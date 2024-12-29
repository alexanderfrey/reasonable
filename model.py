import torch
import torch.nn as nn
from torch.nn import functional as F

class GPT2WithGroupedAttention(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        embed_size: int, 
        num_heads: int, 
        num_layers: int, 
        block_size: int, 
        num_groups: int,
        dropout: float = 0.1,
        causal: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.block_size = block_size
        self.pad_token_id = 0

        # Token & positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)

        # Transformer blocks with grouped query attention
        # Pass dropout & causal flags to each block
        self.layers = nn.ModuleList([
            TransformerBlockWithGroupedAttention(
                embed_size=embed_size,
                num_heads=num_heads,
                num_groups=num_groups,
                dropout=dropout,
                causal=causal
            )
            for _ in range(num_layers)
        ])

        # Final layer norm + output head
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) input token IDs

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = x.shape

        # 1) Truncate if input is too long
        if T > self.block_size:
            x = x[:, :self.block_size]

        # 2) Pad if input is too short
        if T < self.block_size:
            pad_length = self.block_size - T
            pad = torch.full(
                (B, pad_length), 
                self.pad_token_id, 
                device=x.device, 
                dtype=x.dtype
            )
            x = torch.cat([x, pad], dim=1)

        # Update T after truncation/padding
        T = x.shape[1]

        # 3) Create a key padding mask for padded tokens
        # shape: (B, T), True means "ignore this position"
        pad_mask = (x == self.pad_token_id)
        # If no padding at all, we can pass None to skip overhead
        key_padding_mask = pad_mask if pad_mask.any() else None

        # 4) (Optional) Build a causal mask for [T, T]
        # True means "block this attention connection"
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), 
            diagonal=1
        ) if T > 1 else None

        # 5) Token + positional embeddings
        token_emb = self.token_embedding(x)  # (B, T, C)
        positions = torch.arange(T, device=x.device)
        pos_emb = self.position_embedding(positions)  # (T, C)
        pos_emb = pos_emb.unsqueeze(0).expand(B, T, -1)  # (B, T, C)

        # Combine them
        h = token_emb + pos_emb  # (B, T, C)

        # 6) Pass through Transformer layers
        for layer in self.layers:
            # Each layer can handle the causal & key padding mask
            h = layer(h, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        # 7) Final LN + output head
        h = self.ln_f(h)               # (B, T, C)
        logits = self.head(h)          # (B, T, vocab_size)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.ln2 = nn.LayerNorm(embed_size)
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x
    

class TransformerBlockWithGroupedAttention(nn.Module):
    def __init__(
        self, 
        embed_size: int, 
        num_heads: int, 
        num_groups: int, 
        dropout: float = 0.1,
        causal: bool = True
    ):
        """
        Args:
            embed_size (int): Total embedding dimension.
            num_heads (int): Number of attention heads (per group).
            num_groups (int): Number of groups to split embed_size into.
            dropout (float): Dropout probability for attention & FF layers.
            causal (bool): If True, apply a causal mask so each token cannot attend beyond its own position.
        """
        super().__init__()
        self.num_groups = num_groups
        self.embed_size = embed_size
        self.group_size = embed_size // num_groups
        assert self.group_size * num_groups == embed_size, (
            "Embedding size must be divisible by num_groups"
        )

        self.causal = causal
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        # Attention within groups
        self.group_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.group_size, 
                num_heads=num_heads, 
                dropout=dropout,   # dropout in MHA
                batch_first=False  # we’ll permute to (seq, batch, embed)
            )
            for _ in range(num_groups)
        ])

        self.ln1 = nn.LayerNorm(embed_size)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(
        self, 
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Shape (B, T, C) = (batch, seq_len, embed_size)
            attn_mask (torch.Tensor, optional): 
                Typically shape (T, T). Causal or other masking for attention.
            key_padding_mask (torch.Tensor, optional): 
                Shape (B, T), where True indicates masked (ignored) positions.
        """
        B, T, C = x.shape  # Batch, Sequence, Channels

        # 1) Split input into groups: shape -> (num_groups, B, T, group_size)
        #    each group is processed by a separate MultiheadAttention
        grouped_x = x.view(B, T, self.num_groups, self.group_size).permute(2, 0, 1, 3)
        grouped_outputs = []

        # 2) Prepare a causal mask if needed
        #    If self.causal == True and attn_mask not provided, create an upper-triangular mask
        #    for MHA to ensure token i only attends to [0..i]
        #    shape (T, T), True=masked
        if self.causal and attn_mask is None:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), 
                diagonal=1
            )
        else:
            causal_mask = attn_mask  # could be None or user-provided

        # 3) Process each group independently
        for i, attn in enumerate(self.group_attn):
            # group: (B, T, group_size) -> (T, B, group_size) for MultiheadAttention
            group = grouped_x[i].permute(1, 0, 2)

            # attn_out shape: (T, B, group_size)
            attn_out, _ = attn(
                query=group, 
                key=group, 
                value=group,
                attn_mask=causal_mask,          # shape (T, T)
                key_padding_mask=key_padding_mask  # shape (B, T) if provided
            )
            # Convert back to (B, T, group_size)
            attn_out = attn_out.permute(1, 0, 2)

            grouped_outputs.append(attn_out)

        # 4) Concatenate group outputs: shape => (B, T, embed_size)
        attn_output = torch.cat(grouped_outputs, dim=-1)

        # 5) Residual + LayerNorm (with dropout)
        #    GPT-2 style: x + dropout(attn_out), then LN
        x = self.ln1(x + self.dropout_attn(attn_output))

        # 6) Feed-forward network with dropout
        ffn_out = self.ffn(x)                     # (B, T, C)
        x = self.ln2(x + self.dropout_ffn(ffn_out))
        return x
