import torch
import torch.nn as nn
from torch.nn import functional as F
import math



def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Split last dim in half and rotate.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, sin, cos):
    # q, k: shape (B, T, num_heads, head_dim)
    # but sin, cos: shape (1, T, 1, head_dim/2)

    # Split each head-dim in half
    q1, q2 = q.chunk(2, dim=-1)  # each has shape (B, T, num_heads, head_dim/2)
    k1, k2 = k.chunk(2, dim=-1)

    # Apply rotation
    # "classic" formula is often:
    #   q1_rot = q1*cos - q2*sin
    #   q2_rot = q2*cos + q1*sin
    #   (and similarly for k)
    q1_rot = q1 * cos - q2 * sin
    q2_rot = q2 * cos + q1 * sin
    k1_rot = k1 * cos - k2 * sin
    k2_rot = k2 * cos + k1 * sin

    # Recombine
    q_rot = torch.cat([q1_rot, q2_rot], dim=-1)
    k_rot = torch.cat([k1_rot, k2_rot], dim=-1)

    return q_rot, k_rot


class RotaryEmbedding(nn.Module):
    """
    Precomputes sin/cos for rotary embeddings.
    By default, this is sized for max_seq_len. If you 
    need dynamic shapes, you'll need to adapt or 
    compute sin/cos on the fly.
    """
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # We assume head_dim is even; if not, you'd need adjustments.
        # (T, head_dim)
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # (T, 1, head_dim)
        t = torch.arange(self.max_seq_len, dtype=torch.float)
        angles = t[:, None] * freqs[None, :]
        
        # shape: (T, head_dim/2)
        sin = angles.sin()
        cos = angles.cos()

        # We'll expand these in forward() to match [B, T, num_heads, head_dim]
        # But store them as buffers.
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> (torch.Tensor, torch.Tensor):
        """
        Return the slice of precomputed sin, cos up to seq_len,
        reshaped to (1, seq_len, 1, head_dim).
        """
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(2).to(device)
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(2).to(device)
        return sin, cos
    
class GroupedMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, num_groups, dropout=0.1, causal=True):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.causal = causal

        # Per-head dimension
        self.head_dim = embed_size // num_heads
        assert self.head_dim * num_heads == embed_size, \
            "embed_size must be divisible by num_heads"

        # Projection layers for Q, K, V
        self.W_q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_k = nn.Linear(embed_size, embed_size, bias=False)
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)
        self.W_o = nn.Linear(embed_size, embed_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None, rotary_emb=None):
        """
        x: (B, T, C)
        attn_mask: (T, T) or None
        key_padding_mask: (B, T) or None
        rotary_emb: (sin, cos) used to apply rotary embeddings to Q, K
        """
        B, T, C = x.shape

        # 1) Project to Q, K, V
        q = self.W_q(x)  # (B, T, C)
        k = self.W_k(x)  # (B, T, C)
        v = self.W_v(x)  # (B, T, C)

        # 2) Reshape into (B, T, num_heads, head_dim) for multi-head
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # 3) Apply rotary embeddings on Q and K if provided
        if rotary_emb is not None:
            sin, cos = rotary_emb  # each shape: (1, T, 1, head_dim)
            q, k = apply_rotary_pos_emb(q, k, sin, cos)

        # 4) Perform grouped attention (a simplified example).
        #
        # In real "grouped" attention, you'd slice or chunk heads
        # into groups, possibly do separate attention, etc.
        # Below is just standard multi-head for demonstration.
        #
        # a) Compute attention scores: shape (B, num_heads, T, T)
        scores = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1)) 
        scores = scores / math.sqrt(self.head_dim)

        # b) Apply causal mask (if any)
        if self.causal and attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0), float('-inf'))

        # c) Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (B, T)
            # We need to broadcast to (B, num_heads, 1, T)
            pad = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            scores = scores.masked_fill(pad, float('-inf'))

        # d) Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # e) Multiply by V
        out = torch.matmul(attn_weights, v.transpose(1, 2))  # (B, num_heads, T, head_dim)

        # 5) Reshape back
        out = out.transpose(1, 2).contiguous()  # (B, T, num_heads, head_dim)
        out = out.view(B, T, C)

        # 6) Final projection
        out = self.W_o(out)  # (B, T, C)
        return out
    
class TransformerBlockWithGroupedAttention(nn.Module):
    def __init__(self, embed_size, num_heads, num_groups, dropout=0.1, causal=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        
        self.attn = GroupedMultiHeadAttention(
            embed_size=embed_size,
            num_heads=num_heads,
            num_groups=num_groups,
            dropout=dropout,
            causal=causal
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None, rotary_emb=None):
        # Pre-norm
        x_ln = self.ln1(x)
        # Multi-head attn
        attn_out = self.attn(x_ln, attn_mask, key_padding_mask, rotary_emb=rotary_emb)
        x = x + attn_out

        # Feed-forward
        x_ln = self.ln2(x)
        ff_out = self.ff(x_ln)
        x = x + ff_out
        return x

class GPT2WithGroupedAttentionRotary(nn.Module):
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

        # Token embedding only — no learned positional embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_size)

        # Rotary embedding for queries and keys
        # We'll assume each head has embed_size // num_heads dimension
        self.rotary_emb = RotaryEmbedding(head_dim=embed_size // num_heads, 
                                          max_seq_len=block_size)

        # Transformer blocks with grouped query attention
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
            T = self.block_size

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
            T = self.block_size

        # 3) Key padding mask
        pad_mask = (x == self.pad_token_id)
        key_padding_mask = pad_mask if pad_mask.any() else None

        # 4) Causal mask
        # shape: (T, T), True means block that connection
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1
        ) if T > 1 else None

        # 5) Token embeddings
        token_emb = self.token_embedding(x)  # (B, T, C)
        h = token_emb

        # 6) Precompute rotary sin, cos up to T
        sin, cos = self.rotary_emb(seq_len=T, device=x.device)

        # 7) Pass through Transformer layers
        for layer in self.layers:
            # Provide rotary_emb=(sin, cos) so the attention can apply it to Q,K
            h = layer(h, attn_mask=causal_mask, 
                      key_padding_mask=key_padding_mask, 
                      rotary_emb=(sin, cos))

        # 8) Final LN + output head
        h = self.ln_f(h)               # (B, T, C)
        logits = self.head(h)          # (B, T, vocab_size)
        return logits