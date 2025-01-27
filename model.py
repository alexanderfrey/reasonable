import torch
import torch.nn as nn
import math

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs)  # shape (seq_len, dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to the input tensor using the precomputed frequencies.
    """
    # reshape x to have complex values: [batch, seq_len, heads, dim] -> [batch, seq_len, heads, dim//2]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # reshape freqs_cis to match x's dimensions
    freqs_cis = freqs_cis.view(1, x_complex.shape[1], 1, x_complex.shape[-1])
    # apply rotation using complex multiplication
    x_rotated = x_complex * freqs_cis
    # convert back to real values and original shape
    x_out = torch.view_as_real(x_rotated).flatten(3)
    return x_out.type_as(x)

class GPTConfig:
    """Configuration class for the GPT model."""
    def __init__(
        self,
        vocab_size,
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=1024,
        bias=False,
        dropout=0.0,
        rope_theta=10000.0  
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.bias = bias
        self.dropout = dropout
        self.rope_theta = rope_theta  # RoPE frequency scaling

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self Attention module with RoPE."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Core attention parameters
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # RoPE parameters
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)
        self.max_seq_len = config.block_size
        self.freqs_cis = None
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_value = config.dropout
        
        # Flash Attention check
        self.use_flash_attention = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.use_flash_attention:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.size()
        
        # Initialize RoPE frequencies if not already done
        if self.freqs_cis is None or self.freqs_cis.device != x.device:
            self.freqs_cis = precompute_freqs_cis(
                self.head_dim,
                self.max_seq_len,
                self.rope_theta
            ).to(x.device)
        
        # Compute query, key, value projections
        q = self.query(x).view(B, T, self.n_head, self.head_dim)  # (B, T, H, D)
        k = self.key(x).view(B, T, self.n_head, self.head_dim)    # (B, T, H, D)
        v = self.value(x).view(B, T, self.n_head, self.head_dim)  # (B, T, H, D)
        
        # Apply rotary embeddings to query and key
        q = apply_rotary_emb(q, self.freqs_cis[:T])
        k = apply_rotary_emb(k, self.freqs_cis[:T])
        
        # Rearrange for attention computation
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)  # (B, H, T, D)
        v = v.transpose(1, 2)  # (B, H, T, D)

        # Compute attention
        if self.use_flash_attention:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_value if self.training else 0,
                is_causal=True
            )
        else:
            # Manual attention implementation
            att = (q @ k.transpose(-2, -1)) / self.scale
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
            att = torch.softmax(att, dim=-1)
            att = self.dropout(att)
            out = att @ v

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out

class FeedForward(nn.Module):
    """Feed-Forward Network module."""
    def __init__(self, config: GPTConfig):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Transformer Block consisting of Self-Attention and Feed-Forward layers."""
    def __init__(self, config: GPTConfig):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x, mask=None):
        # Self-Attention
        x = x + self.attn(self.ln1(x), mask=mask)
        # Feed-Forward
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    """GPT Language Model."""
    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

        # Initialize weights
        self._init_weights()

        self.gradient_checkpointing = False

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        if self.config.bias:
            nn.init.zeros_(self.lm_head.bias)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, idx, mask=None):
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.config.block_size}")

        positions = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
        tok_emb = self.token_embedding(idx)  # (B, T, C)
        pos_emb = self.position_embedding(positions)  # (1, T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, mask)
            else:
                x = block(x, mask=mask)

        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits
    