import torch
import torch.nn as nn
import math
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
        n_groups=1
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.bias = bias
        self.dropout = dropout
        self.n_groups = n_groups

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self Attention module with optional masking."""
    def __init__(self, config: GPTConfig):
        super(MultiHeadSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0, "Embedding size must be divisible by number of heads."
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        # Compute queries, keys, values
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)    # (B, n_head, T, head_dim)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        # Compute scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / self.scale  # (B, n_head, T, T)
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        # Apply attention to values
        out = att @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
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
        # Self-Attention with residual connection
        att_out = self.attn(self.ln1(x), mask=mask)
        x = x + att_out
        # Feed-Forward with residual connection
        ff_out = self.ff(self.ln2(x))
        x = x + ff_out
        return x

class GPT(nn.Module):
    """GPT Language Model with Causal Masking."""
    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        if self.config.bias:
            nn.init.zeros_(self.output.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)

    def forward(self, idx, mask=None):
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.config.block_size}")

        # Generate causal mask if not provided
        if mask is None:
            mask = self.generate_causal_mask(T, idx.device)  # (1, 1, T, T)

        positions = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
        tok_emb = self.token_embedding(idx)  # (B, T, C)
        pos_emb = self.position_embedding(positions)  # (1, T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)  # (B, T, C)
        logits = self.output(x)  # (B, T, vocab_size)
        return logits

    @staticmethod
    def generate_causal_mask(T, device):
        """Generates a causal mask to ensure that each position can only attend to previous positions."""
        # Create a lower triangular matrix filled with 1s
        mask = torch.tril(torch.ones((T, T), device=device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        return mask