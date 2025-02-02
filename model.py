import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Union, Tuple


# Keeping the existing helper functions unchanged
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x_complex.shape[1], 1, x_complex.shape[-1])
    x_rotated = x_complex * freqs_cis
    return torch.view_as_real(x_rotated).flatten(3).type_as(x)


class HeadConfig:
    """Configuration for a single output head."""

    def __init__(
        self,
        name: str,
        vocab_size: int,
        weight: float = 1.0,
    ):
        self.name = name
        self.vocab_size = vocab_size
        self.weight = weight


class GPTConfig:
    """Configuration class for GPT model with rotary position embeddings."""

    def __init__(
        self,
        vocab_size: int,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        block_size: int = 1024,
        bias: bool = False,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        n_latents: int = 64,
    ):
        """
        Initialize GPT configuration.

        Args:
            vocab_size (int): Size of the vocabulary
            n_layer (int): Number of transformer layers
            n_head (int): Number of attention heads
            n_embd (int): Embedding dimension
            block_size (int): Maximum sequence length
            bias (bool): Whether to use bias in linear layers
            dropout (float): Dropout probability
            rope_theta (float): Base for rotary position embedding
            n_latents (int): Number of latent queries for attention
        """
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.bias = bias
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.n_latents = n_latents

    def __repr__(self):
        """String representation of the configuration."""
        return (
            f"GPTConfig(\n"
            f"  vocab_size: {self.vocab_size}\n"
            f"  n_layer: {self.n_layer}\n"
            f"  n_head: {self.n_head}\n"
            f"  n_embd: {self.n_embd}\n"
            f"  block_size: {self.block_size}\n"
            f"  bias: {self.bias}\n"
            f"  dropout: {self.dropout}\n"
            f"  rope_theta: {self.rope_theta}\n"
            f"  n_latents: {self.n_latents}\n"
            ")"
        )


class MultiLatentAttention(nn.Module):
    """Multi-Head Attention with latent queries, RoPE, and optimized scaling."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Core attention parameters
        self.n_head = config.n_head
        self.n_latents = config.n_latents
        self.head_dim = config.n_embd // config.n_head

        # Dynamic scaling based on sequence length
        self.scale = 1 / math.sqrt(self.head_dim)

        # Latent queries with improved initialization
        self.latent_queries = nn.Parameter(
            torch.randn(1, config.n_latents, config.n_head, self.head_dim) * 0.02
        )

        # Linear projections with improved initialization
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.gate = nn.Linear(
            config.n_embd, config.n_head * config.n_latents, bias=config.bias
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Initialize projection layers with better scaling
        nn.init.normal_(self.key.weight, std=0.02)
        nn.init.normal_(self.value.weight, std=0.02)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

        # RoPE parameters
        self.rope_theta = config.rope_theta
        self.max_seq_len = config.block_size
        self.freqs_cis = None

        # Optimized dropout rates
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_value = config.dropout * 0.8  # Reduced dropout for attention

        # Flash Attention check
        self.use_flash_attention = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )

        # Improved normalization strategy with per-head norms
        self.query_norm = nn.LayerNorm(self.head_dim)
        self.key_norm = nn.LayerNorm(self.head_dim)
        self.value_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.size()

        # Initialize RoPE frequencies if needed
        if self.freqs_cis is None or self.freqs_cis.device != x.device:
            self.freqs_cis = precompute_freqs_cis(
                self.head_dim, self.max_seq_len, self.rope_theta
            ).to(x.device)

        # Dynamic scaling based on sequence length
        length_scale = 1.0 / math.sqrt(T)

        # Compute projections
        k = self.key(x).view(B, T, self.n_head, self.head_dim)
        v = self.value(x).view(B, T, self.n_head, self.head_dim)

        # Per-head normalization
        k = self.key_norm(k.transpose(1, 2)).transpose(1, 2)  # [B, T, H, D]
        v = self.value_norm(v.transpose(1, 2)).transpose(1, 2)  # [B, T, H, D]

        # Apply length-dependent scaling
        k = k * length_scale
        v = v * length_scale

        # Compute gating weights with improved numerics
        gates = self.gate(x).view(B, T, self.n_head, self.n_latents)
        gates = torch.softmax(gates * self.scale, dim=-1)

        # Expand and normalize latent queries
        q = self.latent_queries.expand(B, -1, -1, -1)  # [B, L, H, D]
        q = self.query_norm(q.transpose(1, 2)).transpose(1, 2)
        q = q * length_scale

        # Apply rotary embeddings to keys
        k = apply_rotary_emb(k, self.freqs_cis[:T])

        # Rearrange for attention computation
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)  # [B, H, T, D]
        v = v.transpose(1, 2)  # [B, H, T, D]

        # Compute attention with flash attention when available
        if self.use_flash_attention:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_value if self.training else 0,
                is_causal=True,
            )
        else:
            # Compute attention scores with improved numerical stability
            attn_weights = (q @ k.transpose(-2, -1)) * self.scale

            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = attn_weights @ v

        # Apply gating mechanism
        gates = gates.transpose(1, 2)  # [B, H, T, L]
        weighted_output = torch.einsum("bhls,bhtl->bhts", attn_output, gates)

        # Final projection with gradient stabilization
        out = weighted_output.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    """Optimized Feed-Forward Network with parallel paths and improved activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        # Split the network into two parallel paths
        self.w1_path = nn.Linear(
            config.n_embd, (4 * config.n_embd) // 3, bias=config.bias
        )
        self.w2_path = nn.Linear(
            config.n_embd, (4 * config.n_embd) // 3, bias=config.bias
        )
        self.w3_path = nn.Linear(
            config.n_embd, (4 * config.n_embd) // 3, bias=config.bias
        )

        # Output projection
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        # Improved dropout with path-specific rates
        self.dropout1 = nn.Dropout(config.dropout * 0.9)
        self.dropout2 = nn.Dropout(config.dropout * 0.8)
        self.dropout3 = nn.Dropout(config.dropout * 0.7)
        self.dropout_out = nn.Dropout(config.dropout)

        # Initialize with improved scaling
        self.init_weights()

        # Layer scale parameter
        self.layer_scale = nn.Parameter(torch.ones(1, 1, config.n_embd) * 0.1)

    def init_weights(self):
        # Initialize with scaled factors for better gradient flow
        for module in [self.w1_path, self.w2_path, self.w3_path]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # Special initialization for output projection
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02 / math.sqrt(2))
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # Process parallel paths with different activation functions
        path1 = self.w1_path(x)
        path2 = self.w2_path(x)
        path3 = self.w3_path(x)

        # Apply different activation patterns
        path1 = F.gelu(path1, approximate="tanh")  # Faster approximate GELU
        path2 = F.silu(path2)  # SiLU activation
        path3 = F.relu(path3)  # ReLU activation

        # Apply path-specific dropout
        path1 = self.dropout1(path1)
        path2 = self.dropout2(path2)
        path3 = self.dropout3(path3)

        # Combine paths
        combined = torch.cat([path1, path2, path3], dim=-1)

        # Apply output transformation
        output = self.proj(combined)
        output = self.dropout_out(output)

        # Apply learned scaling
        return output * self.layer_scale


class TransformerBlock(nn.Module):
    """Transformer block with optimized feed-forward network."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiLatentAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

        # Add skip connection scaling
        self.skip_scale = nn.Parameter(torch.ones(1, 1, config.n_embd) * 0.1)

    def forward(self, x, mask=None):
        # Residual connections with learned scaling
        attn_output = self.attn(self.ln1(x), mask=mask)
        x = x + attn_output * self.skip_scale

        ff_output = self.ff(self.ln2(x))
        x = x + ff_output * self.skip_scale

        return x


class GPT(nn.Module):
    """GPT model with multi-latent attention and optimized gradient checkpointing."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

        # Output layers
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

        # Initialize weights
        self._init_weights()

        # Initialize latent queries with small random values
        for block in self.blocks:
            nn.init.normal_(block.attn.latent_queries, mean=0.0, std=0.02)

        # Enable gradient checkpointing by default for better memory efficiency
        self.gradient_checkpointing = True

        # Group layers for efficient checkpointing
        self.checkpoint_every_n = 2  # Checkpoint every 2 layers

    def _init_weights(self):
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Initialize head
        if self.config.bias:
            nn.init.zeros_(self.lm_head.bias)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def _forward_block_group(
        self,
        x: torch.Tensor,
        blocks: List[nn.Module],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through a group of blocks with optional checkpointing."""
        for block in blocks:
            x = block(x, mask=mask)
        return x

    def forward(self, idx, mask=None):
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block size {self.config.block_size}"
            )

        # Generate embeddings
        positions = (
            torch.arange(0, T, dtype=torch.long, device=idx.device)
            .unsqueeze(0)
            .expand(B, -1)
        )

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)
        x = tok_emb + pos_emb

        # Process through transformer blocks with grouped checkpointing
        if self.gradient_checkpointing and self.training:
            num_blocks = len(self.blocks)
            for i in range(0, num_blocks, self.checkpoint_every_n):
                block_group = self.blocks[
                    i : min(i + self.checkpoint_every_n, num_blocks)
                ]
                x = torch.utils.checkpoint.checkpoint(
                    self._forward_block_group,
                    x,
                    block_group,
                    mask,
                    use_reentrant=False,  # More memory efficient
                )
        else:
            x = self._forward_block_group(x, self.blocks, mask)

        # Generate output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


class MultiHeadGPT(nn.Module):
    """GPT model with configurable multiple heads for different objectives."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Calculate single scaling factor based on embedding dimension
        self.embedding_scale = 1.0 / math.sqrt(config.n_embd)

        # Embeddings (shared between heads)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Transformer blocks (shared backbone)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

        # Create separate normalization layers and heads for each output head
        self.ln_fs = nn.ModuleDict(
            {head.name: nn.LayerNorm(config.n_embd) for head in config.heads}
        )

        # Initialize heads with careful scaling
        self.lm_heads = nn.ModuleDict()
        for head in config.heads:
            head_module = nn.Linear(config.n_embd, head.vocab_size, bias=config.bias)
            # Initialize with scaled xavier/glorot
            nn.init.xavier_normal_(head_module.weight, gain=0.1)
            if config.bias:
                nn.init.zeros_(head_module.bias)
            self.lm_heads[head.name] = head_module

        # Store head weights for loss calculation
        self.head_weights = {head.name: head.weight for head in config.heads}

        # Initialize all weights with careful scaling
        self._init_weights()

        # Enable gradient checkpointing by default for better memory efficiency
        self.gradient_checkpointing = True

    def _init_weights(self):
        # Initialize embeddings with smaller standard deviation
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Initialize latent queries for all blocks
        for block in self.blocks:
            if hasattr(block.attn, "latent_queries"):
                nn.init.normal_(block.attn.latent_queries, mean=0.0, std=0.02)

        # Careful initialization of layer norms
        for name, module in self.ln_fs.items():
            nn.init.constant_(module.weight, 0.1)  # Start with smaller scale
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        head_names: Optional[List[str]] = None,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block size {self.config.block_size}"
            )

        # Generate embeddings with single scaling
        positions = (
            torch.arange(0, T, dtype=torch.long, device=idx.device)
            .unsqueeze(0)
            .expand(B, -1)
        )

        # Apply single scaling factor to combined embeddings
        x = (
            self.token_embedding(idx) + self.position_embedding(positions)
        ) * self.embedding_scale

        # Process through transformer blocks with gradient checkpointing
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, mask)
            else:
                x = block(x, mask=mask)

        # Get head names to compute
        heads_to_compute = (
            head_names if head_names is not None else list(self.lm_heads.keys())
        )

        # Compute outputs for each head without additional scaling
        outputs = {}
        for name in heads_to_compute:
            normalized = self.ln_fs[name](x)
            outputs[name] = self.lm_heads[name](normalized)

        return outputs[heads_to_compute[0]] if len(heads_to_compute) == 1 else outputs

    def get_grad_norm(self, norm_type=2.0):
        """Helper method to get total gradient norm for monitoring."""
        parameters = [p for p in self.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return torch.tensor(0.0)
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
        return total_norm

    def enable_gradient_checkpointing(self):
        """Helper method to enable gradient checkpointing."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Helper method to disable gradient checkpointing."""
        self.gradient_checkpointing = False
