"""
bilateral_model.py

This module implements a Bilateral Transformer model with two parallel pathways
that share information through lateral connections. Both pathways are full transformers
that can process the same input for different objectives while sharing information
at multiple levels of processing.
"""

import torch
import torch.nn as nn
import math


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    Args:
        dim (int): Hidden dimension
        end (int): Maximum sequence length
        theta (float): Base value for frequency computation

    Returns:
        torch.Tensor: Complex tensor containing precomputed frequencies
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to the input tensor using precomputed frequencies.

    Args:
        x (torch.Tensor): Input tensor
        freqs_cis (torch.Tensor): Precomputed frequency tensor

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x_complex.shape[1], 1, x_complex.shape[-1])
    x_rotated = x_complex * freqs_cis
    return torch.view_as_real(x_rotated).flatten(3).type_as(x)


class BilateralGPTConfig:
    """Configuration class for bilateral GPT model with lateral connections."""

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
        lateral_dim: int = None,
        pretrain_mode: bool = False,  # New parameter for pre-training mode
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.bias = bias
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.lateral_dim = lateral_dim if lateral_dim is not None else n_embd
        self.pretrain_mode = pretrain_mode  # Store pre-training mode flag


class BilateralGPT(nn.Module):
    """GPT model with two parallel pathways and lateral connections."""

    def __init__(self, config: BilateralGPTConfig):
        super().__init__()
        self.config = config

        # Shared embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Bilateral transformer blocks
        self.blocks = nn.ModuleList(
            [BilateralTransformerBlock(config) for _ in range(config.n_layer)]
        )

        # Output layers for both pathways
        self.ln_f_main = nn.LayerNorm(config.n_embd)
        self.ln_f_analysis = nn.LayerNorm(config.n_embd)

        # In pre-training mode, both heads are language model heads
        if config.pretrain_mode:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
            # Share weights between heads in pre-training mode
            self.analysis_head = self.lm_head
        else:
            # In fine-tuning mode, maintain separate heads
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
            self.analysis_head = nn.Linear(
                config.n_embd, config.vocab_size, bias=config.bias
            )

        # Initialize weights
        self._init_weights()

        self.gradient_checkpointing = False

    def _init_weights(self):
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Initialize head biases if present
        if self.config.bias:
            nn.init.zeros_(self.lm_head.bias)
            if (
                not self.config.pretrain_mode
            ):  # Only initialize separate analysis head if not in pre-training
                nn.init.zeros_(self.analysis_head.bias)

        # Initialize head weights
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if (
            not self.config.pretrain_mode
        ):  # Only initialize separate analysis head if not in pre-training
            nn.init.normal_(self.analysis_head.weight, mean=0.0, std=0.02)

    def forward(self, idx, mask=None):
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block size {self.config.block_size}"
            )

        # Generate embeddings
        positions = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)

        # Initialize both pathways with same embeddings
        x_main = tok_emb + pos_emb
        x_analysis = tok_emb + pos_emb

        # Process through bilateral blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x_main, x_analysis = torch.utils.checkpoint.checkpoint(
                    block, x_main, x_analysis, mask
                )
            else:
                x_main, x_analysis = block(x_main, x_analysis, mask=mask)

        # Generate outputs for both pathways
        x_main = self.ln_f_main(x_main)
        x_analysis = self.ln_f_analysis(x_analysis)

        # Generate logits
        logits_main = self.lm_head(x_main)

        # In pre-training mode, both pathways use the same head
        if self.config.pretrain_mode:
            logits_analysis = self.lm_head(x_analysis)
        else:
            logits_analysis = self.analysis_head(x_analysis)

        return logits_main, logits_analysis


class MultiLatentAttention(nn.Module):
    """Multi-Latent Attention module with RoPE."""

    def __init__(self, config: BilateralGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Core attention parameters
        self.n_head = config.n_head
        self.n_latents = (
            config.n_head // 2
        )  # Number of latent units (typically fewer than heads)
        self.head_dim = config.n_embd // config.n_head
        self.latent_dim = self.head_dim * 2  # Larger dim for latent units
        self.scale = math.sqrt(self.head_dim)

        # Latent units - learned parameters
        self.latent_queries = nn.Parameter(
            torch.randn(1, self.n_latents, self.latent_dim) / math.sqrt(self.latent_dim)
        )
        self.latent_keys = nn.Parameter(
            torch.randn(1, self.n_latents, self.latent_dim) / math.sqrt(self.latent_dim)
        )

        # Projections
        self.to_queries = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.to_keys = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.to_values = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Latent projections
        self.to_latent_q = nn.Linear(self.latent_dim, config.n_embd, bias=config.bias)
        self.to_latent_k = nn.Linear(self.latent_dim, config.n_embd, bias=config.bias)

        # Output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # RoPE parameters
        self.rope_theta = config.rope_theta
        self.max_seq_len = config.block_size
        self.freqs_cis = None

        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_value = config.dropout

        # Flash Attention check
        self.use_flash_attention = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )
        if not self.use_flash_attention:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.size()
        H = self.n_head
        L = self.n_latents

        # Initialize RoPE frequencies if needed
        if self.freqs_cis is None or self.freqs_cis.device != x.device:
            self.freqs_cis = precompute_freqs_cis(
                self.head_dim, self.max_seq_len, self.rope_theta
            ).to(x.device)

        # Project input to queries, keys, and values
        q = self.to_queries(x).view(B, T, H, -1)
        k = self.to_keys(x).view(B, T, H, -1)
        v = self.to_values(x).view(B, T, H, -1)

        # Apply rotary embeddings
        q = apply_rotary_emb(q, self.freqs_cis[:T])
        k = apply_rotary_emb(k, self.freqs_cis[:T])

        # Expand latent units to batch size
        latent_q = self.latent_queries.expand(B, -1, -1)
        latent_k = self.latent_keys.expand(B, -1, -1)

        # Project latent units
        latent_q = self.to_latent_q(latent_q).view(B, L, H, -1)
        latent_k = self.to_latent_k(latent_k).view(B, L, H, -1)

        # Combine with input projections
        q = torch.cat([q, latent_q], dim=1)  # B, T+L, H, D
        k = torch.cat([k, latent_k], dim=1)  # B, T+L, H, D
        v = torch.cat([v, torch.zeros_like(latent_q)], dim=1)  # Pad values

        # Rearrange for attention
        q = q.transpose(1, 2)  # B, H, T+L, D
        k = k.transpose(1, 2)  # B, H, T+L, D
        v = v.transpose(1, 2)  # B, H, T+L, D

        # Compute attention
        if self.use_flash_attention:
            # Adjust causal mask for latent units
            causal_mask = torch.ones((T + L, T + L), device=x.device, dtype=torch.bool)
            causal_mask[:T, :T] = torch.tril(torch.ones((T, T), device=x.device)).bool()
            causal_mask[T:, :] = True  # Latent units can attend to everything

            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=causal_mask,
                dropout_p=self.dropout_value if self.training else 0,
                is_causal=False,  # We handle causality with our custom mask
            )
        else:
            # Manual attention with custom mask for latent units
            att = (q @ k.transpose(-2, -1)) / self.scale

            # Create mask that allows latent units to attend globally
            extended_mask = torch.ones((B, H, T + L, T + L), device=x.device)
            extended_mask[:, :, :T, :T] = self.mask[:, :, :T, :T]
            extended_mask[:, :, T:, :] = 1  # Latent units can attend to everything

            att = att.masked_fill(extended_mask == 0, float("-inf"))
            att = torch.softmax(att, dim=-1)
            att = self.dropout(att)
            out = att @ v

        # Extract only the non-latent outputs
        out = out[:, :, :T, :].transpose(1, 2).contiguous()
        out = out.view(B, T, C)

        # Final projection
        out = self.proj(out)
        out = self.dropout(out)

        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self Attention module with RoPE."""

    def __init__(self, config: BilateralGPTConfig):
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
        self.rope_theta = config.rope_theta
        self.max_seq_len = config.block_size
        self.freqs_cis = None

        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_value = config.dropout

        # Flash Attention check
        self.use_flash_attention = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )
        if not self.use_flash_attention:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.size()

        # Initialize RoPE frequencies if needed
        if self.freqs_cis is None or self.freqs_cis.device != x.device:
            self.freqs_cis = precompute_freqs_cis(
                self.head_dim, self.max_seq_len, self.rope_theta
            ).to(x.device)

        # Compute query, key, value projections
        q = self.query(x).view(B, T, self.n_head, self.head_dim)
        k = self.key(x).view(B, T, self.n_head, self.head_dim)
        v = self.value(x).view(B, T, self.n_head, self.head_dim)

        # Apply rotary embeddings
        q = apply_rotary_emb(q, self.freqs_cis[:T])
        k = apply_rotary_emb(k, self.freqs_cis[:T])

        # Rearrange for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        if self.use_flash_attention:
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_value if self.training else 0,
                is_causal=True,
            )
        else:
            # Manual attention implementation
            att = (q @ k.transpose(-2, -1)) / self.scale
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
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

    def __init__(self, config: BilateralGPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class LateralConnection(nn.Module):
    """Module for lateral connections between parallel transformers."""

    def __init__(self, config: BilateralGPTConfig):
        super().__init__()
        self.proj_main = nn.Linear(config.n_embd, config.lateral_dim, bias=config.bias)
        self.proj_analysis = nn.Linear(
            config.n_embd, config.lateral_dim, bias=config.bias
        )
        self.gate_main = nn.Linear(
            config.n_embd + config.lateral_dim, config.n_embd, bias=config.bias
        )
        self.gate_analysis = nn.Linear(
            config.n_embd + config.lateral_dim, config.n_embd, bias=config.bias
        )
        self.ln_main = nn.LayerNorm(config.n_embd)
        self.ln_analysis = nn.LayerNorm(config.n_embd)

    def forward(self, x_main, x_analysis):
        # Project both pathways to lateral space
        lat_main = self.proj_main(self.ln_main(x_main))
        lat_analysis = self.proj_analysis(self.ln_analysis(x_analysis))

        # Combine information using gating mechanism
        combined_main = torch.cat([x_main, lat_analysis], dim=-1)
        combined_analysis = torch.cat([x_analysis, lat_main], dim=-1)

        # Generate gating weights
        gate_main = torch.sigmoid(self.gate_main(combined_main))
        gate_analysis = torch.sigmoid(self.gate_analysis(combined_analysis))

        # Apply gates
        x_main = x_main * gate_main
        x_analysis = x_analysis * gate_analysis

        return x_main, x_analysis


class BilateralTransformerBlock(nn.Module):
    def __init__(self, config: BilateralGPTConfig):
        super().__init__()
        # Main pathway
        self.ln1_main = nn.LayerNorm(config.n_embd)
        self.attn_main = MultiLatentAttention(config)
        self.ln2_main = nn.LayerNorm(config.n_embd)
        self.ff_main = FeedForward(config)

        # Analysis pathway
        self.ln1_analysis = nn.LayerNorm(config.n_embd)
        self.attn_analysis = MultiLatentAttention(config)
        self.ln2_analysis = nn.LayerNorm(config.n_embd)
        self.ff_analysis = FeedForward(config)

        # Lateral connections remain unchanged
        self.lateral_pre = LateralConnection(config)
        self.lateral_post = LateralConnection(config)

    def forward(self, x_main, x_analysis, mask=None):
        # Lateral connection before attention
        x_main_lat, x_analysis_lat = self.lateral_pre(x_main, x_analysis)

        # Self-attention for both pathways
        x_main = x_main_lat + self.attn_main(self.ln1_main(x_main_lat), mask=mask)
        x_analysis = x_analysis_lat + self.attn_analysis(
            self.ln1_analysis(x_analysis_lat), mask=mask
        )

        # Rest remains unchanged
        x_main_lat, x_analysis_lat = self.lateral_post(x_main, x_analysis)
        x_main = x_main_lat + self.ff_main(self.ln2_main(x_main_lat))
        x_analysis = x_analysis_lat + self.ff_analysis(
            self.ln2_analysis(x_analysis_lat)
        )

        return x_main, x_analysis
