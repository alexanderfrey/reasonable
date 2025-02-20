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
        n_aux_heads: int = 0,
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
            n_aux_heads  # Number of auxiliary heads for auxiliary loss training
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
        self.n_aux_heads = n_aux_heads

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
            f"  n_aux_heads: {self.n_aux_heads}\n"
            ")"
        )


class MultiLatentAttention(nn.Module):
    """Memory-efficient Multi-Head Attention with latent queries and chunked processing."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_latents = config.n_latents
        self.head_dim = config.n_embd // config.n_head
        self.chunk_size = 128  # Process attention in chunks

        # Scale with better numerical stability
        self.scale = 1 / math.sqrt(self.head_dim)

        # Memory-efficient latent queries
        self.latent_queries = nn.Parameter(
            torch.empty(1, config.n_latents, config.n_head, self.head_dim)
        )
        torch.nn.init.kaiming_normal_(self.latent_queries, nonlinearity="linear")

        # Linear layers with memory-efficient initialization
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.gate = nn.Linear(
            config.n_embd, config.n_head * config.n_latents, bias=config.bias
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Initialize with memory-efficient approach
        for layer in [self.key, self.value, self.gate, self.proj]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        self.rope_theta = config.rope_theta
        self.max_seq_len = config.block_size
        self.freqs_cis = None

        self.dropout = nn.Dropout(config.dropout)
        self.use_flash_attention = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )

    def process_chunk(self, q, k, v, chunk_size, mask=None):
        B, H, Lq, D = q.shape  # Lq is n_latents
        _, _, T, _ = k.shape  # T is sequence length

        out = torch.zeros_like(q)

        # Process attention in chunks to save memory
        for i in range(0, T, chunk_size):
            chunk_end = min(i + chunk_size, T)
            k_chunk = k[:, :, i:chunk_end]
            v_chunk = v[:, :, i:chunk_end]

            if mask is not None:
                # Need to adjust mask to match latent query dimension
                # Slice mask to match chunk size and reshape for latent queries
                mask_chunk = mask[:, :, :Lq, i:chunk_end] if mask is not None else None

            if self.use_flash_attention:
                chunk_out = torch.nn.functional.scaled_dot_product_attention(
                    q,  # [B, H, Lq, D]
                    k_chunk,  # [B, H, chunk_size, D]
                    v_chunk,  # [B, H, chunk_size, D]
                    attn_mask=mask_chunk,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=True,
                )
            else:
                attn_weights = (q @ k_chunk.transpose(-2, -1)) * self.scale

                if mask_chunk is not None:
                    attn_weights = attn_weights.masked_fill(
                        mask_chunk == 0, float("-inf")
                    )

                attn_weights = F.softmax(attn_weights, dim=-1)
                attn_weights = self.dropout(attn_weights)
                chunk_out = attn_weights @ v_chunk

            out = out + chunk_out

        return out

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.size()

        # Initialize RoPE frequencies if needed
        if self.freqs_cis is None or self.freqs_cis.device != x.device:
            self.freqs_cis = precompute_freqs_cis(
                self.head_dim, self.max_seq_len, self.rope_theta
            ).to(x.device)

        # Create causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)

        # Handle attention mask
        if mask is None:
            # If no mask provided, use only causal masking
            mask = (~causal_mask.bool()).float()
            # Reshape for batch and heads: [B, 1, 1, T, T]
            mask = mask.unsqueeze(0).unsqueeze(1).unsqueeze(1)
        else:
            # Ensure mask has correct shape [B, 1, 1, T, T]
            if mask.dim() == 3:  # [B, T, T]
                mask = mask.unsqueeze(1).unsqueeze(1)

            # Combine with causal mask
            causal_mask = (
                (~causal_mask.bool()).float().unsqueeze(0).unsqueeze(1).unsqueeze(1)
            )
            mask = mask * causal_mask

        # Now reshape mask for latent queries
        # Reshape from [B, 1, 1, T, T] to [B, H, Lq, T]
        # We only need the target sequence dimension from the last axis
        mask = mask.squeeze(2)[:, :, :, 0:T]  # [B, 1, T, T]
        mask = mask.expand(B, self.n_head, T, T)  # [B, H, T, T]
        # Now adjust for latent queries
        mask = mask[:, :, : self.n_latents, :]  # [B, H, Lq, T]
        k = self.key(x).view(B, T, self.n_head, self.head_dim)
        v = self.value(x).view(B, T, self.n_head, self.head_dim)
        k = apply_rotary_emb(k, self.freqs_cis[:T])
        gates = self.gate(x).view(B, T, self.n_head, self.n_latents)
        gates = F.softmax(gates * self.scale, dim=-1)
        q = self.latent_queries.expand(B, -1, -1, -1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Pass mask to process_chunk
        attn_output = self.process_chunk(q, k, v, self.chunk_size, mask=mask)

        gates = gates.transpose(1, 2)
        weighted_output = torch.einsum("bhld,bhtl->bhtd", attn_output, gates)

        out = weighted_output.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    """Memory-efficient Feed-Forward Network with activation checkpointing."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        # Keep intermediate dimensions consistent with input embedding dimension
        self.hidden_dim = 4 * config.n_embd  # Standard scaling for transformer FFN

        # Up projections for SwiGLU
        self.up_proj = nn.Linear(config.n_embd, self.hidden_dim, bias=config.bias)
        self.gate_proj = nn.Linear(config.n_embd, self.hidden_dim, bias=config.bias)

        # Down projection
        self.down_proj = nn.Linear(self.hidden_dim, config.n_embd, bias=config.bias)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize up projections
        for module in [self.up_proj, self.gate_proj]:
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # Initialize down projection with zeros for better training stability
        nn.init.zeros_(self.down_proj.weight)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def _activation_function(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # Memory-efficient SwiGLU activation
        return F.silu(gate) * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Up projections
        hidden = self.up_proj(x)
        gate = self.gate_proj(x)

        # Checkpoint the activation function to save memory
        if self.training:
            hidden = torch.utils.checkpoint.checkpoint(
                self._activation_function, hidden, gate, use_reentrant=False
            )
        else:
            hidden = self._activation_function(hidden, gate)

        # Down projection and dropout
        out = self.down_proj(hidden)
        return self.dropout(out)


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

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

        # Output layers
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

        self.aux_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.n_embd),
                    nn.Linear(config.n_embd, config.vocab_size, bias=config.bias),
                )
                for _ in range(config.n_aux_heads)
            ]
        )

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

        # Initialize main head
        if self.config.bias:
            nn.init.zeros_(self.lm_head.bias)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        # Initialize auxiliary heads
        for aux_head in self.aux_heads:
            if self.config.bias:
                nn.init.zeros_(aux_head[1].bias)
            nn.init.normal_(aux_head[1].weight, mean=0.0, std=0.02)

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

    def forward(self, idx, attention_mask=None):
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block size {self.config.block_size}"
            )

        # Only use token embeddings - RoPE handles positional information
        x = self.token_embedding(idx)

        # Store intermediate outputs for auxiliary heads
        intermediate_outputs = []
        aux_interval = max(1, self.config.n_layer // (self.config.n_aux_heads + 1))

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
                    attention_mask,
                    use_reentrant=False,
                )

                # Store intermediate outputs at regular intervals
                if self.config.n_aux_heads > 0 and (i + 1) % aux_interval == 0:
                    intermediate_outputs.append(x)
        else:
            for i, block in enumerate(self.blocks):
                x = block(x, mask=attention_mask)
                if self.config.n_aux_heads > 0 and (i + 1) % aux_interval == 0:
                    intermediate_outputs.append(x)

        # Generate main output and normalized final representations
        x = self.ln_f(x)  # Get normalized final representations
        main_logits = self.lm_head(x)

        # Generate auxiliary outputs using the same normalized representations
        aux_logits = []
        for aux_head in self.aux_heads:
            aux_logits.append(aux_head(x))  # Use final normalized representations

        if self.config.n_aux_heads > 0:
            return main_logits, aux_logits
        return main_logits


class ExpertFFN(nn.Module):
    """Expert Feed-Forward Network with SwiGLU activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.hidden_dim = 4 * config.n_embd

        # Expert layers
        self.up_proj = nn.Linear(config.n_embd, self.hidden_dim, bias=config.bias)
        self.gate_proj = nn.Linear(config.n_embd, self.hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(self.hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in [self.up_proj, self.gate_proj]:
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Zero init for better stability
        nn.init.zeros_(self.down_proj.weight)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.up_proj(x)
        gate = self.gate_proj(x)
        hidden = F.silu(gate) * hidden
        out = self.down_proj(hidden)
        return self.dropout(out)


class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing."""

    def __init__(self, config: GPTConfig, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Create experts
        self.experts = nn.ModuleList([ExpertFFN(config) for _ in range(num_experts)])

        # Router with additional capacity factor
        router_hidden = config.n_embd // 2
        self.router = nn.Sequential(
            nn.Linear(config.n_embd, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, num_experts),
        )

        # Balance loss coefficient
        self.balance_coefficient = 0.01

        # Initialize router with small weights
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _compute_routing_probabilities(
        self, router_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing probabilities and selected experts."""
        temperature = 0.1  # Add temperature scaling
        scaled_logits = router_logits / temperature
        top_logits, top_indices = torch.topk(scaled_logits, self.top_k, dim=-1)

        # Compute softmax over selected logits
        router_probs = F.softmax(top_logits, dim=-1)

        # Mask for load balancing
        mask = torch.zeros_like(router_logits).scatter_(-1, top_indices, 1.0)

        return router_probs, top_indices, mask

    def _compute_balance_loss(self, router_mask: torch.Tensor) -> torch.Tensor:
        # Calculate expert assignment ratios
        expert_counts = router_mask.sum(0) + 1e-6  # Add small epsilon for stability
        expert_ratios = expert_counts / expert_counts.sum()

        # Use negative entropy as balance loss
        entropy_loss = -(expert_ratios * torch.log(expert_ratios)).sum()
        return self.balance_coefficient * entropy_loss

    def forward(
        self, x: torch.Tensor, return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, d_model = x.shape

        # Compute router logits
        router_logits = self.router(x)

        # Get routing probabilities and expert assignments
        router_probs, expert_indices, router_mask = self._compute_routing_probabilities(
            router_logits
        )

        # Reshape input for parallel expert processing
        x_reshaped = x.view(-1, d_model)

        # Initialize output tensor
        final_output = torch.zeros_like(x_reshaped)

        # Process inputs through experts
        for k in range(self.top_k):
            expert_idx = expert_indices[..., k].view(-1)
            prob = router_probs[..., k].view(-1, 1)

            # Route inputs to each expert
            for i in range(self.num_experts):
                expert_mask = expert_idx == i
                if expert_mask.any():
                    expert_input = x_reshaped[expert_mask]
                    expert_output = self.experts[i](expert_input)
                    final_output[expert_mask] += prob[expert_mask] * expert_output

        # Reshape output back to original dimensions
        output = final_output.view(batch_size, seq_len, d_model)

        if return_aux_loss:
            aux_loss = self._compute_balance_loss(router_mask)
            return output, aux_loss

        return output, None


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE FFN replacing standard FFN."""

    def __init__(self, config: GPTConfig, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiLatentAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.moe = MoELayer(config, num_experts=num_experts, top_k=top_k)
        self.skip_scale = nn.Parameter(torch.ones(1, 1, config.n_embd) * 0.1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Attention block
        attn_output = self.attn(self.ln1(x), mask=mask)
        x = x + attn_output * self.skip_scale

        # MoE block
        moe_output, aux_loss = self.moe(self.ln2(x))
        x = x + moe_output * self.skip_scale

        return x, aux_loss


class MoEGPT(nn.Module):
    """GPT model with MoE layers."""

    def __init__(
        self,
        config: GPTConfig,
        num_experts: int = 8,
        top_k: int = 2,
        moe_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.config = config

        self.num_experts = num_experts

        # Default to MoE every other layer if not specified
        if moe_layers is None:
            moe_layers = list(range(1, config.n_layer, 2))
        self.moe_layers = set(moe_layers)

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Mix of standard and MoE transformer blocks
        self.blocks = nn.ModuleList(
            [
                (
                    MoETransformerBlock(config, num_experts, top_k)
                    if i in self.moe_layers
                    else TransformerBlock(config)
                )
                for i in range(config.n_layer)
            ]
        )

        # Output layers
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

        # Initialize weights
        self._init_weights()

        # Enable gradient checkpointing by default
        self.gradient_checkpointing = True
        self.checkpoint_every_n = 2

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through a group of blocks with auxiliary loss accumulation."""
        total_aux_loss = 0.0

        for block in blocks:
            if isinstance(block, MoETransformerBlock):
                x, aux_loss = block(x, mask=mask)
                if aux_loss is not None:
                    total_aux_loss += aux_loss
            else:
                x = block(x, mask=mask)

        return x, total_aux_loss

    def forward(
        self, idx: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # Track total auxiliary loss
        total_aux_loss = 0.0

        # Process through transformer blocks with grouped checkpointing
        if self.gradient_checkpointing and self.training:
            num_blocks = len(self.blocks)
            for i in range(0, num_blocks, self.checkpoint_every_n):
                block_group = self.blocks[
                    i : min(i + self.checkpoint_every_n, num_blocks)
                ]
                x, aux_loss = torch.utils.checkpoint.checkpoint(
                    self._forward_block_group, x, block_group, mask, use_reentrant=False
                )
                total_aux_loss += aux_loss
        else:
            x, total_aux_loss = self._forward_block_group(x, self.blocks, mask)

        # Generate output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, total_aux_loss
