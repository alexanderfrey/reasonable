import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
import triton
import triton.language as tl


class GPTConfig:
    """Configuration class for GPT model with all optimization options."""

    def __init__(
        self,
        vocab_size: int = 50257,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        block_size: int = 1024,
        bias: bool = False,
        dropout: float = 0.1,
        # RoPE settings
        rope_theta: float = 10000.0,
        # Quantization settings
        use_quantization: bool = False,
        quantization_bits: int = 8,
        # Attention settings
        use_flash_attn: bool = True,
        use_triton_kernels: bool = True,
        # MoE settings
        use_moe: bool = False,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        expert_capacity_factor: float = 1.0,
        moe_jitter_noise: float = 0.1,
        # Memory optimization settings
        use_activation_checkpointing: bool = False,
        checkpoint_ratio: float = 0.5,
        use_kv_cache: bool = True,
        kv_cache_strategy: str = "dynamic",  # "dynamic" or "static"
        # Mixed precision settings
        mixed_precision: bool = True,
        mixed_precision_dtype: str = "fp16",  # "fp16" or "bf16"
        # Multi-Latent Attention settings
        latent_dim_scale: int = 2,  # multiplier for latent dimension
        n_latents: Optional[int] = None,  # if None, will be n_head // 2
        # Performance settings
        fused_operations: bool = True,
        use_fused_mlp: bool = True,
        use_fused_attention: bool = True,
        # Advanced optimization settings
        use_memory_efficient_attention: bool = True,
        gradient_checkpointing_policy: str = "nothing_saveable",
        use_sdpa: bool = True,  # scaled dot product attention
        use_parallel_attention: bool = True,
        parallel_block_size: int = 256,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.bias = bias
        self.dropout = dropout

        # RoPE settings
        self.rope_theta = rope_theta

        # Quantization settings
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits

        # Attention settings
        self.use_flash_attn = use_flash_attn
        self.use_triton_kernels = use_triton_kernels

        # MoE settings
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_capacity_factor = expert_capacity_factor
        self.moe_jitter_noise = moe_jitter_noise

        # Memory optimization settings
        self.use_activation_checkpointing = use_activation_checkpointing
        self.checkpoint_ratio = checkpoint_ratio
        self.use_kv_cache = use_kv_cache
        self.kv_cache_strategy = kv_cache_strategy

        # Mixed precision settings
        self.mixed_precision = mixed_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # Multi-Latent Attention settings
        self.latent_dim_scale = latent_dim_scale
        self.n_latents = n_latents if n_latents is not None else n_head // 2

        # Performance settings
        self.fused_operations = fused_operations
        self.use_fused_mlp = use_fused_mlp
        self.use_fused_attention = use_fused_attention

        # Advanced optimization settings
        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.gradient_checkpointing_policy = gradient_checkpointing_policy
        self.use_sdpa = use_sdpa
        self.use_parallel_attention = use_parallel_attention
        self.parallel_block_size = parallel_block_size

    def update(self, **kwargs):
        """Update config parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

    @classmethod
    def from_pretrained(cls, model_name: str) -> "GPTConfig":
        """Create config from pretrained model name."""
        # This would typically load from a config file
        # Here we just provide some example configurations
        configs = {
            "gpt2": dict(
                vocab_size=50257,
                n_layer=12,
                n_head=12,
                n_embd=768,
            ),
            "gpt2-medium": dict(
                vocab_size=50257,
                n_layer=24,
                n_head=16,
                n_embd=1024,
            ),
            "gpt2-large": dict(
                vocab_size=50257,
                n_layer=36,
                n_head=20,
                n_embd=1280,
            ),
            "gpt2-xl": dict(
                vocab_size=50257,
                n_layer=48,
                n_head=25,
                n_embd=1600,
            ),
        }

        if model_name not in configs:
            raise ValueError(f"Unknown model name: {model_name}")

        return cls(**configs[model_name])


class OptimizedKVCache:
    def __init__(
        self,
        max_seq_len: int,
        n_head: int,
        head_dim: int,
        device: torch.device,
    ):
        self.max_seq_len = max_seq_len
        self.n_head = n_head
        self.head_dim = head_dim
        self.device = device
        self.current_batch_size = None
        self.cache_k = None
        self.cache_v = None
        self.curr_len = 0

    def _ensure_cache_size(self, batch_size: int):
        """Dynamically resize cache if needed"""
        if self.cache_k is None or self.current_batch_size != batch_size:
            self.current_batch_size = batch_size
            self.cache_k = torch.zeros(
                (batch_size, self.n_head, self.max_seq_len, self.head_dim),
                pin_memory=True,
                device=self.device,
            )
            self.cache_v = torch.zeros_like(self.cache_k)
            self.curr_len = 0

    def update(self, key: torch.Tensor, value: torch.Tensor, position: int):
        """Update cache with new key-value pairs."""
        batch_size = key.size(0)
        self._ensure_cache_size(batch_size)

        self.cache_k[:, :, position : position + key.size(2)] = key
        self.cache_v[:, :, position : position + value.size(2)] = value
        self.curr_len = max(self.curr_len, position + key.size(2))

    def get_kv(self, position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached key-value pairs."""
        if self.cache_k is None:
            raise RuntimeError("Cache not initialized. Call update() first.")
        return (self.cache_k[:, :, :position], self.cache_v[:, :, :position])

    def clear(self):
        """Clear the cache."""
        self.cache_k = None
        self.cache_v = None
        self.curr_len = 0
        self.current_batch_size = None


class SparseMoE(nn.Module):
    """Sparse Mixture of Experts with dynamic pruning."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.sparsity_threshold = 0.1
        self.num_experts = config.num_experts

        # Sparse expert allocation
        self.sparse_allocation = torch.zeros(config.num_experts)
        self.expert_masks = nn.Parameter(torch.ones(config.num_experts))

    def _prune_experts(self, usage_stats: torch.Tensor):
        """Dynamically prune inactive experts."""
        normalized_usage = usage_stats / usage_stats.sum()
        self.expert_masks.data = (normalized_usage > self.sparsity_threshold).float()

    def forward(self, x: torch.Tensor, router_outputs: torch.Tensor) -> torch.Tensor:
        # Apply expert masks
        masked_routing = router_outputs * self.expert_masks
        return masked_routing


@triton.jit
def fused_rope_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    stride_q,
    stride_k,
    stride_v,
    stride_out,
    seq_len,
    head_dim,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RoPE and attention computation."""
    pid = tl.program_id(0)

    # Load inputs
    q = tl.load(q_ptr + pid * stride_q)
    k = tl.load(k_ptr + pid * stride_k)
    v = tl.load(v_ptr + pid * stride_v)

    # Compute RoPE
    position = pid % seq_len
    freq = tl.arange(0, head_dim, 2)
    freq = 1.0 / (10000 ** (freq / head_dim))
    rope = tl.cos(position * freq)

    # Apply RoPE
    q_rotated = q * rope
    k_rotated = k * rope

    # Compute attention
    scores = tl.sum(q_rotated * k_rotated) * scale
    scores = tl.softmax(scores)
    out = scores * v

    # Store output
    tl.store(out_ptr + pid * stride_out, out)


class QuantizedMoERouter(nn.Module):
    """8-bit quantized MoE router with caching."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.cache_size = 1024
        self.router_cache = {}

        # 8-bit quantization parameters
        self.scale = nn.Parameter(torch.ones(1))
        self.zero_point = nn.Parameter(torch.zeros(1))

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x / self.scale + self.zero_point).clamp(0, 255)

    def dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.zero_point) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check cache first
        cache_key = hash(x.data_ptr())
        if cache_key in self.router_cache:
            return self.router_cache[cache_key]

        # Quantize computation
        x_q = self.quantize(x)
        logits = self.dequantize(x_q)

        # Update cache
        if len(self.router_cache) >= self.cache_size:
            self.router_cache.clear()
        self.router_cache[cache_key] = logits

        return logits


class StreamingStateManager:
    """Manages streaming state for efficient inference."""

    def __init__(self, config: GPTConfig):
        self.max_sequence_length = config.block_size
        self.current_position = 0
        self.state_buffers = {}

    def update_state(self, layer_id: int, state: torch.Tensor):
        if layer_id not in self.state_buffers:
            self.state_buffers[layer_id] = torch.zeros(
                self.max_sequence_length, state.shape[-1]
            )
        start_idx = self.current_position
        end_idx = start_idx + state.shape[0]
        self.state_buffers[layer_id][start_idx:end_idx] = state

    def get_state(self, layer_id: int, position: int) -> torch.Tensor:
        return self.state_buffers[layer_id][:position]

    def reset(self):
        self.current_position = 0
        self.state_buffers.clear()


class AdaptiveQuantization(nn.Module):
    """Adaptive quantization based on activation patterns."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.num_bits = 8
        self.history_size = 1000
        self.activation_history = []

    def update_statistics(self, x: torch.Tensor):
        if len(self.activation_history) >= self.history_size:
            self.activation_history.pop(0)
        self.activation_history.append(x.abs().mean().item())

    def get_optimal_scale(self) -> float:
        if not self.activation_history:
            return 1.0
        max_val = max(self.activation_history)
        return (2**self.num_bits - 1) / max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.update_statistics(x)
        scale = self.get_optimal_scale()
        x_q = torch.round(x * scale).clamp(
            -(2 ** (self.num_bits - 1)), 2 ** (self.num_bits - 1) - 1
        )
        return x_q / scale


class FlashAttention(nn.Module):
    """
    Optimized Flash Attention implementation using torch.nn.functional.scaled_dot_product_attention
    with support for memory efficient attention and sliding window attention.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Basic attention parameters
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections for Q, K, V
        self.q_proj = FusedLinear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = FusedLinear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = FusedLinear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = FusedLinear(
            config.n_embd, config.n_embd, bias=config.bias, dropout=config.dropout
        )

        # Additional configurations
        self.use_memory_efficient = config.use_memory_efficient_attention
        self.use_sdpa = config.use_sdpa
        self.use_parallel = config.use_parallel_attention
        self.parallel_block_size = config.parallel_block_size

        # Register buffer for RoPE
        angles = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("rope_angles", angles)

    def _apply_rope(self, x: torch.Tensor, position: int = 0) -> torch.Tensor:
        """Apply Rotary Position Embedding."""
        B, H, T, D = x.shape
        freqs = torch.outer(
            torch.arange(position, position + T, device=x.device), self.rope_angles
        ).view(1, 1, T, D // 2)

        # Reshape x for rotation
        x_rot = x.reshape(B, H, T, D // 2, 2)

        # Apply rotation
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        x_out = torch.stack(
            [
                x_rot[..., 0] * cos - x_rot[..., 1] * sin,
                x_rot[..., 1] * cos + x_rot[..., 0] * sin,
            ],
            dim=-1,
        )

        return x_out.reshape(B, H, T, D)

    def _memory_efficient_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Memory efficient attention implementation."""
        B, H, T, D = q.shape

        # Process in chunks to save memory
        chunk_size = min(T, 1024)  # Adjust based on available memory
        output = torch.zeros_like(q)

        for i in range(0, T, chunk_size):
            chunk_end = min(i + chunk_size, T)
            q_chunk = q[:, :, i:chunk_end]

            # Compute attention scores for chunk
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale

            if mask is not None:
                chunk_mask = (
                    mask[:, :, i:chunk_end] if mask.dim() == 4 else mask[:, i:chunk_end]
                )
                scores = scores + chunk_mask

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training
            )

            # Update output
            output[:, :, i:chunk_end] = torch.matmul(attn_weights, v)

        return output

    def _parallel_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Parallel attention implementation using blocks."""
        B, H, T, D = q.shape
        block_size = self.parallel_block_size

        # Pad sequence length to be divisible by block_size
        pad_len = (block_size - T % block_size) % block_size
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            if mask is not None:
                mask = F.pad(mask, (0, pad_len))

        # Reshape into blocks
        new_T = T + pad_len
        q = q.view(B, H, new_T // block_size, block_size, D)
        k = k.view(B, H, new_T // block_size, block_size, D)
        v = v.view(B, H, new_T // block_size, block_size, D)

        # Compute attention for each block independently
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.view(B, H, new_T // block_size, block_size, block_size)
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        output = torch.matmul(attn_weights, v)

        # Reshape back and remove padding
        output = output.view(B, H, new_T, D)
        if pad_len > 0:
            output = output[:, :, :T]

        return output

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[OptimizedKVCache] = None,
        position: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass with support for various attention optimizations.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, embedding_dim]
            mask: Optional attention mask
            kv_cache: Optional key-value cache for inference
            position: Current position for positional encoding

        Returns:
            Output tensor of shape [batch_size, sequence_length, embedding_dim]
        """
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Apply RoPE
        q = self._apply_rope(q, position)
        k = self._apply_rope(k, position)

        # Handle KV cache during inference
        if kv_cache is not None:
            if position > 0:
                k_cache, v_cache = kv_cache.get_kv(position)
                k = torch.cat([k_cache, k[:, :, -1:]], dim=2)
                v = torch.cat([v_cache, v[:, :, -1:]], dim=2)
            kv_cache.update(k, v, position)

        # Choose attention implementation
        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            # Use native scaled dot product attention if available
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale,
            )
        elif self.use_memory_efficient:
            output = self._memory_efficient_attention(q, k, v, mask)
        elif self.use_parallel:
            output = self._parallel_attention(q, k, v, mask)
        else:
            # Fallback to standard attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores + mask
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training
            )
            output = torch.matmul(attn_weights, v)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(output)


class GPTBlock(nn.Module):
    """Transformer block for GPT."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # Choose between standard attention and optimized multi-latent attention
        if config.use_flash_attn and hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        ):
            self.attn = FlashAttention(
                config
            )  # A hypothetical flash attention implementation
        else:
            self.attn = OptimizedMultiLatentAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)

        # Choose between MoE and standard MLP
        if config.use_moe:
            self.mlp = OptimizedMoELayer(config)
        else:
            self.mlp = nn.Sequential(
                FusedLinear(config.n_embd, 4 * config.n_embd, bias=config.bias),
                nn.GELU(),
                FusedLinear(
                    4 * config.n_embd,
                    config.n_embd,
                    bias=config.bias,
                    dropout=config.dropout,
                ),
            )

        # Optional activation checkpointing
        self.use_checkpoint = config.use_activation_checkpointing

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[OptimizedKVCache] = None,
        position: int = 0,
    ) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mask, kv_cache, position
            )
        return self._forward(x, mask, kv_cache, position)

    def _forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_cache: Optional[OptimizedKVCache],
        position: int,
    ) -> torch.Tensor:
        # Attention with residual connection
        x = x + self.attn(self.ln_1(x), mask=mask, kv_cache=kv_cache, position=position)
        # MLP/MoE with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Complete GPT model with all optimizations."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embeddings with weight tying support
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Initialize weights
        self.apply(self._init_weights)

        # Optional weight quantization
        if config.use_quantization:
            self.quantize_weights()

        # KV cache for efficient inference
        self.kv_cache = (
            None
            if not config.use_kv_cache
            else OptimizedKVCache(
                max_seq_len=config.block_size,
                n_head=config.n_head,
                head_dim=config.n_embd // config.n_head,
                device=next(self.parameters()).device,
            )
        )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def quantize_weights(self):
        """Quantize model weights if quantization is enabled."""
        for module in self.modules():
            if isinstance(module, (QuantizedLinear, QuantizedExpertLayer)):
                if hasattr(module, "quantize_weights"):
                    module.quantize_weights()

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wte.weight.numel()
        return n_params

    def forward(
        self, idx: torch.Tensor, mask: Optional[torch.Tensor] = None, position: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.
        Args:
            idx: Input token indices [batch_size, sequence_length]
            mask: Attention mask [batch_size, sequence_length]
            position: Position for KV cache
        Returns:
            tuple: (logits, loss if training)
        """
        device = idx.device
        b, t = idx.size()

        # Assert sequence length is within block size
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"

        # Get token embeddings
        x = self.wte(idx)

        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask, kv_cache=self.kv_cache, position=position)

        # Final layer norm
        x = self.ln_f(x)

        # Get logits by using embedding weight matrix (weight tying)
        logits = F.linear(x, self.wte.weight)

        # Calculate loss if in training mode
        loss = None
        if self.training:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        Args:
            idx: Context token indices [batch_size, sequence_length]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Number of top tokens to sample from
            eos_token_id: Token ID for end of sequence
        Returns:
            torch.Tensor: Generated token indices
        """
        # Set model to eval mode
        self.eval()

        # Clear KV cache if using it
        if self.kv_cache is not None:
            self.kv_cache.clear()

        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            # Forward pass
            logits, _ = self(idx_cond, position=idx.size(1))
            logits = logits[:, -1, :]  # Take last timestep

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply softmax and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # Check for EOS token
            if eos_token_id is not None and (idx_next == eos_token_id).any():
                break

        return idx

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.AdamW:
        """
        Configure optimizer with weight decay fix.
        """
        # Separate parameters that should and shouldn't have weight decay
        decay = set()
        no_decay = set()

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, (nn.Linear, nn.Embedding)):
                    decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer


class QuantizedLinear(nn.Module):
    """Memory-efficient linear layer with dynamic quantization support."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, bits: int = 8
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Quantization parameters
        self.register_buffer("scale", None)
        self.register_buffer("zero_point", None)

        # Per-channel quantization parameters
        self.register_buffer("weight_scale", None)
        self.register_buffer("weight_zero_point", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _quantize_weight(self):
        """Quantize weights using per-channel dynamic quantization."""
        if self.weight_scale is None:
            # Compute per-channel scaling factors
            weight_max = torch.max(self.weight.abs(), dim=1, keepdim=True)[0]
            self.weight_scale = (2 * weight_max) / (2**self.bits - 1)
            self.weight_zero_point = torch.zeros_like(self.weight_scale)

        # Quantize weights
        weight_q = torch.clamp(
            torch.round(self.weight / self.weight_scale + self.weight_zero_point),
            0,
            2**self.bits - 1,
        )

        # Dequantize for computation
        return (weight_q - self.weight_zero_point) * self.weight_scale

    def _quantize_input(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize input tensor dynamically."""
        with torch.no_grad():
            x_max = torch.max(torch.abs(x))
            scale = (2 * x_max) / (2**self.bits - 1)
            zero_point = torch.zeros_like(scale)

            # Quantize input
            x_q = torch.clamp(torch.round(x / scale + zero_point), 0, 2**self.bits - 1)

            # Dequantize for computation
            x_dq = (x_q - zero_point) * scale
            return x_dq, scale, zero_point

    @torch.jit.script
    def _compute_output(
        self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """JIT-compiled forward computation."""
        output = F.linear(x, weight, bias)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic quantization during inference."""
        if self.training:
            return F.linear(x, self.weight, self.bias)

        # Quantize weights and input
        weight_q = self._quantize_weight()
        x_q, _, _ = self._quantize_input(x)

        # Compute output with quantized values
        return self._compute_output(x_q, weight_q, self.bias)


# 1. Optimized Multi-Latent Attention with Triton kernel
@triton.jit
def fused_multi_latent_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    latent_q_ptr,
    latent_k_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_lqb,
    stride_lqh,
    stride_lqm,
    stride_lkb,
    stride_lkh,
    stride_lkn,
    stride_ob,
    stride_oh,
    stride_om,
    B,
    H,
    M,
    N,
    L,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused multi-latent attention kernel using Triton.
    Handles both regular attention and latent attention components.
    """
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(M + L, BLOCK_M)
    block_id = pid // num_blocks
    block_m = pid % num_blocks

    # Compute offsets
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Load query block (including latent queries)
    is_latent = offs_m >= M
    q = tl.where(
        is_latent,
        tl.load(latent_q_ptr + (offs_m - M)[:, None] * stride_lqm),
        tl.load(q_ptr + offs_m[:, None] * stride_qm),
    )

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Process regular and latent keys/values
    for k in range(0, N + L, BLOCK_N):
        k_is_latent = k >= N
        k_offs = k - (N if k_is_latent else 0)

        # Load keys (regular or latent)
        k_vals = tl.where(
            k_is_latent,
            tl.load(latent_k_ptr + k_offs * stride_lkn),
            tl.load(k_ptr + k_offs * stride_kn),
        )

        # Load values (zero for latent components)
        v_vals = tl.where(
            k_is_latent, tl.zeros_like(k_vals), tl.load(v_ptr + k_offs * stride_vn)
        )

        # Compute attention scores and update accumulator
        acc += tl.dot(q, k_vals) * v_vals

    # Scale and store output
    scale = 1.0 / tl.sqrt(float(N + L))
    acc = acc * scale

    # Only store non-latent outputs
    tl.store(out_ptr + offs_m * stride_om, acc.to(tl.float16), mask=(offs_m < M))


class OptimizedPagedMultiLatentAttention(nn.Module):
    """Paged multi-latent attention with memory optimizations."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_latents = config.n_head // 2
        self.head_dim = config.n_embd // config.n_head
        self.latent_dim = self.head_dim * 2
        self.scale = math.sqrt(self.head_dim)
        self.page_size = getattr(config, "page_size", 16)

        # Latent parameters with memory pinning
        self.latent_queries = nn.Parameter(
            torch.randn(1, self.n_latents, self.latent_dim) / math.sqrt(self.latent_dim)
        )
        self.latent_keys = nn.Parameter(
            torch.randn(1, self.n_latents, self.latent_dim) / math.sqrt(self.latent_dim)
        )

        # Projections
        self.to_queries = FusedLinear(config.n_embd, config.n_embd, bias=config.bias)
        self.to_keys = FusedLinear(config.n_embd, config.n_embd, bias=config.bias)
        self.to_values = FusedLinear(config.n_embd, config.n_embd, bias=config.bias)

        # Latent projections
        self.to_latent_q = QuantizedLinear(
            self.latent_dim, config.n_embd, bias=config.bias
        )
        self.to_latent_k = QuantizedLinear(
            self.latent_dim, config.n_embd, bias=config.bias
        )

        # Output projection
        self.proj = FusedLinear(
            config.n_embd, config.n_embd, bias=config.bias, dropout=config.dropout
        )

        # RoPE parameters
        self.rope_theta = config.rope_theta
        self.max_seq_len = config.block_size
        self.register_buffer("freqs_cis", None)

        # Flash Attention support
        self.use_flash_attention = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )

    def _process_page(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        latent_q: torch.Tensor,
        latent_k: torch.Tensor,
        page_idx: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, H, D = q.shape
        start_idx = page_idx * self.page_size
        end_idx = min(start_idx + self.page_size, T)

        # Get current page
        page_q = q[:, start_idx:end_idx]
        page_k = torch.cat([k[:, :end_idx], latent_k], dim=1)
        page_v = torch.cat([v[:, :end_idx], latent_q], dim=1)

        if torch.cuda.is_available() and q.is_cuda:
            return self._process_page_triton(page_q, page_k, page_v, start_idx, end_idx)
        elif self.use_flash_attention:
            page_mask = mask[:, :, start_idx:end_idx] if mask is not None else None
            return F.scaled_dot_product_attention(
                page_q,
                page_k,
                page_v,
                attn_mask=page_mask,
                scale=self.scale,
                dropout_p=0.0 if not self.training else 0.1,
            )
        else:
            # Standard attention for CPU
            scores = torch.matmul(page_q, page_k.transpose(-2, -1)) / self.scale
            if mask is not None:
                scores = scores + mask[:, :, start_idx:end_idx, :end_idx]
            attn_weights = F.softmax(scores, dim=-1)
            return torch.matmul(attn_weights, page_v)

    def _process_page_triton(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor:
        B = q.size(0)
        page_size = end_idx - start_idx

        # Get appropriate kernel
        kernel = get_paged_attention_kernel(page_size, self.head_dim)
        grid = (B * self.n_head,)

        # Reshape for Triton kernel
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        output = torch.empty_like(q)

        # Launch kernel
        kernel[grid](
            q,
            k,
            v,
            output,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            start_idx,
            end_idx,
            BLOCK_M=32,
            BLOCK_N=32,
        )

        return output.permute(0, 2, 1, 3)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[OptimizedKVCache] = None,
        position: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.size()

        # Initialize RoPE frequencies if needed
        if self.freqs_cis is None or self.freqs_cis.device != x.device:
            self.freqs_cis = optimized_precompute_freqs_cis(
                self.head_dim, self.max_seq_len, self.rope_theta
            ).to(x.device)

        # Project inputs
        q = self.to_queries(x).view(B, T, self.n_head, -1)
        k = self.to_keys(x).view(B, T, self.n_head, -1)
        v = self.to_values(x).view(B, T, self.n_head, -1)

        # Apply RoPE
        q = apply_rotary_emb(q, self.freqs_cis[:T])
        k = apply_rotary_emb(k, self.freqs_cis[:T])

        # Process latent components
        latent_q = self.latent_queries.expand(B, -1, -1)
        latent_k = self.latent_keys.expand(B, -1, -1)

        latent_q = self.to_latent_q(latent_q).view(B, self.n_latents, self.n_head, -1)
        latent_k = self.to_latent_k(latent_k).view(B, self.n_latents, self.n_head, -1)

        # Handle KV cache
        if kv_cache is not None and position > 0:
            k_cached, v_cached = kv_cache.get_kv(position)
            k = torch.cat([k_cached, k[:, -1:]], dim=1)
            v = torch.cat([v_cached, v[:, -1:]], dim=1)
            kv_cache.update(k, v, position)

        # Initialize output
        output = torch.zeros_like(q)

        # Process pages
        n_pages = (T + self.page_size - 1) // self.page_size
        for page_idx in range(n_pages):
            page_output = self._process_page(
                q, k, v, latent_q, latent_k, page_idx, mask
            )
            start_idx = page_idx * self.page_size
            end_idx = min(start_idx + self.page_size, T)
            output[:, start_idx:end_idx] = page_output

        # Final projection
        output = output.view(B, T, C)
        return self.proj(output)


class FusedLinear(nn.Module):
    """Memory-efficient linear layer with fused operations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropout: float = 0.0,
        activation: Optional[str] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.activation = activation

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Compute linear transformation
        output = F.linear(input, self.weight, self.bias)

        # Apply activation if specified
        if self.activation == "gelu":
            output = F.gelu(output)

        # Apply dropout during training
        if self.dropout > 0 and self.training:
            output = F.dropout(output, p=self.dropout, training=True)

        return output


# Optimized precomputation of RoPE frequencies
@torch.jit.script
def optimized_precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0
) -> torch.Tensor:
    """JIT-compiled frequency precomputation for better performance."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


@torch.jit.script
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor using precomputed frequencies.

    Args:
        x: Input tensor of shape [B, T, H, D]
        freqs_cis: Precomputed frequencies of shape [T, D/2]

    Returns:
        Tensor with rotary embeddings applied
    """
    # Extract shapes
    B, T, H, D = x.shape

    # Reshape x for complex multiplication explicitly
    x_reshaped = x.float().reshape(B, T, H, D // 2, 2)
    x_complex = torch.view_as_complex(x_reshaped)

    # Expand freqs_cis to match batch and head dimensions
    freqs_cis = freqs_cis[:T, None, :]  # [T, 1, D/2]
    freqs_cis = freqs_cis.expand(T, H, x_complex.shape[-1])  # [T, H, D/2]

    # Apply rotary embeddings through complex multiplication
    x_out = torch.view_as_real(x_complex * freqs_cis.unsqueeze(0))  # [B, T, H, D/2, 2]

    # Reshape back to original shape
    x_out = x_out.reshape(B, T, H, D)

    return x_out.type_as(x)


@triton.jit
def fused_paged_attention_kernel(
    # Pointers to matrices
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    # Matrix strides
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    # Attention-specific params
    start_idx: tl.constexpr,
    end_idx: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # Optional params
    scale: tl.constexpr = None,
):
    """
    Compute paged attention for a single block.
    Uses Triton for efficient GPU computation.

    Args:
        q_ptr: Query tensor pointer [B, H, page_size, D]
        k_ptr: Key tensor pointer [B, H, N, D]
        v_ptr: Value tensor pointer [B, H, N, D]
        output_ptr: Output tensor pointer [B, H, page_size, D]
        Various strides for tensor access
        start_idx, end_idx: Current page bounds
        BLOCK_M, BLOCK_N: Block sizes for tiling
    """
    # Program ID
    pid = tl.program_id(0)

    # Page dimensions
    page_size = end_idx - start_idx

    # Block dimensions
    n_blocks = tl.cdiv(page_size, BLOCK_M)
    block_id = pid // n_blocks
    block_m = pid % n_blocks

    # Offsets
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load query block
    q = tl.load(q_ptr + offs_m[:, None] * stride_qm)

    # Process key-value pairs in blocks
    for k_start in range(0, end_idx, BLOCK_N):
        k_block = k_start + offs_n
        k_mask = k_block < end_idx

        # Load key and value blocks
        k = tl.load(k_ptr + k_block * stride_kn, mask=k_mask)
        v = tl.load(v_ptr + k_block * stride_vn, mask=k_mask)

        # Compute attention scores for this block
        scores = tl.dot(q, k.transpose(1, 0))
        if scale is not None:
            scores = scores * scale

        # Apply softmax
        scores = tl.softmax(scores, axis=-1)

        # Update accumulator
        acc += tl.dot(scores, v)

    # Write output
    output_offset = block_id * stride_ob + offs_m[:, None] * stride_om
    tl.store(output_ptr + output_offset, acc.to(tl.float16))


# Modified version of fused_paged_attention_kernel for larger pages
@triton.jit
def fused_paged_attention_kernel_large(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    start_idx: tl.constexpr,
    end_idx: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Version of the kernel optimized for larger page sizes.
    Includes additional tiling over the model dimension.
    """
    # Similar structure to above, but with additional tiling over D dimension
    pid = tl.program_id(0)

    # Calculate indices for this block
    num_block_m = tl.cdiv(page_size, BLOCK_M)
    num_block_n = tl.cdiv(end_idx, BLOCK_N)
    num_block_d = tl.cdiv(BLOCK_DMODEL, 32)

    # Get block indices
    block_m = pid % num_block_m
    block_n = (pid // num_block_m) % num_block_n
    block_d = (pid // (num_block_m * num_block_n)) % num_block_d

    # Offsets for this block
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = block_d * 32 + tl.arange(0, 32)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, 32), dtype=tl.float32)

    # Main computation
    q = tl.load(q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :])
    k = tl.load(k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :])
    v = tl.load(v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :])

    # Compute scores and attention
    scores = tl.dot(q, k.transpose(1, 0))
    scores = tl.softmax(scores, axis=-1)
    acc += tl.dot(scores, v)

    # Store results
    output_offset = offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(output_ptr + output_offset, acc.to(tl.float16))


def get_paged_attention_kernel(page_size: int, head_dim: int) -> Callable:
    """
    Returns the appropriate kernel based on page and model size.

    Args:
        page_size: Size of attention pages
        head_dim: Dimension of attention heads

    Returns:
        Appropriate Triton kernel for these dimensions
    """
    if page_size <= 32 and head_dim <= 64:
        return fused_paged_attention_kernel
    else:
        return fused_paged_attention_kernel_large


# Optimized MoE Implementation
@triton.jit
def fused_expert_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_out,
):
    """Fused expert computation kernel using Triton."""
    pid = tl.program_id(0)

    # Compute offsets
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator for matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate through K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K) * BLOCK_K, BLOCK_K):
        # Load x and weight blocks
        x_block_ptr = (
            x_ptr + offs_m[:, None] * stride_xm + (k + offs_k[None, :]) * stride_xk
        )
        w_block_ptr = (
            weight_ptr + (k + offs_k[:, None]) * stride_wk + offs_n[None, :] * stride_wn
        )

        x_block = tl.load(x_block_ptr, mask=offs_m[:, None] < M, other=0.0)
        w_block = tl.load(w_block_ptr, mask=offs_n[None, :] < N, other=0.0)

        # Compute matrix multiplication
        acc += tl.dot(x_block, w_block)

    # Add bias if present
    if has_bias:
        bias = tl.load(bias_ptr + offs_m, mask=offs_m < M, other=0.0)
        acc = acc + bias[:, None]

    # Apply ReLU activation
    acc = tl.maximum(acc, 0.0)

    # Store the result
    for n in range(BLOCK_N):
        out_ptr_n = out_ptr + offs_m * stride_out + n
        mask = offs_m < M
        tl.store(out_ptr_n, acc[:, n], mask=mask)


class OptimizedMoELayer(nn.Module):
    """Optimized Mixture of Experts layer with quantization and Triton acceleration."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.expert_capacity_factor = config.expert_capacity_factor
        self.hidden_dim = config.n_embd
        self.ffn_dim = 4 * config.n_embd
        self.dropout = config.dropout

        # Create experts (quantized)
        self.experts = nn.ModuleList(
            [QuantizedExpertLayer(config) for _ in range(self.num_experts)]
        )

        # Optimized router with quantization
        self.router = QuantizedLinear(
            config.n_embd, self.num_experts, bias=False, bits=config.quantization_bits
        )

        # Load balancing tracking
        self.register_buffer("expert_counts", torch.zeros(self.num_experts))

        # Dropout for routing
        self.route_dropout = nn.Dropout(config.dropout)

        # Initialize router weights
        with torch.no_grad():
            self.router.weight.data.normal_(mean=0.0, std=0.02)

    def _compute_routing_weights(
        self, x: torch.Tensor, importance_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing weights with dynamic capacity and load balancing."""
        batch_size, seq_len, _ = x.shape

        # Get router logits
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]

        # Apply importance scoring if provided
        if importance_scores is not None:
            router_logits = router_logits * importance_scores.unsqueeze(-1)

        # Calculate expert capacity
        capacity = int(
            self.expert_capacity_factor
            * seq_len
            * self.num_experts_per_tok
            / self.num_experts
        )

        # Get top-k experts per token
        router_probs = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )

        # Normalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        # Create dispatch tensors - ensure same dtype as expert_weights
        dispatch_mask = torch.zeros_like(router_logits, dtype=expert_weights.dtype)
        dispatch_mask.scatter_(-1, expert_indices, expert_weights)

        # Implement load balancing
        position_in_expert = torch.cumsum(dispatch_mask, dim=1)
        capacity_mask = position_in_expert <= capacity
        dispatch_mask = dispatch_mask * capacity_mask.float().to(dispatch_mask.dtype)

        # Renormalize weights
        dispatch_mask = dispatch_mask / (dispatch_mask.sum(dim=-1, keepdim=True) + 1e-6)

        return dispatch_mask, expert_indices

    def _update_expert_counts(self, dispatch_mask: torch.Tensor):
        """Track expert usage for load balancing."""
        with torch.no_grad():
            token_counts = dispatch_mask.sum(dim=(0, 1))
            self.expert_counts.mul_(0.9).add_(token_counts, alpha=0.1)

    def get_load_balancing_loss(self, dispatch_mask: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss with improved stability."""
        # Calculate expert assignment ratios
        expert_ratios = dispatch_mask.sum(dim=(0, 1)) / dispatch_mask.sum()
        target_ratio = 1.0 / self.num_experts

        # Compute variance from ideal uniform distribution with stability term
        load_balancing_loss = torch.sum(
            (expert_ratios - target_ratio).pow(2) / (expert_ratios + 1e-6)
        )

        return load_balancing_loss * 0.01  # Scale factor

    def forward(
        self, x: torch.Tensor, importance_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optimized expert computation and load balancing."""
        batch_size, seq_len, hidden_dim = x.shape

        # Get routing weights and indices
        dispatch_mask, expert_indices = self._compute_routing_weights(
            x, importance_scores
        )

        # Update expert usage statistics
        self._update_expert_counts(dispatch_mask)

        # Initialize output tensor
        final_output = torch.zeros_like(x)

        # Process each expert
        for i, expert in enumerate(self.experts):
            # Get tokens routed to this expert
            expert_mask = dispatch_mask[:, :, i].unsqueeze(-1)
            expert_input = x * expert_mask

            # Skip computation if no tokens are routed here
            if expert_mask.sum() == 0:
                continue

            # Process tokens through expert
            if torch.cuda.is_available() and x.is_cuda:
                # Optimize for GPU by doing the computation in a single batch
                # Reshape for efficient computation
                expert_input = expert_input.view(-1, hidden_dim)
                expert_output = expert(expert_input)
                expert_output = expert_output.view(batch_size, seq_len, -1)
            else:
                # Standard CPU processing
                expert_output = expert(expert_input)

            final_output = final_output + expert_output

        return final_output


class QuantizedExpertLayer(nn.Module):
    """Expert layer with dynamic quantization and fused operations."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.w1 = QuantizedLinear(
            config.n_embd,
            4 * config.n_embd,
            bias=config.bias,
            bits=config.quantization_bits,
        )
        self.w2 = QuantizedLinear(
            4 * config.n_embd,
            config.n_embd,
            bias=config.bias,
            bits=config.quantization_bits,
        )
        self.dropout = nn.Dropout(config.dropout)

        # Dynamic quantization parameters
        self.bits = config.quantization_bits
        self.scale_w1 = nn.Parameter(torch.ones(1))
        self.scale_w2 = nn.Parameter(torch.ones(1))
        self.register_buffer("zero_point_w1", torch.zeros(1))
        self.register_buffer("zero_point_w2", torch.zeros(1))

    def quantize_weights(self):
        """Dynamically quantize weights using learned scaling factors."""
        w1_range = self.w1.weight.abs().max()
        w2_range = self.w2.weight.abs().max()

        self.scale_w1.data = (2 * w1_range) / (2**self.bits - 1)
        self.scale_w2.data = (2 * w2_range) / (2**self.bits - 1)

        w1_q = torch.clamp(
            torch.round(self.w1.weight / self.scale_w1 + self.zero_point_w1),
            0,
            2**self.bits - 1,
        )
        w2_q = torch.clamp(
            torch.round(self.w2.weight / self.scale_w2 + self.zero_point_w2),
            0,
            2**self.bits - 1,
        )

        self.w1.weight.data = (w1_q - self.zero_point_w1) * self.scale_w1
        self.w2.weight.data = (w2_q - self.zero_point_w2) * self.scale_w2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic quantization during inference."""
        if not self.training:
            self.quantize_weights()
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class CombinedAttention(nn.Module):
    """Combines Multi-Latent Attention with Paged processing and KV cache."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Multi-Latent parameters
        self.n_head = config.n_head
        self.n_latents = config.n_head // 2
        self.head_dim = config.n_embd // config.n_head
        self.latent_dim = self.head_dim * 2
        self.scale = math.sqrt(self.head_dim)

        # Paging parameters
        self.page_size = 16  # Can be tuned
        self.max_seq_len = config.block_size

        # Latent parameters
        self.latent_queries = nn.Parameter(
            torch.randn(1, self.n_latents, self.latent_dim) / math.sqrt(self.latent_dim)
        )
        self.latent_keys = nn.Parameter(
            torch.randn(1, self.n_latents, self.latent_dim) / math.sqrt(self.latent_dim)
        )

        # Projections
        self.to_queries = FusedLinear(config.n_embd, config.n_embd, bias=config.bias)
        self.to_keys = FusedLinear(config.n_embd, config.n_embd, bias=config.bias)
        self.to_values = FusedLinear(config.n_embd, config.n_embd, bias=config.bias)

        # Latent projections
        self.to_latent_q = QuantizedLinear(
            self.latent_dim, config.n_embd, bias=config.bias
        )
        self.to_latent_k = QuantizedLinear(
            self.latent_dim, config.n_embd, bias=config.bias
        )

        # Output projection
        self.proj = FusedLinear(
            config.n_embd, config.n_embd, bias=config.bias, dropout=config.dropout
        )

        # Flash Attention support
        self.use_flash_attention = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )

    def _process_page(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        latent_q: torch.Tensor,
        latent_k: torch.Tensor,
        mask: Optional[torch.Tensor],
        page_idx: int,
        page_size: int,
    ) -> torch.Tensor:
        """Process a single page of attention including latent components."""
        B, H, T, D = q.shape
        start_idx = page_idx * page_size
        end_idx = min(start_idx + page_size, T)

        # Get current page
        page_q = q[:, :, start_idx:end_idx]
        page_k = k[:, :, :end_idx]  # Use all previous keys up to current page
        page_v = v[:, :, :end_idx]  # Use all previous values up to current page

        # Combine with latent components
        combined_k = torch.cat([page_k, latent_k], dim=2)
        combined_v = torch.cat([page_v, latent_q], dim=2)

        if self.use_flash_attention:
            page_mask = (
                mask[:, :, start_idx:end_idx, :end_idx] if mask is not None else None
            )
            output = F.scaled_dot_product_attention(
                page_q,
                combined_k,
                combined_v,
                attn_mask=page_mask,
                dropout_p=0.0 if not self.training else 0.1,
                scale=self.scale,
            )
        else:
            # Regular attention computation
            scores = torch.matmul(page_q, combined_k.transpose(-2, -1)) / self.scale
            if mask is not None:
                page_mask = mask[:, :, start_idx:end_idx, :end_idx]
                scores = scores + page_mask
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, combined_v)

        return output

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[OptimizedKVCache] = None,
        position: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.size()

        # Project inputs
        q = self.to_queries(x).view(B, T, self.n_head, -1).transpose(1, 2)
        k = self.to_keys(x).view(B, T, self.n_head, -1).transpose(1, 2)
        v = self.to_values(x).view(B, T, self.n_head, -1).transpose(1, 2)

        # Process latent components
        latent_q = self.latent_queries.expand(B, -1, -1)
        latent_k = self.latent_keys.expand(B, -1, -1)

        latent_q = (
            self.to_latent_q(latent_q)
            .view(B, self.n_latents, self.n_head, -1)
            .transpose(1, 2)
        )
        latent_k = (
            self.to_latent_k(latent_k)
            .view(B, self.n_latents, self.n_head, -1)
            .transpose(1, 2)
        )

        # Handle KV cache during inference
        if kv_cache is not None:
            if position > 0:
                # Retrieve cached keys and values
                k_cached, v_cached = kv_cache.get_kv(position)
                # Only compute for new position
                k = torch.cat([k_cached, k[:, :, -1:]], dim=2)
                v = torch.cat([v_cached, v[:, :, -1:]], dim=2)
            # Update cache
            kv_cache.update(k, v, position)

            # For inference with KV cache, we only process the last token
            output = self._process_page(
                q[:, :, -1:],
                k,
                v,
                latent_q,
                latent_k,
                mask,
                0,
                1,  # Process single token as a page
            )
        else:
            # Training or inference without KV cache - use paged attention
            output = torch.zeros_like(q)
            n_pages = (T + self.page_size - 1) // self.page_size

            for page_idx in range(n_pages):
                page_output = self._process_page(
                    q, k, v, latent_q, latent_k, mask, page_idx, self.page_size
                )
                start_idx = page_idx * self.page_size
                end_idx = min(start_idx + self.page_size, T)
                output[:, :, start_idx:end_idx] = page_output

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(output)


class DualHeadGPTConfig(GPTConfig):
    """Extended GPT config with dual head support."""

    def __init__(
        self,
        second_vocab_size: int = 50257,  # Size of second vocabulary
        aux_head_weight: float = 0.5,  # Weight for auxiliary loss
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.second_vocab_size = second_vocab_size
        self.aux_head_weight = aux_head_weight


class DualHeadGPT(GPT):
    """GPT model with dual heads but single input."""

    def __init__(self, config: DualHeadGPTConfig):
        super().__init__(config)
        self.config = config

        # Second head for auxiliary task
        self.second_head = FusedLinear(
            config.n_embd, config.second_vocab_size, bias=config.bias
        )

    def forward(
        self,
        idx: torch.Tensor,
        target_second: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        position: int = 0,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Forward pass with dual heads.

        Args:
            idx: Input token indices [batch_size, sequence_length]
            target_second: Target indices for second head [batch_size, sequence_length]
            mask: Attention mask [batch_size, sequence_length]
            position: Position for KV cache

        Returns:
            tuple: (primary_logits, second_logits, primary_loss, second_loss)
        """
        device = idx.device
        b, t = idx.size()

        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"

        # Get token embeddings
        x = self.wte(idx)

        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask, kv_cache=self.kv_cache, position=position)

        # Final layer norm
        x = self.ln_f(x)

        # Get logits from both heads
        primary_logits = F.linear(x, self.wte.weight)  # Weight tying for primary
        second_logits = self.second_head(x)

        # Calculate losses if in training mode
        primary_loss = None
        second_loss = None
        if self.training:
            primary_loss = F.cross_entropy(
                primary_logits.view(-1, primary_logits.size(-1)), idx.view(-1)
            )
            if target_second is not None:
                second_loss = F.cross_entropy(
                    second_logits.view(-1, second_logits.size(-1)),
                    target_second.view(-1),
                )
                # Apply auxiliary loss weight
                second_loss *= self.config.aux_head_weight

        return primary_logits, second_logits, primary_loss, second_loss

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model."""
        n_params = super().get_num_params(non_embedding)
        n_params += self.second_head.weight.numel()
        if self.second_head.bias is not None:
            n_params += self.second_head.bias.numel()
        return n_params

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        target_head: str = "primary",  # 'primary' or 'second'
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively from either head.

        Args:
            idx: Context token indices [batch_size, sequence_length]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Number of top tokens to sample from
            eos_token_id: Token ID for end of sequence
            target_head: Which head to use for generation ('primary' or 'second')

        Returns:
            torch.Tensor: Generated token indices
        """
        self.eval()

        # Clear KV cache if using it
        if self.kv_cache is not None:
            self.kv_cache.clear()

        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            # Forward pass
            primary_logits, second_logits, _, _ = self(idx_cond, position=idx.size(1))

            # Select appropriate logits
            logits = primary_logits if target_head == "primary" else second_logits
            logits = logits[:, -1, :]  # Take last timestep

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply softmax and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # Check for EOS token
            if eos_token_id is not None and (idx_next == eos_token_id).any():
                break

        return idx
