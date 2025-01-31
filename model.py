import torch
import math
import time
import warnings
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Union, Dict
import triton
import triton.language as tl


# Todos:
# Consider implementing a streaming inference mode for handling long sequences
# Add support for sparse attention patterns


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
        num_experts_per_tok: int = 3,
        expert_capacity_factor: float = 1.5,
        moe_jitter_noise: float = 0.05,
        moe_aux_loss_scale: float = 0.3,
        # Memory optimization settings
        use_activation_checkpointing: bool = False,
        checkpoint_ratio: float = 0.5,
        use_kv_cache: bool = True,
        kv_cache_strategy: str = "dynamic",  # "dynamic" or "static"
        # Mixed precision settings
        mixed_precision: bool = True,
        mixed_precision_dtype: str = "bf16",  # "fp16" or "bf16"
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
        self.moe_aux_loss_scale = moe_aux_loss_scale

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
        self.use_half_precision = True

    def _ensure_cache_size(self, batch_size: int):
        """Dynamically resize cache if needed"""
        if self.cache_k is None or self.current_batch_size != batch_size:
            dtype = torch.float16 if self.use_half_precision else torch.float32
            self.current_batch_size = batch_size
            self.cache_k = torch.zeros(
                (batch_size, self.n_head, self.max_seq_len, self.head_dim),
                dtype=dtype,
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

    def _memory_efficient_attention(self, q, k, v, mask, chunk_size):
        B, H, T, D = q.shape
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

    def compute_optimal_chunk_size(self, seq_len):
        # Heuristic for optimal chunk size based on sequence length
        if seq_len <= 512:
            return 128
        elif seq_len <= 2048:
            return 256
        return 512

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

        chunk_size = min(self.compute_optimal_chunk_size(x.shape[1]), 1024)

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
            output = self._memory_efficient_attention(
                q, k, v, mask, chunk_size=chunk_size
            )

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
            self.attn = FlashAttention(config)
        else:
            self.attn = OptimizedPagedMultiLatentAttention(config)

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
    """
    Memory-efficient linear layer with advanced quantization support.
    Supports both dynamic and static quantization with per-channel or per-tensor schemes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8,
        per_channel: bool = True,
        symmetric: bool = True,
        static_quantization: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.static_quantization = static_quantization
        
        # Quantization parameters
        self.quant_min = 0
        self.quant_max = 2**bits - 1
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Quantization buffers
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_zero_point', None)
        self.register_buffer('input_scale', None)
        self.register_buffer('input_zero_point', None)
        
        # Cached quantized weights for inference
        self.register_buffer('weight_quantized', None)
        
        # Running estimates for static quantization
        if static_quantization:
            self.register_buffer('running_min', torch.zeros(1))
            self.register_buffer('running_max', torch.zeros(1))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with proper scaling."""
        # Use Kaiming initialization with correction for quantization
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        scale = math.sqrt(2. / fan_in) * (2 ** (self.bits - 1))
        nn.init.uniform_(self.weight, -scale, scale)
        
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _compute_weight_scaling_factors(self):
        """Compute weight scaling factors using improved calibration."""
        if self.per_channel:
            axis = 1  # per output channel
            keepdim = True
        else:
            axis = None
            keepdim = False

        if self.symmetric:
            weight_max = torch.max(torch.abs(self.weight), dim=axis, keepdim=keepdim)[0]
            weight_min = -weight_max
        else:
            weight_max = torch.max(self.weight, dim=axis, keepdim=keepdim)[0]
            weight_min = torch.min(self.weight, dim=axis, keepdim=keepdim)[0]

        # Improve stability with moving averages during training
        if self.training and hasattr(self, 'running_max'):
            momentum = 0.1
            self.running_max.mul_(1 - momentum).add_(weight_max.mean() * momentum)
            self.running_min.mul_(1 - momentum).add_(weight_min.mean() * momentum)
            
        scale = (weight_max - weight_min) / (self.quant_max - self.quant_min)
        zero_point = self.quant_min - torch.round(weight_min / scale)
        
        return scale, zero_point

    def quantize_weight(self):
        """Quantize weights with caching for inference."""
        if not self.training and self.weight_quantized is not None:
            return self.weight_quantized

        if self.weight_scale is None or self.training:
            self.weight_scale, self.weight_zero_point = self._compute_weight_scaling_factors()

        # Quantize weights
        weight_q = torch.clamp(
            torch.round(self.weight / self.weight_scale + self.weight_zero_point),
            self.quant_min,
            self.quant_max
        )

        # Dequantize
        weight_dq = (weight_q - self.weight_zero_point) * self.weight_scale

        # Cache for inference
        if not self.training:
            self.weight_quantized = weight_dq

        return weight_dq

    def quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize input with support for static quantization."""
        if self.static_quantization and self.input_scale is not None:
            scale = self.input_scale
            zero_point = self.input_zero_point
        else:
            with torch.no_grad():
                if self.symmetric:
                    x_max = torch.max(torch.abs(x))
                    x_min = -x_max
                else:
                    x_max = torch.max(x)
                    x_min = torch.min(x)

                # Update running estimates for static quantization
                if self.static_quantization and self.training:
                    self.num_batches_tracked += 1
                    momentum = 0.1
                    self.running_max.mul_(1 - momentum).add_(x_max * momentum)
                    self.running_min.mul_(1 - momentum).add_(x_min * momentum)

                scale = (x_max - x_min) / (self.quant_max - self.quant_min)
                zero_point = self.quant_min - torch.round(x_min / scale)

                if self.static_quantization and not self.training:
                    self.input_scale = scale
                    self.input_zero_point = zero_point

        # Quantize and dequantize
        x_q = torch.clamp(
            torch.round(x / scale + zero_point),
            self.quant_min,
            self.quant_max
        )
        return (x_q - zero_point) * scale

    @torch.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization aware training."""
        if self.training or not self.static_quantization:
            # Quantize weights and input
            weight_q = self.quantize_weight()
            x_q = self.quantize_input(x)
            
            # Compute output
            output = F.linear(x_q, weight_q, self.bias)
            
            # Apply straight-through estimator for gradients
            if self.training:
                output = output + (x - x_q).detach() + (self.weight - weight_q).detach()
            
            return output
        else:
            # Use cached quantized weights during inference
            if self.weight_quantized is None:
                self.weight_quantized = self.quantize_weight()
            return F.linear(self.quantize_input(x), self.weight_quantized, self.bias)

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}, '
                f'bits={self.bits}, '
                f'per_channel={self.per_channel}, '
                f'symmetric={self.symmetric}, '
                f'static={self.static_quantization}')


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
        self.use_sdpa = config.use_sdpa
        self.dropout = config.dropout

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

        self.out_proj = FusedLinear(
            config.n_embd, config.n_embd, bias=config.bias, dropout=config.dropout
        )

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

    def _apply_rope(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """
        Applies rotary positional embeddings to x in shape [B, H, T, D].
        """
        B, H, T, D = x.shape

        # Compute if needed
        if self.freqs_cis is None or self.freqs_cis.size(0) < (start_pos + T):
            # Recompute and store
            self.freqs_cis = optimized_precompute_freqs_cis(
                dim=D, end=self.max_seq_len, theta=self.rope_theta
            ).to(x.device)

        # Slice out needed positions
        freqs_slice = self.freqs_cis[start_pos : start_pos + T]  # shape [T, D//2]

        # Permute to [B, T, H, D] for apply_rotary_emb
        x = x.permute(0, 2, 1, 3)
        x = apply_rotary_emb(x, freqs_slice)  # returns [B, T, H, D]
        # Permute back
        x = x.permute(0, 2, 1, 3)
        return x

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
        """Process attention with proper tensor reshaping."""
        try:
            B, H, T, D = q.shape

            # Make tensors contiguous and reshape
            q = q.contiguous().view(B * H, T, D)
            k = k.contiguous().view(B * H, -1, D)  # -1 to handle variable length
            v = v.contiguous().view(B * H, -1, D)

            # Create output tensor
            output = torch.empty_like(q)

            # Get kernel config
            kernel_config = get_paged_attention_kernel(T, D)
            kernel = kernel_config["kernel"]
            kwargs = kernel_config["kwargs"]

            # Launch kernel
            grid = (B * H,)
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
                start_idx=start_idx,
                end_idx=end_idx,
                **kwargs,
            )

            # Reshape output back to original dimensions
            return output.view(B, H, T, D)

        except Exception as e:
            warnings.warn(f"Triton kernel failed, falling back to PyTorch: {str(e)}")
            return self._process_page_pytorch(q, k, v, start_idx, end_idx)

    def _process_page_pytorch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor:
        """PyTorch fallback implementation with proper sequence length handling."""
        # Get input dimensions
        B, H, T, D = q.shape
        _, _, S, _ = k.shape  # S is the source sequence length

        # Handle the page slice
        page_q = q[:, :, start_idx:end_idx]  # Take the current page from query

        # Compute attention with proper scaling
        scale = 1.0 / math.sqrt(D)

        # Compute attention scores for this page
        scores = (
            torch.matmul(page_q, k.transpose(-2, -1)) * scale
        )  # [B, H, page_size, S]

        # Apply softmax over the key dimension
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [B, H, page_size, D]

        # Create output tensor of the same size as input q
        full_output = torch.zeros_like(q)
        full_output[:, :, start_idx:end_idx] = output

        return full_output

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[OptimizedKVCache] = None,
        position: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass with improved tensor handling.
        """
        B, T, C = x.size()

        # Project to Q, K, V with proper reshaping
        q = (
            self.to_queries(x)
            .reshape(B, T, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )
        k = self.to_keys(x).reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = (
            self.to_values(x)
            .reshape(B, T, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )

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

        # Calculate chunk size based on sequence length
        chunk_size = min(self.compute_optimal_chunk_size(T), 1024)

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
            output = self._memory_efficient_attention(
                q, k, v, mask, chunk_size=chunk_size
            )
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

    def compute_optimal_chunk_size(self, seq_len):
        # Heuristic for optimal chunk size based on sequence length
        if seq_len <= 512:
            return 128
        elif seq_len <= 2048:
            return 256
        return 512


class FusedLinear(nn.Module):
    """
    Memory-efficient linear layer with fused operations for linear transform,
    activation, and dropout.
    """

    SUPPORTED_ACTIVATIONS = {
        "gelu": F.gelu,
        "relu": F.relu,
        "silu": F.silu,
        None: lambda x: x
    }

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropout: float = 0.0,
        activation: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize FusedLinear layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to include bias term
            dropout: Dropout probability
            activation: Activation function ('gelu', 'relu', 'silu', or None)
            device: Torch device
            dtype: Tensor dtype
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        # Validate inputs
        if not (0 <= dropout < 1.0):
            raise ValueError(f"Dropout must be in [0, 1), got {dropout}")
        if activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}. "
                           f"Choose from {list(self.SUPPORTED_ACTIVATIONS.keys())}")
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.activation_fn = self.SUPPORTED_ACTIVATIONS[activation]

        # Initialize parameters with proper dtype/device
        self.weight = nn.Parameter(torch.empty(
            (out_features, in_features), **factory_kwargs))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(activation)

    def reset_parameters(self, activation: Optional[str] = None) -> None:
        """
        Reset layer parameters with activation-aware initialization.
        """
        if activation == 'gelu':
            # Use He initialization adjusted for GELU
            gain = 1.0 / math.sqrt(0.8862)
            nn.init.kaiming_normal_(self.weight, a=math.sqrt(5), nonlinearity='leaky_relu')
            self.weight.data *= gain
        elif activation == 'relu':
            nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        else:
            # Default initialization for linear/other activations
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fused operations.
        
        Args:
            input: Input tensor of shape [..., in_features]
            
        Returns:
            Output tensor of shape [..., out_features]
        """
        # Validate input shape
        if input.size(-1) != self.in_features:
            raise ValueError(f"Expected input features: {self.in_features}, "
                           f"got: {input.size(-1)}")

        if torch.jit.is_scripting() or not self.training:
            # Standard forward pass when scripting or in eval mode
            output = F.linear(input, self.weight, self.bias)
            output = self.activation_fn(output)
            if self.dropout > 0 and self.training:
                output = F.dropout(output, p=self.dropout, training=True)
            return output
        else:
            # Fused operations for training mode
            if hasattr(torch.nn.functional, 'fused_linear'):
                # Use fused kernel if available
                return F.fused_linear(
                    input, 
                    self.weight,
                    self.bias,
                    self.activation_fn,
                    self.dropout if self.training else 0.0
                )
            else:
                # Fallback to standard operations
                output = F.linear(input, self.weight, self.bias)
                output = self.activation_fn(output)
                if self.dropout > 0:
                    output = F.dropout(output, p=self.dropout, training=True)
                return output

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}, '
                f'dropout={self.dropout}, '
                f'activation={[k for k, v in self.SUPPORTED_ACTIVATIONS.items() '
                f'if v == self.activation_fn][0]}')


# Optimized precomputation of RoPE frequencies
@torch.jit.script
def optimized_precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    JIT-compiled frequency precomputation with improved numerical stability.

    Args:
        dim: Model dimension
        end: Maximum sequence length
        theta: Base value for frequency computation

    Returns:
        Complex tensor of shape [end, dim//2] containing precomputed frequencies
    """
    # Compute frequencies with better numerical stability
    freq_seq = torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)]
    inv_freq = torch.exp(-torch.log(torch.tensor(theta)) * (freq_seq / dim))

    # Create position sequence
    pos_seq = torch.arange(end, dtype=torch.float32)

    # Compute outer product more efficiently
    freqs = torch.einsum("i,j->ij", pos_seq, inv_freq)

    # Convert to complex numbers using polar coordinates
    return torch.polar(torch.ones_like(freqs), freqs)


@torch.jit.script
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings with improved efficiency and stability.

    Args:
        x: Input tensor of shape [B, T, H, D]
        freqs_cis: Precomputed frequencies of shape [T, D/2]

    Returns:
        Tensor with rotary embeddings applied
    """
    B, T, H, D = x.shape

    # Input validation
    assert D % 2 == 0, f"Dimension {D} must be divisible by 2"
    assert (
        freqs_cis.shape[0] >= T
    ), f"freq_cis length {freqs_cis.shape[0]} too small for seq length {T}"

    # Reshape input maintaining original dtype
    x_reshaped = x.reshape(B, T, H, D // 2, 2)
    x_complex = torch.view_as_complex(x_reshaped)

    # Prepare frequencies more efficiently
    freqs_cis = freqs_cis[:T]  # [T, D/2]

    # Optimize the broadcasting
    x_complex = x_complex.view(B, T, H, -1)
    freqs_cis = freqs_cis.view(T, 1, -1)  # [T, 1, D/2]

    # Apply rotary embeddings
    x_out = torch.view_as_real(x_complex * freqs_cis).reshape(B, T, H, D)

    return x_out


@triton.jit
def fused_paged_attention_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fixed implementation of paged attention kernel."""
    # Get program ID
    pid = tl.program_id(0)

    # Calculate dimensions
    page_size = end_idx - start_idx
    n_blocks = tl.cdiv(page_size, BLOCK_M)

    # Calculate block indices
    block_id = pid // n_blocks
    block_m = pid % n_blocks

    # Calculate offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Add base offset for this block
    offs_m = block_m * BLOCK_M + offs_m

    # Create masks
    m_mask = offs_m < page_size

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load query block with proper pointer arithmetic
    q_block_ptr = q_ptr + (offs_m[:, None] * stride_qm)
    q = tl.load(q_block_ptr, mask=m_mask[:, None], other=0.0)

    # Process key-value pairs in blocks
    for k_start in range(0, end_idx, BLOCK_N):
        k_offs = k_start + offs_n
        k_mask = k_offs < end_idx

        # Load key and value blocks with correct pointer arithmetic
        k_block_ptr = k_ptr + (k_offs * stride_kn)
        v_block_ptr = v_ptr + (k_offs * stride_vn)

        k = tl.load(k_block_ptr, mask=k_mask, other=0.0)
        v = tl.load(v_block_ptr, mask=k_mask, other=0.0)

        # Compute attention scores without explicit indexing
        qk = tl.dot(q, k)
        qk = qk * (1.0 / tl.sqrt(float(BLOCK_K)))

        # Apply masking and softmax
        qk = tl.where(k_mask[None, :], qk, float("-inf"))
        qk = tl.softmax(qk, axis=-1)

        # Compute weighted sum
        acc += tl.dot(qk, v)

    # Store output
    out_ptr = output_ptr + (offs_m * stride_om)
    tl.store(out_ptr, acc.to(tl.float16), mask=m_mask)


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


def get_paged_attention_kernel(page_size: int, head_dim: int) -> dict:
    """Returns kernel configuration with fixed block sizes."""
    # Use smaller block sizes for better stability
    BLOCK_M = min(16, page_size)
    BLOCK_N = min(16, head_dim)
    BLOCK_K = min(16, head_dim)

    return {
        "kernel": fused_paged_attention_kernel,
        "kwargs": {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
    }


@triton.jit
def fused_paged_attention_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Simplified paged attention kernel."""
    # Get program ID
    pid = tl.program_id(0)

    # Calculate dimensions
    page_size = end_idx - start_idx

    # Calculate offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Create masks
    m_mask = offs_m < page_size

    # Load query block
    q = tl.load(q_ptr + offs_m * stride_qm, mask=m_mask, other=0.0)

    # Load key and value
    k = tl.load(k_ptr + offs_n * stride_kn)
    v = tl.load(v_ptr + offs_n * stride_vn)

    # Compute attention scores
    scores = tl.dot(q, k) / tl.sqrt(float(q.shape[-1]))

    # Apply softmax
    scores = tl.softmax(scores, axis=-1)

    # Compute output
    output = tl.dot(scores, v)

    # Store output
    tl.store(output_ptr + offs_m * stride_om, output, mask=m_mask)


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

        # Add routing parameters
        self.moe_jitter_noise = getattr(config, "moe_jitter_noise", 0.1)
        self.temperature = getattr(config, "router_temperature", 0.1)
        self.overflow_factor = getattr(config, "overflow_factor", 0.2)

        # Initialize metrics and losses
        self.entropy_loss = 0.0
        self.aux_loss = 0.0
        self.metrics = {}

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

        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.noise_scale = 0.01

        # Initialize router weights
        with torch.no_grad():
            self.router.weight.data.normal_(mean=0.0, std=0.02)

    def _compute_routing_weights(
        self, x: torch.Tensor, importance_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing weights with stable gradient flow."""
        seq_len = x.size(1)

        # Get router logits
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]

        # Add gating
        gate_value = torch.sigmoid(router_logits.mean(-1, keepdim=True))

        # Calculate expert capacity
        capacity = int(
            self.expert_capacity_factor
            * seq_len
            * self.num_experts_per_tok
            / self.num_experts
        )

        # Get top-k experts per token with temperature
        temperature = getattr(self, "temperature", 0.1)
        router_logits = router_logits / temperature
        router_probs = F.softmax(router_logits, dim=-1)

        # Add auxiliary loss for load balancing
        if self.training:
            # Calculate load balancing loss
            token_usage = router_probs.sum(dim=(0, 1))  # [num_experts]
            target_usage = router_probs.sum() / self.num_experts
            balance_loss = (token_usage - target_usage).pow(2).mean()

            # Calculate entropy loss for exploration
            entropy = -(router_probs * torch.log(router_probs + 1e-10)).sum(-1).mean()

            # Store auxiliary losses
            self.aux_loss = balance_loss * 0.01 + entropy * 0.01

        # Get top-k experts
        expert_weights, expert_indices = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )

        # Convert indices to correct dtype for scattering
        expert_indices = expert_indices.to(router_logits.dtype)

        # Normalize weights with stability ensuring float dtype
        expert_weights = F.softmax(expert_weights / temperature, dim=-1).to(
            router_logits.dtype
        )

        # Create dispatch tensors with gating
        dispatch_mask = torch.zeros_like(router_logits)
        dispatch_mask = dispatch_mask.to(expert_weights.dtype)
        dispatch_mask.scatter_(-1, expert_indices.long(), expert_weights)
        dispatch_mask = dispatch_mask * gate_value.to(dispatch_mask.dtype)

        # Implement capacity-based pruning
        position_in_expert = torch.cumsum(dispatch_mask, dim=1)
        capacity_mask = (position_in_expert <= capacity).to(dispatch_mask.dtype)
        dispatch_mask = dispatch_mask * capacity_mask

        # Final normalization with stability
        normalizer = dispatch_mask.sum(dim=-1, keepdim=True)
        dispatch_mask = dispatch_mask / (normalizer + 1e-6)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient stabilization."""

        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        # Cache router probabilities for statistics
        self._last_router_probs = router_probs.detach()  # Store detached version

        # Get dispatch mask and indices
        dispatch_mask, expert_indices = self._compute_routing_weights(x)

        # Rest of the forward method remains the same...
        final_output = torch.zeros_like(x)
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_mask = dispatch_mask[:, :, i].unsqueeze(-1)
            expert_input = x * expert_mask

            if expert_mask.sum() == 0:
                continue

            expert_output = expert(expert_input)
            expert_outputs.append(expert_output)

        if expert_outputs:
            final_output = sum(expert_outputs)

        gate = torch.sigmoid(router_logits.mean(-1, keepdim=True))
        final_output = gate * final_output + (1 - gate) * x

        if self.training:
            noise = torch.randn_like(final_output) * self.noise_scale
            final_output = final_output + noise

        if hasattr(self, "layer_norm"):
            final_output = self.layer_norm(final_output)

        return final_output

    def get_expert_statistics(self) -> Dict[str, torch.Tensor]:
        """Collect detailed expert usage statistics using cached router probabilities."""
        if not hasattr(self, "_last_router_probs"):
            raise RuntimeError(
                "Statistics not available - forward pass must be run first"
            )

        with torch.no_grad():
            router_probs = self._last_router_probs

            # Calculate expert usage
            expert_usage = router_probs.sum(dim=(0, 1))  # Sum over batch and sequence
            total_tokens = router_probs.size(0) * router_probs.size(1)
            expert_usage = expert_usage / total_tokens  # Normalize

            # Calculate entropy
            entropy = -(router_probs * torch.log(router_probs + 1e-10)).sum(-1).mean()

            # Calculate load balancing score
            target_usage = 1.0 / self.num_experts
            balance_score = 1.0 - (expert_usage - target_usage).abs().mean()

            # Calculate expert capacity utilization
            capacity = int(
                self.expert_capacity_factor * total_tokens / self.num_experts
            )
            usage_per_expert = (router_probs > 0).float().sum(dim=(0, 1))
            capacity_utilization = usage_per_expert / capacity

            return {
                "expert_usage": expert_usage,
                "entropy": entropy,
                "balance_score": balance_score,
                "capacity_utilization": capacity_utilization,
                "router_max_prob": router_probs.max(dim=-1)[0].mean(),
                "unused_expert_count": (expert_usage < 0.01).sum(),
            }

    def log_statistics(self, step: int):
        """Log expert statistics for monitoring."""
        stats = self.get_expert_statistics()

        print(f"\nStep {step} Expert Statistics:")
        print("Expert Usage Distribution:")
        for i, usage in enumerate(stats["expert_usage"]):
            print(f"Expert {i}: {usage:.3f}")

        print(f"\nRouting Metrics:")
        print(f"Entropy: {stats['entropy']:.3f}")
        print(f"Balance Score: {stats['balance_score']:.3f}")
        print(f"Unused Experts: {stats['unused_expert_count']}")
        print(
            f"Average Capacity Utilization: {stats['capacity_utilization'].mean():.3f}"
        )

        return stats


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


class DualHeadGPTConfig(GPTConfig):
    """Extended GPT config with dual head support."""

    def __init__(
        self,
        second_vocab_size: int = 50257,  # Size of second vocabulary
        aux_head_weight: float = 0.5,  # Weight for auxiliary loss
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.second_vocab_size = second_vocab_size
        self.aux_head_weight = aux_head_weight
        self.dropout = dropout

class DualHeadGPT(nn.Module):
    """GPT model with dual heads and improved stability."""

    def __init__(self, config: DualHeadGPTConfig):
        super().__init__()
        self.config = config

        # Core model components
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([self._create_block() for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Initialize second head
        self.second_head = nn.Linear(config.n_embd, config.second_vocab_size, bias=config.bias)
        self._init_weights(self.second_head)
        
        # Performance monitoring with bounded storage
        self.max_stats_size = 1000
        self.perf_stats = {
            'forward_time': deque(maxlen=self.max_stats_size),
            'backward_time': deque(maxlen=self.max_stats_size),
            'memory_used': deque(maxlen=self.max_stats_size)
        }
        
        # Model state
        self.kv_cache = {}
        self._is_static_graph = False
        self.gradient_checkpointing = False
        self.checkpoint_ratio = 0.5
        
        # Loss scaling parameters
        self.loss_scale_factor = 1.0
        self.min_loss_scale = 1e-8
        self.max_loss_scale = 1.0
        self.loss_scale_decay = 0.5
        self.max_loss_value = 100.0

        # Debug settings
        self.debug_mode = False
        
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with improved scaling."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)

    def forward(
        self,
        idx: torch.Tensor,
        target_second: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        position: int = 0,
        return_router_logits: bool = False,
    ) -> Union[ModelOutput, ModelOutputWithAux]:
        """Enhanced forward pass with improved validation and error handling."""
        if position < 0:
            raise ValueError("position must be non-negative")
            
        start_time = time.time()
        batch_size, seq_len = idx.size()
        
        # Validate and adjust mask
        if mask is not None:
            mask = self._validate_and_adjust_mask(mask, batch_size, seq_len)
            
        # Validate sequence length
        if seq_len > self.config.block_size:
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum block size {self.config.block_size}")
            
        # Process embeddings
        x = self.wte(idx)
        x = self.drop(x)
        
        self._debug_log("embeddings", x)
        
        # Forward through transformer blocks
        aux_losses = {}
        for i, block in enumerate(self.blocks):
            x = self._process_block(block, x, mask, position, return_router_logits, i)
            
        # Final processing
        x = self.ln_f(x)
        x = self.drop(x)
        
        # Compute logits and losses
        primary_logits, second_logits = self._compute_logits(x)
        primary_loss, second_loss = self._compute_losses(
            primary_logits, second_logits, idx, target_second
        )
        
        # Update performance stats
        if self.training:
            self._update_perf_stats(start_time)
            
        if return_router_logits:
            return primary_logits, second_logits, primary_loss, second_loss, aux_losses
        return primary_logits, second_logits, primary_loss, second_loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        target_head: str = "primary",
    ) -> torch.Tensor:
        """Generate tokens with improved parameter validation and efficiency."""
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        if target_head not in {"primary", "second"}:
            raise ValueError('target_head must be "primary" or "second"')
            
        self.eval()
        
        # Clear KV cache only if it exists and isn't empty
        if self.kv_cache and len(self.kv_cache) > 0:
            self.kv_cache.clear()
            
        vocab_size = (
            self.config.vocab_size 
            if target_head == "primary" 
            else self.config.second_vocab_size
        )
        
        if top_k is not None:
            top_k = min(top_k, vocab_size)
            
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )
            
            # Forward pass
            logits = self._get_next_token_logits(idx_cond, target_head)
            
            # Apply temperature and sampling
            logits = self._apply_sampling_params(logits, temperature, top_k)
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            
            # Append and check for EOS
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token_id is not None and (idx_next == eos_token_id).any():
                break
                
        return idx

    def _validate_and_adjust_mask(
        self, mask: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        """Validate and adjust attention mask dimensions."""
        expected_shape = (batch_size, 1, seq_len, seq_len)
        
        if mask.shape != expected_shape:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif len(mask.shape) == 3:
                mask = mask.unsqueeze(1)
                
            if mask.shape != expected_shape:
                raise ValueError(
                    f"Mask shape {mask.shape} cannot be adjusted to expected shape {expected_shape}"
                )
                
        return mask

    def _process_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        position: int,
        return_router_logits: bool,
        block_idx: int,
    ) -> torch.Tensor:
        """Process a single transformer block with optional checkpointing."""
        should_checkpoint = (
            self.gradient_checkpointing
            and self.training
            and block_idx % max(1, int(len(self.blocks) * self.checkpoint_ratio)) == 0
        )
        
        if should_checkpoint:
            x = self._forward_block_with_checkpoint(
                block, x, mask, position, return_router_logits
            )
        else:
            x = self._forward_block(block, x, mask, position, return_router_logits)
            
        self._debug_log(f"block_{block_idx}", x)
        return x

    def _compute_losses(
        self,
        primary_logits: torch.Tensor,
        second_logits: torch.Tensor,
        idx: torch.Tensor,
        target_second: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute losses for both heads with proper scaling."""
        if not self.training:
            return None, None
            
        primary_loss = (
            self._compute_loss(primary_logits, idx)
            if idx is not None
            else None
        )
        
        second_loss = (
            self._compute_loss(second_logits, target_second) * self.config.aux_head_weight
            if target_second is not None
            else None
        )
        
        return primary_loss, second_loss

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross entropy loss with improved numerical stability."""
        if logits.shape[:-1] != targets.shape:
            raise ValueError(
                f"Logits shape {logits.shape} incompatible with targets shape {targets.shape}"
            )
            
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            
            scaled_loss = self._scale_loss(loss)
            return scaled_loss

    def _scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss with bounds to prevent numerical instability."""
        if loss.item() > self.max_loss_value:
            self.loss_scale_factor = max(
                self.min_loss_scale,
                self.loss_scale_factor * self.loss_scale_decay
            )
            return loss * self.loss_scale_factor
        return loss

    def _get_next_token_logits(
        self, idx: torch.Tensor, target_head: str
    ) -> torch.Tensor:
        """Get logits for next token generation."""
        primary_logits, second_logits, _, _ = self(idx, position=idx.size(1))
        logits = primary_logits if target_head == "primary" else second_logits
        return logits[:, -1, :]

    def _apply_sampling_params(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
    ) -> torch.Tensor:
        """Apply temperature and top-k sampling to logits."""
        if temperature != 1.0:
            logits = logits / temperature
            
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            
        return logits

    def _debug_log(self, stage: str, x: torch.Tensor) -> None:
        """Log debug information when debug mode is enabled."""
        if not self.debug_mode:
            return
            
        print(f"\nDebug Info - {stage}:")
        print(f"Shape: {x.shape}")
        print(f"Device: {x.device}")
        print(
            f"Stats: min={x.min().item():.3f}, max={x.max().item():.3f}, "
            f"mean={x.mean().item():.3f}, std={x.std().item():.3f}"
        )
        has_nan = torch.isnan(x).any()
        has_inf = torch.isinf(x).any()
        if has_nan:
            print("WARNING: NaN values detected!")
        if has_inf:
            print("WARNING: Inf values detected!")

    def _update_perf_stats(self, start_time: float) -> None:
        """Update performance statistics with time and memory usage."""
        self.perf_stats["forward_time"].append(time.time() - start_time)
        if torch.cuda.is_available():
            self.perf_stats["memory_used"].append(torch.cuda.max_memory_allocated())

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics with proper handling of empty stats."""
        if not self.perf_stats["forward_time"]:
            return {}
            
        stats = {}
        for key, values in self.perf_stats.items():
            if values:
                stats[f"avg_{key}"] = sum(values) / len(values)
                stats[f"max_{key}"] = max(values)
                
        return stats

    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        for key in self.perf_stats:
            self.perf_stats[key].clear()

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get total number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wte.weight.numel()
        return n_params

    def save_state(self, path: str) -> None:
        """Save model state including performance statistics."""
        state = {
            'model_state': self.state_dict(),
            'config': self.config,
            'perf_stats': dict(self.perf_stats),
        }
        torch.save(state, path)