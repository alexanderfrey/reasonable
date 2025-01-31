# config.py
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
