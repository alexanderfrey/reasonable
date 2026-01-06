"""
SOTA GPT Model Implementation.
Features: FlashAttention-2, Static KV Cache, Fused RoPE, Fused MLP, GQA.
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# --- SOTA Imports ---
try:
    from flash_attn import flash_attn_func
    from flash_attn.layers.rotary import apply_rotary_emb
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("WARNING: flash_attn not installed. Code will fail or run slow.")

# --- Optimized Components ---

class OptimizedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # PyTorch 2.4+ fused kernel
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class OptimizedMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Optimization: Fuse Gate (w1) and Up (w3) into one matrix
        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        # 1. Fused Projection
        gate_up = self.gate_up_proj(x)
        # 2. Split
        gate, up = gate_up.chunk(2, dim=-1)
        # 3. SwiGLU & Down Projection
        return self.down_proj(F.silu(gate) * up)


class OptimizedAttention(nn.Module):
    """
    Drop-in replacement for MultiHeadAttention, but requires the
    parent GPT model to handle 'input_pos' and 'kv_cache'.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        self.head_dim = self.d_model // self.n_head
        self.dropout = getattr(config, 'dropout', 0.0)

        # Check GQA constraints
        assert self.n_head % self.n_kv_head == 0
        self.n_rep = self.n_head // self.n_kv_head

        # Optimization: Fuse Q, K, V into one projection
        # Output layout: [Q_heads | K_heads | V_heads]
        op_size = (self.n_head + 2 * self.n_kv_head) * self.head_dim
        self.qkv_proj = nn.Linear(self.d_model, op_size, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor, 
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        B, S, _ = x.shape
        
        # 1. Fused QKV Projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, S, self.n_head + 2 * self.n_kv_head, self.head_dim)
        
        # 2. Split Q, K, V
        q, k, v = qkv.split([self.n_head, self.n_kv_head, self.n_kv_head], dim=2)

        # Ensure flash-attn sees a supported dtype
        target_dtype = q.dtype
        if target_dtype not in (torch.float16, torch.bfloat16):
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16
        if q.dtype != target_dtype:
            q = q.to(dtype=target_dtype)
            k = k.to(dtype=target_dtype)
            v = v.to(dtype=target_dtype)

        # 3. Fused RoPE
        # flash_attn rotary expects inputs as [B, S, H, D]
        # Guard against any mismatch between cached rotary dim and head_dim
        rotary_dim = cos.shape[-1] * 2  # flash-attn treats cache dim as half
        head_dim = q.shape[-1]
        if rotary_dim > head_dim:
            trim = head_dim // 2
            cos = cos[..., :trim]
            sin = sin[..., :trim]
        if cos.dtype != target_dtype:
            cos = cos.to(dtype=target_dtype)
            sin = sin.to(dtype=target_dtype)
        q = apply_rotary_emb(q, cos, sin, interleaved=False)
        k = apply_rotary_emb(k, cos, sin, interleaved=False)

        # 4. KV Cache Management (Zero-Copy)
        if kv_cache is not None:
            # kv_cache is a tuple of (k_cache, v_cache) refs to the large static buffer
            k_cache, v_cache = kv_cache
            
            # Write to cache at specific positions
            # cache shape: [B, Max_Seq, H, D]
            k_cache.index_copy_(1, input_pos, k)
            v_cache.index_copy_(1, input_pos, v)

            # Retrieve the valid portion of the cache for attention
            # For decoding (S=1), we need all history up to current pos
            # For prefill (S>1), we just use the current slice (causal masking handled by flash_attn)
            if S == 1:
                # Decoding: Attend to history + current
                # Note: We can pass the whole buffer to flash_attn if we use 'seqlens', 
                # but slicing is easier to implement for general cases.
                # input_pos is a scalar tensor in decode
                curr_pos = input_pos[-1].item() + 1
                k = k_cache[:, :curr_pos]
                v = v_cache[:, :curr_pos]
            else:
                # Prefill: self-attention within the prompt
                # The cache is updated, but we attend to q, k, v directly
                pass 

        # 5. Flash Attention 2
        # Automatically handles GQA (if n_kv < n_head) and broadcasting

        is_causal = (S > 1)  # Causal masking required during prefill, not single-token decode
        # Only apply dropout during training
        attn_dropout = self.dropout if self.training else 0.0

        output = flash_attn_func(
            q, k, v,
            dropout_p=attn_dropout,
            causal=is_causal,
            window_size=(-1, -1)  # Full context
        )
        # flash_attn returns [B, S, H, D]; merge heads for the output projection
        output = output.reshape(B, S, self.n_head * self.head_dim)
        # Cast to weight dtype for compatibility with gradient checkpointing (which may
        # recompute outside autocast context)
        output = output.to(self.o_proj.weight.dtype)
        return self.o_proj(output)


class MemoryAugmentedAttention(nn.Module):
    """
    Attention layer that can attend to both current context and memory K/V pairs.

    This enables GPT layers to naturally attend to retrieved memories during
    forward pass, rather than post-hoc blending. Memory tokens are prepended
    to the K/V sequence, allowing each query position to attend to relevant
    memory content.

    Key design choices:
    - Memory K/V projections are separate (trainable) from frozen base projections
    - Gating controls how much memory influences output (initialized small)
    - Memory positions are always attendable (no causal mask applied to them)
    """

    def __init__(self, config, base_attn: OptimizedAttention):
        super().__init__()
        self.base_attn = base_attn  # Original attention (can be frozen)
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        self.head_dim = self.d_model // self.n_head

        # Memory K/V projections (trainable even if base is frozen)
        self.mem_k_proj = nn.Linear(self.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.mem_v_proj = nn.Linear(self.d_model, self.n_kv_head * self.head_dim, bias=False)

        # Gate to control memory influence (initialized to ~0.12)
        self.mem_gate = nn.Parameter(torch.tensor(-2.0))

        # Initialize memory projections
        nn.init.normal_(self.mem_k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.mem_v_proj.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        input_pos: Optional[torch.Tensor] = None,
        memory_kv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional memory K/V injection.

        Args:
            x: [B, S, d_model] input hidden states
            cos, sin: RoPE embeddings
            kv_cache: Optional KV cache for generation
            input_pos: Position indices
            memory_kv: [B, M, d_model] memory tokens to attend to, or None

        Returns:
            output: [B, S, d_model] attention output
        """
        if memory_kv is None:
            # No memory - use base attention directly
            return self.base_attn(x, cos, sin, kv_cache, input_pos)

        B, S, _ = x.shape
        M = memory_kv.size(1)  # Number of memory tokens

        # 1. Get Q, K, V from base attention's fused projection
        qkv = self.base_attn.qkv_proj(x)
        qkv = qkv.view(B, S, self.n_head + 2 * self.n_kv_head, self.head_dim)
        q, k, v = qkv.split([self.n_head, self.n_kv_head, self.n_kv_head], dim=2)

        # Ensure flash-attn compatible dtype
        target_dtype = q.dtype
        if target_dtype not in (torch.float16, torch.bfloat16):
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16
        if q.dtype != target_dtype:
            q = q.to(dtype=target_dtype)
            k = k.to(dtype=target_dtype)
            v = v.to(dtype=target_dtype)

        # 2. Apply RoPE to Q and K (not to memory - memory has no position)
        rotary_dim = cos.shape[-1] * 2
        head_dim = q.shape[-1]
        if rotary_dim > head_dim:
            trim = head_dim // 2
            cos = cos[..., :trim]
            sin = sin[..., :trim]
        if cos.dtype != target_dtype:
            cos = cos.to(dtype=target_dtype)
            sin = sin.to(dtype=target_dtype)
        q = apply_rotary_emb(q, cos, sin, interleaved=False)
        k = apply_rotary_emb(k, cos, sin, interleaved=False)

        # 3. KV Cache Management (same as base attention)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache

            # Write current K, V to cache at input_pos
            k_cache.index_copy_(1, input_pos, k)
            v_cache.index_copy_(1, input_pos, v)

            # For decoding (S=1), retrieve history from cache
            if S == 1:
                curr_pos = input_pos[-1].item() + 1
                k = k_cache[:, :curr_pos]
                v = v_cache[:, :curr_pos]

        # 4. Project memory to K, V (no RoPE - memory is "outside" position)
        mem_k = self.mem_k_proj(memory_kv.to(target_dtype))  # [B, M, n_kv_head * head_dim]
        mem_v = self.mem_v_proj(memory_kv.to(target_dtype))  # [B, M, n_kv_head * head_dim]
        # Ensure output dtype matches target (projection weights may be fp32)
        mem_k = mem_k.to(target_dtype).view(B, M, self.n_kv_head, self.head_dim)
        mem_v = mem_v.to(target_dtype).view(B, M, self.n_kv_head, self.head_dim)

        # 5. Compute attention in two parts to handle masking correctly:
        #    - Context self-attention: causal mask (Q[i] sees K[0..i])
        #    - Memory cross-attention: no causal mask (all Q see all memory)
        #    This avoids the bug where prepending memory would shift causal positions.
        is_causal = (S > 1)  # Only apply causal during prefill
        attn_dropout = self.base_attn.dropout if self.base_attn.training else 0.0

        # 5a. Self-attention to context (with causal mask)
        attn_context = flash_attn_func(
            q, k, v,
            dropout_p=attn_dropout,
            causal=is_causal,
            window_size=(-1, -1)
        )

        # 5b. Cross-attention to memory (no causal mask - all positions see all memory)
        attn_memory = flash_attn_func(
            q, mem_k, mem_v,
            dropout_p=attn_dropout,
            causal=False,  # All query positions can attend to all memory
            window_size=(-1, -1)
        )

        # 6. Blend context and memory attention with learned gate
        gate = torch.sigmoid(self.mem_gate)
        output = (1 - gate) * attn_context + gate * attn_memory

        # 7. Reshape and output projection
        output = output.reshape(B, S, self.n_head * self.head_dim)
        output = output.to(self.base_attn.o_proj.weight.dtype)
        return self.base_attn.o_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_attn = OptimizedRMSNorm(config.d_model)
        self.attn = OptimizedAttention(config)
        self.norm_ffn = OptimizedRMSNorm(config.d_model)
        self.ffn = OptimizedMLP(config.d_model, config.d_ff)
        # Residual dropout (applied after attention and FFN)
        dropout = getattr(config, 'dropout', 0.0)
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Flag for memory-augmented attention
        self._has_memory_attn = False

    def upgrade_to_memory_attention(self, config):
        """Replace standard attention with memory-augmented attention."""
        if not self._has_memory_attn:
            self.attn = MemoryAugmentedAttention(config, self.attn)
            self._has_memory_attn = True

    def forward(self, x, cos, sin, kv_cache=None, input_pos=None, memory_kv=None):
        """
        Forward pass through transformer block.

        Args:
            x: [B, S, d_model] input hidden states
            cos, sin: RoPE embeddings
            kv_cache: Optional KV cache tuple for generation
            input_pos: Position indices
            memory_kv: [B, M, d_model] memory tokens for memory-augmented attention

        Returns:
            x: [B, S, d_model] output hidden states
        """
        # Attention Block
        h = self.norm_attn(x)
        if self._has_memory_attn and memory_kv is not None:
            attn_out = self.attn(h, cos, sin, kv_cache, input_pos, memory_kv=memory_kv)
        else:
            attn_out = self.attn(h, cos, sin, kv_cache, input_pos)
        x = x + self.resid_dropout(attn_out)

        # MLP Block
        h = self.norm_ffn(x)
        ffn_out = self.ffn(h)
        x = x + self.resid_dropout(ffn_out)

        return x


# --- Main GPT Model ---

class GPTConfig:
    def __init__(self, vocab_size, d_model, n_head, n_layer, max_seq_len, n_kv_head=None, dropout=0.0, rope_theta=500000.0, use_gradient_checkpointing=False, d_ff=None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.max_seq_len = max_seq_len
        self.n_kv_head = n_kv_head
        self.dropout = dropout  # Dropout rate for attention and residual connections
        self.rope_theta = rope_theta  # RoPE base frequency (500k for long context, 10k original)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        # SwiGLU sizing: use provided d_ff or compute optimal value
        if d_ff is not None:
            self.d_ff = d_ff
        else:
            self.d_ff = int(2 * (4 * d_model) / 3)
            self.d_ff = 256 * ((self.d_ff + 256 - 1) // 256)  # Multiple of 256


class GPT(nn.Module):
    def __init__(self, config: GPTConfig = None, **kwargs):
        super().__init__()
        # Support both GPTConfig object and kwargs for flexibility
        if config is None:
            config = GPTConfig(
                vocab_size=kwargs['vocab_size'],
                d_model=kwargs['d_model'],
                n_head=kwargs['n_head'],
                n_layer=kwargs['n_layer'],
                max_seq_len=kwargs['max_seq_len'],
                n_kv_head=kwargs.get('n_kv_head'),
                dropout=kwargs.get('dropout', 0.0),
                rope_theta=kwargs.get('rope_theta', 500000.0),
                use_gradient_checkpointing=kwargs.get('use_gradient_checkpointing', False),
                d_ff=kwargs.get('d_ff'),
            )
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = OptimizedRMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        # Initialize RoPE Cache (Cos/Sin)
        self.head_dim = config.d_model // config.n_head
        self._init_rope()

        self.kv_caches = None # Placeholder

        # Memory injection layer indices (empty = no memory injection)
        self._memory_layers: List[int] = []

        # Apply SOTA weight initialization
        self._init_weights()

    def enable_memory_layers(self, layer_indices: Optional[List[int]] = None):
        """
        Enable memory K/V injection at specific layers.

        Args:
            layer_indices: List of layer indices to inject memory at.
                          If None, uses default [n_layer//4, n_layer//2, 3*n_layer//4]
                          (1/4, 1/2, 3/4 depth)

        Returns:
            List of actual layer indices enabled
        """
        if layer_indices is None:
            # Default: inject at 1/4, 1/2, 3/4 depth
            n = self.config.n_layer
            layer_indices = [n // 4, n // 2, 3 * n // 4]

        # Validate indices
        layer_indices = [i for i in layer_indices if 0 <= i < self.config.n_layer]

        # Upgrade selected layers to memory-augmented attention
        for idx in layer_indices:
            self.layers[idx].upgrade_to_memory_attention(self.config)

        self._memory_layers = layer_indices
        logger.info(f"Enabled memory injection at layers: {layer_indices}")
        return layer_indices

    def get_memory_layer_indices(self) -> List[int]:
        """Get list of layer indices with memory injection enabled."""
        return self._memory_layers.copy()

    def _init_weights(self):
        """
        GPT-2/GPT-3 style weight initialization:
        - Normal(0, 0.02) for most weights
        - Scaled Normal for residual projections: 0.02 / sqrt(2 * n_layer)
        - Embeddings: Normal(0, 0.02)
        """
        init_std = 0.02
        residual_std = init_std / math.sqrt(2 * self.config.n_layer)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Scale down output projections (o_proj, down_proj) for residual stability
                if "o_proj" in name or "down_proj" in name:
                    nn.init.normal_(module.weight, mean=0.0, std=residual_std)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)

    def _init_rope(self):
        # Precompute cos/sin for the maximum sequence length
        # Using float32 for high precision calculation
        # Larger theta (500k) enables better long-context modeling
        theta = getattr(self.config, 'rope_theta', 500000.0)
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(self.config.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        # flash-attn rotary expects cos/sin shape [S, head_dim/2]; it multiplies by 2 internally
        self.register_buffer("cos_cached", freqs.cos().to(dtype=torch.bfloat16), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().to(dtype=torch.bfloat16), persistent=False)

    def _cache_dtype(self, prefer: Optional[torch.dtype] = None) -> torch.dtype:
        """
        flash-attn kernels only support fp16/bf16. Pick a supported dtype,
        preferring the requested one when valid.
        """
        if prefer in (torch.float16, torch.bfloat16):
            return prefer
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def setup_caches(self, batch_size, dtype: Optional[torch.dtype] = None):
        """
        Allocates the Static KV Cache in VRAM.
        Shape: [n_layer, 2, B, Max_Len, n_kv, head_dim]
        """
        # We store keys and values separately to avoid slicing overhead if desired,
        # or grouped. Here we group them by layer.
        
        # Dimensions: [2(k,v), B, Max_Seq, n_kv_head, head_dim]
        cache_dtype = self._cache_dtype(dtype or self.token_embedding.weight.dtype)
        cache_shape = (
            2, 
            batch_size, 
            self.config.max_seq_len, 
            self.config.n_kv_head, 
            self.head_dim
        )
        
        device = self.token_embedding.weight.device
        self.kv_caches = []
        for _ in range(self.config.n_layer):
            # Pre-allocate zeroed tensor
            k_cache = torch.zeros(cache_shape[1:], dtype=cache_dtype, device=device)
            v_cache = torch.zeros(cache_shape[1:], dtype=cache_dtype, device=device)
            self.kv_caches.append((k_cache, v_cache))

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        memory_kv: Optional[torch.Tensor] = None
    ):
        # input_ids: [B, S]
        # input_pos: [S] (integers indicating position in sequence)
        # return_hidden_states: if True, returns (logits, hidden_states) instead of (logits, None)
        # memory_kv: [B, M, d_model] memory tokens to inject at memory layers (optional)

        if input_pos is None:
            # Default to 0..S if not provided (assume prompt w/o cache)
            input_pos = torch.arange(input_ids.size(1), device=input_ids.device)

        # 1. Fetch RoPE for these positions
        # flash_attn rotary expects [S, D]; broadcast happens inside the kernel
        cos = self.cos_cached[input_pos]  # [S, D]
        sin = self.sin_cached[input_pos]

        # 2. Embeddings
        x = self.token_embedding(input_ids)

        use_cache = bool(self.kv_caches) and len(self.kv_caches) == len(self.layers)
        if use_cache:
            cache_batch = self.kv_caches[0][0].shape[0]
            cache_device = self.kv_caches[0][0].device
            if cache_batch != input_ids.size(0) or cache_device != input_ids.device:
                logger.debug(
                    "Resetting KV cache due to batch/device mismatch (cache_batch=%s, batch=%s, cache_device=%s, input_device=%s).",
                    cache_batch,
                    input_ids.size(0),
                    cache_device,
                    input_ids.device,
                )
                self.kv_caches = None
                use_cache = False

        # 3. Transformer Layers
        use_checkpointing = self.config.use_gradient_checkpointing and self.training and not use_cache
        for i, layer in enumerate(self.layers):
            # Retrieve layer-specific cache tuple (K, V)
            layer_cache = self.kv_caches[i] if use_cache else None

            # Pass memory_kv only to memory-enabled layers
            layer_memory = memory_kv if i in self._memory_layers else None

            if use_checkpointing:
                # Note: gradient_checkpoint doesn't support kwargs well, so we pass memory_kv positionally
                x = gradient_checkpoint(
                    layer, x, cos, sin, layer_cache, input_pos, layer_memory,
                    use_reentrant=False
                )
            else:
                x = layer(x, cos, sin, kv_cache=layer_cache, input_pos=input_pos, memory_kv=layer_memory)

        hidden_states = self.final_norm(x)
        logits = self.lm_head(hidden_states)

        # Return hidden states if requested (for experiential stream, probing, etc.)
        if return_hidden_states:
            return logits, hidden_states

        # Keep API compatible with callers expecting (logits, kv_cache_out)
        return logits, None

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
    ):
        """
        Optimized generation loop with Static Cache and Prefill/Decode separation.
        """
        self.eval()
        B, S = input_ids.shape
        device = input_ids.device

        # 1. Setup Static Cache
        # Crucial: Reset/Allocate cache for this batch size
        cache_dtype = self._cache_dtype(self.token_embedding.weight.dtype)
        self.setup_caches(batch_size=B, dtype=cache_dtype)

        try:
            # 2. Prefill Phase (Process entire prompt at once)
            # We tell the model to write to positions 0...S-1
            input_pos = torch.arange(0, S, device=device)
            logits, _ = self(input_ids, input_pos=input_pos)
            
            # Select last token to start generation
            next_token_logits = logits[:, -1, :]
            if repetition_penalty and repetition_penalty != 1.0:
                # Apply repetition penalty using the existing prompt tokens
                next_token_logits = self._apply_repetition_penalty(next_token_logits, input_ids, repetition_penalty)
            next_token = self._sample_token(next_token_logits, temperature, top_k)
            generated_ids = [next_token]

            # 3. Decode Phase (Token by Token)
            # Compile Hint: The loop body is static shape!
            cur_pos = S
            
            for _ in range(max_new_tokens):
                # Input is just the last generated token [B, 1]
                # input_pos is just the current scalar position [B] or [1]
                pos_tensor = torch.tensor([cur_pos], device=device) # Scalar tensor
                
                # Forward pass writes to cache[cur_pos] and attends to 0...cur_pos
                logits, _ = self(next_token, input_pos=pos_tensor)
                
                next_token_logits = logits[:, -1, :]
                if repetition_penalty and repetition_penalty != 1.0:
                    # Build full history = prompt + generated so far
                    history = torch.cat([input_ids] + generated_ids, dim=1)
                    next_token_logits = self._apply_repetition_penalty(next_token_logits, history, repetition_penalty)
                next_token = self._sample_token(next_token_logits, temperature, top_k)
                
                generated_ids.append(next_token)
                cur_pos += 1

            return torch.cat([input_ids] + generated_ids, dim=1)
        finally:
            # Prevent stale caches from leaking into training passes with different batch sizes
            self.kv_caches = None

    def _sample_token(self, logits, temperature, top_k):
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            if top_k is not None:
                v, _ = torch.topk(probs, top_k)
                probs[probs < v[:, [-1]]] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token

    def _apply_repetition_penalty(self, logits: torch.Tensor, history: torch.Tensor, penalty: float) -> torch.Tensor:
        """
        Applies repetition penalty to logits based on tokens in `history`.
        Supports batch processing; expects logits/history batch sizes to match.
        """
        if penalty == 1.0:
            return logits
        adjusted = logits.clone()
        B = logits.size(0)
        for b in range(B):
            used = torch.unique(history[b])
            vals = adjusted[b, used]
            penalized = torch.where(vals < 0, vals * penalty, vals / penalty)
            adjusted[b, used] = penalized
        return adjusted
