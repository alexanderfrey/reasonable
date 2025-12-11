"""
SOTA GPT Model Implementation.
Features: FlashAttention-2, Static KV Cache, Fused RoPE, Fused MLP, GQA.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

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

        # 3. Fused RoPE
        # flash_attn rotary expects inputs as [B, S, H, D]
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

        return self.o_proj(output)


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

    def forward(self, x, cos, sin, kv_cache=None, input_pos=None):
        # Attention Block
        h = self.norm_attn(x)
        attn_out = self.attn(h, cos, sin, kv_cache, input_pos)
        x = x + self.resid_dropout(attn_out)

        # MLP Block
        h = self.norm_ffn(x)
        ffn_out = self.ffn(h)
        x = x + self.resid_dropout(ffn_out)

        return x


# --- Main GPT Model ---

class GPTConfig:
    def __init__(self, vocab_size, d_model, n_head, n_layer, max_seq_len, n_kv_head=None, dropout=0.0, rope_theta=500000.0):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.max_seq_len = max_seq_len
        self.n_kv_head = n_kv_head
        self.dropout = dropout  # Dropout rate for attention and residual connections
        self.rope_theta = rope_theta  # RoPE base frequency (500k for long context, 10k original)
        # SwiGLU sizing
        self.d_ff = int(2 * (4 * d_model) / 3)
        self.d_ff = 256 * ((self.d_ff + 256 - 1) // 256) # Multiple of 256


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

        # Apply SOTA weight initialization
        self._init_weights()

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
        
        # Concatenate to match head_dim
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Register as buffer to save in state_dict
        self.register_buffer("cos_cached", emb.cos().to(dtype=torch.bfloat16), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype=torch.bfloat16), persistent=False)

    def setup_caches(self, batch_size, dtype=torch.bfloat16):
        """
        Allocates the Static KV Cache in VRAM.
        Shape: [n_layer, 2, B, Max_Len, n_kv, head_dim]
        """
        # We store keys and values separately to avoid slicing overhead if desired,
        # or grouped. Here we group them by layer.
        
        # Dimensions: [2(k,v), B, Max_Seq, n_kv_head, head_dim]
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
            k_cache = torch.zeros(cache_shape[1:], dtype=dtype, device=device)
            v_cache = torch.zeros(cache_shape[1:], dtype=dtype, device=device)
            self.kv_caches.append((k_cache, v_cache))

    def forward(
        self, 
        input_ids: torch.Tensor, 
        input_pos: Optional[torch.Tensor] = None
    ):
        # input_ids: [B, S]
        # input_pos: [S] (integers indicating position in sequence)
        
        if input_pos is None:
            # Default to 0..S if not provided (assume prompt w/o cache)
            input_pos = torch.arange(input_ids.size(1), device=input_ids.device)

        # 1. Fetch RoPE for these positions
        # shape: [S, Head_Dim] -> Reshaped for broadcast
        cos = self.cos_cached[input_pos].unsqueeze(0) # [1, S, D]
        sin = self.sin_cached[input_pos].unsqueeze(0)

        # 2. Embeddings
        x = self.token_embedding(input_ids)

        # 3. Transformer Layers
        for i, layer in enumerate(self.layers):
            # Retrieve layer-specific cache tuple (K, V)
            layer_cache = self.kv_caches[i] if self.kv_caches else None
            x = layer(x, cos, sin, kv_cache=layer_cache, input_pos=input_pos)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None
    ):
        """
        Optimized generation loop with Static Cache and Prefill/Decode separation.
        """
        self.eval()
        B, S = input_ids.shape
        device = input_ids.device

        # 1. Setup Static Cache
        # Crucial: Reset/Allocate cache for this batch size
        self.setup_caches(batch_size=B, dtype=self.token_embedding.weight.dtype)

        # 2. Prefill Phase (Process entire prompt at once)
        # We tell the model to write to positions 0...S-1
        input_pos = torch.arange(0, S, device=device)
        logits = self(input_ids, input_pos=input_pos)
        
        # Select last token to start generation
        next_token_logits = logits[:, -1, :]
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
            logits = self(next_token, input_pos=pos_tensor)
            
            next_token_logits = logits[:, -1, :]
            next_token = self._sample_token(next_token_logits, temperature, top_k)
            
            generated_ids.append(next_token)
            cur_pos += 1

        return torch.cat([input_ids] + generated_ids, dim=1)

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

