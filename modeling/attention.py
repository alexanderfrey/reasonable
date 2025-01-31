# attention.py
import torch
import math
import time
import warnings
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Union, Dict
import triton
import triton.language as tl
from caching import OptimizedKVCache
from gpt import GPTConfig


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
        """
        Apply Rotary Position Embedding (RoPE) to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n_heads, seq_len, head_dim]
            position (int): Starting position for positional encoding

        Returns:
            torch.Tensor: Tensor with RoPE applied, same shape as input

        Raises:
            ValueError: If input tensor dimensions are invalid
            RuntimeError: If tensor device doesn't match rope_angles device
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")

        B, H, T, D = x.shape
        if D != self.head_dim:
            raise ValueError(f"Expected head dimension {self.head_dim}, got {D}")

        # Ensure rope_angles are on the same device as input
        if self.rope_angles.device != x.device:
            self.rope_angles = self.rope_angles.to(x.device)

        # Use cached frequencies if available and valid
        cache_key = (position, T, x.device)
        if not hasattr(self, "_rope_freq_cache"):
            self._rope_freq_cache = {}
            self._rope_cache_size = 0
            self._max_cache_size = 32  # Limit cache size

        if cache_key in self._rope_freq_cache:
            freqs = self._rope_freq_cache[cache_key]
        else:
            # Compute new frequencies
            positions = torch.arange(
                position, position + T, device=x.device, dtype=torch.float
            )
            freqs = torch.outer(positions, self.rope_angles)
            freqs = freqs.view(1, 1, T, D // 2)

            # Cache management: remove oldest if cache is full
            if self._rope_cache_size >= self._max_cache_size:
                oldest_key = next(iter(self._rope_freq_cache))
                del self._rope_freq_cache[oldest_key]
            else:
                self._rope_cache_size += 1

            self._rope_freq_cache[cache_key] = freqs

        # Split input into even and odd dimensions
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Stack for efficient rotation
        x_stacked = torch.stack([-x_odd, x_even], dim=-1)

        # Apply rotation using broadcasting
        cos = torch.cos(freqs).unsqueeze(-2)
        sin = torch.sin(freqs).unsqueeze(-2)

        # Compute rotation: [cos(θ) -sin(θ)] [x_even]
        #                   [sin(θ)  cos(θ)] [x_odd ]
        x_rotated = torch.cat(
            [x_even * cos - x_odd * sin, x_odd * cos + x_even * sin], dim=-1
        )

        # Ensure contiguous memory layout for better performance
        return x_rotated.contiguous()

    def _clear_rope_cache(self):
        """Clear the RoPE frequency cache to free memory."""
        if hasattr(self, "_rope_freq_cache"):
            self._rope_freq_cache.clear()
            self._rope_cache_size = 0

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


class OptimizedPagedMultiLatentAttention(nn.Module):
    """
    Paged multi-latent attention with memory optimizations.
    Processes attention in pages while using learned latent vectors to reduce memory usage.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Core dimensions
        self.n_head = config.n_head
        self.n_latents = config.n_head // 2  # Number of latent vectors per head
        self.head_dim = config.n_embd // config.n_head
        self.latent_dim = (
            self.head_dim
        )  # Keep latent dim same as head dim for consistency
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Configuration
        self.page_size = getattr(config, "page_size", 16)
        self.use_sdpa = getattr(config, "use_sdpa", False)
        self.use_memory_efficient = getattr(config, "use_memory_efficient", False)
        self.use_parallel = getattr(config, "use_parallel", False)
        self.dropout = config.dropout

        # Initialize latent vectors
        self.latent_queries = nn.Parameter(
            torch.randn(1, self.n_latents, self.latent_dim) / math.sqrt(self.latent_dim)
        )
        self.latent_keys = nn.Parameter(
            torch.randn(1, self.n_latents, self.latent_dim) / math.sqrt(self.latent_dim)
        )
        self.latent_values = nn.Parameter(
            torch.randn(1, self.n_latents, self.latent_dim) / math.sqrt(self.latent_dim)
        )

        # Main projections
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
        self.to_latent_v = QuantizedLinear(
            self.latent_dim, config.n_embd, bias=config.bias
        )

        # Single output projection
        self.out_proj = FusedLinear(
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
        """Applies rotary positional embeddings with proper bounds checking."""
        B, H, T, D = x.shape

        # Ensure we don't exceed max sequence length
        if start_pos >= self.max_seq_len:
            raise ValueError(
                f"Position {start_pos} exceeds maximum sequence length {self.max_seq_len}"
            )

        end_pos = min(start_pos + T, self.max_seq_len)

        # Compute frequencies if needed
        if self.freqs_cis is None or self.freqs_cis.size(0) < end_pos:
            self.freqs_cis = optimized_precompute_freqs_cis(
                dim=D, end=self.max_seq_len, theta=self.rope_theta
            ).to(x.device)

        # Apply RoPE
        freqs_slice = self.freqs_cis[start_pos:end_pos]
        x = x.permute(0, 2, 1, 3)  # [B, T, H, D]
        x = apply_rotary_emb(x, freqs_slice)
        return x.permute(0, 2, 1, 3)  # [B, H, T, D]

    def _process_page(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        page_idx: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process a single page of attention including latent vectors."""
        B, H, T, D = q.shape
        start_idx = page_idx * self.page_size
        end_idx = min(start_idx + self.page_size, T)

        # Prepare latent vectors for current batch
        latent_q = self.latent_queries.expand(B, -1, -1)  # [B, L, D]
        latent_k = self.latent_keys.expand(B, -1, -1)  # [B, L, D]
        latent_v = self.latent_values.expand(B, -1, -1)  # [B, L, D]

        # Project and reshape latent vectors
        latent_q = self.to_latent_q(latent_q).view(B, self.n_latents, H, D)
        latent_k = self.to_latent_k(latent_k).view(B, self.n_latents, H, D)
        latent_v = self.to_latent_v(latent_v).view(B, self.n_latents, H, D)

        # Get current page
        page_q = q[:, :, start_idx:end_idx]  # [B, H, page_size, D]

        # Concatenate regular and latent key/value pairs
        page_k = torch.cat([k[:, :, :end_idx], latent_k], dim=2)  # [B, H, T+L, D]
        page_v = torch.cat([v[:, :, :end_idx], latent_v], dim=2)  # [B, H, T+L, D]

        # Choose attention implementation
        if torch.cuda.is_available() and q.is_cuda:
            output = self._process_page_triton(
                page_q, page_k, page_v, start_idx, end_idx
            )
        elif self.use_flash_attention:
            # Adjust mask if provided
            if mask is not None:
                page_mask = mask[:, :, start_idx:end_idx, :end_idx]
                # Add mask entries for latent vectors (typically unmasked)
                latent_mask = torch.zeros(
                    (B, H, end_idx - start_idx, self.n_latents),
                    dtype=mask.dtype,
                    device=mask.device,
                )
                page_mask = torch.cat([page_mask, latent_mask], dim=-1)
            else:
                page_mask = None

            output = F.scaled_dot_product_attention(
                page_q,
                page_k,
                page_v,
                attn_mask=page_mask,
                scale=self.scale,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            output = self._process_page_pytorch(
                page_q, page_k, page_v, start_idx, end_idx
            )

        return output

    def _process_page_pytorch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor:
        """PyTorch implementation of paged attention."""
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Compute attention output
        output = torch.matmul(attn_weights, v)

        # Create output tensor
        full_output = torch.zeros(
            (q.shape[0], q.shape[1], end_idx - start_idx, q.shape[-1]),
            dtype=q.dtype,
            device=q.device,
        )
        full_output[:, :, : (end_idx - start_idx)] = output

        return full_output

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[OptimizedKVCache] = None,
        position: int = 0,
    ) -> torch.Tensor:
        """Forward pass implementing paged multi-latent attention."""
        B, T, C = x.shape
        H = self.n_head
        D = self.head_dim

        # Project inputs
        q = self.to_queries(x).view(B, T, H, D).transpose(1, 2)
        k = self.to_keys(x).view(B, T, H, D).transpose(1, 2)
        v = self.to_values(x).view(B, T, H, D).transpose(1, 2)

        # Apply RoPE
        q = self._apply_rope(q, position)
        k = self._apply_rope(k, position)

        # Handle KV cache
        if kv_cache is not None:
            if position > 0:
                k_cache, v_cache = kv_cache.get_kv(position)
                k = torch.cat([k_cache, k[:, :, -1:]], dim=2)
                v = torch.cat([v_cache, v[:, :, -1:]], dim=2)
            kv_cache.update(k, v, position)

        # Process attention in pages
        num_pages = math.ceil(T / self.page_size)
        outputs = []

        for page_idx in range(num_pages):
            page_output = self._process_page(q, k, v, page_idx, mask)
            outputs.append(page_output)

        # Combine page outputs
        output = torch.cat(outputs, dim=2)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(output)

    def compute_optimal_chunk_size(self, seq_len: int) -> int:
        """Compute optimal chunk size based on sequence length."""
        if seq_len <= 512:
            return 128
        elif seq_len <= 2048:
            return 256
        else:
            return 512

    def _process_page_triton(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor:
        """Process attention with proper tensor reshaping and Triton acceleration."""
        try:
            B, H, T, D = q.shape
            page_size = end_idx - start_idx

            # Make tensors contiguous and reshape properly
            page_q = q[:, :, start_idx:end_idx].contiguous()
            q_reshaped = page_q.view(B * H, page_size, D)
            k_reshaped = k.contiguous().view(B * H, k.size(2), D)
            v_reshaped = v.contiguous().view(B * H, v.size(2), D)

            # Create correctly sized output tensor
            output = torch.empty((B * H, page_size, D), dtype=q.dtype, device=q.device)

            # Get kernel config
            kernel_config = get_paged_attention_kernel(page_size, D)
            kernel = kernel_config["kernel"]
            kwargs = kernel_config["kwargs"]

            # Launch kernel with proper parameters
            grid = (B * H,)
            kernel[grid](
                q_reshaped,
                k_reshaped,
                v_reshaped,
                output,
                q_reshaped.stride(0),
                q_reshaped.stride(1),
                q_reshaped.stride(2),
                k_reshaped.stride(0),
                k_reshaped.stride(1),
                k_reshaped.stride(2),
                v_reshaped.stride(0),
                v_reshaped.stride(1),
                v_reshaped.stride(2),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                self.scale,
                self.dropout if self.training else 0.0,
                start_idx=start_idx,
                end_idx=end_idx,
                **kwargs,
            )

            # Reshape output back to original dimensions
            return output.view(B, H, page_size, D)

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
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """PyTorch fallback implementation with proper sequence length handling."""
        B, H, _, D = q.shape
        page_size = end_idx - start_idx

        # Handle the page slice efficiently
        page_q = q[:, :, start_idx:end_idx].contiguous()  # [B, H, page_size, D]

        # Compute attention scores with proper scaling
        scores = torch.matmul(page_q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask[:, :, start_idx:end_idx]

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [B, H, page_size, D]

        return output


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
    scale: float,
    dropout_p: float,
    start_idx: tl.constexpr,
    end_idx: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Get program ID and batch/head indices
    pid = tl.program_id(0)
    batch_idx = pid // stride_qb
    head_idx = (pid % stride_qb) // stride_qh

    # Calculate dimensions
    page_size = end_idx - start_idx
    seq_len = end_idx  # Total sequence length including current position

    # Create block pointers with proper offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Create proper masks
    m_mask = offs_m < page_size
    n_mask = offs_n < seq_len
    k_mask = offs_k < BLOCK_K

    # Load query block with proper masking and offsets
    q_block_ptr = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    q = tl.load(
        q_block_ptr + offs_m[:, None] * stride_qm + offs_k[None, :],
        mask=m_mask[:, None] & k_mask[None, :],
        other=0.0,
    )

    # Initialize accumulator for output
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # Block-wise attention computation
    for block_start in range(0, seq_len, BLOCK_N):
        k_block_ptr = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
        v_block_ptr = v_ptr + batch_idx * stride_vb + head_idx * stride_vh

        # Load key and value blocks
        k = tl.load(
            k_block_ptr
            + block_start * stride_kn
            + offs_n[:, None] * stride_kn
            + offs_k[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            v_block_ptr
            + block_start * stride_vn
            + offs_n[:, None] * stride_vn
            + offs_k[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # Compute attention scores
        scores = tl.dot(q, k.transpose()) * scale

        # Apply causal masking
        if block_start + offs_n[None, :] <= start_idx + offs_m[:, None]:
            scores = tl.where(True, -float("inf"), scores)

        # Apply softmax and dropout
        scores = tl.softmax(scores, axis=-1)
        if dropout_p > 0.0:
            scores = tl.where(
                tl.rand(*scores.shape) > dropout_p, scores / (1 - dropout_p), 0.0
            )

        # Accumulate weighted values
        acc += tl.dot(scores, v)

    # Store output with proper masking
    output_block_ptr = output_ptr + batch_idx * stride_ob + head_idx * stride_oh
    tl.store(
        output_block_ptr + offs_m[:, None] * stride_om + offs_k[None, :],
        acc,
        mask=m_mask[:, None] & k_mask[None, :],
    )
