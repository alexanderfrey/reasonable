"""
Continuous Thought Machine (CTM) - Faithful Implementation.

Based on the original CTM paper:
- Synapse: z^t + o^t → a^t (pre-activations)
- NLM: processes pre-activation history → z^{t+1} (post-activations)
- Sync: dot products of post-activation histories between neuron pairs
- Sync projected to attention queries/outputs
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from flash_attn import flash_attn_func
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    raise ImportError("flash_attn is required. Install with: pip install flash-attn --no-build-isolation")

from model import OptimizedRMSNorm


@dataclass
class CTMConfig:
    """Configuration for CTM."""
    vocab_size: int
    d_model: int = 512
    n_head: int = 8
    n_kv_head: Optional[int] = None
    max_seq_len: int = 2048

    # CTM-specific
    num_ticks: int = 8
    nlm_hidden: int = 64
    nlm_depth: int = 2
    sync_pairs: int = 512

    # Training
    dropout: float = 0.0
    rope_theta: float = 500000.0

    # Memory optimization
    use_gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head


class SynapseModel(nn.Module):
    """
    Synapse: combines post-activations z^t and attention output o^t
    to produce pre-activations a^t.

    U-Net style MLP with:
    - SwiGLU activations for increased expressiveness
    - Tick-conditioned gating to control observation integration
      (early ticks: more receptive, later ticks: more conservative)
    """

    def __init__(self, d_model: int, num_ticks: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model

        # Input: z + o (both d_model)
        input_dim = d_model * 2

        # Encoder with SwiGLU (need 2x projections for gate and value)
        self.enc1_gate = nn.Linear(input_dim, d_model, bias=False)
        self.enc1_up = nn.Linear(input_dim, d_model, bias=False)
        self.enc2_gate = nn.Linear(d_model, d_model // 2, bias=False)
        self.enc2_up = nn.Linear(d_model, d_model // 2, bias=False)

        # Decoder with SwiGLU
        self.dec2_gate = nn.Linear(d_model // 2, d_model, bias=False)
        self.dec2_up = nn.Linear(d_model // 2, d_model, bias=False)
        self.dec1 = nn.Linear(d_model * 2, d_model, bias=False)  # skip connection

        self.norm1 = OptimizedRMSNorm(d_model)
        self.norm2 = OptimizedRMSNorm(d_model // 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Tick-conditioned gating: controls how much observation to integrate
        # Early ticks should be more receptive (higher gate), later ticks more conservative
        self.tick_gate_proj = nn.Linear(d_model, d_model, bias=False)
        # Initialize so early ticks have higher gate values
        self._init_tick_gate()

    def _init_tick_gate(self):
        # Small initialization for tick gate projection
        nn.init.normal_(self.tick_gate_proj.weight, std=0.01)

    def forward(
        self,
        z: torch.Tensor,
        o: torch.Tensor,
        tick_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z: (B, S, D) post-activations (internal state)
            o: (B, S, D) attention output (observation)
            tick_embed: (D,) tick embedding for current tick
        Returns:
            a: (B, S, D) pre-activations
        """
        # Tick-conditioned gate: modulates how much of o to integrate
        # sigmoid output in [0, 1], learned per-tick behavior
        tick_gate = torch.sigmoid(self.tick_gate_proj(tick_embed))  # (D,)

        # Gate the observation: early ticks can learn to be more receptive
        o_gated = o * tick_gate  # broadcast (B, S, D) * (D,)

        x = torch.cat([z, o_gated], dim=-1)

        # Encoder with SwiGLU
        h1 = F.silu(self.enc1_gate(x)) * self.enc1_up(x)
        h1 = self.norm1(h1)
        h2 = F.silu(self.enc2_gate(h1)) * self.enc2_up(h1)
        h2 = self.norm2(h2)

        # Decoder with SwiGLU and skip
        d2 = F.silu(self.dec2_gate(h2)) * self.dec2_up(h2)
        d2 = self.dropout(d2)
        d1 = torch.cat([d2, h1], dim=-1)
        out = self.dec1(d1)

        return out


class NeuronLevelModel(nn.Module):
    """
    NLM: Each neuron independently processes its pre-activation history
    to produce the next post-activation.

    D independent MLPs, efficiently batched.
    Input: (B, S, T, D) - history of pre-activations per neuron
    Output: (B, S, D) - post-activations

    Features:
    - SwiGLU activation (gated, more expressive)
    - Layer normalization (training stability)
    - Internal residuals in hidden layers (gradient flow)
    - No output residual (forces history-driven dynamics)
    - Per-neuron time constants (heterogeneous temporal scales)
    """

    def __init__(self, d_model: int, nlm_hidden: int, nlm_depth: int, max_ticks: int):
        super().__init__()
        self.d_model = d_model
        self.nlm_hidden = nlm_hidden
        self.max_ticks = max_ticks
        self.nlm_depth = nlm_depth

        # Per-neuron time constants for history decay
        # tau controls how far back each neuron "looks" in history
        # Higher tau = longer memory, lower tau = focuses on recent
        # Initialize with diversity: log-uniform in [0.5, 4.0] ticks
        self.tau_raw = nn.Parameter(torch.empty(d_model))
        self._init_tau()

        # Per-neuron input projection: (D, max_ticks, nlm_hidden)
        # SwiGLU needs 2x hidden for gate
        self.w_in = nn.Parameter(torch.empty(d_model, max_ticks, nlm_hidden * 2))
        self.b_in = nn.Parameter(torch.zeros(d_model, nlm_hidden * 2))

        # Layer norm after input (per-neuron, so we normalize over hidden dim)
        # Using learnable scale/bias per neuron
        self.ln_in_scale = nn.Parameter(torch.ones(d_model, nlm_hidden))
        self.ln_in_bias = nn.Parameter(torch.zeros(d_model, nlm_hidden))

        # Hidden layers with SwiGLU
        self.w_hidden = nn.ParameterList()
        self.b_hidden = nn.ParameterList()
        self.ln_hidden_scale = nn.ParameterList()
        self.ln_hidden_bias = nn.ParameterList()
        for _ in range(nlm_depth - 2):
            # SwiGLU: project to 2x hidden, then gate
            self.w_hidden.append(nn.Parameter(torch.empty(d_model, nlm_hidden, nlm_hidden * 2)))
            self.b_hidden.append(nn.Parameter(torch.zeros(d_model, nlm_hidden * 2)))
            self.ln_hidden_scale.append(nn.Parameter(torch.ones(d_model, nlm_hidden)))
            self.ln_hidden_bias.append(nn.Parameter(torch.zeros(d_model, nlm_hidden)))

        # Output projection: (D, nlm_hidden, 1)
        self.w_out = nn.Parameter(torch.empty(d_model, nlm_hidden, 1))
        self.b_out = nn.Parameter(torch.zeros(d_model, 1))

        self._init_weights()

    def _init_tau(self):
        """Initialize per-neuron time constants with diversity."""
        # Log-uniform initialization in [0.5, 4.0] ticks
        # This creates neurons with different temporal preferences:
        # - Low tau (~0.5): focuses on very recent history
        # - High tau (~4.0): integrates over longer history
        with torch.no_grad():
            log_tau = torch.linspace(math.log(0.5), math.log(4.0), self.d_model)
            log_tau = log_tau[torch.randperm(self.d_model)]  # shuffle
            self.tau_raw.copy_(log_tau)

    @property
    def tau(self):
        """Get positive time constants via exp."""
        return torch.exp(self.tau_raw)

    def _init_weights(self):
        """Initialize with diversity across neurons."""
        std = 0.02
        # Different scales per neuron for diverse dynamics
        scales = torch.exp(torch.linspace(-0.7, 0.7, self.d_model))
        scales = scales[torch.randperm(self.d_model)]

        nn.init.normal_(self.w_in, std=std)
        nn.init.normal_(self.w_out, std=std)

        with torch.no_grad():
            self.w_in.mul_(scales.view(-1, 1, 1))
            self.w_out.mul_(scales.view(-1, 1, 1))

        for w in self.w_hidden:
            nn.init.normal_(w, std=std)
            with torch.no_grad():
                w.mul_(scales.view(-1, 1, 1))

    def _layer_norm(self, x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Per-neuron layer norm over hidden dimension."""
        # x: (B, S, D, H), scale/bias: (D, H)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-6)
        return x_norm * scale + bias

    def _swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation: split into two halves, gate with SiLU."""
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (B, S, T, D) pre-activation history
        Returns:
            z: (B, S, D) post-activations
        """
        B, S, T, D = history.shape

        # Pad to max_ticks if needed
        if T < self.max_ticks:
            pad = torch.zeros(B, S, self.max_ticks - T, D,
                            device=history.device, dtype=history.dtype)
            history = torch.cat([pad, history], dim=2)
        else:
            history = history[:, :, -self.max_ticks:]

        # (B, S, D, max_ticks)
        h = history.to(self.w_in.dtype).transpose(-1, -2)

        # Apply per-neuron temporal decay
        # Each neuron has its own time constant tau controlling how far it looks back
        # decay_weight[t] = exp(-(max_ticks - 1 - t) / tau) for position t
        # Position max_ticks-1 (most recent) has weight 1, older positions decay
        t_idx = torch.arange(self.max_ticks, device=h.device, dtype=h.dtype)
        age = (self.max_ticks - 1) - t_idx  # age: 0 for most recent, max_ticks-1 for oldest
        tau = self.tau.to(h.dtype)  # (D,)
        decay_weights = torch.exp(-age.unsqueeze(0) / tau.unsqueeze(1))  # (D, max_ticks)
        h = h * decay_weights  # (B, S, D, max_ticks) * (D, max_ticks) broadcast

        # Input projection with SwiGLU
        h = torch.einsum('bsdt,dth->bsdh', h, self.w_in) + self.b_in
        h = self._swiglu(h)
        h = self._layer_norm(h, self.ln_in_scale, self.ln_in_bias)

        # Hidden layers with SwiGLU and LayerNorm
        for w, b, ln_s, ln_b in zip(self.w_hidden, self.b_hidden,
                                     self.ln_hidden_scale, self.ln_hidden_bias):
            h_in = h  # For internal residual
            h = torch.einsum('bsdh,dhk->bsdk', h, w) + b
            h = self._swiglu(h)
            h = self._layer_norm(h, ln_s, ln_b)
            # Internal residual within hidden layers
            h = h + h_in

        # Output projection
        out = torch.einsum('bsdh,dho->bsdo', h, self.w_out) + self.b_out
        out = out.squeeze(-1)  # (B, S, D)

        return out


class SynchronizationModule(nn.Module):
    """
    Sync: Computes synchronization as dot products of post-activation
    histories between neuron pairs.

    S_{ij} = sum_t z_i^t * z_j^t (with decay weighting)
    """

    def __init__(self, d_model: int, sync_pairs: int):
        super().__init__()
        self.d_model = d_model
        self.sync_pairs = sync_pairs

        # Sample fixed neuron pairs
        idx1 = torch.randint(0, d_model, (sync_pairs,))
        idx2 = torch.randint(0, d_model, (sync_pairs,))
        self.register_buffer('idx1', idx1)
        self.register_buffer('idx2', idx2)

        # Learnable decay
        self.decay_raw = nn.Parameter(torch.zeros(sync_pairs))

    @property
    def decay(self):
        return F.softplus(self.decay_raw)

    def forward(self, z_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_history: (B, S, T, D) post-activation history
        Returns:
            sync: (B, S, sync_pairs)
        """
        B, S, T, D = z_history.shape
        dtype = self.decay_raw.dtype
        h = z_history.to(dtype)

        # Extract neuron pairs
        h1 = h[..., self.idx1]  # (B, S, T, P)
        h2 = h[..., self.idx2]  # (B, S, T, P)

        # Decay weights
        t_idx = torch.arange(T, device=h.device, dtype=dtype)
        weights = torch.exp(-self.decay.unsqueeze(0) * (T - 1 - t_idx).unsqueeze(1))
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)

        # Weighted dot product over time
        sync = (h1 * h2 * weights.unsqueeze(0).unsqueeze(0)).sum(dim=2)

        return sync


class CTMAttention(nn.Module):
    """
    Attention module where sync is projected to form the query.
    Queries the static input KV.
    """

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.d_model // config.n_head
        self.sync_pairs = config.sync_pairs

        # Sync → Query projection
        self.sync_to_q = nn.Linear(config.sync_pairs, config.n_head * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.d_model, bias=False)

        self.dropout = config.dropout

    def forward(
        self,
        sync: torch.Tensor,      # (B, S, sync_pairs)
        static_k: torch.Tensor,  # (B, S, n_kv_head, head_dim)
        static_v: torch.Tensor,  # (B, S, n_kv_head, head_dim)
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sync is projected to query, which attends to static KV.
        Returns attention output o.
        """
        B, S, _ = sync.shape

        # Project sync to query
        q = self.sync_to_q(sync).view(B, S, self.n_head, self.head_dim)

        # Ensure compatible dtype
        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        q = q.to(target_dtype)
        k = static_k.to(target_dtype)
        v = static_v.to(target_dtype)

        # Apply RoPE to query
        rotary_dim = cos.shape[-1] * 2
        if rotary_dim > self.head_dim:
            cos = cos[..., :self.head_dim // 2]
            sin = sin[..., :self.head_dim // 2]
        cos = cos.to(target_dtype)
        sin = sin.to(target_dtype)

        q = apply_rotary_emb(q, cos, sin, interleaved=False)

        # Flash attention
        attn_dropout = self.dropout if self.training else 0.0
        out = flash_attn_func(q, k, v, dropout_p=attn_dropout, causal=True)

        # Project output
        out = out.reshape(B, S, self.n_head * self.head_dim)
        out = out.to(self.o_proj.weight.dtype)

        return self.o_proj(out)


class InputEncoderLayer(nn.Module):
    """
    Single self-attention layer for input encoding.
    Allows tokens to build contextual representations before CTM processing.
    """

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head

        # Self-attention projections
        self.q_proj = nn.Linear(config.d_model, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.d_model, bias=False)

        # FFN (SwiGLU style)
        self.ffn_up = nn.Linear(config.d_model, config.d_model * 4, bias=False)
        self.ffn_gate = nn.Linear(config.d_model, config.d_model * 4, bias=False)
        self.ffn_down = nn.Linear(config.d_model * 4, config.d_model, bias=False)

        # Norms
        self.attn_norm = OptimizedRMSNorm(config.d_model)
        self.ffn_norm = OptimizedRMSNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) input
            cos, sin: RoPE embeddings
        Returns:
            (B, S, D) contextualized output
        """
        B, S, D = x.shape

        # Self-attention with pre-norm
        residual = x
        x = self.attn_norm(x)

        q = self.q_proj(x).view(B, S, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_head, self.head_dim)

        # Apply RoPE
        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        q = q.to(target_dtype)
        k = k.to(target_dtype)
        v = v.to(target_dtype)

        rotary_dim = cos.shape[-1] * 2
        if rotary_dim > self.head_dim:
            cos_rope = cos[..., :self.head_dim // 2]
            sin_rope = sin[..., :self.head_dim // 2]
        else:
            cos_rope = cos
            sin_rope = sin
        cos_rope = cos_rope.to(target_dtype)
        sin_rope = sin_rope.to(target_dtype)

        q = apply_rotary_emb(q, cos_rope, sin_rope, interleaved=False)
        k = apply_rotary_emb(k, cos_rope, sin_rope, interleaved=False)

        # Flash attention (causal)
        attn_out = flash_attn_func(q, k, v, causal=True)
        attn_out = attn_out.reshape(B, S, self.n_head * self.head_dim)
        attn_out = self.o_proj(attn_out.to(self.o_proj.weight.dtype))
        attn_out = self.dropout(attn_out)

        x = residual + attn_out

        # FFN with pre-norm (SwiGLU)
        residual = x
        x = self.ffn_norm(x)
        gate = F.silu(self.ffn_gate(x))
        up = self.ffn_up(x)
        x = self.ffn_down(gate * up)
        x = self.dropout(x)
        x = residual + x

        return x


class InputEncoder(nn.Module):
    """
    Encodes input tokens to produce static KV for CTM attention.
    Uses self-attention layers to build contextual representations.
    """

    def __init__(self, config: CTMConfig, num_layers: int = 2):
        super().__init__()
        self.d_model = config.d_model
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.d_model // config.n_head
        self.num_layers = num_layers

        # Self-attention layers for contextual encoding
        self.layers = nn.ModuleList([
            InputEncoderLayer(config) for _ in range(num_layers)
        ])

        # Final projection to KV
        self.k_proj = nn.Linear(config.d_model, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_head * self.head_dim, bias=False)
        self.norm = OptimizedRMSNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            encoded: (B, S, D) contextualized representation
            static_k: (B, S, n_kv_head, head_dim) with RoPE
            static_v: (B, S, n_kv_head, head_dim)
        """
        B, S, D = x.shape

        # Run through self-attention layers
        for layer in self.layers:
            x = layer(x, cos, sin)

        # Final norm before KV projection
        x = self.norm(x)

        k = self.k_proj(x).view(B, S, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_head, self.head_dim)

        # Apply RoPE to keys
        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        k = k.to(target_dtype)

        rotary_dim = cos.shape[-1] * 2
        if rotary_dim > self.head_dim:
            cos = cos[..., :self.head_dim // 2]
            sin = sin[..., :self.head_dim // 2]
        cos = cos.to(target_dtype)
        sin = sin.to(target_dtype)

        k = apply_rotary_emb(k, cos, sin, interleaved=False)

        return x, k, v.to(target_dtype)


class CTMCore(nn.Module):
    """
    The CTM iterative core - faithful to original paper.

    Per tick t:
        1. Synapse(z^t, o^t) → a^t (pre-activations)
        2. NLM(a_history) → z^{t+1} (post-activations)
        3. Sync(z_history) → synchronization matrix
        4. Attention(sync → query, static_KV) → o^{t+1}
    """

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.num_ticks = config.num_ticks
        self.d_model = config.d_model
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        # Synapse: z + o → a (with tick-conditioned gating)
        self.synapse = SynapseModel(config.d_model, config.num_ticks, config.dropout)

        # NLM: a_history → z
        self.nlm = NeuronLevelModel(
            config.d_model,
            config.nlm_hidden,
            config.nlm_depth,
            config.num_ticks
        )

        # Sync: z_history → sync matrix
        self.sync = SynchronizationModule(config.d_model, config.sync_pairs)

        # Attention: sync → query → attend to data
        self.attention = CTMAttention(config)

        # Normalization
        self.z_norm = OptimizedRMSNorm(config.d_model)

        # Tick embeddings for temporal identity
        self.tick_embed = nn.Embedding(config.num_ticks, config.d_model)

    def _tick_forward(
        self,
        z: torch.Tensor,
        o: torch.Tensor,
        a_hist_tensor: Optional[torch.Tensor],
        z_hist_tensor: torch.Tensor,
        tick_emb: torch.Tensor,
        static_k: torch.Tensor,
        static_v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single tick forward pass (can be checkpointed).

        Args:
            z: current post-activations (B, S, D)
            o: current attention output (B, S, D)
            a_hist_tensor: pre-activation history (B, S, T, D) or None for first tick
            z_hist_tensor: post-activation history (B, S, T+1, D)
            tick_emb: tick embedding (D,)
            static_k, static_v: static KV from encoder
            cos, sin: RoPE

        Returns:
            z_new, o_new, a_hist_new, z_hist_new, sync_out
        """
        # 1. Synapse: z^t + o^t → a^t (tick_emb controls observation gating)
        a = self.synapse(z, o, tick_emb)
        a = a + tick_emb  # add tick identity

        # 2. Update a_history and run NLM
        if a_hist_tensor is not None:
            a_hist_new = torch.cat([a_hist_tensor, a.unsqueeze(2)], dim=2)
        else:
            a_hist_new = a.unsqueeze(2)

        z_new = self.nlm(a_hist_new)
        z_new = self.z_norm(z_new)

        # 3. Update z_history and compute sync
        z_hist_new = torch.cat([z_hist_tensor, z_new.unsqueeze(2)], dim=2)
        sync_out = self.sync(z_hist_new)

        # 4. Attention: sync → query → o^{t+1}
        o_new = self.attention(sync_out, static_k, static_v, cos, sin)

        return z_new, o_new, a_hist_new, z_hist_new, sync_out

    def forward(
        self,
        initial_z: torch.Tensor,   # (B, S, D) initial post-activations
        static_k: torch.Tensor,    # (B, S, n_kv_head, head_dim)
        static_v: torch.Tensor,    # (B, S, n_kv_head, head_dim)
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_ticks: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Run CTM for num_ticks iterations.

        Returns:
            final_z: (B, S, D) final post-activations
            final_sync: (B, S, sync_pairs) final synchronization
            all_syncs: List of sync at each tick
            z_history: List of z at each tick (includes initial)
        """
        from torch.utils.checkpoint import checkpoint

        B, S, D = initial_z.shape
        num_ticks = num_ticks or self.num_ticks
        device = initial_z.device

        # Initialize
        z = initial_z  # post-activations
        o = torch.zeros_like(z)  # attention output starts at zero

        # Use tensors for history (enables checkpointing)
        a_hist_tensor: Optional[torch.Tensor] = None
        z_hist_tensor = z.unsqueeze(2)  # (B, S, 1, D)

        all_syncs: List[torch.Tensor] = []
        z_history: List[torch.Tensor] = [z]

        use_checkpoint = self.use_gradient_checkpointing and self.training

        for tick in range(num_ticks):
            # Get tick embedding
            tick_idx = torch.tensor(tick, device=device)
            tick_emb = self.tick_embed(tick_idx)

            if use_checkpoint:
                # Gradient checkpointing: recompute forward during backward
                z, o, a_hist_tensor, z_hist_tensor, sync_out = checkpoint(
                    self._tick_forward,
                    z, o, a_hist_tensor, z_hist_tensor,
                    tick_emb, static_k, static_v, cos, sin,
                    use_reentrant=False,
                )
            else:
                z, o, a_hist_tensor, z_hist_tensor, sync_out = self._tick_forward(
                    z, o, a_hist_tensor, z_hist_tensor,
                    tick_emb, static_k, static_v, cos, sin,
                )

            all_syncs.append(sync_out)
            z_history.append(z)

        return z, all_syncs[-1], all_syncs, z_history


class CTMLanguageModel(nn.Module):
    """
    Complete CTM for language modeling.

    Architecture:
        1. Token embedding
        2. Input encoder (produces static KV)
        3. CTM iterative core
        4. Output from sync (per CTM paper, sync IS the representation)
    """

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Input encoder
        self.input_encoder = InputEncoder(config)

        # CTM core
        self.ctm_core = CTMCore(config)

        # Output: sync → logits
        self.sync_norm = OptimizedRMSNorm(config.sync_pairs)
        self.output_head = nn.Linear(config.sync_pairs, config.vocab_size, bias=False)

        # RoPE
        self.head_dim = config.d_model // config.n_head
        self._init_rope()
        self._init_weights()

    def _init_rope(self):
        theta = self.config.rope_theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(self.config.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos().to(torch.bfloat16), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().to(torch.bfloat16), persistent=False)

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        return_all_ticks: bool = False,
        num_ticks: Optional[int] = None,
        return_oscillation_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (B, S) token IDs
            return_all_ticks: if True, return logits for all ticks
            return_oscillation_loss: if True, compute and return oscillation loss

        Returns:
            logits: (B, S, V)
            all_logits: List of logits per tick (if return_all_ticks)
            oscillation_loss: scalar loss penalizing flat z trajectories (if return_oscillation_loss)
        """
        B, S = input_ids.shape

        # RoPE
        if input_pos is None:
            input_pos = torch.arange(S, device=input_ids.device)
        cos = self.cos_cached[input_pos]
        sin = self.sin_cached[input_pos]

        # Embed tokens
        x = self.token_embedding(input_ids)

        # Encode to static KV
        encoded, static_k, static_v = self.input_encoder(x, cos, sin)

        # Run CTM (initial z = encoded input)
        final_z, final_sync, all_syncs, z_history = self.ctm_core(
            encoded, static_k, static_v, cos, sin, num_ticks
        )

        # Compute oscillation loss if requested
        oscillation_loss = None
        if return_oscillation_loss and len(z_history) > 1:
            # Stack z history: (B, S, T, D)
            z_stack = torch.stack(z_history, dim=2)

            # Target thresholds - stop pushing once reached
            target_variance = 0.3
            target_delta = 0.2

            # 1. Variance loss: penalize low variance across ticks (up to target)
            z_var = z_stack.var(dim=2)  # (B, S, D)
            var_mean = z_var.mean()
            # Only penalize if below target, otherwise loss = 0
            var_loss = F.relu(target_variance - var_mean) / target_variance

            # 2. Delta loss: penalize small tick-to-tick changes (up to target)
            z_deltas = z_stack[:, :, 1:, :] - z_stack[:, :, :-1, :]  # (B, S, T-1, D)
            delta_magnitude = z_deltas.abs().mean()
            # Only penalize if below target
            delta_loss = F.relu(target_delta - delta_magnitude) / target_delta

            # 3. Diversity loss: penalize neurons being too correlated
            # Compute correlation between neurons across ticks
            z_flat = z_stack.mean(dim=(0, 1))  # (T, D) - average over batch and seq
            z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
            # Sample neuron pairs for efficiency
            n_pairs = min(256, z_centered.shape[1] // 2)
            idx1 = torch.randperm(z_centered.shape[1], device=z_centered.device)[:n_pairs]
            idx2 = torch.randperm(z_centered.shape[1], device=z_centered.device)[:n_pairs]
            corr = (z_centered[:, idx1] * z_centered[:, idx2]).sum(dim=0)
            corr = corr / (z_centered[:, idx1].norm(dim=0) * z_centered[:, idx2].norm(dim=0) + 1e-6)
            diversity_loss = corr.abs().mean()  # penalize high correlation

            # Combined oscillation loss (will be ~0 once targets reached)
            oscillation_loss = var_loss + delta_loss + 0.5 * diversity_loss

        # Output from sync
        if return_all_ticks:
            all_logits = [self.output_head(self.sync_norm(s)) for s in all_syncs]
            return all_logits[-1], all_logits, oscillation_loss
        else:
            logits = self.output_head(self.sync_norm(final_sync))
            return logits, None, oscillation_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        num_ticks: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple autoregressive generation."""
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_seq_len:]
            logits, _, _ = self(context, num_ticks=num_ticks)
            next_logits = logits[:, -1, :]

            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                if top_k is not None:
                    v, _ = torch.topk(probs, top_k)
                    probs[probs < v[:, [-1]]] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    @torch.no_grad()
    def get_nlm_diagnostics(
        self,
        input_ids: torch.Tensor,
        num_ticks: Optional[int] = None,
        return_full: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward pass and collect neuron dynamics for visualization.
        Memory-efficient: only stores means unless return_full=True.

        Returns:
            z_activations: (num_ticks+1, D) - post-activations at each tick (mean over B, S)
            a_activations: (num_ticks, D) - pre-activations at each tick (mean over B, S)
            sync_values: (num_ticks, sync_pairs) - sync at each tick (mean over B, S)
            z_full: (num_ticks+1, B, S, D) - full post-activations (if return_full)
            a_full: (num_ticks, B, S, D) - full pre-activations (if return_full)
        """
        self.eval()
        B, S = input_ids.shape
        device = input_ids.device
        num_ticks = num_ticks or self.config.num_ticks
        D = self.config.d_model

        # Get RoPE
        input_pos = torch.arange(S, device=device)
        cos = self.cos_cached[input_pos]
        sin = self.sin_cached[input_pos]

        # Embed and encode
        x = self.token_embedding(input_ids)
        encoded, static_k, static_v = self.input_encoder(x, cos, sin)

        # Pre-allocate history tensors (needed for NLM and Sync)
        # These are the minimum required for the forward pass
        a_hist = torch.zeros(B, S, num_ticks, D, device=device, dtype=encoded.dtype)
        z_hist = torch.zeros(B, S, num_ticks + 1, D, device=device, dtype=encoded.dtype)

        # Pre-allocate output tensors (small, on CPU to save GPU memory)
        z_means = torch.zeros(num_ticks + 1, D, device='cpu')
        a_means = torch.zeros(num_ticks, D, device='cpu')
        sync_means = torch.zeros(num_ticks, self.config.sync_pairs, device='cpu')

        # Only allocate full storage if requested
        if return_full:
            z_full = torch.zeros(num_ticks + 1, B, S, D, device='cpu')
            a_full = torch.zeros(num_ticks, B, S, D, device='cpu')

        # Initialize
        z = encoded
        o = torch.zeros_like(z)
        z_hist[:, :, 0, :] = z

        # Store initial z
        z_means[0] = z.mean(dim=(0, 1)).cpu()
        if return_full:
            z_full[0] = z.cpu()

        for tick in range(num_ticks):
            tick_idx = torch.tensor(tick, device=device)
            tick_emb = self.ctm_core.tick_embed(tick_idx)

            # 1. Synapse (with tick-conditioned gating)
            a = self.ctm_core.synapse(z, o, tick_emb)
            a = a + tick_emb
            a_hist[:, :, tick, :] = a

            # Store a stats
            a_means[tick] = a.mean(dim=(0, 1)).cpu()
            if return_full:
                a_full[tick] = a.cpu()

            # 2. NLM (uses history up to current tick)
            z = self.ctm_core.nlm(a_hist[:, :, :tick + 1, :])
            z = self.ctm_core.z_norm(z)
            z_hist[:, :, tick + 1, :] = z

            # Store z stats
            z_means[tick + 1] = z.mean(dim=(0, 1)).cpu()
            if return_full:
                z_full[tick + 1] = z.cpu()

            # 3. Sync (uses z history up to current)
            sync_out = self.ctm_core.sync(z_hist[:, :, :tick + 2, :])
            sync_means[tick] = sync_out.mean(dim=(0, 1)).cpu()

            # 4. Attention
            o = self.ctm_core.attention(sync_out, static_k, static_v, cos, sin)

        # Clean up GPU tensors
        del a_hist, z_hist, z, o, a, sync_out, encoded, static_k, static_v, x
        torch.cuda.empty_cache()

        diagnostics = {
            'z_activations': z_means,
            'a_activations': a_means,
            'sync_values': sync_means,
            'num_ticks': torch.tensor(num_ticks),
        }

        if return_full:
            diagnostics['z_full'] = z_full
            diagnostics['a_full'] = a_full

        return diagnostics
