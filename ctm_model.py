"""
Continuous Thought Machine (CTM) Hybrid Architecture for Language Modeling.

Implements CTM's iterative refinement mechanism with:
- Neuron-Level Models (NLMs): Private MLPs per hidden dimension
- Neural Synchronization: Correlation-based representations
- Internal Ticks: Iterative processing decoupled from sequence length
- Data as Static KV: Input encoded once, queried each tick
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

logger = logging.getLogger(__name__)

# --- SOTA Imports ---
try:
    from flash_attn import flash_attn_func
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    raise ImportError(
        "flash_attn is required for CTM. Install with: pip install flash-attn --no-build-isolation"
    )

# Import shared components from model.py
from model import OptimizedRMSNorm, OptimizedMLP


# --- Configuration ---

@dataclass
class CTMConfig:
    """Configuration for CTM Language Model."""
    # Core dimensions
    vocab_size: int
    d_model: int = 512
    n_head: int = 8
    n_kv_head: Optional[int] = None  # For GQA; defaults to n_head
    n_layer: int = 6
    max_seq_len: int = 2048
    d_ff: Optional[int] = None  # FFN dimension; computed if None

    # CTM-specific
    num_ticks: int = 8
    nlm_hidden: int = 64
    nlm_depth: int = 2
    sync_pairs: int = 512

    # Enhanced sync parameters
    n_sync_heads: int = 8           # Number of sync heads (like attention heads)
    sync_local_window: int = 3      # Window for exponential decay (kept for compat)
    use_enhanced_sync: bool = True  # Use enhanced sync vs original
    sync_order: int = 2             # Correlation order (2=covariance, faithful to CTM paper)

    # Enhanced NLM parameters
    use_enhanced_nlm: bool = True   # Use enhanced NLM with temporal attention + gating

    # Training
    dropout: float = 0.0
    rope_theta: float = 500000.0
    use_gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.d_ff is None:
            # SwiGLU optimal sizing
            self.d_ff = int(2 * (4 * self.d_model) / 3)
            self.d_ff = 256 * ((self.d_ff + 256 - 1) // 256)
        # Ensure sync_pairs is divisible by n_sync_heads
        assert self.sync_pairs % self.n_sync_heads == 0, \
            f"sync_pairs ({self.sync_pairs}) must be divisible by n_sync_heads ({self.n_sync_heads})"
        assert self.sync_order >= 2, "sync_order must be >= 2 for correlation computation"


# --- Neuron-Level Models ---

class NeuronLevelModels(nn.Module):
    """
    D independent MLPs, one per hidden dimension.

    Each NLM processes its temporal history of activations and produces
    an updated activation. Efficiently batched using einsum operations.

    Input: (B, S, T, D) - history of activations per neuron
    Output: (B, S, D) - updated state
    """

    def __init__(self, d_model: int, nlm_hidden: int, nlm_depth: int, max_ticks: int):
        super().__init__()
        self.d_model = d_model
        self.nlm_hidden = nlm_hidden
        self.nlm_depth = nlm_depth
        self.max_ticks = max_ticks

        # Input projection: (D, max_ticks, nlm_hidden)
        self.w_in = nn.Parameter(torch.empty(d_model, max_ticks, nlm_hidden))
        self.b_in = nn.Parameter(torch.zeros(d_model, nlm_hidden))

        # Hidden layers (if depth > 2)
        if nlm_depth > 2:
            self.w_hidden = nn.ParameterList([
                nn.Parameter(torch.empty(d_model, nlm_hidden, nlm_hidden))
                for _ in range(nlm_depth - 2)
            ])
            self.b_hidden = nn.ParameterList([
                nn.Parameter(torch.zeros(d_model, nlm_hidden))
                for _ in range(nlm_depth - 2)
            ])
        else:
            self.w_hidden = nn.ParameterList()
            self.b_hidden = nn.ParameterList()

        # Output projection: (D, nlm_hidden, 1)
        self.w_out = nn.Parameter(torch.empty(d_model, nlm_hidden, 1))
        self.b_out = nn.Parameter(torch.zeros(d_model, 1))

        self._init_weights()

    def _init_weights(self):
        """
        Initialize with DIVERSITY across neurons for heterogeneous temporal dynamics.
        """
        base_std = 0.02

        # Per-neuron scale factors for diverse dynamics
        # Log-uniform in ~[0.5, 2.0] range
        neuron_scales = torch.exp(torch.linspace(-0.7, 0.7, self.d_model))
        neuron_scales = neuron_scales[torch.randperm(self.d_model)]

        nn.init.normal_(self.w_in, mean=0.0, std=base_std)
        nn.init.normal_(self.w_out, mean=0.0, std=base_std)
        with torch.no_grad():
            self.w_in.mul_(neuron_scales.view(-1, 1, 1))
            self.w_out.mul_(neuron_scales.view(-1, 1, 1))

        for w in self.w_hidden:
            nn.init.normal_(w, mean=0.0, std=base_std)
            with torch.no_grad():
                w.mul_(neuron_scales.view(-1, 1, 1))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (B, S, T, D) where T is current tick count (may be < max_ticks)

        Returns:
            (B, S, D) updated activations
        """
        B, S, T, D = history.shape

        # Pad history to max_ticks if needed (for consistent weight shapes)
        if T < self.max_ticks:
            padding = torch.zeros(B, S, self.max_ticks - T, D,
                                  device=history.device, dtype=history.dtype)
            history_padded = torch.cat([padding, history], dim=2)
        else:
            history_padded = history[:, :, -self.max_ticks:]

        # Convert to weight dtype for computation
        compute_dtype = self.w_in.dtype
        h = history_padded.to(compute_dtype)

        # Transpose for batched matmul: (B, S, D, max_ticks)
        h = h.transpose(-1, -2)

        # Input projection: (B, S, D, max_ticks) @ (D, max_ticks, H) -> (B, S, D, H)
        h = torch.einsum('bsdt,dth->bsdh', h, self.w_in) + self.b_in
        h = F.gelu(h)

        # Hidden layers
        for w, b in zip(self.w_hidden, self.b_hidden):
            h = torch.einsum('bsdh,dhk->bsdk', h, w) + b
            h = F.gelu(h)

        # Output projection: (B, S, D, H) @ (D, H, 1) -> (B, S, D, 1)
        out = torch.einsum('bsdh,dho->bsdo', h, self.w_out) + self.b_out

        return out.squeeze(-1)  # (B, S, D)


class EnhancedNeuronLevelModels(nn.Module):
    """
    Enhanced NLM with temporal attention and per-neuron gating (faithful to CTM).

    Improvements over basic NLM:
    1. Temporal attention: Each neuron learns which past states matter most,
       dynamically weighting history entries instead of fixed MLP processing
    2. Per-neuron gating: Each neuron independently decides how much to update
       vs keep previous state (NO cross-channel communication)

    CRITICAL: Neurons remain INDEPENDENT - no cross-channel communication.
    Neurons only "communicate" through sync (correlation patterns).
    The gating is per-neuron with private weights, not a shared linear layer.

    Input: (B, S, T, D) - history of activations per neuron
    Output: (B, S, D) - updated state
    """

    def __init__(
        self,
        d_model: int,
        nlm_hidden: int,
        nlm_depth: int,
        max_ticks: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.nlm_hidden = nlm_hidden
        self.nlm_depth = nlm_depth
        self.max_ticks = max_ticks

        # === 1. Temporal Attention ===
        # Each neuron has a learned query to attend over its history
        # Query: (D, nlm_hidden) - one query vector per neuron
        self.temporal_query = nn.Parameter(torch.randn(d_model, nlm_hidden) * 0.02)

        # Project history to keys and values: (D, 1, nlm_hidden) per time step
        # Using per-neuron projections for independence
        self.temporal_k_proj = nn.Parameter(torch.empty(d_model, 1, nlm_hidden))
        self.temporal_v_proj = nn.Parameter(torch.empty(d_model, 1, nlm_hidden))

        # === 2. Per-Neuron MLP (processes attention output) ===
        # After temporal attention aggregates history, MLP refines it
        self.w_in = nn.Parameter(torch.empty(d_model, nlm_hidden, nlm_hidden))
        self.b_in = nn.Parameter(torch.zeros(d_model, nlm_hidden))

        if nlm_depth > 2:
            self.w_hidden = nn.ParameterList([
                nn.Parameter(torch.empty(d_model, nlm_hidden, nlm_hidden))
                for _ in range(nlm_depth - 2)
            ])
            self.b_hidden = nn.ParameterList([
                nn.Parameter(torch.zeros(d_model, nlm_hidden))
                for _ in range(nlm_depth - 2)
            ])
        else:
            self.w_hidden = nn.ParameterList()
            self.b_hidden = nn.ParameterList()

        self.w_out = nn.Parameter(torch.empty(d_model, nlm_hidden, 1))
        self.b_out = nn.Parameter(torch.zeros(d_model, 1))

        # === 3. Per-Neuron Gating (NO cross-channel mixing!) ===
        # Each neuron has its own gate weights: maps [nlm_out_d, recent_d] -> gate_d
        # Shape: (D, 2) weights + (D,) bias per neuron
        # gate_d = sigmoid(w_gate[d] @ [nlm_out[d], recent[d]] + b_gate[d])
        self.w_gate = nn.Parameter(torch.empty(d_model, 2))
        self.b_gate = nn.Parameter(torch.full((d_model,), -0.5))  # Less conservative for more dynamics

        self._init_weights()

    def _init_weights(self):
        """
        Initialize NLM weights with DIVERSITY across neurons.

        Key insight from CTM: Each NLM should develop different temporal dynamics
        (different "frequencies"). Homogeneous initialization leads to similar
        dynamics across neurons, reducing sync's ability to capture meaningful
        coordination patterns.

        Strategy:
        1. Temporal queries get per-neuron scale factors (diverse attention patterns)
        2. MLP weights get per-neuron scale variation (diverse processing)
        3. Gate biases vary per neuron (diverse update rates)
        """
        base_std = 0.02

        # === Diverse temporal attention initialization ===
        # Each neuron gets a different scale for its temporal query
        # This encourages different neurons to attend to different time scales
        # Scale factors log-uniformly distributed: some neurons 0.5x, some 2x base
        neuron_scales = torch.exp(torch.linspace(-0.7, 0.7, self.d_model))  # ~[0.5, 2.0]
        neuron_scales = neuron_scales[torch.randperm(self.d_model)]  # Shuffle

        # temporal_query: (D, nlm_hidden) - scale each neuron's query differently
        nn.init.normal_(self.temporal_query, std=base_std)
        with torch.no_grad():
            self.temporal_query.mul_(neuron_scales.unsqueeze(1))

        # temporal_k_proj, temporal_v_proj: (D, 1, nlm_hidden)
        nn.init.normal_(self.temporal_k_proj, std=base_std)
        nn.init.normal_(self.temporal_v_proj, std=base_std)
        with torch.no_grad():
            self.temporal_k_proj.mul_(neuron_scales.view(-1, 1, 1))
            self.temporal_v_proj.mul_(neuron_scales.view(-1, 1, 1))

        # === Diverse MLP initialization ===
        # w_in, w_out: different scales per neuron for diverse dynamics
        nn.init.normal_(self.w_in, std=base_std)
        nn.init.normal_(self.w_out, std=base_std)
        with torch.no_grad():
            self.w_in.mul_(neuron_scales.view(-1, 1, 1))
            self.w_out.mul_(neuron_scales.view(-1, 1, 1))

        for w in self.w_hidden:
            nn.init.normal_(w, std=base_std)
            with torch.no_grad():
                w.mul_(neuron_scales.view(-1, 1, 1))

        # === Diverse gating initialization ===
        # Different neurons start with different update tendencies
        # Some neurons more "sticky" (low gate), some more "responsive" (high gate)
        nn.init.normal_(self.w_gate, std=base_std)
        with torch.no_grad():
            # Gate biases: vary from -1.5 to +1.5 around base of -0.5
            # Final range: -2.0 to +1.0
            # sigmoid(-2.0) ≈ 0.12 (slow neurons), sigmoid(+1.0) ≈ 0.73 (fast neurons)
            # This creates diverse update rates across neurons
            gate_bias_variation = torch.linspace(-1.5, 1.5, self.d_model)
            gate_bias_variation = gate_bias_variation[torch.randperm(self.d_model)]
            self.b_gate.add_(gate_bias_variation)

    def forward(
        self, history: torch.Tensor, return_diagnostics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            history: (B, S, T, D) where T is current tick count
            return_diagnostics: If True, return dict with attention weights and gate values

        Returns:
            output: (B, S, D) updated activations
            diagnostics (optional): Dict with:
                - 'attn_weights': (B, S, D, T) temporal attention per neuron
                - 'gate_values': (B, S, D) update gate per neuron
                - 'attn_entropy': (D,) entropy of attention distribution per neuron
                - 'gate_mean': (D,) mean gate value per neuron
        """
        B, S, T, D = history.shape
        compute_dtype = self.temporal_query.dtype
        h = history.to(compute_dtype)

        # Most recent state (for gating)
        most_recent = h[:, :, -1, :]  # (B, S, D)

        # === 1. Temporal Attention ===
        # Project history to keys and values per neuron
        # h: (B, S, T, D) -> transpose to (B, S, D, T)
        h_t = h.transpose(-1, -2)  # (B, S, D, T)

        # Keys: (B, S, D, T) @ (D, 1, H) -> (B, S, D, T, H) via broadcasting
        # We compute this as: for each neuron d, k[d] = h[d,:] @ k_proj[d]
        keys = torch.einsum('bsdt,doh->bsdth', h_t, self.temporal_k_proj)  # (B, S, D, T, H)
        values = torch.einsum('bsdt,doh->bsdth', h_t, self.temporal_v_proj)  # (B, S, D, T, H)

        # Query: (D, H) broadcast to (B, S, D, H)
        query = self.temporal_query.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H)

        # Attention scores: (B, S, D, H) @ (B, S, D, T, H).T -> (B, S, D, T)
        attn_scores = torch.einsum('bsdh,bsdth->bsdt', query.expand(B, S, -1, -1), keys)
        attn_scores = attn_scores / math.sqrt(self.nlm_hidden)

        # Softmax over time dimension
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, S, D, T)

        # Weighted sum of values: (B, S, D, T) @ (B, S, D, T, H) -> (B, S, D, H)
        attn_out = torch.einsum('bsdt,bsdth->bsdh', attn_weights, values)

        # === 2. Per-Neuron MLP ===
        h = torch.einsum('bsdh,dhk->bsdk', attn_out, self.w_in) + self.b_in
        h = F.gelu(h)

        for w, b in zip(self.w_hidden, self.b_hidden):
            h = torch.einsum('bsdh,dhk->bsdk', h, w) + b
            h = F.gelu(h)

        nlm_out = torch.einsum('bsdh,dho->bsdo', h, self.w_out) + self.b_out
        nlm_out = nlm_out.squeeze(-1)  # (B, S, D)

        # === 3. Per-Neuron Gating (NO cross-channel mixing!) ===
        # Stack inputs per neuron: (B, S, D, 2)
        gate_input = torch.stack([nlm_out, most_recent], dim=-1)
        # Per-neuron gate: (B, S, D, 2) @ (D, 2) -> (B, S, D) via einsum
        # Each neuron d uses only its own w_gate[d] weights
        gate_logits = torch.einsum('bsdi,di->bsd', gate_input, self.w_gate) + self.b_gate
        update_gate = torch.sigmoid(gate_logits)  # (B, S, D)

        # Blend NLM output with most recent state (per neuron, independently)
        output = update_gate * nlm_out + (1 - update_gate) * most_recent

        if return_diagnostics:
            with torch.no_grad():
                # Compute attention entropy per neuron (averaged over batch and sequence)
                # High entropy = uniform attention (integrator), low entropy = focused (selective)
                attn_entropy = -(attn_weights * (attn_weights + 1e-8).log()).sum(dim=-1)  # (B, S, D)
                attn_entropy = attn_entropy.mean(dim=(0, 1))  # (D,)

                # Mean gate value per neuron (averaged over batch and sequence)
                # High = responsive (uses NLM output), low = sticky (keeps previous)
                gate_mean = update_gate.mean(dim=(0, 1))  # (D,)

                # Average attention weights per neuron (shows temporal focus)
                # (B, S, D, T) -> (D, T)
                attn_mean = attn_weights.mean(dim=(0, 1))

                diagnostics = {
                    'attn_weights': attn_weights.detach(),  # (B, S, D, T)
                    'gate_values': update_gate.detach(),     # (B, S, D)
                    'attn_entropy': attn_entropy,            # (D,)
                    'gate_mean': gate_mean,                  # (D,)
                    'attn_mean': attn_mean,                  # (D, T)
                }
            return output, diagnostics

        return output


# --- Synchronization Module ---

class SynchronizationModule(nn.Module):
    """
    Computes neural synchronization from post-activation history.

    Synchronization captures the correlation structure between neurons
    over time, which serves as the representation in CTM.

    Input: (B, S, T, D) - post-activation history
    Output: (B, S, sync_pairs) - sync features
    """

    def __init__(self, d_model: int, sync_pairs: int):
        super().__init__()
        self.d_model = d_model
        self.sync_pairs = sync_pairs

        # Pre-sample fixed neuron pairs (registered as buffers for device handling)
        idx1 = torch.randint(0, d_model, (sync_pairs,))
        idx2 = torch.randint(0, d_model, (sync_pairs,))
        self.register_buffer('pair_idx1', idx1)
        self.register_buffer('pair_idx2', idx2)

        # Learnable decay per pair (controls temporal weighting)
        self.decay_raw = nn.Parameter(torch.zeros(sync_pairs))

    @property
    def decay(self) -> torch.Tensor:
        # Ensure non-negative decay via softplus
        return F.softplus(self.decay_raw)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (B, S, T, D) post-activation history

        Returns:
            (B, S, sync_pairs) synchronization values
        """
        B, S, T, D = history.shape

        # Convert to compute dtype (float32 for stability)
        compute_dtype = self.decay_raw.dtype
        history_compute = history.to(compute_dtype)

        # Extract histories for sampled neuron pairs
        # (B, S, T, P) where P = sync_pairs
        h1 = history_compute[..., self.pair_idx1]
        h2 = history_compute[..., self.pair_idx2]

        # Compute decay weights: exp(-decay * (T-1-t)) for t in [0, T-1]
        # This weights recent activations more heavily
        time_indices = torch.arange(T, device=history.device, dtype=compute_dtype)
        decay_weights = torch.exp(-self.decay.unsqueeze(0) * (T - 1 - time_indices).unsqueeze(1))
        # decay_weights: (T, P)

        # Normalize weights
        decay_weights = decay_weights / (decay_weights.sum(dim=0, keepdim=True).sqrt() + 1e-8)

        # Weighted correlation: sum over time of (h1 * h2 * weight)
        # (B, S, T, P) * (T, P) summed over T -> (B, S, P)
        sync = (h1 * h2 * decay_weights.unsqueeze(0).unsqueeze(0)).sum(dim=2)

        return sync


class EnhancedSynchronizationModule(nn.Module):
    """
    Pure correlation-based synchronization module (faithful to CTM).

    CRITICAL DESIGN PRINCIPLE: Sync is a MEASUREMENT of neural coordination,
    not a learned transformation. Temporal integration is NLM's job.

    This module:
    1. Projects history to multiple learned spaces (captures what to correlate)
    2. Computes element-wise products (the actual correlation measurement)
    3. Averages over time with simple exponential decay (not learned attention)
    4. Applies minimal normalization for numerical stability

    What this module does NOT do (separation of concerns):
    - NO learned temporal attention (NLMs handle temporal integration)
    - NO state-dependent gating (sync should be pure function of history)
    - NO learned output projection (sync IS the representation, not input to one)
    - NO tanh squashing (preserves correlation magnitude information)

    Input: (B, S, T, D) - post-activation history
    Output: (B, S, sync_pairs) - sync features (raw correlation patterns)
    """

    def __init__(
        self,
        d_model: int,
        sync_pairs: int,
        n_heads: int = 8,
        local_window: int = 3,  # Kept for config compat, but not used for learned gating
        dropout: float = 0.0,
        order: int = 2,  # Default to 2 for covariance (second-order statistics)
    ):
        super().__init__()
        self.d_model = d_model
        self.sync_pairs = sync_pairs
        self.n_heads = n_heads
        self.head_dim = sync_pairs // n_heads
        self.order = order

        if order < 2:
            raise ValueError("order must be >= 2 for correlation computation")

        # Learned projections define WHAT to correlate (not HOW to weight time)
        # These learn which neuron combinations are meaningful to track
        self.projections = nn.ModuleList([
            nn.Linear(d_model, sync_pairs, bias=False)
            for _ in range(order)
        ])

        # Simple exponential decay for temporal weighting (not learned per-sample)
        # This gives recent states more weight, but uniformly across all inputs
        self.decay_rate = nn.Parameter(torch.tensor(0.5))  # Learnable but global

        # Minimal output normalization for numerical stability only
        # No learned projection - sync IS the representation
        self.norm = OptimizedRMSNorm(sync_pairs)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for proj in self.projections:
            nn.init.normal_(proj.weight, std=0.02)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (B, S, T, D) post-activation history

        Returns:
            (B, S, sync_pairs) synchronization values (raw correlations)
        """
        B, S, T, D = history.shape

        # Use float32 for numerical stability in correlation computation
        compute_dtype = self.projections[0].weight.dtype
        h = history.to(compute_dtype)

        # Project history to correlation spaces
        # NO normalization or squashing - preserve the signal
        proj_terms = []
        for proj in self.projections:
            term = proj(h)  # (B, S, T, sync_pairs)
            proj_terms.append(term)

        # Compute element-wise product (correlation measurement)
        # For order=2: A * B captures covariance-like patterns
        # For order=3: A * B * C captures higher-order coordination
        correlation = proj_terms[0]
        for term in proj_terms[1:]:
            correlation = correlation * term  # (B, S, T, sync_pairs)

        # Simple exponential decay weighting (not learned per-sample)
        # decay_weight[t] = exp(-decay_rate * (T - 1 - t))
        # More recent = higher weight
        decay_rate = F.softplus(self.decay_rate)  # Ensure positive
        time_offsets = torch.arange(T, device=h.device, dtype=compute_dtype)
        time_offsets = T - 1 - time_offsets  # [T-1, T-2, ..., 1, 0]
        decay_weights = torch.exp(-decay_rate * time_offsets)  # (T,)
        decay_weights = decay_weights / (decay_weights.sum() + 1e-8)  # Normalize

        # Weighted average over time: (B, S, T, sync_pairs) * (T,) -> (B, S, sync_pairs)
        sync = torch.einsum('bstp,t->bsp', correlation, decay_weights)

        # Minimal normalization for stability (not a learned transformation)
        sync = self.norm(sync)
        sync = self.dropout(sync)

        return sync


# --- Synapse Model ---

class SynapseModel(nn.Module):
    """
    Integrates current state with observation (cross-attention output).

    U-Net style MLP with skip connections for stable gradient flow.

    Per the CTM paper, sync determines WHERE to look (via attention query),
    and the synapse integrates WHAT was found (observation) with current state.
    Sync does NOT enter the synapse directly.

    Inputs:
        state: (B, S, D) - current hidden state
        observation: (B, S, D) - cross-attention output
    Output: (B, S, D) - integrated update
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()

        input_dim = d_model + d_model  # state + observation only

        # Encoder path
        self.enc1 = nn.Linear(input_dim, d_model, bias=False)
        self.enc2 = nn.Linear(d_model, d_model // 2, bias=False)

        # Decoder path with skip connections
        self.dec2 = nn.Linear(d_model // 2, d_model, bias=False)
        self.dec1 = nn.Linear(d_model * 2, d_model, bias=False)  # *2 for skip

        self.norm1 = OptimizedRMSNorm(d_model)
        self.norm2 = OptimizedRMSNorm(d_model // 2)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        # Concatenate inputs (no sync - it influenced the query instead)
        x = torch.cat([state, observation], dim=-1)

        # Encoder
        h1 = F.gelu(self.enc1(x))
        h1 = self.norm1(h1)
        h2 = F.gelu(self.enc2(h1))
        h2 = self.norm2(h2)

        # Decoder with skip connections
        d2 = F.gelu(self.dec2(h2))
        d2 = self.dropout(d2)
        d1 = torch.cat([d2, h1], dim=-1)  # Skip connection
        out = self.dec1(d1)

        return out


# --- CTM Self-Attention ---

class CTMSelfAttention(nn.Module):
    """
    Self-attention for CTM state during iterative processing.

    Allows tokens to communicate with each other during thinking,
    enabling reasoning across positions within each tick.

    Uses FlashAttention-2 with causal masking for autoregressive modeling.
    Supports Grouped Query Attention (GQA).
    """

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.d_model // config.n_head
        self.dropout = config.dropout

        assert config.n_head % config.n_kv_head == 0
        self.n_rep = config.n_head // config.n_kv_head

        # QKV projections
        self.q_proj = nn.Linear(config.d_model, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_head * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,          # (B, S, D) - CTM state
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, S, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_head, self.head_dim)

        # Ensure flash-attn compatible dtype
        target_dtype = q.dtype
        if target_dtype not in (torch.float16, torch.bfloat16):
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if q.dtype != target_dtype:
            q = q.to(dtype=target_dtype)
            k = k.to(dtype=target_dtype)
            v = v.to(dtype=target_dtype)

        # Apply RoPE to Q and K
        rotary_dim = cos.shape[-1] * 2
        if rotary_dim > self.head_dim:
            trim = self.head_dim // 2
            cos = cos[..., :trim]
            sin = sin[..., :trim]
        if cos.dtype != target_dtype:
            cos = cos.to(dtype=target_dtype)
            sin = sin.to(dtype=target_dtype)

        q = apply_rotary_emb(q, cos, sin, interleaved=False)
        k = apply_rotary_emb(k, cos, sin, interleaved=False)

        # FlashAttention with causal mask
        attn_dropout = self.dropout if self.training else 0.0
        output = flash_attn_func(
            q, k, v,
            dropout_p=attn_dropout,
            causal=True,
            window_size=(-1, -1)
        )

        # Reshape and project output
        output = output.reshape(B, S, self.n_head * self.head_dim)
        output = output.to(self.o_proj.weight.dtype)

        return self.o_proj(output)


# --- CTM Cross-Attention ---

class CTMCrossAttention(nn.Module):
    """
    Cross-attention from CTM state to static input KV.

    Per the CTM paper, sync determines WHERE to look by modulating the query.
    The sync features influence query formation, steering attention to relevant
    parts of the input based on the current synchronization pattern.

    Uses FlashAttention-2 with causal masking for autoregressive modeling.
    Supports Grouped Query Attention (GQA).
    """

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.d_model // config.n_head
        self.dropout = config.dropout
        self.sync_pairs = config.sync_pairs

        assert config.n_head % config.n_kv_head == 0
        self.n_rep = config.n_head // config.n_kv_head

        # Sync projection: maps sync features to query modulation
        # This allows sync to steer WHERE the model looks
        self.sync_proj = nn.Linear(config.sync_pairs, config.d_model, bias=False)

        # Query projection (from CTM state + sync modulation)
        self.q_proj = nn.Linear(config.d_model, config.n_head * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.d_model, bias=False)

    def forward(
        self,
        query: torch.Tensor,      # (B, S, D) - CTM state
        key: torch.Tensor,        # (B, S, n_kv_head, head_dim) - static key
        value: torch.Tensor,      # (B, S, n_kv_head, head_dim) - static value
        cos: torch.Tensor,
        sin: torch.Tensor,
        sync: torch.Tensor,       # (B, S, sync_pairs) - sync features to modulate query
    ) -> torch.Tensor:
        B, S, _ = query.shape

        # Sync modulates the query: determines WHERE to look
        # Project sync to d_model and add to state before query projection
        sync_modulation = self.sync_proj(sync)
        query_input = query + sync_modulation

        # Project query (now influenced by sync)
        q = self.q_proj(query_input)
        q = q.view(B, S, self.n_head, self.head_dim)

        # Ensure flash-attn compatible dtype
        target_dtype = q.dtype
        if target_dtype not in (torch.float16, torch.bfloat16):
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if q.dtype != target_dtype:
            q = q.to(dtype=target_dtype)

        # Apply RoPE to query
        rotary_dim = cos.shape[-1] * 2
        if rotary_dim > self.head_dim:
            trim = self.head_dim // 2
            cos = cos[..., :trim]
            sin = sin[..., :trim]
        if cos.dtype != target_dtype:
            cos = cos.to(dtype=target_dtype)
            sin = sin.to(dtype=target_dtype)

        q = apply_rotary_emb(q, cos, sin, interleaved=False)

        # Key and value already have RoPE applied during encoding
        k = key.to(dtype=target_dtype) if key.dtype != target_dtype else key
        v = value.to(dtype=target_dtype) if value.dtype != target_dtype else value

        # FlashAttention with causal mask
        attn_dropout = self.dropout if self.training else 0.0
        output = flash_attn_func(
            q, k, v,
            dropout_p=attn_dropout,
            causal=True,
            window_size=(-1, -1)
        )

        # Reshape and project output
        output = output.reshape(B, S, self.n_head * self.head_dim)
        output = output.to(self.o_proj.weight.dtype)

        return self.o_proj(output)


# --- CTM Layer ---

class CTMLayer(nn.Module):
    """
    Single CTM layer with per-layer temporal processing.

    Architecture per layer:
        1. Self-Attention: Tokens communicate with each other
        2. Cross-Attention: Query static KV (sync modulates WHERE to look)
        3. Synapse: Integrate state + observation
        4. Temporal NLM: Process layer's history (depth in time)
        5. FFN: Final transformation

    Each layer maintains its own NLM for per-channel temporal processing,
    enabling "depth in time" rather than just depth in layers.
    """

    def __init__(self, config: CTMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config.d_model

        # Self-attention (tokens communicate during thinking)
        self.self_attn = CTMSelfAttention(config)

        # Cross-attention to static KV (sync modulates query inside)
        self.cross_attn = CTMCrossAttention(config)

        # Synapse model (integrates state and observation)
        self.synapse = SynapseModel(
            d_model=config.d_model,
            dropout=config.dropout
        )

        # Per-layer NLM: each layer has private temporal processors
        # Each layer only sees its own history across ticks (num_ticks entries)
        if config.use_enhanced_nlm:
            self.nlm = EnhancedNeuronLevelModels(
                d_model=config.d_model,
                nlm_hidden=config.nlm_hidden,
                nlm_depth=config.nlm_depth,
                max_ticks=config.num_ticks,
            )
        else:
            self.nlm = NeuronLevelModels(
                d_model=config.d_model,
                nlm_hidden=config.nlm_hidden,
                nlm_depth=config.nlm_depth,
                max_ticks=config.num_ticks
            )

        # FFN
        self.ffn = OptimizedMLP(config.d_model, config.d_ff)

        # Normalizations
        self.norm_self = OptimizedRMSNorm(config.d_model)
        self.norm_cross = OptimizedRMSNorm(config.d_model)
        self.norm_nlm = OptimizedRMSNorm(config.d_model)
        self.norm_ffn = OptimizedRMSNorm(config.d_model)
        self.norm_post = OptimizedRMSNorm(config.d_model)

        # Residual dropout
        self.resid_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        state: torch.Tensor,           # (B, S, D) current state
        sync: torch.Tensor,            # (B, S, sync_pairs) from global sync module
        layer_history: Optional[torch.Tensor],  # (B, S, T, D) history for this layer's NLM
        static_k: torch.Tensor,        # (B, S, n_kv_head, head_dim)
        static_v: torch.Tensor,        # (B, S, n_kv_head, head_dim)
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with NLM as the PRIMARY driver of neural dynamics.

        Per the CTM paper, NLM IS the thinking mechanism:
            - NLM processes temporal history to produce the next state
            - Observation from data informs but doesn't dominate
            - Self-attention allows token communication

        Flow:
            1. NLM: PRIMARY state from temporal history (THE thinking)
            2. Cross-attention: sync → WHERE, retrieves observation
            3. Synapse: integrates NLM state with observation
            4. Self-attention: tokens communicate (for language modeling)

        Returns:
            state: (B, S, D) updated state
            post_act: (B, S, D) post-activation for history
        """
        # 1. NLM: THE PRIMARY driver of neural dynamics
        # NLM processes this layer's temporal history to produce next state
        if layer_history is not None and layer_history.size(2) > 0:
            # NLM output IS the new state (not a residual!)
            nlm_state = self.nlm(layer_history)
            nlm_state = self.norm_nlm(nlm_state)
        else:
            # Tick 0: no history yet, use incoming state
            nlm_state = self.norm_nlm(state)

        # 2. Cross-attention: get observation from data
        # Sync modulates WHERE to look, attention retrieves WHAT
        obs = self.cross_attn(self.norm_cross(nlm_state), static_k, static_v, cos, sin, sync)

        # 3. Synapse: integrate NLM-driven state with observation
        # NLM state + observation → combined understanding
        state = self.synapse(nlm_state, obs)
        state = self.resid_dropout(state)

        # 4. Self-attention: tokens communicate (needed for language modeling)
        # This is an addition to pure CTM for sequential reasoning
        self_attn_out = self.self_attn(self.norm_self(state), cos, sin)
        state = state + self.resid_dropout(self_attn_out)

        # 5. FFN for expressiveness
        ffn_out = self.ffn(self.norm_ffn(state))
        state = state + self.resid_dropout(ffn_out)

        # Post-activation for history (this feeds into next tick's NLM)
        post_act = self.norm_post(state)

        return state, post_act


# --- CTM Core ---

class CTMCore(nn.Module):
    """
    The iterative core implementing continuous thought with per-layer history.

    Key insight from the paper:
    - z^t evolution = the neural dynamics (thinking)
    - S^t (sync) = the representation of that thinking

    Architecture with per-layer temporal processing:
        - History updated after EACH layer (not just per tick)
        - Each layer has its own NLM for temporal processing
        - Sync computed from full history up to current step
        - Enables "depth in time" at each layer

    History shape: (B, S, num_ticks * n_layer, D)
        - Entry [tick * n_layer + layer_idx] = post-activation of layer at tick
    """

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.num_ticks = config.num_ticks
        self.n_layer = config.n_layer
        self.d_model = config.d_model
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        # Total history length = num_ticks * n_layer (one entry per layer-tick)
        self.total_history_len = config.num_ticks * config.n_layer

        self.layers = nn.ModuleList([
            CTMLayer(config, i) for i in range(config.n_layer)
        ])

        # Global sync module (shared, computes sync from full history)
        # Use enhanced sync for learned projections + multi-scale attention
        if config.use_enhanced_sync:
            self.global_sync = EnhancedSynchronizationModule(
                d_model=config.d_model,
                sync_pairs=config.sync_pairs,
                n_heads=config.n_sync_heads,
                local_window=config.sync_local_window,
                dropout=config.dropout,
                order=config.sync_order,
            )
        else:
            self.global_sync = SynchronizationModule(
                d_model=config.d_model,
                sync_pairs=config.sync_pairs
            )

    def forward(
        self,
        initial_state: torch.Tensor,      # (B, S, D)
        static_k: torch.Tensor,           # (B, S, n_kv_head, head_dim)
        static_v: torch.Tensor,           # (B, S, n_kv_head, head_dim)
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_ticks: Optional[int] = None,  # Override for adaptive compute
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """
        CTM forward pass with sync computed ONLY at tick boundaries.

        CRITICAL: Sync is computed once per tick (after all layers), not mid-layer.
        This maintains clean separation: layers process within a tick using
        the sync from the PREVIOUS tick's end. Sync captures the state of
        thinking at discrete time points, not continuously during processing.

        Returns:
            final_state: (B, S, D)
            all_states: List of (B, S, D) for each complete tick
            final_sync: (B, S, sync_pairs) - sync at end (THE representation per CTM paper)
            all_syncs: List of (B, S, sync_pairs) for each complete tick (for tick selection)
        """
        B, S, D = initial_state.shape
        num_ticks = num_ticks or self.num_ticks
        device = initial_state.device
        dtype = initial_state.dtype

        state = initial_state
        all_states = []  # States at end of each full tick
        all_syncs = []   # Syncs at end of each full tick (for output/tick selection)

        # History as list of tensors (preserves gradients through time)
        # Each entry is (B, S, D) post-activation from a layer-tick step
        # Indexed as: history_list[tick * n_layer + layer_idx]
        history_list: List[torch.Tensor] = []

        # Sync from previous tick (used by all layers within current tick)
        # This is the key fix: sync is computed once per tick, not per layer
        prev_tick_sync = torch.zeros(B, S, self.global_sync.sync_pairs, device=device, dtype=dtype)

        for tick in range(num_ticks):
            # All layers in this tick use the SAME sync (from previous tick boundary)
            # This maintains separation: sync captures state at tick boundaries only
            current_sync = prev_tick_sync

            for layer_idx, layer in enumerate(self.layers):
                # Per-layer history for NLM (strided view of this layer's history across ticks)
                if len(history_list) > 0:
                    global_hist = torch.stack(history_list, dim=2)  # (B, S, T, D)
                    layer_history = global_hist[:, :, layer_idx::self.n_layer]
                else:
                    layer_history = None

                # Layer forward with per-layer NLM
                # Uses current_sync (from previous tick boundary), not recomputed mid-layer
                if self.use_gradient_checkpointing and self.training:
                    state, post_act = gradient_checkpoint(
                        layer.forward,
                        state, current_sync, layer_history, static_k, static_v, cos, sin,
                        use_reentrant=False
                    )
                else:
                    state, post_act = layer.forward(
                        state, current_sync, layer_history, static_k, static_v, cos, sin
                    )

                # Append to history (preserves gradients)
                history_list.append(post_act)

            # Compute sync ONLY at tick boundary (after all layers processed)
            # This is THE representation per the CTM paper
            tick_global_hist = torch.stack(history_list, dim=2)
            tick_sync = self.global_sync(tick_global_hist)

            # Store for next tick's layers to use
            prev_tick_sync = tick_sync

            # Store state and sync at end of each tick
            all_states.append(state.clone())
            all_syncs.append(tick_sync)

        final_sync = all_syncs[-1] if all_syncs else torch.zeros(B, S, self.global_sync.sync_pairs, device=device, dtype=dtype)
        return state, all_states, final_sync, all_syncs


# --- Input Encoder ---

class InputEncoder(nn.Module):
    """
    Shallow transformer encoder that processes input tokens once.
    Produces static key-value pairs for CTM cross-attention.
    """

    def __init__(self, config: CTMConfig, num_encoder_layers: int = 2):
        super().__init__()
        self.d_model = config.d_model
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.d_model // config.n_head

        # Import TransformerBlock from model.py
        from model import TransformerBlock, GPTConfig

        # Create a GPTConfig-compatible object for TransformerBlock
        encoder_config = GPTConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_head=config.n_head,
            n_layer=num_encoder_layers,
            max_seq_len=config.max_seq_len,
            n_kv_head=config.n_kv_head,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
            d_ff=config.d_ff,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(encoder_config) for _ in range(num_encoder_layers)
        ])

        # KV projection
        self.k_proj = nn.Linear(config.d_model, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_head * self.head_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            encoded: (B, S, D) encoded representation
            static_k: (B, S, n_kv_head, head_dim) keys with RoPE
            static_v: (B, S, n_kv_head, head_dim) values
        """
        B, S, D = x.shape

        # Run through encoder layers
        for layer in self.layers:
            x = layer(x, cos, sin)

        # Project to KV
        k = self.k_proj(x).view(B, S, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_head, self.head_dim)

        # Apply RoPE to keys
        target_dtype = k.dtype
        if target_dtype not in (torch.float16, torch.bfloat16):
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if k.dtype != target_dtype:
            k = k.to(dtype=target_dtype)

        rotary_dim = cos.shape[-1] * 2
        if rotary_dim > self.head_dim:
            trim = self.head_dim // 2
            cos_k = cos[..., :trim]
            sin_k = sin[..., :trim]
        else:
            cos_k, sin_k = cos, sin

        if cos_k.dtype != target_dtype:
            cos_k = cos_k.to(dtype=target_dtype)
            sin_k = sin_k.to(dtype=target_dtype)

        k = apply_rotary_emb(k, cos_k, sin_k, interleaved=False)

        return x, k, v.to(k.dtype)


# --- Main CTM Language Model ---

class CTMLanguageModel(nn.Module):
    """
    Complete CTM for language modeling.

    Architecture:
        1. Token embedding
        2. Input encoder (produces static KV)
        3. CTM iterative core (multiple ticks)
        4. Output projection from SYNC (per CTM paper)

    Per the CTM paper, sync (neural correlation patterns) IS the representation.
    Output is computed from sync, not from the hidden state directly.

    Returns logits for each tick to enable tick selection during training.
    """

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Input encoder
        self.input_encoder = InputEncoder(config, num_encoder_layers=2)

        # CTM core
        self.ctm_core = CTMCore(config)

        # Output head: sync → logits (per CTM paper, sync IS the representation)
        # No weight tying since sync_pairs != d_model
        self.sync_norm = OptimizedRMSNorm(config.sync_pairs)
        self.sync_head = nn.Linear(config.sync_pairs, config.vocab_size, bias=False)

        # RoPE cache
        self.head_dim = config.d_model // config.n_head
        self._init_rope()

        # Initialize weights
        self._init_weights()

    def _init_rope(self):
        """Precompute RoPE cos/sin cache."""
        theta = self.config.rope_theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(self.config.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos().to(torch.bfloat16), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().to(torch.bfloat16), persistent=False)

    def _init_weights(self):
        """CTM-specific weight initialization."""
        init_std = 0.02
        # Scale for both layers and ticks
        residual_std = init_std / math.sqrt(2 * self.config.n_layer * self.config.num_ticks)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "o_proj" in name or "down_proj" in name or "synapse" in name:
                    nn.init.normal_(module.weight, mean=0.0, std=residual_std)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        return_all_ticks: bool = False,
        num_ticks: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor, None], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            input_ids: (B, S) token IDs
            input_pos: (S,) position indices for RoPE
            return_all_ticks: If True, return logits for all ticks
            num_ticks: Override number of ticks (for adaptive compute)

        Returns:
            If return_all_ticks=False: (logits, None) where logits is (B, S, V)
            If return_all_ticks=True: (final_logits, all_logits) where all_logits is List[(B, S, V)]
        """
        B, S = input_ids.shape

        # Get RoPE
        if input_pos is None:
            input_pos = torch.arange(S, device=input_ids.device)
        cos = self.cos_cached[input_pos]
        sin = self.sin_cached[input_pos]

        # 1. Embed tokens
        x = self.token_embedding(input_ids)

        # 2. Encode input (produces static KV)
        encoded, static_k, static_v = self.input_encoder(x, cos, sin)

        # 3. Initialize CTM state from encoded input
        initial_state = encoded

        # 4. Run CTM iterations
        final_state, all_states, final_sync, all_syncs = self.ctm_core(
            initial_state, static_k, static_v, cos, sin, num_ticks=num_ticks
        )

        # 5. Compute logits from SYNC (per CTM paper, sync IS the representation)
        if return_all_ticks:
            all_logits = []
            for sync in all_syncs:
                logits = self.sync_head(self.sync_norm(sync))
                all_logits.append(logits)
            final_logits = self.sync_head(self.sync_norm(final_sync))
            return final_logits, all_logits
        else:
            logits = self.sync_head(self.sync_norm(final_sync))
            return logits, None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        num_ticks: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Simple generation without KV caching (recomputes each step).
        For production, implement proper caching.
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate if exceeds max length
            context = generated[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(context, num_ticks=num_ticks)
            next_token_logits = logits[:, -1, :]

            # Sample
            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                if top_k is not None:
                    v, _ = torch.topk(probs, top_k)
                    probs[probs < v[:, [-1]]] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    @torch.no_grad()
    def get_nlm_diagnostics(
        self,
        input_ids: torch.Tensor,
        num_ticks: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run a forward pass and collect NLM diagnostics for visualization.

        Returns diagnostics:
            - 'tick_activations': (num_ticks, D) - mean neuron activation at each tick boundary
            - 'tick_sync': (num_ticks, sync_pairs) - sync values at each tick boundary
            - 'gate_mean': (n_layer, D) - mean gate value per neuron (last tick)
            - 'attn_entropy': (n_layer, D) - temporal attention entropy per neuron (last tick)

        The tick_activations show how each neuron's output evolves across ticks,
        which is what the sync module measures correlations over.
        """
        self.eval()
        B, S = input_ids.shape
        device = input_ids.device
        num_ticks = num_ticks or self.config.num_ticks

        # Get RoPE
        input_pos = torch.arange(S, device=device)
        cos = self.cos_cached[input_pos]
        sin = self.sin_cached[input_pos]

        # Embed and encode
        x = self.token_embedding(input_ids)
        encoded, static_k, static_v = self.input_encoder(x, cos, sin)

        # Run CTM core manually to collect diagnostics
        state = encoded
        history_list: List[torch.Tensor] = []
        prev_tick_sync = torch.zeros(
            B, S, self.config.sync_pairs, device=device, dtype=encoded.dtype
        )

        # Collect tick-level dynamics (the key insight: sync operates at tick boundaries)
        tick_activations = []  # State at each tick boundary
        tick_sync_values = []  # Sync at each tick boundary
        tick_layer_post_acts = []  # Post-act from each layer at each tick

        # Also collect NLM internals at last tick
        last_tick_attn_entropy = []
        last_tick_gate_mean = []

        n_layer = self.config.n_layer

        for tick in range(num_ticks):
            current_sync = prev_tick_sync
            tick_post_acts = []  # Post-act for each layer in this tick

            for layer_idx, layer in enumerate(self.ctm_core.layers):
                # Get layer history
                if len(history_list) > 0:
                    global_hist = torch.stack(history_list, dim=2)
                    layer_history = global_hist[:, :, layer_idx::n_layer]
                else:
                    layer_history = None

                # Collect NLM diagnostics on last tick
                if layer_history is not None and layer_history.size(2) > 0:
                    if tick == num_ticks - 1:
                        _, nlm_diag = layer.nlm(layer_history, return_diagnostics=True)
                        last_tick_attn_entropy.append(nlm_diag['attn_entropy'])
                        last_tick_gate_mean.append(nlm_diag['gate_mean'])

                # Run full layer forward for state update
                state, post_act = layer.forward(
                    state, current_sync, layer_history, static_k, static_v, cos, sin
                )
                history_list.append(post_act)

                # Capture post_act for this layer (averaged over batch and sequence)
                post_act_mean = post_act.mean(dim=(0, 1))  # (D,)
                tick_post_acts.append(post_act_mean)

            # Stack post_acts for all layers in this tick
            tick_layer_post_acts.append(torch.stack(tick_post_acts, dim=0))  # (n_layer, D)

            # === TICK BOUNDARY: This is where sync is computed ===
            tick_global_hist = torch.stack(history_list, dim=2)
            prev_tick_sync = self.ctm_core.global_sync(tick_global_hist)

            # Capture state at tick boundary (averaged over batch and sequence)
            # state: (B, S, D) -> (D,) mean activation per neuron
            tick_state_mean = state.mean(dim=(0, 1))  # (D,)
            tick_activations.append(tick_state_mean)

            # Capture sync at tick boundary
            tick_sync_mean = prev_tick_sync.mean(dim=(0, 1))  # (sync_pairs,)
            tick_sync_values.append(tick_sync_mean)

        # Stack tick-level data
        tick_activations = torch.stack(tick_activations, dim=0)  # (num_ticks, D)
        tick_sync_values = torch.stack(tick_sync_values, dim=0)  # (num_ticks, sync_pairs)
        layer_post_acts = torch.stack(tick_layer_post_acts, dim=0)  # (num_ticks, n_layer, D)

        result = {
            'tick_activations': tick_activations,  # (num_ticks, D)
            'tick_sync': tick_sync_values,         # (num_ticks, sync_pairs)
            'layer_post_acts': layer_post_acts,    # (num_ticks, n_layer, D)
            'num_ticks': torch.tensor(num_ticks),
            'n_layer': torch.tensor(n_layer),
        }

        # Add NLM internals if available
        if last_tick_attn_entropy:
            result['attn_entropy'] = torch.stack(last_tick_attn_entropy, dim=0)  # (n_layer, D)
            result['gate_mean'] = torch.stack(last_tick_gate_mean, dim=0)        # (n_layer, D)

        return result
