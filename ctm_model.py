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
        # Xavier-like initialization scaled for the architecture
        std = 0.02
        nn.init.normal_(self.w_in, mean=0.0, std=std)
        nn.init.normal_(self.w_out, mean=0.0, std=std)
        for w in self.w_hidden:
            nn.init.normal_(w, mean=0.0, std=std)

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
        obs = self.cross_attn(nlm_state, static_k, static_v, cos, sin, sync)

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

        # History as list of tensors (preserves gradients for temporal credit assignment)
        # Each entry is (B, S, D) post-activation from a layer-tick step
        # Indexed as: history_list[tick * n_layer + layer_idx]
        history_list: List[torch.Tensor] = []

        for tick in range(num_ticks):
            for layer_idx, layer in enumerate(self.layers):
                # Build GLOBAL history by stacking (for sync computation)
                if len(history_list) > 0:
                    # Stack maintains gradient flow through time
                    global_hist = torch.stack(history_list, dim=2)  # (B, S, T, D)
                    sync = self.global_sync(global_hist)
                else:
                    sync = torch.zeros(B, S, self.global_sync.sync_pairs, device=device, dtype=dtype)

                # Build PER-LAYER history for this layer's NLM
                # Layer L's entries are at indices: [L, n_layer+L, 2*n_layer+L, ...]
                if tick > 0:
                    layer_entries = [history_list[t * self.n_layer + layer_idx] for t in range(tick)]
                    layer_history = torch.stack(layer_entries, dim=2)  # (B, S, tick, D)
                else:
                    layer_history = None

                # Layer forward with per-layer NLM
                if self.use_gradient_checkpointing and self.training:
                    state, post_act = gradient_checkpoint(
                        layer.forward,
                        state, sync, layer_history, static_k, static_v, cos, sin,
                        use_reentrant=False
                    )
                else:
                    state, post_act = layer.forward(
                        state, sync, layer_history, static_k, static_v, cos, sin
                    )

                # Append to history (preserves gradients)
                history_list.append(post_act)

            # Compute FINAL sync for this tick (after all layers processed)
            # This is THE representation per the CTM paper
            tick_global_hist = torch.stack(history_list, dim=2)
            tick_sync = self.global_sync(tick_global_hist)

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
