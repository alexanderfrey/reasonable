"""
Global Sync Module - Cross-module synchronization for PEM.

Follows the CTM paper's "Synchronization as a representation" blueprint:

    S_t = Z_t · (Z_t)^T ∈ R^{D×D}

Each module computes its synchronization matrix from post-activation history,
then cross-module sync is computed by comparing these sync patterns.

Extended with Oscillatory World Model:
- Instead of GRU-based persistent state, uses oscillating neurons
- Oscillators at different frequencies capture different timescales
- Learned modulation of oscillators gets gradients (solves gradient flow problem)
- Phase advance happens every forward pass

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         GLOBAL SYNC MODULE                              │
    │                                                                         │
    │   Module Z_histories (post-activation histories):                       │
    │   [Z_pred, Z_surp, ...]  where Z = [z_1, z_2, ..., z_T]                │
    │         │       │                                                       │
    │         ▼       ▼                                                       │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │  Sync Computation: S = Z · Z^T  (per module)                    │  │
    │   │  Then subsample (i,j) pairs to get S_sync                       │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                          │
    │                              ▼                                          │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │  Cross-Module Sync: compare sync patterns between modules       │  │
    │   │  Which modules have similar synchronization dynamics?           │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                          │
    │                              ▼                                          │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │  Oscillatory World Model (replaces GRU-based state)             │  │
    │   │  - Phase advance: deterministic evolution                        │  │
    │   │  - Modulation: learned (gets gradients!)                        │  │
    │   │  - Read: oscillator amplitudes * sin(phases)                    │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                          │
    │                              ▼                                          │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │  Global Sync State                                              │  │
    │   │  (B, S, sync_pairs) - drives attention to features              │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────┘
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .oscillatory_world import (
    OscillatoryWorldState,
    OscillatoryWorldConfig,
    OscillatorMetrics,
    OscillatorOutput,
    create_oscillatory_world,
)


class GlobalSyncOutput(NamedTuple):
    """Output from GlobalSyncModule."""
    sync: torch.Tensor                 # (B, S, sync_pairs) global sync state
    cross_module_sync: torch.Tensor    # (num_modules, num_modules, B, S) sync matrix
    module_contributions: torch.Tensor # (B, S, num_modules) how much each module contributes
    world_state: Optional[torch.Tensor] = None  # (d_world_output,) current oscillatory world state
    # Oscillator-specific monitoring (replaces GRU gate values)
    oscillator_metrics: Optional[OscillatorMetrics] = None  # Detailed oscillator metrics
    oscillator_amplitudes: Optional[torch.Tensor] = None    # (num_oscillators,) current amplitudes
    oscillator_phases: Optional[torch.Tensor] = None        # (num_oscillators,) current phases
    # Auxiliary prediction for future features
    future_prediction: Optional[torch.Tensor] = None  # (d_feature_input,) predicted future features
    # Oscillator cross-attention entropy (measures selection diversity)
    # High entropy = using many oscillators, low = always same ones
    osc_attn_entropy: Optional[float] = None


@dataclass
class GlobalSyncConfig:
    """Configuration for Global Sync Module."""

    # Sync computation
    d_sync_space: int = 128        # Subsampled sync pairs per module

    # Cross-module attention
    n_heads: int = 4
    dropout: float = 0.0
    attention_temperature: float = 1.0  # Higher = softer attention (1.0 = standard)

    # Cross-residual: inject signal from OTHER modules even when attention is weak
    # 0.0 = no cross-residual (standard), 0.1-0.3 = moderate cross-module signal
    cross_residual_strength: float = 0.0

    # Output
    sync_pairs: int = 256          # Output sync dimension

    # Learnable module embeddings
    use_module_embeddings: bool = True

    # Oscillatory world model (content-based memory with surprise gating)
    use_oscillatory_world: bool = True  # Enable/disable oscillatory world model
    num_oscillators: int = 64           # Number of oscillators (memory slots)
    min_period: int = 8                 # Fastest oscillator period
    max_period: int = 4096              # Slowest oscillator period
    d_world_output: int = 256           # Output dimension of world state
    d_feature_input: int = 256          # Dimension of content features for writing
    surprise_gate_bias: float = 0.5     # Base write strength for surprise gating
    surprise_gate_scale: float = 1.0    # How much surprise amplifies writing

    # Sync compression method (now used for QUERYING memory, not writing)
    # "mean": Simple mean
    # "attention": Learned attention-weighted pooling
    # "last": Use last sequence position only (causal)
    sync_compression: str = "attention"

    # Oscillator cross-attention (positions attend to oscillator memory)
    # When True, world_state output is (B, S, d_world_output) - position-specific
    # When False, world_state output is (d_world_output,) - broadcast to all positions
    use_oscillator_cross_attention: bool = True
    osc_cross_attn_heads: int = 4
    d_osc_embed: int = 64  # Per-oscillator embedding dimension for cross-attention

    # Auxiliary prediction loss (train oscillators to predict future)
    # When enabled, oscillator memory states are used to predict future features
    use_auxiliary_prediction: bool = True
    auxiliary_prediction_horizon: int = 8  # How many steps ahead to predict


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class ModuleSyncComputer(nn.Module):
    """
    Computes synchronization from a module's post-activation history.

    Following the paper:
        Z_t = [z_1, z_2, ..., z_t] ∈ R^{D×t}
        S_t = Z_t · (Z_t)^T ∈ R^{D×D}
        S_sync = subsample(S_t, n_pairs)
    """

    def __init__(
        self,
        d_neurons: int,
        d_sync_space: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_neurons = d_neurons
        self.d_sync_space = d_sync_space

        # Register fixed random (i,j) pairs for subsampling sync matrix
        # Sample from upper triangle of D×D matrix
        sync_indices = self._sample_pairs(d_neurons, d_sync_space)
        self.register_buffer('sync_i', sync_indices[:, 0])
        self.register_buffer('sync_j', sync_indices[:, 1])

        # Project subsampled sync to common space
        self.sync_proj = nn.Sequential(
            nn.Linear(d_sync_space, d_sync_space),
            nn.GELU(),
            nn.Linear(d_sync_space, d_sync_space),
        )
        self.norm = RMSNorm(d_sync_space)

        self._init_weights()

    def _sample_pairs(self, D: int, n_pairs: int) -> torch.Tensor:
        """Sample n_pairs (i,j) index pairs from upper triangle of DxD matrix."""
        pairs = []
        for i in range(D):
            for j in range(i, D):
                pairs.append((i, j))

        # Randomly sample pairs
        indices = torch.randperm(len(pairs))[:n_pairs]
        sampled = [pairs[idx] for idx in indices]

        return torch.tensor(sampled, dtype=torch.long)

    def _init_weights(self):
        for m in self.sync_proj.modules():
            if isinstance(m, nn.Linear):
                # Xavier init for proper gradient flow
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        Z_history: List[torch.Tensor],  # List of (B, S, D) post-activations per tick
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute synchronization from post-activation history.

        Args:
            Z_history: List of post-activations [z_1, z_2, ..., z_T]
                      Each z_t has shape (B, S, D)

        Returns:
            S_sync: (B, S, d_sync_space) subsampled sync representation
            S_full: (B, S, D, D) full sync matrix (for analysis)
        """
        # Stack: (B, S, D, T)
        Z_t = torch.stack(Z_history, dim=-1)
        B, S, D, T = Z_t.shape

        # Compute sync matrix: S = Z · Z^T / sqrt(T)
        # (B, S, D, T) @ (B, S, T, D) -> (B, S, D, D)
        S_full = torch.matmul(Z_t, Z_t.transpose(-1, -2))
        S_full = S_full / math.sqrt(T)

        # Subsample (i,j) pairs
        S_sync = S_full[:, :, self.sync_i, self.sync_j]  # (B, S, d_sync_space)

        # Project to common space
        S_sync = self.sync_proj(S_sync)
        S_sync = self.norm(S_sync)

        return S_sync, S_full


class SyncCrossModuleAttention(nn.Module):
    """
    Cross-module attention using sync-derived queries.

    Following the paper:
        q_t = W_in · S_action

    Each module's sync representation generates queries that attend to
    other modules' sync representations.

    Cross-residual mechanism:
        When cross_residual_strength > 0, each module receives a residual
        signal from OTHER modules, ensuring cross-module information flow
        even when learned attention is weak/collapsed.
    """

    def __init__(
        self,
        d_sync_space: int,
        num_modules: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        temperature: float = 1.0,
        cross_residual_strength: float = 0.0,
    ):
        super().__init__()
        self.d_sync_space = d_sync_space
        self.num_modules = num_modules
        self.n_heads = n_heads
        self.head_dim = d_sync_space // n_heads
        self.temperature = temperature
        self.cross_residual_strength = cross_residual_strength

        assert d_sync_space % n_heads == 0

        # Sync -> Query projection (paper: W_in)
        self.sync_to_query = nn.Sequential(
            nn.Linear(d_sync_space, d_sync_space),
            nn.GELU(),
            nn.Linear(d_sync_space, d_sync_space),
        )

        # K, V projections for cross-module attention
        self.k_proj = nn.Linear(d_sync_space, d_sync_space, bias=False)
        self.v_proj = nn.Linear(d_sync_space, d_sync_space, bias=False)
        self.o_proj = nn.Linear(d_sync_space, d_sync_space, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = RMSNorm(d_sync_space)

        self._init_weights()

    def _init_weights(self):
        for m in self.sync_to_query.modules():
            if isinstance(m, nn.Linear):
                # Xavier init for proper gradient flow
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for proj in [self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def forward(
        self,
        module_syncs: torch.Tensor,  # (num_modules, B, S, d_sync_space)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-module attention using sync-derived queries.

        Args:
            module_syncs: Stacked sync representations from all modules

        Returns:
            attended: (num_modules, B, S, d_sync_space) updated sync states
            cross_module_sync: (num_modules, num_modules, B, S) attention weights
        """
        num_modules, B, S, D = module_syncs.shape

        # Reshape: (num_modules, B, S, D) -> (B*S, num_modules, D)
        syncs_flat = module_syncs.permute(1, 2, 0, 3).reshape(B * S, num_modules, D)

        # Generate queries from sync (paper: q_t = W_in · S_action)
        q = self.sync_to_query(syncs_flat)  # (B*S, num_modules, D)

        # K, V from sync representations
        k = self.k_proj(syncs_flat)
        v = self.v_proj(syncs_flat)

        # Reshape for multi-head attention
        q = q.view(B * S, num_modules, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * S, num_modules, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * S, num_modules, self.n_heads, self.head_dim).transpose(1, 2)
        # Now: (B*S, n_heads, num_modules, head_dim)

        # Attention with temperature scaling
        # Higher temperature = softer attention (more uniform)
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores / self.temperature  # Temperature scaling
        attn_weights = F.softmax(scores, dim=-1)  # (B*S, n_heads, num_modules, num_modules)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attended = torch.matmul(attn_weights, v)  # (B*S, n_heads, num_modules, head_dim)
        attended = attended.transpose(1, 2).reshape(B * S, num_modules, D)

        # Output projection
        attended = self.o_proj(attended)
        attended = attended.view(B, S, num_modules, D).permute(2, 0, 1, 3)

        # Residual connection with optional cross-residual
        # Standard residual: add back own module's sync
        # Cross-residual: also add signal from OTHER modules
        if self.cross_residual_strength > 0 and num_modules > 1:
            # Compute mean of all other modules for each module
            # For module i: cross_signal_i = mean(module_sync_j for j != i)
            # Efficient: (sum_all - self) / (n-1) = (n*mean - self) / (n-1)
            all_mean = module_syncs.mean(dim=0, keepdim=True)  # (1, B, S, D)
            # For each module, compute mean of others
            # cross_signal_i = (n * all_mean - module_syncs_i) / (n - 1)
            cross_signal = (num_modules * all_mean - module_syncs) / (num_modules - 1)

            # Blend: (1 - alpha) * self_residual + alpha * cross_residual
            alpha = self.cross_residual_strength
            residual = (1 - alpha) * module_syncs + alpha * cross_signal
            attended = self.norm(attended + residual)
        else:
            # Standard self-residual only
            attended = self.norm(attended + module_syncs)

        # Get cross-module sync matrix (average over heads)
        cross_module_sync = attn_weights.mean(dim=1)  # (B*S, num_modules, num_modules)
        cross_module_sync = cross_module_sync.view(B, S, num_modules, num_modules)
        cross_module_sync = cross_module_sync.permute(2, 3, 0, 1)  # (num_modules, num_modules, B, S)

        return attended, cross_module_sync


class SyncIntegrator(nn.Module):
    """
    Integrate cross-module sync states into global sync output.
    """

    def __init__(
        self,
        d_sync_space: int,
        num_modules: int,
        sync_pairs: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Input: concatenated module sync states
        d_input = d_sync_space * num_modules

        self.integrator = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_input, sync_pairs),
        )

        # Module contribution weights (learned)
        self.module_weights = nn.Linear(d_sync_space, 1)

        self.norm = RMSNorm(sync_pairs)

        self._init_weights()

    def _init_weights(self):
        for m in self.integrator.modules():
            if isinstance(m, nn.Linear):
                # Xavier init for proper gradient flow
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.module_weights.weight)

    def forward(
        self,
        attended_syncs: torch.Tensor,  # (num_modules, B, S, d_sync_space)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate module sync states into global sync.

        Returns:
            sync: (B, S, sync_pairs) global sync state
            contributions: (B, S, num_modules) module contribution weights
        """
        num_modules, B, S, D = attended_syncs.shape

        # Compute module contributions
        contributions = self.module_weights(attended_syncs)  # (num_modules, B, S, 1)
        contributions = F.softmax(contributions.squeeze(-1).permute(1, 2, 0), dim=-1)  # (B, S, num_modules)

        # Concatenate all module sync states
        # (num_modules, B, S, D) -> (B, S, num_modules * D)
        concat = attended_syncs.permute(1, 2, 0, 3).reshape(B, S, -1)

        # Integrate
        sync = self.integrator(concat)  # (B, S, sync_pairs)
        sync = self.norm(sync)

        return sync, contributions




class GlobalSyncModule(nn.Module):
    """
    Global Sync Module - combines post-activation histories from all CTM modules.

    Following the paper's "Synchronization as a representation" blueprint:
    1. Each module provides its Z_history (post-activations at each tick)
    2. Compute S = Z · Z^T for each module (true neural synchronization)
    3. Subsample sync pairs from each module
    4. Use sync-derived queries for cross-module attention
    5. Integrate into global sync state

    Extended with Oscillatory World Model:
    - Uses oscillating neurons at different frequencies instead of GRU-based state
    - Modulation of oscillators is learned (gets gradients!)
    - Phase advance happens automatically each forward pass
    - No need for commit_world_state() - oscillators evolve continuously

    Usage:
        global_sync = GlobalSyncModule(config)

        # Register module sync computers (once at init)
        global_sync.register_module('prediction', d_neurons=256)
        global_sync.register_module('surprise', d_neurons=128)

        # Forward pass
        output = global_sync({
            'prediction': pred_ctm.all_tick_activations,  # List[(B, S, 256)]
            'surprise': surp_ctm.all_tick_activations,    # List[(B, S, 128)]
        })

        # Use global sync for attention
        sync = output.sync  # (B, S, sync_pairs)
        world_state = output.world_state  # (d_world_output,) current oscillatory state
    """

    def __init__(self, config: GlobalSyncConfig):
        super().__init__()
        self.config = config

        # Module sync computers (added dynamically via register_module)
        self.module_sync_computers = nn.ModuleDict()
        self.module_names: List[str] = []

        # Cross-module attention (created after modules registered)
        self._cross_module_attn = None
        self._sync_integrator = None

        # Module embeddings (learnable identity for each module)
        if config.use_module_embeddings:
            self.module_embeddings = nn.ParameterDict()
        else:
            self.module_embeddings = None

        # Oscillatory world model (content-based memory with surprise gating)
        if config.use_oscillatory_world:
            osc_config = OscillatoryWorldConfig(
                num_oscillators=config.num_oscillators,
                min_period=config.min_period,
                max_period=config.max_period,
                d_feature_input=config.d_feature_input,  # Content features for writing
                d_sync_input=config.sync_pairs,  # Legacy, kept for compatibility
                d_output=config.d_world_output,
                surprise_gate_bias=config.surprise_gate_bias,
                surprise_gate_scale=config.surprise_gate_scale,
            )
            self.oscillatory_world = OscillatoryWorldState(osc_config)

            # Feature compression for writing to memory
            # Compress features (B, S, d_model) -> (d_feature_input,)
            self.feature_compressor = nn.Sequential(
                nn.Linear(config.d_feature_input, config.d_feature_input),
                nn.GELU(),
            )
            # Learned attention pooling over sequence positions for memory write
            hidden_dim = max(1, config.d_feature_input // 2)
            self.feature_write_attn = nn.Sequential(
                nn.Linear(config.d_feature_input, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            # Initialize with larger scale (0.5) to break symmetry and allow
            # attention to differentiate positions. Small scale (0.1) leads to
            # near-uniform attention over 512 positions with weak learning signal.
            with torch.no_grad():
                self.feature_write_attn[-1].weight.mul_(0.5)
                self.feature_write_attn[-1].bias.zero_()

            # Sync to feature projection (fallback when features not provided)
            # (sync_pairs,) -> (d_feature_input,)
            self.sync_to_feature = nn.Linear(config.sync_pairs, config.d_feature_input)

            # Attention network for sync compression (learns which positions matter)
            if config.sync_compression == "attention":
                self.sync_compression_attn = nn.Sequential(
                    nn.Linear(config.sync_pairs, config.sync_pairs // 2),
                    nn.GELU(),
                    nn.Linear(config.sync_pairs // 2, 1),  # Score per position
                )
                # Initialize to produce near-uniform weights initially
                with torch.no_grad():
                    self.sync_compression_attn[-1].weight.mul_(0.1)
                    self.sync_compression_attn[-1].bias.zero_()
            else:
                self.sync_compression_attn = None

            # Cross-attention: positions attend to oscillator memory
            # Query: sync (B, S, sync_pairs) -> each position queries based on its sync pattern
            # Key/Value: 64 oscillators, each as a separate key-value pair
            # Output: (B, S, d_world_output) -> position-specific world context
            if config.use_oscillator_cross_attention:
                if config.d_world_output % config.osc_cross_attn_heads != 0:
                    raise ValueError(
                        "d_world_output must be divisible by osc_cross_attn_heads "
                        f"(got {config.d_world_output} and {config.osc_cross_attn_heads})"
                    )
                # Learnable embedding per oscillator (like positional embeddings)
                # Each oscillator gets a learned representation that encodes its "role"
                # Use larger scale (0.5) to ensure oscillators are distinguishable after K/V projection
                # Small scale (0.02) leads to near-uniform attention and vanishing key gradients
                self.osc_embeddings = nn.Parameter(
                    torch.randn(config.num_oscillators, config.d_osc_embed) * 0.5
                )

                # Query projection: sync -> d_world_output
                self.osc_query_proj = nn.Linear(config.sync_pairs, config.d_world_output)

                # Key/Value projections: per-oscillator embedding -> d_world_output
                # Input: (num_oscillators, d_osc_embed), Output: (num_oscillators, d_world_output)
                self.osc_key_proj = nn.Linear(config.d_osc_embed, config.d_world_output)
                self.osc_value_proj = nn.Linear(config.d_osc_embed, config.d_world_output)

                # Use scaled dot-product attention directly over pre-projected Q/K/V
                # to avoid redundant key/value projections inside MultiheadAttention.
                self.osc_cross_attn = None
                self.osc_output_norm = RMSNorm(config.d_world_output)
            else:
                self.osc_embeddings = None
                self.osc_query_proj = None
                self.osc_key_proj = None
                self.osc_value_proj = None
                self.osc_cross_attn = None
                self.osc_output_norm = None
            self._last_osc_attn_out_norm = None
            self._last_osc_query_act_norm = None
            self._last_osc_attn_query_ratio = None
            self._last_osc_attn_top1 = None
            self._last_osc_key_query_cosine = None
            self._last_feature_write_attn_entropy = None
            self._last_feature_write_top1 = None
            self._last_feature_write_top5 = None
            self._last_feature_write_surprise_corr = None
            self._last_write_gate = None

            # Auxiliary prediction head (predict future features from memory)
            if config.use_auxiliary_prediction:
                # Predict future features from oscillator memory states
                # Input: memory_states (num_oscillators,)
                # Output: predicted future features (d_feature_input,)
                self.future_predictor = nn.Sequential(
                    nn.Linear(config.num_oscillators, config.d_feature_input),
                    nn.GELU(),
                    nn.Linear(config.d_feature_input, config.d_feature_input),
                )
                self.auxiliary_prediction_horizon = config.auxiliary_prediction_horizon
            else:
                self.future_predictor = None
                self.auxiliary_prediction_horizon = 0
        else:
            self.oscillatory_world = None
            self.feature_compressor = None
            self.feature_write_attn = None
            self.sync_to_feature = None
            self.sync_compression_attn = None
            self.osc_query_proj = None
            self.osc_key_proj = None
            self.osc_value_proj = None
            self.osc_cross_attn = None
            self.osc_output_norm = None
            self.future_predictor = None
            self.auxiliary_prediction_horizon = 0
            self._last_osc_attn_out_norm = None
            self._last_osc_query_act_norm = None
            self._last_osc_attn_query_ratio = None
            self._last_osc_attn_top1 = None
            self._last_osc_key_query_cosine = None
            self._last_feature_write_attn_entropy = None
            self._last_feature_write_top1 = None
            self._last_feature_write_top5 = None
            self._last_feature_write_surprise_corr = None
            self._last_write_gate = None

    def register_module(self, name: str, d_neurons: int) -> None:
        """
        Register a CTM module for global sync.

        Args:
            name: Module name (e.g., 'prediction', 'surprise')
            d_neurons: Number of neurons in that module's post-activations
        """
        self.module_sync_computers[name] = ModuleSyncComputer(
            d_neurons=d_neurons,
            d_sync_space=self.config.d_sync_space,
            dropout=self.config.dropout,
        )
        self.module_names.append(name)

        if self.module_embeddings is not None:
            # Use larger scale (0.5 instead of 0.02) for meaningful differentiation
            # Small embeddings lead to uniform cross-attention (0.5/0.5 syndrome)
            # Also use orthogonal-like initialization per module
            module_idx = len(self.module_names) - 1  # 0-indexed after append
            # Create semi-orthogonal embeddings: alternating sign pattern + noise
            base = torch.randn(self.config.d_sync_space)
            base = base / base.norm()  # Normalize to unit length
            # Offset by module index to ensure differentiation
            offset = torch.zeros(self.config.d_sync_space)
            offset[module_idx::2] = 1.0  # Alternate dimensions per module
            embedding = (base + offset * 0.5) * 0.5  # Scale to reasonable magnitude
            self.module_embeddings[name] = nn.Parameter(embedding)

        # Recreate cross-module attention with updated num_modules
        self._create_attention_layers()

    def _osc_cross_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled dot-product cross-attention with explicit Q/K/V projections.

        Args:
            query: (B, S, d_world)
            key: (B, N, d_world)
            value: (B, N, d_world)
        Returns:
            attn_out: (B, S, d_world)
            attn_weights: (B, S, N) averaged over heads (for entropy metrics)
        """
        B, S, d_world = query.shape
        n_osc = key.shape[1]
        n_heads = self.config.osc_cross_attn_heads
        head_dim = d_world // n_heads

        # (B, S, d_world) -> (B, H, S, head_dim)
        q = query.view(B, S, n_heads, head_dim).transpose(1, 2)
        k = key.view(B, n_osc, n_heads, head_dim).transpose(1, 2)
        v = value.view(B, n_osc, n_heads, head_dim).transpose(1, 2)

        scale = head_dim ** -0.5
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_logits, dim=-1)

        if self.config.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.config.dropout)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, d_world)

        # Average over heads for entropy monitoring
        attn_weights_avg = attn_weights.mean(dim=1)
        return attn_out, attn_weights_avg

    def _create_attention_layers(self):
        """Create/recreate attention layers based on registered modules."""
        num_modules = len(self.module_names)
        if num_modules < 1:
            return

        self._cross_module_attn = SyncCrossModuleAttention(
            d_sync_space=self.config.d_sync_space,
            num_modules=num_modules,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout,
            temperature=self.config.attention_temperature,
            cross_residual_strength=self.config.cross_residual_strength,
        )

        self._sync_integrator = SyncIntegrator(
            d_sync_space=self.config.d_sync_space,
            num_modules=num_modules,
            sync_pairs=self.config.sync_pairs,
            dropout=self.config.dropout,
        )

    def forward(
        self,
        module_activations: Dict[str, List[torch.Tensor]],  # {name: List[(B, S, d_neurons)]}
        features: Optional[torch.Tensor] = None,  # (B, S, d_feature_input) content for memory
        surprise: Optional[torch.Tensor] = None,  # (B, S, 1) or scalar, importance gate
    ) -> GlobalSyncOutput:
        """
        Compute global sync from module post-activation histories.

        New architecture: Content-based memory with surprise gating.
        - Features (content) write to oscillator memory
        - Surprise gates write strength (unexpected = important)
        - Sync patterns query memory via cross-attention

        Args:
            module_activations: Dict mapping module name to Z_history
                               (list of post-activations at each tick)
            features: Optional content features for writing to memory
            surprise: Optional surprise magnitude for gating writes

        Returns:
            GlobalSyncOutput with sync state and cross-module analysis
        """
        if len(self.module_names) == 0:
            raise RuntimeError("No modules registered. Call register_module() first.")

        # Verify all modules provided
        for name in self.module_names:
            if name not in module_activations:
                raise ValueError(f"Missing activations for module: {name}")

        # 1. Compute sync representation for each module (S = Z · Z^T, subsampled)
        module_syncs = []
        for name in self.module_names:
            Z_history = module_activations[name]
            sync_computer = self.module_sync_computers[name]

            S_sync, _ = sync_computer(Z_history)  # (B, S, d_sync_space)

            # Add module embedding if enabled
            if self.module_embeddings is not None:
                emb = self.module_embeddings[name]  # (d_sync_space,)
                S_sync = S_sync + emb.unsqueeze(0).unsqueeze(0)

            module_syncs.append(S_sync)

        # 2. Stack: (num_modules, B, S, d_sync_space)
        stacked = torch.stack(module_syncs, dim=0)

        # 3. Cross-module attention using sync-derived queries
        attended, cross_module_sync = self._cross_module_attn(stacked)

        # 4. Integrate into global sync
        sync, contributions = self._sync_integrator(attended)

        # 5. Oscillatory world model (replaces GRU-based persistent state)
        world_state = None
        oscillator_metrics = None
        oscillator_amplitudes = None
        oscillator_phases = None
        future_prediction = None

        if self.oscillatory_world is not None:
            B, S, _ = sync.shape

            # === WRITE PATH: Features + Surprise -> Oscillator Memory ===
            if features is not None:
                # Compress features to single vector for writing using attention pooling
                # features: (B, S, d_feature_input) -> (d_feature_input,)
                if self.feature_write_attn is not None:
                    attn_scores = self.feature_write_attn(features).squeeze(-1)  # (B, S)
                    attn_weights = F.softmax(attn_scores, dim=-1)  # (B, S)
                    # Feature write attention diagnostics
                    with torch.no_grad():
                        eps = 1e-10
                        attn_entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
                        self._last_feature_write_attn_entropy = attn_entropy.mean().item()
                        self._last_feature_write_top1 = attn_weights.max(dim=-1).values.mean().item()
                        topk = min(5, attn_weights.shape[-1])
                        self._last_feature_write_top5 = attn_weights.topk(topk, dim=-1).values.sum(dim=-1).mean().item()
                        if surprise is not None:
                            surp = surprise
                            if surp.dim() == 3:
                                surp = surp.squeeze(-1)
                            if surp.shape == attn_weights.shape:
                                a = attn_weights.reshape(-1).float()
                                b = surp.reshape(-1).float()
                                a = a - a.mean()
                                b = b - b.mean()
                                denom = a.std(unbiased=False) * b.std(unbiased=False) + 1e-8
                                self._last_feature_write_surprise_corr = ((a * b).mean() / denom).item()
                            else:
                                self._last_feature_write_surprise_corr = None
                        else:
                            self._last_feature_write_surprise_corr = None
                    attn_weights_exp = attn_weights.unsqueeze(-1)  # (B, S, 1)
                    pooled = (attn_weights_exp * features).sum(dim=1)  # (B, d_feature_input)
                    features_compressed = self.feature_compressor(pooled.mean(dim=0))
                else:
                    features_compressed = self.feature_compressor(features.mean(dim=(0, 1)))
                    self._last_feature_write_attn_entropy = None
                    self._last_feature_write_top1 = None
                    self._last_feature_write_top5 = None
                    self._last_feature_write_surprise_corr = None

                # Compress surprise if provided
                surprise_scalar = None
                if surprise is not None:
                    if self.feature_write_attn is not None:
                        if surprise.dim() == 2:
                            surprise = surprise.unsqueeze(-1)
                        surprise_pooled = (attn_weights_exp * surprise).sum(dim=1)  # (B, 1)
                        surprise_scalar = surprise_pooled.mean(dim=0)  # (1,)
                    else:
                        surprise_scalar = surprise.mean()  # Scalar importance

                # Write to oscillator memory (content gated by surprise)
                if surprise_scalar is not None:
                    self._last_write_gate = torch.sigmoid(
                        self.oscillatory_world.surprise_gate_bias + surprise_scalar * self.oscillatory_world.surprise_gate_scale
                    ).item()
                else:
                    self._last_write_gate = torch.sigmoid(
                        torch.tensor(
                            self.oscillatory_world.surprise_gate_bias,
                            device=features_compressed.device,
                            dtype=features_compressed.dtype,
                        )
                    ).item()

                osc_output = self.oscillatory_world(
                    features=features_compressed,
                    surprise=surprise_scalar,
                    dt=1.0
                )
            else:
                # Fallback: use sync-based writing (legacy behavior)
                self._last_feature_write_attn_entropy = None
                self._last_feature_write_top1 = None
                self._last_feature_write_top5 = None
                self._last_feature_write_surprise_corr = None
                if self.config.sync_compression == "attention" and self.sync_compression_attn is not None:
                    attn_scores = self.sync_compression_attn(sync)
                    attn_weights = F.softmax(attn_scores.view(-1), dim=0)
                    attn_weights = attn_weights.view(B, S, 1)
                    sync_compressed = (attn_weights * sync).sum(dim=(0, 1))
                elif self.config.sync_compression == "last":
                    sync_compressed = sync[:, -1, :].mean(dim=0)
                else:
                    sync_compressed = sync.mean(dim=(0, 1))

                # Project sync to feature dimension (legacy fallback)
                sync_as_features = self.sync_to_feature(sync_compressed)
                self._last_write_gate = torch.sigmoid(
                    torch.tensor(
                        self.oscillatory_world.surprise_gate_bias,
                        device=sync_as_features.device,
                        dtype=sync_as_features.dtype,
                    )
                ).item()
                osc_output = self.oscillatory_world(
                    features=sync_as_features,
                    surprise=None,
                    dt=1.0
                )

            # osc_output.output: (d_world_output,) - projected output
            # osc_output.memory_states: (num_oscillators,) - raw memory for cross-attention

            # === READ PATH: Sync -> Query Memory -> Position-specific Context ===
            if self.config.use_oscillator_cross_attention:
                # Query: sync patterns determine what each position retrieves
                # (B, S, sync_pairs) -> (B, S, d_world_output)
                query = self.osc_query_proj(sync)

                # Key/Value: each oscillator is a separate key-value pair
                # memory_states: (num_oscillators,) - scalar state per oscillator
                # osc_embeddings: (num_oscillators, d_osc_embed) - learned embedding per oscillator
                memory = osc_output.memory_states  # (num_oscillators,)

                # Modulate embeddings by oscillator state: embedding * (1 + state)
                # This allows the memory content to influence what's retrieved
                # state > 0: amplify embedding, state < 0: flip sign, state = 0: suppress
                modulated_embeddings = self.osc_embeddings * (1.0 + memory.unsqueeze(-1))
                # modulated_embeddings: (num_oscillators, d_osc_embed)

                # Project to key/value: (num_oscillators, d_osc_embed) -> (num_oscillators, d_world_output)
                key = self.osc_key_proj(modulated_embeddings)    # (num_osc, d_world)
                value = self.osc_value_proj(modulated_embeddings)  # (num_osc, d_world)

                # Expand for batch: (num_osc, d_world) -> (B, num_osc, d_world)
                key = key.unsqueeze(0).expand(B, -1, -1)
                value = value.unsqueeze(0).expand(B, -1, -1)

                # Cross-attention: each position queries the 64 oscillators
                # Query: (B, S, d_world), Key: (B, 64, d_world), Value: (B, 64, d_world)
                # Output: (B, S, d_world) - position-specific weighted combination of oscillators
                attn_out, attn_weights = self._osc_cross_attention(query, key, value)
                # attn_weights: (B, S, 64) - attention over 64 oscillators per position
                world_state = self.osc_output_norm(attn_out + query)  # Residual + norm
                self._last_osc_attn_out_norm = attn_out.norm(dim=-1).mean().item()
                self._last_osc_query_act_norm = query.norm(dim=-1).mean().item()
                self._last_osc_attn_query_ratio = (
                    self._last_osc_attn_out_norm / (self._last_osc_query_act_norm + 1e-8)
                )
                self._last_osc_attn_top1 = attn_weights.max(dim=-1).values.mean().item()
                with torch.no_grad():
                    q_norm = F.normalize(query, dim=-1)
                    k_norm = F.normalize(key, dim=-1)
                    qk_cos = torch.einsum('bsd,bnd->bsn', q_norm, k_norm).mean()
                    self._last_osc_key_query_cosine = qk_cos.item()

                # Compute attention entropy: measures how distributed attention is
                # High entropy = using many oscillators (good), low = always same ones (bad)
                # Entropy = -sum(p * log(p)), max entropy = log(64) ≈ 4.16 for uniform
                with torch.no_grad():
                    # Average attention weights across batch and positions
                    avg_attn = attn_weights.mean(dim=(0, 1))  # (64,)
                    # Add small epsilon to avoid log(0)
                    avg_attn = avg_attn + 1e-10
                    avg_attn = avg_attn / avg_attn.sum()  # Renormalize
                    osc_attn_entropy = -(avg_attn * torch.log(avg_attn)).sum().item()
            else:
                # No cross-attention: use projected output (broadcast to all positions)
                world_state = osc_output.output
                osc_attn_entropy = None
                self._last_osc_attn_out_norm = None
                self._last_osc_query_act_norm = None
                self._last_osc_attn_query_ratio = None
                self._last_osc_attn_top1 = None
                self._last_osc_key_query_cosine = None

            # Get oscillator state for monitoring
            osc_state = self.oscillatory_world.get_oscillator_state()
            oscillator_amplitudes = osc_state['current_amplitudes']
            oscillator_phases = osc_state['phases']
            oscillator_metrics = self.oscillatory_world.get_metrics()

            # Compute future prediction if enabled
            future_prediction = None
            if self.future_predictor is not None:
                # Use memory states to predict future features
                future_prediction = self.future_predictor(osc_output.memory_states)

        return GlobalSyncOutput(
            sync=sync,
            cross_module_sync=cross_module_sync,
            module_contributions=contributions,
            world_state=world_state,
            oscillator_metrics=oscillator_metrics,
            oscillator_amplitudes=oscillator_amplitudes,
            oscillator_phases=oscillator_phases,
            future_prediction=future_prediction,
            osc_attn_entropy=osc_attn_entropy,
        )

    def get_world_state(self) -> Optional[torch.Tensor]:
        """
        Get the current world state from oscillatory model.

        Returns:
            World state tensor (d_world_output,) or None if oscillatory world disabled.
        """
        if self.oscillatory_world is not None:
            return self.oscillatory_world.read()
        return None

    def get_world_state_stats(self) -> Dict[str, float]:
        """
        Get statistics about the oscillatory world state for logging.

        Returns:
            Dict with oscillator metrics and basic state statistics.
        """
        if self.oscillatory_world is None:
            return {}

        metrics = self.oscillatory_world.get_metrics()
        osc_state = self.oscillatory_world.get_oscillator_state()

        stats = {
            # Basic output stats
            'world_state/norm': metrics.output_norm,
            'world_state/mean': metrics.output_mean,
            'world_state/std': metrics.output_std,
            'world_state/update_count': self.oscillatory_world._update_count.item(),
            # Oscillator-specific stats
            'oscillator/phase_mean': metrics.phase_mean,
            'oscillator/phase_std': metrics.phase_std,
            'oscillator/phase_entropy': metrics.phase_entropy,
            'oscillator/amplitude_mean': metrics.amplitude_mean,
            'oscillator/amplitude_std': metrics.amplitude_std,
            'oscillator/amplitude_max': metrics.amplitude_max,
            'oscillator/active_frac': metrics.active_oscillator_frac,
            'oscillator/freq_weighted_amp': metrics.frequency_weighted_amplitude,
            'oscillator/amp_mod_mean': metrics.amp_mod_mean,
            'oscillator/phase_mod_mean': metrics.phase_mod_mean,
        }
        # Oscillator frequency band utilization (based on learned frequencies)
        freqs = osc_state['frequencies']
        amps = osc_state['current_amplitudes'].abs()
        if freqs.numel() >= 4:
            sorted_idx = torch.argsort(freqs)
            amps_sorted = amps[sorted_idx]
            n = amps_sorted.numel()
            n_band = n // 4
            n_slow = n_band
            n_fast = n_band
            n_mid = n - n_slow - n_fast
            slow_mean = amps_sorted[:n_slow].mean()
            mid_mean = amps_sorted[n_slow:n_slow + n_mid].mean() if n_mid > 0 else amps_sorted[:n_slow].mean()
            fast_mean = amps_sorted[-n_fast:].mean() if n_fast > 0 else amps_sorted[-n_slow:].mean()
            stats.update({
                'osc/slow_amp_mean': slow_mean.item(),
                'osc/mid_amp_mean': mid_mean.item(),
                'osc/fast_amp_mean': fast_mean.item(),
                'osc/freq_band_ratio': (slow_mean / (fast_mean + 1e-8)).item(),
            })
        if self._last_feature_write_attn_entropy is not None:
            stats.update({
                'feature_write/pos_attn_entropy': self._last_feature_write_attn_entropy,
                'feature_write/top1_weight': self._last_feature_write_top1,
                'feature_write/top5_weight': self._last_feature_write_top5,
                'feature_write/surprise_correlation': self._last_feature_write_surprise_corr
                if self._last_feature_write_surprise_corr is not None else 0.0,
            })
        if self._last_write_gate is not None:
            stats['surprise/write_gate_mean'] = self._last_write_gate
        if self._last_osc_attn_top1 is not None:
            stats['osc_xattn/top1_osc_weight'] = self._last_osc_attn_top1
        if self._last_osc_key_query_cosine is not None:
            stats['osc_xattn/key_query_cosine'] = self._last_osc_key_query_cosine
        if self._last_osc_attn_out_norm is not None:
            stats.update({
                'osc_xattn/attn_out_norm': self._last_osc_attn_out_norm,
                'osc_xattn/query_act_norm': self._last_osc_query_act_norm,
                'osc_xattn/attn_query_ratio': self._last_osc_attn_query_ratio,
            })
        return stats

    def reset_oscillator_phases(self, random: bool = True):
        """
        Reset oscillator phases (mainly for testing/ablation).

        Args:
            random: If True, randomize phases. If False, set to zero.
        """
        if self.oscillatory_world is not None:
            self.oscillatory_world.reset_phases(random=random)


def create_global_sync(
    d_sync_space: int = 128,
    sync_pairs: int = 256,
    n_heads: int = 4,
    num_oscillators: int = 64,
    min_period: int = 8,
    max_period: int = 4096,
    d_world_output: int = 256,
    **kwargs,
) -> GlobalSyncModule:
    """Factory function to create GlobalSyncModule with oscillatory world model."""
    config = GlobalSyncConfig(
        d_sync_space=d_sync_space,
        sync_pairs=sync_pairs,
        n_heads=n_heads,
        num_oscillators=num_oscillators,
        min_period=min_period,
        max_period=max_period,
        d_world_output=d_world_output,
        **kwargs,
    )
    return GlobalSyncModule(config)
