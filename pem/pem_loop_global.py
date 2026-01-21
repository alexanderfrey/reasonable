"""
PEM Loop with Global Sync - Experience loop using cross-module synchronization.

This implements the Global Sync Architecture where:
1. Multiple CTM-based modules process features in parallel
2. Each module exposes its NLM post-activations
3. GlobalSyncModule computes cross-module synchronization
4. Global sync state drives attention to features

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         PEM Loop (Global Sync)                              │
    │                                                                             │
    │   Features ───────────────────────────────────────────────────────┐         │
    │       │                                                           │         │
    │       ▼                                                           │         │
    │   ┌─────────────────────────────────────────────────────────┐     │         │
    │   │             CTM Modules (parallel)                      │     │         │
    │   │                                                         │     │         │
    │   │  ┌─────────────────┐      ┌─────────────────┐          │     │         │
    │   │  │  PredictionCTM  │      │   SurpriseCTM   │          │     │         │
    │   │  │  NLM→Sync→Syn   │      │  NLM→Sync→Syn   │          │     │         │
    │   │  │       │         │      │       │         │          │     │         │
    │   │  │       ▼         │      │       ▼         │          │     │         │
    │   │  │  predictions    │      │  surprise       │          │     │         │
    │   │  │  h_pred         │      │  h_surp         │          │     │         │
    │   │  └───────┬─────────┘      └───────┬─────────┘          │     │         │
    │   │          │                        │                     │     │         │
    │   │          │   Post-Activations     │                     │     │         │
    │   │          └──────────┬─────────────┘                     │     │         │
    │   └─────────────────────┼───────────────────────────────────┘     │         │
    │                         │                                         │         │
    │                         ▼                                         │         │
    │   ┌─────────────────────────────────────────────────────────┐     │         │
    │   │                  GlobalSyncModule                       │     │         │
    │   │                                                         │     │         │
    │   │   [h_pred, h_surp] → Cross-Module Sync → sync_global    │     │         │
    │   │                                                         │     │         │
    │   └──────────────────────────┬──────────────────────────────┘     │         │
    │                              │                                    │         │
    │                              ▼                                    │         │
    │   ┌─────────────────────────────────────────────────────────┐     │         │
    │   │                PerceptionAttention                      │     │         │
    │   │                                                         │     │         │
    │   │   sync_global → Query → Attend to Features → observation│◄────┘         │
    │   │                                                         │               │
    │   └──────────────────────────┬──────────────────────────────┘               │
    │                              │                                              │
    │                              ▼                                              │
    │                        observation ─────────────────────────────────────────┘
    │                              │                                   (loops back)
    │                              ▼
    │                       PEMLoopOutput
    └─────────────────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .prediction_ctm import PredictionCTM, PredictionCTMConfig, PredictionCTMOutput
from .surprise_ctm import SurpriseCTM, SurpriseCTMConfig, SurpriseCTMOutput
from .global_sync import GlobalSyncModule, GlobalSyncConfig, GlobalSyncOutput
from .prediction_module import PredictionTargets
from .ctm_prediction_module import CTMLoss


def compute_tick_certainties(
    all_tick_outputs: List[torch.Tensor],
    window: int = 4,
) -> List[torch.Tensor]:
    """
    Compute certainty at each tick based on output stability.

    Certainty is based on how stable the outputs are - low variance
    and small changes indicate high certainty.

    Args:
        all_tick_outputs: List of y_t tensors at each tick
        window: Number of recent ticks to consider for variance

    Returns:
        List of certainty values (scalar tensors) for each tick
    """
    certainties = []
    for t in range(len(all_tick_outputs)):
        if t == 0:
            # First tick - low certainty (no history)
            certainties.append(torch.tensor(0.1, device=all_tick_outputs[0].device))
            continue

        # Use recent outputs for stability measure
        start = max(0, t - window + 1)
        recent = torch.stack(all_tick_outputs[start:t+1], dim=0)

        # Variance-based certainty
        mean_output = recent.mean(dim=0)
        variance = ((recent - mean_output) ** 2).mean()

        # Change from previous tick
        change = (all_tick_outputs[t] - all_tick_outputs[t-1]).norm(dim=-1).mean()

        # Convert to certainty (high stability = high certainty)
        variance_certainty = torch.exp(-variance * 10.0)
        change_certainty = torch.exp(-change * 5.0)
        certainty = 0.5 * variance_certainty + 0.5 * change_certainty

        certainties.append(certainty)

    return certainties


def compute_ctm_loss(
    all_tick_losses: List[torch.Tensor],
    all_tick_certainties: List[torch.Tensor],
) -> Tuple[torch.Tensor, int, int]:
    """
    Compute CTM paper loss: L = (L_t1 + L_t2) / 2

    Where:
        t1 = argmin(L) - tick with minimum loss
        t2 = argmax(C) - tick with maximum certainty

    This loss function encourages the model to:
    1. Produce the best answer at SOME tick (not necessarily the last)
    2. Be confident when it has the right answer

    Args:
        all_tick_losses: List of loss tensors, one per tick
        all_tick_certainties: List of certainty tensors, one per tick

    Returns:
        loss: The CTM loss (L_t1 + L_t2) / 2
        t1: Index of minimum loss tick
        t2: Index of maximum certainty tick
    """
    if len(all_tick_losses) == 0:
        raise ValueError("No tick losses provided")

    device = all_tick_losses[0].device

    # Stack losses and certainties
    losses = torch.stack(all_tick_losses)  # (T,) or (T, ...)
    certainties = torch.stack(all_tick_certainties)  # (T,) or (T, ...)

    # Find t1 = argmin(losses) and t2 = argmax(certainties)
    # Handle case where losses/certainties might have extra dimensions
    if losses.dim() > 1:
        # Average over batch/spatial dims for argmin/argmax
        losses_for_argmin = losses.view(losses.shape[0], -1).mean(dim=-1)
        certainties_for_argmax = certainties.view(certainties.shape[0], -1).mean(dim=-1)
    else:
        losses_for_argmin = losses
        certainties_for_argmax = certainties

    t1 = torch.argmin(losses_for_argmin).item()
    t2 = torch.argmax(certainties_for_argmax).item()

    # CTM loss: average of loss at best tick and loss at most certain tick
    L_t1 = all_tick_losses[t1]
    L_t2 = all_tick_losses[t2]

    ctm_loss = (L_t1 + L_t2) / 2

    return ctm_loss, t1, t2


def compute_prediction_error_surprise(
    predicted: torch.Tensor,
    actual: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute surprise signal from prediction error.

    Args:
        predicted: (B, S, D) predicted features
        actual: (B, S, D) actual features
        normalize: If True, normalize the output to [0, 1] range

    Returns:
        surprise: (B, S, 1) surprise magnitude
    """
    # Compute per-position error
    error = (predicted - actual).pow(2).mean(dim=-1, keepdim=True)  # (B, S, 1)

    if normalize:
        # Normalize to [0, 1] using sigmoid
        error = torch.sigmoid(error - error.mean())

    return error


def compute_attention_entropy_surprise(
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute surprise signal from attention entropy.

    High entropy = attention is spread out = uncertain = high surprise
    Low entropy = attention is focused = confident = low surprise

    Args:
        attention_weights: (B, H, S, S) attention weights

    Returns:
        surprise: (B, S, 1) surprise magnitude
    """
    # Average over heads: (B, H, S, S) -> (B, S, S)
    attn = attention_weights.mean(dim=1)

    # Compute entropy for each query position
    # H = -sum(p * log(p))
    eps = 1e-10
    entropy = -(attn * torch.log(attn + eps)).sum(dim=-1, keepdim=True)  # (B, S, 1)

    # Normalize to [0, 1] range
    # Max entropy is log(S) when attention is uniform
    max_entropy = torch.log(torch.tensor(attn.shape[-1], dtype=attn.dtype, device=attn.device))
    surprise = entropy / (max_entropy + eps)

    return surprise


class PEMLoopGlobalOutput(NamedTuple):
    """Output from PEM loop with global sync."""
    predictions: Dict[str, torch.Tensor]  # {immediate, shortterm, longterm}
    prediction_output: PredictionCTMOutput # Full prediction output with activations
    surprise: SurpriseCTMOutput           # Surprise with post-activations
    global_sync: GlobalSyncOutput          # Cross-module sync
    observation: torch.Tensor              # (B, S, D) attended observation
    attention_weights: torch.Tensor        # (B, H, S, S) attention pattern
    world_state: Optional[torch.Tensor] = None  # (d_world_output,) current oscillatory world state (for monitoring)
    loop_features: Optional[torch.Tensor] = None  # (B, S, D) features used in this step (for aux prediction loss)


class PEMLoopGlobalState(NamedTuple):
    """State carried between PEM loop iterations."""
    observation: torch.Tensor              # (B, S, D) last observation
    cumulative_sync: torch.Tensor          # (B, S, sync_pairs) accumulated sync
    world_state: Optional[torch.Tensor] = None  # (d_world_output,) differentiable world state from previous step


@dataclass
class PEMLoopGlobalConfig:
    """Configuration for PEM loop with global sync."""

    # Dimensions
    d_model: int = 1536          # Feature dimension from backbone

    # PredictionCTM config
    pred_d_neurons: int = 256
    pred_T: int = 4
    pred_M: int = 8
    pred_synapse_hidden: int = 1024   # Hidden dim in synapse U-NET (matches train_prediction.py)
    pred_nlm_hidden: int = 64         # Hidden dim in per-neuron MLPs (matches train_prediction.py)
    pred_d_sync_out: int = 256        # Sync pairs for output (matches train_prediction.py)
    pred_d_sync_internal: int = 256   # Sync pairs for internal (matches train_prediction.py)

    # SurpriseCTM config
    surp_d_neurons: int = 128
    surp_T: int = 3
    surp_M: int = 4
    surp_synapse_hidden: int = 512    # Hidden dim in synapse U-NET
    surp_nlm_hidden: int = 32         # Hidden dim in per-neuron MLPs
    surp_d_sync_out: int = 128        # Sync pairs for output
    surp_d_sync_internal: int = 128   # Sync pairs for internal

    # GlobalSync config
    d_sync_space: int = 128
    sync_pairs: int = 256
    sync_n_heads: int = 4
    sync_attention_temperature: float = 2.0  # Higher = softer cross-module attention
    sync_cross_residual_strength: float = 0.0  # Cross-module residual (0=off, 0.1-0.3=moderate)

    # Oscillatory world model (replaces GRU-based persistent state)
    use_oscillatory_world: bool = True  # Enable/disable oscillatory world model
    num_oscillators: int = 64           # Number of oscillators
    min_period: int = 8                 # Fastest oscillator period
    max_period: int = 4096              # Slowest oscillator period
    d_world_output: int = 256           # Output dimension of world state
    d_osc_embed: int = 64               # Per-oscillator embedding dim for cross-attention
    surprise_gate_bias: float = 0.5     # Base write strength for surprise gating
    surprise_gate_scale: float = 1.0    # How much surprise amplifies writing

    # Auxiliary prediction (train oscillators to predict future)
    # NOTE: Ablation tests showed this HURTS performance - disabled by default
    use_auxiliary_prediction: bool = False
    auxiliary_prediction_horizon: int = 8  # How many steps ahead to predict
    auxiliary_prediction_weight: float = 0.1  # Loss weight for auxiliary prediction

    # Multi-tick world state injection
    multi_tick_world_injection: bool = False  # Inject world state at each CTM tick

    # Alternative surprise signals (for oscillator memory gating)
    # Options: "ctm" (uses SurpriseCTM), "prediction_error", "attention_entropy"
    # NOTE: Ablation tests showed attention_entropy gives best certainty
    surprise_signal_type: str = "attention_entropy"

    # Prediction horizons
    immediate_horizon: int = 8
    shortterm_horizon: int = 64
    longterm_horizon: int = 256

    # Attention config
    attention_n_heads: int = 8

    # Loop config
    sync_decay: float = 0.9      # Decay for cumulative sync
    observation_residual: float = 0.2  # Blend factor for observation update (0=replace, 1=keep)
    state_combiner_gate_init: float = -0.85  # Initial bias for state combiner gate (sigmoid of this value)

    # Internal tick config (within CTM modules)
    # Lower values = more responsive ticks. 0.35 causes plateau, try 0.1-0.15
    internal_obs_residual: float = 0.1  # Blend factor within CTM tick loop (prevents fixed-point)

    # Memory optimization
    gradient_checkpointing: bool = False  # Recompute activations in backward (saves VRAM)
    backprop_steps: int = -1              # Only backprop through last N steps (-1 = all)

    # Loss weights
    surprise_loss_weight: float = 0.1     # Weight for surprise calibration loss
    cross_attn_diversity_weight: float = 0.0  # Penalize degenerate cross-module attention
    loop_improvement_weight: float = 0.5  # Penalize loop steps that regress (step N worse than N-1)

    dropout: float = 0.0

    # Debug/ablation flags
    disable_surprise: bool = False        # Disable surprise module (prediction-only mode)
    bypass_state_combiner: bool = False   # Skip state combiner, use raw features (for comparison with train_prediction.py)


class SimpleAttention(nn.Module):
    """
    Simplified attention for attending to features based on sync.

    Uses global sync to build query, attends to feature KV cache.
    """

    def __init__(
        self,
        d_model: int,
        sync_pairs: int,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Sync -> Query
        self.sync_to_query = nn.Sequential(
            nn.Linear(sync_pairs, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Feature -> K, V
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Project surprise direction to query space
        self.surprise_to_query = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learnable scale for surprise contribution (start small)
        self.surprise_scale = nn.Parameter(torch.tensor(0.1))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        sync: torch.Tensor,      # (B, S, sync_pairs) global sync state
        features: torch.Tensor,  # (B, S, d_model) features to attend to
        state: torch.Tensor,     # (B, S, d_model) current state
        surprise_direction: Optional[torch.Tensor] = None,  # (B, S, d_model)
        surprise_magnitude: Optional[torch.Tensor] = None,  # (B, S, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attend to features based on global sync and surprise.

        Args:
            sync: Global sync state from GlobalSyncModule
            features: Features from backbone
            state: Current observation state
            surprise_direction: Direction of surprise (normalize(actual - predicted))
            surprise_magnitude: Magnitude of surprise (scalar per position)

        Returns:
            observation: (B, S, d_model) attended features
            attn_weights: (B, n_heads, S, S) attention pattern
        """
        B, S, D = features.shape

        # Build query from sync + state
        sync_query = self.sync_to_query(sync)  # (B, S, d_model)
        q = self.q_proj(sync_query + state)    # Combine sync and state

        # Inject surprise into query: "attend to what surprised me"
        if surprise_direction is not None:
            surprise_query = self.surprise_to_query(surprise_direction)
            if surprise_magnitude is not None:
                # Scale by magnitude: bigger surprise → stronger steering
                surprise_query = surprise_query * surprise_magnitude * self.surprise_scale
            q = q + surprise_query

        # K, V from features
        k = self.k_proj(features)
        v = self.v_proj(features)

        # Reshape for multi-head attention
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).reshape(B, S, D)

        # Output projection
        observation = self.o_proj(attended)

        return observation, attn_weights


class PEMLoopGlobal(nn.Module):
    """
    PEM Experience Loop with Global Sync Architecture.

    This version uses:
    1. PredictionCTM and SurpriseCTM (both CTM-based)
    2. GlobalSyncModule to combine their post-activations
    3. Global sync drives attention to features

    The key difference from the original PEM loop:
    - Both modules expose their NLM states
    - Cross-module synchronization determines attention
    - More modular - easy to add more CTM modules later
    """

    def __init__(self, config: PEMLoopGlobalConfig):
        super().__init__()
        self.config = config

        # 1. PredictionCTM (with world state support for oscillatory world model)
        pred_config = PredictionCTMConfig(
            d_input=config.d_model,
            d_output=config.d_model,
            d_neurons=config.pred_d_neurons,
            T=config.pred_T,
            M=config.pred_M,
            synapse_hidden=config.pred_synapse_hidden,
            nlm_hidden=config.pred_nlm_hidden,
            d_sync_out=config.pred_d_sync_out,
            d_sync_internal=config.pred_d_sync_internal,
            immediate_horizon=config.immediate_horizon,
            shortterm_horizon=config.shortterm_horizon,
            longterm_horizon=config.longterm_horizon,
            dropout=config.dropout,
            internal_obs_residual=config.internal_obs_residual,
            # World state config (for oscillatory world model)
            d_world_state=config.d_world_output,
            use_world_state=config.use_oscillatory_world,
            multi_tick_world_injection=config.multi_tick_world_injection,
        )
        self.prediction = PredictionCTM(pred_config)

        # 2. SurpriseCTM
        surp_config = SurpriseCTMConfig(
            d_model=config.d_model,
            d_input=config.d_model * 3,
            d_output=config.d_model,
            d_neurons=config.surp_d_neurons,
            T=config.surp_T,
            M=config.surp_M,
            synapse_hidden=config.surp_synapse_hidden,
            nlm_hidden=config.surp_nlm_hidden,
            d_sync_out=config.surp_d_sync_out,
            d_sync_internal=config.surp_d_sync_internal,
            dropout=config.dropout,
            internal_obs_residual=config.internal_obs_residual,
        )
        self.surprise = SurpriseCTM(surp_config)

        # 3. GlobalSyncModule (with oscillatory world model)
        sync_config = GlobalSyncConfig(
            d_sync_space=config.d_sync_space,
            sync_pairs=config.sync_pairs,
            n_heads=config.sync_n_heads,
            dropout=config.dropout,
            attention_temperature=config.sync_attention_temperature,
            cross_residual_strength=config.sync_cross_residual_strength,
            use_oscillatory_world=config.use_oscillatory_world,
            num_oscillators=config.num_oscillators,
            min_period=config.min_period,
            max_period=config.max_period,
            d_world_output=config.d_world_output,
            d_osc_embed=config.d_osc_embed,  # Per-oscillator embedding for cross-attention
            d_feature_input=config.d_model,  # Content features for memory writing
            surprise_gate_bias=config.surprise_gate_bias,
            surprise_gate_scale=config.surprise_gate_scale,
            use_auxiliary_prediction=config.use_auxiliary_prediction,
            auxiliary_prediction_horizon=config.auxiliary_prediction_horizon,
        )
        self.global_sync = GlobalSyncModule(sync_config)

        # Register modules with GlobalSync
        self.global_sync.register_module('prediction', config.pred_d_neurons)
        if not config.disable_surprise:
            self.global_sync.register_module('surprise', config.surp_d_neurons)

        # 4. Attention (sync -> attend to features)
        self.attention = SimpleAttention(
            d_model=config.d_model,
            sync_pairs=config.sync_pairs,
            n_heads=config.attention_n_heads,
            dropout=config.dropout,
        )

        # 5. Target computer
        self.target_computer = PredictionTargets(
            immediate_horizon=config.immediate_horizon,
            shortterm_horizon=config.shortterm_horizon,
            longterm_horizon=config.longterm_horizon,
        )

        # 6. CTM Loss (same as train_prediction.py)
        self.ctm_loss = CTMLoss(
            immediate_weight=1.0,
            shortterm_weight=0.5,
            longterm_weight=0.3,
            use_cosine=True,
            use_mse=True,
            mse_weight=0.1,
        )

        # 7. Gated state combiner (for loop)
        # Uses a learned gate to decide how much observation info to incorporate
        # This prevents the loop from corrupting features - it can only ADD info
        self.state_combiner_transform = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        # Gate: sigmoid output determines how much transformed info to use vs raw features
        self.state_combiner_gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid(),
        )
        # Initialize gate bias from config
        # Default -0.85 gives sigmoid ≈ 0.30 (moderate observation influence)
        # Previous value of -2.0 (sigmoid ≈ 0.12) was too conservative - loop couldn't learn
        with torch.no_grad():
            self.state_combiner_gate[0].bias.data.fill_(config.state_combiner_gate_init)

        # 8. Initial observation transform (breaks symmetry in first loop step)
        # Without this, first step sees state_combiner([features, features]) = no signal
        self.observation_init = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        # Initialize with small weights to start close to features but not identical
        with torch.no_grad():
            # Make it approximately identity + small perturbation
            self.observation_init[0].weight.data = torch.eye(config.d_model) * 0.9 + torch.randn(config.d_model, config.d_model) * 0.1
            self.observation_init[2].weight.data = torch.eye(config.d_model) * 0.9 + torch.randn(config.d_model, config.d_model) * 0.1

    def init_state(self, features: torch.Tensor) -> PEMLoopGlobalState:
        """Initialize loop state.

        IMPORTANT: observation is initialized as a TRANSFORMED version of features,
        not the raw features themselves. This prevents the first loop step from
        seeing state_combiner([features, features]) which provides no useful signal.
        """
        B, S, D = features.shape
        device = features.device

        # Transform features for initial observation to break symmetry
        # Uses the observation_init projection to create a differentiated starting point
        initial_obs = self.observation_init(features)

        return PEMLoopGlobalState(
            observation=initial_obs,
            cumulative_sync=torch.zeros(B, S, self.config.sync_pairs, device=device),
            world_state=None,  # First step uses None, subsequent steps use differentiable world_state
        )

    def step(
        self,
        features: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        state: Optional[PEMLoopGlobalState] = None,
    ) -> Tuple[PEMLoopGlobalOutput, PEMLoopGlobalState]:
        """
        Execute one step of the PEM loop.

        Args:
            features: (B, S, d_model) from backbone
            targets: Prediction targets (computed if None)
            state: Previous loop state (initialized if None)

        Returns:
            output: PEMLoopGlobalOutput
            new_state: Updated state for next iteration
        """
        B, S, D = features.shape

        # Initialize state if needed
        if state is None:
            state = self.init_state(features)

        # Compute targets if needed
        if targets is None:
            targets = self.target_computer.compute_targets_efficient(features)

        # 1. Combine features with previous observation using gated skip connection
        # Gate determines how much observation info to incorporate (vs using raw features)
        # This prevents the loop from corrupting features - it can only ADD information
        if self.config.bypass_state_combiner:
            # Bypass: use raw features directly (equivalent to train_prediction.py)
            loop_features = features
        else:
            combined = torch.cat([features, state.observation], dim=-1)
            transformed = self.state_combiner_transform(combined)
            gate = self.state_combiner_gate(combined)  # (B, S, d_model), values in [0, 1]
            # Gated combination: features + gate * (transformed - features)
            # When gate ≈ 0: output ≈ features (safe default)
            # When gate ≈ 1: output ≈ transformed (full combination)
            loop_features = features + gate * (transformed - features)

        # 2. Get world state from state (differentiable from previous step's global_sync)
        # For first step, state.world_state is None; oscillatory world model will still
        # produce a differentiable output via its forward() method later
        # Shape is either:
        #   - (d_world_output,) when use_oscillator_cross_attention=False (broadcast)
        #   - (B, S, d_world_output) when use_oscillator_cross_attention=True (position-specific)
        world_state = state.world_state

        # 3. PredictionCTM: generate predictions (with world state influencing z_0)
        pred_output = self.prediction(loop_features, world_state=world_state)
        predictions = pred_output.predictions

        # 4. SurpriseCTM: compute surprise (using immediate scale)
        if self.config.disable_surprise:
            surp_output = None
        else:
            surp_output = self.surprise(
                predicted=predictions['immediate'],
                actual=targets['immediate'],
                valid_mask=targets.get('immediate_valid', None),
            )

        # 4.5. Compute surprise signal for oscillator memory gating
        # Can use CTM surprise, prediction error, or attention entropy
        surprise_signal = None
        if self.config.surprise_signal_type == "ctm":
            # Use SurpriseCTM magnitude (default)
            if surp_output is not None:
                surprise_signal = surp_output.magnitude
        elif self.config.surprise_signal_type == "prediction_error":
            # Use simple prediction error as surprise
            surprise_signal = compute_prediction_error_surprise(
                predicted=predictions['immediate'],
                actual=targets['immediate'],
            )
        elif self.config.surprise_signal_type == "attention_entropy":
            # Use attention entropy as surprise (computed after attention step)
            # We'll compute this after step 7 and pass to next iteration
            # For now, use CTM surprise if available, else None
            if surp_output is not None:
                surprise_signal = surp_output.magnitude

        # 5. GlobalSyncModule: cross-module synchronization (computes updated world state)
        # Pass Z_history (all tick activations) for true sync computation (S = Z·Z^T)
        # NEW: Also pass features (content) and surprise (importance) for world model
        #      - Features write WHAT to remember
        #      - Surprise gates HOW STRONGLY to write (unexpected = important)
        if self.config.disable_surprise:
            # Prediction-only mode: no cross-module sync, just use prediction activations
            global_sync_output = self.global_sync(
                module_activations={
                    'prediction': pred_output.all_tick_activations,  # List[(B, S, pred_d_neurons)]
                },
                features=loop_features,  # Content to write to memory
                surprise=surprise_signal,  # May be None, prediction_error, or ctm
            )
        else:
            global_sync_output = self.global_sync(
                module_activations={
                    'prediction': pred_output.all_tick_activations,  # List[(B, S, pred_d_neurons)]
                    'surprise': surp_output.all_tick_activations,    # List[(B, S, surp_d_neurons)]
                },
                features=loop_features,  # Content to write to memory
                surprise=surprise_signal,  # Importance gate: high surprise = important
            )

        # 6. Update cumulative sync
        cumulative_sync = (
            self.config.sync_decay * state.cumulative_sync +
            (1 - self.config.sync_decay) * global_sync_output.sync
        )

        # 7. Attention: use global sync to attend to features
        # Surprise steers attention: direction points to "what was unexpected"
        attended_obs, attn_weights = self.attention(
            sync=global_sync_output.sync,
            features=features,
            state=state.observation,
            surprise_direction=surp_output.direction if surp_output is not None else None,
            surprise_magnitude=surp_output.magnitude if surp_output is not None else None,
        )

        # 8. Observation residual connection (prevents fixed points)
        # Blend new attended observation with previous observation
        alpha = self.config.observation_residual
        observation = alpha * state.observation + (1 - alpha) * attended_obs

        # 8.5. If using attention entropy as surprise, compute it now for next iteration
        # (This is used in the NEXT step's oscillator memory write)
        if self.config.surprise_signal_type == "attention_entropy":
            attention_entropy_surprise = compute_attention_entropy_surprise(attn_weights)
            # Store in state for next iteration (we could also re-run global_sync with this)
            # For simplicity, this affects the next iteration's memory write

        # Build output and new state
        # Include world_state for monitoring (oscillatory world evolves continuously, no commit needed)
        output = PEMLoopGlobalOutput(
            predictions=predictions,
            prediction_output=pred_output,
            surprise=surp_output,
            global_sync=global_sync_output,
            observation=observation,
            attention_weights=attn_weights,
            world_state=global_sync_output.world_state,  # Current oscillatory world state for monitoring
            loop_features=loop_features,  # For auxiliary prediction loss
        )

        new_state = PEMLoopGlobalState(
            observation=observation,
            cumulative_sync=cumulative_sync,
            world_state=global_sync_output.world_state,  # Differentiable world state for next step
        )

        return output, new_state

    def get_world_state_stats(self) -> Dict[str, float]:
        """Get statistics about the world state for logging."""
        return self.global_sync.get_world_state_stats()

    def _step_for_checkpoint(
        self,
        features: torch.Tensor,
        targets_immediate: torch.Tensor,
        targets_shortterm: torch.Tensor,
        targets_longterm: torch.Tensor,
        targets_immediate_valid: Optional[torch.Tensor],
        targets_shortterm_valid: Optional[torch.Tensor],
        targets_longterm_valid: Optional[torch.Tensor],
        observation: torch.Tensor,
        cumulative_sync: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Wrapper for step() that works with gradient checkpointing.

        Gradient checkpointing requires all inputs/outputs to be tensors,
        so we flatten the Dict/NamedTuple structures.
        """
        # Reconstruct targets dict
        targets = {
            'immediate': targets_immediate,
            'shortterm': targets_shortterm,
            'longterm': targets_longterm,
        }
        if targets_immediate_valid is not None:
            targets['immediate_valid'] = targets_immediate_valid
        if targets_shortterm_valid is not None:
            targets['shortterm_valid'] = targets_shortterm_valid
        if targets_longterm_valid is not None:
            targets['longterm_valid'] = targets_longterm_valid

        # Reconstruct state (world_state is None in checkpointed path for simplicity)
        # This means gradient checkpointing may not fully support world_state gradients
        state = PEMLoopGlobalState(
            observation=observation,
            cumulative_sync=cumulative_sync,
            world_state=None,  # TODO: Pass world_state through checkpoint if needed
        )

        # Run actual step
        output, new_state = self.step(features, targets, state)

        # Return flattened tensors (checkpointing needs tensor outputs)
        # We'll reconstruct the NamedTuple after
        pred_out = output.prediction_output
        surp_out = output.surprise

        # Stack all-tick data for CTM loss
        # Prediction: all_tick_outputs (y_t) used directly for CTM loss
        pred_outputs_stacked = torch.stack(pred_out.all_tick_outputs, dim=0)

        # Surprise: handle disabled case with placeholder tensors
        if surp_out is not None:
            surp_mag_stacked = torch.stack(surp_out.all_tick_magnitudes, dim=0)
            surp_outputs_stacked = torch.stack(surp_out.all_tick_outputs, dim=0)
            surp_acts_stacked = torch.stack(surp_out.all_tick_activations, dim=0)
            surp_magnitude = surp_out.magnitude
            surp_raw = surp_out.raw
            surp_certainty = surp_out.certainty
        else:
            # Placeholders for disabled surprise (single-element tensors)
            device = features.device
            surp_mag_stacked = torch.zeros(1, device=device)
            surp_outputs_stacked = torch.zeros(1, device=device)
            surp_acts_stacked = torch.zeros(1, device=device)
            surp_magnitude = torch.zeros(1, device=device)
            surp_raw = torch.zeros(1, device=device)
            surp_certainty = torch.zeros(1, device=device)

        return (
            output.predictions['immediate'],
            output.predictions['shortterm'],
            output.predictions['longterm'],
            pred_out.certainty,
            surp_magnitude,
            surp_raw,
            surp_certainty,
            output.global_sync.sync,
            output.global_sync.cross_module_sync,
            output.global_sync.module_contributions,
            output.observation,
            output.attention_weights,
            new_state.observation,
            new_state.cumulative_sync,
            # Pass through activations for global sync reconstruction
            torch.stack(pred_out.all_tick_activations, dim=0),
            surp_acts_stacked,
            # CTM loss data
            pred_outputs_stacked,      # y_t at each tick (for prediction CTM loss)
            surp_mag_stacked,          # magnitude at each tick (for surprise CTM loss)
            surp_outputs_stacked,      # y_t at each tick (for surprise certainty)
        )

    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        num_steps: int = 1,
    ) -> Tuple[List[PEMLoopGlobalOutput], PEMLoopGlobalState]:
        """
        Run PEM loop for multiple steps.

        Args:
            features: (B, S, d_model) from backbone
            targets: Prediction targets
            num_steps: Number of loop iterations

        Returns:
            outputs: List of outputs for each step
            final_state: Final loop state

        Memory optimization options (set in config):
            - gradient_checkpointing: Recompute activations during backward pass
              (reduces VRAM ~2-3x at cost of ~30% slower training)
            - backprop_steps: Only backprop through last N steps (-1 = all)
              (reduces VRAM linearly with steps, but may affect learning)
        """
        if targets is None:
            targets = self.target_computer.compute_targets_efficient(features)

        state = self.init_state(features)
        outputs = []

        # Determine which steps need gradients
        backprop_steps = self.config.backprop_steps
        if backprop_steps < 0:
            backprop_steps = num_steps  # All steps
        first_grad_step = max(0, num_steps - backprop_steps)

        for step in range(num_steps):
            # Truncated backprop: detach state for early steps
            if step < first_grad_step:
                state = PEMLoopGlobalState(
                    observation=state.observation.detach(),
                    cumulative_sync=state.cumulative_sync.detach(),
                    world_state=state.world_state.detach() if state.world_state is not None else None,
                )

            # Use gradient checkpointing if enabled
            if self.config.gradient_checkpointing and self.training and step >= first_grad_step:
                # Flatten inputs for checkpoint (needs all tensor args)
                ckpt_result = checkpoint(
                    self._step_for_checkpoint,
                    features,
                    targets['immediate'],
                    targets['shortterm'],
                    targets['longterm'],
                    targets.get('immediate_valid'),
                    targets.get('shortterm_valid'),
                    targets.get('longterm_valid'),
                    state.observation,
                    state.cumulative_sync,
                    use_reentrant=False,
                )

                # Reconstruct output from checkpoint result
                (pred_imm, pred_short, pred_long, pred_cert, surp_mag, surp_raw, surp_cert,
                 sync, cross_sync, contrib, obs, attn_w, new_obs, new_cum_sync,
                 pred_acts_stacked, surp_acts_stacked,
                 pred_outputs_stacked, surp_mag_stacked, surp_outputs_stacked) = ckpt_result

                # Reconstruct prediction output
                # all_tick_outputs (y_t) used directly for CTM loss
                num_pred_ticks = pred_outputs_stacked.shape[0]
                all_tick_outputs = [pred_outputs_stacked[t] for t in range(num_pred_ticks)]

                pred_output = PredictionCTMOutput(
                    predictions={'immediate': pred_imm, 'shortterm': pred_short, 'longterm': pred_long},
                    post_activations=pred_acts_stacked[-1],  # Final tick
                    sync_matrix=torch.zeros(1, device=pred_imm.device),  # Placeholder
                    certainty=pred_cert,
                    all_tick_outputs=all_tick_outputs,  # y_t at each tick for CTM loss
                    all_tick_activations=[pred_acts_stacked[i] for i in range(pred_acts_stacked.shape[0])],
                )

                # Reconstruct surprise output (or None if disabled)
                if self.config.disable_surprise:
                    surp_output = None
                else:
                    num_surp_ticks = surp_mag_stacked.shape[0]
                    all_tick_magnitudes = [surp_mag_stacked[t] for t in range(num_surp_ticks)]
                    surp_all_tick_outputs = [surp_outputs_stacked[t] for t in range(surp_outputs_stacked.shape[0])]

                    surp_output = SurpriseCTMOutput(
                        magnitude=surp_mag,
                        direction=torch.zeros(1, device=surp_mag.device),  # Placeholder
                        raw=surp_raw,
                        post_activations=surp_acts_stacked[-1],
                        sync_matrix=torch.zeros(1, device=surp_mag.device),
                        certainty=surp_cert,
                        all_tick_outputs=surp_all_tick_outputs,  # For certainty computation
                        all_tick_activations=[surp_acts_stacked[i] for i in range(surp_acts_stacked.shape[0])],
                        all_tick_magnitudes=all_tick_magnitudes,
                    )

                # Reconstruct global sync output (oscillator metrics not available in checkpoint)
                global_sync_output = GlobalSyncOutput(
                    sync=sync,
                    cross_module_sync=cross_sync,
                    module_contributions=contrib,
                    world_state=None,  # Not reconstructed from checkpoint
                    oscillator_metrics=None,
                    oscillator_amplitudes=None,
                    oscillator_phases=None,
                )

                output = PEMLoopGlobalOutput(
                    predictions={'immediate': pred_imm, 'shortterm': pred_short, 'longterm': pred_long},
                    prediction_output=pred_output,
                    surprise=surp_output,
                    global_sync=global_sync_output,
                    observation=obs,
                    attention_weights=attn_w,
                )

                state = PEMLoopGlobalState(
                    observation=new_obs,
                    cumulative_sync=new_cum_sync,
                    world_state=None,  # Checkpoint path doesn't preserve world_state gradients
                )
            else:
                # Normal forward pass
                output, state = self.step(features, targets, state)

            outputs.append(output)

        return outputs, state

    def compute_loss(
        self,
        outputs: List[PEMLoopGlobalOutput],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute CTM paper loss for training.

        CTM Loss Formula (per module):
            t1 = argmin(L)  - tick with minimum loss
            t2 = argmax(C)  - tick with maximum certainty
            L = (L_t1 + L_t2) / 2

        For Prediction (using CTMLoss - same as train_prediction.py):
            - Applies readout heads at EACH tick (not just final)
            - Computes weighted loss for all 3 horizons (immediate, shortterm, longterm)
            - Uses cosine similarity + MSE components

        For Surprise:
            - Uses all_tick_magnitudes (cheap scalar output per tick)
            - Calibration: magnitude should track raw_surprise

        Also includes:
        - Cross-module sync variance (encourages meaningful synchronization)
        - Optional cross-attention diversity loss
        - Optional loop improvement loss
        """
        device = outputs[0].predictions['immediate'].device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        for step_idx, output in enumerate(outputs):
            pred_output = output.prediction_output
            surp_output = output.surprise

            # ========== 1. PREDICTION CTM LOSS ==========
            # Use CTMLoss (same as train_prediction.py):
            # - Applies readout heads at each tick
            # - Computes loss for all 3 horizons (immediate, shortterm, longterm)
            # - Uses cosine similarity + MSE
            # - Finds t1=argmin(loss), t2=argmax(certainty), returns (L_t1 + L_t2) / 2
            all_tick_outputs = pred_output.all_tick_outputs

            if len(all_tick_outputs) > 0:
                pred_ctm_loss, pred_loss_dict = self.ctm_loss(
                    all_tick_outputs,
                    targets,
                    self.prediction.readout_immediate,
                    self.prediction.readout_shortterm,
                    self.prediction.readout_longterm,
                )
                # Extract tick selection info
                pred_t1 = pred_loss_dict.get('t1', torch.tensor(0))
                pred_t2 = pred_loss_dict.get('t2', torch.tensor(0))
                if torch.is_tensor(pred_t1):
                    pred_t1 = pred_t1.item()
                if torch.is_tensor(pred_t2):
                    pred_t2 = pred_t2.item()

                loss_dict[f'step{step_idx}_pred_ctm_loss'] = pred_ctm_loss.detach()
                loss_dict[f'step{step_idx}_pred_best_tick'] = float(pred_t1)
                loss_dict[f'step{step_idx}_pred_certain_tick'] = float(pred_t2)
                # Also log per-scale losses from final tick
                loss_dict[f'step{step_idx}_immediate_loss'] = pred_loss_dict.get('immediate_loss', torch.tensor(0.0)).detach()
                loss_dict[f'step{step_idx}_shortterm_loss'] = pred_loss_dict.get('shortterm_loss', torch.tensor(0.0)).detach()
                loss_dict[f'step{step_idx}_longterm_loss'] = pred_loss_dict.get('longterm_loss', torch.tensor(0.0)).detach()
                total_loss = total_loss + pred_ctm_loss

            # ========== 2. SURPRISE CTM LOSS ==========
            # Skip if surprise is disabled
            if surp_output is not None:
                # Use all_tick_magnitudes (cheap - just scalar per tick)
                all_tick_surp_losses = []
                for tick_mag in surp_output.all_tick_magnitudes:
                    # Surprise calibration: magnitude should track raw surprise
                    surp_cal_loss = F.mse_loss(tick_mag, surp_output.raw)
                    all_tick_surp_losses.append(surp_cal_loss)

                # Compute certainties from surprise outputs
                all_tick_surp_certainties = compute_tick_certainties(surp_output.all_tick_outputs)

                # Apply CTM loss formula
                if len(all_tick_surp_losses) > 0 and len(all_tick_surp_certainties) > 0:
                    surp_ctm_loss, surp_t1, surp_t2 = compute_ctm_loss(
                        all_tick_surp_losses,
                        all_tick_surp_certainties,
                    )
                    loss_dict[f'step{step_idx}_surp_ctm_loss'] = surp_ctm_loss.detach()
                    loss_dict[f'step{step_idx}_surp_best_tick'] = float(surp_t1)
                    loss_dict[f'step{step_idx}_surp_certain_tick'] = float(surp_t2)
                    total_loss = total_loss + self.config.surprise_loss_weight * surp_ctm_loss
                else:
                    # Fallback
                    surp_cal_loss = F.mse_loss(surp_output.magnitude, surp_output.raw)
                    total_loss = total_loss + self.config.surprise_loss_weight * surp_cal_loss

            # ========== 3. CROSS-MODULE SYNC VARIANCE ==========
            # Encourage meaningful cross-module synchronization
            cross_sync = output.global_sync.cross_module_sync
            sync_var = cross_sync.var()
            sync_var_loss = -sync_var * 0.01  # Negative = want MORE variance
            loss_dict[f'step{step_idx}_sync_var'] = sync_var.detach()
            total_loss = total_loss + sync_var_loss

            # ========== 4. CROSS-ATTENTION DIVERSITY LOSS ==========
            # Penalize degenerate attention (when modules only attend to themselves)
            # cross_sync shape: (num_modules, num_modules, B, S)
            # For 2 modules: [0,1] = P→S, [1,0] = S→P (off-diagonal)
            if self.config.cross_attn_diversity_weight > 0 and cross_sync.shape[0] >= 2:
                # Get off-diagonal attention (cross-module attention)
                p_to_s = cross_sync[0, 1].mean()  # Prediction attending to Surprise
                s_to_p = cross_sync[1, 0].mean()  # Surprise attending to Prediction

                # We want both to be meaningful (not 0 or 1)
                # Target: ~0.5 for balanced attention, penalize deviation toward 0 or 1
                # Loss: -log(p) - log(1-p) is minimized at p=0.5 (cross-entropy style)
                eps = 1e-6
                diversity_loss = -(
                    torch.log(p_to_s + eps) + torch.log(1 - p_to_s + eps) +
                    torch.log(s_to_p + eps) + torch.log(1 - s_to_p + eps)
                ) / 4  # Average over 4 terms

                # Subtract baseline (value at p=0.5) so loss is 0 when balanced
                baseline = -torch.log(torch.tensor(0.5 + eps)) * 2
                diversity_loss = diversity_loss - baseline

                loss_dict[f'step{step_idx}_cross_attn_diversity'] = diversity_loss.detach()
                total_loss = total_loss + self.config.cross_attn_diversity_weight * diversity_loss

        # Average across loop steps
        num_steps = len(outputs)
        total_loss = total_loss / num_steps

        # ========== 5. AUXILIARY PREDICTION LOSS ==========
        # Train oscillators to predict future features
        if self.config.auxiliary_prediction_weight > 0 and self.config.use_auxiliary_prediction:
            aux_loss = torch.tensor(0.0, device=device)
            aux_count = 0
            horizon = self.global_sync.config.auxiliary_prediction_horizon

            for step_idx, output in enumerate(outputs):
                future_pred = output.global_sync.future_prediction
                if future_pred is None:
                    continue

                # Target is average of features from future steps
                # Use loop_features from future outputs if available
                future_features = []
                for future_idx in range(step_idx + 1, min(step_idx + 1 + horizon, num_steps)):
                    if outputs[future_idx].loop_features is not None:
                        # Average over batch and sequence
                        future_features.append(
                            outputs[future_idx].loop_features.mean(dim=(0, 1))
                        )

                if future_features:
                    # Average future features
                    future_target = torch.stack(future_features).mean(dim=0)

                    # Cosine similarity loss
                    cos_sim = F.cosine_similarity(
                        future_pred.unsqueeze(0),
                        future_target.unsqueeze(0),
                        dim=-1
                    )
                    aux_loss = aux_loss + (1 - cos_sim.mean())
                    aux_count += 1

            if aux_count > 0:
                aux_loss = aux_loss / aux_count
                loss_dict['auxiliary_prediction_loss'] = aux_loss.detach()
                total_loss = total_loss + self.config.auxiliary_prediction_weight * aux_loss

        # ========== 6. LOOP IMPROVEMENT LOSS ==========
        # Penalize regression: if step N has worse loss than step N-1, add penalty
        # This encourages the loop to make progress (or at least not regress)
        if self.config.loop_improvement_weight > 0 and num_steps > 1:
            # Compute prediction loss at each step
            step_losses = []
            for output in outputs:
                pred = output.predictions['immediate']
                target = targets['immediate']
                valid = targets.get('immediate_valid', None)
                if valid is not None and valid.any():
                    cos_sim = F.cosine_similarity(pred[valid], target[valid], dim=-1)
                    step_loss = (1 - cos_sim).mean()
                else:
                    step_loss = torch.tensor(0.0, device=device)
                step_losses.append(step_loss)

            # Sum of ReLU(loss[i+1] - loss[i]) - penalize only regression
            improvement_loss = torch.tensor(0.0, device=device)
            for i in range(len(step_losses) - 1):
                regression = step_losses[i + 1] - step_losses[i]
                improvement_loss = improvement_loss + F.relu(regression)

            # Normalize by number of transitions
            improvement_loss = improvement_loss / (num_steps - 1)
            loss_dict['loop_improvement_loss'] = improvement_loss.detach()
            total_loss = total_loss + self.config.loop_improvement_weight * improvement_loss

        loss_dict['loss'] = total_loss.detach()

        return total_loss, loss_dict


def create_pem_loop_global(
    d_model: int = 1536,
    pred_d_neurons: int = 256,
    surp_d_neurons: int = 128,
    sync_pairs: int = 256,
    **kwargs,
) -> PEMLoopGlobal:
    """Factory function to create PEM loop with global sync."""
    config = PEMLoopGlobalConfig(
        d_model=d_model,
        pred_d_neurons=pred_d_neurons,
        surp_d_neurons=surp_d_neurons,
        sync_pairs=sync_pairs,
        **kwargs,
    )
    return PEMLoopGlobal(config)
