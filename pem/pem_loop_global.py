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


class PEMLoopGlobalOutput(NamedTuple):
    """Output from PEM loop with global sync."""
    predictions: Dict[str, torch.Tensor]  # {immediate, shortterm, longterm}
    prediction_output: PredictionCTMOutput # Full prediction output with activations
    surprise: SurpriseCTMOutput           # Surprise with post-activations
    global_sync: GlobalSyncOutput          # Cross-module sync
    observation: torch.Tensor              # (B, S, D) attended observation
    attention_weights: torch.Tensor        # (B, H, S, S) attention pattern


class PEMLoopGlobalState(NamedTuple):
    """State carried between PEM loop iterations."""
    observation: torch.Tensor              # (B, S, D) last observation
    cumulative_sync: torch.Tensor          # (B, S, sync_pairs) accumulated sync


@dataclass
class PEMLoopGlobalConfig:
    """Configuration for PEM loop with global sync."""

    # Dimensions
    d_model: int = 1536          # Feature dimension from backbone

    # PredictionCTM config
    pred_d_neurons: int = 256
    pred_T: int = 4
    pred_M: int = 8

    # SurpriseCTM config
    surp_d_neurons: int = 128
    surp_T: int = 3
    surp_M: int = 4

    # GlobalSync config
    d_sync_space: int = 128
    sync_pairs: int = 256
    sync_n_heads: int = 4
    sync_attention_temperature: float = 2.0  # Higher = softer cross-module attention
    sync_cross_residual_strength: float = 0.0  # Cross-module residual (0=off, 0.1-0.3=moderate)

    # Prediction horizons
    immediate_horizon: int = 8
    shortterm_horizon: int = 64
    longterm_horizon: int = 256

    # Attention config
    attention_n_heads: int = 8

    # Loop config
    sync_decay: float = 0.9      # Decay for cumulative sync
    observation_residual: float = 0.3  # Blend factor for observation update (0=replace, 1=keep)

    # Internal tick config (within CTM modules)
    internal_obs_residual: float = 0.2  # Blend factor within CTM tick loop (prevents fixed-point)

    # Memory optimization
    gradient_checkpointing: bool = False  # Recompute activations in backward (saves VRAM)
    backprop_steps: int = -1              # Only backprop through last N steps (-1 = all)

    # Loss weights
    surprise_loss_weight: float = 0.1     # Weight for surprise calibration loss

    dropout: float = 0.0


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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attend to features based on global sync.

        Args:
            sync: Global sync state from GlobalSyncModule
            features: Features from backbone
            state: Current observation state

        Returns:
            observation: (B, S, d_model) attended features
            attn_weights: (B, n_heads, S, S) attention pattern
        """
        B, S, D = features.shape

        # Build query from sync + state
        sync_query = self.sync_to_query(sync)  # (B, S, d_model)
        q = self.q_proj(sync_query + state)    # Combine sync and state

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

        # 1. PredictionCTM
        pred_config = PredictionCTMConfig(
            d_input=config.d_model,
            d_output=config.d_model,
            d_neurons=config.pred_d_neurons,
            T=config.pred_T,
            M=config.pred_M,
            immediate_horizon=config.immediate_horizon,
            shortterm_horizon=config.shortterm_horizon,
            longterm_horizon=config.longterm_horizon,
            dropout=config.dropout,
            internal_obs_residual=config.internal_obs_residual,
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
            dropout=config.dropout,
            internal_obs_residual=config.internal_obs_residual,
        )
        self.surprise = SurpriseCTM(surp_config)

        # 3. GlobalSyncModule
        sync_config = GlobalSyncConfig(
            d_sync_space=config.d_sync_space,
            sync_pairs=config.sync_pairs,
            n_heads=config.sync_n_heads,
            dropout=config.dropout,
            attention_temperature=config.sync_attention_temperature,
            cross_residual_strength=config.sync_cross_residual_strength,
        )
        self.global_sync = GlobalSyncModule(sync_config)

        # Register modules with GlobalSync
        self.global_sync.register_module('prediction', config.pred_d_neurons)
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

        # 6. State combiner (for loop)
        self.state_combiner = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

    def init_state(self, features: torch.Tensor) -> PEMLoopGlobalState:
        """Initialize loop state."""
        B, S, D = features.shape
        device = features.device

        return PEMLoopGlobalState(
            observation=features,
            cumulative_sync=torch.zeros(B, S, self.config.sync_pairs, device=device),
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

        # 1. Combine features with previous observation
        combined = torch.cat([features, state.observation], dim=-1)
        loop_features = self.state_combiner(combined)

        # 2. PredictionCTM: generate predictions
        pred_output = self.prediction(loop_features)
        predictions = pred_output.predictions

        # 3. SurpriseCTM: compute surprise (using immediate scale)
        surp_output = self.surprise(
            predicted=predictions['immediate'],
            actual=targets['immediate'],
            valid_mask=targets.get('immediate_valid', None),
        )

        # 4. GlobalSyncModule: cross-module synchronization
        # Pass Z_history (all tick activations) for true sync computation (S = Z·Z^T)
        global_sync_output = self.global_sync({
            'prediction': pred_output.all_tick_activations,  # List[(B, S, pred_d_neurons)]
            'surprise': surp_output.all_tick_activations,    # List[(B, S, surp_d_neurons)]
        })

        # 5. Update cumulative sync
        cumulative_sync = (
            self.config.sync_decay * state.cumulative_sync +
            (1 - self.config.sync_decay) * global_sync_output.sync
        )

        # 6. Attention: use global sync to attend to features
        attended_obs, attn_weights = self.attention(
            sync=global_sync_output.sync,
            features=features,
            state=state.observation,
        )

        # 7. Observation residual connection (prevents fixed points)
        # Blend new attended observation with previous observation
        alpha = self.config.observation_residual
        observation = alpha * state.observation + (1 - alpha) * attended_obs

        # Build output and new state
        output = PEMLoopGlobalOutput(
            predictions=predictions,
            prediction_output=pred_output,
            surprise=surp_output,
            global_sync=global_sync_output,
            observation=observation,
            attention_weights=attn_weights,
        )

        new_state = PEMLoopGlobalState(
            observation=observation,
            cumulative_sync=cumulative_sync,
        )

        return output, new_state

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

        # Reconstruct state
        state = PEMLoopGlobalState(
            observation=observation,
            cumulative_sync=cumulative_sync,
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
        # Surprise: all_tick_magnitudes (cheap scalar per tick)
        surp_mag_stacked = torch.stack(surp_out.all_tick_magnitudes, dim=0)
        surp_outputs_stacked = torch.stack(surp_out.all_tick_outputs, dim=0)

        return (
            output.predictions['immediate'],
            output.predictions['shortterm'],
            output.predictions['longterm'],
            pred_out.certainty,
            surp_out.magnitude,
            surp_out.raw,
            surp_out.certainty,
            output.global_sync.sync,
            output.global_sync.cross_module_sync,
            output.global_sync.module_contributions,
            output.observation,
            output.attention_weights,
            new_state.observation,
            new_state.cumulative_sync,
            # Pass through activations for global sync reconstruction
            torch.stack(pred_out.all_tick_activations, dim=0),
            torch.stack(surp_out.all_tick_activations, dim=0),
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

                # Reconstruct surprise output
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

                # Reconstruct global sync output
                global_sync_output = GlobalSyncOutput(
                    sync=sync,
                    cross_module_sync=cross_sync,
                    module_contributions=contrib,
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

        For Prediction:
            - Uses all_tick_outputs (y_t) directly - no readout heads per tick
            - Compares y_t to immediate target (primary prediction task)

        For Surprise:
            - Uses all_tick_magnitudes (cheap scalar output per tick)
            - Calibration: magnitude should track raw_surprise

        Also includes:
        - Cross-module sync variance (encourages meaningful synchronization)
        """
        device = outputs[0].predictions['immediate'].device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        for step_idx, output in enumerate(outputs):
            pred_output = output.prediction_output
            surp_output = output.surprise

            # ========== 1. PREDICTION CTM LOSS ==========
            # Use y_t (all_tick_outputs) directly - no per-tick readouts needed
            # Compare to immediate target (primary task)
            all_tick_outputs = pred_output.all_tick_outputs
            target = targets['immediate']
            valid = targets.get('immediate_valid', None)

            # Compute loss at each tick using raw y_t
            all_tick_pred_losses = []
            for y_t in all_tick_outputs:
                if valid is not None and valid.any():
                    cos_sim = F.cosine_similarity(y_t[valid], target[valid], dim=-1)
                    tick_loss = (1 - cos_sim).mean()
                else:
                    tick_loss = torch.tensor(0.0, device=device)
                all_tick_pred_losses.append(tick_loss)

            # Compute certainties from output stability
            all_tick_pred_certainties = compute_tick_certainties(all_tick_outputs)

            # Apply CTM loss formula: L = (L_t1 + L_t2) / 2
            if len(all_tick_pred_losses) > 0:
                pred_ctm_loss, pred_t1, pred_t2 = compute_ctm_loss(
                    all_tick_pred_losses,
                    all_tick_pred_certainties,
                )
                loss_dict[f'step{step_idx}_pred_ctm_loss'] = pred_ctm_loss.detach()
                loss_dict[f'step{step_idx}_pred_best_tick'] = float(pred_t1)
                loss_dict[f'step{step_idx}_pred_certain_tick'] = float(pred_t2)
                total_loss = total_loss + pred_ctm_loss

            # ========== 2. SURPRISE CTM LOSS ==========
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

        # Average across loop steps
        num_steps = len(outputs)
        total_loss = total_loss / num_steps
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
