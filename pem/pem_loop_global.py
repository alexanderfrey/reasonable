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

from .prediction_ctm import PredictionCTM, PredictionCTMConfig, PredictionCTMOutput
from .surprise_ctm import SurpriseCTM, SurpriseCTMConfig, SurpriseCTMOutput
from .global_sync import GlobalSyncModule, GlobalSyncConfig, GlobalSyncOutput
from .prediction_module import PredictionTargets


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

    # Prediction horizons
    immediate_horizon: int = 8
    shortterm_horizon: int = 64
    longterm_horizon: int = 256

    # Attention config
    attention_n_heads: int = 8

    # Loop config
    sync_decay: float = 0.9      # Decay for cumulative sync

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
        )
        self.surprise = SurpriseCTM(surp_config)

        # 3. GlobalSyncModule
        sync_config = GlobalSyncConfig(
            d_sync_space=config.d_sync_space,
            sync_pairs=config.sync_pairs,
            n_heads=config.sync_n_heads,
            dropout=config.dropout,
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
        global_sync_output = self.global_sync({
            'prediction': pred_output.post_activations,
            'surprise': surp_output.post_activations,
        })

        # 5. Update cumulative sync
        cumulative_sync = (
            self.config.sync_decay * state.cumulative_sync +
            (1 - self.config.sync_decay) * global_sync_output.sync
        )

        # 6. Attention: use global sync to attend to features
        observation, attn_weights = self.attention(
            sync=global_sync_output.sync,
            features=features,
            state=state.observation,
        )

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
        """
        if targets is None:
            targets = self.target_computer.compute_targets_efficient(features)

        state = self.init_state(features)
        outputs = []

        for step in range(num_steps):
            output, state = self.step(features, targets, state)
            outputs.append(output)

        return outputs, state

    def compute_loss(
        self,
        outputs: List[PEMLoopGlobalOutput],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss for training.

        Combines:
        1. Prediction loss (cosine similarity to targets)
        2. Surprise calibration (magnitude tracks raw)
        3. Global sync coherence (modules should synchronize)
        """
        device = outputs[0].predictions['immediate'].device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        for step_idx, output in enumerate(outputs):
            # 1. Prediction loss
            for scale in ['immediate', 'shortterm', 'longterm']:
                pred = output.predictions[scale]
                target = targets[scale]
                valid = targets.get(f'{scale}_valid', None)

                if valid is not None and valid.any():
                    pred_valid = pred[valid]
                    target_valid = target[valid]
                    cos_sim = F.cosine_similarity(pred_valid, target_valid, dim=-1)
                    pred_loss = (1 - cos_sim).mean()
                else:
                    pred_loss = torch.tensor(0.0, device=device)

                loss_dict[f'step{step_idx}_{scale}_loss'] = pred_loss.detach()
                total_loss = total_loss + pred_loss

            # 2. Surprise calibration
            mag = output.surprise.magnitude
            raw = output.surprise.raw
            surp_cal_loss = F.mse_loss(mag, raw)
            loss_dict[f'step{step_idx}_surprise_cal'] = surp_cal_loss.detach()
            total_loss = total_loss + 0.1 * surp_cal_loss

            # 3. Cross-module sync should be meaningful
            # Encourage some variance in cross-module sync (not all 0.5)
            cross_sync = output.global_sync.cross_module_sync  # (num_modules, num_modules, B, S)
            sync_var = cross_sync.var()
            sync_var_loss = -sync_var * 0.01  # Negative because we want MORE variance
            loss_dict[f'step{step_idx}_sync_var'] = sync_var.detach()
            total_loss = total_loss + sync_var_loss

        # Average across steps
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
