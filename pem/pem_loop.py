"""
PEM Experience Loop - The core predictive processing loop.

This module connects:
    CTMPrediction → SurpriseModule → PerceptionAttention

The loop:
1. CTMPrediction generates predictions of future features
2. SurpriseModule computes prediction errors (magnitude + direction)
3. PerceptionAttention uses surprise to modulate WHERE to attend next
4. The attended observation feeds back into CTMPrediction

This implements the core PEM theory: the brain is a prediction machine
that constantly generates expectations and learns from prediction errors.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                         PEM Loop                                 │
    │                                                                 │
    │   Features ──────►  CTMPrediction  ──────► predictions          │
    │       ▲                   │                    │                │
    │       │                   │                    ▼                │
    │       │                   │            SurpriseModule           │
    │       │                   │                    │                │
    │       │                   │         ┌─────────┴─────────┐       │
    │       │                   │         │mag          dir   │       │
    │       │                   │         └─────────┬─────────┘       │
    │       │                   ▼                   │                 │
    │       │          PerceptionAttention ◄────────┘                 │
    │       │                   │                                     │
    │       │                   ▼                                     │
    │       └──────────── observation                                 │
    └─────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctm_prediction_module import (
    CTMPredictionConfig,
    CTMPrediction,
    CTMPredictionOutput,
    CTMLoss,
)
from .surprise_module import (
    SurpriseConfig,
    SurpriseModule,
)
from .perception_attention import (
    PerceptionConfig,
    PerceptionAttention,
)
from .prediction_module import PredictionTargets


class PEMLoopOutput(NamedTuple):
    """Output from a single PEM loop step."""
    predictions: Dict[str, torch.Tensor]  # immediate/shortterm/longterm
    surprises: Dict[str, Dict[str, torch.Tensor]]  # {scale: {magnitude, direction, raw}}
    observation: torch.Tensor  # (B, S, D) attended observation
    attention_weights: torch.Tensor  # (B, H, S, S_kv)
    ctm_output: CTMPredictionOutput  # Full CTM output for analysis


class PEMLoopState(NamedTuple):
    """State carried between PEM loop iterations."""
    observation: torch.Tensor  # (B, S, D) last observation
    surprise_magnitude: torch.Tensor  # (B, S, 1) last surprise magnitude
    surprise_direction: torch.Tensor  # (B, S, D) last surprise direction
    cumulative_surprise: torch.Tensor  # (B, S, 1) accumulated surprise


@dataclass
class PEMLoopConfig:
    """Configuration for the PEM experience loop."""

    # Dimensions
    d_model: int = 1536  # Feature dimension (from backbone)
    d_perception: int = 1536  # Perception features (from backbone)

    # CTM Configuration
    d_neurons: int = 512
    d_sync_out: int = 256
    d_sync_action: int = 256
    M: int = 16  # Pre-activation history
    T: int = 8   # Internal ticks
    synapse_hidden: int = 1024
    nlm_hidden: int = 64
    n_attention_heads: int = 8

    # Prediction horizons
    immediate_horizon: int = 8
    shortterm_horizon: int = 64
    longterm_horizon: int = 256

    # Surprise configuration
    surprise_hidden_dim: int = 1536
    surprise_n_layers: int = 2

    # Perception attention configuration
    perception_n_heads: int = 8
    sync_pairs: int = 512
    num_oscillators: int = 32

    # Loop configuration
    surprise_scale: float = 1.0  # How much surprise modulates attention
    cumulative_surprise_decay: float = 0.9  # Decay for accumulated surprise

    dropout: float = 0.0

    def to_ctm_config(self) -> CTMPredictionConfig:
        """Convert to CTMPredictionConfig."""
        return CTMPredictionConfig(
            d_model=self.d_model,
            d_neurons=self.d_neurons,
            d_sync_out=self.d_sync_out,
            d_sync_action=self.d_sync_action,
            M=self.M,
            T=self.T,
            synapse_hidden=self.synapse_hidden,
            nlm_hidden=self.nlm_hidden,
            n_attention_heads=self.n_attention_heads,
            immediate_horizon=self.immediate_horizon,
            shortterm_horizon=self.shortterm_horizon,
            longterm_horizon=self.longterm_horizon,
            dropout=self.dropout,
        )

    def to_surprise_config(self) -> SurpriseConfig:
        """Convert to SurpriseConfig."""
        return SurpriseConfig(
            d_model=self.d_model,
            hidden_dim=self.surprise_hidden_dim,
            n_layers=self.surprise_n_layers,
            dropout=self.dropout,
        )

    def to_perception_config(self) -> PerceptionConfig:
        """Convert to PerceptionConfig."""
        return PerceptionConfig(
            d_model=self.d_model,
            d_perception=self.d_perception,
            n_heads=self.perception_n_heads,
            sync_pairs=self.sync_pairs,
            num_oscillators=self.num_oscillators,
            dropout=self.dropout,
        )


class PEMLoop(nn.Module):
    """
    The core PEM experience loop.

    Connects prediction, surprise, and attention into a unified
    processing loop that implements predictive processing.

    Usage:
        # Initialize
        pem = PEMLoop(config)

        # Process features (e.g., from Janus Pro)
        # Option 1: Single step
        output = pem(features, targets)

        # Option 2: Multi-step loop with state
        state = pem.init_state(features)
        for step in range(num_steps):
            output, state = pem.step(features, targets, state)
    """

    def __init__(self, config: PEMLoopConfig):
        super().__init__()
        self.config = config

        # 1. CTM Prediction Module
        self.prediction = CTMPrediction(config.to_ctm_config())

        # 2. Surprise Module
        self.surprise = SurpriseModule(config.to_surprise_config())

        # 3. Perception Attention (uses surprise to modulate attention)
        self.perception = PerceptionAttention(config.to_perception_config())

        # 4. Target computer
        self.target_computer = PredictionTargets(
            immediate_horizon=config.immediate_horizon,
            shortterm_horizon=config.shortterm_horizon,
            longterm_horizon=config.longterm_horizon,
        )

        # 5. CTM Loss
        self.ctm_loss = CTMLoss()

        # 6. Projections for loop
        # Project observation back to feature space for next prediction
        self.observation_to_features = nn.Linear(config.d_model, config.d_model)

        # Combine original features with observation
        self.feature_combiner = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Personality and intention (simplified - just learned embeddings for now)
        # In full system, these come from SyncModule
        self.personality = nn.Parameter(torch.randn(config.d_model) * 0.02)
        self.intention = nn.Parameter(torch.randn(config.d_model) * 0.02)

        # Sync (simplified - project from prediction sync)
        self.sync_proj = nn.Linear(config.d_sync_out, config.sync_pairs)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in [self.observation_to_features, self.feature_combiner]:
            for m in module.modules() if hasattr(module, 'modules') else [module]:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def init_state(self, features: torch.Tensor) -> PEMLoopState:
        """
        Initialize loop state from features.

        Args:
            features: (B, S, D) input features

        Returns:
            Initial PEMLoopState
        """
        B, S, D = features.shape
        device = features.device

        return PEMLoopState(
            observation=features,  # Start with raw features
            surprise_magnitude=torch.zeros(B, S, 1, device=device),
            surprise_direction=torch.zeros(B, S, D, device=device),
            cumulative_surprise=torch.zeros(B, S, 1, device=device),
        )

    def step(
        self,
        features: torch.Tensor,  # (B, S, D) original features
        targets: Optional[Dict[str, torch.Tensor]] = None,
        state: Optional[PEMLoopState] = None,
        tick: int = 0,
        return_all_ticks: bool = True,
    ) -> Tuple[PEMLoopOutput, PEMLoopState]:
        """
        Execute one step of the PEM loop.

        Args:
            features: Original input features from backbone
            targets: Prediction targets (computed if None)
            state: Previous loop state (initialized if None)
            tick: Current tick for oscillation phase
            return_all_ticks: Whether to return all CTM tick outputs

        Returns:
            output: PEMLoopOutput with predictions, surprises, observation
            new_state: Updated PEMLoopState for next iteration
        """
        B, S, D = features.shape
        device = features.device

        # Initialize state if needed
        if state is None:
            state = self.init_state(features)

        # Compute targets if not provided
        if targets is None:
            targets = self.target_computer.compute_targets_efficient(features)

        # 1. Combine original features with previous observation
        # This lets the loop build on what it attended to
        combined_input = torch.cat([features, state.observation], dim=-1)
        loop_features = self.feature_combiner(combined_input)

        # 2. CTM Prediction
        ctm_output = self.prediction(loop_features, return_all_ticks=return_all_ticks)
        predictions = ctm_output.predictions

        # 3. Compute surprise
        surprises = self.surprise(predictions, targets, features)

        # Get aggregate surprise (use immediate scale as primary)
        if 'immediate' in surprises:
            surprise_mag = surprises['immediate']['magnitude']
            surprise_dir = surprises['immediate']['direction']
        else:
            # Fallback
            first_scale = next(iter(surprises.keys()))
            surprise_mag = surprises[first_scale]['magnitude']
            surprise_dir = surprises[first_scale]['direction']

        # 4. Update cumulative surprise (exponential decay)
        cumulative = (
            self.config.cumulative_surprise_decay * state.cumulative_surprise +
            (1 - self.config.cumulative_surprise_decay) * surprise_mag
        )

        # 5. Perception attention modulated by surprise
        # Cache perception features
        self.perception.cache_perception(features)

        # Build personality/intention signals
        personality_signal = self.personality.unsqueeze(0).unsqueeze(0).expand(B, S, -1)
        intention_signal = self.intention.unsqueeze(0).unsqueeze(0).expand(B, S, -1)

        # Build sync from CTM sync matrix
        # Flatten sync matrix to get sync pairs
        sync_matrix = ctm_output.sync_matrix  # (B, S, D_neurons, D_neurons)
        B_s, S_s, D_n, _ = sync_matrix.shape
        # Take upper triangle elements
        sync_flat = sync_matrix.reshape(B_s, S_s, -1)[:, :, :self.config.d_sync_out]
        sync = self.sync_proj(sync_flat)  # (B, S, sync_pairs)

        # Attend with surprise modulation
        perception_output = self.perception(
            state=state.observation,
            personality_signal=personality_signal,
            intention_signal=intention_signal,
            sync=sync,
            tick=tick,
            surprise_magnitude=surprise_mag * self.config.surprise_scale,
            surprise_direction=surprise_dir,
            causal=False,  # Allow full attention for now
        )

        observation = perception_output.observation
        attention_weights = perception_output.attention_weights

        # Clear perception cache
        self.perception.clear_cache()

        # 6. Build output and new state
        output = PEMLoopOutput(
            predictions=predictions,
            surprises=surprises,
            observation=observation,
            attention_weights=attention_weights,
            ctm_output=ctm_output,
        )

        new_state = PEMLoopState(
            observation=observation,
            surprise_magnitude=surprise_mag,
            surprise_direction=surprise_dir,
            cumulative_surprise=cumulative,
        )

        return output, new_state

    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        num_steps: int = 1,
        return_all_ticks: bool = True,
    ) -> Tuple[List[PEMLoopOutput], PEMLoopState]:
        """
        Run the PEM loop for multiple steps.

        Args:
            features: (B, S, D) input features
            targets: Prediction targets (computed if None)
            num_steps: Number of loop iterations
            return_all_ticks: Whether to return all CTM tick outputs

        Returns:
            outputs: List of PEMLoopOutput for each step
            final_state: Final PEMLoopState
        """
        if targets is None:
            targets = self.target_computer.compute_targets_efficient(features)

        state = self.init_state(features)
        outputs = []

        for step in range(num_steps):
            output, state = self.step(
                features=features,
                targets=targets,
                state=state,
                tick=step,
                return_all_ticks=return_all_ticks,
            )
            outputs.append(output)

        return outputs, state

    def compute_loss(
        self,
        outputs: List[PEMLoopOutput],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss across all loop steps.

        The loss combines:
        1. CTM prediction loss (at each step)
        2. Surprise calibration (learned magnitude should track raw)
        3. Attention efficiency (encourage focused attention)

        Args:
            outputs: List of PEMLoopOutput from forward()
            targets: Prediction targets

        Returns:
            total_loss: Scalar loss
            loss_dict: Breakdown by component
        """
        total_loss = torch.tensor(0.0, device=outputs[0].predictions['immediate'].device)
        loss_dict = {}

        for step_idx, output in enumerate(outputs):
            # 1. CTM prediction loss
            step_loss, step_loss_dict = self.ctm_loss(
                output.ctm_output.all_outputs,
                targets,
                self.prediction.readout_immediate,
                self.prediction.readout_shortterm,
                self.prediction.readout_longterm,
            )

            total_loss = total_loss + step_loss

            # Add step prefix to loss dict
            for k, v in step_loss_dict.items():
                loss_dict[f'step{step_idx}_{k}'] = v

            # 2. Surprise should be meaningful (magnitude correlates with actual error)
            # This is a soft regularization
            for scale, surprise_data in output.surprises.items():
                mag = surprise_data['magnitude']
                raw = surprise_data['raw']

                # Magnitude should track raw surprise
                surprise_calibration = F.mse_loss(mag, raw)
                loss_dict[f'step{step_idx}_{scale}_surprise_cal'] = surprise_calibration.detach()
                total_loss = total_loss + 0.1 * surprise_calibration

            # 3. Attention should become more focused over steps
            # (later steps should have lower entropy attention)
            if step_idx > 0:
                attn = output.attention_weights  # (B, H, S, S_kv)
                # Compute attention entropy
                attn_entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1).mean()
                loss_dict[f'step{step_idx}_attn_entropy'] = attn_entropy.detach()
                # Encourage lower entropy (more focused) at later steps
                total_loss = total_loss + 0.01 * attn_entropy

        # Average across steps
        num_steps = len(outputs)
        total_loss = total_loss / num_steps
        loss_dict['loss'] = total_loss.detach()

        return total_loss, loss_dict


def create_pem_loop(
    d_model: int = 1536,
    d_neurons: int = 512,
    T: int = 8,
    **kwargs,
) -> PEMLoop:
    """Factory function to create PEM loop."""
    config = PEMLoopConfig(
        d_model=d_model,
        d_neurons=d_neurons,
        T=T,
        **kwargs,
    )
    return PEMLoop(config)
