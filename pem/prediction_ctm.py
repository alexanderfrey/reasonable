"""
Prediction CTM Module - CTM-based prediction using shared base.

Generates predictions at multiple temporal scales using CTM architecture.
Exposes post-activations for global sync.
"""

from dataclasses import dataclass
from typing import Optional, Dict, NamedTuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctm_base import CTMBaseConfig, CTMModule, CTMModuleOutput, RMSNorm


class PredictionCTMOutput(NamedTuple):
    """Output from PredictionCTM."""
    predictions: Dict[str, torch.Tensor]  # {immediate, shortterm, longterm} from final tick
    post_activations: torch.Tensor        # (B, S, D_neurons) for global sync
    sync_matrix: torch.Tensor             # (B, S, D_n, D_n)
    all_tick_outputs: List[torch.Tensor]  # Raw CTM outputs at each tick
    certainty: torch.Tensor               # Confidence at final tick
    all_tick_activations: List[torch.Tensor]  # NLM activations at each tick
    # CTM loss support: predictions at each internal tick
    all_tick_predictions: List[Dict[str, torch.Tensor]]  # [{immediate, shortterm, longterm}, ...] per tick
    all_tick_certainties: List[torch.Tensor]  # Certainty at each tick


@dataclass
class PredictionCTMConfig(CTMBaseConfig):
    """Configuration for Prediction CTM module."""

    # Prediction-specific
    immediate_horizon: int = 8
    shortterm_horizon: int = 64
    longterm_horizon: int = 256

    # Override defaults for prediction task
    d_input: int = 1536
    d_neurons: int = 256
    d_output: int = 1536
    d_sync_out: int = 128
    d_sync_internal: int = 128
    M: int = 8
    T: int = 4
    synapse_hidden: int = 512
    nlm_hidden: int = 32


class PredictionCTM(CTMModule):
    """
    CTM-based prediction module.

    Uses the shared CTM core (Synapse -> NLM -> Sync) and adds:
    - Multi-scale prediction readouts (immediate, shortterm, longterm)
    - Exposes post-activations for global sync

    This is a simplified version compared to the original CTMPrediction,
    designed to work with the global sync architecture.
    """

    def __init__(self, config: PredictionCTMConfig):
        # Initialize with base config
        super().__init__(config)
        self.prediction_config = config

        # Prediction readouts (multi-scale)
        self.readout_immediate = nn.Linear(config.d_output, config.d_output)
        self.readout_shortterm = nn.Linear(config.d_output, config.d_output)
        self.readout_longterm = nn.Linear(config.d_output, config.d_output)

        # Input projection (features -> d_input)
        # Identity if d_input == d_model, otherwise project
        if config.d_input != config.d_output:
            self.input_proj = nn.Linear(config.d_output, config.d_input)
        else:
            self.input_proj = nn.Identity()

        self._init_prediction_weights()

    def _init_prediction_weights(self):
        """Initialize prediction-specific weights."""
        for module in [self.readout_immediate, self.readout_shortterm, self.readout_longterm]:
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def input_projection(self, features: torch.Tensor) -> torch.Tensor:
        """Project backbone features to CTM input space."""
        return self.input_proj(features)

    def output_projection(self, core_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate multi-scale predictions from core output."""
        return {
            'immediate': self.readout_immediate(core_output),
            'shortterm': self.readout_shortterm(core_output),
            'longterm': self.readout_longterm(core_output),
        }

    def forward(self, features: torch.Tensor) -> PredictionCTMOutput:
        """
        Generate predictions from features.

        Args:
            features: (B, S, d_model) from backbone

        Returns:
            PredictionCTMOutput with predictions and post-activations
        """
        # 1. Input projection
        input_features = self.input_projection(features)

        # 2. Run core CTM loop
        post_activations, sync_matrix, output, all_outputs, all_activations = self.core(input_features)

        # 3. Generate multi-scale predictions at EACH tick (for CTM loss)
        all_tick_predictions = []
        all_tick_certainties = []
        for t, tick_output in enumerate(all_outputs):
            tick_preds = self.output_projection(tick_output)
            all_tick_predictions.append(tick_preds)
            # Compute certainty up to this tick
            tick_certainty = self.core.compute_certainty(all_outputs[:t+1])
            all_tick_certainties.append(tick_certainty)

        # Final predictions (from last tick)
        predictions = all_tick_predictions[-1] if all_tick_predictions else self.output_projection(output)
        certainty = all_tick_certainties[-1] if all_tick_certainties else self.core.compute_certainty(all_outputs)

        return PredictionCTMOutput(
            predictions=predictions,
            post_activations=post_activations,
            sync_matrix=sync_matrix,
            all_tick_outputs=all_outputs,
            certainty=certainty,
            all_tick_activations=all_activations,
            all_tick_predictions=all_tick_predictions,
            all_tick_certainties=all_tick_certainties,
        )


def create_prediction_ctm(
    d_model: int = 1536,
    d_neurons: int = 256,
    T: int = 4,
    **kwargs,
) -> PredictionCTM:
    """Factory function to create PredictionCTM."""
    config = PredictionCTMConfig(
        d_input=d_model,
        d_output=d_model,
        d_neurons=d_neurons,
        T=T,
        **kwargs,
    )
    return PredictionCTM(config)
