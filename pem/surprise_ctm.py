"""
Surprise CTM Module - CTM-based surprise computation using shared base.

Computes meaningful, context-dependent surprise using CTM architecture.
The CTM processes the prediction error to produce nuanced surprise signals.

Key insight: Surprise isn't just L2 distance - it's MEANINGFUL difference
computed through internal "thinking" ticks.
"""

from dataclasses import dataclass
from typing import Optional, Dict, NamedTuple, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctm_base import CTMBaseConfig, CTMModule, CTMModuleOutput, RMSNorm


class SurpriseCTMOutput(NamedTuple):
    """Output from SurpriseCTM."""
    magnitude: torch.Tensor           # (B, S, 1) learned surprise magnitude from final tick
    direction: torch.Tensor           # (B, S, D) what was unexpected (unit vector)
    raw: torch.Tensor                 # (B, S, 1) raw cosine distance (target for calibration)
    post_activations: torch.Tensor    # (B, S, D_neurons) for global sync
    sync_matrix: torch.Tensor         # (B, S, D_n, D_n)
    all_tick_outputs: List[torch.Tensor]  # Raw CTM outputs (y_t) at each tick
    certainty: torch.Tensor           # Confidence at final tick
    all_tick_activations: List[torch.Tensor]  # NLM activations at each tick
    # CTM loss: magnitude at each tick (cheap - just a small MLP to scalar)
    all_tick_magnitudes: List[torch.Tensor]  # (B, S, 1) magnitude at each tick


@dataclass
class SurpriseCTMConfig(CTMBaseConfig):
    """Configuration for Surprise CTM module."""

    # Surprise-specific
    d_model: int = 1536        # Feature dimension (predictions/targets)

    # Override defaults for surprise task (smaller than prediction)
    d_input: int = 1536 * 3    # concat(predicted, actual, diff)
    d_neurons: int = 128       # Fewer neurons than prediction
    d_output: int = 1536       # Output dimension for direction
    d_sync_out: int = 64
    d_sync_internal: int = 64
    M: int = 4                 # Shorter history
    T: int = 3                 # Fewer ticks (surprise is simpler than prediction)
    synapse_hidden: int = 256
    nlm_hidden: int = 16


class SurpriseCTM(CTMModule):
    """
    CTM-based surprise module.

    Uses the shared CTM core (Synapse -> NLM -> Sync) to process
    prediction errors and produce nuanced surprise signals.

    The CTM "thinks" about the error to determine:
    - magnitude: HOW surprising (context-dependent)
    - direction: WHAT was unexpected (unit vector)

    Exposes post-activations for global sync.
    """

    def __init__(self, config: SurpriseCTMConfig):
        super().__init__(config)
        self.surprise_config = config

        # Error encoding: concat(predicted, actual, diff) -> d_input
        self.error_encoder = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_input),
            nn.GELU(),
            nn.Linear(config.d_input, config.d_input),
        )

        # Magnitude head: core output -> scalar
        # Note: Using Softplus instead of Sigmoid to avoid vanishing gradients
        # Softplus is smooth, non-saturating, and outputs [0, inf)
        # We scale it to roughly match [0, 1] range for raw_surprise calibration
        self.magnitude_head = nn.Sequential(
            nn.Linear(config.d_output, config.d_output // 4),
            nn.GELU(),
            nn.Linear(config.d_output // 4, 1),
            nn.Softplus(beta=2.0),  # Smooth ReLU, beta=2 makes it steeper near 0
        )

        # Direction head: core output -> unit vector
        self.direction_head = nn.Sequential(
            nn.Linear(config.d_output, config.d_output),
            nn.GELU(),
            nn.Linear(config.d_output, config.d_model),
        )

        self._init_surprise_weights()

    def _init_surprise_weights(self):
        """Initialize surprise-specific weights."""
        for module in [self.error_encoder, self.magnitude_head, self.direction_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def input_projection(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode prediction error for CTM processing.

        Args:
            predicted: (B, S, D) predictions
            actual: (B, S, D) actual features (targets)

        Returns:
            error_encoding: (B, S, d_input) encoded error
        """
        raw_diff = actual - predicted
        error_input = torch.cat([predicted, actual, raw_diff], dim=-1)
        return self.error_encoder(error_input)

    def output_projection(self, core_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate surprise magnitude and direction from core output.

        Args:
            core_output: (B, S, d_output) from CTM core

        Returns:
            Dict with magnitude and direction
        """
        magnitude = self.magnitude_head(core_output)  # (B, S, 1)
        direction = self.direction_head(core_output)  # (B, S, d_model)
        direction = F.normalize(direction, dim=-1)     # Unit vector

        return {
            'magnitude': magnitude,
            'direction': direction,
        }

    def forward(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> SurpriseCTMOutput:
        """
        Compute surprise from prediction error.

        CTM Loss Note:
            For surprise calibration, we need magnitude at each tick to find
            which tick best matches raw_surprise. The magnitude_head is cheap
            (small MLP → scalar), so we compute it at each tick.

        Args:
            predicted: (B, S, D) predictions from PredictionCTM
            actual: (B, S, D) actual features (targets)
            valid_mask: (B, S) optional mask for valid positions

        Returns:
            SurpriseCTMOutput with magnitude, direction, and post-activations
        """
        # 1. Compute raw surprise (cosine distance) - this is the calibration target
        cos_sim = F.cosine_similarity(predicted, actual, dim=-1, eps=1e-8)
        raw_surprise = (1 - cos_sim).unsqueeze(-1)  # (B, S, 1)

        # 2. Input projection (error encoding)
        input_features = self.input_projection(predicted, actual)

        # 3. Run core CTM loop
        core_output = self.core(input_features)
        post_activations = core_output.post_activations
        sync_matrix = core_output.sync_matrix
        output = core_output.output
        all_outputs = core_output.all_outputs
        all_activations = core_output.all_activations

        # 4. Compute magnitude at each tick for CTM loss (cheap operation)
        all_tick_magnitudes = []
        for tick_output in all_outputs:
            tick_mag = self.magnitude_head(tick_output)  # (B, S, 1)
            if valid_mask is not None:
                tick_mag = tick_mag * valid_mask.unsqueeze(-1)
            all_tick_magnitudes.append(tick_mag)

        # 5. Final outputs from last tick
        magnitude = all_tick_magnitudes[-1]
        direction = self.direction_head(output)
        direction = F.normalize(direction, dim=-1)

        # Apply validity mask to raw_surprise
        if valid_mask is not None:
            raw_surprise = raw_surprise * valid_mask.unsqueeze(-1)

        # 6. Compute certainty from full output history
        certainty = self.core.compute_certainty(all_outputs)

        return SurpriseCTMOutput(
            magnitude=magnitude,
            direction=direction,
            raw=raw_surprise,
            post_activations=post_activations,
            sync_matrix=sync_matrix,
            all_tick_outputs=all_outputs,
            certainty=certainty,
            all_tick_activations=all_activations,
            all_tick_magnitudes=all_tick_magnitudes,
        )

    def forward_multiscale(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, SurpriseCTMOutput]:
        """
        Compute surprise for multiple prediction scales.

        Convenience method for compatibility with existing code.

        Args:
            predictions: Dict with 'immediate', 'shortterm', 'longterm'
            targets: Dict with targets and validity masks

        Returns:
            Dict mapping scale -> SurpriseCTMOutput
        """
        surprises = {}

        for scale in ['immediate', 'shortterm', 'longterm']:
            if scale not in predictions:
                continue

            pred = predictions[scale]
            actual = targets[scale]
            valid_mask = targets.get(f'{scale}_valid', None)

            surprises[scale] = self.forward(pred, actual, valid_mask)

        return surprises


def create_surprise_ctm(
    d_model: int = 1536,
    d_neurons: int = 128,
    T: int = 3,
    **kwargs,
) -> SurpriseCTM:
    """Factory function to create SurpriseCTM."""
    config = SurpriseCTMConfig(
        d_model=d_model,
        d_input=d_model * 3,
        d_output=d_model,
        d_neurons=d_neurons,
        T=T,
        **kwargs,
    )
    return SurpriseCTM(config)
