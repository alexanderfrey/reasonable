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

from .ctm_base import CTMBaseConfig, CTMModule, CTMModuleOutput, CTMCoreOutput, RMSNorm


class TokenHead(nn.Module):
    """Token prediction head with bottleneck to reduce parameters.

    Instead of d_output -> vocab_size directly (~154M params for 1536->100000),
    uses d_output -> bottleneck -> vocab_size (~26M params for 1536->256->100000).
    """

    def __init__(self, d_input: int, vocab_size: int, bottleneck: int = 256):
        super().__init__()
        self.proj = nn.Linear(d_input, bottleneck)
        self.head = nn.Linear(bottleneck, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_input) -> (B, S, vocab_size)
        x = self.proj(x)
        x = F.gelu(x)
        return self.head(x)


class PredictionCTMOutput(NamedTuple):
    """Output from PredictionCTM."""
    predictions: Dict[str, torch.Tensor]  # {immediate, shortterm, longterm} from final tick
    post_activations: torch.Tensor        # (B, S, D_neurons) for global sync
    sync_matrix: torch.Tensor             # (B, S, D_n, D_n)
    all_tick_outputs: List[torch.Tensor]  # Raw CTM outputs (y_t) at each tick - used for CTM loss
    certainty: torch.Tensor               # Confidence at final tick
    all_tick_activations: List[torch.Tensor]  # NLM activations at each tick
    token_logits: Optional[Dict[str, torch.Tensor]] = None  # {immediate, shortterm, longterm} token logits
    # World state monitoring
    z_init: Optional[torch.Tensor] = None   # (B, S, d_neurons) z before world state added
    z_world: Optional[torch.Tensor] = None  # (d_neurons,) world state projection


@dataclass
class PredictionCTMConfig(CTMBaseConfig):
    """Configuration for Prediction CTM module."""

    # Prediction-specific
    immediate_horizon: int = 8
    shortterm_horizon: int = 64
    longterm_horizon: int = 256

    # Token prediction (auxiliary task)
    vocab_size: int = 0  # 0 = disabled, >0 = predict token IDs
    token_prediction_weight: float = 0.1  # Weight for token prediction loss
    token_bottleneck: int = 256  # Bottleneck dim to reduce params (d_output -> bottleneck -> vocab)

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

    # World state (inherited from CTMBaseConfig, but explicit here for clarity)
    d_world_state: int = 256     # Dimension of persistent world state
    use_world_state: bool = True # Enable world state initialization


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

        # Prediction readouts (multi-scale) for feature prediction
        self.readout_immediate = nn.Linear(config.d_output, config.d_output)
        self.readout_shortterm = nn.Linear(config.d_output, config.d_output)
        self.readout_longterm = nn.Linear(config.d_output, config.d_output)

        # Token prediction heads (auxiliary task - harder than feature prediction)
        # These predict discrete token IDs, forcing more meaningful representations
        # Uses bottleneck to reduce params: d_output -> bottleneck -> vocab_size
        if config.vocab_size > 0:
            self.token_head_immediate = TokenHead(config.d_output, config.vocab_size, config.token_bottleneck)
            self.token_head_shortterm = TokenHead(config.d_output, config.vocab_size, config.token_bottleneck)
            self.token_head_longterm = TokenHead(config.d_output, config.vocab_size, config.token_bottleneck)
        else:
            self.token_head_immediate = None
            self.token_head_shortterm = None
            self.token_head_longterm = None

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
        # Normalize before readout for consistency with loss computation.
        # This ensures train/inference consistency and prevents magnitude bias.
        core_output_norm = F.normalize(core_output, dim=-1)
        return {
            'immediate': self.readout_immediate(core_output_norm),
            'shortterm': self.readout_shortterm(core_output_norm),
            'longterm': self.readout_longterm(core_output_norm),
        }

    def forward(
        self,
        features: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None,  # (B, S, K, d_model) retrieved memories
        world_state: Optional[torch.Tensor] = None,  # (d_world_state,) persistent world state
    ) -> PredictionCTMOutput:
        """
        Generate predictions from features.

        CTM Loss Note:
            all_tick_outputs contains y_t (raw sync-derived outputs) at each tick.
            These are used directly for CTM loss computation - no need to run
            readout heads at every tick. The readouts are only applied to the
            final tick to produce the actual predictions.

        Args:
            features: (B, S, d_model) from backbone
            memory_context: Optional (B, S, K, d_model) retrieved memories.
                           If provided, CTM cross-attention KV includes both
                           features and memories, allowing dynamic attention.
            world_state: Optional (d_world_state,) persistent world state.
                        If provided, biases z_0 initialization via learned projection.
                        This is the emergent world model influencing initial attention.

        Returns:
            PredictionCTMOutput with predictions and post-activations
        """
        # 1. Input projection
        input_features = self.input_projection(features)

        # Project memory context if provided
        if memory_context is not None:
            B, S, K, D = memory_context.shape
            memory_flat = memory_context.reshape(B * S * K, D)
            memory_proj = self.input_projection(memory_flat)
            memory_context_proj = memory_proj.reshape(B, S, K, -1)
        else:
            memory_context_proj = None

        # 2. Run core CTM loop with optional memory context and world state
        # all_outputs contains y_t at each tick - used for CTM loss
        # world_state biases z_0 to incorporate accumulated sync patterns
        core_output = self.core(
            input_features, memory_context=memory_context_proj, world_state=world_state
        )

        # 3. Generate predictions from FINAL tick only
        # Readout heads are task-specific projections, not part of CTM core
        predictions = self.output_projection(core_output.output)

        # 4. Generate token logits if enabled (auxiliary task)
        if self.token_head_immediate is not None:
            token_logits = {
                'immediate': self.token_head_immediate(core_output.output),
                'shortterm': self.token_head_shortterm(core_output.output),
                'longterm': self.token_head_longterm(core_output.output),
            }
        else:
            token_logits = None

        # 5. Compute certainty from full output history
        certainty = self.core.compute_certainty(core_output.all_outputs)

        return PredictionCTMOutput(
            predictions=predictions,
            post_activations=core_output.post_activations,
            sync_matrix=core_output.sync_matrix,
            all_tick_outputs=core_output.all_outputs,  # y_t values for CTM loss
            certainty=certainty,
            all_tick_activations=core_output.all_activations,
            token_logits=token_logits,
            # World state monitoring
            z_init=core_output.z_init,
            z_world=core_output.z_world,
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
