"""
Memory-Augmented Prediction Module.

Combines PredictionCTM + SemanticMemory + Surprise computation:

```
Features (Janus) ─┬─> MemoryRead() ──────────────────────────┐
                  │                                          │
                  └─> PredictionCTM ─> predictions ──────────┼─> SurpriseCompute ─> surprise
                            │                                │         │
                      post_activations                       │         │
                            │                                │    magnitude
                            └── MemoryIntegrator ────────────┘         │
                                      │                                │
                                      └───────> MemoryWrite(importance=surprise)
```

Key insight: Memory is read BEFORE prediction to augment features,
and written AFTER with importance weighted by surprise.
"""

from dataclasses import dataclass
from typing import Optional, Dict, NamedTuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prediction_ctm import PredictionCTM, PredictionCTMConfig, PredictionCTMOutput
from .semantic_memory import (
    SemanticMemory,
    SemanticMemoryConfig,
    MemoryIntegrator,
    MemoryReadOutput,
)


class SurpriseOutput(NamedTuple):
    """Output from surprise computation."""
    magnitude: torch.Tensor    # (B, S) learned/calibrated surprise magnitude
    raw: torch.Tensor          # (B, S) raw cosine distance (1 - cos_sim)
    direction: torch.Tensor    # (B, S, D) direction of surprise (actual - predicted, normalized)


class SurpriseCompute(nn.Module):
    """
    Simple surprise computation module.

    Computes surprise as the difference between predictions and targets.
    Outputs:
    - raw: 1 - cosine_similarity (range [0, 2])
    - magnitude: learned calibration of raw surprise
    - direction: normalized difference vector
    """

    def __init__(
        self,
        d_model: int = 1536,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model

        # Learned calibration: raw surprise -> calibrated magnitude
        # This allows the model to learn context-dependent surprise weighting
        self.calibration = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),  # concat(diff, raw)
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        predicted: torch.Tensor,  # (B, S, D)
        actual: torch.Tensor,     # (B, S, D)
    ) -> SurpriseOutput:
        """
        Compute surprise between prediction and actual.

        Args:
            predicted: Predicted features (normalized)
            actual: Actual target features (normalized)

        Returns:
            SurpriseOutput with magnitude, raw, and direction
        """
        # Normalize inputs for cosine computation
        pred_norm = F.normalize(predicted, dim=-1)
        actual_norm = F.normalize(actual, dim=-1)

        # Raw cosine-based surprise: 1 - cos_sim
        cos_sim = (pred_norm * actual_norm).sum(dim=-1)  # (B, S)
        raw = 1.0 - cos_sim  # (B, S), range [0, 2]

        # Direction of surprise (what was unexpected)
        diff = actual - predicted  # (B, S, D)
        direction = F.normalize(diff, dim=-1)  # (B, S, D)

        # Learned calibration
        calib_input = torch.cat([diff, raw.unsqueeze(-1)], dim=-1)  # (B, S, D+1)
        magnitude = self.calibration(calib_input).squeeze(-1)  # (B, S)

        return SurpriseOutput(
            magnitude=magnitude,
            raw=raw,
            direction=direction,
        )


class MemoryAugmentedPredictionOutput(NamedTuple):
    """Output from MemoryAugmentedPrediction."""
    predictions: Dict[str, torch.Tensor]  # {immediate, shortterm, longterm}
    post_activations: torch.Tensor        # (B, S, D_neurons) for potential downstream use
    sync_matrix: torch.Tensor             # (B, S, D_n, D_n)
    all_tick_outputs: List[torch.Tensor]  # Raw CTM outputs at each tick
    certainty: torch.Tensor               # Confidence at final tick
    all_tick_activations: List[torch.Tensor]  # NLM activations at each tick

    # Surprise outputs (only present when targets provided)
    surprise: Optional[SurpriseOutput] = None

    # Memory outputs
    memory_retrieved: Optional[torch.Tensor] = None  # (B, S, feature_dim)
    memory_attention: Optional[torch.Tensor] = None  # (B, S, num_slots)
    memory_max_attention: Optional[torch.Tensor] = None  # (B, S)

    # Token logits (if enabled)
    token_logits: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class MemoryAugmentedPredictionConfig:
    """Configuration for memory-augmented prediction."""

    # Prediction CTM config
    d_input: int = 1536
    d_output: int = 1536
    d_neurons: int = 512
    d_sync_out: int = 256
    d_sync_internal: int = 256
    M: int = 16
    T: int = 8
    synapse_hidden: int = 1024
    nlm_hidden: int = 64
    internal_obs_residual: float = 0.1

    # Prediction horizons
    immediate_horizon: int = 8
    shortterm_horizon: int = 64
    longterm_horizon: int = 256

    # Token prediction
    vocab_size: int = 0
    token_prediction_weight: float = 0.5
    token_bottleneck: int = 256

    # Memory config
    memory_slots: int = 2048
    memory_key_dim: int = 256
    memory_value_dim: int = 1536  # Now matches feature_dim for storing target features
    memory_context_window: int = 8  # Local context window for key encoding (total 2*8+1=17 positions)
    memory_retrieval_temperature: float = 0.1
    memory_importance_decay: float = 0.8    # Fast decay to evict stale memories
    memory_write_threshold: float = 0.1

    # Memory integration
    memory_attention_threshold: float = 0.5  # Higher threshold - only use memory when confident

    # Surprise config
    surprise_hidden_dim: int = 256


class MemoryAugmentedPrediction(nn.Module):
    """
    Memory-augmented prediction module.

    Combines:
    1. SemanticMemory - episodic memory with importance-based management
    2. PredictionCTM - CTM-based multi-horizon prediction
    3. SurpriseCompute - surprise computation for memory importance

    Flow:
    1. Read from memory using input features
    2. If memory has relevant content (high attention), augment features
    3. Run prediction on (augmented) features
    4. If targets provided, compute surprise
    5. Write to memory with importance = surprise magnitude
    """

    def __init__(self, config: MemoryAugmentedPredictionConfig):
        super().__init__()
        self.config = config

        # 1. Semantic Memory
        memory_config = SemanticMemoryConfig(
            num_slots=config.memory_slots,
            key_dim=config.memory_key_dim,
            value_dim=config.memory_value_dim,
            feature_dim=config.d_input,
            context_window=config.memory_context_window,
            retrieval_temperature=config.memory_retrieval_temperature,
            importance_decay=config.memory_importance_decay,
            write_threshold=config.memory_write_threshold,
        )
        self.memory = SemanticMemory(memory_config)

        # 2. Memory Integrator
        self.memory_integrator = MemoryIntegrator(
            feature_dim=config.d_input,
            hidden_dim=config.d_input // 2,
        )

        # 3. Prediction CTM
        pred_config = PredictionCTMConfig(
            d_input=config.d_input,
            d_output=config.d_output,
            d_neurons=config.d_neurons,
            d_sync_out=config.d_sync_out,
            d_sync_internal=config.d_sync_internal,
            M=config.M,
            T=config.T,
            synapse_hidden=config.synapse_hidden,
            nlm_hidden=config.nlm_hidden,
            internal_obs_residual=config.internal_obs_residual,
            immediate_horizon=config.immediate_horizon,
            shortterm_horizon=config.shortterm_horizon,
            longterm_horizon=config.longterm_horizon,
            vocab_size=config.vocab_size,
            token_prediction_weight=config.token_prediction_weight,
            token_bottleneck=config.token_bottleneck,
        )
        self.prediction = PredictionCTM(pred_config)

        # 4. Surprise Compute
        self.surprise = SurpriseCompute(
            d_model=config.d_output,
            hidden_dim=config.surprise_hidden_dim,
        )

        # Note: memory_value_dim should now match d_output (feature_dim) since we store
        # target features instead of post_activations

    def forward(
        self,
        features: torch.Tensor,                    # (B, S, d_input)
        targets: Optional[Dict[str, torch.Tensor]] = None,  # For computing surprise
        write_to_memory: bool = True,              # Whether to write to memory
    ) -> MemoryAugmentedPredictionOutput:
        """
        Forward pass with memory augmentation.

        Args:
            features: Input features from backbone
            targets: Optional targets for surprise computation
                     Should have 'immediate' key with (B, S, d_output) tensor
            write_to_memory: Whether to write to memory after prediction

        Returns:
            MemoryAugmentedPredictionOutput with predictions and memory info
        """
        B, S, D = features.shape

        # 1. Read from memory
        memory_read = self.memory.read(features)  # MemoryReadOutput

        # 2. Augment features if memory has relevant content
        max_attn = memory_read.max_attention  # (B, S)
        use_memory_mask = max_attn > self.config.memory_attention_threshold

        if use_memory_mask.any():
            # Augment features with memory
            augmented_features = self.memory_integrator(
                features,
                memory_read.retrieved,
                memory_strength=max_attn,
            )
            # Blend: use augmented where memory is strong, original elsewhere
            augmented_features = torch.where(
                use_memory_mask.unsqueeze(-1),
                augmented_features,
                features,
            )
        else:
            augmented_features = features

        # 3. Run prediction
        pred_output: PredictionCTMOutput = self.prediction(augmented_features)

        # 4. Compute surprise (if targets provided)
        surprise_output = None
        if targets is not None and 'immediate' in targets:
            # Use immediate predictions vs targets for surprise
            surprise_output = self.surprise(
                predicted=pred_output.predictions['immediate'],
                actual=targets['immediate'],
            )

            # 5. Write to memory with importance = surprise magnitude
            if write_to_memory:
                # Store TARGET FEATURES (what actually came next) - enables learning
                # "Given context X, outcome Y happened" -> better predictions
                self.memory.write(
                    features=features,
                    values=targets['immediate'],  # Now storing targets, not post_activations
                    importance_scores=surprise_output.magnitude,
                )

        return MemoryAugmentedPredictionOutput(
            predictions=pred_output.predictions,
            post_activations=pred_output.post_activations,
            sync_matrix=pred_output.sync_matrix,
            all_tick_outputs=pred_output.all_tick_outputs,
            certainty=pred_output.certainty,
            all_tick_activations=pred_output.all_tick_activations,
            surprise=surprise_output,
            memory_retrieved=memory_read.retrieved,
            memory_attention=memory_read.attention_weights,
            memory_max_attention=memory_read.max_attention,
            token_logits=pred_output.token_logits,
        )

    def decay_memory_importance(self):
        """Decay memory importance (call once per batch)."""
        self.memory.decay_importance()

    def get_memory_stats(self) -> dict:
        """Get memory statistics for logging."""
        return self.memory.get_stats()

    def reset_memory(self):
        """Clear all memory contents."""
        self.memory.reset()

    def refresh_memory_keys(self) -> int:
        """
        Re-encode all stored memory keys with current key_encoder.

        Call periodically during training to update old memories with
        the improved key_encoder. Returns number of keys refreshed.
        """
        return self.memory.refresh_keys()


def create_memory_augmented_prediction(
    d_model: int = 1536,
    d_neurons: int = 512,
    T: int = 8,
    memory_slots: int = 2048,
    memory_context_window: int = 8,
    vocab_size: int = 0,
    **kwargs,
) -> MemoryAugmentedPrediction:
    """Factory function to create memory-augmented prediction module."""
    config = MemoryAugmentedPredictionConfig(
        d_input=d_model,
        d_output=d_model,
        d_neurons=d_neurons,
        T=T,
        memory_slots=memory_slots,
        memory_context_window=memory_context_window,
        vocab_size=vocab_size,
        memory_value_dim=d_model,  # Now stores target features (same dim as input)
        **kwargs,
    )
    return MemoryAugmentedPrediction(config)
