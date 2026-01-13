"""
Activation Module - Arousal/Engagement dimension of experience.

Arousal determines HOW INTENSELY to engage with an experience:
- High arousal: Heightened attention, more processing, vivid memory
- Low arousal: Relaxed attention, less processing, weaker memory

Arousal is computed from:
- Surprise magnitude (unexpected → arousing)
- Valence extremity (very good OR very bad → arousing)
- Novelty (never seen before → arousing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, NamedTuple
import math


@dataclass
class ActivationConfig:
    """Configuration for the Activation (Arousal) module."""

    d_model: int = 1536
    hidden_dim: Optional[int] = None  # Defaults to d_model // 4

    # Arousal computation
    n_layers: int = 2
    use_context: bool = True  # Context-dependent arousal
    dropout: float = 0.0

    # Component weights (learned, but initialized here)
    surprise_weight: float = 0.4
    valence_weight: float = 0.3
    novelty_weight: float = 0.3

    # Temporal smoothing
    use_temporal_smoothing: bool = True
    smoothing_alpha: float = 0.3  # EMA smoothing factor

    # Modulation ranges
    tick_multiplier_range: Tuple[float, float] = (0.5, 2.0)  # fewer to more ticks
    attention_temperature_range: Tuple[float, float] = (0.5, 2.0)  # sharp to broad
    memory_strength_range: Tuple[float, float] = (0.5, 2.0)  # weak to strong

    def __post_init__(self):
        if self.hidden_dim is None:
            self.hidden_dim = self.d_model // 4


class ActivationOutput(NamedTuple):
    """Output from the Activation module."""

    arousal: torch.Tensor  # (B, S, 1) arousal level in [0, 1]
    tick_multiplier: torch.Tensor  # (B, S, 1) multiplier for number of ticks
    attention_temperature: torch.Tensor  # (B, S, 1) temperature for attention
    memory_strength: torch.Tensor  # (B, S, 1) encoding strength for memory
    raw_components: Optional[torch.Tensor] = None  # (B, S, 3) surprise/valence/novelty contributions


class ArousalComputer(nn.Module):
    """
    Computes arousal from surprise, valence extremity, and novelty.

    Arousal is high when:
    - Surprise is high (unexpected events demand attention)
    - Valence is extreme (very good or very bad)
    - Novelty is high (unfamiliar requires more processing)
    """

    def __init__(self, config: ActivationConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        # Learned component weights
        self.component_weights = nn.Parameter(
            torch.tensor([config.surprise_weight, config.valence_weight, config.novelty_weight])
        )

        # Optional context modulation
        if config.use_context:
            self.context_modulator = nn.Sequential(
                nn.Linear(config.d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_dim, 3),  # Modulate each component
                nn.Sigmoid(),
            )
        else:
            self.context_modulator = None

        # Final arousal network
        layers = []
        in_dim = 3  # surprise, |valence|, novelty
        for i in range(config.n_layers):
            out_dim = hidden_dim if i < config.n_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < config.n_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(config.dropout))
            in_dim = out_dim
        layers.append(nn.Sigmoid())  # Output in [0, 1]

        self.arousal_net = nn.Sequential(*layers)

    def forward(
        self,
        surprise_magnitude: torch.Tensor,  # (B, S, 1)
        valence: torch.Tensor,  # (B, S, 1)
        novelty: Optional[torch.Tensor] = None,  # (B, S, 1)
        context: Optional[torch.Tensor] = None,  # (B, S, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute arousal from input components.

        Returns:
            arousal: (B, S, 1) arousal level in [0, 1]
            raw_components: (B, S, 3) weighted component contributions
        """
        B, S, _ = surprise_magnitude.shape

        # Valence extremity (absolute value)
        valence_extremity = valence.abs()

        # Default novelty to 0.5 if not provided
        if novelty is None:
            novelty = torch.full_like(surprise_magnitude, 0.5)

        # Stack components: (B, S, 3)
        components = torch.cat([surprise_magnitude, valence_extremity, novelty], dim=-1)

        # Get weights (normalized)
        weights = F.softmax(self.component_weights, dim=0)  # (3,)

        # Context-dependent weight modulation
        if self.context_modulator is not None and context is not None:
            context_mod = self.context_modulator(context)  # (B, S, 3)
            # Modulate weights per position
            weights = weights.unsqueeze(0).unsqueeze(0) * (0.5 + context_mod)  # (B, S, 3)
            weights = weights / weights.sum(dim=-1, keepdim=True)  # Renormalize
        else:
            weights = weights.unsqueeze(0).unsqueeze(0).expand(B, S, -1)

        # Weighted components
        weighted_components = components * weights  # (B, S, 3)

        # Compute arousal
        arousal = self.arousal_net(weighted_components)  # (B, S, 1)

        return arousal, weighted_components


class TemporalSmoother(nn.Module):
    """
    Smooths arousal over time using exponential moving average.

    Prevents jarring transitions in activation level.
    """

    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("prev_arousal", None)

    def forward(self, arousal: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal smoothing.

        Args:
            arousal: (B, S, 1) current arousal

        Returns:
            smoothed: (B, S, 1) temporally smoothed arousal
        """
        if self.prev_arousal is None or self.prev_arousal.shape != arousal.shape:
            self.prev_arousal = arousal.detach().clone()
            return arousal

        # EMA: smoothed = alpha * current + (1 - alpha) * previous
        smoothed = self.alpha * arousal + (1 - self.alpha) * self.prev_arousal
        self.prev_arousal = smoothed.detach().clone()

        return smoothed

    def reset(self):
        """Reset temporal state."""
        self.prev_arousal = None


class ModulationComputer(nn.Module):
    """
    Computes modulation signals from arousal level.

    Maps arousal [0, 1] to various modulation ranges.
    """

    def __init__(self, config: ActivationConfig):
        super().__init__()
        self.config = config

        # Small networks to compute each modulation
        # This allows non-linear mappings from arousal to modulation

        hidden = 32

        self.tick_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

        self.attention_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

        self.memory_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, arousal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute modulation signals from arousal.

        Args:
            arousal: (B, S, 1) arousal level in [0, 1]

        Returns:
            tick_multiplier: (B, S, 1) in tick_multiplier_range
            attention_temperature: (B, S, 1) in attention_temperature_range
            memory_strength: (B, S, 1) in memory_strength_range
        """
        # Tick multiplier: more arousal → more ticks
        tick_raw = self.tick_net(arousal)  # (B, S, 1) in [0, 1]
        tick_min, tick_max = self.config.tick_multiplier_range
        tick_multiplier = tick_min + (tick_max - tick_min) * tick_raw

        # Attention temperature: more arousal → LOWER temperature (sharper focus)
        # So we invert: high arousal → low temperature
        attn_raw = self.attention_net(arousal)  # (B, S, 1) in [0, 1]
        attn_min, attn_max = self.config.attention_temperature_range
        # Invert: high arousal (1) → min temperature, low arousal (0) → max temperature
        attention_temperature = attn_max - (attn_max - attn_min) * attn_raw

        # Memory strength: more arousal → stronger encoding
        mem_raw = self.memory_net(arousal)  # (B, S, 1) in [0, 1]
        mem_min, mem_max = self.config.memory_strength_range
        memory_strength = mem_min + (mem_max - mem_min) * mem_raw

        return tick_multiplier, attention_temperature, memory_strength


class ActivationModule(nn.Module):
    """
    Activation (Arousal) module for the Predictive Experience Machine.

    Computes how intensely to engage with an experience based on:
    - Surprise magnitude
    - Valence extremity
    - Novelty

    Outputs modulation signals for:
    - Tick count (how much processing)
    - Attention temperature (how focused)
    - Memory strength (how vivid)
    """

    def __init__(self, config: ActivationConfig):
        super().__init__()
        self.config = config

        # Core arousal computation
        self.arousal_computer = ArousalComputer(config)

        # Temporal smoothing
        if config.use_temporal_smoothing:
            self.smoother = TemporalSmoother(config.smoothing_alpha)
        else:
            self.smoother = None

        # Modulation computation
        self.modulation = ModulationComputer(config)

    def forward(
        self,
        surprise_magnitude: torch.Tensor,  # (B, S, 1)
        valence: Optional[torch.Tensor] = None,  # (B, S, 1)
        novelty: Optional[torch.Tensor] = None,  # (B, S, 1)
        context: Optional[torch.Tensor] = None,  # (B, S, D)
    ) -> ActivationOutput:
        """
        Compute activation/arousal and modulation signals.

        Args:
            surprise_magnitude: (B, S, 1) surprise magnitude
            valence: (B, S, 1) valence in [-1, 1], optional
            novelty: (B, S, 1) novelty in [0, 1], optional
            context: (B, S, D) context features, optional

        Returns:
            ActivationOutput with arousal and modulation signals
        """
        B, S, _ = surprise_magnitude.shape

        # Default valence to 0 (neutral) if not provided
        if valence is None:
            valence = torch.zeros_like(surprise_magnitude)

        # Compute raw arousal
        arousal, raw_components = self.arousal_computer(
            surprise_magnitude=surprise_magnitude,
            valence=valence,
            novelty=novelty,
            context=context,
        )

        # Apply temporal smoothing if enabled
        if self.smoother is not None:
            arousal = self.smoother(arousal)

        # Compute modulation signals
        tick_multiplier, attention_temperature, memory_strength = self.modulation(arousal)

        return ActivationOutput(
            arousal=arousal,
            tick_multiplier=tick_multiplier,
            attention_temperature=attention_temperature,
            memory_strength=memory_strength,
            raw_components=raw_components,
        )

    def reset(self):
        """Reset temporal state."""
        if self.smoother is not None:
            self.smoother.reset()

    def get_arousal_only(
        self,
        surprise_magnitude: torch.Tensor,
        valence: Optional[torch.Tensor] = None,
        novelty: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get just the arousal value without modulation signals.

        Useful for simple integration scenarios.
        """
        output = self.forward(surprise_magnitude, valence, novelty, context)
        return output.arousal


class ActivationLoss(nn.Module):
    """
    Loss function for training the Activation module.

    Encourages:
    1. Arousal to correlate with actual processing difficulty
    2. Temporal consistency (smooth transitions)
    3. Appropriate modulation ranges
    """

    def __init__(
        self,
        consistency_weight: float = 0.1,
        range_weight: float = 0.05,
    ):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.range_weight = range_weight

    def forward(
        self,
        activation_output: ActivationOutput,
        processing_difficulty: Optional[torch.Tensor] = None,  # (B, S, 1) target
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute activation loss.

        Args:
            activation_output: Output from ActivationModule
            processing_difficulty: Optional target for arousal (e.g., from loss magnitude)

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        arousal = activation_output.arousal
        losses = {}

        # 1. Target matching (if provided)
        if processing_difficulty is not None:
            target_loss = F.mse_loss(arousal, processing_difficulty)
            losses["target_loss"] = target_loss
        else:
            target_loss = torch.tensor(0.0, device=arousal.device)
            losses["target_loss"] = target_loss

        # 2. Temporal consistency (encourage smooth changes)
        if arousal.shape[1] > 1:
            arousal_diff = arousal[:, 1:] - arousal[:, :-1]
            consistency_loss = (arousal_diff ** 2).mean()
        else:
            consistency_loss = torch.tensor(0.0, device=arousal.device)
        losses["consistency_loss"] = consistency_loss

        # 3. Range utilization (encourage using full range, avoid collapse)
        arousal_mean = arousal.mean()
        arousal_std = arousal.std()
        # Penalize if mean is too extreme or std is too low
        range_loss = (arousal_mean - 0.5) ** 2 + F.relu(0.1 - arousal_std)
        losses["range_loss"] = range_loss

        # Total loss
        total_loss = (
            target_loss
            + self.consistency_weight * consistency_loss
            + self.range_weight * range_loss
        )

        return total_loss, losses


def create_activation_module(
    d_model: int = 1536,
    hidden_dim: Optional[int] = None,
    use_context: bool = True,
    use_temporal_smoothing: bool = True,
    **kwargs,
) -> ActivationModule:
    """
    Factory function to create an ActivationModule.

    Args:
        d_model: Model dimension
        hidden_dim: Hidden layer dimension
        use_context: Whether to use context for modulation
        use_temporal_smoothing: Whether to smooth arousal over time
        **kwargs: Additional config parameters

    Returns:
        Configured ActivationModule
    """
    config = ActivationConfig(
        d_model=d_model,
        hidden_dim=hidden_dim,
        use_context=use_context,
        use_temporal_smoothing=use_temporal_smoothing,
        **kwargs,
    )
    return ActivationModule(config)
