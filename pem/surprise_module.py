"""
Surprise Module for PEM.

Compare predictions to reality. Compute meaningful, context-dependent surprise.

Key insight: Surprise isn't just L2 distance - it's MEANINGFUL difference.
- "cat" vs "dog" after "The ___ barked" → HIGH surprise
- "cat" vs "dog" after "The pet was" → LOW surprise
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SurpriseConfig:
    """Configuration for the surprise module."""

    # Feature dimensions
    d_model: int = 1536

    # Architecture
    hidden_dim: Optional[int] = None  # Defaults to d_model
    n_layers: int = 2                 # Depth of encoders
    dropout: float = 0.0

    # Scales to compute surprise for
    scales: Tuple[str, ...] = ('immediate', 'shortterm', 'longterm')

    def __post_init__(self):
        if self.hidden_dim is None:
            self.hidden_dim = self.d_model


class ErrorEncoder(nn.Module):
    """
    Encode prediction error into a meaningful representation.

    Takes (predicted, actual, raw_diff) and produces an error embedding
    that captures what KIND of error occurred.
    """

    def __init__(self, d_model: int, hidden_dim: int, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model

        # Input: concat of [predicted, actual, diff] = 3 * d_model
        layers = []
        in_dim = d_model * 3
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else d_model
            layers.append(nn.Linear(in_dim, out_dim, bias=False))
            if i < n_layers - 1:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        predicted: torch.Tensor,  # (B, S, D)
        actual: torch.Tensor,     # (B, S, D)
    ) -> torch.Tensor:
        """Encode the prediction error."""
        raw_diff = actual - predicted
        error_input = torch.cat([predicted, actual, raw_diff], dim=-1)
        return self.net(error_input)  # (B, S, D)


class ContextGate(nn.Module):
    """
    Context-dependent surprise gating.

    The same prediction error can be more or less surprising
    depending on context. This module learns that distinction.
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        # Input: [error_encoded, context_summary] = 2 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1, bias=False),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        error_encoded: torch.Tensor,  # (B, S, D)
        context: torch.Tensor,        # (B, S, D)
    ) -> torch.Tensor:
        """Compute context-gated surprise magnitude."""
        # Use cumulative mean as context summary at each position
        # Position t sees context from 0..t
        # This is causal - doesn't look ahead
        cumsum = torch.cumsum(context, dim=1)
        positions = torch.arange(1, context.shape[1] + 1, device=context.device).float()
        context_summary = cumsum / positions.view(1, -1, 1)  # (B, S, D)

        gate_input = torch.cat([error_encoded, context_summary], dim=-1)
        return self.net(gate_input)  # (B, S, 1)


class DirectionEncoder(nn.Module):
    """
    Encode WHAT was unexpected (not just that something was).

    The direction vector captures the nature of the surprise,
    suitable for storage in memory and injection into sync.
    """

    def __init__(self, d_model: int, hidden_dim: int, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()

        # Input: [error_encoded, raw_diff] = 2 * d_model
        layers = []
        in_dim = d_model * 2
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else d_model
            layers.append(nn.Linear(in_dim, out_dim, bias=False))
            if i < n_layers - 1:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        error_encoded: torch.Tensor,  # (B, S, D)
        raw_diff: torch.Tensor,       # (B, S, D)
    ) -> torch.Tensor:
        """Encode the direction of surprise."""
        direction_input = torch.cat([error_encoded, raw_diff], dim=-1)
        direction = self.net(direction_input)  # (B, S, D)
        # Normalize to unit vector (direction, not magnitude)
        return F.normalize(direction, dim=-1)


class ScaleSurpriseComputer(nn.Module):
    """
    Compute surprise for a single prediction scale.

    Combines error encoding, context gating, and direction encoding.
    """

    def __init__(self, config: SurpriseConfig):
        super().__init__()
        self.config = config

        self.error_encoder = ErrorEncoder(
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )

        self.context_gate = ContextGate(
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

        self.direction_encoder = DirectionEncoder(
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )

    def forward(
        self,
        predicted: torch.Tensor,  # (B, S, D)
        actual: torch.Tensor,     # (B, S, D)
        context: torch.Tensor,    # (B, S, D)
        valid_mask: Optional[torch.Tensor] = None,  # (B, S)
    ) -> Dict[str, torch.Tensor]:
        """
        Compute surprise for this scale.

        Returns:
            Dict with:
                'magnitude': (B, S, 1) learned context-dependent surprise
                'direction': (B, S, D) what was unexpected (unit vector)
                'raw': (B, S, 1) raw cosine distance (1 - cos_sim)
        """
        # Raw cosine-based surprise
        cos_sim = F.cosine_similarity(predicted, actual, dim=-1, eps=1e-8)
        raw_surprise = (1 - cos_sim).unsqueeze(-1)  # (B, S, 1)

        # Encoded error
        error_encoded = self.error_encoder(predicted, actual)  # (B, S, D)

        # Context-gated magnitude
        magnitude = self.context_gate(error_encoded, context)  # (B, S, 1)

        # Direction of surprise
        raw_diff = actual - predicted
        direction = self.direction_encoder(error_encoded, raw_diff)  # (B, S, D)

        # Apply validity mask if provided
        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1)  # (B, S, 1)
            magnitude = magnitude * mask
            raw_surprise = raw_surprise * mask
            # Direction stays as-is (will be masked when used)

        return {
            'magnitude': magnitude,
            'direction': direction,
            'raw': raw_surprise,
        }


class SurpriseModule(nn.Module):
    """
    Multi-scale surprise computation for PEM.

    Computes surprise at each prediction scale (immediate, shortterm, longterm),
    outputting both learned context-dependent surprise and raw cosine distance.

    Usage:
        surprises = surprise_module(predictions, targets, features)
        # surprises['immediate']['magnitude'] -> (B, S, 1)
        # surprises['immediate']['direction'] -> (B, S, D)
        # surprises['immediate']['raw'] -> (B, S, 1)
    """

    def __init__(self, config: SurpriseConfig):
        super().__init__()
        self.config = config

        # Create a surprise computer for each scale
        self.scale_computers = nn.ModuleDict({
            scale: ScaleSurpriseComputer(config)
            for scale in config.scales
        })

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],  # From PredictionModule
        targets: Dict[str, torch.Tensor],       # From PredictionTargets
        context: torch.Tensor,                  # Features from FeatureExtractor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute multi-scale surprise.

        Args:
            predictions: Dict with 'immediate', 'shortterm', 'longterm' predictions
            targets: Dict with targets and validity masks from PredictionTargets
            context: Feature context from FeatureExtractor (B, S, D)

        Returns:
            Dict mapping scale -> {'magnitude', 'direction', 'raw'}
        """
        surprises = {}

        for scale in self.config.scales:
            if scale not in predictions:
                continue

            pred = predictions[scale]
            actual = targets[scale]
            valid_mask = targets.get(f'{scale}_valid', None)

            surprises[scale] = self.scale_computers[scale](
                predicted=pred,
                actual=actual,
                context=context,
                valid_mask=valid_mask,
            )

        return surprises

    def compute_single(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
        context: torch.Tensor,
        scale: str = 'immediate',
    ) -> Dict[str, torch.Tensor]:
        """
        Compute surprise for a single prediction (convenience method).

        Useful for online inference where you have one prediction/actual pair.
        """
        if scale not in self.scale_computers:
            raise ValueError(f"Unknown scale: {scale}. Available: {list(self.scale_computers.keys())}")

        return self.scale_computers[scale](predicted, actual, context)


class SurpriseLoss(nn.Module):
    """
    Loss for training the surprise module.

    Self-supervised approach: surprise magnitude should correlate with
    prediction difficulty (entropy/uncertainty of the prediction task).

    The intuition:
    - If many things could follow → low surprise for any of them
    - If one thing strongly expected → high surprise for deviations
    """

    def __init__(
        self,
        calibration_weight: float = 1.0,
        direction_weight: float = 0.5,
        raw_correlation_weight: float = 0.3,
    ):
        super().__init__()
        self.calibration_weight = calibration_weight
        self.direction_weight = direction_weight
        self.raw_correlation_weight = raw_correlation_weight

    def forward(
        self,
        surprises: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute surprise calibration loss.

        Args:
            surprises: Output from SurpriseModule
            targets: Targets with validity masks

        Returns:
            total_loss: Scalar loss
            loss_dict: Breakdown by component
        """
        loss_dict = {}
        total_loss = 0.0

        for scale, surprise_data in surprises.items():
            magnitude = surprise_data['magnitude']  # (B, S, 1)
            direction = surprise_data['direction']  # (B, S, D)
            raw = surprise_data['raw']              # (B, S, 1)

            valid_mask = targets.get(f'{scale}_valid', None)

            # 1. Calibration: learned magnitude should correlate with raw
            # (but can be more nuanced due to context gating)
            if valid_mask is not None:
                valid = valid_mask.unsqueeze(-1)
                mag_valid = magnitude[valid.expand_as(magnitude)].view(-1)
                raw_valid = raw[valid.expand_as(raw)].view(-1)
            else:
                mag_valid = magnitude.view(-1)
                raw_valid = raw.view(-1)

            if mag_valid.numel() > 1:
                # Correlation loss: encourage positive correlation with raw
                # Using MSE as a soft constraint (magnitude should track raw)
                calibration_loss = F.mse_loss(mag_valid, raw_valid)
                loss_dict[f'{scale}_calibration'] = calibration_loss.detach()
                total_loss = total_loss + self.calibration_weight * calibration_loss

            # 2. Direction should point toward actual (reconstruction signal)
            # The direction + predicted should approximate actual
            pred = targets.get(f'{scale}', None)
            if pred is not None and valid_mask is not None:
                # This is a soft constraint - direction helps reconstruct
                # actual = predicted + alpha * direction (for some alpha)
                # We just check that direction points in the right general direction
                actual = targets[scale]
                diff = actual - pred  # What we need to add to predicted
                diff_norm = F.normalize(diff, dim=-1)

                # Direction should align with normalized diff
                dir_valid = direction[valid_mask]
                diff_valid = diff_norm[valid_mask]

                if dir_valid.numel() > 0:
                    alignment = F.cosine_similarity(dir_valid, diff_valid, dim=-1)
                    direction_loss = (1 - alignment).mean()
                    loss_dict[f'{scale}_direction'] = direction_loss.detach()
                    total_loss = total_loss + self.direction_weight * direction_loss

        return total_loss, loss_dict


@dataclass
class ValenceConfig:
    """Configuration for the valence module.

    Valence = the affective dimension of experience (good vs bad).
    Measures alignment between surprise and personality goals.
    """

    d_model: int = 1536              # Feature dimension
    personality_dim: int = 512       # Personality embedding dimension
    hidden_dim: Optional[int] = None # Hidden layer size (defaults to d_model // 2)
    n_layers: int = 2                # Depth of alignment network
    use_context: bool = True         # Context-dependent valence
    dropout: float = 0.0

    def __post_init__(self):
        if self.hidden_dim is None:
            self.hidden_dim = self.d_model // 2


class PersonalityProjector(nn.Module):
    """
    Project personality embedding into surprise direction space.

    Personality lives in a different representational space than surprise.
    This module learns the mapping so we can compute alignment.
    """

    def __init__(self, personality_dim: int, d_model: int, hidden_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(personality_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, personality: torch.Tensor) -> torch.Tensor:
        """
        Args:
            personality: (D_p,) or (B, D_p) personality embedding

        Returns:
            projected: (D,) or (B, D) in surprise direction space
        """
        return self.net(personality)


class ContextualGoalModulator(nn.Module):
    """
    Context-dependent goal interpretation.

    The same personality goal can mean different things in different contexts.
    "Be helpful" means different things when reading code vs poetry.

    This module learns how context modulates the effective goal representation.
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        # Input: [projected_personality, context] = 2 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, d_model),
        )

        # Residual gate: how much should context modulate?
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
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
        personality_proj: torch.Tensor,  # (B, S, D) or (B, D)
        context: torch.Tensor,           # (B, S, D)
    ) -> torch.Tensor:
        """
        Modulate goal representation based on context.

        Returns:
            modulated: (B, S, D) context-dependent goal direction
        """
        # Expand personality if needed
        if personality_proj.dim() == 2:
            personality_proj = personality_proj.unsqueeze(1).expand(-1, context.shape[1], -1)

        combined = torch.cat([personality_proj, context], dim=-1)

        # Compute modulation
        modulation = self.net(combined)
        gate = self.gate(combined)

        # Residual connection with learned gate
        modulated = personality_proj + gate * modulation

        # Normalize to unit vector (it's a direction)
        return F.normalize(modulated, dim=-1)


class AlignmentComputer(nn.Module):
    """
    Compute alignment between surprise direction and goal direction.

    Goes beyond simple cosine similarity to learn nuanced alignment:
    - Some surprise directions are more important than others
    - Alignment might be non-linear (small deviations OK, large ones bad)
    """

    def __init__(self, d_model: int, hidden_dim: int, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()

        # Input: [surprise_direction, goal_direction, element_wise_product]
        # The element-wise product captures interaction patterns
        layers = []
        in_dim = d_model * 3

        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim, bias=True))
            if i < n_layers - 1:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        layers.append(nn.Tanh())  # Output in [-1, +1]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        surprise_direction: torch.Tensor,  # (B, S, D) unit vector
        goal_direction: torch.Tensor,      # (B, S, D) unit vector
    ) -> torch.Tensor:
        """
        Compute valence as learned alignment.

        Returns:
            valence: (B, S, 1) in [-1, +1]
                +1 = surprise moves toward goals (positive)
                -1 = surprise moves away from goals (negative)
                 0 = surprise is orthogonal to goals (neutral)
        """
        # Element-wise product captures interaction
        interaction = surprise_direction * goal_direction

        # Concatenate all inputs
        alignment_input = torch.cat([
            surprise_direction,
            goal_direction,
            interaction,
        ], dim=-1)

        return self.net(alignment_input)


class ValenceModule(nn.Module):
    """
    Compute affective valence of surprise: is it good or bad?

    Core equation:
        valence = alignment(surprise_direction, personality_goals, context)

    Where:
        - surprise_direction: WHAT was unexpected (unit vector)
        - personality_goals: WHAT the agent wants (from PersonalityModule)
        - context: current situation (modulates goal interpretation)

    Output:
        - valence in [-1, +1]: positive = good for goals, negative = bad

    Usage:
        valence_module = ValenceModule(config)
        valence = valence_module(
            surprise_direction=surprise['direction'],
            personality=sync_module.personality.base_personality,
            context=features,
        )
    """

    def __init__(self, config: ValenceConfig):
        super().__init__()
        self.config = config

        # 1. Project personality to surprise space
        self.personality_proj = PersonalityProjector(
            personality_dim=config.personality_dim,
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
        )

        # 2. Context-dependent goal modulation (optional)
        if config.use_context:
            self.context_modulator = ContextualGoalModulator(
                d_model=config.d_model,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
        else:
            self.context_modulator = None

        # 3. Alignment computation
        self.alignment = AlignmentComputer(
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )

    def forward(
        self,
        surprise_direction: torch.Tensor,  # (B, S, D) from SurpriseModule
        personality: torch.Tensor,          # (D,) or (D_p,) from PersonalityModule
        context: Optional[torch.Tensor] = None,  # (B, S, D) features/context
    ) -> torch.Tensor:
        """
        Compute valence of surprise relative to personality goals.

        Args:
            surprise_direction: Direction of prediction error (unit vector)
            personality: Personality embedding (base_personality from PersonalityModule)
            context: Optional context for goal modulation

        Returns:
            valence: (B, S, 1) in [-1, +1]
        """
        B, S, D = surprise_direction.shape

        # 1. Project personality to surprise direction space
        personality_proj = self.personality_proj(personality)  # (D,) -> (D,)

        # Expand to match surprise shape
        personality_proj = personality_proj.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        personality_proj = personality_proj.expand(B, S, -1)  # (B, S, D)

        # 2. Modulate by context (optional)
        if self.context_modulator is not None and context is not None:
            goal_direction = self.context_modulator(personality_proj, context)
        else:
            goal_direction = F.normalize(personality_proj, dim=-1)

        # 3. Compute alignment (valence)
        valence = self.alignment(surprise_direction, goal_direction)

        return valence

    def compute_from_surprises(
        self,
        surprises: Dict[str, Dict[str, torch.Tensor]],  # From SurpriseModule
        personality: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method to compute valence for all surprise scales.

        Args:
            surprises: Output from SurpriseModule with 'direction' for each scale
            personality: Personality embedding
            context: Feature context

        Returns:
            Dict mapping scale -> valence tensor
        """
        valences = {}

        for scale, surprise_data in surprises.items():
            if 'direction' in surprise_data:
                valences[scale] = self.forward(
                    surprise_direction=surprise_data['direction'],
                    personality=personality,
                    context=context,
                )

        return valences


class ValenceLoss(nn.Module):
    """
    Loss for training the valence module.

    Self-supervised approach based on prediction improvement:
    - If surprise helped improve next prediction → positive valence was correct
    - If surprise hurt next prediction → negative valence was correct

    Also includes consistency losses:
    - Valence should be smooth over time (no random flips)
    - Valence magnitude should correlate with surprise magnitude
    """

    def __init__(
        self,
        consistency_weight: float = 0.3,
        magnitude_correlation_weight: float = 0.2,
    ):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.magnitude_correlation_weight = magnitude_correlation_weight

    def forward(
        self,
        valences: Dict[str, torch.Tensor],       # {scale: (B, S, 1)}
        surprises: Dict[str, Dict[str, torch.Tensor]],  # From SurpriseModule
        valid_masks: Dict[str, torch.Tensor],    # Validity masks
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute valence training loss.

        Returns:
            total_loss: Scalar loss
            loss_dict: Breakdown by component
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=next(iter(valences.values())).device)

        for scale, valence in valences.items():
            valid_mask = valid_masks.get(f'{scale}_valid', None)
            magnitude = surprises[scale]['magnitude']

            # 1. Temporal consistency: valence shouldn't flip randomly
            if valence.shape[1] > 1:
                valence_diff = (valence[:, 1:] - valence[:, :-1]).abs()
                consistency_loss = valence_diff.mean()
                loss_dict[f'{scale}_consistency'] = consistency_loss.detach()
                total_loss = total_loss + self.consistency_weight * consistency_loss

            # 2. Magnitude correlation: high surprise = high |valence|
            # (both very good and very bad things are surprising)
            if valid_mask is not None and valid_mask.any():
                valence_valid = valence.abs()[valid_mask.unsqueeze(-1).expand_as(valence)].view(-1)
                mag_valid = magnitude[valid_mask.unsqueeze(-1).expand_as(magnitude)].view(-1)

                if valence_valid.numel() > 1:
                    # Soft correlation: |valence| should increase with magnitude
                    # Using ranking loss: if mag_i > mag_j, then |val_i| > |val_j|
                    mag_corr_loss = F.mse_loss(
                        valence_valid,
                        mag_valid / (mag_valid.max() + 1e-8)  # Normalize magnitude
                    )
                    loss_dict[f'{scale}_mag_correlation'] = mag_corr_loss.detach()
                    total_loss = total_loss + self.magnitude_correlation_weight * mag_corr_loss

        return total_loss, loss_dict


def create_valence_module(
    d_model: int = 1536,
    personality_dim: int = 512,
    hidden_dim: Optional[int] = None,
    use_context: bool = True,
    **kwargs,
) -> ValenceModule:
    """Factory function to create a valence module."""
    config = ValenceConfig(
        d_model=d_model,
        personality_dim=personality_dim,
        hidden_dim=hidden_dim,
        use_context=use_context,
        **kwargs,
    )
    return ValenceModule(config)


def create_surprise_module(
    d_model: int = 1536,
    hidden_dim: Optional[int] = None,
    n_layers: int = 2,
    scales: Tuple[str, ...] = ('immediate', 'shortterm', 'longterm'),
    **kwargs,
) -> SurpriseModule:
    """Factory function to create a surprise module."""
    config = SurpriseConfig(
        d_model=d_model,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        scales=scales,
        **kwargs,
    )
    return SurpriseModule(config)
