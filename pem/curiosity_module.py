"""
Curiosity Module for PEM.

Implements epistemic drive - the motivation to seek information and reduce uncertainty.

Core insight:
- Surprise tells us WHAT was unexpected (reactive)
- Curiosity drives us to SEEK the unexpected (proactive)

Two types of value in decision-making:
1. Pragmatic value (valence): "Is this good for my goals?"
2. Epistemic value (curiosity): "Will this teach me something?"

A truly experiencing system doesn't just react to surprise - it actively
seeks to understand the world.

Architecture:
    Predictions → UncertaintyEstimator → uncertainty
                                              ↓
    Context → InformationGainComputer → expected_info_gain
                                              ↓
                                        CuriosityModule
                                              ↓
                            curiosity signal (where to explore)
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CuriosityOutput(NamedTuple):
    """Output from CuriosityModule forward pass."""
    curiosity: torch.Tensor           # (B, S, 1) curiosity intensity
    uncertainty: torch.Tensor         # (B, S, 1) prediction uncertainty
    information_gain: torch.Tensor    # (B, S, 1) expected information gain
    exploration_bonus: torch.Tensor   # (B, S, D) attention modulation


@dataclass
class CuriosityConfig:
    """Configuration for the curiosity module.

    Curiosity = epistemic drive to reduce uncertainty and gain information.
    """

    d_model: int = 1536              # Feature dimension
    hidden_dim: Optional[int] = None # Hidden layer size (defaults to d_model // 2)
    n_layers: int = 2                # Depth of networks

    # Uncertainty estimation
    use_ensemble: bool = False       # Use ensemble for uncertainty (expensive)
    n_ensemble: int = 5              # Number of ensemble members
    dropout_uncertainty: float = 0.1 # MC dropout for uncertainty estimation

    # Information gain
    use_temporal_novelty: bool = True   # Track what's been seen before
    novelty_memory_size: int = 256      # How many past states to remember
    novelty_decay: float = 0.99         # How fast novelty decays

    # Exploration-exploitation balance
    exploration_weight: float = 0.5     # Weight of curiosity in attention
    curiosity_temperature: float = 1.0  # Sharpness of curiosity signal

    def __post_init__(self):
        if self.hidden_dim is None:
            self.hidden_dim = self.d_model // 2


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class UncertaintyEstimator(nn.Module):
    """
    Estimates prediction uncertainty.

    Uncertainty has two components:
    1. Epistemic uncertainty: "I don't know" (reducible with more data)
    2. Aleatoric uncertainty: "It's inherently random" (irreducible)

    We care mainly about epistemic uncertainty - that's what curiosity can reduce.

    Methods:
    - MC Dropout: Run multiple forward passes with dropout, measure variance
    - Direct estimation: Learn to predict uncertainty from features
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.1,
        n_samples: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_samples = n_samples

        # Uncertainty estimation network
        # Takes features and predictions, outputs uncertainty estimate
        self.uncertainty_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),  # features + predictions
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Uncertainty must be positive
        )

        # Prediction variance estimator (for MC dropout approach)
        self.variance_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
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

    def forward(
        self,
        features: torch.Tensor,      # (B, S, D) current features
        predictions: torch.Tensor,   # (B, S, D) model predictions
        use_mc_dropout: bool = False,
    ) -> torch.Tensor:
        """
        Estimate prediction uncertainty.

        Args:
            features: Current context features
            predictions: What the model predicted
            use_mc_dropout: Use MC dropout for uncertainty (slower but more accurate)

        Returns:
            uncertainty: (B, S, 1) uncertainty estimate (higher = more uncertain)
        """
        B, S, D = features.shape

        if use_mc_dropout and self.training:
            # MC Dropout: Run multiple forward passes, measure variance
            samples = []
            for _ in range(self.n_samples):
                sample = self.variance_proj(predictions)
                samples.append(sample)

            samples = torch.stack(samples, dim=0)  # (n_samples, B, S, D)
            variance = samples.var(dim=0).mean(dim=-1, keepdim=True)  # (B, S, 1)

            # Combine with learned uncertainty
            combined = torch.cat([features, predictions], dim=-1)
            learned_uncertainty = self.uncertainty_net(combined)

            uncertainty = 0.5 * variance + 0.5 * learned_uncertainty

        else:
            # Direct estimation
            combined = torch.cat([features, predictions], dim=-1)
            uncertainty = self.uncertainty_net(combined)

        return uncertainty

    def estimate_epistemic(
        self,
        features: torch.Tensor,
        prediction_fn: callable,
        n_samples: int = 10,
    ) -> torch.Tensor:
        """
        Estimate epistemic uncertainty via MC dropout on a prediction function.

        This is more expensive but more accurate for measuring "what we don't know".
        """
        samples = []
        self.train()  # Enable dropout

        for _ in range(n_samples):
            with torch.no_grad():
                pred = prediction_fn(features)
                samples.append(pred)

        samples = torch.stack(samples, dim=0)
        epistemic_uncertainty = samples.var(dim=0).mean(dim=-1, keepdim=True)

        return epistemic_uncertainty


class NoveltyMemory(nn.Module):
    """
    Tracks what has been seen before to compute novelty.

    Novelty = how different is this from what I've seen?

    Uses a memory bank of past states and computes distance to nearest neighbors.
    """

    def __init__(
        self,
        d_model: int,
        memory_size: int = 256,
        decay: float = 0.99,
    ):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        self.decay = decay

        # Memory bank (not a parameter - updated during forward pass)
        self.register_buffer('memory', torch.zeros(memory_size, d_model))
        self.register_buffer('memory_age', torch.zeros(memory_size))
        self.register_buffer('write_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('is_filled', torch.tensor(False, dtype=torch.bool))

        # Projection for memory comparison
        self.memory_proj = nn.Linear(d_model, d_model // 2, bias=False)

    def compute_novelty(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute novelty of features relative to memory.

        Args:
            features: (B, S, D) current features

        Returns:
            novelty: (B, S, 1) novelty score (higher = more novel)
        """
        B, S, D = features.shape

        if not self.is_filled and self.write_ptr == 0:
            # Empty memory: everything is maximally novel
            return torch.ones(B, S, 1, device=features.device)

        # Project features for comparison
        feat_proj = self.memory_proj(features)  # (B, S, D//2)

        # Get valid memory slots
        if self.is_filled:
            valid_memory = self.memory
        else:
            valid_memory = self.memory[:self.write_ptr]

        mem_proj = self.memory_proj(valid_memory)  # (M, D//2)

        # Compute distance to all memory slots
        # feat_proj: (B, S, D//2), mem_proj: (M, D//2)
        feat_norm = F.normalize(feat_proj, dim=-1)
        mem_norm = F.normalize(mem_proj, dim=-1)

        # Cosine similarity to all memories
        similarity = torch.einsum('bsd,md->bsm', feat_norm, mem_norm)  # (B, S, M)

        # Novelty = 1 - max similarity (most similar memory)
        max_similarity = similarity.max(dim=-1, keepdim=True)[0]  # (B, S, 1)
        novelty = 1.0 - max_similarity

        return novelty

    def update(self, features: torch.Tensor) -> None:
        """
        Add new features to memory.

        Args:
            features: (B, S, D) features to remember
        """
        B, S, D = features.shape

        # Flatten batch and sequence
        flat_features = features.reshape(-1, D)  # (B*S, D)

        # Subsample if too many
        n_new = flat_features.shape[0]
        if n_new > self.memory_size // 4:
            indices = torch.randperm(n_new, device=features.device)[:self.memory_size // 4]
            flat_features = flat_features[indices]
            n_new = flat_features.shape[0]

        # Write to memory (circular buffer)
        for i in range(n_new):
            self.memory[self.write_ptr] = flat_features[i].detach()
            self.memory_age[self.write_ptr] = 0
            self.write_ptr = (self.write_ptr + 1) % self.memory_size

            if self.write_ptr == 0:
                self.is_filled = torch.tensor(True, device=features.device)

        # Age all memories
        self.memory_age += 1

    def reset(self):
        """Clear memory."""
        self.memory.zero_()
        self.memory_age.zero_()
        self.write_ptr.zero_()
        self.is_filled.fill_(False)


class InformationGainComputer(nn.Module):
    """
    Computes expected information gain from attending to different positions.

    Information gain = how much would attending to X reduce my uncertainty?

    Based on the principle that we should attend to positions where:
    1. We are uncertain (high epistemic uncertainty)
    2. The content is novel (not seen before)
    3. Attending would be informative (high mutual information)
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        use_novelty: bool = True,
        novelty_memory_size: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.use_novelty = use_novelty

        # Novelty memory
        if use_novelty:
            self.novelty_memory = NoveltyMemory(
                d_model=d_model,
                memory_size=novelty_memory_size,
            )
        else:
            self.novelty_memory = None

        # Information gain estimator
        # Takes uncertainty + novelty + context, outputs expected info gain
        input_dim = 2 if use_novelty else 1  # uncertainty + novelty
        self.info_gain_net = nn.Sequential(
            nn.Linear(d_model + input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Info gain must be positive
        )

        # Mutual information estimator (simplified)
        # Estimates how much we'd learn by attending to each position
        self.mi_estimator = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
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
        features: torch.Tensor,      # (B, S, D) current features
        uncertainty: torch.Tensor,   # (B, S, 1) prediction uncertainty
        context: Optional[torch.Tensor] = None,  # (B, S, D) context for MI
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute expected information gain.

        Args:
            features: Current features
            uncertainty: Prediction uncertainty
            context: Optional context for mutual information

        Returns:
            info_gain: (B, S, 1) expected information gain
            novelty: (B, S, 1) novelty score
        """
        B, S, D = features.shape

        # Compute novelty
        if self.novelty_memory is not None:
            novelty = self.novelty_memory.compute_novelty(features)
        else:
            novelty = torch.zeros(B, S, 1, device=features.device)

        # Combine signals for info gain estimation
        if self.use_novelty:
            combined_signals = torch.cat([uncertainty, novelty], dim=-1)  # (B, S, 2)
        else:
            combined_signals = uncertainty  # (B, S, 1)

        # Compute expected information gain
        info_input = torch.cat([features, combined_signals], dim=-1)
        info_gain = self.info_gain_net(info_input)  # (B, S, 1)

        # Optional: Mutual information with context
        if context is not None:
            # How much would attending to each position tell us about context?
            mi_input = torch.cat([features, context], dim=-1)
            mi_weight = self.mi_estimator(mi_input)  # (B, S, 1)
            info_gain = info_gain * (1 + mi_weight)  # Boost info gain where MI is high

        return info_gain, novelty

    def update_novelty(self, features: torch.Tensor) -> None:
        """Update novelty memory with new features."""
        if self.novelty_memory is not None:
            self.novelty_memory.update(features)

    def reset_novelty(self):
        """Reset novelty memory."""
        if self.novelty_memory is not None:
            self.novelty_memory.reset()


class ExplorationBonusComputer(nn.Module):
    """
    Computes exploration bonus for attention.

    The exploration bonus modulates attention to encourage looking at
    uncertain/novel/informative regions.

    exploration_bonus = f(curiosity, features) → attention modulation
    """

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()

        # Project curiosity signal to attention-compatible space
        self.bonus_proj = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),  # features + curiosity
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

        # Learnable scale for exploration bonus
        self.exploration_scale = nn.Parameter(torch.tensor(0.5))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        features: torch.Tensor,   # (B, S, D)
        curiosity: torch.Tensor,  # (B, S, 1)
    ) -> torch.Tensor:
        """
        Compute exploration bonus for attention.

        Returns:
            bonus: (B, S, D) attention query modulation
        """
        combined = torch.cat([features, curiosity], dim=-1)
        bonus = self.bonus_proj(combined)

        # Scale by learnable parameter
        bonus = self.exploration_scale * bonus

        return bonus


class CuriosityModule(nn.Module):
    """
    Compute curiosity/epistemic drive.

    Curiosity = the drive to reduce uncertainty and gain information.

    Core equation:
        curiosity = f(uncertainty, novelty, expected_information_gain)

    Where:
        - uncertainty: How confident are my predictions?
        - novelty: Have I seen this before?
        - information_gain: How much would I learn by attending here?

    Curiosity modulates:
        - Attention: Look at uncertain/novel things
        - Memory: Remember informative experiences
        - Intention: Balance goal pursuit with exploration

    Usage:
        curiosity_module = CuriosityModule(config)
        output = curiosity_module(
            features=qwen_features,
            predictions=pred_module(sync, features),
            context=features,
        )
        # output.curiosity: (B, S, 1) curiosity intensity
        # output.exploration_bonus: (B, S, D) attention modulation
    """

    def __init__(self, config: CuriosityConfig):
        super().__init__()
        self.config = config

        # 1. Uncertainty estimation
        self.uncertainty_estimator = UncertaintyEstimator(
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout_uncertainty,
        )

        # 2. Information gain computation
        self.info_gain_computer = InformationGainComputer(
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
            use_novelty=config.use_temporal_novelty,
            novelty_memory_size=config.novelty_memory_size,
        )

        # 3. Exploration bonus for attention
        self.exploration_bonus = ExplorationBonusComputer(
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
        )

        # 4. Curiosity combiner
        # Combines uncertainty, novelty, info gain into single curiosity signal
        self.curiosity_combiner = nn.Sequential(
            nn.Linear(3, config.hidden_dim // 4),  # uncertainty + novelty + info_gain
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid(),  # Curiosity in [0, 1]
        )

        # Temperature for sharpening curiosity
        self.temperature = config.curiosity_temperature

        # Exploration weight
        self.exploration_weight = config.exploration_weight

        self.norm = RMSNorm(config.d_model)

    def forward(
        self,
        features: torch.Tensor,                  # (B, S, D) current features
        predictions: Optional[torch.Tensor] = None,  # (B, S, D) model predictions
        context: Optional[torch.Tensor] = None,  # (B, S, D) context for info gain
        update_novelty: bool = True,             # Update novelty memory?
    ) -> CuriosityOutput:
        """
        Compute curiosity signal.

        Args:
            features: Current features (from Qwen or CTM state)
            predictions: Model predictions (for uncertainty estimation)
            context: Optional context for mutual information
            update_novelty: Whether to update novelty memory

        Returns:
            CuriosityOutput with:
                - curiosity: (B, S, 1) curiosity intensity
                - uncertainty: (B, S, 1) prediction uncertainty
                - information_gain: (B, S, 1) expected info gain
                - exploration_bonus: (B, S, D) attention modulation
        """
        B, S, D = features.shape

        # 1. Estimate uncertainty
        if predictions is not None:
            uncertainty = self.uncertainty_estimator(features, predictions)
        else:
            # Without predictions, use feature variance as proxy
            uncertainty = features.var(dim=-1, keepdim=True)
            uncertainty = uncertainty / (uncertainty.max() + 1e-8)  # Normalize

        # 2. Compute information gain (includes novelty)
        info_gain, novelty = self.info_gain_computer(
            features=features,
            uncertainty=uncertainty,
            context=context,
        )

        # 3. Update novelty memory
        if update_novelty:
            self.info_gain_computer.update_novelty(features)

        # 4. Combine into curiosity signal
        curiosity_input = torch.cat([uncertainty, novelty, info_gain], dim=-1)
        curiosity = self.curiosity_combiner(curiosity_input)  # (B, S, 1)

        # Apply temperature (sharpen or soften)
        if self.temperature != 1.0:
            curiosity = torch.sigmoid((curiosity - 0.5) / self.temperature + 0.5)

        # 5. Compute exploration bonus for attention
        exploration = self.exploration_bonus(features, curiosity)
        exploration = self.norm(exploration)

        # Scale by exploration weight
        exploration = self.exploration_weight * exploration

        return CuriosityOutput(
            curiosity=curiosity,
            uncertainty=uncertainty,
            information_gain=info_gain,
            exploration_bonus=exploration,
        )

    def compute_epistemic_value(
        self,
        features: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pure epistemic value (expected uncertainty reduction).

        This is useful for decision-making: actions that reduce uncertainty
        have high epistemic value.
        """
        output = self.forward(features, predictions, update_novelty=False)

        # Epistemic value = curiosity * (1 - certainty)
        certainty = 1.0 - output.uncertainty
        epistemic_value = output.curiosity * (1.0 - certainty)

        return epistemic_value

    def reset_novelty(self):
        """Reset novelty memory (e.g., at start of new document)."""
        self.info_gain_computer.reset_novelty()


class CuriosityLoss(nn.Module):
    """
    Loss for training the curiosity module.

    Key insight: Curiosity should be high where predictions are wrong.
    If we were curious about X and X surprised us, curiosity was calibrated.
    If we weren't curious about X but X surprised us, curiosity was too low.

    Losses:
    1. Calibration: High curiosity should correlate with high surprise
    2. Exploration reward: Encourage exploring uncertain regions
    3. Novelty correlation: Novel things should trigger curiosity
    """

    def __init__(
        self,
        calibration_weight: float = 1.0,
        exploration_weight: float = 0.5,
        novelty_weight: float = 0.3,
    ):
        super().__init__()
        self.calibration_weight = calibration_weight
        self.exploration_weight = exploration_weight
        self.novelty_weight = novelty_weight

    def forward(
        self,
        curiosity_output: CuriosityOutput,
        surprise_magnitude: torch.Tensor,  # (B, S, 1) from SurpriseModule
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute curiosity training loss.

        Args:
            curiosity_output: Output from CuriosityModule
            surprise_magnitude: How surprising each position was
            valid_mask: Which positions are valid

        Returns:
            total_loss: Scalar loss
            loss_dict: Breakdown by component
        """
        loss_dict = {}
        device = curiosity_output.curiosity.device
        total_loss = torch.tensor(0.0, device=device)

        curiosity = curiosity_output.curiosity
        uncertainty = curiosity_output.uncertainty

        # 1. Calibration loss: curiosity should predict surprise
        # If curiosity was high and surprise was high → good
        # If curiosity was low and surprise was high → bad (should've been curious)
        calibration_loss = F.mse_loss(curiosity, surprise_magnitude)
        loss_dict['calibration'] = calibration_loss.detach()
        total_loss = total_loss + self.calibration_weight * calibration_loss

        # 2. Exploration reward: uncertainty reduction is good
        # This is a pseudo-reward: if we attended to uncertain things, that's good
        # Measured by: were uncertain things attended to?
        exploration_reward = (curiosity * uncertainty).mean()
        exploration_loss = -exploration_reward  # Maximize this
        loss_dict['exploration'] = exploration_loss.detach()
        total_loss = total_loss + self.exploration_weight * exploration_loss

        # 3. Novelty correlation: novel things should be curious
        novelty = curiosity_output.information_gain  # Proxy for novelty
        novelty_correlation = F.cosine_similarity(
            curiosity.view(-1), novelty.view(-1), dim=0
        )
        novelty_loss = 1.0 - novelty_correlation
        loss_dict['novelty_correlation'] = novelty_loss.detach()
        total_loss = total_loss + self.novelty_weight * novelty_loss

        return total_loss, loss_dict


def create_curiosity_module(
    d_model: int = 1536,
    hidden_dim: Optional[int] = None,
    use_temporal_novelty: bool = True,
    exploration_weight: float = 0.5,
    **kwargs,
) -> CuriosityModule:
    """Factory function to create a curiosity module."""
    config = CuriosityConfig(
        d_model=d_model,
        hidden_dim=hidden_dim,
        use_temporal_novelty=use_temporal_novelty,
        exploration_weight=exploration_weight,
        **kwargs,
    )
    return CuriosityModule(config)
