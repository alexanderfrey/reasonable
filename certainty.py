"""
Unified certainty estimation module.

Combines multiple uncertainty signals into a single calibrated certainty score.
Inspired by CTM (Continuous Thought Machines) but adapted for experiential architecture.

Phase 1 of certainty integration:
- CertaintyHead: unifies surprise, meta_surprise, confidence_gate, self_confidence
- Calibration loss: trains certainty to match actual prediction correctness
- ECE metric: measures calibration quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CertaintyOutput:
    """Output from CertaintyHead."""
    certainty: torch.Tensor           # [B] in [0, 1], main output
    uncertainty_sources: torch.Tensor  # [B, n_sources] contribution of each source
    raw_signals: Dict[str, torch.Tensor]  # Original signals for debugging


class CertaintyHead(nn.Module):
    """
    Estimates calibrated certainty from multiple uncertainty signals.

    This module unifies:
    - surprise: prediction error (inverse relationship to certainty)
    - meta_surprise: self-calibration error (inverse relationship)
    - confidence_gate: per-dimension confidence from self-modulator
    - self_confidence: confidence in self-model predictions
    - soma: internal state context (optional)

    The output is trained to match actual prediction correctness (calibration).
    """

    # Fixed number of certainty sources (hardcoded, not configurable)
    # Sources: surprise, meta_surprise, confidence_gate, self_confidence, hidden
    N_SOURCES = 5

    def __init__(
        self,
        d_model: int,
        d_soma: int = 64,
        use_soma: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_soma = d_soma
        self.use_soma = use_soma
        n_sources = self.N_SOURCES  # Use class constant

        # Source-specific processors
        # Each source gets its own small network to extract certainty contribution

        # 1. Surprise-based certainty (inverse: low surprise = high certainty)
        self.surprise_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # 2. Meta-surprise-based certainty (inverse: good self-knowledge = certain)
        self.meta_surprise_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # 3. Confidence gate aggregation (from self-modulator)
        self.confidence_gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
        )

        # 4. Self-confidence processing (from SelfModel)
        self.self_confidence_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # 5. Hidden state certainty (learned pattern from representation)
        self.hidden_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
        )

        # Soma context modulation (optional)
        if use_soma:
            self.soma_context = nn.Sequential(
                nn.Linear(d_soma, 32),
                nn.GELU(),
                nn.Linear(32, n_sources),  # Modulates source weights based on internal state
            )

        # Learnable source weights (how much each source matters)
        self.source_weights = nn.Parameter(torch.ones(n_sources) / n_sources)

        # Output calibration network
        self.output_calibration = nn.Sequential(
            nn.Linear(n_sources, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, seq_len, d_model] or [B, d_model]
        surprise: Optional[torch.Tensor] = None,        # [B]
        meta_surprise: Optional[torch.Tensor] = None,   # [B]
        confidence_gate: Optional[torch.Tensor] = None,  # [B, d_model]
        self_confidence: Optional[torch.Tensor] = None,  # [B]
        soma: Optional[torch.Tensor] = None,            # [B, d_soma]
        return_sources: bool = False,
    ) -> CertaintyOutput:
        """
        Compute unified certainty score.

        Args:
            hidden_states: Model hidden states
            surprise: Prediction surprise [0, 1]
            meta_surprise: Self-prediction error [0, 1]
            confidence_gate: Per-dimension confidence [0, 1]
            self_confidence: Confidence in self-model [0, 1]
            soma: Internal state vector
            return_sources: Whether to return per-source contributions

        Returns:
            CertaintyOutput with certainty score and diagnostics
        """
        # Handle sequence input - use last position
        if hidden_states.dim() == 3:
            hidden = hidden_states[:, -1, :]  # [B, d_model]
        else:
            hidden = hidden_states  # [B, d_model]

        B = hidden.shape[0]
        device = hidden.device

        # Compute per-source certainty contributions
        source_contributions = []
        raw_signals = {}

        # 1. Surprise -> certainty (inverse relationship)
        if surprise is not None:
            raw_signals['surprise'] = surprise
            # Ensure proper shape
            s = surprise.view(B, 1) if surprise.dim() == 1 else surprise
            surprise_cert = self.surprise_net(s)  # [B, 1]
            # Low surprise = high certainty (apply sigmoid after negation)
            surprise_cert = torch.sigmoid(-surprise_cert)
        else:
            surprise_cert = torch.full((B, 1), 0.5, device=device)
        source_contributions.append(surprise_cert)

        # 2. Meta-surprise -> certainty (inverse: good self-knowledge = certain)
        if meta_surprise is not None:
            raw_signals['meta_surprise'] = meta_surprise
            ms = meta_surprise.view(B, 1) if meta_surprise.dim() == 1 else meta_surprise
            meta_cert = self.meta_surprise_net(ms)
            meta_cert = torch.sigmoid(-meta_cert)
        else:
            meta_cert = torch.full((B, 1), 0.5, device=device)
        source_contributions.append(meta_cert)

        # 3. Confidence gate -> certainty (aggregate per-dimension confidence)
        if confidence_gate is not None:
            raw_signals['confidence_gate'] = confidence_gate
            conf_cert = self.confidence_gate_net(confidence_gate)  # [B, 1]
            conf_cert = torch.sigmoid(conf_cert)
        else:
            conf_cert = torch.full((B, 1), 0.5, device=device)
        source_contributions.append(conf_cert)

        # 4. Self-confidence -> certainty (direct relationship)
        if self_confidence is not None:
            raw_signals['self_confidence'] = self_confidence
            sc = self_confidence.view(B, 1) if self_confidence.dim() == 1 else self_confidence
            self_cert = self.self_confidence_net(sc)
            self_cert = torch.sigmoid(self_cert)
        else:
            self_cert = torch.full((B, 1), 0.5, device=device)
        source_contributions.append(self_cert)

        # 5. Hidden state -> certainty (learned pattern)
        raw_signals['hidden'] = hidden
        hidden_cert = self.hidden_net(hidden)  # [B, 1]
        hidden_cert = torch.sigmoid(hidden_cert)
        source_contributions.append(hidden_cert)

        # Stack sources: [B, n_sources]
        sources = torch.cat(source_contributions, dim=-1)

        # Apply soma-based context modulation if available
        if self.use_soma and soma is not None:
            raw_signals['soma'] = soma
            soma_modulation = torch.sigmoid(self.soma_context(soma))  # [B, n_sources]
            sources = sources * soma_modulation

        # Weighted aggregation
        weights = F.softmax(self.source_weights, dim=0)  # [n_sources]
        weighted_sources = sources * weights.unsqueeze(0)  # [B, n_sources]

        # Final calibrated output
        certainty = self.output_calibration(weighted_sources).squeeze(-1)  # [B]

        return CertaintyOutput(
            certainty=certainty,
            uncertainty_sources=sources if return_sources else weighted_sources,
            raw_signals=raw_signals if return_sources else {},
        )


def certainty_calibration_loss(
    certainty: torch.Tensor,      # [B] predicted certainty
    was_correct: torch.Tensor,    # [B] binary or soft correctness
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Calibration loss: certainty should match actual correctness.

    Uses BCE so that:
    - High certainty + correct -> low loss (appropriate confidence)
    - High certainty + wrong -> high loss (overconfidence penalty)
    - Low certainty + wrong -> low loss (honest uncertainty)
    - Low certainty + correct -> medium loss (underconfidence)

    Args:
        certainty: [B] predicted certainty scores in [0, 1]
        was_correct: [B] binary or soft correctness in [0, 1]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Calibration loss
    """
    return F.binary_cross_entropy(
        certainty,
        was_correct.float(),
        reduction=reduction,
    )


def expected_calibration_error(
    certainty: torch.Tensor,      # [N] predicted certainties
    was_correct: torch.Tensor,    # [N] binary correctness
    n_bins: int = 10,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures how well-calibrated the certainty estimates are:
    - Bucket samples by certainty
    - Compare mean certainty vs mean accuracy per bucket
    - Weighted average of |certainty - accuracy| per bucket

    Args:
        certainty: [N] predicted certainty scores
        was_correct: [N] binary correctness indicators
        n_bins: Number of calibration bins

    Returns:
        ece: Scalar ECE value
        details: Per-bucket statistics for visualization
    """
    # Flatten inputs
    certainty = certainty.flatten()
    was_correct = was_correct.flatten().float()

    # Create bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=certainty.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1, device=certainty.device)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (certainty > bin_lower) & (certainty <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if in_bin.sum() > 0:
            accuracy_in_bin = was_correct[in_bin].mean()
            confidence_in_bin = certainty[in_bin].mean()

            ece += prop_in_bin * torch.abs(accuracy_in_bin - confidence_in_bin)

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(torch.tensor(0.0, device=certainty.device))
            bin_confidences.append(torch.tensor(0.0, device=certainty.device))
            bin_counts.append(torch.tensor(0, device=certainty.device))

    return ece, {
        'bin_accuracies': torch.stack(bin_accuracies),
        'bin_confidences': torch.stack(bin_confidences),
        'bin_counts': torch.stack(bin_counts),
        'bin_boundaries': bin_boundaries,
    }
