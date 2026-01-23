"""
Self-State Analyzer - Tools for interpreting the self-state representation.

Provides:
1. Collection buffer for self-states with metadata
2. Correlation analysis (which dimensions correlate with surprise/confidence)
3. Probing classifiers (what can be decoded from self-state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np


@dataclass
class SelfStateSnapshot:
    """A single snapshot of self-state with associated metadata."""
    self_state: torch.Tensor      # (d_self_state,)
    surprise: float               # Surprise level at this moment
    confidence: float             # Prediction confidence
    s_write_gate: float           # Self-write gate value
    did_write: bool               # Whether self-oscillators were written
    step: int                     # Training step
    # Optional content statistics
    input_norm: Optional[float] = None
    sync_norm: Optional[float] = None
    pred_norm: Optional[float] = None


class SelfStateBuffer:
    """
    Ring buffer to collect self-states during training.

    Stores snapshots with metadata for later analysis.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self._step = 0

    def add(
        self,
        self_state: torch.Tensor,
        surprise: float,
        confidence: float,
        s_write_gate: float,
        did_write: bool,
        input_norm: Optional[float] = None,
        sync_norm: Optional[float] = None,
        pred_norm: Optional[float] = None,
    ):
        """Add a snapshot to the buffer."""
        snapshot = SelfStateSnapshot(
            self_state=self_state.detach().cpu().clone(),
            surprise=surprise,
            confidence=confidence,
            s_write_gate=s_write_gate,
            did_write=did_write,
            step=self._step,
            input_norm=input_norm,
            sync_norm=sync_norm,
            pred_norm=pred_norm,
        )
        self.buffer.append(snapshot)
        self._step += 1

    def get_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Extract all data as tensors for analysis.

        Returns:
            Dict with keys: self_states, surprise, confidence, s_write_gate, did_write
        """
        if len(self.buffer) == 0:
            return {}

        self_states = torch.stack([s.self_state for s in self.buffer])
        surprise = torch.tensor([s.surprise for s in self.buffer])
        confidence = torch.tensor([s.confidence for s in self.buffer])
        s_write_gate = torch.tensor([s.s_write_gate for s in self.buffer])
        did_write = torch.tensor([float(s.did_write) for s in self.buffer])

        return {
            'self_states': self_states,      # (N, d_self_state)
            'surprise': surprise,            # (N,)
            'confidence': confidence,        # (N,)
            's_write_gate': s_write_gate,    # (N,)
            'did_write': did_write,          # (N,)
        }

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


class CorrelationAnalyzer:
    """
    Analyze correlations between self-state dimensions and metadata.
    """

    @staticmethod
    def compute_correlations(buffer: SelfStateBuffer) -> Dict[str, torch.Tensor]:
        """
        Compute correlations between self-state dimensions and surprise/confidence.

        Returns:
            Dict with:
            - dim_surprise_corr: (d_self_state,) correlation of each dim with surprise
            - dim_confidence_corr: (d_self_state,) correlation with confidence
            - dim_write_corr: (d_self_state,) correlation with write events
            - dim_variance: (d_self_state,) variance of each dimension
            - top_surprise_dims: indices of dims most correlated with surprise
            - top_confidence_dims: indices of dims most correlated with confidence
        """
        data = buffer.get_tensors()
        if not data:
            return {}

        self_states = data['self_states']  # (N, d)
        surprise = data['surprise']        # (N,)
        confidence = data['confidence']    # (N,)
        did_write = data['did_write']      # (N,)

        N, d = self_states.shape

        # Compute per-dimension correlations
        dim_surprise_corr = torch.zeros(d)
        dim_confidence_corr = torch.zeros(d)
        dim_write_corr = torch.zeros(d)

        # Center the data
        ss_centered = self_states - self_states.mean(dim=0, keepdim=True)
        surp_centered = surprise - surprise.mean()
        conf_centered = confidence - confidence.mean()
        write_centered = did_write - did_write.mean()

        # Variances
        ss_std = ss_centered.std(dim=0) + 1e-8
        surp_std = surp_centered.std() + 1e-8
        conf_std = conf_centered.std() + 1e-8
        write_std = write_centered.std() + 1e-8

        for i in range(d):
            dim_i = ss_centered[:, i]
            dim_std = ss_std[i]

            # Pearson correlation
            dim_surprise_corr[i] = (dim_i * surp_centered).mean() / (dim_std * surp_std)
            dim_confidence_corr[i] = (dim_i * conf_centered).mean() / (dim_std * conf_std)
            dim_write_corr[i] = (dim_i * write_centered).mean() / (dim_std * write_std)

        # Dimension variance (which dims vary most)
        dim_variance = self_states.var(dim=0)

        # Top correlated dimensions
        k = min(10, d)
        top_surprise_dims = dim_surprise_corr.abs().topk(k).indices
        top_confidence_dims = dim_confidence_corr.abs().topk(k).indices
        top_write_dims = dim_write_corr.abs().topk(k).indices
        top_variance_dims = dim_variance.topk(k).indices

        return {
            'dim_surprise_corr': dim_surprise_corr,
            'dim_confidence_corr': dim_confidence_corr,
            'dim_write_corr': dim_write_corr,
            'dim_variance': dim_variance,
            'top_surprise_dims': top_surprise_dims,
            'top_confidence_dims': top_confidence_dims,
            'top_write_dims': top_write_dims,
            'top_variance_dims': top_variance_dims,
        }

    @staticmethod
    def summarize(corr_results: Dict[str, torch.Tensor]) -> str:
        """Generate a human-readable summary of correlation analysis."""
        if not corr_results:
            return "No data for correlation analysis"

        lines = ["=== Self-State Correlation Analysis ==="]

        # Top surprise-correlated dimensions
        top_surp = corr_results['top_surprise_dims']
        surp_corr = corr_results['dim_surprise_corr']
        lines.append(f"\nTop dims correlated with SURPRISE:")
        for i, idx in enumerate(top_surp[:5]):
            lines.append(f"  dim[{idx:2d}]: r={surp_corr[idx]:+.3f}")

        # Top confidence-correlated dimensions
        top_conf = corr_results['top_confidence_dims']
        conf_corr = corr_results['dim_confidence_corr']
        lines.append(f"\nTop dims correlated with CONFIDENCE:")
        for i, idx in enumerate(top_conf[:5]):
            lines.append(f"  dim[{idx:2d}]: r={conf_corr[idx]:+.3f}")

        # Top write-correlated dimensions
        top_write = corr_results['top_write_dims']
        write_corr = corr_results['dim_write_corr']
        lines.append(f"\nTop dims correlated with WRITE events:")
        for i, idx in enumerate(top_write[:5]):
            lines.append(f"  dim[{idx:2d}]: r={write_corr[idx]:+.3f}")

        # Highest variance dimensions
        top_var = corr_results['top_variance_dims']
        variance = corr_results['dim_variance']
        lines.append(f"\nHighest variance dims:")
        for i, idx in enumerate(top_var[:5]):
            lines.append(f"  dim[{idx:2d}]: var={variance[idx]:.4f}")

        # Summary statistics
        lines.append(f"\nOverall statistics:")
        lines.append(f"  Max |surprise corr|: {surp_corr.abs().max():.3f}")
        lines.append(f"  Max |confidence corr|: {conf_corr.abs().max():.3f}")
        lines.append(f"  Max |write corr|: {write_corr.abs().max():.3f}")
        lines.append(f"  Mean variance: {variance.mean():.4f}")

        return "\n".join(lines)


class SelfStateProbe(nn.Module):
    """
    Probing classifier to test what can be decoded from self-state.

    Supports different architectures:
    - linear: Single linear layer (tests if information is linearly separable)
    - shallow: One hidden layer (tests simple nonlinear encoding)
    - deep: Two hidden layers (tests complex nonlinear encoding)
    """

    def __init__(
        self,
        d_self_state: int,
        target_type: str = "regression",  # "regression" or "classification"
        n_classes: int = 1,  # For classification
        hidden_dim: int = 32,
        architecture: str = "shallow",  # "linear", "shallow", or "deep"
    ):
        super().__init__()
        self.target_type = target_type
        self.n_classes = n_classes
        self.architecture = architecture

        out_dim = n_classes if target_type == "classification" else 1

        if architecture == "linear":
            self.probe = nn.Linear(d_self_state, out_dim)
        elif architecture == "shallow":
            self.probe = nn.Sequential(
                nn.Linear(d_self_state, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        elif architecture == "deep":
            self.probe = nn.Sequential(
                nn.Linear(d_self_state, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, self_state: torch.Tensor) -> torch.Tensor:
        return self.probe(self_state)

    def compute_loss(
        self,
        self_state: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.forward(self_state)
        if self.target_type == "classification":
            return F.cross_entropy(pred, target.long())
        else:
            return F.mse_loss(pred.squeeze(-1), target)

    def compute_accuracy(
        self,
        self_state: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """For classification: accuracy. For regression: R^2."""
        pred = self.forward(self_state)
        if self.target_type == "classification":
            pred_class = pred.argmax(dim=-1)
            return (pred_class == target.long()).float().mean().item()
        else:
            # R^2 for regression
            pred = pred.squeeze(-1)
            ss_res = ((target - pred) ** 2).sum()
            ss_tot = ((target - target.mean()) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            return r2.item()


class ProbeTrainer:
    """
    Train and evaluate probing classifiers on self-state data.
    """

    def __init__(self, d_self_state: int):
        self.d_self_state = d_self_state
        self.probes: Dict[str, SelfStateProbe] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}

    def add_probe(
        self,
        name: str,
        target_type: str = "regression",
        n_classes: int = 1,
        architecture: str = "shallow",
    ):
        """Add a new probe for a specific target."""
        probe = SelfStateProbe(
            d_self_state=self.d_self_state,
            target_type=target_type,
            n_classes=n_classes,
            architecture=architecture,
        )
        self.probes[name] = probe
        self.optimizers[name] = torch.optim.Adam(probe.parameters(), lr=1e-3)

    def train_step(
        self,
        buffer: SelfStateBuffer,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Train all probes for one step using data from buffer.

        Returns dict of losses per probe.
        """
        data = buffer.get_tensors()
        if not data or len(buffer) < batch_size:
            return {}

        # Sample a batch
        N = len(buffer)
        indices = torch.randperm(N)[:batch_size]

        self_states = data['self_states'][indices]
        surprise = data['surprise'][indices]
        confidence = data['confidence'][indices]
        did_write = data['did_write'][indices]

        losses = {}

        # Define targets for each probe (handle _deep suffix variants)
        targets = {
            'surprise': surprise,
            'confidence': confidence,
            'confidence_deep': confidence,  # Same target, different architecture
            'write': did_write,
        }

        for name, probe in self.probes.items():
            if name not in targets:
                continue

            target = targets[name]
            opt = self.optimizers[name]

            opt.zero_grad()
            loss = probe.compute_loss(self_states, target)
            loss.backward()
            opt.step()

            losses[name] = loss.item()

        return losses

    def evaluate(self, buffer: SelfStateBuffer) -> Dict[str, float]:
        """
        Evaluate all probes on the full buffer.

        Returns dict of accuracy/R^2 per probe.
        """
        data = buffer.get_tensors()
        if not data:
            return {}

        self_states = data['self_states']
        targets = {
            'surprise': data['surprise'],
            'confidence': data['confidence'],
            'confidence_deep': data['confidence'],  # Same target, different architecture
            'write': data['did_write'],
        }

        results = {}

        for name, probe in self.probes.items():
            if name not in targets:
                continue

            with torch.no_grad():
                acc = probe.compute_accuracy(self_states, targets[name])
                results[f'{name}_r2' if probe.target_type == 'regression' else f'{name}_acc'] = acc

        return results

    def summarize(self, buffer: SelfStateBuffer) -> str:
        """Generate summary of probe performance."""
        results = self.evaluate(buffer)
        if not results:
            return "No probe results"

        lines = ["=== Self-State Probe Results ==="]
        for name, value in results.items():
            lines.append(f"  {name}: {value:.3f}")

        # Interpretation
        lines.append("\nInterpretation:")
        for name, value in results.items():
            if 'r2' in name:
                if value > 0.5:
                    lines.append(f"  {name}: STRONG encoding (R^2={value:.2f})")
                elif value > 0.2:
                    lines.append(f"  {name}: moderate encoding (R^2={value:.2f})")
                else:
                    lines.append(f"  {name}: weak/no encoding (R^2={value:.2f})")

        return "\n".join(lines)


class SelfStateAnalyzer:
    """
    Main analyzer class combining buffer, correlations, and probes.
    """

    def __init__(
        self,
        d_self_state: int,
        buffer_size: int = 10000,
        enable_probes: bool = True,
    ):
        self.buffer = SelfStateBuffer(max_size=buffer_size)
        self.correlation_analyzer = CorrelationAnalyzer()

        self.probe_trainer = None
        if enable_probes:
            self.probe_trainer = ProbeTrainer(d_self_state)
            # Linear probes (test if info is linearly separable)
            self.probe_trainer.add_probe('surprise', target_type='regression', architecture='linear')
            self.probe_trainer.add_probe('confidence', target_type='regression', architecture='linear')
            self.probe_trainer.add_probe('write', target_type='regression', architecture='linear')
            # Deep MLP probes (test if info is nonlinearly encoded)
            self.probe_trainer.add_probe('confidence_deep', target_type='regression', architecture='deep')
            self.probe_trainer.add_probe('write', target_type='regression')

    def record(
        self,
        self_state: torch.Tensor,
        surprise: float,
        confidence: float,
        s_write_gate: float,
        did_write: bool,
        **kwargs,
    ):
        """Record a self-state snapshot."""
        self.buffer.add(
            self_state=self_state,
            surprise=surprise,
            confidence=confidence,
            s_write_gate=s_write_gate,
            did_write=did_write,
            **kwargs,
        )

    def train_probes(self, batch_size: int = 64) -> Dict[str, float]:
        """Train probes for one step."""
        if self.probe_trainer is None:
            return {}
        return self.probe_trainer.train_step(self.buffer, batch_size)

    def analyze(self) -> Dict[str, any]:
        """
        Run full analysis on collected data.

        Returns dict with correlation and probe results.
        """
        results = {}

        # Correlation analysis
        corr_results = self.correlation_analyzer.compute_correlations(self.buffer)
        results['correlations'] = corr_results

        # Probe evaluation
        if self.probe_trainer is not None:
            probe_results = self.probe_trainer.evaluate(self.buffer)
            results['probes'] = probe_results

        return results

    def get_summary(self) -> str:
        """Get human-readable summary of all analyses."""
        lines = [f"=== Self-State Analysis (n={len(self.buffer)} samples) ===\n"]

        # Correlation summary
        corr_results = self.correlation_analyzer.compute_correlations(self.buffer)
        lines.append(self.correlation_analyzer.summarize(corr_results))

        # Probe summary
        if self.probe_trainer is not None:
            lines.append("\n")
            lines.append(self.probe_trainer.summarize(self.buffer))

        return "\n".join(lines)
