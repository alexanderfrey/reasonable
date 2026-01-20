"""
World State Monitor - Comprehensive monitoring for the oscillatory world model.

Tracks detailed metrics about the oscillatory world state to understand:
1. How oscillators evolve over time (phase/amplitude dynamics)
2. Which frequency bands are active
3. How modulation affects oscillator behavior
4. Whether gradients are flowing through modulation networks

Usage with oscillatory world model:
    # Oscillator metrics are computed directly in OscillatoryWorldState.get_metrics()
    # This module provides additional tracking and visualization utilities

    from pem.oscillatory_world import OscillatoryWorldState, OscillatorMetrics

    osc_world = OscillatoryWorldState(config)
    metrics = osc_world.get_metrics()

Legacy GRU-based world state monitoring classes are preserved for backwards compatibility
but are no longer actively used.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WorldStateMetrics:
    """Container for world state metrics."""

    # === Basic Statistics ===
    norm: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0

    # === Distribution Shape ===
    skewness: float = 0.0          # Asymmetry of distribution
    kurtosis: float = 0.0          # "Peakedness" (>3 = heavy tails)
    sparsity: float = 0.0          # Fraction of near-zero values
    effective_rank: float = 0.0    # How many dimensions are "used"
    entropy: float = 0.0           # Information content estimate

    # === Update Dynamics ===
    update_gate_mean: float = 0.0  # How much new info incorporated
    update_gate_std: float = 0.0   # Variability in gating
    reset_gate_mean: float = 0.0   # How much old state is "forgotten"
    reset_gate_std: float = 0.0
    candidate_norm: float = 0.0    # Magnitude of proposed update

    # === Change Metrics ===
    delta_norm: float = 0.0        # ||S_new - S_old||
    delta_relative: float = 0.0    # ||S_new - S_old|| / ||S_old||
    cosine_similarity: float = 0.0 # cos(S_new, S_old) - direction change

    # === Influence Metrics ===
    z_world_norm: float = 0.0      # Magnitude of world->z0 projection
    z_world_contribution: float = 0.0  # ||z_world|| / ||z_init + z_world||
    z_world_cosine_with_init: float = 0.0  # Alignment with input-based init

    # === Temporal Dynamics ===
    velocity: float = 0.0          # Smoothed rate of change
    acceleration: float = 0.0      # Change in velocity
    stability_score: float = 0.0   # 1 - normalized variance of recent changes

    # === Health Indicators ===
    is_saturated: bool = False     # Values near boundaries
    is_collapsed: bool = False     # All values similar
    is_exploding: bool = False     # Norm growing too fast
    is_dead: bool = False          # Not changing at all


class WorldStateMonitor(nn.Module):
    """
    Comprehensive monitoring for the emergent world model.

    Tracks the persistent sync state over time to understand:
    - What information is being stored
    - How quickly/slowly the state adapts
    - Whether it's learning useful representations
    - Health indicators for training stability
    """

    def __init__(
        self,
        d_sync_state: int = 256,
        history_length: int = 100,
        sparsity_threshold: float = 0.01,
        saturation_threshold: float = 0.95,
        collapse_threshold: float = 0.01,
        explosion_threshold: float = 100.0,
        dead_threshold: float = 1e-6,
    ):
        super().__init__()
        self.d_sync_state = d_sync_state
        self.history_length = history_length
        self.sparsity_threshold = sparsity_threshold
        self.saturation_threshold = saturation_threshold
        self.collapse_threshold = collapse_threshold
        self.explosion_threshold = explosion_threshold
        self.dead_threshold = dead_threshold

        # History tracking (not parameters, just state)
        self.norm_history: deque = deque(maxlen=history_length)
        self.delta_history: deque = deque(maxlen=history_length)
        self.state_snapshots: deque = deque(maxlen=10)  # Keep last 10 states

        # For velocity/acceleration
        self.prev_velocity: float = 0.0
        self.update_count: int = 0

    def reset(self):
        """Reset all tracking history."""
        self.norm_history.clear()
        self.delta_history.clear()
        self.state_snapshots.clear()
        self.prev_velocity = 0.0
        self.update_count = 0

    @torch.no_grad()
    def compute_basic_stats(self, S: torch.Tensor) -> Dict[str, float]:
        """Compute basic statistics of world state."""
        return {
            'norm': S.norm().item(),
            'mean': S.mean().item(),
            'std': S.std().item(),
            'min_val': S.min().item(),
            'max_val': S.max().item(),
        }

    @torch.no_grad()
    def compute_distribution_metrics(self, S: torch.Tensor) -> Dict[str, float]:
        """Compute distribution shape metrics."""
        # Normalize for moment calculations
        S_centered = S - S.mean()
        std = S.std()

        if std < 1e-8:
            return {
                'skewness': 0.0,
                'kurtosis': 0.0,
                'sparsity': 1.0,
                'effective_rank': 0.0,
                'entropy': 0.0,
            }

        S_norm = S_centered / std

        # Skewness: E[(X-mu)^3] / sigma^3
        skewness = (S_norm ** 3).mean().item()

        # Kurtosis: E[(X-mu)^4] / sigma^4 (excess kurtosis, normal = 0)
        kurtosis = (S_norm ** 4).mean().item() - 3.0

        # Sparsity: fraction of values near zero
        sparsity = (S.abs() < self.sparsity_threshold).float().mean().item()

        # Effective rank via entropy of squared values
        S_sq = S ** 2
        S_sq_sum = S_sq.sum()
        if S_sq_sum > 1e-8:
            p = S_sq / S_sq_sum  # "Probability" distribution
            entropy = -(p * torch.log(p + 1e-10)).sum().item()
            effective_rank = math.exp(entropy)  # Perplexity
        else:
            entropy = 0.0
            effective_rank = 0.0

        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'sparsity': sparsity,
            'effective_rank': effective_rank,
            'entropy': entropy,
        }

    @torch.no_grad()
    def compute_change_metrics(
        self,
        S_old: torch.Tensor,
        S_new: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute metrics about how the state changed."""
        delta = S_new - S_old
        delta_norm = delta.norm().item()
        old_norm = S_old.norm().item()

        # Relative change
        delta_relative = delta_norm / (old_norm + 1e-8)

        # Direction change (cosine similarity)
        if old_norm > 1e-8 and S_new.norm() > 1e-8:
            cosine_sim = F.cosine_similarity(
                S_old.unsqueeze(0),
                S_new.unsqueeze(0)
            ).item()
        else:
            cosine_sim = 1.0  # No change in direction if zero

        return {
            'delta_norm': delta_norm,
            'delta_relative': delta_relative,
            'cosine_similarity': cosine_sim,
        }

    @torch.no_grad()
    def compute_gate_metrics(
        self,
        update_gate: Optional[torch.Tensor] = None,
        reset_gate: Optional[torch.Tensor] = None,
        candidate: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute metrics about GRU gate values."""
        metrics = {}

        if update_gate is not None:
            metrics['update_gate_mean'] = update_gate.mean().item()
            metrics['update_gate_std'] = update_gate.std().item()
            metrics['update_gate_min'] = update_gate.min().item()
            metrics['update_gate_max'] = update_gate.max().item()

        if reset_gate is not None:
            metrics['reset_gate_mean'] = reset_gate.mean().item()
            metrics['reset_gate_std'] = reset_gate.std().item()
            metrics['reset_gate_min'] = reset_gate.min().item()
            metrics['reset_gate_max'] = reset_gate.max().item()

        if candidate is not None:
            metrics['candidate_norm'] = candidate.norm().item()
            metrics['candidate_mean'] = candidate.mean().item()
            metrics['candidate_std'] = candidate.std().item()

        return metrics

    @torch.no_grad()
    def compute_influence_metrics(
        self,
        z_world: Optional[torch.Tensor] = None,
        z_init: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute metrics about world state influence on CTM."""
        metrics = {}

        if z_world is not None:
            z_world_flat = z_world.reshape(-1)
            metrics['z_world_norm'] = z_world_flat.norm().item()
            metrics['z_world_mean'] = z_world_flat.mean().item()
            metrics['z_world_std'] = z_world_flat.std().item()

            if z_init is not None:
                z_init_flat = z_init.reshape(-1)
                z_combined = z_init_flat + z_world_flat
                combined_norm = z_combined.norm().item()

                # Contribution ratio
                if combined_norm > 1e-8:
                    contribution = z_world_flat.norm().item() / combined_norm
                else:
                    contribution = 0.0
                metrics['z_world_contribution'] = contribution

                # Alignment with input-based initialization
                if z_init_flat.norm() > 1e-8 and z_world_flat.norm() > 1e-8:
                    cosine = F.cosine_similarity(
                        z_init_flat.unsqueeze(0),
                        z_world_flat.unsqueeze(0)
                    ).item()
                else:
                    cosine = 0.0
                metrics['z_world_cosine_with_init'] = cosine

        return metrics

    @torch.no_grad()
    def compute_temporal_metrics(self, delta_norm: float) -> Dict[str, float]:
        """Compute temporal dynamics metrics."""
        self.delta_history.append(delta_norm)

        if len(self.delta_history) < 2:
            return {
                'velocity': delta_norm,
                'acceleration': 0.0,
                'stability_score': 1.0,
            }

        # Exponential moving average for velocity
        alpha = 0.1
        velocity = alpha * delta_norm + (1 - alpha) * self.prev_velocity

        # Acceleration
        acceleration = velocity - self.prev_velocity
        self.prev_velocity = velocity

        # Stability: inverse of coefficient of variation
        deltas = list(self.delta_history)
        mean_delta = sum(deltas) / len(deltas)
        if mean_delta > 1e-8:
            std_delta = (sum((d - mean_delta)**2 for d in deltas) / len(deltas)) ** 0.5
            cv = std_delta / mean_delta
            stability_score = 1.0 / (1.0 + cv)  # Higher = more stable
        else:
            stability_score = 1.0

        return {
            'velocity': velocity,
            'acceleration': acceleration,
            'stability_score': stability_score,
        }

    @torch.no_grad()
    def compute_health_indicators(
        self,
        S: torch.Tensor,
        delta_norm: float,
    ) -> Dict[str, bool]:
        """Compute health indicators for the world state."""
        norm = S.norm().item()
        std = S.std().item()
        mean_abs = S.abs().mean().item()

        # Track norm history
        self.norm_history.append(norm)

        # Saturation: values near tanh boundaries (-1, 1)
        # (since world_to_z0 uses Tanh)
        saturated_frac = ((S.abs() > self.saturation_threshold).float().mean().item())
        is_saturated = saturated_frac > 0.1  # >10% saturated

        # Collapse: all values too similar
        is_collapsed = std < self.collapse_threshold

        # Explosion: norm growing too fast
        is_exploding = norm > self.explosion_threshold
        if len(self.norm_history) >= 10:
            recent_growth = self.norm_history[-1] / (self.norm_history[-10] + 1e-8)
            is_exploding = is_exploding or recent_growth > 2.0  # Doubled in 10 steps

        # Dead: not changing at all
        is_dead = delta_norm < self.dead_threshold

        return {
            'is_saturated': is_saturated,
            'is_collapsed': is_collapsed,
            'is_exploding': is_exploding,
            'is_dead': is_dead,
            'saturated_fraction': saturated_frac,
        }

    @torch.no_grad()
    def compute_all_metrics(
        self,
        S_world_old: torch.Tensor,
        S_world_new: torch.Tensor,
        update_gate: Optional[torch.Tensor] = None,
        reset_gate: Optional[torch.Tensor] = None,
        candidate: Optional[torch.Tensor] = None,
        z_world: Optional[torch.Tensor] = None,
        z_init: Optional[torch.Tensor] = None,
    ) -> WorldStateMetrics:
        """
        Compute all world state metrics.

        Args:
            S_world_old: World state before update
            S_world_new: World state after update
            update_gate: GRU update gate values (z in GRU equations)
            reset_gate: GRU reset gate values (r in GRU equations)
            candidate: GRU candidate values (h_tilde in GRU equations)
            z_world: Projection of world state to z_0 space
            z_init: Input-based z_0 initialization (before world state added)

        Returns:
            WorldStateMetrics with all computed values
        """
        self.update_count += 1

        # Basic stats (on new state)
        basic = self.compute_basic_stats(S_world_new)

        # Distribution shape
        dist = self.compute_distribution_metrics(S_world_new)

        # Change metrics
        change = self.compute_change_metrics(S_world_old, S_world_new)

        # Gate metrics
        gates = self.compute_gate_metrics(update_gate, reset_gate, candidate)

        # Influence metrics
        influence = self.compute_influence_metrics(z_world, z_init)

        # Temporal metrics
        temporal = self.compute_temporal_metrics(change['delta_norm'])

        # Health indicators
        health = self.compute_health_indicators(S_world_new, change['delta_norm'])

        # Store snapshot periodically
        if self.update_count % 10 == 0:
            self.state_snapshots.append(S_world_new.clone().cpu())

        # Combine into metrics object
        metrics = WorldStateMetrics(
            # Basic
            norm=basic['norm'],
            mean=basic['mean'],
            std=basic['std'],
            min_val=basic['min_val'],
            max_val=basic['max_val'],
            # Distribution
            skewness=dist['skewness'],
            kurtosis=dist['kurtosis'],
            sparsity=dist['sparsity'],
            effective_rank=dist['effective_rank'],
            entropy=dist['entropy'],
            # Gates
            update_gate_mean=gates.get('update_gate_mean', 0.0),
            update_gate_std=gates.get('update_gate_std', 0.0),
            reset_gate_mean=gates.get('reset_gate_mean', 0.0),
            reset_gate_std=gates.get('reset_gate_std', 0.0),
            candidate_norm=gates.get('candidate_norm', 0.0),
            # Change
            delta_norm=change['delta_norm'],
            delta_relative=change['delta_relative'],
            cosine_similarity=change['cosine_similarity'],
            # Influence
            z_world_norm=influence.get('z_world_norm', 0.0),
            z_world_contribution=influence.get('z_world_contribution', 0.0),
            z_world_cosine_with_init=influence.get('z_world_cosine_with_init', 0.0),
            # Temporal
            velocity=temporal['velocity'],
            acceleration=temporal['acceleration'],
            stability_score=temporal['stability_score'],
            # Health
            is_saturated=health['is_saturated'],
            is_collapsed=health['is_collapsed'],
            is_exploding=health['is_exploding'],
            is_dead=health['is_dead'],
        )

        return metrics

    def format_for_logging(
        self,
        metrics: WorldStateMetrics,
        prefix: str = 'world_state',
    ) -> Dict[str, float]:
        """
        Format metrics for wandb/tensorboard logging.

        Returns dict with prefixed keys.
        """
        return {
            # === Basic Statistics ===
            f'{prefix}/norm': metrics.norm,
            f'{prefix}/mean': metrics.mean,
            f'{prefix}/std': metrics.std,
            f'{prefix}/min': metrics.min_val,
            f'{prefix}/max': metrics.max_val,

            # === Distribution Shape ===
            f'{prefix}/dist/skewness': metrics.skewness,
            f'{prefix}/dist/kurtosis': metrics.kurtosis,
            f'{prefix}/dist/sparsity': metrics.sparsity,
            f'{prefix}/dist/effective_rank': metrics.effective_rank,
            f'{prefix}/dist/entropy': metrics.entropy,

            # === GRU Gate Dynamics ===
            f'{prefix}/gate/update_mean': metrics.update_gate_mean,
            f'{prefix}/gate/update_std': metrics.update_gate_std,
            f'{prefix}/gate/reset_mean': metrics.reset_gate_mean,
            f'{prefix}/gate/reset_std': metrics.reset_gate_std,
            f'{prefix}/gate/candidate_norm': metrics.candidate_norm,

            # === Change Metrics ===
            f'{prefix}/change/delta_norm': metrics.delta_norm,
            f'{prefix}/change/delta_relative': metrics.delta_relative,
            f'{prefix}/change/cosine_sim': metrics.cosine_similarity,

            # === Influence on CTM ===
            f'{prefix}/influence/z_world_norm': metrics.z_world_norm,
            f'{prefix}/influence/contribution': metrics.z_world_contribution,
            f'{prefix}/influence/alignment': metrics.z_world_cosine_with_init,

            # === Temporal Dynamics ===
            f'{prefix}/temporal/velocity': metrics.velocity,
            f'{prefix}/temporal/acceleration': metrics.acceleration,
            f'{prefix}/temporal/stability': metrics.stability_score,

            # === Health Indicators (as floats for logging) ===
            f'{prefix}/health/saturated': float(metrics.is_saturated),
            f'{prefix}/health/collapsed': float(metrics.is_collapsed),
            f'{prefix}/health/exploding': float(metrics.is_exploding),
            f'{prefix}/health/dead': float(metrics.is_dead),
        }

    def get_health_summary(self, metrics: WorldStateMetrics) -> str:
        """Get a human-readable health summary."""
        issues = []

        if metrics.is_saturated:
            issues.append("SATURATED (values near boundaries)")
        if metrics.is_collapsed:
            issues.append("COLLAPSED (no variance)")
        if metrics.is_exploding:
            issues.append("EXPLODING (norm growing)")
        if metrics.is_dead:
            issues.append("DEAD (not updating)")

        if not issues:
            return "HEALTHY"
        return " | ".join(issues)

    def get_state_trajectory(self) -> Optional[torch.Tensor]:
        """
        Get trajectory of recent state snapshots.

        Returns:
            Tensor of shape (num_snapshots, d_sync_state) or None if no snapshots.
        """
        if len(self.state_snapshots) == 0:
            return None
        return torch.stack(list(self.state_snapshots), dim=0)


def create_world_state_visualizations(
    metrics: WorldStateMetrics,
    state_trajectory: Optional[torch.Tensor] = None,
    S_world: Optional[torch.Tensor] = None,
) -> Dict[str, 'plt.Figure']:
    """
    Create visualization figures for world state monitoring.

    Args:
        metrics: WorldStateMetrics from compute_all_metrics()
        state_trajectory: Optional (num_snapshots, d_sync_state) tensor of state history
        S_world: Optional current world state tensor

    Returns:
        Dict of figure names to matplotlib figures
    """
    import matplotlib.pyplot as plt
    import numpy as np

    figures = {}

    # === 1. Gate Dynamics Visualization ===
    if metrics.update_gate_mean > 0 or metrics.reset_gate_mean > 0:
        fig_gates, axes = plt.subplots(1, 3, figsize=(12, 3))

        # Update gate histogram (simulated from mean/std)
        ax = axes[0]
        x = np.linspace(0, 1, 100)
        # Approximate beta distribution from mean/std
        mean, std = metrics.update_gate_mean, max(metrics.update_gate_std, 0.01)
        ax.axvline(mean, color='blue', linewidth=2, label=f'Mean: {mean:.3f}')
        ax.axvspan(max(0, mean - std), min(1, mean + std), alpha=0.3, color='blue', label=f'Std: {std:.3f}')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Update Gate Value')
        ax.set_ylabel('Density')
        ax.set_title('Update Gate (z)')
        ax.legend(fontsize=8)

        # Reset gate
        ax = axes[1]
        mean, std = metrics.reset_gate_mean, max(metrics.reset_gate_std, 0.01)
        ax.axvline(mean, color='red', linewidth=2, label=f'Mean: {mean:.3f}')
        ax.axvspan(max(0, mean - std), min(1, mean + std), alpha=0.3, color='red', label=f'Std: {std:.3f}')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Reset Gate Value')
        ax.set_title('Reset Gate (r)')
        ax.legend(fontsize=8)

        # Candidate magnitude
        ax = axes[2]
        ax.bar(['Candidate\nNorm', 'Delta\nNorm', 'State\nNorm'],
               [metrics.candidate_norm, metrics.delta_norm, metrics.norm],
               color=['green', 'orange', 'purple'])
        ax.set_ylabel('Magnitude')
        ax.set_title('Update Magnitudes')

        plt.tight_layout()
        figures['gate_dynamics'] = fig_gates

    # === 2. World State Distribution ===
    if S_world is not None:
        fig_dist, axes = plt.subplots(1, 3, figsize=(12, 3))

        S_np = S_world.detach().cpu().numpy()

        # Histogram of values
        ax = axes[0]
        ax.hist(S_np, bins=50, density=True, alpha=0.7, color='blue')
        ax.axvline(S_np.mean(), color='red', linestyle='--', label=f'Mean: {S_np.mean():.3f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('World State Distribution')
        ax.legend(fontsize=8)

        # Sorted values (to see rank structure)
        ax = axes[1]
        sorted_vals = np.sort(np.abs(S_np))[::-1]
        ax.plot(sorted_vals, 'b-', linewidth=1)
        ax.set_xlabel('Dimension (sorted by |value|)')
        ax.set_ylabel('|Value|')
        ax.set_title(f'Sorted Magnitudes (eff. rank: {metrics.effective_rank:.1f})')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Cumulative energy
        ax = axes[2]
        energy = np.cumsum(sorted_vals ** 2) / (np.sum(sorted_vals ** 2) + 1e-8)
        ax.plot(energy, 'g-', linewidth=2)
        ax.axhline(0.9, color='red', linestyle='--', label='90% energy')
        dims_90 = np.searchsorted(energy, 0.9)
        ax.axvline(dims_90, color='red', linestyle=':', alpha=0.5)
        ax.set_xlabel('Number of dimensions')
        ax.set_ylabel('Cumulative energy fraction')
        ax.set_title(f'Energy concentration ({dims_90} dims for 90%)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        figures['state_distribution'] = fig_dist

    # === 3. State Trajectory (if available) ===
    if state_trajectory is not None and len(state_trajectory) > 1:
        fig_traj, axes = plt.subplots(2, 2, figsize=(10, 8))

        traj_np = state_trajectory.detach().cpu().numpy()
        num_snapshots, d_sync = traj_np.shape

        # Norm over time
        ax = axes[0, 0]
        norms = np.linalg.norm(traj_np, axis=1)
        ax.plot(norms, 'b-o', markersize=4)
        ax.set_xlabel('Snapshot')
        ax.set_ylabel('Norm')
        ax.set_title('World State Norm Over Time')
        ax.grid(True, alpha=0.3)

        # Inter-snapshot cosine similarity
        ax = axes[0, 1]
        cosines = []
        for i in range(1, num_snapshots):
            cos = np.dot(traj_np[i], traj_np[i-1]) / (
                np.linalg.norm(traj_np[i]) * np.linalg.norm(traj_np[i-1]) + 1e-8
            )
            cosines.append(cos)
        ax.plot(range(1, num_snapshots), cosines, 'r-o', markersize=4)
        ax.set_xlabel('Snapshot')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Direction Stability (cos with previous)')
        ax.set_ylim(-1, 1)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Heatmap of top dimensions
        ax = axes[1, 0]
        # Select top-k most varying dimensions
        var_per_dim = np.var(traj_np, axis=0)
        top_k = min(32, d_sync)
        top_dims = np.argsort(var_per_dim)[-top_k:]
        im = ax.imshow(traj_np[:, top_dims].T, aspect='auto', cmap='RdBu_r')
        ax.set_xlabel('Snapshot')
        ax.set_ylabel('Dimension (top varying)')
        ax.set_title(f'Top {top_k} Varying Dimensions')
        plt.colorbar(im, ax=ax)

        # PCA projection (2D)
        ax = axes[1, 1]
        if num_snapshots >= 2:
            # Simple 2D PCA
            centered = traj_np - traj_np.mean(axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            proj_2d = centered @ Vt[:2].T
            colors = np.arange(num_snapshots)
            scatter = ax.scatter(proj_2d[:, 0], proj_2d[:, 1], c=colors, cmap='viridis', s=50)
            # Draw trajectory line
            ax.plot(proj_2d[:, 0], proj_2d[:, 1], 'k-', alpha=0.3, linewidth=1)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('Trajectory in PCA Space')
            plt.colorbar(scatter, ax=ax, label='Snapshot')

        plt.tight_layout()
        figures['state_trajectory'] = fig_traj

    # === 4. Health Dashboard ===
    fig_health, ax = plt.subplots(1, 1, figsize=(8, 4))

    health_indicators = [
        ('Saturated', metrics.is_saturated, 'Values near ±1'),
        ('Collapsed', metrics.is_collapsed, 'No variance'),
        ('Exploding', metrics.is_exploding, 'Norm growing'),
        ('Dead', metrics.is_dead, 'Not updating'),
    ]

    colors = []
    for name, is_bad, desc in health_indicators:
        colors.append('red' if is_bad else 'green')

    y_pos = np.arange(len(health_indicators))
    bars = ax.barh(y_pos, [1] * len(health_indicators), color=colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{name}: {desc}" for name, _, desc in health_indicators])
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])

    # Add status text
    for i, (name, is_bad, _) in enumerate(health_indicators):
        status = "ISSUE" if is_bad else "OK"
        ax.text(1.1, i, status, va='center', fontweight='bold',
                color='red' if is_bad else 'green')

    ax.set_title('World State Health Check')
    ax.invert_yaxis()

    plt.tight_layout()
    figures['health_dashboard'] = fig_health

    return figures


# ============================================================================
# OSCILLATOR MONITORING (NEW)
# ============================================================================

@dataclass
class OscillatorTrajectory:
    """Container for oscillator trajectory over time."""
    phases: List[torch.Tensor]           # List of (num_oscillators,) phase tensors
    amplitudes: List[torch.Tensor]       # List of (num_oscillators,) amplitude tensors
    amp_modulations: List[torch.Tensor]  # List of (num_oscillators,) amp mod tensors
    phase_modulations: List[torch.Tensor]  # List of (num_oscillators,) phase mod tensors
    outputs: List[torch.Tensor]          # List of (d_output,) output tensors


class OscillatorMonitor:
    """
    Tracks oscillator state over time for visualization and analysis.

    Unlike the GRU-based WorldStateMonitor, this is designed for the
    oscillatory world model where state evolves continuously.
    """

    def __init__(
        self,
        num_oscillators: int = 64,
        history_length: int = 100,
    ):
        self.num_oscillators = num_oscillators
        self.history_length = history_length

        # History tracking
        self.phase_history: deque = deque(maxlen=history_length)
        self.amplitude_history: deque = deque(maxlen=history_length)
        self.amp_mod_history: deque = deque(maxlen=history_length)
        self.phase_mod_history: deque = deque(maxlen=history_length)
        self.output_history: deque = deque(maxlen=history_length)

        self.update_count: int = 0

    def reset(self):
        """Reset all tracking history."""
        self.phase_history.clear()
        self.amplitude_history.clear()
        self.amp_mod_history.clear()
        self.phase_mod_history.clear()
        self.output_history.clear()
        self.update_count = 0

    @torch.no_grad()
    def record(
        self,
        phases: torch.Tensor,
        amplitudes: torch.Tensor,
        amp_mod: torch.Tensor,
        phase_mod: torch.Tensor,
        output: torch.Tensor,
    ):
        """Record current oscillator state."""
        self.phase_history.append(phases.clone().cpu())
        self.amplitude_history.append(amplitudes.clone().cpu())
        self.amp_mod_history.append(amp_mod.clone().cpu())
        self.phase_mod_history.append(phase_mod.clone().cpu())
        self.output_history.append(output.clone().cpu())
        self.update_count += 1

    def get_trajectory(self) -> Optional[OscillatorTrajectory]:
        """Get the recorded trajectory."""
        if len(self.phase_history) == 0:
            return None
        return OscillatorTrajectory(
            phases=list(self.phase_history),
            amplitudes=list(self.amplitude_history),
            amp_modulations=list(self.amp_mod_history),
            phase_modulations=list(self.phase_mod_history),
            outputs=list(self.output_history),
        )


def create_oscillator_visualizations(
    osc_state: Dict[str, torch.Tensor],
    trajectory: Optional[OscillatorTrajectory] = None,
) -> Dict[str, 'plt.Figure']:
    """
    Create visualization figures for oscillatory world model.

    Args:
        osc_state: Current oscillator state from OscillatoryWorldState.get_oscillator_state()
        trajectory: Optional trajectory from OscillatorMonitor

    Returns:
        Dict of figure names to matplotlib figures
    """
    import matplotlib.pyplot as plt
    import numpy as np

    figures = {}

    # === 1. Current Oscillator State ===
    fig_state, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Frequencies (log scale)
    ax = axes[0, 0]
    freqs = osc_state['frequencies'].cpu().numpy()
    periods = 1.0 / freqs
    ax.bar(range(len(freqs)), periods)
    ax.set_xlabel('Oscillator Index')
    ax.set_ylabel('Period (steps)')
    ax.set_title('Oscillator Periods')
    ax.set_yscale('log')

    # Current amplitudes
    ax = axes[0, 1]
    amps = osc_state['current_amplitudes'].cpu().numpy()
    base_amps = osc_state['base_amplitudes'].cpu().numpy()
    x = range(len(amps))
    ax.bar(x, base_amps, alpha=0.5, label='Base')
    ax.bar(x, amps, alpha=0.7, label='Current')
    ax.set_xlabel('Oscillator Index')
    ax.set_ylabel('Amplitude')
    ax.set_title('Amplitudes (Base vs Modulated)')
    ax.legend()

    # Current phases (polar plot)
    ax = axes[1, 0]
    phases = osc_state['phases'].cpu().numpy()
    # Color by frequency (slow=blue, fast=red)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(phases)))
    ax.scatter(range(len(phases)), phases, c=colors, s=50)
    ax.axhline(np.pi, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Oscillator Index')
    ax.set_ylabel('Phase (radians)')
    ax.set_title('Current Phases')
    ax.set_ylim(0, 2 * np.pi)

    # Modulation values
    ax = axes[1, 1]
    amp_mod = osc_state['last_amp_mod'].cpu().numpy()
    phase_mod = osc_state['last_phase_mod'].cpu().numpy()
    width = 0.35
    x = np.arange(len(amp_mod))
    ax.bar(x - width/2, amp_mod, width, label='Amplitude mod', alpha=0.7)
    ax.bar(x + width/2, phase_mod, width, label='Phase mod', alpha=0.7)
    ax.set_xlabel('Oscillator Index')
    ax.set_ylabel('Modulation')
    ax.set_title('Current Modulation Values')
    ax.legend()
    ax.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    figures['oscillator_state'] = fig_state

    # === 2. Trajectory visualization (if available) ===
    if trajectory is not None and len(trajectory.phases) > 1:
        num_snapshots = len(trajectory.phases)
        phases_np = torch.stack(trajectory.phases, dim=0).numpy()  # (T, N)
        amps_np = torch.stack(trajectory.amplitudes, dim=0).numpy()  # (T, N)

        fig_traj, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Phase evolution heatmap
        ax = axes[0, 0]
        im = ax.imshow(phases_np.T, aspect='auto', cmap='hsv', vmin=0, vmax=2*np.pi)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Oscillator Index')
        ax.set_title('Phase Evolution')
        plt.colorbar(im, ax=ax, label='Phase (rad)')

        # Amplitude evolution heatmap
        ax = axes[0, 1]
        im = ax.imshow(amps_np.T, aspect='auto', cmap='viridis')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Oscillator Index')
        ax.set_title('Amplitude Evolution')
        plt.colorbar(im, ax=ax, label='Amplitude')

        # Sample oscillator traces
        ax = axes[1, 0]
        sample_indices = np.linspace(0, len(phases_np[0]) - 1, min(8, len(phases_np[0]))).astype(int)
        for i in sample_indices:
            # Compute oscillator output: amp * sin(phase)
            output = amps_np[:, i] * np.sin(phases_np[:, i])
            period = 1.0 / osc_state['frequencies'][i].cpu().numpy()
            ax.plot(output, alpha=0.7, label=f'T={period:.0f}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Output Value')
        ax.set_title('Sample Oscillator Outputs')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Output norm over time
        ax = axes[1, 1]
        output_norms = [o.norm().item() for o in trajectory.outputs]
        ax.plot(output_norms, 'b-', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Output Norm')
        ax.set_title('World State Output Norm Over Time')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        figures['oscillator_trajectory'] = fig_traj

    return figures


# ============================================================================
# LEGACY GRU-BASED MONITORING (Preserved for backwards compatibility)
# ============================================================================

class SyncUpdateGateWithMonitoring(nn.Module):
    """
    [LEGACY] Extended SyncUpdateGate that exposes all internal values for monitoring.

    NOTE: This class is preserved for backwards compatibility but is no longer
    used with the new oscillatory world model.
    """

    def __init__(self, d_sync: int = 256):
        super().__init__()
        self.d_sync = d_sync

        # Reset gate: what to forget from old state
        self.reset_gate = nn.Linear(d_sync * 2, d_sync)
        # Update gate: how much to update
        self.update_gate = nn.Linear(d_sync * 2, d_sync)
        # Candidate: what new information to add
        self.candidate = nn.Linear(d_sync * 2, d_sync)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable learning."""
        for module in [self.reset_gate, self.update_gate, self.candidate]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        # Initialize update gate bias negative so initial gate is ~0.1
        with torch.no_grad():
            self.update_gate.bias.data.fill_(-2.0)  # sigmoid(-2) ≈ 0.12

    def forward(
        self,
        S_world: torch.Tensor,
        S_new: torch.Tensor,
        return_gates: bool = False,
    ) -> torch.Tensor:
        """
        GRU-style update with optional gate value returns.

        Args:
            S_world: Current world state
            S_new: New sync observation
            return_gates: If True, return (output, gate_dict)

        Returns:
            Updated world state, or (updated, gate_dict) if return_gates=True
        """
        combined = torch.cat([S_world, S_new], dim=-1)

        r = torch.sigmoid(self.reset_gate(combined))    # Reset gate
        z = torch.sigmoid(self.update_gate(combined))   # Update gate

        reset_combined = torch.cat([r * S_world, S_new], dim=-1)
        candidate = torch.tanh(self.candidate(reset_combined))

        # GRU update: (1-z) keeps old, z incorporates new
        output = (1 - z) * S_world + z * candidate

        if return_gates:
            gate_dict = {
                'reset_gate': r,
                'update_gate': z,
                'candidate': candidate,
            }
            return output, gate_dict

        return output
