"""
Training script for PEM Loop with Global Sync Architecture.

Trains PredictionCTM + SurpriseCTM + GlobalSyncModule end-to-end.

Usage:
    python -m pem.train_pem_global --wandb_project pem-global --batch_size 4

Monitors:
    - Loss breakdown (prediction, surprise calibration, sync variance)
    - Cross-module synchronization patterns
    - Module contributions over time
    - Attention entropy
    - Per-tick outputs from CTM modules
"""

import argparse
import math
import os
import random
import time
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from .pem_loop_global import PEMLoopGlobal, PEMLoopGlobalConfig, create_pem_loop_global
from .janus_pro_feature_extractor import JanusProFeatureExtractor, JanusProConfig


def create_nlm_activation_grid(
    outputs: List,
    max_neurons: int = 64,
    max_positions: int = 32,
) -> plt.Figure:
    """
    Create a grid visualization of NLM activations over time.

    Grid layout:
        Rows: Loop steps (e.g., step 0, step 1, ...)
        Columns: Modules (Prediction, Surprise)

    Each cell shows: Neurons (y-axis) x Ticks (x-axis) heatmap

    Args:
        outputs: List of PEMLoopGlobalOutput from forward pass
        max_neurons: Max neurons to show (subsample if more)
        max_positions: Max sequence positions to average over

    Returns:
        matplotlib Figure
    """
    num_steps = len(outputs)
    # Check if surprise is enabled
    has_surprise = outputs[0].surprise is not None
    num_modules = 2 if has_surprise else 1

    fig, axes = plt.subplots(
        num_steps, num_modules,
        figsize=(4 * num_modules, 3 * num_steps),
        squeeze=False,
    )

    module_names = ['Prediction', 'Surprise'] if has_surprise else ['Prediction']

    # First pass: collect all data to find global min/max for colorbar
    all_values = []

    for step_idx, output in enumerate(outputs):
        pred_activations = output.prediction_output.all_tick_activations
        surp_activations = output.surprise.all_tick_activations if has_surprise else None

        activations_list = [pred_activations]
        if surp_activations is not None:
            activations_list.append(surp_activations)

        for activations in activations_list:
            if not activations:
                continue
            stacked = torch.stack(activations, dim=0)
            T, B, S, D = stacked.shape
            pos_subset = min(S, max_positions)
            neuron_subset = min(D, max_neurons)
            neuron_indices = torch.linspace(0, D-1, neuron_subset).long()
            averaged = stacked[:, 0, :pos_subset, :][:, :, neuron_indices].mean(dim=1)
            all_values.extend(averaged.detach().cpu().numpy().flatten().tolist())

    # Compute symmetric color limits centered at 0
    if all_values:
        abs_max = max(abs(min(all_values)), abs(max(all_values)))
        if abs_max < 1e-6:
            abs_max = 0.1  # Minimum range for visibility
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = -1, 1

    # Second pass: plot
    for step_idx, output in enumerate(outputs):
        pred_activations = output.prediction_output.all_tick_activations
        surp_activations = output.surprise.all_tick_activations if has_surprise else None

        module_activations = [pred_activations]
        if surp_activations is not None:
            module_activations.append(surp_activations)

        for mod_idx, (name, activations) in enumerate(zip(module_names, module_activations)):
            ax = axes[step_idx, mod_idx]

            if not activations:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'{name} (Step {step_idx})')
                continue

            stacked = torch.stack(activations, dim=0)
            T, B, S, D = stacked.shape

            pos_subset = min(S, max_positions)
            averaged = stacked[:, 0, :pos_subset, :].mean(dim=1)

            neuron_subset = min(D, max_neurons)
            neuron_indices = torch.linspace(0, D-1, neuron_subset).long()
            averaged = averaged[:, neuron_indices]

            data = averaged.detach().cpu().numpy()

            im = ax.imshow(
                data.T,
                aspect='auto',
                cmap='RdBu_r',
                vmin=vmin, vmax=vmax,
            )

            ax.set_xlabel('Tick')
            ax.set_ylabel('Neuron')
            ax.set_title(f'{name} (Step {step_idx})')

            ax.set_xticks(range(T))
            ax.set_xticklabels([f't{t}' for t in range(T)])

    # Add colorbar with actual range info
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, label=f'Activation (range: ±{vmax:.4f})')

    plt.tight_layout()
    return fig


def create_nlm_neuron_lines(
    outputs: List,
    num_neurons: int = 16,
    max_positions: int = 32,
) -> plt.Figure:
    """
    Create a grid of line plots showing individual neuron activations over ticks.

    Grid layout:
        Rows: Neurons (showing num_neurons neurons)
        Columns: Modules (Prediction, Surprise)

    Each cell shows: Line plot of activation value (y) vs tick (x)
    Different colors for different loop steps.

    Args:
        outputs: List of PEMLoopGlobalOutput from forward pass
        num_neurons: Number of neurons to show (evenly sampled)
        max_positions: Max sequence positions to average over

    Returns:
        matplotlib Figure
    """
    num_steps = len(outputs)
    # Check if surprise is enabled
    has_surprise = outputs[0].surprise is not None
    num_modules = 2 if has_surprise else 1

    # Calculate grid size (sqrt layout for neurons)
    grid_rows = int(np.ceil(np.sqrt(num_neurons)))
    grid_cols = int(np.ceil(num_neurons / grid_rows))

    # Create figure with subplots for each module
    fig, axes = plt.subplots(
        grid_rows, grid_cols * num_modules,
        figsize=(3 * grid_cols * num_modules, 2 * grid_rows),
        squeeze=False,
    )

    module_names = ['Prediction', 'Surprise'] if has_surprise else ['Prediction']
    colors = plt.cm.viridis(np.linspace(0, 1, num_steps))

    # First pass: collect all data to determine y-axis range per module
    all_data = {0: [], 1: []}  # mod_idx -> list of values

    for mod_idx, name in enumerate(module_names):
        if mod_idx == 0:
            sample_activations = outputs[0].prediction_output.all_tick_activations
        else:
            sample_activations = outputs[0].surprise.all_tick_activations if has_surprise else None

        if not sample_activations:
            continue

        total_neurons = sample_activations[0].shape[-1]
        neuron_indices = torch.linspace(0, total_neurons - 1, num_neurons).long()

        for neuron_plot_idx in range(num_neurons):
            neuron_idx = neuron_indices[neuron_plot_idx].item()

            for step_idx, output in enumerate(outputs):
                if mod_idx == 0:
                    activations = output.prediction_output.all_tick_activations
                else:
                    activations = output.surprise.all_tick_activations if has_surprise else None

                if not activations:
                    continue

                stacked = torch.stack(activations, dim=0)
                T, B, S, D = stacked.shape
                pos_subset = min(S, max_positions)
                neuron_activation = stacked[:, 0, :pos_subset, neuron_idx].mean(dim=1)
                all_data[mod_idx].extend(neuron_activation.detach().cpu().numpy().tolist())

    # Compute y-axis limits per module (with padding)
    y_limits = {}
    for mod_idx in range(num_modules):
        if all_data[mod_idx]:
            data_min = min(all_data[mod_idx])
            data_max = max(all_data[mod_idx])
            data_range = data_max - data_min
            if data_range < 1e-6:  # Nearly flat - use small fixed range
                mid = (data_min + data_max) / 2
                y_limits[mod_idx] = (mid - 0.1, mid + 0.1)
            else:
                padding = data_range * 0.15
                y_limits[mod_idx] = (data_min - padding, data_max + padding)
        else:
            y_limits[mod_idx] = (-1, 1)

    # Second pass: plot with proper y-limits
    for mod_idx, name in enumerate(module_names):
        if mod_idx == 0:
            sample_activations = outputs[0].prediction_output.all_tick_activations
        else:
            sample_activations = outputs[0].surprise.all_tick_activations if has_surprise else None

        if not sample_activations:
            continue

        total_neurons = sample_activations[0].shape[-1]
        neuron_indices = torch.linspace(0, total_neurons - 1, num_neurons).long()

        for neuron_plot_idx in range(num_neurons):
            row = neuron_plot_idx // grid_cols
            col = (mod_idx * grid_cols) + (neuron_plot_idx % grid_cols)
            ax = axes[row, col]

            neuron_idx = neuron_indices[neuron_plot_idx].item()

            for step_idx, output in enumerate(outputs):
                if mod_idx == 0:
                    activations = output.prediction_output.all_tick_activations
                else:
                    activations = output.surprise.all_tick_activations if has_surprise else None

                if not activations:
                    continue

                stacked = torch.stack(activations, dim=0)
                T, B, S, D = stacked.shape
                pos_subset = min(S, max_positions)
                neuron_activation = stacked[:, 0, :pos_subset, neuron_idx].mean(dim=1)

                data = neuron_activation.detach().cpu().numpy()
                ax.plot(range(T), data, 'o-', color=colors[step_idx],
                       label=f'Step {step_idx}' if neuron_plot_idx == 0 else None,
                       linewidth=1.5, markersize=4)

            ax.set_title(f'{name} N{neuron_idx}', fontsize=8)
            ax.set_xlabel('Tick', fontsize=7)
            ax.set_ylabel('Act', fontsize=7)
            ax.tick_params(axis='both', labelsize=6)
            ax.grid(True, alpha=0.3)

            # Auto-scaled y-axis per module
            ax.set_ylim(y_limits[mod_idx])

    # Add legend to first subplot
    if num_steps > 1:
        axes[0, 0].legend(fontsize=6, loc='upper right')

    # Add module labels with y-range info
    pred_range = y_limits.get(0, (-1, 1))
    if has_surprise:
        surp_range = y_limits.get(1, (-1, 1))
        fig.text(0.25, 0.98, f'Prediction Module (y: {pred_range[0]:.3f} to {pred_range[1]:.3f})',
                 ha='center', fontsize=10, fontweight='bold')
        fig.text(0.75, 0.98, f'Surprise Module (y: {surp_range[0]:.3f} to {surp_range[1]:.3f})',
                 ha='center', fontsize=10, fontweight='bold')
    else:
        fig.text(0.5, 0.98, f'Prediction Module (y: {pred_range[0]:.3f} to {pred_range[1]:.3f})',
                 ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def create_cross_module_sync_plot(
    outputs: List,
) -> Optional[plt.Figure]:
    """
    Create a plot showing cross-module synchronization over loop steps.

    Shows how Prediction and Surprise modules synchronize over time.
    Returns None if surprise is disabled (single module mode).
    """
    num_steps = len(outputs)

    # Check if we have 2 modules
    cross_sync = outputs[0].global_sync.cross_module_sync
    if cross_sync.shape[0] < 2:
        # Single module mode - no cross-module sync to plot
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Cross-module sync values over steps
    ax1 = axes[0]
    pred_surp_sync = []
    surp_pred_sync = []

    for output in outputs:
        cross_sync = output.global_sync.cross_module_sync  # (2, 2, B, S)
        pred_surp_sync.append(cross_sync[0, 1].mean().item())
        surp_pred_sync.append(cross_sync[1, 0].mean().item())

    steps = range(num_steps)
    ax1.plot(steps, pred_surp_sync, 'b-o', label='Pred→Surp')
    ax1.plot(steps, surp_pred_sync, 'r-o', label='Surp→Pred')
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax1.set_xlabel('Loop Step')
    ax1.set_ylabel('Cross-Module Sync')
    ax1.set_title('Cross-Module Synchronization')
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Plot 2: Module contributions over steps
    ax2 = axes[1]
    pred_contrib = []
    surp_contrib = []

    for output in outputs:
        contrib = output.global_sync.module_contributions  # (B, S, 2)
        pred_contrib.append(contrib[..., 0].mean().item())
        surp_contrib.append(contrib[..., 1].mean().item())

    ax2.bar(np.array(steps) - 0.15, pred_contrib, 0.3, label='Prediction', color='blue', alpha=0.7)
    ax2.bar(np.array(steps) + 0.15, surp_contrib, 0.3, label='Surprise', color='red', alpha=0.7)
    ax2.set_xlabel('Loop Step')
    ax2.set_ylabel('Module Contribution')
    ax2.set_title('Module Contributions to Global Sync')
    ax2.legend()
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    return fig


def fig_to_image(fig: plt.Figure):
    """Convert matplotlib figure to PIL Image for wandb logging."""
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    d_model: int = 1536
    pred_d_neurons: int = 256
    surp_d_neurons: int = 128
    pred_T: int = 4
    surp_T: int = 3
    sync_pairs: int = 256

    # Training
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 100
    grad_clip: float = 1.0
    num_epochs: int = 0  # If >0, train for this many epochs (overrides max_steps)

    # Loop
    num_loop_steps: int = 2

    # Logging
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 1000

    # Data
    max_length: int = 512

    # WandB
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


def create_optimizer(model: nn.Module, config: TrainingConfig):
    """Create AdamW optimizer with weight decay."""
    # Separate parameters that should/shouldn't have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'embedding' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=config.learning_rate)

    return optimizer


def create_scheduler(optimizer, config: TrainingConfig):
    """Create learning rate scheduler with warmup."""
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        # Cosine decay
        progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_detailed_metrics(
    outputs: List,
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute consolidated metrics for logging.

    Now tracks TWO levels of iteration:
    1. Loop steps (outer PEM loop iterations)
    2. Internal ticks (within each CTM module) - used by CTM loss

    Returns:
    - Overall best loss/certainty step (loop level)
    - Per-module best steps (loop level)
    - Internal tick selection by CTM loss (tick level)
    - Key summary metrics (surprise, sync)
    """
    metrics = {}

    def safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.float().reshape(-1)
        b = b.float().reshape(-1)
        if a.numel() == 0 or b.numel() == 0:
            return 0.0
        a = a - a.mean()
        b = b - b.mean()
        denom = a.std(unbiased=False) * b.std(unbiased=False) + 1e-8
        denom_val = denom.item()
        if denom_val == 0:
            return 0.0
        return (a * b).mean().item() / denom_val

    # Track per-step metrics for finding best step (LOOP level)
    step_losses = []           # Overall prediction loss
    step_certainties = []      # Combined certainty

    # Per-module tracking (LOOP level)
    step_pred_losses = []      # PredictionCTM loss only
    step_pred_certainties = [] # PredictionCTM certainty
    step_surp_losses = []      # SurpriseCTM calibration loss
    step_surp_certainties = [] # SurpriseCTM certainty

    # Internal TICK tracking (for CTM loss monitoring)
    pred_best_ticks = []       # Which internal tick was best for prediction
    pred_certain_ticks = []    # Which internal tick was most certain
    surp_best_ticks = []       # Which internal tick was best for surprise
    surp_certain_ticks = []    # Which internal tick was most certain
    pred_tick_spreads = []     # Std of tick losses (diversity)

    for step_idx, output in enumerate(outputs):
        step_loss = 0.0

        # Prediction loss (cosine similarity) - from FINAL tick
        for scale in ['immediate', 'shortterm', 'longterm']:
            pred = output.predictions[scale]
            target = targets[scale]
            valid = targets.get(f'{scale}_valid', None)

            if valid is not None and valid.any():
                pred_valid = pred[valid]
                target_valid = target[valid]
                cos_sim = F.cosine_similarity(pred_valid, target_valid, dim=-1).mean()
                step_loss += (1 - cos_sim).item()

        # Module-specific metrics
        pred_certainty = output.prediction_output.certainty.item()
        surp_out = output.surprise

        if surp_out is not None:
            surp_certainty = surp_out.certainty.item()
            # Surprise calibration loss (how well magnitude tracks raw)
            surp_cal_loss = F.mse_loss(surp_out.magnitude, surp_out.raw).item()
        else:
            surp_certainty = 0.0
            surp_cal_loss = 0.0

        # Store per-step values (LOOP level)
        step_losses.append(step_loss)
        step_certainties.append(pred_certainty if surp_out is None else (pred_certainty + surp_certainty) / 2)

        step_pred_losses.append(step_loss)
        step_pred_certainties.append(pred_certainty)
        step_surp_losses.append(surp_cal_loss)
        step_surp_certainties.append(surp_certainty)

        # ===== CTM INTERNAL TICK ANALYSIS =====
        # Find which internal ticks were selected by CTM loss
        pred_out = output.prediction_output

        # Prediction: compute loss at each internal tick using y_t directly
        if hasattr(pred_out, 'all_tick_outputs') and pred_out.all_tick_outputs:
            tick_losses = []
            target = targets['immediate']  # Use immediate target for y_t comparison
            valid = targets.get('immediate_valid', None)

            for y_t in pred_out.all_tick_outputs:
                if valid is not None and valid.any():
                    cos_sim = F.cosine_similarity(y_t[valid], target[valid], dim=-1).mean()
                    tick_loss = (1 - cos_sim).item()
                else:
                    tick_loss = 0.0
                tick_losses.append(tick_loss)

            pred_best_ticks.append(int(np.argmin(tick_losses)))
            if tick_losses:
                pred_tick_spreads.append(float(np.std(tick_losses)))

            # Compute certainties from output stability
            if len(pred_out.all_tick_outputs) > 1:
                tick_certs = []
                for t in range(len(pred_out.all_tick_outputs)):
                    if t == 0:
                        tick_certs.append(0.1)
                    else:
                        # Simple stability measure
                        change = (pred_out.all_tick_outputs[t] - pred_out.all_tick_outputs[t-1]).norm().item()
                        tick_certs.append(np.exp(-change * 5.0))
                pred_certain_ticks.append(int(np.argmax(tick_certs)))

        # Surprise: find best internal ticks
        if surp_out is not None and hasattr(surp_out, 'all_tick_magnitudes') and surp_out.all_tick_magnitudes:
            tick_losses = []
            for tick_mag in surp_out.all_tick_magnitudes:
                surp_loss = F.mse_loss(tick_mag, surp_out.raw).item()
                tick_losses.append(surp_loss)
            surp_best_ticks.append(int(np.argmin(tick_losses)))

            # Compute certainties from output stability
            if hasattr(surp_out, 'all_tick_outputs') and len(surp_out.all_tick_outputs) > 1:
                tick_certs = []
                for t in range(len(surp_out.all_tick_outputs)):
                    if t == 0:
                        tick_certs.append(0.1)
                    else:
                        change = (surp_out.all_tick_outputs[t] - surp_out.all_tick_outputs[t-1]).norm().item()
                        tick_certs.append(np.exp(-change * 5.0))
                surp_certain_ticks.append(int(np.argmax(tick_certs)))

    # Find best steps for this batch (LOOP level)
    if len(outputs) > 0:
        # Overall best steps
        best_loss_step = int(np.argmin(step_losses))
        best_certainty_step = int(np.argmax(step_certainties))

        metrics['best_loss_step'] = best_loss_step
        metrics['best_loss_value'] = step_losses[best_loss_step]
        metrics['best_certainty_step'] = best_certainty_step
        metrics['best_certainty_value'] = step_certainties[best_certainty_step]

        # PredictionCTM best steps (LOOP level)
        metrics['pred_best_loss_step'] = int(np.argmin(step_pred_losses))
        metrics['pred_best_loss_value'] = step_pred_losses[metrics['pred_best_loss_step']]
        metrics['pred_best_cert_step'] = int(np.argmax(step_pred_certainties))
        metrics['pred_best_cert_value'] = step_pred_certainties[metrics['pred_best_cert_step']]

        # SurpriseCTM best steps (LOOP level)
        metrics['surp_best_loss_step'] = int(np.argmin(step_surp_losses))
        metrics['surp_best_loss_value'] = step_surp_losses[metrics['surp_best_loss_step']]
        metrics['surp_best_cert_step'] = int(np.argmax(step_surp_certainties))
        metrics['surp_best_cert_value'] = step_surp_certainties[metrics['surp_best_cert_step']]

        # ===== CTM INTERNAL TICK METRICS =====
        # Average which internal ticks are being selected
        if pred_best_ticks:
            metrics['ctm_pred_avg_best_tick'] = float(np.mean(pred_best_ticks))
            metrics['ctm/pred_t1_mean'] = float(np.mean(pred_best_ticks))
        if pred_certain_ticks:
            metrics['ctm_pred_avg_certain_tick'] = float(np.mean(pred_certain_ticks))
            metrics['ctm/pred_t2_mean'] = float(np.mean(pred_certain_ticks))
        if surp_best_ticks:
            metrics['ctm_surp_avg_best_tick'] = float(np.mean(surp_best_ticks))
        if surp_certain_ticks:
            metrics['ctm_surp_avg_certain_tick'] = float(np.mean(surp_certain_ticks))
        if pred_best_ticks and pred_certain_ticks:
            pair_count = min(len(pred_best_ticks), len(pred_certain_ticks))
            agreement = sum(
                1 for i in range(pair_count) if pred_best_ticks[i] == pred_certain_ticks[i]
            ) / max(pair_count, 1)
            metrics['ctm/t1_t2_agreement'] = float(agreement)
        if pred_tick_spreads:
            metrics['ctm/tick_loss_spread'] = float(np.mean(pred_tick_spreads))

        # Final step metrics (for monitoring convergence)
        metrics['final_step_loss'] = step_losses[-1]
        metrics['final_step_certainty'] = step_certainties[-1]
        metrics['final_step_surprise'] = step_surp_losses[-1]

        # Cross-module sync (from final step)
        final_sync = outputs[-1].global_sync
        cross_sync = final_sync.cross_module_sync  # (num_modules, num_modules, B, S)
        if cross_sync.shape[0] >= 2:
            metrics['cross_sync_pred_surp'] = cross_sync[0, 1].mean().item()
            metrics['cross_sync_surp_pred'] = cross_sync[1, 0].mean().item()
        else:
            # Single module mode - no cross-module sync
            metrics['cross_sync_pred_surp'] = 0.0
            metrics['cross_sync_surp_pred'] = 0.0

        # Loop loss trajectory
        metrics['loop/loss_step0'] = step_losses[0] if len(step_losses) > 0 else 0.0
        metrics['loop/loss_step1'] = step_losses[1] if len(step_losses) > 1 else 0.0
        metrics['loop/loss_stepN'] = step_losses[-1] if len(step_losses) > 0 else 0.0
        if len(step_losses) > 1:
            monotonic = sum(
                1 for i in range(1, len(step_losses)) if step_losses[i] <= step_losses[i - 1]
            ) / (len(step_losses) - 1)
            metrics['loop/monotonic_improve'] = float(monotonic)

        # Oscillator attention entropy evolution across loop steps
        e0 = outputs[0].global_sync.osc_attn_entropy if outputs[0].global_sync is not None else None
        eN = outputs[-1].global_sync.osc_attn_entropy if outputs[-1].global_sync is not None else None
        if e0 is not None and eN is not None:
            metrics['osc_xattn/entropy_step0'] = float(e0)
            metrics['osc_xattn/entropy_stepN'] = float(eN)
            metrics['osc_xattn/entropy_delta'] = float(eN - e0)

    # Improvement across steps (did iterating help?)
    if len(outputs) > 1:
        metrics['loss_improvement'] = step_losses[0] - step_losses[-1]  # Positive = improved
        metrics['certainty_improvement'] = step_certainties[-1] - step_certainties[0]  # Positive = improved
        obs_0 = outputs[0].observation
        obs_1 = outputs[1].observation
        obs_n = outputs[-1].observation
        metrics['loop/obs_delta_01'] = (obs_1 - obs_0).norm().item() / (obs_0.numel() ** 0.5)
        metrics['loop/obs_delta_total'] = (obs_n - obs_0).norm().item() / (obs_0.numel() ** 0.5)

    # ===== TICK EVOLUTION METRICS (for internal obs residual monitoring) =====
    # Compute how much activations change across internal ticks
    if len(outputs) > 0:
        final_out = outputs[-1]  # Use final step for monitoring
        pred_acts = final_out.prediction_output.all_tick_activations
        surp_out = final_out.surprise
        surp_acts = surp_out.all_tick_activations if surp_out is not None else None

        # Prediction tick evolution
        if len(pred_acts) > 1:
            pred_deltas = []
            for t in range(1, len(pred_acts)):
                delta = (pred_acts[t] - pred_acts[t-1]).norm().item() / pred_acts[t].numel() ** 0.5
                pred_deltas.append(delta)
            metrics['tick_pred_total_delta'] = sum(pred_deltas)
            metrics['tick_pred_late_delta'] = sum(pred_deltas[-2:]) if len(pred_deltas) >= 2 else pred_deltas[-1]
            metrics['tick_pred_first_delta'] = pred_deltas[0] if pred_deltas else 0
            # Detect plateau (last delta < 10% of first)
            metrics['tick_pred_plateau'] = 1.0 if (pred_deltas and pred_deltas[-1] < pred_deltas[0] * 0.1) else 0.0

        # Surprise tick evolution
        if surp_acts is not None and len(surp_acts) > 1:
            surp_deltas = []
            for t in range(1, len(surp_acts)):
                delta = (surp_acts[t] - surp_acts[t-1]).norm().item() / surp_acts[t].numel() ** 0.5
                surp_deltas.append(delta)
            metrics['tick_surp_total_delta'] = sum(surp_deltas)
            metrics['tick_surp_late_delta'] = sum(surp_deltas[-2:]) if len(surp_deltas) >= 2 else surp_deltas[-1]
            metrics['tick_surp_first_delta'] = surp_deltas[0] if surp_deltas else 0
            metrics['tick_surp_plateau'] = 1.0 if (surp_deltas and surp_deltas[-1] < surp_deltas[0] * 0.1) else 0.0

        # Surprise-prediction coupling (final step)
        if surp_out is not None:
            pred = final_out.predictions['immediate']
            target = targets['immediate']
            valid = targets.get('immediate_valid', None)
            pred_err = 1 - F.cosine_similarity(pred, target, dim=-1)  # (B, S)
            surp_mag = surp_out.magnitude.squeeze(-1)
            surp_raw = surp_out.raw.squeeze(-1)
            if valid is not None and valid.any():
                pred_err = pred_err[valid]
                surp_mag = surp_mag[valid]
                surp_raw = surp_raw[valid]
            metrics['surprise/pred_error_corr'] = safe_corr(pred_err, surp_mag)
            metrics['surprise/raw_vs_calibrated'] = safe_corr(surp_raw, surp_mag)

    return metrics


@torch.no_grad()
def compute_oscillator_ablation(
    model: PEMLoopGlobal,
    output,  # PEMLoopGlobalOutput from final step
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute prediction degradation when oscillator contribution is disabled.

    This measures how much the oscillators actually help predictions by
    comparing loss with vs without world_state.

    Returns:
        Dict with:
            - osc_ablation/loss_with: Loss with oscillators (normal)
            - osc_ablation/loss_without: Loss without oscillators
            - osc_ablation/degradation: How much worse without oscillators (positive = oscillators help)
            - osc_ablation/relative_help: Relative improvement from oscillators
    """
    metrics = {}

    # Get the loop features from the output (this is what prediction module uses)
    loop_features = output.loop_features
    if loop_features is None:
        return metrics

    # Compute loss with normal predictions (oscillators enabled)
    pred_with = output.predictions['immediate']
    target = targets['immediate']
    valid = targets.get('immediate_valid', None)

    if valid is not None and valid.any():
        cos_sim_with = F.cosine_similarity(pred_with[valid], target[valid], dim=-1).mean()
        loss_with = (1 - cos_sim_with).item()
    else:
        loss_with = 0.0

    # Run prediction WITHOUT world_state (oscillators disabled)
    pred_output_without = model.prediction(loop_features, world_state=None)
    pred_without = pred_output_without.predictions['immediate']

    if valid is not None and valid.any():
        cos_sim_without = F.cosine_similarity(pred_without[valid], target[valid], dim=-1).mean()
        loss_without = (1 - cos_sim_without).item()
    else:
        loss_without = 0.0

    # Compute degradation metrics
    degradation = loss_without - loss_with  # Positive = oscillators help
    relative_help = degradation / (loss_without + 1e-8)  # Fraction of loss explained by oscillators

    metrics['osc_ablation/loss_with'] = loss_with
    metrics['osc_ablation/loss_without'] = loss_without
    metrics['osc_ablation/degradation'] = degradation
    metrics['osc_ablation/relative_help'] = relative_help

    return metrics


def print_diagnostic_report(
    outputs: List,
    targets: Dict[str, torch.Tensor],
    step: int,
    model: Optional[nn.Module] = None,
    metrics: Optional[Dict[str, float]] = None,
):
    """
    Print compact diagnostic report to terminal.
    Shows key health indicators without verbose per-tick/per-step output.
    """
    print("\n" + "="*70)
    print(f"DIAGNOSTIC - Step {step}")
    print("="*70)

    num_steps = len(outputs)

    # Helper for tick evolution visualization
    def tick_bar(deltas: list, threshold: float = 0.001) -> str:
        if not deltas:
            return "?"
        chars = []
        for d in deltas:
            if d > threshold * 10:
                chars.append("█")
            elif d > threshold:
                chars.append("▓")
            elif d > threshold * 0.1:
                chars.append("░")
            else:
                chars.append("·")
        return "".join(chars)

    def find_plateau(deltas: list, threshold: float = 0.0001) -> int:
        for t, d in enumerate(deltas):
            if d < threshold:
                return t + 1
        return -1

    # ===== 1. GLOBAL LOOP: Loss per step (compact) =====
    step_losses = []
    for out in outputs:
        pred = out.predictions['immediate']
        target = targets['immediate']
        valid = targets.get('immediate_valid', None)
        if valid is not None and valid.any():
            cos_sim = F.cosine_similarity(pred[valid], target[valid], dim=-1).mean()
            step_losses.append((1 - cos_sim).item())
        else:
            step_losses.append(0.0)

    best_step = int(np.argmin(step_losses))
    improvement = step_losses[0] - step_losses[-1]

    # Compact loss display
    loss_str = " → ".join([f"{l:.3f}" for l in step_losses[:4]])
    if num_steps > 4:
        loss_str += f" → ... → {step_losses[-1]:.3f}"

    print(f"\n[Loop] {loss_str}")
    print(f"       Best: step {best_step} ({step_losses[best_step]:.4f}) | Δ: {improvement:+.4f} {'✓' if improvement > 0 else '✗'}")

    # ===== 2. TICK EVOLUTION (final step only) =====
    final_out = outputs[-1]
    pred_acts = final_out.prediction_output.all_tick_activations
    surp_out = final_out.surprise
    surp_acts = surp_out.all_tick_activations if surp_out is not None else None

    # Compute deltas
    pred_deltas = [(pred_acts[t] - pred_acts[t-1]).norm().item() / pred_acts[t].numel() ** 0.5
                   for t in range(1, len(pred_acts))]

    pred_plateau = find_plateau(pred_deltas)
    pred_status = f"⚠plateau@{pred_plateau}" if pred_plateau > 0 else "✓"

    print(f"\n[Ticks] Pred({len(pred_acts)}): [{tick_bar(pred_deltas)}] {pred_status}")

    if surp_acts is not None:
        surp_deltas = [(surp_acts[t] - surp_acts[t-1]).norm().item() / surp_acts[t].numel() ** 0.5
                       for t in range(1, len(surp_acts))]
        surp_plateau = find_plateau(surp_deltas)
        surp_status = f"⚠plateau@{surp_plateau}" if surp_plateau > 0 else "✓"
        print(f"        Surp({len(surp_acts)}): [{tick_bar(surp_deltas)}] {surp_status}")
    else:
        surp_deltas = []
        surp_plateau = -1
        print(f"        Surp: DISABLED")

    # ===== 3. ACTIVATION HEALTH (final step only) =====
    pred_z = pred_acts[-1]
    print(f"\n[Health] Pred: std={pred_z.std():.3f} mean={pred_z.mean():.3f}", end="")
    if surp_acts is not None:
        surp_z = surp_acts[-1]
        print(f" | Surp: std={surp_z.std():.3f} mean={surp_z.mean():.3f}")
    else:
        print()

    # ===== 4. CROSS-MODULE SYNC (final step only) =====
    cross_sync = final_out.global_sync.cross_module_sync
    if cross_sync.shape[0] >= 2:
        p2s = cross_sync[0, 1].mean().item()
        s2p = cross_sync[1, 0].mean().item()
        contrib = final_out.global_sync.module_contributions
        p_contrib = contrib[..., 0].mean().item()
        attn_status = "⚠degenerate" if (p2s > 0.95 or s2p > 0.95) else "✓"
        print(f"\n[CrossSync] P→S: {p2s:.2f} S→P: {s2p:.2f} | Contrib: P={p_contrib:.2f} S={1-p_contrib:.2f} {attn_status}")
    else:
        p2s, s2p = 0.0, 0.0  # For summary section
        print(f"\n[CrossSync] Single module mode (no cross-module sync)")

    # ===== 5. BEST TICK (final step only) =====
    pred_out = final_out.prediction_output

    if hasattr(pred_out, 'all_tick_outputs') and pred_out.all_tick_outputs:
        target = targets['immediate']
        valid = targets.get('immediate_valid', None)
        tick_losses = []
        for y_t in pred_out.all_tick_outputs:
            if valid is not None and valid.any():
                cos_sim = F.cosine_similarity(y_t[valid], target[valid], dim=-1).mean()
                tick_losses.append((1 - cos_sim).item())
            else:
                tick_losses.append(0.0)
        pred_best_tick = int(np.argmin(tick_losses))
        pred_tick_improve = tick_losses[0] - tick_losses[-1]
    else:
        pred_best_tick = 0
        pred_tick_improve = 0

    best_tick_str = f"[BestTick] Pred: {pred_best_tick}/{len(pred_acts)-1} (Δ={pred_tick_improve:+.3f})"

    if surp_out is not None and hasattr(surp_out, 'all_tick_magnitudes') and surp_out.all_tick_magnitudes:
        surp_tick_losses = [F.mse_loss(m, surp_out.raw).item() for m in surp_out.all_tick_magnitudes]
        surp_best_tick = int(np.argmin(surp_tick_losses))
        best_tick_str += f" | Surp: {surp_best_tick}/{len(surp_acts)-1}"
    else:
        surp_best_tick = 0

    print(f"\n{best_tick_str}")

    # ===== 6. OBSERVATION CONVERGENCE =====
    if num_steps >= 2:
        obs_0 = outputs[0].observation
        obs_f = outputs[-1].observation
        obs_delta = (obs_f - obs_0).norm().item() / obs_0.numel() ** 0.5
        obs_cos = F.cosine_similarity(obs_0.reshape(1, -1), obs_f.reshape(1, -1)).item()
        conv_status = "⚠fixed-point" if obs_cos > 0.9999 else "✓"
        print(f"\n[Obs] Δ(0→{num_steps-1}): {obs_delta:.4f} | cos: {obs_cos:.6f} {conv_status}")

    # ===== 7. NEW METRICS (from model and metrics dict) =====
    if model is not None and hasattr(model, 'global_sync'):
        world_stats = model.global_sync.get_world_state_stats()

        # Oscillator frequency bands
        slow_amp = world_stats.get('osc/slow_amp_mean', None)
        mid_amp = world_stats.get('osc/mid_amp_mean', None)
        fast_amp = world_stats.get('osc/fast_amp_mean', None)
        freq_ratio = world_stats.get('osc/freq_band_ratio', None)
        if slow_amp is not None:
            print(f"\n[OscBands] slow={slow_amp:.3f} mid={mid_amp:.3f} fast={fast_amp:.3f} ratio={freq_ratio:.2f}")

        # Feature write attention
        fw_entropy = world_stats.get('feature_write/pos_attn_entropy', None)
        fw_top1 = world_stats.get('feature_write/top1_weight', None)
        fw_top5 = world_stats.get('feature_write/top5_weight', None)
        fw_surp_corr = world_stats.get('feature_write/surprise_correlation', None)
        fw_surp_abs_corr = world_stats.get('feature_write/surprise_abs_correlation', None)
        surp_mean = world_stats.get('surprise/mean', None)
        surp_std = world_stats.get('surprise/std', None)
        surp_pos = world_stats.get('surprise/pos_frac', None)
        if fw_entropy is not None:
            surp_str = f" surp_corr={fw_surp_corr:.2f}" if fw_surp_corr is not None else ""
            abs_str = f" abs_corr={fw_surp_abs_corr:.2f}" if fw_surp_abs_corr is not None else ""
            print(f"[FeatWrite] entropy={fw_entropy:.2f} top1={fw_top1:.3f} top5={fw_top5:.3f}{surp_str}{abs_str}")
        if surp_mean is not None:
            print(f"[Surprise] mean={surp_mean:.3f} std={surp_std:.3f} pos={surp_pos:.2f}")

        # Write gate
        write_gate = world_stats.get('surprise/write_gate_mean', None)
        if write_gate is not None:
            print(f"[WriteGate] mean={write_gate:.3f}")

        # Oscillator cross-attention extra metrics
        osc_top1 = world_stats.get('osc_xattn/top1_osc_weight', None)
        osc_kq_cos = world_stats.get('osc_xattn/key_query_cosine', None)
        if osc_top1 is not None:
            print(f"[OscXAttn] top1={osc_top1:.3f} kq_cos={osc_kq_cos:.3f}")

    if metrics is not None:
        # Loop trajectory
        l0 = metrics.get('loop/loss_step0', None)
        l1 = metrics.get('loop/loss_step1', None)
        ln = metrics.get('loop/loss_stepN', None)
        mono = metrics.get('loop/monotonic_improve', None)
        obs_d01 = metrics.get('loop/obs_delta_01', None)
        obs_dtot = metrics.get('loop/obs_delta_total', None)
        if l0 is not None:
            mono_str = f" mono={mono:.2f}" if mono is not None else ""
            obs_str = f" obs_Δ01={obs_d01:.3f} tot={obs_dtot:.3f}" if obs_d01 is not None else ""
            print(f"\n[LoopTraj] L0={l0:.3f} L1={l1:.3f} LN={ln:.3f}{mono_str}{obs_str}")

        # CTM tick selection
        t1 = metrics.get('ctm/pred_t1_mean', None)
        t2 = metrics.get('ctm/pred_t2_mean', None)
        agree = metrics.get('ctm/t1_t2_agreement', None)
        spread = metrics.get('ctm/tick_loss_spread', None)
        if t1 is not None:
            print(f"[CTM] t1={t1:.1f} t2={t2:.1f} agree={agree:.2f} spread={spread:.4f}")

        # Oscillator attention entropy evolution
        e0 = metrics.get('osc_xattn/entropy_step0', None)
        en = metrics.get('osc_xattn/entropy_stepN', None)
        ed = metrics.get('osc_xattn/entropy_delta', None)
        if e0 is not None:
            print(f"[OscEntropy] step0={e0:.2f} stepN={en:.2f} Δ={ed:+.3f}")

        # Surprise-prediction correlation
        surp_pred_corr = metrics.get('surprise/pred_error_corr', None)
        surp_raw_cal = metrics.get('surprise/raw_vs_calibrated', None)
        if surp_pred_corr is not None:
            print(f"[Surprise] pred_err_corr={surp_pred_corr:.3f} raw_cal_corr={surp_raw_cal:.3f}")

        # Oscillator ablation (most important - does the oscillator actually help?)
        osc_abl_loss_with = metrics.get('osc_ablation/loss_with', None)
        osc_abl_loss_without = metrics.get('osc_ablation/loss_without', None)
        osc_abl_deg = metrics.get('osc_ablation/degradation', None)
        osc_abl_rel = metrics.get('osc_ablation/relative_help', None)
        if osc_abl_deg is not None:
            status = "✓ OSCILLATORS HELPING" if osc_abl_deg > 0.001 else (
                "≈ NEUTRAL" if abs(osc_abl_deg) < 0.001 else "✗ OSCILLATORS HURTING")
            print(f"\n[OscAblation] loss_with={osc_abl_loss_with:.4f} loss_without={osc_abl_loss_without:.4f}")
            print(f"              degradation={osc_abl_deg:+.4f} relative_help={osc_abl_rel:.1%} {status}")

    # ===== SUMMARY =====
    issues = []
    if improvement <= 0:
        issues.append("loop not helping")
    if pred_plateau > 0 and pred_plateau < len(pred_acts) - 2:
        issues.append(f"pred plateau@{pred_plateau}")
    if surp_acts is not None and surp_plateau > 0 and surp_plateau < len(surp_acts) - 2:
        issues.append(f"surp plateau@{surp_plateau}")
    if surp_acts is not None and (p2s > 0.95 or s2p > 0.95):
        issues.append("degenerate cross-attn")
    if num_steps >= 2 and obs_cos > 0.9999:
        issues.append("obs fixed-point")

    print("\n" + "-"*70)
    if issues:
        print(f"⚠ Issues: {', '.join(issues)}")
    else:
        print("✓ All systems nominal")
    print("="*70 + "\n")


def compute_nlm_metrics(model: PEMLoopGlobal) -> Dict[str, float]:
    """
    Compute NLM (Neuron Level Model) specific metrics for debugging.

    Tracks:
    - Weight norms (are weights changing?)
    - Gradient norms (are gradients flowing?)
    """
    metrics = {}

    # PredictionCTM NLM
    pred_nlm = model.prediction.core.nlm
    metrics['nlm_pred/w1_norm'] = pred_nlm.w1.norm().item()
    metrics['nlm_pred/w2_norm'] = pred_nlm.w2.norm().item()
    if pred_nlm.w1.grad is not None:
        metrics['nlm_pred/w1_grad_norm'] = pred_nlm.w1.grad.norm().item()
        metrics['nlm_pred/w2_grad_norm'] = pred_nlm.w2.grad.norm().item()
    else:
        metrics['nlm_pred/w1_grad_norm'] = 0.0
        metrics['nlm_pred/w2_grad_norm'] = 0.0

    # SurpriseCTM NLM
    surp_nlm = model.surprise.core.nlm
    metrics['nlm_surp/w1_norm'] = surp_nlm.w1.norm().item()
    metrics['nlm_surp/w2_norm'] = surp_nlm.w2.norm().item()
    if surp_nlm.w1.grad is not None:
        metrics['nlm_surp/w1_grad_norm'] = surp_nlm.w1.grad.norm().item()
        metrics['nlm_surp/w2_grad_norm'] = surp_nlm.w2.grad.norm().item()
    else:
        metrics['nlm_surp/w1_grad_norm'] = 0.0
        metrics['nlm_surp/w2_grad_norm'] = 0.0

    # Synapse gradients (upstream of NLM)
    pred_synapse = model.prediction.core.synapse
    surp_synapse = model.surprise.core.synapse
    if hasattr(pred_synapse, 'out') and pred_synapse.out.weight.grad is not None:
        metrics['synapse_pred/out_grad_norm'] = pred_synapse.out.weight.grad.norm().item()
    if hasattr(surp_synapse, 'out') and surp_synapse.out.weight.grad is not None:
        metrics['synapse_surp/out_grad_norm'] = surp_synapse.out.weight.grad.norm().item()

    # Oscillator cross-attention gradients (critical for oscillator selection learning)
    global_sync = model.global_sync
    if hasattr(global_sync, 'osc_embeddings') and global_sync.osc_embeddings is not None:
        metrics['osc_xattn/embed_norm'] = global_sync.osc_embeddings.norm().item()
        if global_sync.osc_embeddings.grad is not None:
            metrics['osc_xattn/embed_grad'] = global_sync.osc_embeddings.grad.norm().item()
        else:
            metrics['osc_xattn/embed_grad'] = 0.0

    if hasattr(global_sync, 'osc_query_proj') and global_sync.osc_query_proj is not None:
        metrics['osc_xattn/query_norm'] = global_sync.osc_query_proj.weight.norm().item()
        if global_sync.osc_query_proj.weight.grad is not None:
            metrics['osc_xattn/query_grad'] = global_sync.osc_query_proj.weight.grad.norm().item()
        else:
            metrics['osc_xattn/query_grad'] = 0.0

    if hasattr(global_sync, 'osc_key_proj') and global_sync.osc_key_proj is not None:
        metrics['osc_xattn/key_norm'] = global_sync.osc_key_proj.weight.norm().item()
        if global_sync.osc_key_proj.weight.grad is not None:
            metrics['osc_xattn/key_grad'] = global_sync.osc_key_proj.weight.grad.norm().item()
        else:
            metrics['osc_xattn/key_grad'] = 0.0

    if hasattr(global_sync, 'osc_value_proj') and global_sync.osc_value_proj is not None:
        metrics['osc_xattn/value_norm'] = global_sync.osc_value_proj.weight.norm().item()
        if global_sync.osc_value_proj.weight.grad is not None:
            metrics['osc_xattn/value_grad'] = global_sync.osc_value_proj.weight.grad.norm().item()
        else:
            metrics['osc_xattn/value_grad'] = 0.0

    # Gate gradient monitoring
    gate_params = list(model.state_combiner_gate.parameters())
    gate_grad_sq = 0.0
    for p in gate_params:
        if p.grad is not None:
            gate_grad_sq += p.grad.norm().item() ** 2
    metrics['gate/grad_norm'] = math.sqrt(gate_grad_sq) if gate_grad_sq > 0 else 0.0

    # Relative update monitoring (are queries doing all the learning?)
    query_grad = metrics.get('osc_xattn/query_grad', 0.0)
    key_grad = metrics.get('osc_xattn/key_grad', 0.0)
    value_grad = metrics.get('osc_xattn/value_grad', 0.0)
    query_norm = metrics.get('osc_xattn/query_norm', 0.0)
    key_norm = metrics.get('osc_xattn/key_norm', 0.0)
    value_norm = metrics.get('osc_xattn/value_norm', 0.0)
    eps = 1e-12
    if query_norm > 0:
        metrics['osc_xattn/query_rel_update'] = query_grad / (query_norm + eps)
    else:
        metrics['osc_xattn/query_rel_update'] = 0.0
    if key_norm > 0:
        metrics['osc_xattn/key_rel_update'] = key_grad / (key_norm + eps)
    else:
        metrics['osc_xattn/key_rel_update'] = 0.0
    if value_norm > 0:
        metrics['osc_xattn/value_rel_update'] = value_grad / (value_norm + eps)
    else:
        metrics['osc_xattn/value_rel_update'] = 0.0
    metrics['osc_xattn/qk_grad_ratio'] = query_grad / (key_grad + eps)
    metrics['osc_xattn/qk_rel_update_ratio'] = (
        metrics['osc_xattn/query_rel_update'] / (metrics['osc_xattn/key_rel_update'] + eps)
    )

    return metrics


def train_step(
    model: PEMLoopGlobal,
    features: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    return_outputs: bool = False,
):
    """Execute one training step.

    Returns:
        metrics: Dict of scalar metrics
        outputs: (optional) List of PEMLoopGlobalOutput for visualization
        targets: (optional) Dict of target tensors (returned with outputs)
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    targets = model.target_computer.compute_targets_efficient(features)
    outputs, final_state = model(features, targets, num_steps=config.num_loop_steps)

    # Compute loss
    loss, loss_dict = model.compute_loss(outputs, targets)

    # Backward pass
    loss.backward()

    # Compute NLM metrics BEFORE optimizer step (to capture gradients)
    nlm_metrics = compute_nlm_metrics(model)

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    # Optimizer step
    optimizer.step()

    # Get final output for world state processing
    final_output = outputs[-1]

    # Compute oscillator metrics for monitoring
    world_state_metrics = {}
    osc_metrics = final_output.global_sync.oscillator_metrics
    if osc_metrics is not None:
        world_state_metrics = {
            'oscillator/phase_mean': osc_metrics.phase_mean,
            'oscillator/phase_std': osc_metrics.phase_std,
            'oscillator/phase_entropy': osc_metrics.phase_entropy,
            'oscillator/amplitude_mean': osc_metrics.amplitude_mean,
            'oscillator/amplitude_std': osc_metrics.amplitude_std,
            'oscillator/amplitude_max': osc_metrics.amplitude_max,
            'oscillator/active_frac': osc_metrics.active_oscillator_frac,
            'oscillator/freq_weighted_amp': osc_metrics.frequency_weighted_amplitude,
            'oscillator/amp_mod_mean': osc_metrics.amp_mod_mean,
            'oscillator/phase_mod_mean': osc_metrics.phase_mod_mean,
            'oscillator/output_norm': osc_metrics.output_norm,
            'oscillator/output_mean': osc_metrics.output_mean,
            'oscillator/output_std': osc_metrics.output_std,
        }
        # Add cross-attention entropy (measures diversity of oscillator selection)
        if final_output.global_sync.osc_attn_entropy is not None:
            world_state_metrics['oscillator/attn_entropy'] = final_output.global_sync.osc_attn_entropy

    # NOTE: Oscillatory world model evolves continuously during forward pass
    # No commit_world_state() needed - oscillators advance and modulate automatically

    # Compute detailed metrics
    with torch.no_grad():
        metrics = compute_detailed_metrics(outputs, targets)

    # Add loss and grad norm
    metrics['loss'] = loss.item()
    metrics['grad_norm'] = grad_norm.item()

    # Add NLM metrics
    metrics.update(nlm_metrics)

    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            metrics[f'loss/{k}'] = v.item()
        elif isinstance(v, (int, float)):
            metrics[f'loss/{k}'] = v

    # Cumulative sync stats
    metrics['cumulative_sync_mean'] = final_state.cumulative_sync.mean().item()
    metrics['cumulative_sync_std'] = final_state.cumulative_sync.std().item()

    # World state stats (emergent world model) - basic stats
    world_state_stats = model.get_world_state_stats()
    metrics.update(world_state_stats)

    # Add detailed world state metrics from monitor
    metrics.update(world_state_metrics)

    # Compute state combiner gate metrics (critical for loop learning)
    with torch.no_grad():
        combined = torch.cat([features, final_state.observation], dim=-1)
        gate_values = model.state_combiner_gate(combined)
        metrics['gate/mean'] = gate_values.mean().item()
        metrics['gate/std'] = gate_values.std().item()
        metrics['gate/min'] = gate_values.min().item()
        metrics['gate/max'] = gate_values.max().item()
        metrics['gate/position_variance'] = gate_values.var(dim=1, unbiased=False).mean().item()
        # Also track how much loop_features differ from raw features
        transformed = model.state_combiner_transform(combined)
        loop_features = features + gate_values * (transformed - features)
        loop_features_diff = (loop_features - features).abs().mean().item()
        metrics['gate/loop_features_diff'] = loop_features_diff
        if outputs:
            combined_0 = torch.cat([features, outputs[0].observation], dim=-1)
            gate_0 = model.state_combiner_gate(combined_0)
            combined_n = torch.cat([features, outputs[-1].observation], dim=-1)
            gate_n = model.state_combiner_gate(combined_n)
            gate_0_mean = gate_0.mean().item()
            gate_n_mean = gate_n.mean().item()
            metrics['gate/mean_step0'] = gate_0_mean
            metrics['gate/mean_stepN'] = gate_n_mean
            metrics['gate/step0_vs_stepN'] = gate_n_mean - gate_0_mean

    # Add oscillator metrics if available (from oscillatory world model)
    if final_output.global_sync.oscillator_metrics is not None:
        osc_metrics = final_output.global_sync.oscillator_metrics
        # OscillatorMetrics is a NamedTuple, convert to dict
        osc_dict = osc_metrics._asdict() if hasattr(osc_metrics, '_asdict') else osc_metrics
        for key, value in osc_dict.items():
            if isinstance(value, (int, float)):
                metrics[f'world_state/osc_{key}'] = value

    if return_outputs:
        return metrics, outputs, targets
    return metrics


@torch.no_grad()
def eval_step(
    model: PEMLoopGlobal,
    features: torch.Tensor,
    config: TrainingConfig,
) -> Dict[str, float]:
    """Execute one evaluation step."""
    model.eval()

    targets = model.target_computer.compute_targets_efficient(features)
    outputs, final_state = model(features, targets, num_steps=config.num_loop_steps)

    loss, loss_dict = model.compute_loss(outputs, targets)
    metrics = compute_detailed_metrics(outputs, targets)

    metrics['eval/loss'] = loss.item()
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            metrics[f'eval/loss/{k}'] = v.item()
        elif isinstance(v, (int, float)):
            metrics[f'eval/loss/{k}'] = v

    return metrics


def find_text_files(data_dir: str, num_files: int = 0, seed: int = 42) -> List[Path]:
    """Find and optionally sample text files from directory."""
    print(f"Scanning for text files in {data_dir}...")

    all_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.txt') and not f.startswith('._'):
                all_files.append(Path(root) / f)

    print(f"Found {len(all_files)} text files")

    # Sample random subset if num_files > 0
    if num_files > 0 and len(all_files) > num_files:
        random.seed(seed)
        sampled = random.sample(all_files, num_files)
        print(f"Selected {len(sampled)} files for training")
        return sampled

    return all_files


def load_and_clean_text(file_path: Path, min_length: int = 1000) -> Optional[str]:
    """Load text file and perform basic cleaning."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        text = text.strip()
        if len(text) < min_length:
            return None
        return text
    except Exception as e:
        return None


class LocalTextDataset(IterableDataset):
    """Iterable dataset that streams text chunks from local text files."""

    def __init__(
        self,
        file_paths: List[Path],
        chunk_size: int = 2048,  # Characters per chunk
        min_text_length: int = 1000,
        shuffle: bool = True,
    ):
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.min_text_length = min_text_length
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[dict]:
        """Yield text chunks from files."""
        file_paths = self.file_paths.copy()

        if self.shuffle:
            random.shuffle(file_paths)

        for file_path in file_paths:
            text = load_and_clean_text(file_path, self.min_text_length)
            if text is None:
                continue

            # Create overlapping chunks
            stride = self.chunk_size // 2
            for i in range(0, len(text) - self.chunk_size, stride):
                chunk = text[i:i + self.chunk_size]
                yield {'text': chunk}


def create_dataloader(
    config: TrainingConfig,
    split: str = 'train',
    dataset_type: str = 'fineweb',
    data_dir: Optional[str] = None,
    num_files: int = 0,
):
    """Create dataloader from HuggingFace dataset or local text files.

    Args:
        config: Training configuration
        split: 'train' or 'val'
        dataset_type: 'fineweb' for HuggingFace FineWeb-Edu, 'local' for local text files
        data_dir: Directory containing text files (required if dataset_type='local')
        num_files: Number of files to use (0 = all files)
    """
    if dataset_type == 'local':
        if data_dir is None:
            raise ValueError("data_dir required for local dataset")

        file_paths = find_text_files(data_dir, num_files=num_files)
        if not file_paths:
            raise ValueError(f"No text files found in {data_dir}")

        # Split for validation
        if split == 'val':
            random.seed(42)
            random.shuffle(file_paths)
            val_size = max(1, len(file_paths) // 20)  # 5% for validation
            file_paths = file_paths[:val_size]
        else:
            random.seed(42)
            random.shuffle(file_paths)
            val_size = max(1, len(file_paths) // 20)
            file_paths = file_paths[val_size:]

        dataset = LocalTextDataset(
            file_paths,
            chunk_size=config.max_length * 4,  # Rough char estimate
            shuffle=(split == 'train'),
        )

        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            return texts

    else:  # fineweb
        from datasets import load_dataset

        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            split="train",
            streaming=True,
        )

        if split == 'val':
            dataset = dataset.skip(10000).take(1000)

        def collate_fn(batch):
            texts = [item['text'][:config.max_length * 4] for item in batch]
            return texts

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )

    return loader


def main():
    parser = argparse.ArgumentParser(description='Train PEM Loop with Global Sync')

    # Model args
    parser.add_argument('--d_model', type=int, default=1536)
    parser.add_argument('--pred_d_neurons', type=int, default=256)
    parser.add_argument('--surp_d_neurons', type=int, default=128)
    parser.add_argument('--pred_T', type=int, default=4)
    parser.add_argument('--surp_T', type=int, default=3)
    parser.add_argument('--sync_pairs', type=int, default=256)

    # PredictionCTM architecture (matching train_prediction.py defaults)
    parser.add_argument('--pred_synapse_hidden', type=int, default=1024,
                        help='Hidden dim in Prediction synapse U-NET')
    parser.add_argument('--pred_nlm_hidden', type=int, default=64,
                        help='Hidden dim in Prediction per-neuron MLPs')
    parser.add_argument('--pred_d_sync_out', type=int, default=256,
                        help='Prediction sync pairs for output')
    parser.add_argument('--pred_d_sync_internal', type=int, default=256,
                        help='Prediction sync pairs for internal')

    # SurpriseCTM architecture
    parser.add_argument('--surp_synapse_hidden', type=int, default=512,
                        help='Hidden dim in Surprise synapse U-NET')
    parser.add_argument('--surp_nlm_hidden', type=int, default=32,
                        help='Hidden dim in Surprise per-neuron MLPs')
    parser.add_argument('--surp_d_sync_out', type=int, default=128,
                        help='Surprise sync pairs for output')
    parser.add_argument('--surp_d_sync_internal', type=int, default=128,
                        help='Surprise sync pairs for internal')

    # Training args
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--num_epochs', type=int, default=0,
                        help='Train for this many epochs (0 = use max_steps instead)')
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_loop_steps', type=int, default=2)
    parser.add_argument('--observation_residual', type=float, default=0.2,
                        help='Global loop observation blend factor (0=replace, 0.2=default, 1=keep)')
    parser.add_argument('--internal_obs_residual', type=float, default=0.2,
                        help='Internal tick observation blend factor (0=replace, 0.2=default, 1=keep)')
    parser.add_argument('--state_combiner_gate_init', type=float, default=-0.85,
                        help='State combiner gate init bias (sigmoid: -2→0.12, -0.85→0.30, 0→0.50)')
    parser.add_argument('--attention_temperature', type=float, default=2.0,
                        help='Cross-module attention temperature (higher=softer, default=2.0)')
    parser.add_argument('--cross_residual_strength', type=float, default=0.0,
                        help='Cross-module residual strength (0=off, 0.1-0.3=moderate, forces cross-module info flow)')
    parser.add_argument('--surprise_loss_weight', type=float, default=0.1,
                        help='Weight for surprise calibration loss (default=0.1, try 0.5-1.0 if surprise gradients vanish)')
    parser.add_argument('--cross_attn_diversity_weight', type=float, default=0.0,
                        help='Penalize degenerate cross-attention (0=off, 0.1-0.5=moderate, prevents attention collapse)')
    parser.add_argument('--loop_improvement_weight', type=float, default=0.5,
                        help='Penalize loop regression (0=off, 0.5=default, encourages later steps to improve or maintain)')
    parser.add_argument('--max_length', type=int, default=512)

    # Dataset args
    parser.add_argument('--dataset', type=str, default='fineweb',
                        choices=['fineweb', 'local'],
                        help='Dataset to use: fineweb (HuggingFace) or local (text files)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing text files (required if --dataset=local)')
    parser.add_argument('--num_files', type=int, default=0,
                        help='Number of text files to use (0=all files)')

    # Oscillatory world model args
    parser.add_argument('--num_oscillators', type=int, default=64,
                        help='Number of oscillators in world model (memory slots)')
    parser.add_argument('--min_period', type=int, default=8,
                        help='Fastest oscillator period (phrase-level ~8 tokens)')
    parser.add_argument('--max_period', type=int, default=4096,
                        help='Slowest oscillator period (document-level ~4096 tokens)')
    parser.add_argument('--d_world_output', type=int, default=256,
                        help='Output dimension of world state')
    parser.add_argument('--d_osc_embed', type=int, default=64,
                        help='Per-oscillator embedding dimension for cross-attention')
    parser.add_argument('--surprise_gate_bias', type=float, default=0.5,
                        help='Base write strength for surprise gating (0.5 = moderate baseline)')
    parser.add_argument('--surprise_gate_scale', type=float, default=1.0,
                        help='How much surprise amplifies writing (higher = more surprise-sensitive)')
    parser.add_argument('--disable_oscillatory_world', action='store_true',
                        help='Disable oscillatory world model (for ablation)')
    parser.add_argument('--auxiliary_prediction_weight', type=float, default=0.1,
                        help='Weight for auxiliary prediction loss (trains oscillators to predict future)')
    parser.add_argument('--auxiliary_prediction_horizon', type=int, default=8,
                        help='How many steps ahead to predict for auxiliary loss')
    parser.add_argument('--enable_auxiliary_prediction', action='store_true',
                        help='Enable auxiliary prediction loss (disabled by default - ablation showed it hurts)')
    parser.add_argument('--multi_tick_world_injection', action='store_true',
                        help='Inject world state at each CTM tick (not just z_0)')
    parser.add_argument('--surprise_signal_type', type=str, default='attention_entropy',
                        choices=['ctm', 'prediction_error', 'attention_entropy'],
                        help='Surprise signal type for oscillator memory gating (attention_entropy best per ablation)')

    # Prediction horizon args
    parser.add_argument('--immediate_horizon', type=int, default=8,
                        help='Tokens ahead for immediate prediction (1=exact next token, 8=mean of next 8)')
    parser.add_argument('--shortterm_horizon', type=int, default=64,
                        help='Tokens ahead for shortterm prediction')
    parser.add_argument('--longterm_horizon', type=int, default=256,
                        help='Tokens ahead for longterm prediction')

    # Memory optimization args
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Recompute activations during backward (saves ~2-3x VRAM, ~30%% slower)')
    parser.add_argument('--backprop_steps', type=int, default=-1,
                        help='Only backprop through last N loop steps (-1 = all)')

    # Ablation/debug args
    parser.add_argument('--disable_surprise', action='store_true',
                        help='Disable surprise module (prediction-only mode for debugging)')
    parser.add_argument('--bypass_state_combiner', action='store_true',
                        help='Skip state combiner, use raw features (for comparison with train_prediction.py)')

    # Logging args
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--diagnostic_every', type=int, default=0,
                        help='Print diagnostic report every N steps (0=disabled, 1=every step)')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        d_model=args.d_model,
        pred_d_neurons=args.pred_d_neurons,
        surp_d_neurons=args.surp_d_neurons,
        pred_T=args.pred_T,
        surp_T=args.surp_T,
        sync_pairs=args.sync_pairs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        num_loop_steps=args.num_loop_steps,
        max_length=args.max_length,
        log_every=args.log_every,
        eval_every=args.eval_every,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize wandb
    if config.wandb_project:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )

    # Create feature extractor (frozen)
    print("Loading Janus Pro feature extractor...")
    from .janus_pro_feature_extractor import LearningMode
    janus_config = JanusProConfig(
        model_name_or_path="deepseek-ai/Janus-Pro-1B",
        output_dim=config.d_model,
        learning_mode=LearningMode.FROZEN,
    )
    feature_extractor = JanusProFeatureExtractor(janus_config)
    feature_extractor.eval()
    print(f"Feature extractor loaded: {sum(p.numel() for p in feature_extractor.parameters()):,} params")

    # Create PEM loop with global sync
    print("Creating PEM Loop with Global Sync...")
    pem_config = PEMLoopGlobalConfig(
        d_model=config.d_model,
        pred_d_neurons=config.pred_d_neurons,
        surp_d_neurons=config.surp_d_neurons,
        pred_T=config.pred_T,
        surp_T=config.surp_T,
        sync_pairs=config.sync_pairs,
        # PredictionCTM architecture
        pred_synapse_hidden=args.pred_synapse_hidden,
        pred_nlm_hidden=args.pred_nlm_hidden,
        pred_d_sync_out=args.pred_d_sync_out,
        pred_d_sync_internal=args.pred_d_sync_internal,
        # SurpriseCTM architecture
        surp_synapse_hidden=args.surp_synapse_hidden,
        surp_nlm_hidden=args.surp_nlm_hidden,
        surp_d_sync_out=args.surp_d_sync_out,
        surp_d_sync_internal=args.surp_d_sync_internal,
        # Oscillatory world model config
        use_oscillatory_world=not args.disable_oscillatory_world,
        num_oscillators=args.num_oscillators,
        min_period=args.min_period,
        max_period=args.max_period,
        d_world_output=args.d_world_output,
        d_osc_embed=args.d_osc_embed,
        surprise_gate_bias=args.surprise_gate_bias,
        surprise_gate_scale=args.surprise_gate_scale,
        # Auxiliary prediction (disabled by default per ablation results)
        use_auxiliary_prediction=args.enable_auxiliary_prediction,
        auxiliary_prediction_horizon=args.auxiliary_prediction_horizon,
        auxiliary_prediction_weight=args.auxiliary_prediction_weight,
        # Multi-tick world injection
        multi_tick_world_injection=args.multi_tick_world_injection,
        # Surprise signal type
        surprise_signal_type=args.surprise_signal_type,
        # Other config
        sync_attention_temperature=args.attention_temperature,
        sync_cross_residual_strength=args.cross_residual_strength,
        observation_residual=args.observation_residual,
        state_combiner_gate_init=args.state_combiner_gate_init,
        internal_obs_residual=args.internal_obs_residual,
        # Prediction horizons
        immediate_horizon=args.immediate_horizon,
        shortterm_horizon=args.shortterm_horizon,
        longterm_horizon=args.longterm_horizon,
        gradient_checkpointing=args.gradient_checkpointing,
        backprop_steps=args.backprop_steps,
        surprise_loss_weight=args.surprise_loss_weight,
        cross_attn_diversity_weight=args.cross_attn_diversity_weight,
        loop_improvement_weight=args.loop_improvement_weight,
        disable_surprise=args.disable_surprise,
        bypass_state_combiner=args.bypass_state_combiner,
    )
    model = PEMLoopGlobal(pem_config).to(device)
    print(f"PEM Loop created: {sum(p.numel() for p in model.parameters()):,} params")
    if args.observation_residual > 0:
        print(f"  [Loop] Observation residual: {args.observation_residual}")
    if args.internal_obs_residual > 0:
        print(f"  [CTM] Internal observation residual: {args.internal_obs_residual}")
    if args.attention_temperature != 1.0:
        print(f"  [Sync] Attention temperature: {args.attention_temperature}")
    if args.cross_residual_strength > 0:
        print(f"  [Sync] Cross-residual strength: {args.cross_residual_strength}")
    if args.surprise_loss_weight != 0.1:
        print(f"  [Loss] Surprise loss weight: {args.surprise_loss_weight}")
    if args.cross_attn_diversity_weight > 0:
        print(f"  [Loss] Cross-attention diversity weight: {args.cross_attn_diversity_weight}")
    if args.loop_improvement_weight > 0:
        print(f"  [Loss] Loop improvement weight: {args.loop_improvement_weight}")
    if args.disable_surprise:
        print("  [Ablation] Surprise module DISABLED (prediction-only mode)")
    if args.bypass_state_combiner:
        print("  [Ablation] State combiner BYPASSED (raw features mode)")
    if args.gradient_checkpointing:
        print("  [Memory] Gradient checkpointing ENABLED")
    if args.backprop_steps > 0:
        print(f"  [Memory] Truncated backprop: last {args.backprop_steps} steps")
    if args.disable_oscillatory_world:
        print("  [Ablation] Oscillatory world model DISABLED")
    else:
        print(f"  [Oscillator] n={args.num_oscillators} periods=[{args.min_period}, {args.max_period}] "
              f"gate_bias={args.surprise_gate_bias} gate_scale={args.surprise_gate_scale}")
        if args.enable_auxiliary_prediction:
            print(f"  [AuxPred] ENABLED horizon={args.auxiliary_prediction_horizon} weight={args.auxiliary_prediction_weight}")
        else:
            print("  [AuxPred] Disabled (default - ablation showed it hurts performance)")
        if args.multi_tick_world_injection:
            print("  [World] Multi-tick injection ENABLED (world state at every CTM tick)")
        if args.surprise_signal_type != 'ctm':
            print(f"  [Surprise] Signal type: {args.surprise_signal_type}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Note: WorldStateMonitor is now less relevant with oscillatory world model
    # The oscillator-specific metrics are computed directly in the forward pass
    world_state_monitor = None  # Oscillator metrics computed in forward pass
    print(f"  [Monitor] Oscillatory world model enabled (n_osc={pem_config.num_oscillators})")

    # Create dataloader
    print(f"Creating dataloader (dataset={args.dataset})...")
    if args.dataset == 'local' and args.data_dir is None:
        raise ValueError("--data_dir required when using --dataset=local")
    train_loader = create_dataloader(
        config,
        split='train',
        dataset_type=args.dataset,
        data_dir=args.data_dir,
        num_files=args.num_files,
    )
    train_iter = iter(train_loader)

    # Determine training mode
    if config.num_epochs > 0:
        training_mode = "epochs"
        print(f"\nStarting training for {config.num_epochs} epochs...")
    else:
        training_mode = "steps"
        print(f"\nStarting training for {config.max_steps} steps...")

    print(f"  Batch size: {config.batch_size}")
    print(f"  Loop steps: {config.num_loop_steps}")
    print(f"  Pred: neurons={config.pred_d_neurons}, T={config.pred_T}, synapse={args.pred_synapse_hidden}, nlm={args.pred_nlm_hidden}")
    print(f"  Surp: neurons={config.surp_d_neurons}, T={config.surp_T}, synapse={args.surp_synapse_hidden}, nlm={args.surp_nlm_hidden}")
    print(f"  Sync pairs: {config.sync_pairs}")

    # Log oscillator architecture (world vs self split)
    n_osc = model.global_sync.config.num_oscillators
    n_self = model.global_sync.oscillatory_world.config.num_self_oscillators
    n_world = n_osc - n_self
    d_self_state = model.global_sync.oscillatory_world.config.d_self_state
    print(f"  Oscillators: {n_world} world + {n_self} self = {n_osc} total (d_self_state={d_self_state})")

    if args.diagnostic_every > 0:
        print(f"  [Diagnostic] Report every {args.diagnostic_every} steps")
    print()

    global_step = 0
    running_loss = 0.0
    start_time = time.time()
    current_epoch = 0
    total_documents = 0
    documents_in_epoch = 0

    # Track average best steps across training (overall and per-module)
    best_loss_step_sum = 0
    best_certainty_step_sum = 0
    # PredictionCTM
    pred_best_loss_step_sum = 0
    pred_best_cert_step_sum = 0
    # SurpriseCTM
    surp_best_loss_step_sum = 0
    surp_best_cert_step_sum = 0
    num_logged = 0
    qk_rel_ratio_warn_threshold = 1e3
    qk_rel_ratio_warn_min_logs = 3
    qk_rel_ratio_warn_streak = 0

    # Training loop condition
    def should_continue_training():
        if training_mode == "epochs":
            return current_epoch < config.num_epochs
        else:
            return global_step < config.max_steps

    while should_continue_training():
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            # Epoch complete
            current_epoch += 1
            documents_in_epoch = 0
            print(f"\n{'='*60}")
            print(f"EPOCH {current_epoch} COMPLETE (step {global_step}, {total_documents} docs)")
            print(f"{'='*60}\n")

            if training_mode == "epochs" and current_epoch >= config.num_epochs:
                break

            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Tokenize and extract features
        with torch.no_grad():
            # Tokenize
            tokenized = feature_extractor.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length,
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            # Extract features (cast to float32 for PEM model)
            features = feature_extractor(input_ids, attention_mask=attention_mask)
            features = features.float()  # Cast from bfloat16 to float32

        # Skip if too short
        if features.shape[1] < 32:
            continue

        # Track documents
        total_documents += len(batch)
        documents_in_epoch += len(batch)

        # Training step (return outputs periodically for visualization and diagnostics)
        should_visualize = config.wandb_project and (global_step + 1) % (config.log_every * 10) == 0
        should_diagnose = args.diagnostic_every > 0 and (
            global_step == 0 or (global_step + 1) % args.diagnostic_every == 0
        )
        need_outputs = should_visualize or should_diagnose
        result = train_step(
            model, features, optimizer, config,
            return_outputs=need_outputs,
        )

        if need_outputs:
            metrics, outputs_for_diag, targets_for_diag = result
        else:
            metrics = result
            outputs_for_diag = None
            targets_for_diag = None

        # Print diagnostic report if requested
        if should_diagnose and outputs_for_diag is not None:
            # Compute oscillator ablation metric
            osc_ablation_metrics = compute_oscillator_ablation(model, outputs_for_diag[-1], targets_for_diag)
            metrics.update(osc_ablation_metrics)
            print_diagnostic_report(outputs_for_diag, targets_for_diag, global_step + 1, model=model, metrics=metrics)

        scheduler.step()

        running_loss += metrics['loss']
        global_step += 1

        # Logging
        if global_step % config.log_every == 0:
            avg_loss = running_loss / config.log_every
            elapsed = time.time() - start_time
            steps_per_sec = global_step / elapsed

            # Track best steps for averaging (overall)
            best_loss_step = int(metrics.get('best_loss_step', 0))
            best_cert_step = int(metrics.get('best_certainty_step', 0))
            best_loss_step_sum += best_loss_step
            best_certainty_step_sum += best_cert_step

            # Track per-module best steps
            pred_best_loss_step = int(metrics.get('pred_best_loss_step', 0))
            pred_best_cert_step = int(metrics.get('pred_best_cert_step', 0))
            surp_best_loss_step = int(metrics.get('surp_best_loss_step', 0))
            surp_best_cert_step = int(metrics.get('surp_best_cert_step', 0))

            pred_best_loss_step_sum += pred_best_loss_step
            pred_best_cert_step_sum += pred_best_cert_step
            surp_best_loss_step_sum += surp_best_loss_step
            surp_best_cert_step_sum += surp_best_cert_step

            num_logged += 1

            # Compute running averages (overall)
            avg_best_loss_step = best_loss_step_sum / num_logged
            avg_best_cert_step = best_certainty_step_sum / num_logged

            # Compute running averages (per-module)
            avg_pred_best_loss_step = pred_best_loss_step_sum / num_logged
            avg_pred_best_cert_step = pred_best_cert_step_sum / num_logged
            avg_surp_best_loss_step = surp_best_loss_step_sum / num_logged
            avg_surp_best_cert_step = surp_best_cert_step_sum / num_logged

            # Console output (consolidated)
            best_loss_val = metrics.get('best_loss_value', 0)
            best_cert_val = metrics.get('best_certainty_value', 0)
            loss_impr = metrics.get('loss_improvement', 0)
            qk_rel_ratio = metrics.get('osc_xattn/qk_rel_update_ratio', 0)
            has_xattn = (
                metrics.get('osc_xattn/query_norm', 0) > 0 and
                metrics.get('osc_xattn/key_norm', 0) > 0
            )
            if has_xattn:
                if qk_rel_ratio > qk_rel_ratio_warn_threshold:
                    qk_rel_ratio_warn_streak += 1
                else:
                    qk_rel_ratio_warn_streak = 0
            else:
                qk_rel_ratio_warn_streak = 0

            epoch_str = f"E{current_epoch}" if training_mode == "epochs" else ""
            print(f"Step {global_step:5d} {epoch_str}| "
                  f"Loss: {avg_loss:.4f} (loop-best: {best_loss_val:.3f} @step{best_loss_step}) | "
                  f"Cert: {best_cert_val:.2f} @step{best_cert_step} | "
                  f"ΔLoss: {loss_impr:+.3f} | "
                  f"Docs: {total_documents} | "
                  f"{steps_per_sec:.2f} it/s")

            # Compact tick evolution monitoring (internal obs residual check)
            pred_total = metrics.get('tick_pred_total_delta', 0)
            pred_late = metrics.get('tick_pred_late_delta', 0)
            surp_total = metrics.get('tick_surp_total_delta', 0)
            surp_late = metrics.get('tick_surp_late_delta', 0)
            pred_plateau = "⚠PLATEAU" if metrics.get('tick_pred_plateau', 0) > 0.5 else "✓"
            surp_plateau = "⚠PLATEAU" if metrics.get('tick_surp_plateau', 0) > 0.5 else "✓"
            print(f"         Ticks | Pred: Δ={pred_total:.4f} (late={pred_late:.4f}) {pred_plateau} | "
                  f"Surp: Δ={surp_total:.4f} (late={surp_late:.4f}) {surp_plateau}")

            # NLM gradient health monitoring
            pred_w1_grad = metrics.get('nlm_pred/w1_grad_norm', 0)
            pred_w2_grad = metrics.get('nlm_pred/w2_grad_norm', 0)
            surp_w1_grad = metrics.get('nlm_surp/w1_grad_norm', 0)
            surp_w2_grad = metrics.get('nlm_surp/w2_grad_norm', 0)
            pred_grad_status = "⚠VANISH" if pred_w1_grad < 1e-6 else "✓"
            surp_grad_status = "⚠VANISH" if surp_w1_grad < 1e-6 else "✓"
            print(f"          NLM | Pred: ∇w1={pred_w1_grad:.2e} ∇w2={pred_w2_grad:.2e} {pred_grad_status} | "
                  f"Surp: ∇w1={surp_w1_grad:.2e} ∇w2={surp_w2_grad:.2e} {surp_grad_status}")

            # Oscillator health monitoring
            osc_amp_mean = metrics.get('oscillator/amplitude_mean', 0)
            osc_phase_entropy = metrics.get('oscillator/phase_entropy', 0)
            osc_attn_entropy = metrics.get('oscillator/attn_entropy', 0)
            osc_active_frac = metrics.get('oscillator/active_frac', 0)
            osc_amp_mod = metrics.get('oscillator/amp_mod_mean', 0)
            osc_phase_mod = metrics.get('oscillator/phase_mod_mean', 0)
            osc_out_norm = metrics.get('oscillator/output_norm', 0)
            osc_health_issues = []
            if osc_active_frac < 0.1:
                osc_health_issues.append("FEW_ACTIVE")
            if osc_amp_mod < 0.001 and osc_phase_mod < 0.001:
                osc_health_issues.append("NO_MODULATION")
            if osc_out_norm > 100:
                osc_health_issues.append("EXPLODING")
            # Low attention entropy means always selecting same oscillators
            if osc_attn_entropy > 0 and osc_attn_entropy < 1.0:
                osc_health_issues.append("LOW_ATTN_DIV")
            osc_health_str = " ".join(f"⚠{i}" for i in osc_health_issues) if osc_health_issues else "✓"
            print(f"   Oscillator | amp={osc_amp_mean:.3f} φH={osc_phase_entropy:.2f} aH={osc_attn_entropy:.2f} "
                  f"active={osc_active_frac:.2f} mod(a/φ)={osc_amp_mod:.3f}/{osc_phase_mod:.3f} "
                  f"out={osc_out_norm:.2f} {osc_health_str}")

            # Oscillator cross-attention gradient monitoring
            osc_embed_grad = metrics.get('osc_xattn/embed_grad', 0)
            osc_query_grad = metrics.get('osc_xattn/query_grad', 0)
            osc_key_grad = metrics.get('osc_xattn/key_grad', 0)
            osc_value_grad = metrics.get('osc_xattn/value_grad', 0)
            osc_qk_ratio = metrics.get('osc_xattn/qk_grad_ratio', 0)
            osc_query_rel = metrics.get('osc_xattn/query_rel_update', 0)
            osc_key_rel = metrics.get('osc_xattn/key_rel_update', 0)
            osc_attn_ratio = metrics.get('osc_xattn/attn_query_ratio', 0)
            xattn_grad_issues = []
            if osc_embed_grad == 0:
                xattn_grad_issues.append("EMBED_NOGRAD")
            elif osc_embed_grad < 1e-6:
                xattn_grad_issues.append("EMBED_VANISH")
            if osc_query_grad == 0:
                xattn_grad_issues.append("QUERY_NOGRAD")
            elif osc_query_grad < 1e-6:
                xattn_grad_issues.append("QUERY_VANISH")
            if osc_key_grad == 0:
                xattn_grad_issues.append("KEY_NOGRAD")
            elif osc_key_grad < 1e-6:
                xattn_grad_issues.append("KEY_VANISH")
            if osc_value_grad == 0:
                xattn_grad_issues.append("VALUE_NOGRAD")
            elif osc_value_grad < 1e-6:
                xattn_grad_issues.append("VALUE_VANISH")
            if qk_rel_ratio_warn_streak >= qk_rel_ratio_warn_min_logs:
                xattn_grad_issues.append(f"QUERY_DOMINANTx{qk_rel_ratio_warn_streak}")
            xattn_status = " ".join(f"⚠{i}" for i in xattn_grad_issues) if xattn_grad_issues else "✓"
            print(f"    Osc XAttn | ∇embed={osc_embed_grad:.2e} ∇query={osc_query_grad:.2e} "
                  f"∇key={osc_key_grad:.2e} ∇value={osc_value_grad:.2e} "
                  f"q/k={osc_qk_ratio:.1e} relΔq={osc_query_rel:.1e} relΔk={osc_key_rel:.1e} "
                  f"a/q={osc_attn_ratio:.1e} {xattn_status}")

            # Gate health monitoring (critical for loop learning)
            gate_mean = metrics.get('gate/mean', 0)
            gate_std = metrics.get('gate/std', 0)
            gate_min = metrics.get('gate/min', 0)
            gate_max = metrics.get('gate/max', 0)
            loop_feat_diff = metrics.get('gate/loop_features_diff', 0)
            gate_health_issues = []
            if gate_mean < 0.15:
                gate_health_issues.append("TOO_SMALL")
            elif gate_mean > 0.85:
                gate_health_issues.append("TOO_LARGE")
            if loop_feat_diff < 0.01:
                gate_health_issues.append("NO_INFLUENCE")
            gate_health_str = " ".join(f"⚠{i}" for i in gate_health_issues) if gate_health_issues else "✓"
            print(f"         Gate | mean={gate_mean:.3f} std={gate_std:.3f} "
                  f"[{gate_min:.2f}-{gate_max:.2f}] loop_diff={loop_feat_diff:.4f} {gate_health_str}")

            # Per-horizon loss breakdown (shows first vs last loop step)
            # Lower = better. If world model helps, long should improve more than imm
            imm_0 = metrics.get('loss/step0_immediate_loss', 0)
            short_0 = metrics.get('loss/step0_shortterm_loss', 0)
            long_0 = metrics.get('loss/step0_longterm_loss', 0)
            # Try to get last step (step2 for 3-step loop)
            last_step = config.num_loop_steps - 1
            imm_last = metrics.get(f'loss/step{last_step}_immediate_loss', imm_0)
            short_last = metrics.get(f'loss/step{last_step}_shortterm_loss', short_0)
            long_last = metrics.get(f'loss/step{last_step}_longterm_loss', long_0)
            # Show deltas (negative = improvement)
            print(f"      Horizon | imm: {imm_0:.3f}→{imm_last:.3f} | short: {short_0:.3f}→{short_last:.3f} | long: {long_0:.3f}→{long_last:.3f}")

            # === NEW METRICS ===
            # Oscillator frequency bands
            world_stats = model.global_sync.get_world_state_stats()
            slow_amp = world_stats.get('osc/slow_amp_mean')
            mid_amp = world_stats.get('osc/mid_amp_mean')
            fast_amp = world_stats.get('osc/fast_amp_mean')
            freq_ratio = world_stats.get('osc/freq_band_ratio')
            if slow_amp is not None:
                print(f"    Osc Bands | slow={slow_amp:.3f} mid={mid_amp:.3f} fast={fast_amp:.3f} ratio={freq_ratio:.2f}")

            # Feature write attention
            fw_entropy = world_stats.get('feature_write/pos_attn_entropy')
            fw_top1 = world_stats.get('feature_write/top1_weight')
            fw_top5 = world_stats.get('feature_write/top5_weight')
            fw_surp_corr = world_stats.get('feature_write/surprise_correlation')
            fw_surp_abs_corr = world_stats.get('feature_write/surprise_abs_correlation')
            surp_mean = world_stats.get('surprise/mean')
            surp_std = world_stats.get('surprise/std')
            surp_pos = world_stats.get('surprise/pos_frac')
            if fw_entropy is not None:
                surp_str = f" surp_corr={fw_surp_corr:.2f}" if fw_surp_corr is not None else ""
                abs_str = f" abs_corr={fw_surp_abs_corr:.2f}" if fw_surp_abs_corr is not None else ""
                print(f"   Feat Write | entropy={fw_entropy:.2f} top1={fw_top1:.3f} top5={fw_top5:.3f}{surp_str}{abs_str}")
            if surp_mean is not None:
                print(f"     Surprise | mean={surp_mean:.3f} std={surp_std:.3f} pos={surp_pos:.2f}")

            # Loop trajectory
            l0 = metrics.get('loop/loss_step0')
            l1 = metrics.get('loop/loss_step1')
            ln = metrics.get('loop/loss_stepN')
            mono = metrics.get('loop/monotonic_improve')
            if l0 is not None:
                mono_str = f" mono={mono:.0%}" if mono is not None else ""
                print(f"    Loop Traj | L0={l0:.3f}→L1={l1:.3f}→LN={ln:.3f}{mono_str}")

            # CTM tick selection
            t1 = metrics.get('ctm/pred_t1_mean')
            t2 = metrics.get('ctm/pred_t2_mean')
            agree = metrics.get('ctm/t1_t2_agreement')
            spread = metrics.get('ctm/tick_loss_spread')
            if t1 is not None:
                print(f"          CTM | t1={t1:.1f} t2={t2:.1f} agree={agree:.0%} spread={spread:.4f}")

            # Oscillator ablation (only on diagnostic steps)
            osc_abl_deg = metrics.get('osc_ablation/degradation')
            osc_abl_rel = metrics.get('osc_ablation/relative_help')
            if osc_abl_deg is not None:
                abl_status = "✓ HELPING" if osc_abl_deg > 0.001 else ("≈ NEUTRAL" if abs(osc_abl_deg) < 0.001 else "✗ HURTING")
                print(f"  Osc Ablation | degradation={osc_abl_deg:+.4f} relative_help={osc_abl_rel:.1%} {abl_status}")

            # Temporal awareness metrics (phase-tagged writes, self-state)
            phase_dist = world_stats.get('temporal/phase_dist_mean')
            write_frac = world_stats.get('temporal/write_fraction')
            self_state_div = world_stats.get('temporal/self_state_diversity')
            self_world_align = world_stats.get('temporal/self_world_alignment')
            world_write_frac = world_stats.get('temporal/world_write_fraction')
            self_write_frac = world_stats.get('temporal/self_write_fraction')
            if phase_dist is not None:
                print(f"     Temporal | phase_dist={phase_dist:.2f} write_frac={write_frac:.2f} self_div={self_state_div:.3f}")
                print(f"   Self-World | align={self_world_align:.3f} w_write={world_write_frac:.0%} s_write={self_write_frac:.0%}")

            # Self-state mechanism metrics (autobiographical memory health)
            ss_contrib = world_stats.get('self_state/contribution_norm')
            ss_ratio = world_stats.get('self_state/value_ratio')
            ss_coverage = world_stats.get('self_state/stored_coverage')
            ss_comp_std = world_stats.get('self_state/compressor_std')
            ss_step_change = world_stats.get('self_state/step_change')
            ss_write_gate = world_stats.get('self_state/write_gate')
            ss_gated_amp_mean = world_stats.get('self_state/gated_amp_mean')
            ss_gated_amp_max = world_stats.get('self_state/gated_amp_max')
            ss_gated_amp_over = world_stats.get('self_state/gated_amp_over_threshold')
            if ss_contrib is not None:
                gate_str = f" gate={ss_write_gate:.3f}" if ss_write_gate is not None else ""
                amp_str = ""
                if ss_gated_amp_mean is not None and ss_gated_amp_max is not None and ss_gated_amp_over is not None:
                    phase_write_threshold = model.global_sync.oscillatory_world.config.phase_write_threshold_self
                    amp_str = (
                        f" amp_mean={ss_gated_amp_mean:.3e} amp_max={ss_gated_amp_max:.3e} "
                        f"over={ss_gated_amp_over:.1%} thr={phase_write_threshold:.2f}"
                    )
                print(
                    f"   Self-State | contrib={ss_contrib:.3f} ratio={ss_ratio:.3f} "
                    f"coverage={ss_coverage:.2f} comp_std={ss_comp_std:.3f} Δstep={ss_step_change:.3f}{gate_str}{amp_str}"
                )

            # Write time spread metrics (is autobiographical memory accumulating?)
            wt_spread = world_stats.get('temporal/write_phase_spread')
            wt_age_range = world_stats.get('temporal/estimated_age_range')
            wt_uniqueness = world_stats.get('temporal/self_state_uniqueness')
            wt_similarity = world_stats.get('temporal/self_state_avg_similarity')
            if wt_spread is not None:
                print(
                    f"  Write Spread | phase_std={wt_spread:.3f} age_range={wt_age_range:.1f} "
                    f"uniqueness={wt_uniqueness:.3f} similarity={wt_similarity:.3f}"
                )

            # Self-state INPUT diagnostics (why is diversity low?)
            ss_in_sync = world_stats.get('ss_input/sync_norm')
            ss_in_pred = world_stats.get('ss_input/pred_norm')
            ss_in_surp = world_stats.get('ss_input/surprise')
            ss_in_conf = world_stats.get('ss_input/confidence')
            ss_normed_sync = world_stats.get('ss_normed/sync_norm')
            ss_normed_pred = world_stats.get('ss_normed/pred_norm')
            ss_normed_surp = world_stats.get('ss_normed/surprise_norm')
            ss_normed_conf = world_stats.get('ss_normed/confidence_norm')
            ss_in_change = world_stats.get('ss_input/step_change')
            ss_out_change = world_stats.get('self_state/step_change')
            if ss_in_sync is not None:
                # Compare input change vs output change to see if compressor is collapsing
                ratio = ss_out_change / (ss_in_change + 1e-8) if ss_in_change else 0
                # Show raw vs normalized to verify normalization is working
                print(
                    f"    SS Raw    | sync={ss_in_sync:.2f} pred={ss_in_pred:.2f} "
                    f"surp={ss_in_surp:.2f} conf={ss_in_conf:.2f}"
                )
                if ss_normed_sync is not None:
                    print(
                        f"    SS Normed | sync={ss_normed_sync:.2f} pred={ss_normed_pred:.2f} "
                        f"surp={ss_normed_surp:.2f} conf={ss_normed_conf:.2f} | "
                        f"Δin={ss_in_change:.2f} Δout={ss_out_change:.2f} ratio={ratio:.2f}"
                    )

            # WandB logging (consolidated - only essential metrics)
            if config.wandb_project:
                log_dict = {
                    # Training basics
                    'loss': avg_loss,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'steps_per_sec': steps_per_sec,

                    # Epoch/document tracking
                    'epoch': current_epoch,
                    'total_documents': total_documents,
                    'documents_in_epoch': documents_in_epoch,

                    # === OVERALL BEST STEPS ===
                    'best/loss_value': metrics.get('best_loss_value', 0),
                    'best/loss_step': best_loss_step,
                    'best/certainty_value': metrics.get('best_certainty_value', 0),
                    'best/certainty_step': best_cert_step,

                    # Running averages (overall - which step is best ON AVERAGE)
                    'avg_best/loss_step': avg_best_loss_step,
                    'avg_best/certainty_step': avg_best_cert_step,

                    # === PREDICTION MODULE BEST STEPS ===
                    'pred/best_loss_step': pred_best_loss_step,
                    'pred/best_loss_value': metrics.get('pred_best_loss_value', 0),
                    'pred/best_cert_step': pred_best_cert_step,
                    'pred/best_cert_value': metrics.get('pred_best_cert_value', 0),
                    # Running averages for PredictionCTM
                    'pred/avg_best_loss_step': avg_pred_best_loss_step,
                    'pred/avg_best_cert_step': avg_pred_best_cert_step,

                    # === SURPRISE MODULE BEST STEPS ===
                    'surp/best_loss_step': surp_best_loss_step,
                    'surp/best_loss_value': metrics.get('surp_best_loss_value', 0),
                    'surp/best_cert_step': surp_best_cert_step,
                    'surp/best_cert_value': metrics.get('surp_best_cert_value', 0),
                    # Running averages for SurpriseCTM
                    'surp/avg_best_loss_step': avg_surp_best_loss_step,
                    'surp/avg_best_cert_step': avg_surp_best_cert_step,

                    # Final step metrics
                    'final/loss': metrics.get('final_step_loss', 0),
                    'final/certainty': metrics.get('final_step_certainty', 0),
                    'final/surprise': metrics.get('final_step_surprise', 0),

                    # Improvement from iterating
                    'improvement/loss': metrics.get('loss_improvement', 0),
                    'improvement/certainty': metrics.get('certainty_improvement', 0),

                    # Cross-module sync
                    'sync/pred_to_surp': metrics.get('cross_sync_pred_surp', 0.5),

                    # === CTM INTERNAL TICK SELECTION ===
                    # Which internal tick the CTM loss selects (within each module)
                    'ctm_ticks/pred_best': metrics.get('ctm_pred_avg_best_tick', 0),
                    'ctm_ticks/pred_certain': metrics.get('ctm_pred_avg_certain_tick', 0),
                    'ctm_ticks/surp_best': metrics.get('ctm_surp_avg_best_tick', 0),
                    'ctm_ticks/surp_certain': metrics.get('ctm_surp_avg_certain_tick', 0),
                    'ctm/pred_t1_mean': metrics.get('ctm/pred_t1_mean', 0),
                    'ctm/pred_t2_mean': metrics.get('ctm/pred_t2_mean', 0),
                    'ctm/t1_t2_agreement': metrics.get('ctm/t1_t2_agreement', 0),
                    'ctm/tick_loss_spread': metrics.get('ctm/tick_loss_spread', 0),
                    'sync/surp_to_pred': metrics.get('cross_sync_surp_pred', 0.5),

                    # === TICK EVOLUTION (internal obs residual monitoring) ===
                    'tick_evolution/pred_total_delta': metrics.get('tick_pred_total_delta', 0),
                    'tick_evolution/pred_late_delta': metrics.get('tick_pred_late_delta', 0),
                    'tick_evolution/pred_plateau': metrics.get('tick_pred_plateau', 0),
                    'tick_evolution/surp_total_delta': metrics.get('tick_surp_total_delta', 0),
                    'tick_evolution/surp_late_delta': metrics.get('tick_surp_late_delta', 0),
                    'tick_evolution/surp_plateau': metrics.get('tick_surp_plateau', 0),
                    # === LOOP REFINEMENT ===
                    'loop/loss_step0': metrics.get('loop/loss_step0', 0),
                    'loop/loss_step1': metrics.get('loop/loss_step1', 0),
                    'loop/loss_stepN': metrics.get('loop/loss_stepN', 0),
                    'loop/monotonic_improve': metrics.get('loop/monotonic_improve', 0),
                    'loop/obs_delta_01': metrics.get('loop/obs_delta_01', 0),
                    'loop/obs_delta_total': metrics.get('loop/obs_delta_total', 0),

                    # === NLM GRADIENT/WEIGHT MONITORING ===
                    'nlm/pred_w1_norm': metrics.get('nlm_pred/w1_norm', 0),
                    'nlm/pred_w2_norm': metrics.get('nlm_pred/w2_norm', 0),
                    'nlm/pred_w1_grad': metrics.get('nlm_pred/w1_grad_norm', 0),
                    'nlm/pred_w2_grad': metrics.get('nlm_pred/w2_grad_norm', 0),
                    'nlm/surp_w1_norm': metrics.get('nlm_surp/w1_norm', 0),
                    'nlm/surp_w2_norm': metrics.get('nlm_surp/w2_norm', 0),
                    'nlm/surp_w1_grad': metrics.get('nlm_surp/w1_grad_norm', 0),
                    'nlm/surp_w2_grad': metrics.get('nlm_surp/w2_grad_norm', 0),

                    # === OSCILLATORY WORLD MODEL ===
                    # Oscillator state that evolves continuously
                    'world_state/norm': metrics.get('world_state/norm', 0),
                    'world_state/mean': metrics.get('world_state/mean', 0),
                    'world_state/std': metrics.get('world_state/std', 0),
                    'world_state/update_count': metrics.get('world_state/update_count', 0),

                    # === OSCILLATOR METRICS ===
                    # Phase dynamics
                    'oscillator/phase_mean': metrics.get('oscillator/phase_mean', 0),
                    'oscillator/phase_std': metrics.get('oscillator/phase_std', 0),
                    'oscillator/phase_entropy': metrics.get('oscillator/phase_entropy', 0),

                    # Amplitude dynamics
                    'oscillator/amplitude_mean': metrics.get('oscillator/amplitude_mean', 0),
                    'oscillator/amplitude_std': metrics.get('oscillator/amplitude_std', 0),
                    'oscillator/amplitude_max': metrics.get('oscillator/amplitude_max', 0),

                    # Frequency utilization
                    'oscillator/active_frac': metrics.get('oscillator/active_frac', 0),
                    'oscillator/freq_weighted_amp': metrics.get('oscillator/freq_weighted_amp', 0),

                    # Modulation strength (key metric - shows gradients flowing!)
                    'oscillator/amp_mod_mean': metrics.get('oscillator/amp_mod_mean', 0),
                    'oscillator/phase_mod_mean': metrics.get('oscillator/phase_mod_mean', 0),

                    # Output statistics
                    'oscillator/output_norm': metrics.get('oscillator/output_norm', 0),
                    'oscillator/output_mean': metrics.get('oscillator/output_mean', 0),
                    'oscillator/output_std': metrics.get('oscillator/output_std', 0),

                    # Frequency band utilization
                    'osc/slow_amp_mean': metrics.get('osc/slow_amp_mean', 0),
                    'osc/mid_amp_mean': metrics.get('osc/mid_amp_mean', 0),
                    'osc/fast_amp_mean': metrics.get('osc/fast_amp_mean', 0),
                    'osc/freq_band_ratio': metrics.get('osc/freq_band_ratio', 0),

                    # Cross-attention entropy (measures oscillator selection diversity)
                    'oscillator/attn_entropy': metrics.get('oscillator/attn_entropy', 0),
                    'osc_xattn/entropy_step0': metrics.get('osc_xattn/entropy_step0', 0),
                    'osc_xattn/entropy_stepN': metrics.get('osc_xattn/entropy_stepN', 0),
                    'osc_xattn/entropy_delta': metrics.get('osc_xattn/entropy_delta', 0),

                    # Cross-attention gradients (critical for oscillator selection learning)
                    'osc_xattn/embed_grad': metrics.get('osc_xattn/embed_grad', 0),
                    'osc_xattn/query_grad': metrics.get('osc_xattn/query_grad', 0),
                    'osc_xattn/key_grad': metrics.get('osc_xattn/key_grad', 0),
                    'osc_xattn/value_grad': metrics.get('osc_xattn/value_grad', 0),
                    'osc_xattn/embed_norm': metrics.get('osc_xattn/embed_norm', 0),
                    'osc_xattn/query_norm': metrics.get('osc_xattn/query_norm', 0),
                    'osc_xattn/key_norm': metrics.get('osc_xattn/key_norm', 0),
                    'osc_xattn/value_norm': metrics.get('osc_xattn/value_norm', 0),
                    'osc_xattn/query_rel_update': metrics.get('osc_xattn/query_rel_update', 0),
                    'osc_xattn/key_rel_update': metrics.get('osc_xattn/key_rel_update', 0),
                    'osc_xattn/value_rel_update': metrics.get('osc_xattn/value_rel_update', 0),
                    'osc_xattn/qk_grad_ratio': metrics.get('osc_xattn/qk_grad_ratio', 0),
                    'osc_xattn/qk_rel_update_ratio': metrics.get('osc_xattn/qk_rel_update_ratio', 0),
                    'osc_xattn/attn_out_norm': metrics.get('osc_xattn/attn_out_norm', 0),
                    'osc_xattn/query_act_norm': metrics.get('osc_xattn/query_act_norm', 0),
                    'osc_xattn/attn_query_ratio': metrics.get('osc_xattn/attn_query_ratio', 0),
                    'osc_xattn/top1_osc_weight': metrics.get('osc_xattn/top1_osc_weight', 0),
                    'osc_xattn/key_query_cosine': metrics.get('osc_xattn/key_query_cosine', 0),

                    # Auxiliary prediction loss
                    'loss/auxiliary_prediction': metrics.get('loss/auxiliary_prediction_loss', 0),

                    # === STATE COMBINER GATE METRICS ===
                    # Critical for loop learning - gate controls observation influence
                    'gate/mean': metrics.get('gate/mean', 0),
                    'gate/std': metrics.get('gate/std', 0),
                    'gate/min': metrics.get('gate/min', 0),
                    'gate/max': metrics.get('gate/max', 0),
                    'gate/loop_features_diff': metrics.get('gate/loop_features_diff', 0),
                    'gate/grad_norm': metrics.get('gate/grad_norm', 0),
                    'gate/position_variance': metrics.get('gate/position_variance', 0),
                    'gate/mean_step0': metrics.get('gate/mean_step0', 0),
                    'gate/mean_stepN': metrics.get('gate/mean_stepN', 0),
                    'gate/step0_vs_stepN': metrics.get('gate/step0_vs_stepN', 0),

                    # === FEATURE WRITE ATTENTION ===
                    'feature_write/pos_attn_entropy': metrics.get('feature_write/pos_attn_entropy', 0),
                    'feature_write/top1_weight': metrics.get('feature_write/top1_weight', 0),
                    'feature_write/top5_weight': metrics.get('feature_write/top5_weight', 0),
                    'feature_write/surprise_correlation': metrics.get('feature_write/surprise_correlation', 0),
                    'feature_write/surprise_abs_correlation': metrics.get('feature_write/surprise_abs_correlation', 0),

                    # === SURPRISE COUPLING ===
                    'surprise/pred_error_corr': metrics.get('surprise/pred_error_corr', 0),
                    'surprise/raw_vs_calibrated': metrics.get('surprise/raw_vs_calibrated', 0),
                    'surprise/write_gate_mean': metrics.get('surprise/write_gate_mean', 0),
                    'surprise/mean': metrics.get('surprise/mean', 0),
                    'surprise/std': metrics.get('surprise/std', 0),
                    'surprise/pos_frac': metrics.get('surprise/pos_frac', 0),

                    # === TEMPORAL AWARENESS (phase-tagged writes) ===
                    'temporal/phase_dist_mean': metrics.get('temporal/phase_dist_mean', 0),
                    'temporal/phase_dist_std': metrics.get('temporal/phase_dist_std', 0),
                    'temporal/estimated_age_mean': metrics.get('temporal/estimated_age_mean', 0),
                    'temporal/estimated_age_std': metrics.get('temporal/estimated_age_std', 0),
                    'temporal/write_fraction': metrics.get('temporal/write_fraction', 0),
                    'temporal/write_strength_mean': metrics.get('temporal/write_strength_mean', 0),
                    # World vs self oscillator breakdowns
                    'temporal/world_phase_dist_mean': metrics.get('temporal/world_phase_dist_mean', 0),
                    'temporal/world_estimated_age_mean': metrics.get('temporal/world_estimated_age_mean', 0),
                    'temporal/world_write_fraction': metrics.get('temporal/world_write_fraction', 0),
                    'temporal/self_phase_dist_mean': metrics.get('temporal/self_phase_dist_mean', 0),
                    'temporal/self_estimated_age_mean': metrics.get('temporal/self_estimated_age_mean', 0),
                    'temporal/self_write_fraction': metrics.get('temporal/self_write_fraction', 0),

                    # === SELF-STATE TRACKING (autobiographical memory) ===
                    'temporal/self_state_diversity': metrics.get('temporal/self_state_diversity', 0),
                    'temporal/self_state_recency_ratio': metrics.get('temporal/self_state_recency_ratio', 0),

                    # === WRITE TIME SPREAD (memory accumulation) ===
                    'temporal/write_phase_spread': metrics.get('temporal/write_phase_spread', 0),
                    'temporal/write_phase_spread_world': metrics.get('temporal/write_phase_spread_world', 0),
                    'temporal/write_phase_spread_self': metrics.get('temporal/write_phase_spread_self', 0),
                    'temporal/estimated_age_range': metrics.get('temporal/estimated_age_range', 0),
                    'temporal/estimated_age_range_world': metrics.get('temporal/estimated_age_range_world', 0),
                    'temporal/estimated_age_range_self': metrics.get('temporal/estimated_age_range_self', 0),
                    'temporal/self_state_uniqueness': metrics.get('temporal/self_state_uniqueness', 0),
                    'temporal/self_state_avg_similarity': metrics.get('temporal/self_state_avg_similarity', 0),

                    # === SELF-WORLD COHERENCE ===
                    'temporal/self_world_alignment': metrics.get('temporal/self_world_alignment', 0),
                    'temporal/world_coherence': metrics.get('temporal/world_coherence', 0),
                    'temporal/self_coherence': metrics.get('temporal/self_coherence', 0),
                    'temporal/write_timing_coherence': metrics.get('temporal/write_timing_coherence', 0),

                    # === SELF-STATE MECHANISM (autobiographical memory health) ===
                    'self_state/contribution_norm': metrics.get('self_state/contribution_norm', 0),
                    'self_state/value_ratio': metrics.get('self_state/value_ratio', 0),
                    'self_state/stored_coverage': metrics.get('self_state/stored_coverage', 0),
                    'self_state/compressor_std': metrics.get('self_state/compressor_std', 0),
                    'self_state/step_change': metrics.get('self_state/step_change', 0),
                    'self_state/write_gate': metrics.get('self_state/write_gate', 0),
                    'self_state/gated_amp_mean': metrics.get('self_state/gated_amp_mean', 0),
                    'self_state/gated_amp_max': metrics.get('self_state/gated_amp_max', 0),
                    'self_state/gated_amp_over_threshold': metrics.get('self_state/gated_amp_over_threshold', 0),
                    'self_state/change': metrics.get('self_state/change', 0),
                }
                wandb.log(log_dict, step=global_step)

                # NLM activation visualization (less frequent)
                if should_visualize and outputs_for_diag is not None:
                    try:
                        # Create NLM activation heatmap grid
                        fig_nlm = create_nlm_activation_grid(outputs_for_diag)
                        wandb.log({
                            "visualizations/nlm_activations_heatmap": wandb.Image(fig_to_image(fig_nlm))
                        }, step=global_step)

                        # Create NLM neuron line plots
                        fig_lines = create_nlm_neuron_lines(outputs_for_diag)
                        wandb.log({
                            "visualizations/nlm_neuron_lines": wandb.Image(fig_to_image(fig_lines))
                        }, step=global_step)

                        # Create cross-module sync plot (only if surprise is enabled)
                        fig_sync = create_cross_module_sync_plot(outputs_for_diag)
                        if fig_sync is not None:
                            wandb.log({
                                "visualizations/cross_module_sync": wandb.Image(fig_to_image(fig_sync))
                            }, step=global_step)

                        # Create oscillator visualization
                        try:
                            osc_world = model.global_sync.oscillatory_world
                            if osc_world is not None:
                                # Log oscillator state as histograms
                                osc_state = osc_world.get_oscillator_state()
                                wandb.log({
                                    "oscillator_viz/frequencies": wandb.Histogram(osc_state['frequencies'].cpu().numpy()),
                                    "oscillator_viz/amplitudes": wandb.Histogram(osc_state['current_amplitudes'].cpu().numpy()),
                                    "oscillator_viz/phases": wandb.Histogram(osc_state['phases'].cpu().numpy()),
                                    "oscillator_viz/amp_modulation": wandb.Histogram(osc_state['last_amp_mod'].cpu().numpy()),
                                    "oscillator_viz/phase_modulation": wandb.Histogram(osc_state['last_phase_mod'].cpu().numpy()),
                                }, step=global_step)
                                print(f"  [Viz] Logged oscillator histograms")
                        except Exception as e:
                            print(f"  [Viz] Warning: Failed to create oscillator visualization: {e}")

                        if fig_sync is not None:
                            print(f"  [Viz] Logged NLM heatmap, neuron lines, and cross-module sync plots")
                        else:
                            print(f"  [Viz] Logged NLM heatmap and neuron lines (no cross-module sync in single-module mode)")
                    except Exception as e:
                        print(f"  [Viz] Warning: Failed to create visualization: {e}")

            running_loss = 0.0

        # Evaluation
        if global_step % config.eval_every == 0:
            # Use same batch for eval (simpler)
            eval_metrics = eval_step(model, features, config)

            print(f"  [Eval] Loss: {eval_metrics['eval/loss']:.4f}")

            if config.wandb_project:
                wandb.log(eval_metrics, step=global_step)

    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nTraining Stats:")
    print(f"  Total steps: {global_step}")
    print(f"  Total epochs: {current_epoch}")
    print(f"  Total documents: {total_documents}")
    elapsed_total = time.time() - start_time
    print(f"  Total time: {elapsed_total/60:.1f} minutes")
    if num_logged > 0:
        print(f"\nOverall (Global Loop):")
        print(f"  Average best loss step:      {best_loss_step_sum / num_logged:.2f}")
        print(f"  Average best certainty step: {best_certainty_step_sum / num_logged:.2f}")
        print(f"\nPredictionCTM Module:")
        print(f"  Average best loss step:      {pred_best_loss_step_sum / num_logged:.2f}")
        print(f"  Average best certainty step: {pred_best_cert_step_sum / num_logged:.2f}")
        print(f"\nSurpriseCTM Module:")
        print(f"  Average best loss step:      {surp_best_loss_step_sum / num_logged:.2f}")
        print(f"  Average best certainty step: {surp_best_cert_step_sum / num_logged:.2f}")
        print(f"\n(out of {config.num_loop_steps} loop steps, 0-indexed)")
    print("="*60)

    if config.wandb_project:
        # Log final summary
        wandb.log({
            # Overall
            'summary/avg_best_loss_step': best_loss_step_sum / num_logged if num_logged > 0 else 0,
            'summary/avg_best_certainty_step': best_certainty_step_sum / num_logged if num_logged > 0 else 0,
            # PredictionCTM
            'summary/pred_avg_best_loss_step': pred_best_loss_step_sum / num_logged if num_logged > 0 else 0,
            'summary/pred_avg_best_cert_step': pred_best_cert_step_sum / num_logged if num_logged > 0 else 0,
            # SurpriseCTM
            'summary/surp_avg_best_loss_step': surp_best_loss_step_sum / num_logged if num_logged > 0 else 0,
            'summary/surp_avg_best_cert_step': surp_best_cert_step_sum / num_logged if num_logged > 0 else 0,
        })
        wandb.finish()


if __name__ == '__main__':
    main()
