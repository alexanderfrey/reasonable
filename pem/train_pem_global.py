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
        if pred_certain_ticks:
            metrics['ctm_pred_avg_certain_tick'] = float(np.mean(pred_certain_ticks))
        if surp_best_ticks:
            metrics['ctm_surp_avg_best_tick'] = float(np.mean(surp_best_ticks))
        if surp_certain_ticks:
            metrics['ctm_surp_avg_certain_tick'] = float(np.mean(surp_certain_ticks))

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

    # Improvement across steps (did iterating help?)
    if len(outputs) > 1:
        metrics['loss_improvement'] = step_losses[0] - step_losses[-1]  # Positive = improved
        metrics['certainty_improvement'] = step_certainties[-1] - step_certainties[0]  # Positive = improved

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

    return metrics


def print_diagnostic_report(
    outputs: List,
    targets: Dict[str, torch.Tensor],
    step: int,
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
        # Also track how much loop_features differ from raw features
        transformed = model.state_combiner_transform(combined)
        loop_features = features + gate_values * (transformed - features)
        loop_features_diff = (loop_features - features).abs().mean().item()
        metrics['gate/loop_features_diff'] = loop_features_diff

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
            print_diagnostic_report(outputs_for_diag, targets_for_diag, global_step + 1)

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

            epoch_str = f"E{current_epoch}" if training_mode == "epochs" else ""
            print(f"Step {global_step:5d} {epoch_str}| "
                  f"Loss: {avg_loss:.4f} (best: {best_loss_val:.3f} @step{best_loss_step}) | "
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
            osc_health_str = " ".join(f"⚠{i}" for i in osc_health_issues) if osc_health_issues else "✓"
            print(f"   Oscillator | amp={osc_amp_mean:.3f} φH={osc_phase_entropy:.2f} "
                  f"active={osc_active_frac:.2f} mod(a/φ)={osc_amp_mod:.3f}/{osc_phase_mod:.3f} "
                  f"out={osc_out_norm:.2f} {osc_health_str}")

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
                    'sync/surp_to_pred': metrics.get('cross_sync_surp_pred', 0.5),

                    # === TICK EVOLUTION (internal obs residual monitoring) ===
                    'tick_evolution/pred_total_delta': metrics.get('tick_pred_total_delta', 0),
                    'tick_evolution/pred_late_delta': metrics.get('tick_pred_late_delta', 0),
                    'tick_evolution/pred_plateau': metrics.get('tick_pred_plateau', 0),
                    'tick_evolution/surp_total_delta': metrics.get('tick_surp_total_delta', 0),
                    'tick_evolution/surp_late_delta': metrics.get('tick_surp_late_delta', 0),
                    'tick_evolution/surp_plateau': metrics.get('tick_surp_plateau', 0),

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

                    # Auxiliary prediction loss
                    'loss/auxiliary_prediction': metrics.get('loss/auxiliary_prediction_loss', 0),

                    # === STATE COMBINER GATE METRICS ===
                    # Critical for loop learning - gate controls observation influence
                    'gate/mean': metrics.get('gate/mean', 0),
                    'gate/std': metrics.get('gate/std', 0),
                    'gate/min': metrics.get('gate/min', 0),
                    'gate/max': metrics.get('gate/max', 0),
                    'gate/loop_features_diff': metrics.get('gate/loop_features_diff', 0),
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
