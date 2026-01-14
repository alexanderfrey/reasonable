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
import time
import io
from dataclasses import dataclass
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
    num_modules = 2  # Prediction, Surprise

    fig, axes = plt.subplots(
        num_steps, num_modules,
        figsize=(4 * num_modules, 3 * num_steps),
        squeeze=False,
    )

    module_names = ['Prediction', 'Surprise']

    for step_idx, output in enumerate(outputs):
        # Get activations for each module
        pred_activations = output.prediction_output.all_tick_activations  # List of (B, S, D_neurons)
        surp_activations = output.surprise.all_tick_activations  # List of (B, S, D_neurons)

        module_activations = [pred_activations, surp_activations]

        for mod_idx, (name, activations) in enumerate(zip(module_names, module_activations)):
            ax = axes[step_idx, mod_idx]

            if not activations:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'{name} (Step {step_idx})')
                continue

            # Stack: (T, B, S, D_neurons) -> average over B, S -> (T, D_neurons)
            stacked = torch.stack(activations, dim=0)  # (T, B, S, D)
            T, B, S, D = stacked.shape

            # Average over batch and positions (take subset of positions)
            pos_subset = min(S, max_positions)
            averaged = stacked[:, 0, :pos_subset, :].mean(dim=1)  # (T, D)

            # Subsample neurons if needed
            neuron_subset = min(D, max_neurons)
            neuron_indices = torch.linspace(0, D-1, neuron_subset).long()
            averaged = averaged[:, neuron_indices]  # (T, neuron_subset)

            # Convert to numpy for plotting
            data = averaged.detach().cpu().numpy()  # (T, neurons)

            # Plot heatmap: x=ticks, y=neurons
            im = ax.imshow(
                data.T,  # (neurons, T)
                aspect='auto',
                cmap='RdBu_r',
                vmin=-2, vmax=2,
            )

            ax.set_xlabel('Tick')
            ax.set_ylabel('Neuron')
            ax.set_title(f'{name} (Step {step_idx})')

            # Add tick labels
            ax.set_xticks(range(T))
            ax.set_xticklabels([f't{t}' for t in range(T)])

    # Add colorbar
    fig.colorbar(im, ax=axes, shrink=0.6, label='Activation')

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
    num_modules = 2  # Prediction, Surprise

    # Calculate grid size (sqrt layout for neurons)
    grid_rows = int(np.ceil(np.sqrt(num_neurons)))
    grid_cols = int(np.ceil(num_neurons / grid_rows))

    # Create figure with subplots for each module
    fig, axes = plt.subplots(
        grid_rows, grid_cols * num_modules,
        figsize=(3 * grid_cols * num_modules, 2 * grid_rows),
        squeeze=False,
    )

    module_names = ['Prediction', 'Surprise']
    colors = plt.cm.viridis(np.linspace(0, 1, num_steps))

    for mod_idx, name in enumerate(module_names):
        # Get activations from first output to determine neuron count
        if mod_idx == 0:
            sample_activations = outputs[0].prediction_output.all_tick_activations
        else:
            sample_activations = outputs[0].surprise.all_tick_activations

        if not sample_activations:
            continue

        # Get total neuron count
        total_neurons = sample_activations[0].shape[-1]
        neuron_indices = torch.linspace(0, total_neurons - 1, num_neurons).long()

        for neuron_plot_idx in range(num_neurons):
            row = neuron_plot_idx // grid_cols
            col = (mod_idx * grid_cols) + (neuron_plot_idx % grid_cols)
            ax = axes[row, col]

            neuron_idx = neuron_indices[neuron_plot_idx].item()

            # Plot each loop step as a different colored line
            for step_idx, output in enumerate(outputs):
                if mod_idx == 0:
                    activations = output.prediction_output.all_tick_activations
                else:
                    activations = output.surprise.all_tick_activations

                if not activations:
                    continue

                # Stack: (T, B, S, D_neurons)
                stacked = torch.stack(activations, dim=0)
                T, B, S, D = stacked.shape

                # Average over batch and positions, get single neuron
                pos_subset = min(S, max_positions)
                neuron_activation = stacked[:, 0, :pos_subset, neuron_idx].mean(dim=1)  # (T,)

                # Convert to numpy and plot
                data = neuron_activation.detach().cpu().numpy()
                ax.plot(range(T), data, 'o-', color=colors[step_idx],
                       label=f'Step {step_idx}' if neuron_plot_idx == 0 else None,
                       linewidth=1.5, markersize=4)

            ax.set_title(f'{name} N{neuron_idx}', fontsize=8)
            ax.set_xlabel('Tick', fontsize=7)
            ax.set_ylabel('Act', fontsize=7)
            ax.tick_params(axis='both', labelsize=6)
            ax.grid(True, alpha=0.3)

            # Set consistent y-axis limits
            ax.set_ylim(-3, 3)

    # Add legend to first subplot
    if num_steps > 1:
        axes[0, 0].legend(fontsize=6, loc='upper right')

    # Add module labels
    fig.text(0.25, 0.98, 'Prediction Module', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.75, 0.98, 'Surprise Module', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def create_cross_module_sync_plot(
    outputs: List,
) -> plt.Figure:
    """
    Create a plot showing cross-module synchronization over loop steps.

    Shows how Prediction and Surprise modules synchronize over time.
    """
    num_steps = len(outputs)

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
    """Compute detailed metrics for logging."""
    metrics = {}

    # Track per-step metrics for finding best step
    step_losses = []
    step_certainties = []

    for step_idx, output in enumerate(outputs):
        prefix = f"step{step_idx}"
        step_loss = 0.0

        # Prediction metrics
        for scale in ['immediate', 'shortterm', 'longterm']:
            pred = output.predictions[scale]
            target = targets[scale]
            valid = targets.get(f'{scale}_valid', None)

            if valid is not None and valid.any():
                pred_valid = pred[valid]
                target_valid = target[valid]
                cos_sim = F.cosine_similarity(pred_valid, target_valid, dim=-1).mean()
                metrics[f'{prefix}/{scale}_cos_sim'] = cos_sim.item()
                step_loss += (1 - cos_sim).item()

        # Surprise metrics
        metrics[f'{prefix}/surprise_magnitude_mean'] = output.surprise.magnitude.mean().item()
        metrics[f'{prefix}/surprise_magnitude_std'] = output.surprise.magnitude.std().item()
        metrics[f'{prefix}/surprise_raw_mean'] = output.surprise.raw.mean().item()
        metrics[f'{prefix}/surprise_certainty'] = output.surprise.certainty.item()

        # Prediction certainty
        metrics[f'{prefix}/prediction_certainty'] = output.prediction_output.certainty.item()

        # Combined certainty (average of prediction and surprise)
        combined_certainty = (output.prediction_output.certainty.item() + output.surprise.certainty.item()) / 2
        metrics[f'{prefix}/combined_certainty'] = combined_certainty

        # Global sync metrics
        sync = output.global_sync
        metrics[f'{prefix}/global_sync_mean'] = sync.sync.mean().item()
        metrics[f'{prefix}/global_sync_std'] = sync.sync.std().item()

        # Cross-module sync (2x2 matrix for pred-surp)
        cross_sync = sync.cross_module_sync  # (2, 2, B, S)
        metrics[f'{prefix}/cross_sync_pred_pred'] = cross_sync[0, 0].mean().item()
        metrics[f'{prefix}/cross_sync_pred_surp'] = cross_sync[0, 1].mean().item()
        metrics[f'{prefix}/cross_sync_surp_pred'] = cross_sync[1, 0].mean().item()
        metrics[f'{prefix}/cross_sync_surp_surp'] = cross_sync[1, 1].mean().item()

        # Module contributions
        contrib = sync.module_contributions  # (B, S, 2)
        metrics[f'{prefix}/contrib_prediction'] = contrib[..., 0].mean().item()
        metrics[f'{prefix}/contrib_surprise'] = contrib[..., 1].mean().item()

        # Attention metrics
        attn = output.attention_weights  # (B, H, S, S)
        attn_entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1).mean()
        metrics[f'{prefix}/attention_entropy'] = attn_entropy.item()

        # Attention focus (how peaked is attention?)
        attn_max = attn.max(dim=-1).values.mean()
        metrics[f'{prefix}/attention_max'] = attn_max.item()

        # Store per-step loss
        metrics[f'{prefix}/step_loss'] = step_loss
        step_losses.append(step_loss)
        step_certainties.append(combined_certainty)

    # Find best steps
    if len(outputs) > 0:
        best_loss_step = int(np.argmin(step_losses))
        best_certainty_step = int(np.argmax(step_certainties))

        metrics['best_loss_step'] = best_loss_step
        metrics['best_loss_value'] = step_losses[best_loss_step]
        metrics['best_certainty_step'] = best_certainty_step
        metrics['best_certainty_value'] = step_certainties[best_certainty_step]

    # Cross-step metrics
    if len(outputs) > 1:
        # Surprise change across steps
        first_surp = outputs[0].surprise.magnitude.mean()
        last_surp = outputs[-1].surprise.magnitude.mean()
        metrics['surprise_change'] = (last_surp - first_surp).item()

        # Attention entropy change
        first_entropy = -(outputs[0].attention_weights * (outputs[0].attention_weights + 1e-8).log()).sum(dim=-1).mean()
        last_entropy = -(outputs[-1].attention_weights * (outputs[-1].attention_weights + 1e-8).log()).sum(dim=-1).mean()
        metrics['attention_entropy_change'] = (last_entropy - first_entropy).item()

        # Certainty change
        metrics['certainty_change'] = step_certainties[-1] - step_certainties[0]

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

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    # Optimizer step
    optimizer.step()

    # Compute detailed metrics
    with torch.no_grad():
        metrics = compute_detailed_metrics(outputs, targets)

    # Add loss and grad norm
    metrics['loss'] = loss.item()
    metrics['grad_norm'] = grad_norm.item()

    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            metrics[f'loss/{k}'] = v.item()

    # Cumulative sync stats
    metrics['cumulative_sync_mean'] = final_state.cumulative_sync.mean().item()
    metrics['cumulative_sync_std'] = final_state.cumulative_sync.std().item()

    if return_outputs:
        return metrics, outputs
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

    return metrics


def create_dataloader(config: TrainingConfig, split: str = 'train'):
    """Create dataloader from HuggingFace dataset."""
    from datasets import load_dataset

    # Load FineWeb-Edu sample
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
    )

    if split == 'val':
        dataset = dataset.skip(10000).take(1000)

    def collate_fn(batch):
        texts = [item['text'][:config.max_length * 4] for item in batch]  # Rough char estimate
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

    # Training args
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_loop_steps', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=512)

    # Logging args
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=100)
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
    )
    model = PEMLoopGlobal(pem_config).to(device)
    print(f"PEM Loop created: {sum(p.numel() for p in model.parameters()):,} params")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Create dataloader
    print("Creating dataloader...")
    train_loader = create_dataloader(config, split='train')
    train_iter = iter(train_loader)

    # Training loop
    print(f"\nStarting training for {config.max_steps} steps...")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Loop steps: {config.num_loop_steps}")
    print(f"  Pred neurons: {config.pred_d_neurons}, T={config.pred_T}")
    print(f"  Surp neurons: {config.surp_d_neurons}, T={config.surp_T}")
    print(f"  Sync pairs: {config.sync_pairs}")
    print()

    global_step = 0
    running_loss = 0.0
    start_time = time.time()

    while global_step < config.max_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
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

        # Training step (return outputs periodically for visualization)
        should_visualize = config.wandb_project and (global_step + 1) % (config.log_every * 10) == 0
        result = train_step(model, features, optimizer, config, return_outputs=should_visualize)

        if should_visualize:
            metrics, outputs_for_viz = result
        else:
            metrics = result
            outputs_for_viz = None

        scheduler.step()

        running_loss += metrics['loss']
        global_step += 1

        # Logging
        if global_step % config.log_every == 0:
            avg_loss = running_loss / config.log_every
            elapsed = time.time() - start_time
            steps_per_sec = global_step / elapsed

            # Console output
            best_loss_step = int(metrics.get('best_loss_step', 0))
            best_cert_step = int(metrics.get('best_certainty_step', 0))
            best_cert_val = metrics.get('best_certainty_value', 0)

            print(f"Step {global_step:5d} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Surp: {metrics['step0/surprise_magnitude_mean']:.3f}→{metrics.get('step1/surprise_magnitude_mean', metrics['step0/surprise_magnitude_mean']):.3f} | "
                  f"CrossSync: {metrics['step0/cross_sync_pred_surp']:.3f} | "
                  f"Best: L{best_loss_step} C{best_cert_step}({best_cert_val:.2f}) | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"{steps_per_sec:.2f} steps/s")

            # WandB logging
            if config.wandb_project:
                log_dict = {
                    'train/loss': avg_loss,
                    'train/learning_rate': scheduler.get_last_lr()[0],
                    'train/steps_per_sec': steps_per_sec,
                }
                log_dict.update({f'train/{k}': v for k, v in metrics.items()})
                wandb.log(log_dict, step=global_step)

                # NLM activation visualization (less frequent)
                if outputs_for_viz is not None:
                    try:
                        # Create NLM activation heatmap grid
                        fig_nlm = create_nlm_activation_grid(outputs_for_viz)
                        wandb.log({
                            "visualizations/nlm_activations_heatmap": wandb.Image(fig_to_image(fig_nlm))
                        }, step=global_step)

                        # Create NLM neuron line plots
                        fig_lines = create_nlm_neuron_lines(outputs_for_viz)
                        wandb.log({
                            "visualizations/nlm_neuron_lines": wandb.Image(fig_to_image(fig_lines))
                        }, step=global_step)

                        # Create cross-module sync plot
                        fig_sync = create_cross_module_sync_plot(outputs_for_viz)
                        wandb.log({
                            "visualizations/cross_module_sync": wandb.Image(fig_to_image(fig_sync))
                        }, step=global_step)

                        print(f"  [Viz] Logged NLM heatmap, neuron lines, and cross-module sync plots")
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

    print("\nTraining complete!")

    if config.wandb_project:
        wandb.finish()


if __name__ == '__main__':
    main()
