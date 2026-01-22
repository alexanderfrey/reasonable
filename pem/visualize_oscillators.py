"""
Visualize oscillator states and dynamics.

Usage:
    python -m pem.visualize_oscillators --checkpoint path/to/model.pt
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List, Dict
import math


def visualize_oscillator_state(
    model,
    features: torch.Tensor,
    targets: dict,
    save_path: Optional[str] = None,
    title: str = "Oscillator State",
) -> plt.Figure:
    """
    Create a comprehensive visualization of oscillator states.

    Shows:
    1. Oscillator values as heatmap (8x8 grid)
    2. Frequency spectrum (amplitude by period)
    3. Phase distribution
    4. Memory write attention
    """
    model.eval()

    with torch.no_grad():
        outputs, _ = model(features, targets, num_steps=2)

    # Get oscillator state from the model
    osc_world = model.global_sync.oscillatory_world
    osc_state = osc_world.get_oscillator_state()

    amplitudes = osc_state['current_amplitudes'].detach().cpu().numpy()
    phases = osc_state['phases'].detach().cpu().numpy()
    frequencies = osc_world.frequencies.detach().cpu().numpy()
    periods = 1.0 / (frequencies + 1e-8)

    # Compute oscillator values
    osc_values = amplitudes * np.sin(phases)

    # Get world state stats
    world_stats = model.global_sync.get_world_state_stats()

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Oscillator values as 8x8 heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    osc_grid = osc_values.reshape(8, 8)
    im1 = ax1.imshow(osc_grid, cmap='RdBu', vmin=-1, vmax=1, aspect='equal')
    ax1.set_title('Oscillator States (8x8)', fontsize=12)
    ax1.set_xlabel('Oscillator Index (mod 8)')
    ax1.set_ylabel('Oscillator Index (div 8)')
    plt.colorbar(im1, ax=ax1, label='Value')

    # Add grid lines
    for i in range(9):
        ax1.axhline(i - 0.5, color='black', linewidth=0.5)
        ax1.axvline(i - 0.5, color='black', linewidth=0.5)

    # 2. Amplitude by period (frequency spectrum)
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_idx = np.argsort(periods)
    ax2.bar(range(len(periods)), amplitudes[sorted_idx], color='steelblue', alpha=0.7)
    ax2.set_xlabel('Oscillator (sorted by period)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Amplitude Spectrum', fontsize=12)

    # Color by frequency band
    n_osc = len(periods)
    colors = ['red'] * (n_osc // 3) + ['green'] * (n_osc // 3) + ['blue'] * (n_osc - 2 * (n_osc // 3))
    for i, (amp, c) in enumerate(zip(amplitudes[sorted_idx], colors)):
        ax2.bar(i, amp, color=c, alpha=0.7)
    ax2.legend(['Fast', 'Medium', 'Slow'], loc='upper right')

    # 3. Phase distribution (polar plot)
    ax3 = fig.add_subplot(gs[0, 2], projection='polar')
    ax3.scatter(phases, amplitudes, c=np.log10(periods), cmap='viridis', alpha=0.7, s=50)
    ax3.set_title('Phase Distribution', fontsize=12)
    ax3.set_rlabel_position(45)

    # 4. Oscillator values over frequency (line plot)
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.plot(periods[sorted_idx], osc_values[sorted_idx], 'o-', markersize=4, alpha=0.7)
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xscale('log')
    ax4.set_xlabel('Period (steps)')
    ax4.set_ylabel('Oscillator Value (amp * sin(phase))')
    ax4.set_title('Memory State by Timescale', fontsize=12)
    ax4.grid(True, alpha=0.3)

    # Add frequency band annotations
    ax4.axvspan(periods.min(), periods[sorted_idx[n_osc//3]], alpha=0.1, color='red', label='Fast')
    ax4.axvspan(periods[sorted_idx[n_osc//3]], periods[sorted_idx[2*n_osc//3]], alpha=0.1, color='green', label='Medium')
    ax4.axvspan(periods[sorted_idx[2*n_osc//3]], periods.max(), alpha=0.1, color='blue', label='Slow')

    # 5. Amplitude modulation histogram
    ax5 = fig.add_subplot(gs[1, 2])
    amp_mod = osc_world._last_amp_mod.detach().cpu().numpy()
    ax5.hist(amp_mod, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax5.axvline(0, color='red', linestyle='--', label='No modulation')
    ax5.set_xlabel('Amplitude Modulation')
    ax5.set_ylabel('Count')
    ax5.set_title('Amp Modulation Distribution', fontsize=12)
    ax5.legend()

    # 6. Key metrics text box
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.axis('off')

    metrics_text = f"""
    Oscillator Metrics:
    ─────────────────────
    Amplitude mean: {amplitudes.mean():.3f}
    Amplitude std:  {amplitudes.std():.3f}
    Phase entropy:  {world_stats.get('oscillator/phase_entropy', 0):.2f}
    Active frac:    {world_stats.get('oscillator/active_frac', 0):.2f}

    Frequency Bands:
    ─────────────────────
    Slow amp:  {world_stats.get('osc/slow_amp_mean', 0):.3f}
    Mid amp:   {world_stats.get('osc/mid_amp_mean', 0):.3f}
    Fast amp:  {world_stats.get('osc/fast_amp_mean', 0):.3f}

    Write Gate:
    ─────────────────────
    Gate mean: {world_stats.get('surprise/write_gate_mean', 0):.3f}
    """
    ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='top')

    # 7. Oscillator correlation matrix (which oscillators co-activate?)
    ax7 = fig.add_subplot(gs[2, 1:])

    # Show period labels on x-axis
    period_labels = [f'{int(p)}' if p < 1000 else f'{p/1000:.1f}k' for p in periods[sorted_idx]]
    ax7.bar(range(len(periods)), osc_values[sorted_idx], color='steelblue', alpha=0.7)
    ax7.set_xticks(range(0, len(periods), 8))
    ax7.set_xticklabels([period_labels[i] for i in range(0, len(periods), 8)], rotation=45)
    ax7.set_xlabel('Period (steps)')
    ax7.set_ylabel('Memory Value')
    ax7.set_title('Current Memory State by Frequency', fontsize=12)
    ax7.axhline(0, color='gray', linestyle='--', alpha=0.5)

    fig.suptitle(title, fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def visualize_oscillator_evolution(
    model,
    features: torch.Tensor,
    targets: dict,
    num_steps: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize how oscillator states evolve over multiple loop steps.

    Shows a heatmap of oscillator values over time.
    """
    model.eval()

    osc_world = model.global_sync.oscillatory_world
    frequencies = osc_world.frequencies.detach().cpu().numpy()
    periods = 1.0 / (frequencies + 1e-8)
    sorted_idx = np.argsort(periods)

    # Collect oscillator states over steps
    states_over_time = []

    # Reset to get clean evolution
    osc_world.reset_phases(random=True)

    with torch.no_grad():
        state = model.init_state(features)

        for step in range(num_steps):
            output, state = model.step(features, targets, state)

            osc_state = osc_world.get_oscillator_state()
            amplitudes = osc_state['current_amplitudes'].detach().cpu().numpy()
            phases = osc_state['phases'].detach().cpu().numpy()
            osc_values = amplitudes * np.sin(phases)
            states_over_time.append(osc_values[sorted_idx])

    states_matrix = np.array(states_over_time)  # (num_steps, 64)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 1. Heatmap of oscillator evolution
    ax1 = axes[0]
    im = ax1.imshow(states_matrix.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    ax1.set_xlabel('Loop Step')
    ax1.set_ylabel('Oscillator (sorted by period)')
    ax1.set_title('Oscillator States Over Time', fontsize=12)
    plt.colorbar(im, ax=ax1, label='Value')

    # Add frequency band separators
    n_osc = states_matrix.shape[1]
    ax1.axhline(n_osc // 3 - 0.5, color='white', linewidth=2, linestyle='--')
    ax1.axhline(2 * n_osc // 3 - 0.5, color='white', linewidth=2, linestyle='--')

    # Labels for bands
    ax1.text(-0.5, n_osc // 6, 'Fast', ha='right', va='center', fontweight='bold', color='red')
    ax1.text(-0.5, n_osc // 2, 'Med', ha='right', va='center', fontweight='bold', color='green')
    ax1.text(-0.5, 5 * n_osc // 6, 'Slow', ha='right', va='center', fontweight='bold', color='blue')

    # 2. Line plot of selected oscillators
    ax2 = axes[1]
    # Pick representative oscillators from each band
    fast_idx = sorted_idx[5]
    med_idx = sorted_idx[n_osc // 2]
    slow_idx = sorted_idx[-5]

    steps = range(num_steps)
    ax2.plot(steps, [s[5] for s in states_over_time], 'r-', label=f'Fast (T={int(periods[fast_idx])})', linewidth=2)
    ax2.plot(steps, [s[n_osc // 2] for s in states_over_time], 'g-', label=f'Med (T={int(periods[med_idx])})', linewidth=2)
    ax2.plot(steps, [s[-5] for s in states_over_time], 'b-', label=f'Slow (T={int(periods[slow_idx])})', linewidth=2)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Loop Step')
    ax2.set_ylabel('Oscillator Value')
    ax2.set_title('Representative Oscillator Trajectories', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def visualize_cross_attention(
    model,
    features: torch.Tensor,
    targets: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize how different sequence positions attend to different oscillators.
    """
    model.eval()

    with torch.no_grad():
        outputs, _ = model(features, targets, num_steps=2)

    # Get attention weights from the last step
    # This requires the model to store attention weights
    gs = model.global_sync

    if not hasattr(gs, '_last_osc_attn_weights') or gs._last_osc_attn_weights is None:
        print("Note: Attention weights not stored. Add storage in global_sync.py")
        return None

    attn_weights = gs._last_osc_attn_weights.cpu().numpy()  # (B, S, 64)

    # Average over batch
    attn_weights = attn_weights.mean(axis=0)  # (S, 64)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Full attention heatmap
    ax1 = axes[0]
    im = ax1.imshow(attn_weights.T, aspect='auto', cmap='viridis')
    ax1.set_xlabel('Sequence Position')
    ax1.set_ylabel('Oscillator Index')
    ax1.set_title('Position → Oscillator Attention', fontsize=12)
    plt.colorbar(im, ax=ax1, label='Attention Weight')

    # 2. Attention entropy per position
    ax2 = axes[1]
    entropy = -np.sum(attn_weights * np.log(attn_weights + 1e-10), axis=1)
    ax2.plot(entropy, color='steelblue')
    ax2.axhline(np.log(64), color='red', linestyle='--', label=f'Max entropy ({np.log(64):.2f})')
    ax2.set_xlabel('Sequence Position')
    ax2.set_ylabel('Attention Entropy')
    ax2.set_title('Attention Selectivity by Position', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


if __name__ == "__main__":
    import argparse
    from .pem_loop_global import PEMLoopGlobal, PEMLoopGlobalConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='pem_analysis', help='Output directory')
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Create or load model
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        config = checkpoint.get('config', PEMLoopGlobalConfig())
        model = PEMLoopGlobal(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.checkpoint}")
    else:
        config = PEMLoopGlobalConfig()
        model = PEMLoopGlobal(config)
        print("Using randomly initialized model")

    # Create dummy features
    features = torch.randn(2, 512, config.d_model)
    targets = model.target_computer.compute_targets_efficient(features)

    # Generate visualizations
    print("Generating oscillator state visualization...")
    fig1 = visualize_oscillator_state(model, features, targets,
                                       save_path=f"{args.output_dir}/osc_state.png")

    print("Generating oscillator evolution visualization...")
    fig2 = visualize_oscillator_evolution(model, features, targets, num_steps=10,
                                           save_path=f"{args.output_dir}/osc_evolution.png")

    print("Done!")
    plt.show()
