#!/usr/bin/env python3
"""
Ablation tests for PEM Global Sync improvements.

Tests the effectiveness of:
1. Auxiliary prediction loss
2. Multi-tick world state injection
3. Different surprise signal types
4. Oscillator configurations

Usage:
    python -m pem.run_ablation_tests --device cuda --steps 200
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np

from .pem_loop_global import PEMLoopGlobal, PEMLoopGlobalConfig


@dataclass
class ExperimentConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str
    # Overrides for PEMLoopGlobalConfig
    overrides: Dict


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    name: str
    config: Dict
    final_loss: float
    final_certainty: float
    loss_improvement: float  # Improvement across loop steps
    world_state_benefit: float  # Benefit from world state (ablation)
    oscillator_active_frac: float
    oscillator_amp_mod: float
    training_time: float
    metrics_history: List[Dict]


def create_synthetic_data(
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: torch.device,
    with_structure: bool = True,
) -> torch.Tensor:
    """
    Create synthetic data for testing.

    If with_structure=True, creates data with temporal patterns that
    should benefit from the oscillatory world model.
    """
    if with_structure:
        # Create data with repeating patterns at different frequencies
        t = torch.arange(seq_len, device=device).float()

        # Multiple frequency components
        features = torch.zeros(batch_size, seq_len, d_model, device=device)

        # Fast patterns (every 8 tokens)
        fast = torch.sin(2 * np.pi * t / 8).unsqueeze(0).unsqueeze(-1)
        features[..., :d_model//4] = fast.expand(batch_size, -1, d_model//4)

        # Medium patterns (every 64 tokens)
        medium = torch.sin(2 * np.pi * t / 64).unsqueeze(0).unsqueeze(-1)
        features[..., d_model//4:d_model//2] = medium.expand(batch_size, -1, d_model//4)

        # Slow patterns (every 256 tokens)
        slow = torch.sin(2 * np.pi * t / 256).unsqueeze(0).unsqueeze(-1)
        features[..., d_model//2:3*d_model//4] = slow.expand(batch_size, -1, d_model//4)

        # Random component
        features[..., 3*d_model//4:] = torch.randn(batch_size, seq_len, d_model//4, device=device) * 0.1

        # Add some noise
        features = features + torch.randn_like(features) * 0.05
    else:
        # Pure random data (should not benefit from structure)
        features = torch.randn(batch_size, seq_len, d_model, device=device)

    return features


def run_single_experiment(
    config: PEMLoopGlobalConfig,
    device: torch.device,
    num_steps: int = 100,
    batch_size: int = 4,
    seq_len: int = 128,
) -> Tuple[List[Dict], float]:
    """Run a single training experiment and return metrics history."""

    model = PEMLoopGlobal(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    metrics_history = []
    start_time = time.time()

    for step in range(num_steps):
        # Generate synthetic data
        features = create_synthetic_data(
            batch_size, seq_len, config.d_model, device, with_structure=True
        )

        optimizer.zero_grad()

        # Compute targets
        targets = model.target_computer.compute_targets_efficient(features)

        # Forward pass
        outputs, final_state = model(features, targets, num_steps=2)

        # Compute loss
        loss, loss_dict = model.compute_loss(outputs, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Collect metrics
        with torch.no_grad():
            final_output = outputs[-1]

            # Compute certainty
            certainty = final_output.prediction_output.certainty.item()

            # Compute loop improvement
            first_loss = 0.0
            last_loss = 0.0
            for scale in ['immediate', 'shortterm', 'longterm']:
                target = targets[scale]
                valid = targets.get(f'{scale}_valid', None)
                if valid is not None and valid.any():
                    first_cos = F.cosine_similarity(
                        outputs[0].predictions[scale][valid],
                        target[valid], dim=-1
                    ).mean()
                    last_cos = F.cosine_similarity(
                        outputs[-1].predictions[scale][valid],
                        target[valid], dim=-1
                    ).mean()
                    first_loss += (1 - first_cos).item()
                    last_loss += (1 - last_cos).item()

            loop_improvement = first_loss - last_loss

            # Get oscillator metrics
            osc_metrics = final_output.global_sync.oscillator_metrics
            if osc_metrics is not None:
                osc_active = osc_metrics.active_oscillator_frac
                osc_amp_mod = osc_metrics.amp_mod_mean
            else:
                osc_active = 0.0
                osc_amp_mod = 0.0

            metrics = {
                'step': step,
                'loss': loss.item(),
                'certainty': certainty,
                'loop_improvement': loop_improvement,
                'osc_active_frac': osc_active,
                'osc_amp_mod': osc_amp_mod,
            }

            # Add loss components
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    metrics[f'loss_{k}'] = v.item()
                elif isinstance(v, (int, float)):
                    metrics[f'loss_{k}'] = v

            metrics_history.append(metrics)

        if (step + 1) % 50 == 0:
            print(f"    Step {step+1}/{num_steps}: loss={loss.item():.4f}, "
                  f"certainty={certainty:.3f}, loop_impr={loop_improvement:+.4f}")

    training_time = time.time() - start_time

    # Compute world state benefit via quick ablation
    # Run one batch with world state disabled
    model.eval()
    with torch.no_grad():
        features = create_synthetic_data(
            batch_size, seq_len, config.d_model, device, with_structure=True
        )
        targets = model.target_computer.compute_targets_efficient(features)

        # With world state
        outputs_with, _ = model(features, targets, num_steps=2)
        loss_with = 0.0
        for scale in ['immediate']:
            target = targets[scale]
            valid = targets.get(f'{scale}_valid', None)
            if valid is not None and valid.any():
                cos = F.cosine_similarity(
                    outputs_with[-1].predictions[scale][valid],
                    target[valid], dim=-1
                ).mean()
                loss_with += (1 - cos).item()

        # Without world state (disable temporarily)
        original_use_world = model.prediction.core.world_to_z0 is not None
        if original_use_world:
            # Temporarily disable world state influence
            model.prediction.core.world_to_z0 = None
            outputs_without, _ = model(features, targets, num_steps=2)

            loss_without = 0.0
            for scale in ['immediate']:
                target = targets[scale]
                valid = targets.get(f'{scale}_valid', None)
                if valid is not None and valid.any():
                    cos = F.cosine_similarity(
                        outputs_without[-1].predictions[scale][valid],
                        target[valid], dim=-1
                    ).mean()
                    loss_without += (1 - cos).item()

            world_state_benefit = (loss_without - loss_with) / (loss_without + 1e-8) * 100
        else:
            world_state_benefit = 0.0

    return metrics_history, training_time, world_state_benefit


def run_ablation_experiments(
    device: torch.device,
    num_steps: int = 100,
    d_model: int = 256,
) -> List[ExperimentResult]:
    """Run all ablation experiments."""

    # Base configuration
    base_config = {
        'd_model': d_model,
        'pred_d_neurons': 64,
        'surp_d_neurons': 32,
        'pred_T': 3,
        'surp_T': 2,
        'sync_pairs': 64,
        'use_oscillatory_world': True,
        'num_oscillators': 32,
        'min_period': 4,
        'max_period': 256,
        'd_world_output': 64,
    }

    # Define experiments
    experiments = [
        ExperimentConfig(
            name="baseline",
            description="Baseline: aux_pred=True, multi_tick=False, surprise=ctm",
            overrides={
                'use_auxiliary_prediction': True,
                'auxiliary_prediction_weight': 0.1,
                'multi_tick_world_injection': False,
                'surprise_signal_type': 'ctm',
            }
        ),
        ExperimentConfig(
            name="no_aux_pred",
            description="No auxiliary prediction loss",
            overrides={
                'use_auxiliary_prediction': False,
                'multi_tick_world_injection': False,
                'surprise_signal_type': 'ctm',
            }
        ),
        ExperimentConfig(
            name="aux_pred_high_weight",
            description="Auxiliary prediction with higher weight (0.3)",
            overrides={
                'use_auxiliary_prediction': True,
                'auxiliary_prediction_weight': 0.3,
                'multi_tick_world_injection': False,
                'surprise_signal_type': 'ctm',
            }
        ),
        ExperimentConfig(
            name="multi_tick_injection",
            description="Multi-tick world state injection",
            overrides={
                'use_auxiliary_prediction': True,
                'auxiliary_prediction_weight': 0.1,
                'multi_tick_world_injection': True,
                'surprise_signal_type': 'ctm',
            }
        ),
        ExperimentConfig(
            name="surprise_pred_error",
            description="Surprise signal: prediction error",
            overrides={
                'use_auxiliary_prediction': True,
                'auxiliary_prediction_weight': 0.1,
                'multi_tick_world_injection': False,
                'surprise_signal_type': 'prediction_error',
            }
        ),
        ExperimentConfig(
            name="surprise_attn_entropy",
            description="Surprise signal: attention entropy",
            overrides={
                'use_auxiliary_prediction': True,
                'auxiliary_prediction_weight': 0.1,
                'multi_tick_world_injection': False,
                'surprise_signal_type': 'attention_entropy',
            }
        ),
        ExperimentConfig(
            name="combined_best",
            description="Combined: aux_pred + multi_tick + pred_error",
            overrides={
                'use_auxiliary_prediction': True,
                'auxiliary_prediction_weight': 0.2,
                'multi_tick_world_injection': True,
                'surprise_signal_type': 'prediction_error',
            }
        ),
        ExperimentConfig(
            name="no_world_model",
            description="Ablation: No oscillatory world model",
            overrides={
                'use_oscillatory_world': False,
                'use_auxiliary_prediction': False,
                'multi_tick_world_injection': False,
                'surprise_signal_type': 'ctm',
            }
        ),
    ]

    results = []

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {exp.name}")
        print(f"  {exp.description}")
        print('='*60)

        # Create config with overrides
        config_dict = {**base_config, **exp.overrides}
        config = PEMLoopGlobalConfig(**config_dict)

        # Run experiment
        metrics_history, training_time, world_state_benefit = run_single_experiment(
            config=config,
            device=device,
            num_steps=num_steps,
            batch_size=4,
            seq_len=128,
        )

        # Compute final metrics (average of last 10 steps)
        final_metrics = {}
        for key in metrics_history[0].keys():
            if key != 'step':
                values = [m[key] for m in metrics_history[-10:]]
                final_metrics[key] = sum(values) / len(values)

        result = ExperimentResult(
            name=exp.name,
            config=config_dict,
            final_loss=final_metrics['loss'],
            final_certainty=final_metrics['certainty'],
            loss_improvement=final_metrics['loop_improvement'],
            world_state_benefit=world_state_benefit,
            oscillator_active_frac=final_metrics.get('osc_active_frac', 0),
            oscillator_amp_mod=final_metrics.get('osc_amp_mod', 0),
            training_time=training_time,
            metrics_history=metrics_history,
        )

        results.append(result)

        print(f"\n  Results for {exp.name}:")
        print(f"    Final loss:           {result.final_loss:.4f}")
        print(f"    Final certainty:      {result.final_certainty:.3f}")
        print(f"    Loop improvement:     {result.loss_improvement:+.4f}")
        print(f"    World state benefit:  {result.world_state_benefit:+.1f}%")
        print(f"    Oscillator active:    {result.oscillator_active_frac:.2f}")
        print(f"    Oscillator amp mod:   {result.oscillator_amp_mod:.4f}")
        print(f"    Training time:        {result.training_time:.1f}s")

    return results


def print_comparison_table(results: List[ExperimentResult]):
    """Print a comparison table of all results."""
    print("\n" + "="*100)
    print("ABLATION RESULTS COMPARISON")
    print("="*100)

    # Header
    print(f"{'Experiment':<25} {'Loss':>10} {'Certainty':>10} {'Loop Impr':>10} "
          f"{'WS Benefit':>10} {'Osc Active':>10} {'Time':>8}")
    print("-"*100)

    # Sort by loss (ascending)
    sorted_results = sorted(results, key=lambda r: r.final_loss)

    for r in sorted_results:
        print(f"{r.name:<25} {r.final_loss:>10.4f} {r.final_certainty:>10.3f} "
              f"{r.loss_improvement:>+10.4f} {r.world_state_benefit:>+9.1f}% "
              f"{r.oscillator_active_frac:>10.2f} {r.training_time:>7.1f}s")

    print("-"*100)

    # Find best for each metric
    best_loss = min(results, key=lambda r: r.final_loss)
    best_cert = max(results, key=lambda r: r.final_certainty)
    best_impr = max(results, key=lambda r: r.loss_improvement)
    best_ws = max(results, key=lambda r: r.world_state_benefit)

    print(f"\nBest configurations:")
    print(f"  Lowest loss:            {best_loss.name} ({best_loss.final_loss:.4f})")
    print(f"  Highest certainty:      {best_cert.name} ({best_cert.final_certainty:.3f})")
    print(f"  Best loop improvement:  {best_impr.name} ({best_impr.loss_improvement:+.4f})")
    print(f"  Best world state ben:   {best_ws.name} ({best_ws.world_state_benefit:+.1f}%)")

    # Compare to baseline
    baseline = next((r for r in results if r.name == "baseline"), None)
    no_world = next((r for r in results if r.name == "no_world_model"), None)

    if baseline and no_world:
        ws_contribution = (no_world.final_loss - baseline.final_loss) / no_world.final_loss * 100
        print(f"\n  World model contribution: {ws_contribution:+.1f}% loss reduction vs no world model")


def main():
    parser = argparse.ArgumentParser(description="Run PEM ablation tests")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--steps', type=int, default=200, help='Training steps per experiment')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--output', type=str, default=None, help='Save results to JSON file')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Running ablation tests on {device}")
    print(f"Steps per experiment: {args.steps}")
    print(f"Model dimension: {args.d_model}")

    # Run experiments
    results = run_ablation_experiments(
        device=device,
        num_steps=args.steps,
        d_model=args.d_model,
    )

    # Print comparison
    print_comparison_table(results)

    # Save results if requested
    if args.output:
        output_data = []
        for r in results:
            output_data.append({
                'name': r.name,
                'config': r.config,
                'final_loss': r.final_loss,
                'final_certainty': r.final_certainty,
                'loss_improvement': r.loss_improvement,
                'world_state_benefit': r.world_state_benefit,
                'oscillator_active_frac': r.oscillator_active_frac,
                'oscillator_amp_mod': r.oscillator_amp_mod,
                'training_time': r.training_time,
            })

        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
