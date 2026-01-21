#!/usr/bin/env python3
"""
Diagnostic script to analyze PEM loop degradation.

Investigates why later loop steps perform worse than earlier steps.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from .pem_loop_global import PEMLoopGlobal, PEMLoopGlobalConfig


@dataclass
class LoopDiagnostics:
    """Diagnostics for loop iteration behavior."""
    step_losses: List[float]
    step_certainties: List[float]
    gate_values: List[float]  # State combiner gate values
    observation_changes: List[float]  # How much observation changes between steps
    world_state_norms: List[float]  # World state magnitude at each step
    loop_features_diff: List[float]  # Difference between loop_features and raw features
    prediction_similarities: List[float]  # Similarity between consecutive predictions
    loop_features_target_alignment: List[float] = None  # How well loop_features align with targets


def diagnose_loop(
    model: PEMLoopGlobal,
    features: torch.Tensor,
    num_steps: int = 3,
) -> LoopDiagnostics:
    """
    Run loop with detailed diagnostics.

    Args:
        model: PEM loop model
        features: (B, S, d_model) input features
        num_steps: Number of loop iterations

    Returns:
        LoopDiagnostics with detailed metrics
    """
    model.eval()
    device = features.device

    with torch.no_grad():
        targets = model.target_computer.compute_targets_efficient(features)
        state = model.init_state(features)

        step_losses = []
        step_certainties = []
        gate_values = []
        observation_changes = []
        world_state_norms = []
        loop_features_diffs = []
        prediction_similarities = []
        loop_features_target_alignment = []  # NEW: how well loop_features align with targets

        prev_observation = state.observation.clone()
        prev_predictions = None

        for step in range(num_steps):
            # Manually compute state combiner to get gate values
            combined = torch.cat([features, state.observation], dim=-1)
            gate = model.state_combiner_gate(combined)
            gate_mean = gate.mean().item()
            gate_values.append(gate_mean)

            # Compute loop_features manually
            transformed = model.state_combiner_transform(combined)
            loop_features = features + gate * (transformed - features)

            # Measure difference from raw features
            feat_diff = (loop_features - features).abs().mean().item()
            loop_features_diffs.append(feat_diff)

            # NEW: Check if loop_features still align with targets
            # Targets are computed from raw features, so if loop_features diverge,
            # the prediction task becomes harder
            target = targets['immediate']
            valid = targets.get('immediate_valid', None)
            if valid is not None and valid.any():
                # How well do loop_features correlate with the target they need to predict?
                lf_target_sim = F.cosine_similarity(
                    loop_features[valid], target[valid], dim=-1
                ).mean().item()
            else:
                lf_target_sim = 0.0
            loop_features_target_alignment.append(lf_target_sim)

            # Run the step
            output, state = model.step(features, targets, state)

            # Compute loss at this step
            pred = output.predictions['immediate']
            if valid is not None and valid.any():
                cos_sim = F.cosine_similarity(pred[valid], target[valid], dim=-1)
                step_loss = (1 - cos_sim).mean().item()
            else:
                step_loss = 0.0
            step_losses.append(step_loss)

            # Certainty
            cert = output.prediction_output.certainty.mean().item()
            step_certainties.append(cert)

            # Observation change
            obs_change = (state.observation - prev_observation).abs().mean().item()
            observation_changes.append(obs_change)
            prev_observation = state.observation.clone()

            # World state norm
            if state.world_state is not None:
                ws_norm = state.world_state.norm().item()
            else:
                ws_norm = 0.0
            world_state_norms.append(ws_norm)

            # Prediction similarity to previous
            if prev_predictions is not None:
                pred_sim = F.cosine_similarity(
                    pred.view(-1, pred.shape[-1]),
                    prev_predictions.view(-1, prev_predictions.shape[-1]),
                    dim=-1
                ).mean().item()
            else:
                pred_sim = 1.0
            prediction_similarities.append(pred_sim)
            prev_predictions = pred.clone()

    return LoopDiagnostics(
        step_losses=step_losses,
        step_certainties=step_certainties,
        gate_values=gate_values,
        observation_changes=observation_changes,
        world_state_norms=world_state_norms,
        loop_features_diff=loop_features_diffs,
        prediction_similarities=prediction_similarities,
        loop_features_target_alignment=loop_features_target_alignment,
    )


def print_diagnostics(diag: LoopDiagnostics):
    """Print formatted diagnostics."""
    print("\n" + "="*70)
    print("LOOP DIAGNOSTICS")
    print("="*70)

    print("\n1. LOSS PER STEP:")
    for i, loss in enumerate(diag.step_losses):
        delta = ""
        if i > 0:
            diff = loss - diag.step_losses[i-1]
            delta = f" (Δ={diff:+.4f})"
        print(f"   Step {i}: {loss:.4f}{delta}")

    print("\n2. STATE COMBINER GATE VALUES:")
    for i, gate in enumerate(diag.gate_values):
        print(f"   Step {i}: {gate:.4f}")
    print(f"   → Gate ≈ {np.mean(diag.gate_values):.3f} means {np.mean(diag.gate_values)*100:.1f}% "
          f"observation influence")

    print("\n3. LOOP FEATURES DIFF FROM RAW:")
    for i, diff in enumerate(diag.loop_features_diff):
        print(f"   Step {i}: {diff:.6f}")
    if np.mean(diag.loop_features_diff) < 0.01:
        print("   ⚠️  Loop features very close to raw features - observation not influencing!")

    print("\n4. OBSERVATION CHANGES:")
    for i, change in enumerate(diag.observation_changes):
        print(f"   Step {i}: {change:.4f}")

    print("\n5. WORLD STATE NORMS:")
    for i, norm in enumerate(diag.world_state_norms):
        print(f"   Step {i}: {norm:.4f}")

    print("\n6. PREDICTION SIMILARITY (to previous step):")
    for i, sim in enumerate(diag.prediction_similarities):
        print(f"   Step {i}: {sim:.4f}")
    if np.mean(diag.prediction_similarities[1:]) > 0.99:
        print("   ⚠️  Predictions almost identical between steps - loop not refining!")

    print("\n7. CERTAINTY PER STEP:")
    for i, cert in enumerate(diag.step_certainties):
        print(f"   Step {i}: {cert:.4f}")

    if diag.loop_features_target_alignment:
        print("\n8. LOOP_FEATURES → TARGET ALIGNMENT:")
        for i, align in enumerate(diag.loop_features_target_alignment):
            delta = ""
            if i > 0:
                diff = align - diag.loop_features_target_alignment[i-1]
                delta = f" (Δ={diff:+.4f})"
            print(f"   Step {i}: {align:.4f}{delta}")
        if len(diag.loop_features_target_alignment) > 1:
            if diag.loop_features_target_alignment[-1] < diag.loop_features_target_alignment[0]:
                print("   ⚠️  Loop features DIVERGE from targets - this hurts prediction!")

    # Summary
    print("\n" + "-"*70)
    print("DIAGNOSIS:")

    issues = []

    # Check if loss increases
    if len(diag.step_losses) > 1:
        loss_increases = sum(1 for i in range(1, len(diag.step_losses))
                           if diag.step_losses[i] > diag.step_losses[i-1])
        if loss_increases > 0:
            issues.append(f"Loss INCREASES in {loss_increases}/{len(diag.step_losses)-1} steps")

    # Check gate values
    avg_gate = np.mean(diag.gate_values)
    if avg_gate < 0.15:
        issues.append(f"Gate too small ({avg_gate:.3f}) - observation has little influence")

    # Check loop features diff
    avg_diff = np.mean(diag.loop_features_diff)
    if avg_diff < 0.01:
        issues.append(f"Loop features ≈ raw features (diff={avg_diff:.6f}) - state combiner not working")

    # Check prediction similarity
    if len(diag.prediction_similarities) > 1:
        avg_sim = np.mean(diag.prediction_similarities[1:])
        if avg_sim > 0.995:
            issues.append(f"Predictions too similar ({avg_sim:.4f}) - loop not refining")

    # Check certainty
    if all(c < 0.55 for c in diag.step_certainties):
        issues.append("Certainty stuck near 0.5 - model not learning confidence")

    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   • {issue}")

        print("\n💡 SUGGESTED FIXES:")
        if avg_gate < 0.15:
            print("   • Increase state_combiner_gate bias initialization (currently -2.0)")
            print("     Try: self.state_combiner_gate[0].bias.data.fill_(-1.0)  # sigmoid(-1) ≈ 0.27")
        if avg_diff < 0.01:
            print("   • State combiner not incorporating observation - check if gate is learned")
        if loss_increases > 0:
            print("   • Increase loop_improvement_weight (currently 0.5) to penalize regression more")
            print("   • Or decrease observation_residual (currently 0.3) to allow more change")
    else:
        print("\n✓ No obvious issues found")

    print("="*70)


def run_diagnostic(
    config: PEMLoopGlobalConfig = None,
    device: str = 'cuda',
    batch_size: int = 2,
    seq_len: int = 64,
    num_steps: int = 3,
):
    """Run full diagnostic on a fresh model."""
    if config is None:
        config = PEMLoopGlobalConfig(
            d_model=256,
            pred_d_neurons=64,
            surp_d_neurons=32,
        )

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = PEMLoopGlobal(config).to(device)

    # Create synthetic data with structure
    features = torch.randn(batch_size, seq_len, config.d_model, device=device)

    diag = diagnose_loop(model, features, num_steps)
    print_diagnostics(diag)

    return diag


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_steps', type=int, default=3)
    args = parser.parse_args()

    run_diagnostic(device=args.device, num_steps=args.num_steps)
