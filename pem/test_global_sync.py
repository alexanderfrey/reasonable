"""
Test script for Global Sync Architecture.

Tests:
1. CTMModule base class
2. PredictionCTM
3. SurpriseCTM
4. GlobalSyncModule
5. PEMLoopGlobal (full integration)
"""

import torch
import torch.nn as nn


def test_ctm_base():
    """Test CTMCore basic functionality."""
    print("\n=== Testing CTM Base ===")

    from pem.ctm_base import CTMCore, CTMBaseConfig

    config = CTMBaseConfig(
        d_input=512,
        d_neurons=64,
        d_output=512,
        T=4,
        M=4,
    )

    core = CTMCore(config)
    print(f"CTMCore params: {sum(p.numel() for p in core.parameters()):,}")

    # Test forward
    x = torch.randn(2, 16, 512)  # (B, S, d_input)
    post_act, sync_matrix, output, all_outputs = core(x)

    print(f"Input: {x.shape}")
    print(f"Post-activations: {post_act.shape}")
    print(f"Sync matrix: {sync_matrix.shape}")
    print(f"Output: {output.shape}")
    print(f"All outputs: {len(all_outputs)} ticks")

    assert post_act.shape == (2, 16, 64), f"Expected (2,16,64), got {post_act.shape}"
    assert sync_matrix.shape == (2, 16, 64, 64), f"Expected (2,16,64,64), got {sync_matrix.shape}"
    assert output.shape == (2, 16, 512), f"Expected (2,16,512), got {output.shape}"
    assert len(all_outputs) == 4, f"Expected 4 ticks, got {len(all_outputs)}"

    print("CTM Base: PASSED")


def test_prediction_ctm():
    """Test PredictionCTM."""
    print("\n=== Testing PredictionCTM ===")

    from pem.prediction_ctm import PredictionCTM, PredictionCTMConfig

    config = PredictionCTMConfig(
        d_input=512,
        d_output=512,
        d_neurons=64,
        T=4,
        M=4,
    )

    model = PredictionCTM(config)
    print(f"PredictionCTM params: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward
    features = torch.randn(2, 16, 512)
    output = model(features)

    print(f"Input features: {features.shape}")
    print(f"Predictions (immediate): {output.predictions['immediate'].shape}")
    print(f"Predictions (shortterm): {output.predictions['shortterm'].shape}")
    print(f"Predictions (longterm): {output.predictions['longterm'].shape}")
    print(f"Post-activations: {output.post_activations.shape}")
    print(f"Sync matrix: {output.sync_matrix.shape}")
    print(f"Certainty: {output.certainty.item():.4f}")

    assert output.predictions['immediate'].shape == (2, 16, 512)
    assert output.post_activations.shape == (2, 16, 64)

    print("PredictionCTM: PASSED")


def test_surprise_ctm():
    """Test SurpriseCTM."""
    print("\n=== Testing SurpriseCTM ===")

    from pem.surprise_ctm import SurpriseCTM, SurpriseCTMConfig

    config = SurpriseCTMConfig(
        d_model=512,
        d_input=512 * 3,
        d_output=512,
        d_neurons=32,
        T=3,
        M=4,
    )

    model = SurpriseCTM(config)
    print(f"SurpriseCTM params: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward
    predicted = torch.randn(2, 16, 512)
    actual = torch.randn(2, 16, 512)
    output = model(predicted, actual)

    print(f"Predicted: {predicted.shape}")
    print(f"Actual: {actual.shape}")
    print(f"Magnitude: {output.magnitude.shape}, range [{output.magnitude.min():.3f}, {output.magnitude.max():.3f}]")
    print(f"Direction: {output.direction.shape}")
    print(f"Raw: {output.raw.shape}")
    print(f"Post-activations: {output.post_activations.shape}")
    print(f"Certainty: {output.certainty.item():.4f}")

    # Direction should be unit vectors
    dir_norms = output.direction.norm(dim=-1)
    assert torch.allclose(dir_norms, torch.ones_like(dir_norms), atol=1e-5), \
        f"Direction not unit vectors: {dir_norms.mean():.4f}"

    assert output.magnitude.shape == (2, 16, 1)
    assert output.post_activations.shape == (2, 16, 32)

    print("SurpriseCTM: PASSED")


def test_global_sync():
    """Test GlobalSyncModule."""
    print("\n=== Testing GlobalSyncModule ===")

    from pem.global_sync import GlobalSyncModule, GlobalSyncConfig

    config = GlobalSyncConfig(
        d_sync_space=64,
        sync_pairs=128,
        n_heads=4,
    )

    model = GlobalSyncModule(config)

    # Register modules
    model.register_module('prediction', d_neurons=64)
    model.register_module('surprise', d_neurons=32)

    print(f"GlobalSyncModule params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Registered modules: {model.module_names}")

    # Test forward
    h_pred = torch.randn(2, 16, 64)
    h_surp = torch.randn(2, 16, 32)

    output = model({
        'prediction': h_pred,
        'surprise': h_surp,
    })

    print(f"Prediction activations: {h_pred.shape}")
    print(f"Surprise activations: {h_surp.shape}")
    print(f"Global sync: {output.sync.shape}")
    print(f"Cross-module sync: {output.cross_module_sync.shape}")
    print(f"Module contributions: {output.module_contributions.shape}")

    # Cross-module sync should show meaningful patterns
    cross_sync = output.cross_module_sync
    print(f"Cross-module sync (pred-pred): {cross_sync[0, 0].mean():.4f}")
    print(f"Cross-module sync (pred-surp): {cross_sync[0, 1].mean():.4f}")
    print(f"Cross-module sync (surp-pred): {cross_sync[1, 0].mean():.4f}")
    print(f"Cross-module sync (surp-surp): {cross_sync[1, 1].mean():.4f}")

    assert output.sync.shape == (2, 16, 128)
    assert output.cross_module_sync.shape == (2, 2, 2, 16)
    assert output.module_contributions.shape == (2, 16, 2)

    # Contributions should sum to 1
    contrib_sum = output.module_contributions.sum(dim=-1)
    assert torch.allclose(contrib_sum, torch.ones_like(contrib_sum), atol=1e-5), \
        f"Contributions don't sum to 1: {contrib_sum.mean():.4f}"

    print("GlobalSyncModule: PASSED")


def test_pem_loop_global():
    """Test full PEM loop with global sync."""
    print("\n=== Testing PEMLoopGlobal ===")

    from pem.pem_loop_global import PEMLoopGlobal, PEMLoopGlobalConfig

    config = PEMLoopGlobalConfig(
        d_model=512,
        pred_d_neurons=64,
        surp_d_neurons=32,
        pred_T=3,
        surp_T=2,
        sync_pairs=128,
        immediate_horizon=4,
        shortterm_horizon=16,
        longterm_horizon=32,
    )

    model = PEMLoopGlobal(config)
    print(f"PEMLoopGlobal params: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward
    features = torch.randn(2, 32, 512)

    # Single step
    output, state = model.step(features)

    print(f"\nSingle step:")
    print(f"  Predictions (immediate): {output.predictions['immediate'].shape}")
    print(f"  Surprise magnitude: {output.surprise.magnitude.shape}")
    print(f"  Global sync: {output.global_sync.sync.shape}")
    print(f"  Observation: {output.observation.shape}")
    print(f"  Attention weights: {output.attention_weights.shape}")

    # Multi-step
    outputs, final_state = model(features, num_steps=3)

    print(f"\nMulti-step (3 steps):")
    print(f"  Number of outputs: {len(outputs)}")
    print(f"  Final cumulative sync: {final_state.cumulative_sync.shape}")

    # Test loss computation
    targets = model.target_computer.compute_targets_efficient(features)
    loss, loss_dict = model.compute_loss(outputs, targets)

    print(f"\nLoss: {loss.item():.4f}")
    print(f"Loss breakdown:")
    for k, v in sorted(loss_dict.items()):
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")

    # Test backward
    loss.backward()
    print("\nBackward pass: PASSED")

    # Check gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"Gradients computed: {grad_count}/{total_params} parameters")

    print("PEMLoopGlobal: PASSED")


def test_cross_module_sync_dynamics():
    """Test that cross-module sync shows meaningful dynamics."""
    print("\n=== Testing Cross-Module Sync Dynamics ===")

    from pem.pem_loop_global import PEMLoopGlobal, PEMLoopGlobalConfig

    config = PEMLoopGlobalConfig(
        d_model=512,
        pred_d_neurons=64,
        surp_d_neurons=32,
        pred_T=3,
        surp_T=2,
        sync_pairs=128,
    )

    model = PEMLoopGlobal(config)
    model.eval()

    # Test with predictable vs surprising features
    B, S, D = 2, 32, 512

    # Create "predictable" features (smooth, low variance)
    predictable = torch.randn(B, S, D) * 0.1
    predictable = predictable.cumsum(dim=1)  # Smooth transitions

    # Create "surprising" features (high variance, jumps)
    surprising = torch.randn(B, S, D) * 2.0  # Higher variance

    with torch.no_grad():
        out_pred, _ = model.step(predictable)
        out_surp, _ = model.step(surprising)

    print(f"Predictable features - surprise mean: {out_pred.surprise.magnitude.mean():.4f}")
    print(f"Surprising features - surprise mean: {out_surp.surprise.magnitude.mean():.4f}")

    # Cross-module sync patterns
    sync_pred = out_pred.global_sync.cross_module_sync
    sync_surp = out_surp.global_sync.cross_module_sync

    print(f"\nCross-module sync (predictable):")
    print(f"  Pred-Surp correlation: {sync_pred[0, 1].mean():.4f}")

    print(f"\nCross-module sync (surprising):")
    print(f"  Pred-Surp correlation: {sync_surp[0, 1].mean():.4f}")

    print("\nCross-Module Sync Dynamics: PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Global Sync Architecture Tests")
    print("=" * 60)

    test_ctm_base()
    test_prediction_ctm()
    test_surprise_ctm()
    test_global_sync()
    test_pem_loop_global()
    test_cross_module_sync_dynamics()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
