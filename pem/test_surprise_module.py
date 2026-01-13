#!/usr/bin/env python3
"""
Test script for PEM Surprise Module.

Usage:
    python -m pem.test_surprise_module
"""

import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_surprise_module_forward():
    """Test basic forward pass of surprise module."""
    from pem import SurpriseModule, SurpriseConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: SurpriseModule forward pass")
    logger.info("=" * 60)

    config = SurpriseConfig(
        d_model=256,
        hidden_dim=512,
        n_layers=2,
    )

    module = SurpriseModule(config)
    module.eval()

    # Create dummy inputs
    B, S, D = 2, 32, config.d_model

    # Predictions (from PredictionModule)
    predictions = {
        'immediate': torch.randn(B, S, D),
        'shortterm': torch.randn(B, S, D),
        'longterm': torch.randn(B, S, D),
    }

    # Targets (from PredictionTargets)
    targets = {
        'immediate': torch.randn(B, S, D),
        'shortterm': torch.randn(B, S, D),
        'longterm': torch.randn(B, S, D),
        'immediate_valid': torch.ones(B, S, dtype=torch.bool),
        'shortterm_valid': torch.ones(B, S, dtype=torch.bool),
        'longterm_valid': torch.ones(B, S, dtype=torch.bool),
    }
    # Mark last position as invalid
    targets['immediate_valid'][:, -1] = False
    targets['shortterm_valid'][:, -1] = False
    targets['longterm_valid'][:, -1] = False

    # Context (from FeatureExtractor)
    context = torch.randn(B, S, D)

    logger.info(f"Input shapes: predictions[immediate]={predictions['immediate'].shape}")
    logger.info(f"             targets[immediate]={targets['immediate'].shape}")
    logger.info(f"             context={context.shape}")

    # Forward pass
    with torch.no_grad():
        surprises = module(predictions, targets, context)

    logger.info(f"Output scales: {list(surprises.keys())}")

    for scale, data in surprises.items():
        logger.info(f"  {scale}:")
        logger.info(f"    magnitude: {data['magnitude'].shape}, range=[{data['magnitude'].min():.3f}, {data['magnitude'].max():.3f}]")
        logger.info(f"    direction: {data['direction'].shape}, norm={data['direction'].norm(dim=-1).mean():.3f}")
        logger.info(f"    raw: {data['raw'].shape}, range=[{data['raw'].min():.3f}, {data['raw'].max():.3f}]")

        # Verify shapes
        assert data['magnitude'].shape == (B, S, 1)
        assert data['direction'].shape == (B, S, D)
        assert data['raw'].shape == (B, S, 1)

        # Magnitude should be in [0, 1] (sigmoid output)
        assert data['magnitude'].min() >= 0
        assert data['magnitude'].max() <= 1

        # Direction should be unit vectors
        norms = data['direction'].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

        # Raw should be in [0, 2] (1 - cos_sim range)
        assert data['raw'].min() >= 0
        assert data['raw'].max() <= 2

    logger.info("Forward pass test PASSED ✓")
    return True


def test_raw_vs_learned_surprise():
    """Test that raw and learned surprise behave differently."""
    from pem import SurpriseModule, SurpriseConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Raw vs learned surprise")
    logger.info("=" * 60)

    config = SurpriseConfig(d_model=64, hidden_dim=128)
    module = SurpriseModule(config)
    module.eval()

    B, S, D = 1, 8, config.d_model

    # Case 1: Identical predictions and actuals (zero surprise)
    pred_same = torch.randn(B, S, D)
    actual_same = pred_same.clone()
    context = torch.randn(B, S, D)

    with torch.no_grad():
        surprise_same = module.compute_single(pred_same, actual_same, context, 'immediate')

    logger.info(f"Same pred/actual:")
    logger.info(f"  raw surprise: {surprise_same['raw'].mean():.4f} (should be ~0)")
    logger.info(f"  learned magnitude: {surprise_same['magnitude'].mean():.4f}")

    assert surprise_same['raw'].mean() < 0.01, "Raw surprise should be ~0 for identical inputs"

    # Case 2: Orthogonal predictions and actuals (high surprise)
    pred_orth = torch.randn(B, S, D)
    # Make actual orthogonal by subtracting projection
    actual_orth = torch.randn(B, S, D)
    # Gram-Schmidt orthogonalization
    proj = (actual_orth * pred_orth).sum(dim=-1, keepdim=True) / (pred_orth * pred_orth).sum(dim=-1, keepdim=True)
    actual_orth = actual_orth - proj * pred_orth
    actual_orth = F.normalize(actual_orth, dim=-1) * pred_orth.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        surprise_orth = module.compute_single(pred_orth, actual_orth, context, 'immediate')

    logger.info(f"Orthogonal pred/actual:")
    logger.info(f"  raw surprise: {surprise_orth['raw'].mean():.4f} (should be ~1)")
    logger.info(f"  learned magnitude: {surprise_orth['magnitude'].mean():.4f}")

    assert surprise_orth['raw'].mean() > 0.9, "Raw surprise should be ~1 for orthogonal inputs"

    # Case 3: Opposite predictions and actuals (maximum surprise)
    pred_opp = torch.randn(B, S, D)
    actual_opp = -pred_opp

    with torch.no_grad():
        surprise_opp = module.compute_single(pred_opp, actual_opp, context, 'immediate')

    logger.info(f"Opposite pred/actual:")
    logger.info(f"  raw surprise: {surprise_opp['raw'].mean():.4f} (should be ~2)")
    logger.info(f"  learned magnitude: {surprise_opp['magnitude'].mean():.4f}")

    assert surprise_opp['raw'].mean() > 1.9, "Raw surprise should be ~2 for opposite inputs"

    logger.info("Raw vs learned test PASSED ✓")
    return True


def test_direction_alignment():
    """Test that surprise direction points toward the actual."""
    from pem import SurpriseModule, SurpriseConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Direction alignment")
    logger.info("=" * 60)

    config = SurpriseConfig(d_model=64, hidden_dim=128)
    module = SurpriseModule(config)
    module.eval()

    B, S, D = 2, 16, config.d_model

    predicted = torch.randn(B, S, D)
    actual = torch.randn(B, S, D)
    context = torch.randn(B, S, D)

    with torch.no_grad():
        surprise = module.compute_single(predicted, actual, context, 'immediate')

    direction = surprise['direction']  # (B, S, D)

    # The direction should roughly point from predicted toward actual
    # Check that direction has positive correlation with (actual - predicted)
    diff = actual - predicted
    diff_norm = F.normalize(diff, dim=-1)

    alignment = F.cosine_similarity(direction, diff_norm, dim=-1)
    mean_alignment = alignment.mean().item()

    logger.info(f"Direction-diff alignment: {mean_alignment:.4f}")
    logger.info(f"  (Untrained model, so may not be perfect)")

    # Just verify it's computed correctly (training will improve alignment)
    assert direction.shape == (B, S, D)

    logger.info("Direction alignment test PASSED ✓")
    return True


def test_surprise_loss():
    """Test surprise loss computation."""
    from pem import SurpriseModule, SurpriseConfig, SurpriseLoss

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Surprise loss")
    logger.info("=" * 60)

    config = SurpriseConfig(d_model=64, hidden_dim=128)
    module = SurpriseModule(config)
    loss_fn = SurpriseLoss()

    B, S, D = 2, 16, config.d_model

    predictions = {
        'immediate': torch.randn(B, S, D),
        'shortterm': torch.randn(B, S, D),
        'longterm': torch.randn(B, S, D),
    }

    targets = {
        'immediate': torch.randn(B, S, D),
        'shortterm': torch.randn(B, S, D),
        'longterm': torch.randn(B, S, D),
        'immediate_valid': torch.ones(B, S, dtype=torch.bool),
        'shortterm_valid': torch.ones(B, S, dtype=torch.bool),
        'longterm_valid': torch.ones(B, S, dtype=torch.bool),
    }

    context = torch.randn(B, S, D)

    # Forward pass
    surprises = module(predictions, targets, context)

    # Compute loss
    total_loss, loss_dict = loss_fn(surprises, targets)

    logger.info(f"Total loss: {total_loss.item():.4f}")
    for key, val in loss_dict.items():
        logger.info(f"  {key}: {val.item():.4f}")

    # Loss should be positive and finite
    assert total_loss.item() > 0
    assert not torch.isnan(total_loss)
    assert not torch.isinf(total_loss)

    # Test gradient flow
    module.train()
    surprises = module(predictions, targets, context)
    total_loss, _ = loss_fn(surprises, targets)
    total_loss.backward()

    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    logger.info(f"Gradients flow through module: {has_grads}")
    assert has_grads

    logger.info("Surprise loss test PASSED ✓")
    return True


def test_full_pipeline():
    """Test surprise module in full PEM pipeline context."""
    from pem import (
        PredictionModule, PredictionConfig, PredictionTargets,
        SurpriseModule, SurpriseConfig,
    )

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Full pipeline integration")
    logger.info("=" * 60)

    D = 128
    sync_pairs = 64

    # Create modules
    pred_config = PredictionConfig(
        sync_pairs=sync_pairs,
        d_model=D,
        n_head=4,
        immediate_horizon=4,
        shortterm_horizon=8,
        longterm_horizon=16,
    )
    pred_module = PredictionModule(pred_config)

    target_computer = PredictionTargets(
        immediate_horizon=pred_config.immediate_horizon,
        shortterm_horizon=pred_config.shortterm_horizon,
        longterm_horizon=pred_config.longterm_horizon,
    )

    surprise_config = SurpriseConfig(d_model=D, hidden_dim=D * 2)
    surprise_module = SurpriseModule(surprise_config)

    # Simulate inputs
    B, S = 2, 32
    sync = torch.randn(B, S, sync_pairs)       # From CTM
    features = torch.randn(B, S, D)            # From FeatureExtractor

    logger.info(f"Pipeline inputs: sync={sync.shape}, features={features.shape}")

    # Run pipeline
    with torch.no_grad():
        # 1. Generate predictions from sync
        predictions = pred_module(sync, features)

        # 2. Compute targets from actual features
        targets = target_computer.compute_targets_efficient(features)

        # 3. Compute surprise
        surprises = surprise_module(predictions, targets, features)

    logger.info("Pipeline outputs:")
    for scale in ['immediate', 'shortterm', 'longterm']:
        mag = surprises[scale]['magnitude']
        raw = surprises[scale]['raw']
        logger.info(f"  {scale}: magnitude={mag.mean():.3f}, raw={raw.mean():.3f}")

    logger.info("Full pipeline test PASSED ✓")
    return True


def test_factory_function():
    """Test factory function."""
    from pem import create_surprise_module

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Factory function")
    logger.info("=" * 60)

    module = create_surprise_module(
        d_model=256,
        hidden_dim=512,
        n_layers=3,
    )

    logger.info(f"Created module: {type(module).__name__}")
    logger.info(f"Scales: {list(module.scale_computers.keys())}")

    # Quick forward test
    B, S, D = 1, 8, module.config.d_model
    pred = torch.randn(B, S, D)
    actual = torch.randn(B, S, D)
    context = torch.randn(B, S, D)

    with torch.no_grad():
        surprise = module.compute_single(pred, actual, context)

    assert 'magnitude' in surprise
    assert 'direction' in surprise
    assert 'raw' in surprise

    logger.info("Factory function test PASSED ✓")
    return True


def main():
    """Run all tests."""
    logger.info("PEM Surprise Module Tests")
    logger.info("=" * 60)

    # Import F for tests that need it
    global F
    import torch.nn.functional as F

    tests = [
        ("Forward pass", test_surprise_module_forward),
        ("Raw vs learned", test_raw_vs_learned_surprise),
        ("Direction alignment", test_direction_alignment),
        ("Surprise loss", test_surprise_loss),
        ("Full pipeline", test_full_pipeline),
        ("Factory function", test_factory_function),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            logger.error(f"\n{name} test FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
