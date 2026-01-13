#!/usr/bin/env python3
"""
Test script for PEM Prediction Module.

Usage:
    python -m pem.test_prediction_module
"""

import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_prediction_module_forward():
    """Test basic forward pass of prediction module."""
    from pem import PredictionModule, PredictionConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: PredictionModule forward pass")
    logger.info("=" * 60)

    config = PredictionConfig(
        sync_pairs=512,
        d_model=1536,
        n_head=8,
        shortterm_horizon=64,
    )

    module = PredictionModule(config)
    module.eval()

    # Create dummy inputs
    B, S = 2, 128
    sync = torch.randn(B, S, config.sync_pairs)
    context = torch.randn(B, S, config.d_model)

    logger.info(f"Input shapes: sync={sync.shape}, context={context.shape}")

    # Forward pass
    with torch.no_grad():
        predictions = module(sync, context)

    logger.info(f"Output keys: {list(predictions.keys())}")
    for key, val in predictions.items():
        logger.info(f"  {key}: shape={val.shape}, dtype={val.dtype}")

    # Verify shapes
    assert predictions['immediate'].shape == (B, S, config.d_model)
    assert predictions['shortterm'].shape == (B, S, config.d_model)
    assert predictions['longterm'].shape == (B, S, config.d_model)
    assert predictions['basis'].shape == (B, S, config.d_model)

    logger.info("Forward pass test PASSED ✓")
    return True


def test_causal_masking():
    """Test that causal masking works correctly."""
    from pem.prediction_module import CausalCrossAttention

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Causal masking")
    logger.info("=" * 60)

    d_model = 64
    n_head = 4
    attn = CausalCrossAttention(d_model, n_head, use_flash=False)
    attn.eval()

    # Create inputs where each position has distinct content
    B, S = 1, 8
    query = torch.randn(B, S, d_model)
    key = torch.randn(B, S, d_model)
    value = torch.randn(B, S, d_model)

    # Forward pass
    with torch.no_grad():
        out1 = attn(query, key, value)

    # Now modify key/value at position 5 and check that outputs at positions < 5 don't change
    key_modified = key.clone()
    value_modified = value.clone()
    key_modified[:, 5:, :] = torch.randn(B, S - 5, d_model)
    value_modified[:, 5:, :] = torch.randn(B, S - 5, d_model)

    with torch.no_grad():
        out2 = attn(query, key_modified, value_modified)

    # Positions 0-4 should be unchanged (can't attend to positions >= 5)
    early_diff = (out1[:, :5] - out2[:, :5]).abs().max().item()
    # Position 5+ should change
    late_diff = (out1[:, 5:] - out2[:, 5:]).abs().max().item()

    logger.info(f"Max diff at positions 0-4 (should be ~0): {early_diff:.6f}")
    logger.info(f"Max diff at positions 5+ (should be >0): {late_diff:.6f}")

    assert early_diff < 1e-5, f"Causal masking failed: early positions affected by future changes"
    assert late_diff > 0.01, f"Causal masking may be wrong: late positions unchanged"

    logger.info("Causal masking test PASSED ✓")
    return True


def test_prediction_targets():
    """Test target computation for training."""
    from pem import PredictionTargets

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Prediction targets computation")
    logger.info("=" * 60)

    # Test with all horizons explicit
    # immediate=2, shortterm=4, longterm=None (rest of sequence)
    target_computer = PredictionTargets(
        immediate_horizon=2,
        shortterm_horizon=4,
        longterm_horizon=None,
    )

    # Create simple features for easy verification
    B, S, D = 2, 10, 8
    features = torch.arange(S).float().unsqueeze(0).unsqueeze(-1).expand(B, S, D)
    # features[b, t, :] = t (all dims same value)

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Features[0, :, 0]: {features[0, :, 0].tolist()}")

    # Compute targets
    targets = target_computer.compute_targets_efficient(features)

    logger.info(f"Target keys: {list(targets.keys())}")

    # Verify immediate targets (mean of next 2 tokens)
    # immediate[3] = mean(features[4:6]) = mean([4,5]) = 4.5
    immediate_at_3 = targets['immediate'][0, 3, 0].item()
    expected_immediate_3 = (4 + 5) / 2  # 4.5
    logger.info(f"  Position 3 immediate (horizon=2): {immediate_at_3:.2f} (expected: {expected_immediate_3})")

    # Verify shortterm targets (mean of next 4 tokens)
    # shortterm[3] = mean(features[4:8]) = mean([4,5,6,7]) = 5.5
    shortterm_at_3 = targets['shortterm'][0, 3, 0].item()
    expected_shortterm_3 = (4 + 5 + 6 + 7) / 4  # 5.5
    logger.info(f"  Position 3 shortterm (horizon=4): {shortterm_at_3:.2f} (expected: {expected_shortterm_3})")

    # Verify longterm targets (mean of all remaining when longterm_horizon=None)
    # longterm[3] = mean(features[4:10]) = mean([4,5,6,7,8,9]) = 6.5
    longterm_at_3 = targets['longterm'][0, 3, 0].item()
    expected_longterm_3 = (4 + 5 + 6 + 7 + 8 + 9) / 6  # 6.5
    logger.info(f"  Position 3 longterm (horizon=None): {longterm_at_3:.2f} (expected: {expected_longterm_3})")

    assert abs(immediate_at_3 - expected_immediate_3) < 0.01
    assert abs(shortterm_at_3 - expected_shortterm_3) < 0.01
    assert abs(longterm_at_3 - expected_longterm_3) < 0.01

    # Test with fixed longterm_horizon=4
    logger.info("\nTesting with fixed longterm_horizon=4:")
    target_computer_fixed = PredictionTargets(immediate_horizon=2, shortterm_horizon=4, longterm_horizon=4)
    targets_fixed = target_computer_fixed.compute_targets_efficient(features)

    # longterm[3] with horizon=4 = mean(features[4:8]) = mean([4,5,6,7]) = 5.5
    longterm_fixed_at_3 = targets_fixed['longterm'][0, 3, 0].item()
    expected_longterm_fixed_3 = (4 + 5 + 6 + 7) / 4  # 5.5
    logger.info(f"  Position 3 longterm (horizon=4): {longterm_fixed_at_3:.2f} (expected: {expected_longterm_fixed_3})")
    assert abs(longterm_fixed_at_3 - expected_longterm_fixed_3) < 0.01

    # Test default horizons (immediate=8)
    logger.info("\nTesting default horizons (immediate=8):")
    target_computer_default = PredictionTargets()  # immediate=8, shortterm=64, longterm=2048
    targets_default = target_computer_default.compute_targets_efficient(features)
    # immediate[3] with horizon=8 = mean(features[4:10]) = mean([4,5,6,7,8,9]) = 6.5 (capped at S=10)
    immediate_default_at_3 = targets_default['immediate'][0, 3, 0].item()
    expected_immediate_default_3 = (4 + 5 + 6 + 7 + 8 + 9) / 6  # 6.5
    logger.info(f"  Position 3 immediate (horizon=8, capped): {immediate_default_at_3:.2f} (expected: {expected_immediate_default_3})")
    assert abs(immediate_default_at_3 - expected_immediate_default_3) < 0.01

    # Check validity masks
    logger.info(f"\nValidity masks:")
    logger.info(f"  Immediate valid positions: {targets['immediate_valid'].sum(dim=1).tolist()}")
    logger.info(f"  Shortterm valid positions: {targets['shortterm_valid'].sum(dim=1).tolist()}")
    logger.info(f"  Longterm valid positions: {targets['longterm_valid'].sum(dim=1).tolist()}")

    logger.info("Prediction targets test PASSED ✓")
    return True


def test_prediction_loss():
    """Test loss computation."""
    from pem import PredictionModule, PredictionConfig, PredictionTargets, PredictionLoss

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Prediction loss")
    logger.info("=" * 60)

    config = PredictionConfig(
        sync_pairs=64, d_model=128, n_head=4,
        immediate_horizon=4, shortterm_horizon=8, longterm_horizon=16
    )
    module = PredictionModule(config)
    target_computer = PredictionTargets(
        immediate_horizon=config.immediate_horizon,
        shortterm_horizon=config.shortterm_horizon,
        longterm_horizon=config.longterm_horizon,
    )
    loss_fn = PredictionLoss(use_cosine=True, use_mse=False)

    # Create inputs
    B, S = 2, 32
    sync = torch.randn(B, S, config.sync_pairs)
    features = torch.randn(B, S, config.d_model)

    # Forward pass
    predictions = module(sync, features)
    targets = target_computer.compute_targets_efficient(features)

    # Compute loss
    total_loss, loss_dict = loss_fn(predictions, targets)

    logger.info(f"Total loss: {total_loss.item():.4f}")
    for key, val in loss_dict.items():
        logger.info(f"  {key}: {val.item():.4f}")

    # Loss should be positive and finite
    assert total_loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(total_loss), "Loss should not be NaN"
    assert not torch.isinf(total_loss), "Loss should not be infinite"

    # Test gradient flow
    module.train()
    predictions = module(sync, features)
    total_loss, _ = loss_fn(predictions, targets)
    total_loss.backward()

    # Check gradients exist
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    logger.info(f"Gradients flow through module: {has_grads}")
    assert has_grads, "Gradients should flow through module"

    logger.info("Prediction loss test PASSED ✓")
    return True


def test_factory_function():
    """Test factory function."""
    from pem import create_prediction_module

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Factory function")
    logger.info("=" * 60)

    module = create_prediction_module(
        sync_pairs=256,
        d_model=768,
        n_head=12,
        shortterm_horizon=64,
    )

    logger.info(f"Created module: {type(module).__name__}")
    logger.info(f"Config: sync_pairs={module.config.sync_pairs}, d_model={module.config.d_model}")

    # Quick forward test
    B, S = 1, 16
    sync = torch.randn(B, S, module.config.sync_pairs)
    context = torch.randn(B, S, module.config.d_model)

    with torch.no_grad():
        predictions = module(sync, context)

    assert 'immediate' in predictions
    logger.info("Factory function test PASSED ✓")
    return True


def main():
    """Run all tests."""
    logger.info("PEM Prediction Module Tests")
    logger.info("=" * 60)

    tests = [
        ("Forward pass", test_prediction_module_forward),
        ("Causal masking", test_causal_masking),
        ("Prediction targets", test_prediction_targets),
        ("Prediction loss", test_prediction_loss),
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
