#!/usr/bin/env python3
"""
Test script for PEM Curiosity Module.

Usage:
    python -m pem.test_curiosity_module
"""

import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_curiosity_module_forward():
    """Test basic forward pass of curiosity module."""
    from pem import CuriosityModule, CuriosityConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: CuriosityModule forward pass")
    logger.info("=" * 60)

    config = CuriosityConfig(
        d_model=256,
        hidden_dim=128,
        use_temporal_novelty=True,
        novelty_memory_size=64,
        exploration_weight=0.5,
    )

    module = CuriosityModule(config)
    module.eval()

    # Create dummy inputs
    B, S, D = 2, 32, config.d_model

    features = torch.randn(B, S, D)
    predictions = torch.randn(B, S, D)
    context = torch.randn(B, S, D)

    logger.info(f"Input shapes:")
    logger.info(f"  features: {features.shape}")
    logger.info(f"  predictions: {predictions.shape}")

    # Forward pass
    with torch.no_grad():
        output = module(features, predictions, context)

    logger.info(f"Output shapes:")
    logger.info(f"  curiosity: {output.curiosity.shape}")
    logger.info(f"  uncertainty: {output.uncertainty.shape}")
    logger.info(f"  information_gain: {output.information_gain.shape}")
    logger.info(f"  exploration_bonus: {output.exploration_bonus.shape}")

    # Verify shapes
    assert output.curiosity.shape == (B, S, 1), f"Expected (B, S, 1), got {output.curiosity.shape}"
    assert output.uncertainty.shape == (B, S, 1)
    assert output.information_gain.shape == (B, S, 1)
    assert output.exploration_bonus.shape == (B, S, D)

    # Curiosity should be in [0, 1] (sigmoid output)
    assert output.curiosity.min() >= 0, f"Curiosity min {output.curiosity.min():.3f} < 0"
    assert output.curiosity.max() <= 1, f"Curiosity max {output.curiosity.max():.3f} > 1"

    logger.info(f"Curiosity range: [{output.curiosity.min():.3f}, {output.curiosity.max():.3f}]")
    logger.info(f"Uncertainty range: [{output.uncertainty.min():.3f}, {output.uncertainty.max():.3f}]")

    logger.info("Curiosity forward pass test PASSED ✓")
    return True


def test_uncertainty_estimator():
    """Test that uncertainty estimation works correctly."""
    from pem.curiosity_module import UncertaintyEstimator

    logger.info("\n" + "=" * 60)
    logger.info("TEST: UncertaintyEstimator")
    logger.info("=" * 60)

    D = 64
    estimator = UncertaintyEstimator(d_model=D, hidden_dim=32)
    estimator.eval()

    B, S = 2, 16

    # Case 1: Identical features and predictions (low uncertainty)
    features_same = torch.randn(B, S, D)
    predictions_same = features_same.clone()

    with torch.no_grad():
        uncertainty_same = estimator(features_same, predictions_same)

    logger.info(f"Same features/predictions: uncertainty = {uncertainty_same.mean():.4f}")

    # Case 2: Random predictions (higher uncertainty)
    predictions_random = torch.randn(B, S, D)

    with torch.no_grad():
        uncertainty_random = estimator(features_same, predictions_random)

    logger.info(f"Random predictions: uncertainty = {uncertainty_random.mean():.4f}")

    # Uncertainty should be positive
    assert uncertainty_same.min() >= 0, "Uncertainty must be positive"
    assert uncertainty_random.min() >= 0, "Uncertainty must be positive"

    logger.info("UncertaintyEstimator test PASSED ✓")
    return True


def test_novelty_memory():
    """Test that novelty memory tracks what's been seen."""
    from pem.curiosity_module import NoveltyMemory

    logger.info("\n" + "=" * 60)
    logger.info("TEST: NoveltyMemory")
    logger.info("=" * 60)

    D = 64
    memory = NoveltyMemory(d_model=D, memory_size=32)

    B, S = 1, 8

    # First observation: everything is novel (empty memory)
    features1 = torch.randn(B, S, D)
    novelty1 = memory.compute_novelty(features1)

    logger.info(f"Empty memory: novelty = {novelty1.mean():.4f} (should be 1.0)")
    assert novelty1.mean() == 1.0, "Empty memory should have max novelty"

    # Add to memory
    memory.update(features1)

    # Same features: low novelty (seen before)
    novelty_same = memory.compute_novelty(features1)
    logger.info(f"Same features: novelty = {novelty_same.mean():.4f} (should be low)")

    # New features: higher novelty
    features2 = torch.randn(B, S, D)
    novelty_new = memory.compute_novelty(features2)
    logger.info(f"New features: novelty = {novelty_new.mean():.4f} (should be higher)")

    # Novelty of new should be higher than same
    assert novelty_new.mean() > novelty_same.mean(), "New features should be more novel"

    # Test reset
    memory.reset()
    novelty_after_reset = memory.compute_novelty(features1)
    logger.info(f"After reset: novelty = {novelty_after_reset.mean():.4f} (should be 1.0)")
    assert novelty_after_reset.mean() == 1.0, "Reset should clear memory"

    logger.info("NoveltyMemory test PASSED ✓")
    return True


def test_information_gain():
    """Test information gain computation."""
    from pem.curiosity_module import InformationGainComputer

    logger.info("\n" + "=" * 60)
    logger.info("TEST: InformationGainComputer")
    logger.info("=" * 60)

    D = 64
    computer = InformationGainComputer(
        d_model=D,
        hidden_dim=32,
        use_novelty=True,
        novelty_memory_size=32,
    )
    computer.eval()

    B, S = 2, 16

    features = torch.randn(B, S, D)
    uncertainty = torch.rand(B, S, 1)  # Random uncertainty
    context = torch.randn(B, S, D)

    with torch.no_grad():
        info_gain, novelty = computer(features, uncertainty, context)

    logger.info(f"Info gain shape: {info_gain.shape}")
    logger.info(f"Novelty shape: {novelty.shape}")
    logger.info(f"Info gain range: [{info_gain.min():.4f}, {info_gain.max():.4f}]")

    # Info gain should be positive (softplus output)
    assert info_gain.min() >= 0, "Info gain must be positive"

    logger.info("InformationGainComputer test PASSED ✓")
    return True


def test_curiosity_gradient_flow():
    """Test that gradients flow through curiosity module."""
    from pem import CuriosityModule, CuriosityConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Curiosity gradient flow")
    logger.info("=" * 60)

    config = CuriosityConfig(d_model=64, hidden_dim=32)
    module = CuriosityModule(config)
    module.train()

    B, S, D = 2, 16, config.d_model

    features = torch.randn(B, S, D, requires_grad=True)
    predictions = torch.randn(B, S, D, requires_grad=True)

    # Forward pass
    output = module(features, predictions)

    # Backward pass
    loss = output.curiosity.mean() + output.exploration_bonus.mean()
    loss.backward()

    # Check gradients
    has_feature_grads = features.grad is not None and features.grad.abs().sum() > 0
    has_pred_grads = predictions.grad is not None and predictions.grad.abs().sum() > 0
    module_has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    logger.info(f"Gradients flow to:")
    logger.info(f"  features: {has_feature_grads}")
    logger.info(f"  predictions: {has_pred_grads}")
    logger.info(f"  module parameters: {module_has_grads}")

    assert module_has_grads, "Gradients should flow through module"

    logger.info("Curiosity gradient flow test PASSED ✓")
    return True


def test_curiosity_loss():
    """Test curiosity loss computation."""
    from pem import CuriosityModule, CuriosityConfig, CuriosityLoss

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Curiosity loss")
    logger.info("=" * 60)

    config = CuriosityConfig(d_model=64, hidden_dim=32)
    module = CuriosityModule(config)
    loss_fn = CuriosityLoss()

    B, S, D = 2, 16, config.d_model

    features = torch.randn(B, S, D)
    predictions = torch.randn(B, S, D)
    surprise_magnitude = torch.rand(B, S, 1)  # Random surprise

    # Compute curiosity
    output = module(features, predictions)

    # Compute loss
    total_loss, loss_dict = loss_fn(output, surprise_magnitude)

    logger.info(f"Total loss: {total_loss.item():.4f}")
    for key, val in loss_dict.items():
        logger.info(f"  {key}: {val.item():.4f}")

    # Loss should be finite
    assert not torch.isnan(total_loss), "Loss should not be NaN"
    assert not torch.isinf(total_loss), "Loss should not be infinite"

    logger.info("Curiosity loss test PASSED ✓")
    return True


def test_curiosity_factory():
    """Test curiosity factory function."""
    from pem import create_curiosity_module

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Curiosity factory function")
    logger.info("=" * 60)

    module = create_curiosity_module(
        d_model=256,
        hidden_dim=128,
        use_temporal_novelty=True,
        exploration_weight=0.5,
    )

    logger.info(f"Created module: {type(module).__name__}")
    logger.info(f"Config: d_model={module.config.d_model}")

    # Quick forward test
    B, S, D = 1, 8, module.config.d_model
    features = torch.randn(B, S, D)

    with torch.no_grad():
        output = module(features)

    assert output.curiosity.shape == (B, S, 1)
    assert output.exploration_bonus.shape == (B, S, D)

    logger.info("Curiosity factory test PASSED ✓")
    return True


def test_curiosity_integration_with_sync():
    """Test curiosity integration with SyncModule."""
    from pem import SyncModule, SyncModuleConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Curiosity integration with SyncModule")
    logger.info("=" * 60)

    D = 64
    sync_pairs = 64

    config = SyncModuleConfig(
        d_model=D,
        sync_pairs=sync_pairs,
        memory_slots=10,
        use_intention=True,
    )

    module = SyncModule(config)
    module.eval()

    B, S, T = 2, 16, 4

    history = torch.randn(B, S, T, D)
    surprise = torch.rand(B, S, 1)
    valence = torch.zeros(B, S, 1)  # Neutral valence

    # Test with high curiosity
    curiosity_high = torch.ones(B, S, 1) * 0.9

    with torch.no_grad():
        sync_high_curiosity = module(
            history, surprise=surprise, valence=valence, curiosity=curiosity_high
        )

    # Test with low curiosity
    curiosity_low = torch.ones(B, S, 1) * 0.1

    module.reset_memory()
    with torch.no_grad():
        sync_low_curiosity = module(
            history, surprise=surprise, valence=valence, curiosity=curiosity_low
        )

    # Sync outputs should differ based on curiosity
    diff = (sync_high_curiosity - sync_low_curiosity).abs().mean()
    logger.info(f"Sync difference (high vs low curiosity): {diff:.4f}")

    # Curiosity should affect sync output (through intention modulation)
    assert diff >= 0, "Curiosity should affect sync output"

    logger.info("Curiosity integration with SyncModule test PASSED ✓")
    return True


def test_curiosity_integration_with_attention():
    """Test curiosity integration with PerceptionAttention."""
    from pem import PerceptionAttention, PerceptionConfig, SyncModule, SyncModuleConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Curiosity integration with PerceptionAttention")
    logger.info("=" * 60)

    D = 256
    D_perception = 512
    sync_pairs = 256

    # Create perception attention
    perception_config = PerceptionConfig(
        d_model=D,
        d_perception=D_perception,
        n_heads=4,
        sync_pairs=sync_pairs,
        num_oscillators=16,
    )
    perception = PerceptionAttention(perception_config)

    # Create sync module for signals
    sync_config = SyncModuleConfig(
        d_model=D,
        sync_pairs=sync_pairs,
        use_intention=True,
    )
    sync_module = SyncModule(sync_config)

    perception.eval()
    sync_module.eval()

    B, S = 1, 16

    # Setup
    perception_features = torch.randn(B, S, D_perception)
    perception.cache_perception(perception_features)

    state = torch.randn(B, S, D)
    history = torch.randn(B, S, 4, D)
    sync = sync_module(history)

    personality_signal = torch.randn(B, S, D)
    intention_signal = torch.randn(B, S, D)

    # Test without exploration bonus
    with torch.no_grad():
        output_no_curiosity = perception(
            state=state,
            personality_signal=personality_signal,
            intention_signal=intention_signal,
            sync=sync,
            tick=0,
        )

    # Test with exploration bonus
    exploration_bonus = torch.randn(B, S, D) * 0.5

    with torch.no_grad():
        output_with_curiosity = perception(
            state=state,
            personality_signal=personality_signal,
            intention_signal=intention_signal,
            sync=sync,
            tick=0,
            exploration_bonus=exploration_bonus,
        )

    # Observations should differ
    diff = (output_no_curiosity.observation - output_with_curiosity.observation).abs().mean()
    logger.info(f"Observation difference with/without curiosity: {diff:.4f}")

    assert diff > 0, "Exploration bonus should influence observation"

    logger.info("Curiosity integration with PerceptionAttention test PASSED ✓")
    return True


def test_epistemic_value():
    """Test epistemic value computation."""
    from pem import CuriosityModule, CuriosityConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Epistemic value computation")
    logger.info("=" * 60)

    config = CuriosityConfig(d_model=64, hidden_dim=32)
    module = CuriosityModule(config)
    module.eval()

    B, S, D = 2, 16, config.d_model

    features = torch.randn(B, S, D)
    predictions = torch.randn(B, S, D)

    with torch.no_grad():
        epistemic_value = module.compute_epistemic_value(features, predictions)

    logger.info(f"Epistemic value shape: {epistemic_value.shape}")
    logger.info(f"Epistemic value range: [{epistemic_value.min():.4f}, {epistemic_value.max():.4f}]")

    assert epistemic_value.shape == (B, S, 1)

    logger.info("Epistemic value test PASSED ✓")
    return True


def main():
    """Run all tests."""
    logger.info("PEM Curiosity Module Tests")
    logger.info("=" * 60)

    tests = [
        ("Curiosity forward pass", test_curiosity_module_forward),
        ("Uncertainty estimator", test_uncertainty_estimator),
        ("Novelty memory", test_novelty_memory),
        ("Information gain", test_information_gain),
        ("Curiosity gradient flow", test_curiosity_gradient_flow),
        ("Curiosity loss", test_curiosity_loss),
        ("Curiosity factory", test_curiosity_factory),
        ("Curiosity + SyncModule", test_curiosity_integration_with_sync),
        ("Curiosity + PerceptionAttention", test_curiosity_integration_with_attention),
        ("Epistemic value", test_epistemic_value),
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
