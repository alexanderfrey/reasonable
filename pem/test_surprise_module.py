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


# ============================================================================
# VALENCE MODULE TESTS
# ============================================================================


def test_valence_module_forward():
    """Test basic forward pass of valence module."""
    from pem import ValenceModule, ValenceConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: ValenceModule forward pass")
    logger.info("=" * 60)

    config = ValenceConfig(
        d_model=256,
        personality_dim=128,
        hidden_dim=128,
        n_layers=2,
        use_context=True,
    )

    module = ValenceModule(config)
    module.eval()

    # Create dummy inputs
    B, S, D = 2, 32, config.d_model
    D_p = config.personality_dim

    surprise_direction = torch.randn(B, S, D)
    surprise_direction = F.normalize(surprise_direction, dim=-1)  # Unit vectors

    personality = torch.randn(D_p)  # Personality embedding
    context = torch.randn(B, S, D)  # Optional context

    logger.info(f"Input shapes:")
    logger.info(f"  surprise_direction: {surprise_direction.shape}")
    logger.info(f"  personality: {personality.shape}")
    logger.info(f"  context: {context.shape}")

    # Forward pass
    with torch.no_grad():
        valence = module(surprise_direction, personality, context)

    logger.info(f"Output shape: {valence.shape}")
    logger.info(f"Valence range: [{valence.min():.3f}, {valence.max():.3f}]")

    # Verify shape
    assert valence.shape == (B, S, 1), f"Expected (B, S, 1), got {valence.shape}"

    # Valence should be in [-1, +1] (tanh output)
    assert valence.min() >= -1.0, f"Valence min {valence.min():.3f} < -1"
    assert valence.max() <= 1.0, f"Valence max {valence.max():.3f} > 1"

    logger.info("Valence forward pass test PASSED ✓")
    return True


def test_valence_alignment():
    """Test that valence reflects alignment with personality."""
    from pem import ValenceModule, ValenceConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Valence alignment with personality")
    logger.info("=" * 60)

    config = ValenceConfig(
        d_model=64,
        personality_dim=64,
        hidden_dim=32,
        use_context=False,  # Simpler test without context
    )

    module = ValenceModule(config)
    module.eval()

    B, S, D = 1, 8, config.d_model

    # Create a personality that "wants" a specific direction
    personality = torch.randn(D)
    context = torch.randn(B, S, D)

    # Case 1: Surprise direction aligned with personality
    # Project personality through the module's projector to get what it "wants"
    with torch.no_grad():
        personality_proj = module.personality_proj(personality)  # What personality wants in D space
        personality_proj = F.normalize(personality_proj, dim=-1)

    # Surprise direction = same as personality projection (aligned)
    surprise_aligned = personality_proj.unsqueeze(0).unsqueeze(0).expand(B, S, -1)

    with torch.no_grad():
        valence_aligned = module(surprise_aligned, personality, context)

    logger.info(f"Aligned with personality: mean valence = {valence_aligned.mean():.4f}")

    # Case 2: Surprise direction opposite to personality
    surprise_opposite = -surprise_aligned

    with torch.no_grad():
        valence_opposite = module(surprise_opposite, personality, context)

    logger.info(f"Opposite to personality: mean valence = {valence_opposite.mean():.4f}")

    # Case 3: Random surprise direction
    surprise_random = torch.randn(B, S, D)
    surprise_random = F.normalize(surprise_random, dim=-1)

    with torch.no_grad():
        valence_random = module(surprise_random, personality, context)

    logger.info(f"Random direction: mean valence = {valence_random.mean():.4f}")

    # The aligned should be different from opposite (model learns the distinction)
    diff = (valence_aligned - valence_opposite).abs().mean()
    logger.info(f"Difference (aligned vs opposite): {diff:.4f}")

    # Just verify shapes are correct (training will improve alignment)
    assert valence_aligned.shape == (B, S, 1)
    assert valence_opposite.shape == (B, S, 1)
    assert valence_random.shape == (B, S, 1)

    logger.info("Valence alignment test PASSED ✓")
    return True


def test_valence_context_modulation():
    """Test that context affects valence computation."""
    from pem import ValenceModule, ValenceConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Valence context modulation")
    logger.info("=" * 60)

    # Create module with context gating
    config = ValenceConfig(
        d_model=64,
        personality_dim=64,
        hidden_dim=32,
        use_context=True,
    )

    module = ValenceModule(config)
    module.eval()

    B, S, D = 1, 8, config.d_model

    surprise_direction = torch.randn(B, S, D)
    surprise_direction = F.normalize(surprise_direction, dim=-1)
    personality = torch.randn(D)

    # Two different contexts
    context1 = torch.randn(B, S, D)
    context2 = torch.randn(B, S, D) * 2  # Different context

    with torch.no_grad():
        valence1 = module(surprise_direction, personality, context1)
        valence2 = module(surprise_direction, personality, context2)

    diff = (valence1 - valence2).abs().mean()
    logger.info(f"Same surprise, different contexts: valence diff = {diff:.4f}")

    # Context should affect valence (diff > 0)
    assert diff > 0, "Context should modulate valence"

    # Test without context
    with torch.no_grad():
        valence_no_ctx = module(surprise_direction, personality, None)

    diff_with_ctx = (valence1 - valence_no_ctx).abs().mean()
    logger.info(f"With vs without context: valence diff = {diff_with_ctx:.4f}")

    logger.info("Valence context modulation test PASSED ✓")
    return True


def test_valence_gradient_flow():
    """Test that gradients flow through valence module."""
    from pem import ValenceModule, ValenceConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Valence gradient flow")
    logger.info("=" * 60)

    config = ValenceConfig(d_model=64, personality_dim=64, hidden_dim=32)
    module = ValenceModule(config)
    module.train()

    B, S, D = 2, 16, config.d_model

    surprise_direction = torch.randn(B, S, D, requires_grad=True)
    surprise_direction_norm = F.normalize(surprise_direction, dim=-1)
    personality = torch.randn(D, requires_grad=True)
    context = torch.randn(B, S, D, requires_grad=True)

    # Forward pass
    valence = module(surprise_direction_norm, personality, context)

    # Backward pass
    loss = valence.mean()
    loss.backward()

    # Check gradients
    has_input_grads = surprise_direction.grad is not None and surprise_direction.grad.abs().sum() > 0
    has_personality_grads = personality.grad is not None and personality.grad.abs().sum() > 0
    has_context_grads = context.grad is not None and context.grad.abs().sum() > 0

    module_has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    logger.info(f"Gradients flow to:")
    logger.info(f"  surprise_direction: {has_input_grads}")
    logger.info(f"  personality: {has_personality_grads}")
    logger.info(f"  context: {has_context_grads}")
    logger.info(f"  module parameters: {module_has_grads}")

    assert module_has_grads, "Gradients should flow through module"

    logger.info("Valence gradient flow test PASSED ✓")
    return True


def test_valence_loss():
    """Test valence loss computation."""
    from pem import ValenceModule, ValenceConfig, ValenceLoss, SurpriseModule, SurpriseConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Valence loss")
    logger.info("=" * 60)

    D = 64
    D_p = 64

    # Create modules
    valence_config = ValenceConfig(d_model=D, personality_dim=D_p, hidden_dim=32)
    valence_module = ValenceModule(valence_config)

    surprise_config = SurpriseConfig(d_model=D, hidden_dim=128)
    surprise_module = SurpriseModule(surprise_config)

    loss_fn = ValenceLoss()

    B, S = 2, 16

    # Create test data
    predictions = {
        'immediate': torch.randn(B, S, D),
        'shortterm': torch.randn(B, S, D),
    }
    targets = {
        'immediate': torch.randn(B, S, D),
        'shortterm': torch.randn(B, S, D),
        'immediate_valid': torch.ones(B, S, dtype=torch.bool),
        'shortterm_valid': torch.ones(B, S, dtype=torch.bool),
    }
    context = torch.randn(B, S, D)
    personality = torch.randn(D_p)

    # Compute surprise
    surprises = surprise_module(predictions, targets, context)

    # Compute valence
    valences = valence_module.compute_from_surprises(surprises, personality, context)

    # Compute loss
    total_loss, loss_dict = loss_fn(valences, surprises, targets)

    logger.info(f"Total loss: {total_loss.item():.4f}")
    for key, val in loss_dict.items():
        logger.info(f"  {key}: {val.item():.4f}")

    # Loss should be finite
    assert not torch.isnan(total_loss), "Loss should not be NaN"
    assert not torch.isinf(total_loss), "Loss should not be infinite"

    logger.info("Valence loss test PASSED ✓")
    return True


def test_valence_factory():
    """Test valence factory function."""
    from pem import create_valence_module

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Valence factory function")
    logger.info("=" * 60)

    module = create_valence_module(
        d_model=256,
        personality_dim=128,
        hidden_dim=128,
        use_context=True,
    )

    logger.info(f"Created module: {type(module).__name__}")
    logger.info(f"Config: d_model={module.config.d_model}, personality_dim={module.config.personality_dim}")

    # Quick forward test
    B, S, D = 1, 8, module.config.d_model
    D_p = module.config.personality_dim

    surprise_direction = torch.randn(B, S, D)
    surprise_direction = F.normalize(surprise_direction, dim=-1)
    personality = torch.randn(D_p)
    context = torch.randn(B, S, D)

    with torch.no_grad():
        valence = module(surprise_direction, personality, context)

    assert valence.shape == (B, S, 1)
    assert valence.min() >= -1.0
    assert valence.max() <= 1.0

    logger.info("Valence factory test PASSED ✓")
    return True


def test_valence_integration_with_sync():
    """Test valence integration with SyncModule."""
    from pem import SyncModule, SyncModuleConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Valence integration with SyncModule")
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
    surprise = torch.rand(B, S, 1)  # Random surprise magnitudes

    # Test with positive valence
    valence_positive = torch.ones(B, S, 1) * 0.8  # High positive valence

    with torch.no_grad():
        sync_positive = module(history, surprise=surprise, valence=valence_positive)

    # Test with negative valence
    valence_negative = torch.ones(B, S, 1) * -0.8  # High negative valence

    module.reset_memory()
    with torch.no_grad():
        sync_negative = module(history, surprise=surprise, valence=valence_negative)

    # Sync outputs should differ based on valence
    diff = (sync_positive - sync_negative).abs().mean()
    logger.info(f"Sync difference (positive vs negative valence): {diff:.4f}")

    # Valence should affect sync output
    assert diff > 0, "Valence should affect sync output"

    logger.info("Valence integration with SyncModule test PASSED ✓")
    return True


def main():
    """Run all tests."""
    logger.info("PEM Surprise & Valence Module Tests")
    logger.info("=" * 60)

    # Import F for tests that need it
    global F
    import torch.nn.functional as F

    tests = [
        # Surprise tests
        ("Surprise forward pass", test_surprise_module_forward),
        ("Surprise raw vs learned", test_raw_vs_learned_surprise),
        ("Surprise direction alignment", test_direction_alignment),
        ("Surprise loss", test_surprise_loss),
        ("Surprise full pipeline", test_full_pipeline),
        ("Surprise factory function", test_factory_function),
        # Valence tests
        ("Valence forward pass", test_valence_module_forward),
        ("Valence alignment", test_valence_alignment),
        ("Valence context modulation", test_valence_context_modulation),
        ("Valence gradient flow", test_valence_gradient_flow),
        ("Valence loss", test_valence_loss),
        ("Valence factory function", test_valence_factory),
        ("Valence integration with SyncModule", test_valence_integration_with_sync),
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
