#!/usr/bin/env python3
"""
Test script for PEM Imagination (Mental Simulation) Module.

Usage:
    python -m pem.test_imagination_module
"""

import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_imagination_module_forward():
    """Test basic forward pass of imagination module."""
    from pem import ImaginationModule, ImaginationConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: ImaginationModule forward pass")
    logger.info("=" * 60)

    config = ImaginationConfig(
        d_model=256,
        hidden_dim=128,
        use_scene_generation=True,
        use_mind_modeling=True,
        use_counterfactuals=True,
        num_counterfactuals=3,
    )

    module = ImaginationModule(config)
    module.eval()

    # Create dummy inputs
    B, S, D = 2, 32, config.d_model

    features = torch.randn(B, S, D)
    memory = torch.randn(B, S, D)
    personality = torch.randn(D)

    logger.info(f"Input shapes:")
    logger.info(f"  features: {features.shape}")
    logger.info(f"  memory: {memory.shape}")
    logger.info(f"  personality: {personality.shape}")

    # Forward pass
    with torch.no_grad():
        output = module(features, memory=memory, personality=personality)

    logger.info(f"Output shapes:")
    logger.info(f"  imagined_features: {output.imagined_features.shape}")
    logger.info(f"  vividness: {output.vividness.shape}")
    logger.info(f"  mind_states: {output.mind_states.shape if output.mind_states is not None else None}")
    logger.info(f"  counterfactuals: {output.counterfactuals.shape if output.counterfactuals is not None else None}")
    logger.info(f"  imagination_mask: {output.imagination_mask.shape if output.imagination_mask is not None else None}")

    # Verify shapes
    assert output.imagined_features.shape == (B, S, D), f"Expected (B, S, D), got {output.imagined_features.shape}"
    assert output.vividness.shape == (B, S, 1)
    assert output.mind_states.shape == (B, S, D)
    assert output.counterfactuals.shape == (B, config.num_counterfactuals, S, D)
    assert output.imagination_mask.shape == (B, S, 1)

    # Vividness should be in [0, 1]
    assert output.vividness.min() >= 0, f"Vividness min {output.vividness.min():.3f} < 0"
    assert output.vividness.max() <= 1, f"Vividness max {output.vividness.max():.3f} > 1"

    logger.info(f"Vividness range: [{output.vividness.min():.3f}, {output.vividness.max():.3f}]")

    logger.info("Imagination forward pass test PASSED")
    return True


def test_scene_generator():
    """Test scene generation component."""
    from pem.imagination_module import SceneGenerator

    logger.info("\n" + "=" * 60)
    logger.info("TEST: SceneGenerator")
    logger.info("=" * 60)

    D = 64
    generator = SceneGenerator(d_model=D, hidden_dim=32, n_layers=2)
    generator.eval()

    B, S = 2, 16

    features = torch.randn(B, S, D)
    memory = torch.randn(B, S, D)
    personality = torch.randn(D)

    with torch.no_grad():
        imagined, vividness = generator(features, memory=memory, personality=personality)

    logger.info(f"Input shape: {features.shape}")
    logger.info(f"Imagined shape: {imagined.shape}")
    logger.info(f"Vividness shape: {vividness.shape}")

    assert imagined.shape == features.shape
    assert vividness.shape == (B, S, 1)

    # Imagined should be different from input
    diff = (imagined - features).abs().mean()
    logger.info(f"Difference from input: {diff:.4f}")
    assert diff > 0, "Imagined features should differ from input"

    logger.info("SceneGenerator test PASSED")
    return True


def test_mind_modeler():
    """Test theory of mind component."""
    from pem.imagination_module import MindModeler

    logger.info("\n" + "=" * 60)
    logger.info("TEST: MindModeler (Theory of Mind)")
    logger.info("=" * 60)

    D = 64
    modeler = MindModeler(d_model=D, hidden_dim=32, n_layers=2, max_entities=4)
    modeler.eval()

    B, S = 2, 16

    features = torch.randn(B, S, D)
    context = torch.randn(B, S, D)

    with torch.no_grad():
        mind_states = modeler(features, context=context)

    logger.info(f"Input shape: {features.shape}")
    logger.info(f"Mind states shape: {mind_states.shape}")

    assert mind_states.shape == features.shape

    # Mind states should be different from input
    diff = (mind_states - features).abs().mean()
    logger.info(f"Difference from input: {diff:.4f}")

    logger.info("MindModeler test PASSED")
    return True


def test_counterfactual_generator():
    """Test counterfactual generation component."""
    from pem.imagination_module import CounterfactualGenerator

    logger.info("\n" + "=" * 60)
    logger.info("TEST: CounterfactualGenerator")
    logger.info("=" * 60)

    D = 64
    num_cf = 3
    generator = CounterfactualGenerator(d_model=D, hidden_dim=32, num_counterfactuals=num_cf)
    generator.eval()

    B, S = 2, 16

    features = torch.randn(B, S, D)

    with torch.no_grad():
        counterfactuals, plausibilities = generator(features)

    logger.info(f"Input shape: {features.shape}")
    logger.info(f"Counterfactuals shape: {counterfactuals.shape}")
    logger.info(f"Plausibilities shape: {plausibilities.shape}")

    assert counterfactuals.shape == (B, num_cf, S, D)
    assert plausibilities.shape == (B, num_cf, S, 1)

    # Counterfactuals should be different from each other
    cf_diffs = []
    for i in range(num_cf):
        for j in range(i + 1, num_cf):
            diff = (counterfactuals[:, i] - counterfactuals[:, j]).abs().mean()
            cf_diffs.append(diff.item())

    logger.info(f"Mean counterfactual diversity: {sum(cf_diffs) / len(cf_diffs):.4f}")

    logger.info("CounterfactualGenerator test PASSED")
    return True


def test_trigger_detection():
    """Test imagination trigger detection."""
    from pem.imagination_module import ImaginationTriggerDetector

    logger.info("\n" + "=" * 60)
    logger.info("TEST: ImaginationTriggerDetector")
    logger.info("=" * 60)

    D = 64
    detector = ImaginationTriggerDetector(d_model=D, hidden_dim=32)
    detector.eval()

    B, S = 2, 16

    features = torch.randn(B, S, D)

    with torch.no_grad():
        trigger_probs = detector(features)

    logger.info(f"Input shape: {features.shape}")
    logger.info(f"Trigger probs shape: {trigger_probs.shape}")
    logger.info(f"Trigger probs range: [{trigger_probs.min():.3f}, {trigger_probs.max():.3f}]")

    assert trigger_probs.shape == (B, S, 1)
    assert trigger_probs.min() >= 0
    assert trigger_probs.max() <= 1

    logger.info("ImaginationTriggerDetector test PASSED")
    return True


def test_imagination_gradient_flow():
    """Test that gradients flow through imagination module."""
    from pem import ImaginationModule, ImaginationConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Imagination gradient flow")
    logger.info("=" * 60)

    config = ImaginationConfig(d_model=64, hidden_dim=32)
    module = ImaginationModule(config)
    module.train()

    B, S, D = 2, 16, config.d_model

    features = torch.randn(B, S, D, requires_grad=True)
    memory = torch.randn(B, S, D, requires_grad=True)

    # Forward pass
    output = module(features, memory=memory)

    # Backward pass
    loss = output.imagined_features.mean() + output.vividness.mean()
    loss.backward()

    # Check gradients
    has_feature_grads = features.grad is not None and features.grad.abs().sum() > 0
    has_memory_grads = memory.grad is not None and memory.grad.abs().sum() > 0
    module_has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    logger.info(f"Gradients flow to:")
    logger.info(f"  features: {has_feature_grads}")
    logger.info(f"  memory: {has_memory_grads}")
    logger.info(f"  module parameters: {module_has_grads}")

    assert module_has_grads, "Gradients should flow through module"

    logger.info("Imagination gradient flow test PASSED")
    return True


def test_imagination_loss():
    """Test imagination loss computation."""
    from pem import ImaginationModule, ImaginationConfig, ImaginationLoss

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Imagination loss")
    logger.info("=" * 60)

    config = ImaginationConfig(d_model=64, hidden_dim=32, num_counterfactuals=3)
    module = ImaginationModule(config)
    loss_fn = ImaginationLoss()

    B, S, D = 2, 16, config.d_model

    features = torch.randn(B, S, D)

    # Compute imagination
    output = module(features)

    # Compute loss
    total_loss, loss_dict = loss_fn(output, features)

    logger.info(f"Total loss: {total_loss.item():.4f}")
    for key, val in loss_dict.items():
        logger.info(f"  {key}: {val.item():.4f}")

    # Loss should be finite
    assert not torch.isnan(total_loss), "Loss should not be NaN"
    assert not torch.isinf(total_loss), "Loss should not be infinite"

    logger.info("Imagination loss test PASSED")
    return True


def test_imagination_factory():
    """Test imagination factory function."""
    from pem import create_imagination_module

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Imagination factory function")
    logger.info("=" * 60)

    module = create_imagination_module(
        d_model=256,
        hidden_dim=128,
        use_scene_generation=True,
        use_mind_modeling=True,
        use_counterfactuals=True,
    )

    logger.info(f"Created module: {type(module).__name__}")
    logger.info(f"Config: d_model={module.config.d_model}")

    # Quick forward test
    B, S, D = 1, 8, module.config.d_model
    features = torch.randn(B, S, D)

    with torch.no_grad():
        output = module(features)

    assert output.imagined_features.shape == (B, S, D)
    assert output.vividness.shape == (B, S, 1)

    logger.info("Imagination factory test PASSED")
    return True


def test_imagine_scenario():
    """Test explicit scenario imagination."""
    from pem import ImaginationModule, ImaginationConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Explicit scenario imagination")
    logger.info("=" * 60)

    config = ImaginationConfig(d_model=64, hidden_dim=32)
    module = ImaginationModule(config)
    module.eval()

    B, S, D = 1, 16, config.d_model

    features = torch.randn(B, S, D)
    scenario_prompt = torch.randn(B, S, D)

    with torch.no_grad():
        imagined = module.imagine_scenario(features, scenario_prompt=scenario_prompt)

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Imagined scenario shape: {imagined.shape}")

    assert imagined.shape == features.shape

    # Scenario prompt should influence output
    imagined_no_prompt = module.imagine_scenario(features)
    diff = (imagined - imagined_no_prompt).abs().mean()
    logger.info(f"Difference with/without prompt: {diff:.4f}")

    logger.info("Explicit scenario imagination test PASSED")
    return True


def test_infer_mind():
    """Test explicit mind inference."""
    from pem import ImaginationModule, ImaginationConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Explicit mind inference")
    logger.info("=" * 60)

    config = ImaginationConfig(d_model=64, hidden_dim=32, use_mind_modeling=True)
    module = ImaginationModule(config)
    module.eval()

    B, S, D = 1, 16, config.d_model

    features = torch.randn(B, S, D)
    entity_context = torch.randn(B, S, D)

    with torch.no_grad():
        mind_states = module.infer_mind(features, entity_context=entity_context)

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Mind states shape: {mind_states.shape}")

    assert mind_states.shape == features.shape

    logger.info("Explicit mind inference test PASSED")
    return True


def test_imagination_with_memory():
    """Test that memory enriches imagination."""
    from pem import ImaginationModule, ImaginationConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Memory-enriched imagination")
    logger.info("=" * 60)

    config = ImaginationConfig(d_model=64, hidden_dim=32, use_memory=True)
    module = ImaginationModule(config)
    module.eval()

    B, S, D = 1, 16, config.d_model

    features = torch.randn(B, S, D)
    memory = torch.randn(B, S, D)

    with torch.no_grad():
        output_with_memory = module(features, memory=memory)
        output_without_memory = module(features, memory=None)

    diff = (output_with_memory.imagined_features - output_without_memory.imagined_features).abs().mean()
    logger.info(f"Difference with/without memory: {diff:.4f}")

    # Memory should influence imagination
    assert diff > 0, "Memory should influence imagination"

    logger.info("Memory-enriched imagination test PASSED")
    return True


def test_imagination_with_personality():
    """Test that personality colors imagination."""
    from pem import ImaginationModule, ImaginationConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Personality-colored imagination")
    logger.info("=" * 60)

    config = ImaginationConfig(d_model=64, hidden_dim=32, use_personality=True)
    module = ImaginationModule(config)
    module.eval()

    B, S, D = 1, 16, config.d_model

    features = torch.randn(B, S, D)
    personality1 = torch.randn(D)
    personality2 = torch.randn(D) * 2  # Different personality

    with torch.no_grad():
        output1 = module(features, personality=personality1)
        output2 = module(features, personality=personality2)

    diff = (output1.imagined_features - output2.imagined_features).abs().mean()
    logger.info(f"Difference with different personalities: {diff:.4f}")

    logger.info("Personality-colored imagination test PASSED")
    return True


def test_imagination_minimal_config():
    """Test imagination with minimal config (only scene generation)."""
    from pem import ImaginationModule, ImaginationConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Minimal imagination config")
    logger.info("=" * 60)

    config = ImaginationConfig(
        d_model=64,
        hidden_dim=32,
        use_scene_generation=True,
        use_mind_modeling=False,
        use_counterfactuals=False,
    )
    module = ImaginationModule(config)
    module.eval()

    B, S, D = 1, 16, config.d_model
    features = torch.randn(B, S, D)

    with torch.no_grad():
        output = module(features)

    assert output.imagined_features.shape == (B, S, D)
    assert output.mind_states is None
    assert output.counterfactuals is None

    logger.info("Minimal imagination config test PASSED")
    return True


def main():
    """Run all tests."""
    logger.info("PEM Imagination (Mental Simulation) Module Tests")
    logger.info("=" * 60)

    tests = [
        ("Imagination forward pass", test_imagination_module_forward),
        ("Scene generator", test_scene_generator),
        ("Mind modeler (ToM)", test_mind_modeler),
        ("Counterfactual generator", test_counterfactual_generator),
        ("Trigger detection", test_trigger_detection),
        ("Imagination gradient flow", test_imagination_gradient_flow),
        ("Imagination loss", test_imagination_loss),
        ("Imagination factory", test_imagination_factory),
        ("Explicit scenario imagination", test_imagine_scenario),
        ("Explicit mind inference", test_infer_mind),
        ("Memory-enriched imagination", test_imagination_with_memory),
        ("Personality-colored imagination", test_imagination_with_personality),
        ("Minimal config", test_imagination_minimal_config),
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
