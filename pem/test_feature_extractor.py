#!/usr/bin/env python3
"""
Test script for PEM Feature Extractor.

Usage:
    python -m pem.test_feature_extractor

Requirements:
    - transformers >= 4.57.0
    - torch
    - qwen-vl-utils >= 0.0.14 (for image/video processing)
"""

import sys
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    issues = []

    # Check transformers version
    try:
        import transformers
        version = transformers.__version__
        major, minor = map(int, version.split(".")[:2])
        if major < 4 or (major == 4 and minor < 57):
            issues.append(f"transformers {version} < 4.57.0 - upgrade needed")
        else:
            logger.info(f"transformers: {version} ✓")
    except ImportError:
        issues.append("transformers not installed")

    # Check torch
    try:
        logger.info(f"torch: {torch.__version__} ✓")
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}, Device: {torch.cuda.get_device_name(0)} ✓")
        else:
            logger.warning("CUDA not available - will run on CPU (slow)")
    except Exception as e:
        issues.append(f"torch issue: {e}")

    # Check flash-attn (optional but recommended)
    try:
        from flash_attn import flash_attn_func
        logger.info("flash_attn: available ✓")
    except ImportError:
        logger.warning("flash_attn not installed - will use slower attention")

    return issues


def test_text_only():
    """Test feature extraction with text-only input."""
    from pem import Qwen3VLFeatureExtractor, FeatureExtractorConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Text-only feature extraction")
    logger.info("=" * 60)

    # Create feature extractor
    config = FeatureExtractorConfig(
        model_name_or_path="Qwen/Qwen3-VL-2B-Instruct",
        output_dim=1536,
        learning_mode="frozen",
        torch_dtype="bfloat16",
    )

    extractor = Qwen3VLFeatureExtractor(config)
    extractor.eval()

    # Test with simple tokenized input
    # Note: In production you'd use the processor, but for quick test we create dummy input
    device = next(extractor.parameters()).device

    # Create dummy input (simulating tokenized text)
    batch_size = 2
    seq_len = 32
    vocab_size = 151936  # Qwen3 vocab size

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)

    logger.info(f"Input shape: {input_ids.shape}")

    # Extract features
    with torch.no_grad():
        features = extractor(input_ids, attention_mask)

    logger.info(f"Output shape: {features.shape}")
    logger.info(f"Output dtype: {features.dtype}")
    logger.info(f"Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")

    # Verify output shape
    expected_shape = (batch_size, seq_len, config.output_dim)
    assert features.shape == expected_shape, f"Expected {expected_shape}, got {features.shape}"

    logger.info("Text-only test PASSED ✓")
    return True


def test_with_processor():
    """Test feature extraction using the full processor pipeline."""
    from pem import Qwen3VLFeatureExtractor, FeatureExtractorConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Feature extraction with processor")
    logger.info("=" * 60)

    config = FeatureExtractorConfig(
        model_name_or_path="Qwen/Qwen3-VL-2B-Instruct",
        output_dim=1536,
        learning_mode="frozen",
        torch_dtype="bfloat16",
    )

    extractor = Qwen3VLFeatureExtractor(config)
    extractor.eval()

    # Process text through the official processor
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A neural network learns from data.",
    ]

    # Use the convenience method
    inputs = extractor.process_inputs(texts=texts)

    logger.info(f"Processed input_ids shape: {inputs['input_ids'].shape}")

    # Move to device
    device = next(extractor.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Extract features
    with torch.no_grad():
        features = extractor(**inputs)

    logger.info(f"Output shape: {features.shape}")
    logger.info(f"Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")

    # Features should be (B, S, output_dim)
    B, S, D = features.shape
    assert B == len(texts), f"Batch size mismatch: {B} vs {len(texts)}"
    assert D == config.output_dim, f"Output dim mismatch: {D} vs {config.output_dim}"

    logger.info("Processor test PASSED ✓")
    return True


def test_learning_modes():
    """Test different learning modes and parameter groups."""
    from pem import Qwen3VLFeatureExtractor, FeatureExtractorConfig, LearningMode

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Learning modes")
    logger.info("=" * 60)

    config = FeatureExtractorConfig(
        model_name_or_path="Qwen/Qwen3-VL-2B-Instruct",
        output_dim=1536,
        learning_mode="frozen",
        torch_dtype="bfloat16",
    )

    extractor = Qwen3VLFeatureExtractor(config)

    # Test frozen mode
    logger.info("\nFrozen mode:")
    frozen_params = sum(p.requires_grad for p in extractor.parameters())
    total_params = sum(1 for p in extractor.parameters())
    logger.info(f"  Trainable params: {frozen_params}/{total_params}")

    # In frozen mode, only projection should be trainable
    proj_trainable = sum(p.requires_grad for p in extractor.projection.parameters()) if extractor.projection else 0
    logger.info(f"  Projection trainable: {proj_trainable}")

    param_groups = extractor.get_param_groups(base_lr=1e-4)
    logger.info(f"  Param groups for optimizer: {len(param_groups)}")

    # Test unfreezing
    logger.info("\nUnfrozen (slow learning) mode:")
    extractor.unfreeze(LearningMode.SLOW_LEARNING)
    unfrozen_params = sum(p.requires_grad for p in extractor.parameters())
    logger.info(f"  Trainable params: {unfrozen_params}/{total_params}")

    param_groups = extractor.get_param_groups(base_lr=1e-4)
    logger.info(f"  Param groups: {len(param_groups)}")
    for i, group in enumerate(param_groups):
        logger.info(f"    Group {i}: lr={group['lr']}, params={len(list(group['params']))}")

    logger.info("Learning modes test PASSED ✓")
    return True


def test_factory_function():
    """Test the factory function."""
    from pem.feature_extractor import create_feature_extractor

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Factory function")
    logger.info("=" * 60)

    extractor = create_feature_extractor(
        model_name_or_path="Qwen/Qwen3-VL-2B-Instruct",
        output_dim=1024,  # Different output dim
        learning_mode="frozen",
    )

    logger.info(f"Created extractor: {type(extractor).__name__}")
    logger.info(f"Hidden size: {extractor.get_hidden_size()}")
    logger.info(f"Output dim: {extractor.config.output_dim}")

    # Verify projection was created for dimension mismatch
    assert extractor.projection is not None, "Projection should be created for dim mismatch"

    logger.info("Factory function test PASSED ✓")
    return True


def main():
    """Run all tests."""
    logger.info("PEM Feature Extractor Tests")
    logger.info("=" * 60)

    # Check dependencies first
    issues = check_dependencies()
    if issues:
        logger.error("Dependency issues found:")
        for issue in issues:
            logger.error(f"  - {issue}")
        logger.error("\nFix dependencies before running tests.")
        sys.exit(1)

    # Run tests
    tests = [
        ("Text-only", test_text_only),
        ("With processor", test_with_processor),
        ("Learning modes", test_learning_modes),
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
    success = main()
    sys.exit(0 if success else 1)
