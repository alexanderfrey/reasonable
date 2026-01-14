#!/usr/bin/env python3
"""
Test script for Janus Pro Feature Extractor.

This is the first module to test - everything else depends on it.

Usage:
    python -m pem.test_janus_pro

Requirements:
    - torch
    - transformers
    - janus (from https://github.com/deepseek-ai/Janus)
"""

import sys
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    issues = []

    # Check torch
    try:
        logger.info(f"torch: {torch.__version__} ✓")
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}, Device: {torch.cuda.get_device_name(0)} ✓")
            # Check VRAM
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"VRAM: {total_mem:.1f} GB")
        else:
            logger.warning("CUDA not available - will run on CPU (slow)")
    except Exception as e:
        issues.append(f"torch issue: {e}")

    # Check transformers
    try:
        import transformers
        logger.info(f"transformers: {transformers.__version__} ✓")
    except ImportError:
        issues.append("transformers not installed")

    # Check janus
    try:
        from janus.models import VLChatProcessor, MultiModalityCausalLM
        logger.info("janus: available ✓")
    except ImportError:
        issues.append("janus not installed - clone from https://github.com/deepseek-ai/Janus")

    return issues


def test_lazy_loading():
    """Test that the extractor initializes without loading the model."""
    from pem import create_feature_extractor

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Lazy loading (no model download yet)")
    logger.info("=" * 60)

    # This should be fast - no model loading
    extractor = create_feature_extractor(
        model_name_or_path="deepseek-ai/Janus-Pro-1B",
        output_dim=1536,
        learning_mode="frozen",
    )

    logger.info(f"Created extractor: {type(extractor).__name__}")
    logger.info(f"Model loaded: {extractor._loaded}")

    assert not extractor._loaded, "Model should not be loaded yet"

    logger.info("Lazy loading test PASSED ✓")
    return True


def test_text_features():
    """Test feature extraction with text input."""
    from pem import create_feature_extractor

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Text feature extraction")
    logger.info("=" * 60)

    extractor = create_feature_extractor(
        model_name_or_path="deepseek-ai/Janus-Pro-1B",
        output_dim=1536,
        learning_mode="frozen",
    )

    # This triggers model loading
    logger.info("Loading model (first use)...")

    # Create simple tokenized input
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use the tokenizer to encode text
    extractor._ensure_loaded()
    tokenizer = extractor.tokenizer

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A neural network learns to predict the next token.",
    ]

    # Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Input tokens (first): {tokenizer.decode(input_ids[0][:20])}")

    # Extract features
    with torch.no_grad():
        features = extractor(input_ids, attention_mask=attention_mask)

    logger.info(f"Output shape: {features.shape}")
    logger.info(f"Output dtype: {features.dtype}")
    logger.info(f"Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")

    # Verify output shape
    B, S, D = features.shape
    assert B == len(texts), f"Batch mismatch: {B} vs {len(texts)}"
    assert D == 1536, f"Output dim mismatch: {D} vs 1536"

    logger.info("Text feature extraction PASSED ✓")
    return True


def test_imagination():
    """Test text-to-image generation (imagination)."""
    from pem import create_feature_extractor
    import os

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Text-to-image imagination")
    logger.info("=" * 60)

    extractor = create_feature_extractor(
        model_name_or_path="deepseek-ai/Janus-Pro-1B",
        output_dim=1536,
        learning_mode="frozen",
    )

    prompt = "A cat sitting on a windowsill watching rain"
    logger.info(f"Prompt: {prompt}")

    # Generate features (not images)
    logger.info("Generating imagined features...")
    with torch.no_grad():
        features = extractor.imagine(
            prompt=prompt,
            num_images=1,
            return_features=True,
        )

    logger.info(f"Imagined features shape: {features.shape}")
    logger.info(f"Feature stats: min={features.min():.4f}, max={features.max():.4f}")

    # Optionally generate actual images
    logger.info("\nGenerating actual images...")
    with torch.no_grad():
        images = extractor.imagine(
            prompt=prompt,
            num_images=2,
            return_features=False,
            cfg_weight=5.0,
        )

    logger.info(f"Generated images shape: {images.shape}")
    logger.info(f"Image dtype: {images.dtype}, range: [{images.min()}, {images.max()}]")

    # Save a sample image
    output_dir = "/tmp/pem_test_images"
    os.makedirs(output_dir, exist_ok=True)

    from PIL import Image
    img = Image.fromarray(images[0])
    img_path = os.path.join(output_dir, "test_imagination.jpg")
    img.save(img_path)
    logger.info(f"Saved test image to: {img_path}")

    logger.info("Imagination test PASSED ✓")
    return True


def test_imagine_from_features():
    """Test feature-conditioned imagination (for PEM loop)."""
    from pem import create_feature_extractor

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Imagine from features (PEM integration)")
    logger.info("=" * 60)

    extractor = create_feature_extractor(
        model_name_or_path="deepseek-ai/Janus-Pro-1B",
        output_dim=1536,
        learning_mode="frozen",
    )

    # First get some features from text
    extractor._ensure_loaded()
    tokenizer = extractor.tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = "The dark forest was filled with mysterious shadows."
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    with torch.no_grad():
        features = extractor(input_ids)

    logger.info(f"Input features shape: {features.shape}")

    # Now imagine from those features
    with torch.no_grad():
        imagined = extractor.imagine_from_features(features, mode="scene")

    logger.info(f"Imagined features shape: {imagined.shape}")
    logger.info(f"Same shape as input: {imagined.shape == features.shape}")

    assert imagined.shape == features.shape, "Imagined features should have same shape"

    logger.info("Imagine from features test PASSED ✓")
    return True


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Janus Pro Feature Extractor Tests")
    logger.info("=" * 60)

    # Check dependencies first
    issues = check_dependencies()
    if issues:
        logger.error("\nDependency issues found:")
        for issue in issues:
            logger.error(f"  - {issue}")
        logger.error("\nFix dependencies before running tests.")
        sys.exit(1)

    # Run tests in order
    tests = [
        ("Lazy loading", test_lazy_loading),
        ("Text features", test_text_features),
        ("Imagination", test_imagination),
        ("Imagine from features", test_imagine_from_features),
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
