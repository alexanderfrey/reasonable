#!/usr/bin/env python3
"""
Run the PEM pipeline end-to-end.

Usage:
    python -m pem.run_pipeline --text "Your input text here"
    python -m pem.run_pipeline --mock  # Use mock features (no GPU needed)
"""

import argparse
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_with_mock_features(text: str = "The quick brown fox jumps over the lazy dog."):
    """Run pipeline with mock features (no real model needed)."""
    from pem import (
        PredictionModule, PredictionConfig, PredictionTargets,
        SurpriseModule, SurpriseConfig,
    )

    logger.info("Running PEM pipeline with MOCK features")
    logger.info("=" * 60)

    # Config
    D = 256  # Smaller for testing
    sync_pairs = 64
    seq_len = 32  # Simulated sequence length

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

    # Mock inputs
    B = 1
    sync = torch.randn(B, seq_len, sync_pairs)  # Would come from CTM
    features = torch.randn(B, seq_len, D)       # Would come from FeatureExtractor

    logger.info(f"Input: '{text[:50]}...' (mocked as {seq_len} tokens)")
    logger.info(f"Sync shape: {sync.shape}")
    logger.info(f"Features shape: {features.shape}")

    # Run pipeline
    with torch.no_grad():
        # 1. Generate predictions from sync
        predictions = pred_module(sync, features)
        logger.info(f"\nPredictions generated:")
        for scale, pred in predictions.items():
            logger.info(f"  {scale}: {pred.shape}")

        # 2. Compute targets from actual features
        targets = target_computer.compute_targets_efficient(features)
        logger.info(f"\nTargets computed:")
        for key, val in targets.items():
            if isinstance(val, torch.Tensor):
                logger.info(f"  {key}: {val.shape}")

        # 3. Compute surprise
        surprises = surprise_module(predictions, targets, features)
        logger.info(f"\nSurprise computed:")
        for scale, data in surprises.items():
            mag_mean = data['magnitude'].mean().item()
            raw_mean = data['raw'].mean().item()
            logger.info(f"  {scale}: magnitude={mag_mean:.3f}, raw={raw_mean:.3f}")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    return predictions, targets, surprises


def run_with_real_features(text: str, device: str = "cuda", model: str = "showlab/show-o2-1.5B"):
    """Run pipeline with real features from Show-o2 (or legacy Qwen3-VL)."""
    from pem import (
        create_feature_extractor,
        PredictionModule, PredictionConfig, PredictionTargets,
        SurpriseModule, SurpriseConfig,
    )

    logger.info("Running PEM pipeline with REAL features")
    logger.info("=" * 60)

    # Load feature extractor (Show-o2 by default)
    logger.info(f"Loading feature extractor: {model}...")
    feature_extractor = create_feature_extractor(
        model_name_or_path=model,
        learning_mode="frozen",
        device_map=device,  # Put entire model on specified device
    )
    D = feature_extractor.config.output_dim
    logger.info(f"Feature extractor loaded, output_dim={D}")

    # Config - match feature extractor dimensions
    sync_pairs = 512  # Realistic CTM size
    pred_config = PredictionConfig(
        sync_pairs=sync_pairs,
        d_model=D,
        n_head=8,
        immediate_horizon=8,
        shortterm_horizon=64,
        longterm_horizon=256,
    )
    pred_module = PredictionModule(pred_config).to(device)

    target_computer = PredictionTargets(
        immediate_horizon=pred_config.immediate_horizon,
        shortterm_horizon=pred_config.shortterm_horizon,
        longterm_horizon=pred_config.longterm_horizon,
    )

    surprise_config = SurpriseConfig(d_model=D, hidden_dim=D * 2)
    surprise_module = SurpriseModule(surprise_config).to(device)

    # Extract real features
    logger.info(f"\nProcessing text: '{text[:100]}...'")

    # First, process text into tensors
    inputs = feature_extractor.process_inputs(texts=[text])
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logger.info(f"Tokenized: {inputs['input_ids'].shape}")

    # Extract features
    with torch.no_grad():
        features = feature_extractor(**inputs)
    seq_len = features.shape[1]
    logger.info(f"Extracted features: {features.shape} ({seq_len} tokens)")

    # Convert features to float32 for compatibility with other modules
    features = features.float()

    # Create dummy sync (would come from CTM in real usage)
    B = features.shape[0]
    sync = torch.randn(B, seq_len, sync_pairs, device=device)
    logger.info(f"Using dummy sync: {sync.shape}")

    # Run pipeline
    with torch.no_grad():
        # 1. Generate predictions from sync
        predictions = pred_module(sync, features)
        logger.info(f"\nPredictions generated:")
        for scale, pred in predictions.items():
            logger.info(f"  {scale}: {pred.shape}")

        # 2. Compute targets from actual features
        targets = target_computer.compute_targets_efficient(features)
        logger.info(f"\nTargets computed:")
        for key, val in targets.items():
            if isinstance(val, torch.Tensor):
                logger.info(f"  {key}: {val.shape}")

        # 3. Compute surprise
        surprises = surprise_module(predictions, targets, features)
        logger.info(f"\nSurprise computed:")
        for scale, data in surprises.items():
            mag_mean = data['magnitude'].mean().item()
            raw_mean = data['raw'].mean().item()
            dir_norm = data['direction'].norm(dim=-1).mean().item()
            logger.info(f"  {scale}: magnitude={mag_mean:.3f}, raw={raw_mean:.3f}, dir_norm={dir_norm:.3f}")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    return predictions, targets, surprises


def main():
    parser = argparse.ArgumentParser(description="Run PEM pipeline")
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog. It was a sunny day in the forest.")
    parser.add_argument("--mock", action="store_true", help="Use mock features instead of real model")
    parser.add_argument("--device", type=str, default="cuda", help="Device for real features")
    parser.add_argument("--model", type=str, default="showlab/show-o2-1.5B",
                       help="Model to use (default: showlab/show-o2-1.5B, legacy: Qwen/Qwen3-VL-2B-Instruct)")
    args = parser.parse_args()

    if args.mock:
        run_with_mock_features(args.text)
    else:
        run_with_real_features(args.text, args.device, args.model)


if __name__ == "__main__":
    main()
