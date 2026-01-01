"""
Test meta-surprise on real training data.

Quick experiment: can the system learn to predict its own surprise
when processing real text (not random data)?

Usage:
    python test_meta_surprise_real.py --max_steps 500
"""

import argparse
import json
import logging
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiential import (
    ExperientialStream,
    combined_experiential_loss,
    prediction_accuracy,
)
from train_experiential import (
    HiddenStateExtractor,
    load_model,
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_with_meta_surprise(
    extractor: HiddenStateExtractor,
    experiential: ExperientialStream,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int = 500,
    log_interval: int = 50,
    meta_weight: float = 0.5,
):
    """
    Train experiential module with meta-surprise tracking.
    """
    extractor.model.eval()
    for param in extractor.model.parameters():
        param.requires_grad = False

    experiential.train()

    history = {
        'exp_loss': [],
        'meta_loss': [],
        'meta_surprise': [],
        'accuracy': [],
        'surprise': [],
    }

    start_time = time.time()
    step = 0

    pbar = tqdm(total=max_steps, desc="Training")

    for batch in dataloader:
        if step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)

        # Reset state for each batch
        experiential.reset_state(batch_size=input_ids.size(0))

        # Get hidden states from frozen backbone
        with torch.no_grad():
            _, hidden_states = extractor(input_ids)

        # Forward through experiential module
        exp_output = experiential(hidden_states)

        # Combined loss (world prediction + self prediction)
        loss, loss_dict = combined_experiential_loss(
            exp_output,
            exp_weight=1.0,
            meta_weight=meta_weight
        )

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(experiential.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        acc = prediction_accuracy(exp_output['prediction'], exp_output['target'])

        history['exp_loss'].append(loss_dict['exp_loss'])
        history['meta_loss'].append(loss_dict.get('meta_loss', 0))
        history['meta_surprise'].append(loss_dict.get('mean_meta_surprise', 0))
        history['accuracy'].append(acc)
        history['surprise'].append(exp_output['surprise'].mean().item())

        # Log
        if step % log_interval == 0:
            pbar.set_postfix({
                'exp': f"{loss_dict['exp_loss']:.3f}",
                'meta': f"{loss_dict.get('meta_loss', 0):.4f}",
                'ms': f"{loss_dict.get('mean_meta_surprise', 0):.3f}",
                'acc': f"{acc:.3f}",
            })

        pbar.update(1)
        step += 1

    pbar.close()
    elapsed = time.time() - start_time
    logger.info(f"Training complete: {step} steps in {elapsed:.1f}s")

    return history


def analyze_results(history: dict):
    """Analyze and report training results."""
    n = len(history['meta_surprise'])

    # Split into early/late for comparison
    early = n // 5
    late_start = n - n // 5

    early_meta = sum(history['meta_surprise'][:early]) / early
    late_meta = sum(history['meta_surprise'][late_start:]) / (n - late_start)

    early_exp = sum(history['exp_loss'][:early]) / early
    late_exp = sum(history['exp_loss'][late_start:]) / (n - late_start)

    early_acc = sum(history['accuracy'][:early]) / early
    late_acc = sum(history['accuracy'][late_start:]) / (n - late_start)

    early_surprise = sum(history['surprise'][:early]) / early
    late_surprise = sum(history['surprise'][late_start:]) / (n - late_start)

    print("\n" + "=" * 60)
    print("META-SURPRISE ANALYSIS ON REAL DATA")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Early':>12} {'Late':>12} {'Change':>12}")
    print("-" * 60)
    print(f"{'Meta-surprise':<25} {early_meta:>12.4f} {late_meta:>12.4f} {late_meta - early_meta:>+12.4f}")
    print(f"{'Experiential loss':<25} {early_exp:>12.4f} {late_exp:>12.4f} {late_exp - early_exp:>+12.4f}")
    print(f"{'Prediction accuracy':<25} {early_acc:>12.4f} {late_acc:>12.4f} {late_acc - early_acc:>+12.4f}")
    print(f"{'Mean surprise':<25} {early_surprise:>12.4f} {late_surprise:>12.4f} {late_surprise - early_surprise:>+12.4f}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if late_meta < early_meta:
        reduction = (early_meta - late_meta) / early_meta * 100
        print(f"\n✓ Meta-surprise DECREASED by {reduction:.1f}%")
        print("  → System learned to predict its own surprise!")
        print("  → This is evidence of self-calibration.")
    else:
        print("\n✗ Meta-surprise did not decrease")
        print("  → System may need more training or different hyperparameters")

    if late_exp < early_exp:
        print(f"\n✓ Experiential loss decreased")
        print("  → System is learning to predict the world")

    if late_acc > early_acc:
        print(f"\n✓ Prediction accuracy improved")

    print("\n" + "=" * 60)

    return {
        'early_meta': early_meta,
        'late_meta': late_meta,
        'meta_reduction': (early_meta - late_meta) / early_meta if early_meta > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Test meta-surprise on real data")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--meta_weight", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    extractor = HiddenStateExtractor(model)

    # Create experiential module with meta-surprise
    experiential = ExperientialStream(
        d_model=config.d_model,
        use_meta_surprise=True,
        use_affect=True,
        use_persistent_state=True
    ).to(device)

    n_params = sum(p.numel() for p in experiential.parameters())
    logger.info(f"ExperientialStream: {n_params:,} parameters")

    # Load data
    from pretrain import PretokenizedDataset
    import glob

    meta_files = glob.glob(os.path.join(args.data_dir, "training*_metadata.json"))
    if not meta_files:
        raise FileNotFoundError(f"No training metadata found in {args.data_dir}")

    with open(meta_files[0]) as f:
        meta = json.load(f)

    token_file = meta.get('token_file') or meta_files[0].replace('_metadata.json', '_tokens.bin')
    num_examples = min(meta.get('num_examples', 10000), args.max_steps * args.batch_size * 2)

    logger.info(f"Loading data from {token_file}")
    dataset = PretokenizedDataset(
        token_file, num_examples, args.seq_len, args.seq_len, data_type="Train"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    logger.info(f"Loaded {len(dataset)} examples")

    # Optimizer
    optimizer = torch.optim.AdamW(
        experiential.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    # Train
    logger.info(f"\nTraining with meta_weight={args.meta_weight}")
    history = train_with_meta_surprise(
        extractor=extractor,
        experiential=experiential,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        max_steps=args.max_steps,
        log_interval=50,
        meta_weight=args.meta_weight,
    )

    # Analyze
    results = analyze_results(history)

    # Save
    output_path = "meta_surprise_results.pt"
    torch.save({
        'history': history,
        'results': results,
        'args': vars(args),
    }, output_path)
    logger.info(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
