"""
Test how meta-surprise affects crystallization into episodic memory.

Hypothesis: Moments with high meta-surprise (self-ignorance) should be
more likely to be crystallized because salience is boosted.

Usage:
    python test_meta_surprise_crystallization.py --max_steps 500
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
    EpisodicMemory,
    combined_experiential_loss,
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


def run_crystallization_experiment(
    extractor: HiddenStateExtractor,
    experiential: ExperientialStream,
    memory: EpisodicMemory,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int = 500,
    log_interval: int = 50,
    meta_weight: float = 0.5,
):
    """
    Run experiment tracking crystallization patterns.
    """
    extractor.model.eval()
    for param in extractor.model.parameters():
        param.requires_grad = False

    experiential.train()

    # Track crystallization events
    crystallization_events = []

    history = {
        'meta_surprise': [],
        'salience': [],
        'crystallized': [],
        'memory_size': [],
    }

    start_time = time.time()
    step = 0

    pbar = tqdm(total=max_steps, desc="Training")

    for batch in dataloader:
        if step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        batch_size = input_ids.size(0)

        # Reset state for each batch
        experiential.reset_state(batch_size=batch_size)

        # Get hidden states from frozen backbone
        with torch.no_grad():
            _, hidden_states = extractor(input_ids)

        # Forward through experiential module
        exp_output = experiential(hidden_states)

        # Combined loss
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

        # Track metrics per sample
        meta_surp = exp_output['meta_surprise'].detach()
        salience = exp_output['salience'].detach()

        # Try to crystallize each sample
        crystallized_this_step = 0
        for i in range(batch_size):
            sal = salience[i].item()
            ms = meta_surp[i].item()

            if memory.should_crystallize(sal):
                memory.store(
                    content=exp_output['target'][i].detach(),
                    context=exp_output['prediction'][i].detach(),
                    salience=sal,
                    valence=exp_output['valence'][i].item(),
                    arousal=exp_output['arousal'][i].item()
                )
                crystallized_this_step += 1

                # Record this crystallization event
                crystallization_events.append({
                    'step': step,
                    'meta_surprise': ms,
                    'salience': sal,
                    'surprise': exp_output['surprise'][i].item(),
                })

        # Track batch averages
        history['meta_surprise'].append(meta_surp.mean().item())
        history['salience'].append(salience.mean().item())
        history['crystallized'].append(crystallized_this_step)
        history['memory_size'].append(memory.size)

        # Log
        if step % log_interval == 0:
            pbar.set_postfix({
                'ms': f"{meta_surp.mean():.3f}",
                'sal': f"{salience.mean():.4f}",
                'cryst': crystallized_this_step,
                'mem': memory.size,
            })

        pbar.update(1)
        step += 1

    pbar.close()
    elapsed = time.time() - start_time
    logger.info(f"Experiment complete: {step} steps in {elapsed:.1f}s")

    return history, crystallization_events


def analyze_crystallization(history: dict, events: list):
    """Analyze crystallization patterns."""

    print("\n" + "=" * 60)
    print("CRYSTALLIZATION ANALYSIS")
    print("=" * 60)

    total_crystallized = sum(history['crystallized'])
    final_memory_size = history['memory_size'][-1]

    print(f"\nTotal crystallization events: {total_crystallized}")
    print(f"Final memory size: {final_memory_size}")

    if not events:
        print("\nNo crystallization events to analyze.")
        return {}

    # Analyze meta-surprise of crystallized moments
    ms_values = [e['meta_surprise'] for e in events]
    sal_values = [e['salience'] for e in events]
    surp_values = [e['surprise'] for e in events]

    avg_ms = sum(ms_values) / len(ms_values)
    avg_sal = sum(sal_values) / len(sal_values)
    avg_surp = sum(surp_values) / len(surp_values)

    # Compare to overall averages
    overall_ms = sum(history['meta_surprise']) / len(history['meta_surprise'])
    overall_sal = sum(history['salience']) / len(history['salience'])

    print(f"\n{'Metric':<30} {'Crystallized':>15} {'Overall':>15} {'Ratio':>10}")
    print("-" * 70)
    print(f"{'Mean meta-surprise':<30} {avg_ms:>15.4f} {overall_ms:>15.4f} {avg_ms/overall_ms:>10.2f}x")
    print(f"{'Mean salience':<30} {avg_sal:>15.4f} {overall_sal:>15.4f} {avg_sal/overall_sal:>10.2f}x")
    print(f"{'Mean surprise':<30} {avg_surp:>15.4f} {'—':>15} {'—':>10}")

    # Split events by meta-surprise level
    sorted_events = sorted(events, key=lambda e: e['meta_surprise'])
    n = len(sorted_events)

    if n >= 10:
        low_ms = sorted_events[:n//3]
        high_ms = sorted_events[-n//3:]

        low_ms_avg = sum(e['meta_surprise'] for e in low_ms) / len(low_ms)
        high_ms_avg = sum(e['meta_surprise'] for e in high_ms) / len(high_ms)

        print(f"\n{'Distribution of crystallized events by meta-surprise:'}")
        print(f"  Low meta-surprise (bottom 1/3):  {len(low_ms)} events, avg ms={low_ms_avg:.4f}")
        print(f"  High meta-surprise (top 1/3):    {len(high_ms)} events, avg ms={high_ms_avg:.4f}")

    # Timeline analysis
    n_steps = len(history['crystallized'])
    early_cryst = sum(history['crystallized'][:n_steps//3])
    late_cryst = sum(history['crystallized'][-n_steps//3:])

    early_ms = sum(history['meta_surprise'][:n_steps//3]) / (n_steps//3)
    late_ms = sum(history['meta_surprise'][-n_steps//3:]) / (n_steps//3)

    print(f"\n{'Timeline:'}")
    print(f"  Early (first 1/3):  {early_cryst} crystallizations, avg meta-surprise={early_ms:.4f}")
    print(f"  Late (last 1/3):    {late_cryst} crystallizations, avg meta-surprise={late_ms:.4f}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if avg_ms > overall_ms:
        boost = (avg_ms / overall_ms - 1) * 100
        print(f"\n✓ Crystallized moments have {boost:.1f}% HIGHER meta-surprise than average")
        print("  → System is remembering moments of self-ignorance!")
        print("  → The salience boost from meta-surprise is working.")
    else:
        print("\n~ Crystallized moments have similar meta-surprise to average")
        print("  → Other factors (surprise, arousal, valence) may dominate salience")

    if late_cryst < early_cryst and late_ms < early_ms:
        print(f"\n✓ Crystallization decreased as meta-surprise decreased")
        print("  → System is learning about itself, fewer 'unknown' moments")

    print("\n" + "=" * 60)

    return {
        'total_crystallized': total_crystallized,
        'avg_ms_crystallized': avg_ms,
        'avg_ms_overall': overall_ms,
        'ms_boost_ratio': avg_ms / overall_ms if overall_ms > 0 else 1,
    }


def main():
    parser = argparse.ArgumentParser(description="Test meta-surprise crystallization")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--meta_weight", type=float, default=0.5)
    parser.add_argument("--crystallization_threshold", type=float, default=0.01,
                        help="Salience threshold for crystallization (lower = more memories)")
    parser.add_argument("--memory_capacity", type=int, default=1000,
                        help="Maximum episodic memory capacity")
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

    # Create episodic memory
    memory = EpisodicMemory(
        d_model=config.d_model,
        capacity=args.memory_capacity,
        crystallization_threshold=args.crystallization_threshold
    )

    n_params = sum(p.numel() for p in experiential.parameters())
    logger.info(f"ExperientialStream: {n_params:,} parameters")
    logger.info(f"Crystallization threshold: {args.crystallization_threshold}")
    logger.info(f"Memory capacity: {args.memory_capacity}")

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

    # Run experiment
    logger.info(f"\nRunning crystallization experiment...")
    history, events = run_crystallization_experiment(
        extractor=extractor,
        experiential=experiential,
        memory=memory,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        max_steps=args.max_steps,
        log_interval=50,
        meta_weight=args.meta_weight,
    )

    # Analyze
    results = analyze_crystallization(history, events)

    # Save
    output_path = "crystallization_results.pt"
    torch.save({
        'history': history,
        'events': events,
        'results': results,
        'args': vars(args),
    }, output_path)
    logger.info(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
