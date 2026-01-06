"""
Evaluate whether memory retrieval improves prediction.

Compares perplexity with and without memory retrieval to measure
the actual benefit of the episodic memory system.

This evaluation has two phases:
1. Memory Building Phase: Run through training data with crystallize=True
   to accumulate episodic memories.
2. Evaluation Phase: Compare perplexity on evaluation data with and without
   using the accumulated memories.

Usage:
    python eval_memory_benefit.py --checkpoint memory_augmented_books_v2/memory_gpt_epoch_1.pt
    python eval_memory_benefit.py --checkpoint ckpt.pt --run_controls --resume_eval
"""

import argparse
import json
import logging
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiential import MemoryAugmentedGPT, Episode
from model import GPT, GPTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device) -> MemoryAugmentedGPT:
    """Load a memory-augmented GPT model."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract config
    config_dict = checkpoint.get('config', {})
    args = checkpoint.get('args', {})

    # Build GPT config
    config = GPTConfig(
        vocab_size=config_dict.get('vocab_size', 128256),
        d_model=config_dict.get('d_model', 1024),
        n_head=config_dict.get('n_head', 8),
        n_layer=config_dict.get('n_layer', 12),
        max_seq_len=config_dict.get('max_seq_len', 1024),
        n_kv_head=config_dict.get('n_kv_head', 4),
        d_ff=config_dict.get('d_ff', 4096),
    )

    logger.info(f"Model config: d_model={config.d_model}, n_layer={config.n_layer}")

    # Build model
    gpt = GPT(config)

    # Get memory params from checkpoint args
    memory_capacity = args.get('memory_capacity', 2000) if isinstance(args, dict) else 2000
    threshold = args.get('crystallization_threshold', 1.0) if isinstance(args, dict) else 1.0
    integration = args.get('integration', 'gated') if isinstance(args, dict) else 'gated'

    memory_gpt = MemoryAugmentedGPT(
        gpt,
        memory_capacity=memory_capacity,
        crystallization_threshold=threshold,
        memory_integration=integration,
        use_experiential=True,
    ).to(device)

    # Load weights
    if 'memory_gpt_state_dict' in checkpoint:
        memory_gpt.load_state_dict(checkpoint['memory_gpt_state_dict'], strict=False)
    else:
        raise KeyError("Checkpoint missing 'memory_gpt_state_dict'")

    logger.info(f"Memory size: {memory_gpt.memory.size}")

    return memory_gpt


def create_dataloader(
    data_dir: str,
    batch_size: int,
    seq_len: int,
    split: str = "evaluation",
    num_workers: int = 2
):
    """Create dataloader from pretokenized data."""
    from pretrain import PretokenizedDataset
    import glob

    pattern = os.path.join(data_dir, f"{split}*_metadata.json")
    meta_files = glob.glob(pattern)
    # Exclude document-only metadata files
    meta_files = [p for p in meta_files if "doc_metadata" not in os.path.basename(p)]

    if not meta_files:
        raise FileNotFoundError(f"No metadata found matching {pattern}")

    meta_files.sort()
    meta = None
    for path in meta_files:
        with open(path) as f:
            candidate = json.load(f)
        if 'num_examples' in candidate:
            meta = candidate
            break
    if meta is None:
        raise KeyError(f"No split metadata with num_examples found for {pattern}")

    token_file = meta.get('token_file')
    if token_file and not os.path.isabs(token_file):
        token_file = os.path.join(data_dir, os.path.basename(token_file))

    num_examples = meta['num_examples']
    logger.info(f"Loading {split} data: {num_examples:,} examples")

    dataset = PretokenizedDataset(
        token_file, num_examples, seq_len, seq_len, data_type=split.capitalize()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic for fair comparison
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


def snapshot_memory(model: MemoryAugmentedGPT) -> Dict[str, object]:
    """Capture episodic memory contents for later restoration."""
    episodes = []
    for ep in model.memory.episodes:
        episodes.append(Episode(
            timestamp=ep.timestamp,
            content=ep.content.detach().clone(),
            context=ep.context.detach().clone(),
            salience=ep.salience,
            valence=ep.valence,
            arousal=ep.arousal,
            retrieval_count=ep.retrieval_count,
            text=ep.text,
            token_ids=list(ep.token_ids) if ep.token_ids is not None else None,
            hint_text=ep.hint_text,
            hint_token_ids=list(ep.hint_token_ids) if ep.hint_token_ids is not None else None,
        ))
    return {
        'episodes': episodes,
        'global_step': model.memory._global_step
    }


def restore_memory(model: MemoryAugmentedGPT, snapshot: Dict[str, object]) -> None:
    """Restore episodic memory from a snapshot."""
    model.memory.episodes = snapshot.get('episodes', [])
    model.memory._global_step = snapshot.get('global_step', 0)


def apply_memory_shuffle(model: MemoryAugmentedGPT, seed: int) -> None:
    """Shuffle episodic contents/contexts as a negative control."""
    if model.memory.size < 2:
        return
    rng = random.Random(seed)
    indices = list(range(model.memory.size))
    rng.shuffle(indices)
    shuffled_contents = [model.memory.episodes[i].content for i in indices]
    shuffled_contexts = [model.memory.episodes[i].context for i in indices]
    for ep, content, context in zip(model.memory.episodes, shuffled_contents, shuffled_contexts):
        ep.content = content
        ep.context = context


def apply_memory_random(model: MemoryAugmentedGPT, seed: int) -> None:
    """Replace episodic contents/contexts with random noise."""
    if model.memory.size == 0:
        return
    device = model.memory.episodes[0].content.device
    torch.manual_seed(seed)
    for ep in model.memory.episodes:
        ep.content = torch.randn(ep.content.shape, device=device, dtype=ep.content.dtype)
        ep.context = torch.randn(ep.context.shape, device=device, dtype=ep.context.dtype)


def bootstrap_mean_ci(
    samples: List[float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Bootstrap mean with (1-alpha) CI."""
    if not samples:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(samples)
    means = []
    for _ in range(n_boot):
        boot = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(boot) / n)
    means.sort()
    lower = means[int((alpha / 2) * n_boot)]
    upper = means[int((1 - alpha / 2) * n_boot) - 1]
    return sum(samples) / n, lower, upper


def summarize_step_deltas(
    step_losses_with: List[float],
    step_losses_without: List[float],
    step_weights: List[float],
    n_boot: int,
    seed: int
) -> Dict[str, object]:
    """Summarize per-step loss deltas and retrieval-strength bins."""
    n = min(len(step_losses_with), len(step_losses_without), len(step_weights))
    deltas = [step_losses_with[i] - step_losses_without[i] for i in range(n)]
    mean_delta, ci_low, ci_high = bootstrap_mean_ci(deltas, n_boot=n_boot, seed=seed)

    bins = [0.0, 0.05, 0.15, 0.3, 1.0]
    bin_stats = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        indices = [i for i in range(n) if lo <= step_weights[i] < hi]
        if indices:
            bin_delta = sum(deltas[i] for i in indices) / len(indices)
        else:
            bin_delta = 0.0
        bin_stats.append({
            'bin': f"[{lo:.2f}, {hi:.2f})",
            'count': len(indices),
            'mean_delta_loss': bin_delta
        })

    return {
        'mean_delta_loss': mean_delta,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'bins': bin_stats
    }


def evaluate_perplexity(
    model: MemoryAugmentedGPT,
    dataloader: DataLoader,
    device: torch.device,
    use_memory: bool,
    max_steps: Optional[int] = None,
    desc: str = "Evaluating",
    return_steps: bool = False
) -> Dict[str, float]:
    """Evaluate perplexity with or without memory.

    For causal memory retrieval, we chain prev_memory_query between batches:
    - Each forward pass returns 'next_memory_query'
    - This is passed as 'prev_memory_query' to the next forward pass
    - Retrieval happens on batch 2+ (after first batch generates a query)
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    n_steps = 0
    n_retrievals = 0
    step_losses = []
    step_weights = []

    # Chain prev_memory_query between batches for causal retrieval
    prev_memory_query = None

    # Reset experiential state at start
    if model.experiential is not None:
        model.experiential.reset_state(batch_size=dataloader.batch_size)

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc)
        for step, batch in enumerate(pbar):
            if max_steps and step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()

            # Forward pass with chained memory query
            logits, hidden, mem_out = model(
                input_ids,
                crystallize=False,  # Don't modify memory during eval
                use_memory=use_memory,
                return_memory_weights=True,
                prev_memory_query=prev_memory_query  # Chain from previous batch
            )

            # Update prev_memory_query for next batch
            if 'next_memory_query' in mem_out:
                prev_memory_query = mem_out['next_memory_query']

            # Track if retrieval happened
            max_weight = 0.0
            if 'episodic_weights' in mem_out and mem_out['episodic_weights'] is not None:
                if mem_out['episodic_weights'].numel() > 0:
                    n_retrievals += 1
                    max_weight = mem_out['episodic_weights'].max().item()

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='sum'
            )

            batch_tokens = targets.numel()
            total_loss += loss.item()
            total_tokens += batch_tokens
            n_steps += 1
            if return_steps:
                step_losses.append(loss.item() / batch_tokens)
                step_weights.append(max_weight)

            # Update progress bar
            current_ppl = math.exp(total_loss / total_tokens)
            pbar.set_postfix({'ppl': f'{current_ppl:.2f}', 'ret': n_retrievals})

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'tokens': total_tokens,
        'steps': n_steps,
        'steps_with_retrieval': n_retrievals,
        'step_losses': step_losses if return_steps else None,
        'step_weights': step_weights if return_steps else None
    }


def build_memories(
    model: MemoryAugmentedGPT,
    dataloader: DataLoader,
    device: torch.device,
    max_steps: int = 500,
    desc: str = "Building memories"
) -> Dict[str, float]:
    """Build episodic memories by running through data with crystallize=True.

    Chain prev_memory_query between batches for causal retrieval during building.
    """
    model.eval()

    initial_memory_size = model.memory.size
    n_crystallized = 0
    n_steps = 0
    total_salience = 0.0

    # Chain prev_memory_query between batches
    prev_memory_query = None

    # Reset experiential state at start
    if model.experiential is not None:
        model.experiential.reset_state(batch_size=dataloader.batch_size)

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc)
        for step, batch in enumerate(pbar):
            if max_steps and step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            input_ids = input_ids[:, :-1].contiguous()

            prev_size = model.memory.size

            # Forward pass WITH crystallization and chained query
            logits, hidden, mem_out = model(
                input_ids,
                crystallize=True,  # Build memories!
                use_memory=True,
                prev_memory_query=prev_memory_query
            )

            # Update prev_memory_query for next batch
            if 'next_memory_query' in mem_out:
                prev_memory_query = mem_out['next_memory_query']

            # Track crystallization
            new_memories = model.memory.size - prev_size
            n_crystallized += new_memories
            n_steps += 1

            if 'salience' in mem_out and mem_out['salience'] is not None:
                total_salience += mem_out['salience'].mean().item()

            # Update progress bar
            pbar.set_postfix({
                'memories': model.memory.size,
                'new': n_crystallized
            })

    return {
        'initial_size': initial_memory_size,
        'final_size': model.memory.size,
        'memories_added': n_crystallized,
        'steps': n_steps,
        'avg_salience': total_salience / max(n_steps, 1)
    }


def evaluate_retrieval_stats(
    model: MemoryAugmentedGPT,
    dataloader: DataLoader,
    device: torch.device,
    max_steps: int = 100
) -> Dict[str, float]:
    """Evaluate retrieval statistics.

    Chain prev_memory_query between batches for causal retrieval.
    """
    model.eval()

    total_retrieval_weight = 0.0
    total_semantic_weight = 0.0
    n_samples = 0
    n_with_retrieval = 0

    # Chain prev_memory_query between batches
    prev_memory_query = None

    # Reset state at start
    if model.experiential is not None:
        model.experiential.reset_state(batch_size=dataloader.batch_size)

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            input_ids = input_ids[:, :-1].contiguous()

            logits, hidden, mem_out = model(
                input_ids,
                crystallize=False,
                use_memory=True,
                return_memory_weights=True,
                prev_memory_query=prev_memory_query
            )

            # Update prev_memory_query for next batch
            if 'next_memory_query' in mem_out:
                prev_memory_query = mem_out['next_memory_query']

            n_samples += 1

            # Track retrieval weights
            if 'episodic_weights' in mem_out and mem_out['episodic_weights'] is not None:
                weights = mem_out['episodic_weights']
                if weights.numel() > 0:
                    max_weight = weights.max().item()
                    total_retrieval_weight += max_weight
                    if max_weight > 0.01:  # Significant retrieval
                        n_with_retrieval += 1

            if 'semantic_weights' in mem_out and mem_out['semantic_weights'] is not None:
                weights = mem_out['semantic_weights']
                if weights.numel() > 0:
                    total_semantic_weight += weights.max().item()

    return {
        'avg_max_episodic_weight': total_retrieval_weight / max(n_samples, 1),
        'avg_max_semantic_weight': total_semantic_weight / max(n_samples, 1),
        'samples_with_retrieval': n_with_retrieval,
        'total_samples': n_samples
    }


def evaluate_resume_interruption(
    model: MemoryAugmentedGPT,
    dataloader: DataLoader,
    device: torch.device,
    use_memory: bool,
    interrupt_ratio: float = 0.5,
    max_steps: Optional[int] = None,
    desc: str = "Resume eval",
    return_steps: bool = False
) -> Dict[str, float]:
    """Evaluate memory benefit after an interruption (reset hidden state)."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    n_steps = 0
    n_retrievals = 0
    step_losses = []
    step_weights = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc)
        for step, batch in enumerate(pbar):
            if max_steps and step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            seq_len = input_ids.size(1)
            split_idx = max(2, int(seq_len * interrupt_ratio))
            if split_idx >= seq_len - 1:
                continue

            # Reset experiential state per batch (no cross-batch hidden state)
            if model.experiential is not None:
                model.experiential.reset_state(batch_size=input_ids.size(0))

            prev_memory_query = None

            # Phase A: read first chunk to build memory + query
            first = input_ids[:, :split_idx]
            first_inputs = first[:, :-1].contiguous()
            _, _, mem_out = model(
                first_inputs,
                crystallize=True,
                use_memory=True,
                prev_memory_query=prev_memory_query
            )
            prev_memory_query = mem_out.get('next_memory_query')

            # Simulate interruption
            model.reset_hidden_state()

            # Phase B: evaluate second chunk
            second = input_ids[:, split_idx:]
            if second.size(1) < 2:
                continue
            second_inputs = second[:, :-1].contiguous()
            targets = second[:, 1:].contiguous()

            logits, _, mem_out = model(
                second_inputs,
                crystallize=False,
                use_memory=use_memory,
                return_memory_weights=True,
                prev_memory_query=prev_memory_query
            )

            max_weight = 0.0
            if 'episodic_weights' in mem_out and mem_out['episodic_weights'] is not None:
                if mem_out['episodic_weights'].numel() > 0:
                    n_retrievals += 1
                    max_weight = mem_out['episodic_weights'].max().item()

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='sum'
            )

            batch_tokens = targets.numel()
            total_loss += loss.item()
            total_tokens += batch_tokens
            n_steps += 1
            if return_steps:
                step_losses.append(loss.item() / batch_tokens)
                step_weights.append(max_weight)

            current_ppl = math.exp(total_loss / total_tokens)
            pbar.set_postfix({'ppl': f'{current_ppl:.2f}', 'ret': n_retrievals})

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if total_tokens > 0 else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'tokens': total_tokens,
        'steps': n_steps,
        'steps_with_retrieval': n_retrievals,
        'step_losses': step_losses if return_steps else None,
        'step_weights': step_weights if return_steps else None
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate memory retrieval benefit")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--data_dir", default="book_corpus_output", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--build_steps", type=int, default=500, help="Steps to build memories")
    parser.add_argument("--eval_steps", type=int, default=200, help="Steps to evaluate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--skip_build", action="store_true", help="Skip memory building (use existing)")
    parser.add_argument("--bootstrap_samples", type=int, default=1000, help="Bootstrap samples for CI")
    parser.add_argument("--bootstrap_seed", type=int, default=0, help="Seed for bootstrap")
    parser.add_argument("--run_controls", action="store_true", help="Run shuffled/random memory controls")
    parser.add_argument("--control_seed", type=int, default=0, help="Seed for control shuffles")
    parser.add_argument("--resume_eval", action="store_true", help="Run resume-after-interruption eval")
    parser.add_argument("--resume_ratio", type=float, default=0.5, help="Split ratio for interruption eval")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Load model
    model = load_model(args.checkpoint, device)

    print("\n" + "=" * 70)
    print("MEMORY RETRIEVAL BENEFIT EVALUATION")
    print("=" * 70)

    # Phase 1: Build memories from training data
    build_stats = None
    if not args.skip_build:
        print("\n[Phase 1] Building memories from training data...")
        try:
            train_dataloader = create_dataloader(
                args.data_dir,
                args.batch_size,
                args.seq_len,
                split="training",
                num_workers=args.num_workers
            )
            build_stats = build_memories(
                model, train_dataloader, device,
                max_steps=args.build_steps,
                desc="Building memories"
            )
            print(f"\n  Memories built: {build_stats['initial_size']} -> {build_stats['final_size']}")
            print(f"  New memories: {build_stats['memories_added']}")
            print(f"  Avg salience: {build_stats['avg_salience']:.4f}")
        except FileNotFoundError as e:
            logger.warning(f"Could not load training data: {e}")
            logger.warning("Skipping memory building phase")
    else:
        print("\n[Phase 1] Skipping memory building (--skip_build)")

    print(f"\n  Current memory size: {model.memory.size}")

    if model.memory.size == 0:
        print("\n" + "!" * 70)
        print("WARNING: No memories in store! Cannot evaluate retrieval benefit.")
        print("Run without --skip_build to build memories first.")
        print("!" * 70)

    # Phase 2: Load evaluation data
    print("\n[Phase 2] Loading evaluation data...")
    eval_dataloader = create_dataloader(
        args.data_dir,
        args.batch_size,
        args.seq_len,
        split="evaluation",
        num_workers=args.num_workers
    )

    # Phase 3: Evaluate WITH memory
    print("\n[Phase 3a] Evaluating WITH memory retrieval...")
    results_with = evaluate_perplexity(
        model, eval_dataloader, device,
        use_memory=True,
        max_steps=args.eval_steps,
        desc="With memory",
        return_steps=True
    )

    # Phase 3b: Evaluate WITHOUT memory
    print("\n[Phase 3b] Evaluating WITHOUT memory retrieval...")
    results_without = evaluate_perplexity(
        model, eval_dataloader, device,
        use_memory=False,
        max_steps=args.eval_steps,
        desc="Without memory",
        return_steps=True
    )

    # Phase 4: Retrieval statistics
    print("\n[Phase 4] Analyzing retrieval statistics...")
    retrieval_stats = evaluate_retrieval_stats(
        model, eval_dataloader, device,
        max_steps=min(args.eval_steps, 100)
    )

    # Compute benefit
    ppl_benefit = results_without['perplexity'] - results_with['perplexity']
    ppl_benefit_pct = 100 * (results_without['perplexity'] - results_with['perplexity']) / results_without['perplexity']
    loss_benefit = results_without['loss'] - results_with['loss']

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nMemory Status:")
    print(f"  Memory size:              {model.memory.size}")
    if build_stats:
        print(f"  Memories added:           {build_stats['memories_added']}")
        print(f"  Avg salience (build):     {build_stats['avg_salience']:.4f}")

    print(f"\nPerplexity:")
    print(f"  With memory:    {results_with['perplexity']:.3f}")
    print(f"  Without memory: {results_without['perplexity']:.3f}")
    print(f"  Benefit:        {ppl_benefit:+.3f} ({ppl_benefit_pct:+.2f}%)")

    print(f"\nCross-Entropy Loss:")
    print(f"  With memory:    {results_with['loss']:.4f}")
    print(f"  Without memory: {results_without['loss']:.4f}")
    print(f"  Benefit:        {loss_benefit:+.4f}")

    if results_with['step_losses'] and results_without['step_losses']:
        delta_summary = summarize_step_deltas(
            results_with['step_losses'],
            results_without['step_losses'],
            results_with['step_weights'],
            n_boot=args.bootstrap_samples,
            seed=args.bootstrap_seed
        )
        print("\nPer-step loss delta (with - without):")
        print(f"  Mean Δloss:     {delta_summary['mean_delta_loss']:.5f} "
              f"(95% CI: {delta_summary['ci_low']:.5f}, {delta_summary['ci_high']:.5f})")
        print("  By retrieval strength (max episodic weight):")
        for bin_stat in delta_summary['bins']:
            print(f"    {bin_stat['bin']}: n={bin_stat['count']}, "
                  f"mean Δloss={bin_stat['mean_delta_loss']:.5f}")
    else:
        delta_summary = None

    print(f"\nRetrieval Statistics:")
    print(f"  Avg max episodic weight:  {retrieval_stats['avg_max_episodic_weight']:.4f}")
    print(f"  Avg max semantic weight:  {retrieval_stats['avg_max_semantic_weight']:.4f}")
    print(f"  Samples w/ retrieval:     {retrieval_stats['samples_with_retrieval']}/{retrieval_stats['total_samples']}")

    print("\n" + "=" * 70)
    if model.memory.size == 0:
        print("○ INCONCLUSIVE: No memories to retrieve")
    elif ppl_benefit > 0:
        print(f"✓ Memory retrieval HELPS: {ppl_benefit:.3f} perplexity reduction ({ppl_benefit_pct:+.2f}%)")
    elif ppl_benefit < -0.1:
        print(f"✗ Memory retrieval HURTS: {-ppl_benefit:.3f} perplexity increase ({-ppl_benefit_pct:.2f}%)")
    else:
        print(f"○ Memory retrieval has NO SIGNIFICANT EFFECT (Δppl={ppl_benefit:+.3f})")
    print("=" * 70)

    # Optional negative controls
    control_results = None
    if args.run_controls and model.memory.size > 0:
        print("\n[Controls] Evaluating shuffled memory...")
        snapshot = snapshot_memory(model)
        apply_memory_shuffle(model, seed=args.control_seed)
        shuffled = evaluate_perplexity(
            model, eval_dataloader, device,
            use_memory=True,
            max_steps=args.eval_steps,
            desc="Shuffled memory",
            return_steps=False
        )
        restore_memory(model, snapshot)

        print("\n[Controls] Evaluating random memory...")
        apply_memory_random(model, seed=args.control_seed + 1)
        random_mem = evaluate_perplexity(
            model, eval_dataloader, device,
            use_memory=True,
            max_steps=args.eval_steps,
            desc="Random memory",
            return_steps=False
        )
        restore_memory(model, snapshot)

        control_results = {
            'shuffled': shuffled,
            'random': random_mem
        }
        print("\nControl summary:")
        print(f"  Shuffled PPL: {shuffled['perplexity']:.3f}")
        print(f"  Random PPL:   {random_mem['perplexity']:.3f}")

    # Optional resume-after-interruption evaluation
    resume_results = None
    if args.resume_eval:
        print("\n[Resume Eval] Evaluating WITH memory after interruption...")
        resume_snapshot = snapshot_memory(model)
        resume_with = evaluate_resume_interruption(
            model, eval_dataloader, device,
            use_memory=True,
            interrupt_ratio=args.resume_ratio,
            max_steps=args.eval_steps,
            desc="Resume (with memory)",
            return_steps=True
        )
        restore_memory(model, resume_snapshot)

        print("\n[Resume Eval] Evaluating WITHOUT memory after interruption...")
        resume_without = evaluate_resume_interruption(
            model, eval_dataloader, device,
            use_memory=False,
            interrupt_ratio=args.resume_ratio,
            max_steps=args.eval_steps,
            desc="Resume (without memory)",
            return_steps=True
        )
        restore_memory(model, resume_snapshot)

        resume_delta = summarize_step_deltas(
            resume_with['step_losses'],
            resume_without['step_losses'],
            resume_with['step_weights'],
            n_boot=args.bootstrap_samples,
            seed=args.bootstrap_seed + 1
        )
        resume_results = {
            'with_memory': resume_with,
            'without_memory': resume_without,
            'delta_summary': resume_delta
        }
        print("\nResume eval delta (with - without):")
        print(f"  Mean Δloss: {resume_delta['mean_delta_loss']:.5f} "
              f"(95% CI: {resume_delta['ci_low']:.5f}, {resume_delta['ci_high']:.5f})")

    # Save results
    if args.output:
        output_data = {
            'with_memory': results_with,
            'without_memory': results_without,
            'benefit': {
                'perplexity': ppl_benefit,
                'perplexity_pct': ppl_benefit_pct,
                'loss': loss_benefit
            },
            'delta_summary': delta_summary,
            'retrieval_stats': retrieval_stats,
            'control_results': control_results,
            'resume_results': resume_results,
            'memory_size': model.memory.size,
            'build_stats': build_stats,
            'args': vars(args)
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
