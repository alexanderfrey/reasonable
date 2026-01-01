"""
Analyze salience moments from memory-augmented training.

Shows:
1. Distribution of salience values
2. What content triggers high salience
3. Surprise patterns across text
"""

import argparse
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

from model import GPT, GPTConfig
from experiential import MemoryAugmentedGPT, EpisodicMemory


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load trained memory-augmented model."""
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Reconstruct config
    config_dict = ckpt.get('config', {})
    args = ckpt.get('args', {})

    config = GPTConfig(
        vocab_size=config_dict.get('vocab_size', 128256),
        d_model=config_dict.get('d_model', 1024),
        n_head=config_dict.get('n_head', 16),
        n_layer=config_dict.get('n_layer', 12),
        max_seq_len=config_dict.get('max_seq_len', 2048),
        n_kv_head=config_dict.get('n_kv_head'),
        d_ff=config_dict.get('d_ff'),
    )

    # Create model
    gpt = GPT(config)
    memory_gpt = MemoryAugmentedGPT(
        gpt,
        memory_capacity=args.get('memory_capacity', 1000),
        crystallization_threshold=args.get('crystallization_threshold', 0.2),
        memory_integration=args.get('integration', 'gated'),
    )

    # Load weights (strict=False to handle persistent state buffers)
    memory_gpt.load_state_dict(ckpt['memory_gpt_state_dict'], strict=False)
    memory_gpt = memory_gpt.to(device)
    memory_gpt.eval()

    return memory_gpt, config


def analyze_memory_stats(memory: EpisodicMemory):
    """Analyze the stored episodes."""
    if memory.size == 0:
        print("Memory is empty!")
        return

    episodes = memory.episodes

    # Extract stats
    saliences = [ep.salience for ep in episodes]
    valences = [ep.valence for ep in episodes]
    arousals = [ep.arousal for ep in episodes]
    retrieval_counts = [ep.retrieval_count for ep in episodes]

    print("\n" + "=" * 60)
    print("MEMORY STATISTICS")
    print("=" * 60)
    print(f"Total episodes: {len(episodes)}")
    print()

    print("Salience distribution:")
    print(f"  Min:    {min(saliences):.4f}")
    print(f"  Max:    {max(saliences):.4f}")
    print(f"  Mean:   {np.mean(saliences):.4f}")
    print(f"  Median: {np.median(saliences):.4f}")
    print(f"  Std:    {np.std(saliences):.4f}")

    # Histogram
    print("\n  Salience histogram:")
    hist, bins = np.histogram(saliences, bins=10, range=(0, 0.5))
    for i in range(len(hist)):
        bar = "#" * (hist[i] // 2)
        print(f"    [{bins[i]:.2f}-{bins[i+1]:.2f}]: {hist[i]:3d} {bar}")

    print("\nValence distribution:")
    print(f"  Min:    {min(valences):.4f}")
    print(f"  Max:    {max(valences):.4f}")
    print(f"  Mean:   {np.mean(valences):.4f}")

    print("\nArousal distribution:")
    print(f"  Min:    {min(arousals):.4f}")
    print(f"  Max:    {max(arousals):.4f}")
    print(f"  Mean:   {np.mean(arousals):.4f}")

    print("\nRetrieval counts:")
    print(f"  Total retrievals: {sum(retrieval_counts)}")
    print(f"  Max for single episode: {max(retrieval_counts)}")
    print(f"  Episodes never retrieved: {sum(1 for c in retrieval_counts if c == 0)}")

    # Top salient episodes
    print("\n" + "-" * 60)
    print("TOP 10 MOST SALIENT EPISODES:")
    print("-" * 60)
    sorted_eps = sorted(episodes, key=lambda x: x.salience, reverse=True)[:10]
    for i, ep in enumerate(sorted_eps):
        print(f"\n  #{i+1}: salience={ep.salience:.4f}, valence={ep.valence:.4f}, "
              f"arousal={ep.arousal:.4f}, retrievals={ep.retrieval_count}")
        print(f"       timestamp={ep.timestamp}, content_norm={ep.content.norm():.2f}")


def run_live_analysis(
    memory_gpt: MemoryAugmentedGPT,
    tokenizer,
    text: str,
    device: torch.device,
    show_tokens: bool = True
):
    """
    Run text through the model and show salience at each position.
    """
    print("\n" + "=" * 60)
    print("LIVE SALIENCE ANALYSIS")
    print("=" * 60)

    # Tokenize
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    seq_len = tokens.size(1)
    print(f"Input: {seq_len} tokens")

    if seq_len < 32:
        print("Text too short for meaningful analysis (need at least 32 tokens)")
        return

    # Clear memory for fresh analysis
    memory_gpt.reset_memory()

    # Process in chunks to see crystallization happen
    chunk_size = 64
    all_surprises = []
    all_saliences = []
    crystallization_points = []

    for start in range(0, seq_len - chunk_size, chunk_size // 2):
        end = min(start + chunk_size, seq_len)
        chunk = tokens[:, start:end]

        with torch.no_grad():
            logits, hidden, mem_out = memory_gpt(
                chunk,
                crystallize=True,
                use_memory=True
            )

        if mem_out.get('surprise') is not None:
            surprise = mem_out['surprise'].item()
            salience = mem_out['salience'].item() if mem_out.get('salience') is not None else 0
            all_surprises.append((start, end, surprise))
            all_saliences.append((start, end, salience))

            if mem_out.get('crystallized'):
                crystallization_points.append((start, end, salience))

    # Show results
    print(f"\nProcessed {len(all_surprises)} chunks")
    print(f"Crystallized {len(crystallization_points)} episodes")
    print(f"Memory size: {memory_gpt.memory.size}")

    if all_surprises:
        surprises = [s[2] for s in all_surprises]
        saliences = [s[2] for s in all_saliences]
        print(f"\nSurprise: min={min(surprises):.4f}, max={max(surprises):.4f}, mean={np.mean(surprises):.4f}")
        print(f"Salience: min={min(saliences):.4f}, max={max(saliences):.4f}, mean={np.mean(saliences):.4f}")

    if crystallization_points and show_tokens:
        print("\n" + "-" * 60)
        print("CRYSTALLIZED MOMENTS:")
        print("-" * 60)
        for start, end, salience in crystallization_points[:5]:  # Show first 5
            chunk_tokens = tokens[0, start:end].tolist()
            chunk_text = tokenizer.decode(chunk_tokens)
            print(f"\n  Position {start}-{end}, salience={salience:.4f}")
            print(f"  Text: {chunk_text[:200]}...")

    # Show high-surprise moments
    if all_surprises:
        print("\n" + "-" * 60)
        print("HIGHEST SURPRISE MOMENTS:")
        print("-" * 60)
        sorted_surprises = sorted(all_surprises, key=lambda x: x[2], reverse=True)[:5]
        for start, end, surprise in sorted_surprises:
            chunk_tokens = tokens[0, start:end].tolist()
            chunk_text = tokenizer.decode(chunk_tokens)
            print(f"\n  Position {start}-{end}, surprise={surprise:.4f}")
            print(f"  Text: {chunk_text[:200]}...")


def analyze_from_data(
    memory_gpt: MemoryAugmentedGPT,
    tokenizer,
    data_path: str,
    device: torch.device,
    n_samples: int = 5
):
    """Analyze salience on samples from training data."""
    from pretrain import PretokenizedDataset

    print("\n" + "=" * 60)
    print(f"ANALYZING SAMPLES FROM: {data_path}")
    print("=" * 60)

    # Load a few samples
    meta_path = data_path.replace('_tokens.bin', '_metadata.json')
    with open(meta_path) as f:
        meta = json.load(f)

    dataset = PretokenizedDataset(
        data_path,
        min(1000, meta['num_examples']),
        max_seq_len=1024,
        stride=1024,
        data_type="Analysis"
    )

    # Analyze random samples
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for idx in indices:
        sample = dataset[int(idx)]
        tokens = sample['input_ids'].unsqueeze(0).to(device)

        # Reset memory for each sample
        memory_gpt.reset_memory()

        print(f"\n--- Sample {idx} ---")
        text = tokenizer.decode(tokens[0].tolist()[:100])
        print(f"Start: {text}...")

        # Process and collect stats
        with torch.no_grad():
            logits, hidden, mem_out = memory_gpt(
                tokens,
                crystallize=True,
                use_memory=False  # No retrieval for first pass
            )

        if mem_out.get('salience') is not None:
            print(f"Salience: {mem_out['salience'].item():.4f}")
            print(f"Surprise: {mem_out['surprise'].item():.4f}")
            print(f"Valence: {mem_out['valence'].item():.4f}")
            print(f"Arousal: {mem_out['arousal'].item():.4f}")
            print(f"Crystallized: {mem_out['crystallized']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze salience moments")
    parser.add_argument("--checkpoint", default="memory_augmented_1024/memory_gpt_epoch_1.pt")
    parser.add_argument("--data", default="tiny_pretrain_output/training_meta-llama_Meta-Llama-3-8B_tokens.bin")
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--text", type=str, help="Custom text to analyze")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Load model
    memory_gpt, config = load_checkpoint(args.checkpoint, device)

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Analyze stored memory
    analyze_memory_stats(memory_gpt.memory)

    # Analyze from data
    if os.path.exists(args.data):
        analyze_from_data(memory_gpt, tokenizer, args.data, device, args.n_samples)

    # Custom text analysis
    if args.text:
        run_live_analysis(memory_gpt, tokenizer, args.text, device)
    else:
        # Use a sample text
        sample_text = """
        The quick brown fox jumps over the lazy dog. This is a simple sentence.
        However, what happens next is truly surprising! A massive earthquake struck
        the city, causing buildings to collapse and people to run in panic.
        The death toll was estimated at over 10,000. Markets crashed worldwide.
        But then, miraculously, everything returned to normal. The sun rose again.
        """
        run_live_analysis(memory_gpt, tokenizer, sample_text, device)


if __name__ == "__main__":
    main()
