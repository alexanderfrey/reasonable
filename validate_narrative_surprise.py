"""
Validate: Does surprise correlate with meaningful narrative events?

Hypothesis: High surprise should occur at:
- Plot twists and revelations
- Character introductions
- Emotional peaks
- Scene transitions
- Conflict escalation

This script:
1. Processes narrative text through the model
2. Identifies high-surprise moments
3. Shows the text at those moments for human evaluation
4. Computes correlation with structural markers (chapter/paragraph breaks)
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

from model import GPT, GPTConfig
from experiential import MemoryAugmentedGPT


@dataclass
class SurpriseMoment:
    """A moment of notable surprise in the narrative."""
    position: int           # Token position
    surprise: float         # Surprise value
    meta_surprise: float    # Meta-surprise (self-prediction error)
    salience: float         # Salience score
    text_before: str        # Context before
    text_at: str            # The surprising text
    text_after: str         # Context after
    crystallized: bool      # Whether this became a memory


def load_model(
    checkpoint_path: str,
    device: torch.device,
    crystallization_threshold: Optional[float] = None,
    meta_surprise_salience_weight: float = 1.0,
):
    """Load trained memory-augmented model."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    config_dict = ckpt.get('config', {})
    args = ckpt.get('args', {})

    # If config is empty, try to load from base checkpoint referenced in args
    if not config_dict and 'checkpoint' in args:
        base_ckpt_path = args['checkpoint']
        print(f"Loading base config from: {base_ckpt_path}")
        if os.path.exists(base_ckpt_path):
            base_ckpt = torch.load(base_ckpt_path, map_location='cpu', weights_only=False)
            config_dict = base_ckpt.get('config', {})

    config = GPTConfig(
        vocab_size=config_dict.get('vocab_size', 128256),
        d_model=config_dict.get('d_model', 1024),
        n_head=config_dict.get('n_head', 8),
        n_layer=config_dict.get('n_layer', 12),
        max_seq_len=config_dict.get('max_seq_len', 1024),
        n_kv_head=config_dict.get('n_kv_head', 4),
        d_ff=config_dict.get('d_ff', 4096),
    )

    print(f"Model config: d_model={config.d_model}, n_head={config.n_head}, "
          f"n_kv_head={config.n_kv_head}, d_ff={config.d_ff}")

    # Use provided threshold or default from checkpoint
    threshold = crystallization_threshold if crystallization_threshold is not None else args.get('crystallization_threshold', 0.2)
    print(f"Crystallization threshold: {threshold}")
    print(f"Meta-surprise salience weight: {meta_surprise_salience_weight}")

    gpt = GPT(config)
    memory_gpt = MemoryAugmentedGPT(
        gpt,
        memory_capacity=args.get('memory_capacity', 1000),
        crystallization_threshold=threshold,
        memory_integration=args.get('integration', 'gated'),
        meta_surprise_salience_weight=meta_surprise_salience_weight,
    )

    memory_gpt.load_state_dict(ckpt['memory_gpt_state_dict'], strict=False)
    memory_gpt = memory_gpt.to(device)
    memory_gpt.eval()

    return memory_gpt, config


def analyze_narrative(
    memory_gpt: MemoryAugmentedGPT,
    tokenizer,
    text: str,
    device: torch.device,
    chunk_size: int = 128,
    stride: int = 64,
    context_chars: int = 100
) -> Tuple[List[SurpriseMoment], dict]:
    """
    Analyze a narrative for surprise patterns.

    Returns:
        moments: List of SurpriseMoment for each chunk
        stats: Overall statistics
    """
    # Tokenize
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    seq_len = tokens.size(1)
    print(f"Analyzing {seq_len} tokens...")

    # Clear memory for fresh analysis
    memory_gpt.reset_memory()

    moments = []
    prev_memory_query = None

    for start in range(0, seq_len - chunk_size, stride):
        end = min(start + chunk_size, seq_len)
        chunk = tokens[:, start:end]

        with torch.no_grad():
            logits, hidden, mem_out = memory_gpt(
                chunk,
                crystallize=True,
                use_memory=True,
                prev_memory_query=prev_memory_query
            )

        # Update query for next chunk
        prev_memory_query = mem_out.get('next_memory_query')

        if mem_out.get('surprise') is not None:
            surprise = mem_out['surprise'].item()
            meta_surprise = mem_out.get('meta_surprise')
            meta_surprise = meta_surprise.item() if meta_surprise is not None else 0.0
            salience = mem_out['salience'].item() if mem_out.get('salience') is not None else surprise

            # Get text context
            char_start = len(tokenizer.decode(tokens[0, :start].tolist()))
            char_end = len(tokenizer.decode(tokens[0, :end].tolist()))

            text_before = text[max(0, char_start - context_chars):char_start]
            text_at = text[char_start:char_end]
            text_after = text[char_end:char_end + context_chars]

            moments.append(SurpriseMoment(
                position=start,
                surprise=surprise,
                meta_surprise=meta_surprise,
                salience=salience,
                text_before=text_before,
                text_at=text_at[:200],  # Truncate for display
                text_after=text_after,
                crystallized=mem_out.get('crystallized', False)
            ))

    # Compute statistics
    surprises = [m.surprise for m in moments]
    meta_surprises = [m.meta_surprise for m in moments]
    saliences = [m.salience for m in moments]
    crystallized_count = sum(1 for m in moments if m.crystallized)

    stats = {
        'n_chunks': len(moments),
        'surprise_mean': np.mean(surprises),
        'surprise_std': np.std(surprises),
        'surprise_min': min(surprises),
        'surprise_max': max(surprises),
        'meta_surprise_mean': np.mean(meta_surprises),
        'salience_mean': np.mean(saliences),
        'crystallized_count': crystallized_count,
        'crystallization_rate': crystallized_count / len(moments) if moments else 0,
        'memory_size': memory_gpt.memory.size,
    }

    return moments, stats


def find_structural_markers(text: str) -> List[Tuple[int, str]]:
    """Find structural markers in text (chapters, paragraphs, dialogue)."""
    markers = []

    # Chapter headings
    for match in re.finditer(r'\n\s*(Chapter|CHAPTER)\s+\w+', text):
        markers.append((match.start(), 'chapter'))

    # Paragraph breaks (double newline)
    for match in re.finditer(r'\n\s*\n', text):
        markers.append((match.start(), 'paragraph'))

    # Dialogue starts
    for match in re.finditer(r'["""]', text):
        markers.append((match.start(), 'dialogue'))

    # Exclamations (emotional peaks)
    for match in re.finditer(r'[!?]{1,3}', text):
        markers.append((match.start(), 'exclamation'))

    return sorted(markers, key=lambda x: x[0])


def correlate_with_structure(
    moments: List[SurpriseMoment],
    text: str,
    tokenizer,
    tokens: torch.Tensor
) -> dict:
    """Check if high surprise correlates with structural markers."""
    markers = find_structural_markers(text)

    if not markers:
        return {'correlation': 0.0, 'marker_count': 0}

    # Convert marker positions to token positions
    marker_tokens = set()
    for char_pos, marker_type in markers:
        # Approximate: find token position from character position
        prefix = text[:char_pos]
        token_pos = len(tokenizer.encode(prefix))
        marker_tokens.add(token_pos)

    # Check overlap with high-surprise moments
    high_surprise_threshold = np.percentile([m.surprise for m in moments], 75)
    high_surprise_positions = set(m.position for m in moments if m.surprise > high_surprise_threshold)

    # Count how many high-surprise moments are near structural markers
    near_marker = 0
    window = 20  # tokens

    for pos in high_surprise_positions:
        for marker_pos in marker_tokens:
            if abs(pos - marker_pos) < window:
                near_marker += 1
                break

    return {
        'high_surprise_count': len(high_surprise_positions),
        'marker_count': len(markers),
        'near_marker_count': near_marker,
        'correlation': near_marker / len(high_surprise_positions) if high_surprise_positions else 0,
    }


def display_results(moments: List[SurpriseMoment], stats: dict, top_k: int = 10):
    """Display analysis results."""
    print("\n" + "=" * 70)
    print("NARRATIVE SURPRISE ANALYSIS RESULTS")
    print("=" * 70)

    print("\nOverall Statistics:")
    print(f"  Chunks analyzed:      {stats['n_chunks']}")
    print(f"  Surprise mean:        {stats['surprise_mean']:.4f} ± {stats['surprise_std']:.4f}")
    print(f"  Surprise range:       [{stats['surprise_min']:.4f}, {stats['surprise_max']:.4f}]")
    print(f"  Meta-surprise mean:   {stats['meta_surprise_mean']:.4f}")
    print(f"  Salience mean:        {stats['salience_mean']:.4f}")
    print(f"  Crystallization rate: {stats['crystallization_rate']:.1%}")
    print(f"  Memories formed:      {stats['crystallized_count']}")
    print(f"  Final memory size:    {stats['memory_size']}")

    # Sort by surprise and show top moments
    sorted_moments = sorted(moments, key=lambda x: x.surprise, reverse=True)

    print("\n" + "-" * 70)
    print(f"TOP {top_k} HIGHEST SURPRISE MOMENTS:")
    print("-" * 70)

    for i, m in enumerate(sorted_moments[:top_k]):
        print(f"\n#{i+1} Position {m.position} | Surprise: {m.surprise:.4f} | "
              f"Meta-surprise: {m.meta_surprise:.4f} | Crystallized: {m.crystallized}")
        print(f"  Context: ...{m.text_before[-50:]}")
        print(f"  >>> {m.text_at[:150]}...")
        print(f"  ...{m.text_after[:50]}...")

    # Show crystallized moments (if different from high surprise)
    crystallized = [m for m in moments if m.crystallized]
    if crystallized:
        print("\n" + "-" * 70)
        print(f"CRYSTALLIZED MOMENTS ({len(crystallized)} total):")
        print("-" * 70)

        for i, m in enumerate(crystallized[:5]):
            print(f"\n  Memory #{i+1} at position {m.position}")
            print(f"  Salience: {m.salience:.4f} | Surprise: {m.surprise:.4f}")
            print(f"  Text: {m.text_at[:100]}...")


def load_narrative_text(source: str, max_chars: int = 50000) -> str:
    """Load narrative text from file or use built-in sample."""
    if os.path.exists(source):
        with open(source, 'r', encoding='utf-8', errors='ignore') as f:
            # Read only max_chars without loading entire file
            text = f.read(max_chars)
        print(f"Loaded {len(text)} characters from {source}")
        return text

    # Built-in sample narrative for testing
    sample = """
    Chapter 1: The Beginning

    Sarah had always known she was different. Growing up in the small town of
    Millbrook, she felt like an outsider, watching life happen around her while
    she remained a spectator.

    That all changed on her eighteenth birthday.

    "Happy birthday, sweetheart," her mother said, placing a worn leather box
    on the breakfast table. Sarah looked up, surprised. Her family wasn't one
    for extravagant gifts.

    "What's this?" she asked, reaching for the box.

    "It belonged to your grandmother. She wanted you to have it when you came
    of age."

    Sarah opened the box and gasped. Inside was an amulet, ancient and glowing
    with an otherworldly light. The moment her fingers touched it, the world
    around her shifted.

    Everything went dark.

    When she opened her eyes, she was no longer in her kitchen. She stood in a
    vast hall, surrounded by hooded figures. And they were all looking at her.

    "The prophecy has begun," one of them whispered. "The last guardian has awakened."

    Chapter 2: The Revelation

    Sarah's heart pounded as she looked around the hall. Massive stone pillars
    stretched toward a ceiling lost in shadow. Torches flickered along the walls,
    casting dancing shadows across the hooded figures.

    "Where am I?" she demanded, her voice echoing in the vast space. "Who are you?"

    The figure closest to her stepped forward and lowered its hood. Sarah stumbled
    backward in shock.

    It was her grandmother. But that was impossible—her grandmother had died
    three years ago!

    "Don't be afraid, child," the woman said, her voice warm and familiar.
    "There is much you don't understand. Much that has been hidden from you
    for your own protection."

    "You're dead," Sarah whispered. "I was at your funeral!"

    Her grandmother smiled sadly. "Death is not always what it seems in our
    world, Sarah. You carry the blood of the Guardians—those who stand between
    the realms. And now, a great darkness is rising. One that only you can stop."
    """

    print("Using built-in sample narrative")
    return sample


def main():
    parser = argparse.ArgumentParser(description="Validate surprise on narrative data")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--narrative", default="sample", help="Path to narrative text or 'sample'")
    parser.add_argument("--max_chars", type=int, default=50000, help="Max characters to analyze")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for analysis")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top moments to show")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="Optional: save results to JSON")
    # Threshold calibration arguments
    parser.add_argument("--threshold", type=float, default=None,
                        help="Crystallization threshold (default: from checkpoint)")
    parser.add_argument("--meta_surprise_weight", type=float, default=1.0,
                        help="Meta-surprise salience weight (default: 1.0, was 3.0)")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Load model
    memory_gpt, config = load_model(
        args.checkpoint,
        device,
        crystallization_threshold=args.threshold,
        meta_surprise_salience_weight=args.meta_surprise_weight,
    )

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load narrative
    text = load_narrative_text(args.narrative, args.max_chars)

    # Analyze
    moments, stats = analyze_narrative(
        memory_gpt, tokenizer, text, device,
        chunk_size=args.chunk_size
    )

    # Check correlation with structure
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    structure_correlation = correlate_with_structure(moments, text, tokenizer, tokens)
    stats.update(structure_correlation)

    # Display
    display_results(moments, stats, top_k=args.top_k)

    print("\n" + "=" * 70)
    print("STRUCTURAL CORRELATION ANALYSIS")
    print("=" * 70)
    print(f"  High-surprise moments:       {stats.get('high_surprise_count', 0)}")
    print(f"  Structural markers found:    {stats.get('marker_count', 0)}")
    print(f"  High-surprise near markers:  {stats.get('near_marker_count', 0)}")
    print(f"  Correlation:                 {stats.get('correlation', 0):.1%}")

    # Save results if requested
    if args.output:
        results = {
            'stats': stats,
            'moments': [
                {
                    'position': m.position,
                    'surprise': m.surprise,
                    'meta_surprise': m.meta_surprise,
                    'salience': m.salience,
                    'crystallized': m.crystallized,
                    'text': m.text_at[:200]
                }
                for m in moments
            ]
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
