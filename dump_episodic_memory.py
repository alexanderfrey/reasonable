#!/usr/bin/env python3
"""
Dump episodic memory snapshots from a checkpoint into a readable text report.

Example:
  python dump_episodic_memory.py --checkpoint memory_augmented_output/memory_gpt_epoch_1.pt
"""

import argparse
import sys
from typing import List, Optional

import torch


def _format_text(text: Optional[str], max_chars: int) -> str:
    if not text:
        return "(no text stored)"
    cleaned = " ".join(text.split())
    if max_chars > 0 and len(cleaned) > max_chars:
        return cleaned[: max_chars - 3] + "..."
    return cleaned


def _format_tokens(token_ids: Optional[List[int]], max_tokens: int) -> str:
    if not token_ids:
        return "(no token ids stored)"
    if max_tokens > 0:
        token_ids = token_ids[:max_tokens]
    return " ".join(str(t) for t in token_ids)


def _sort_episodes(episodes, sort_key: str):
    if sort_key == "time":
        return sorted(episodes, key=lambda e: e.timestamp, reverse=True)
    if sort_key == "retrievals":
        return sorted(episodes, key=lambda e: e.retrieval_count, reverse=True)
    return sorted(episodes, key=lambda e: e.salience, reverse=True)


def load_episodes(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    snapshot = checkpoint.get("episodic_memory")
    if snapshot is None:
        raise KeyError("Checkpoint does not contain 'episodic_memory'.")
    episodes = snapshot.get("episodes", [])
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump episodic memory from a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--output", default=None, help="Write report to file instead of stdout")
    parser.add_argument("--limit", type=int, default=20, help="Max episodes to print")
    parser.add_argument("--sort", choices=["salience", "time", "retrievals"], default="salience")
    parser.add_argument("--min_salience", type=float, default=0.0, help="Filter by salience")
    parser.add_argument("--max_chars", type=int, default=240, help="Max chars of text snippet")
    parser.add_argument("--show_tokens", action="store_true", help="Print token id snippets")
    parser.add_argument("--max_tokens", type=int, default=40, help="Max token ids to print")

    args = parser.parse_args()

    try:
        episodes = load_episodes(args.checkpoint)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not episodes:
        print("No episodes found.")
        return 0

    filtered = [ep for ep in episodes if ep.salience >= args.min_salience]
    sorted_eps = _sort_episodes(filtered, args.sort)
    limited = sorted_eps[: args.limit]

    lines = []
    lines.append(f"Total episodes: {len(episodes)}")
    lines.append(f"Filtered episodes: {len(filtered)} (min_salience={args.min_salience})")
    lines.append(f"Showing: {len(limited)} sorted by {args.sort}")
    lines.append("")

    for idx, ep in enumerate(limited, start=1):
        lines.append(
            f"[{idx}] ts={ep.timestamp} salience={ep.salience:.4f} "
            f"retrievals={ep.retrieval_count} valence={ep.valence:.3f} arousal={ep.arousal:.3f}"
        )
        lines.append(f"  text: {_format_text(ep.text, args.max_chars)}")
        if args.show_tokens:
            lines.append(f"  token_ids: {_format_tokens(ep.token_ids, args.max_tokens)}")
        lines.append("")

    output = "\n".join(lines).rstrip() + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        sys.stdout.write(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
