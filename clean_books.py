#!/usr/bin/env python3
"""
Clean converted book .txt files for pretraining.

Features:
- Recursive *.txt discovery under an input directory.
- Deterministic cleaning (unicode/control chars, soft hyphens, repeated headers/footers, Gutenberg boilerplate).
- Heuristic structure detection to preserve verse/code while reflowing prose.
- Optional LLM-assisted filtering/classification for boilerplate/OCR/unsafe/off-topic blocks (OpenAI-compatible API).
- Deduplicates identical documents (blake2s hash) across all inputs and drops very short docs.
- Writes per-book cleaned files mirroring the input tree and optionally concatenates into one corpus.
"""
import argparse
import json
import hashlib
import os
import re
import unicodedata
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


logger = logging.getLogger("clean_books")


def is_binary_file(path: Path, blocksize: int = 1024) -> bool:
    """Heuristically detect binary files."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(blocksize)
    except OSError:
        return True
    if not chunk:
        return False
    if b"\x00" in chunk:
        return True
    text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
    nontext = sum(b not in text_chars for b in chunk)
    return (nontext / len(chunk)) > 0.30


def read_text(path: Path) -> str:
    """UTF-8 with latin-1 fallback, ignoring undecodable chars."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")


def drop_noise_lines(lines: List[str]) -> List[str]:
    """Remove obvious noise like standalone page numbers or URLs."""
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        low = stripped.lower()
        if stripped.isdigit():
            continue
        if re.fullmatch(r"page\s*\d+", low):
            continue
        if re.fullmatch(r"\d+\s*/\s*\d+", stripped):
            continue
        if re.search(r"https?://", low):
            continue
        if re.fullmatch(r"[ivxlcdm]+", low) and len(stripped) <= 5:
            continue
        if len(stripped) <= 4 and sum(c.isalpha() for c in stripped) <= 1:
            continue
        cleaned.append(stripped)
    return cleaned


def drop_repeated_short_lines(lines: List[str], min_repeats: int) -> List[str]:
    """Remove headers/footers that appear on many pages."""
    freq: Dict[str, int] = {}
    for line in lines:
        if not line:
            continue
        if len(line) > 120:
            continue
        freq[line] = freq.get(line, 0) + 1

    to_remove: Set[str] = set()
    for line, count in freq.items():
        if count < min_repeats:
            continue
        alpha_ratio = sum(ch.isalpha() for ch in line) / max(1, len(line))
        if len(line) <= 8 or alpha_ratio < 0.35:
            to_remove.add(line)
        elif count >= 12 and len(line) <= 120:
            to_remove.add(line)

    return [line for line in lines if (line and line not in to_remove) or line == ""]


def strip_gutenberg_boilerplate(lines: List[str]) -> List[str]:
    """Drop Project Gutenberg style boilerplate if present."""
    start_pat = re.compile(r"\*\*\*\s*START OF (THE )?PROJECT GUTENBERG", re.IGNORECASE)
    end_pat = re.compile(r"\*\*\*\s*END OF (THE )?PROJECT GUTENBERG", re.IGNORECASE)

    start_idx = next((i for i, line in enumerate(lines) if start_pat.search(line)), None)
    end_idx = next((i for i, line in enumerate(lines) if end_pat.search(line)), None)
    if start_idx is not None and end_idx is not None and start_idx < end_idx:
        # keep only the slice between the boilerplate markers
        return lines[start_idx + 1 : end_idx]

    # Fallback: drop early/late license chatter
    drop_prefix = min(len(lines), 80)
    drop_suffix = max(len(lines) - 120, 0)
    trimmed = lines[:]
    for idx in range(drop_prefix):
        if re.search(r"project gutenberg|license", trimmed[idx], re.IGNORECASE):
            trimmed[idx] = ""
    for idx in range(drop_suffix, len(trimmed)):
        if re.search(r"project gutenberg|license", trimmed[idx], re.IGNORECASE):
            trimmed[idx] = ""
    return trimmed


def is_structured_block(block: Sequence[str]) -> bool:
    """Detect verse/code/lists where line breaks should be preserved."""
    if not block:
        return False
    if any(line.startswith(("    ", "\t")) for line in block):
        return True
    if any(re.match(r"^[>*\-•\d]+\s+", line) for line in block):
        return True
    short_lines = sum(len(line) <= 45 for line in block)
    if short_lines >= max(2, len(block) // 2):
        return True
    if any("{" in line or "}" in line or ";" in line for line in block):
        return True
    return False


def merge_lines(block: Sequence[str]) -> str:
    """Merge soft line breaks inside a block while keeping paragraph semantics."""
    buffer: List[str] = []
    for line in block:
        if buffer and buffer[-1].endswith("-"):
            buffer[-1] = buffer[-1][:-1] + line.lstrip()
        else:
            buffer.append(line)
    para = " ".join(buffer)
    para = re.sub(r"[ \t]+", " ", para).strip()
    return para


def blockify(lines: List[str]) -> List[List[str]]:
    """Split lines into blocks separated by blank lines."""
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def normalize_blocks(blocks: List[List[str]], *, llm_decider=None, llm_max_blocks: int = 300) -> Tuple[str, Dict[str, int]]:
    """Process blocks with heuristics plus optional LLM decisions."""
    stats = {"llm_dropped": 0, "llm_structured": 0}
    decisions: List[Dict[str, bool]] = []
    for block in blocks:
        decisions.append(
            {
                "keep": True,
                "reflow": not is_structured_block(block),
            }
        )

    if llm_decider and blocks:
        total = len(blocks)
        budget = min(llm_max_blocks, total)
        head = (budget + 1) // 2
        tail = budget - head
        indices: List[int] = list(range(head))
        if tail > 0:
            tail_start = max(head, total - tail)
            indices.extend(range(tail_start, total))
        # Remove any accidental duplicates while preserving order
        seen_idx = set()
        indices = [i for i in indices if not (i in seen_idx or seen_idx.add(i))]
        logger.info(
            "LLM sampling %d blocks (head=%d, tail=%d) out of %d total",
            len(indices),
            head,
            tail,
            total,
        )
        sampled_blocks = [blocks[i] for i in indices]
        llm_results = llm_decider(sampled_blocks)
        for local_idx, decision in llm_results.items():
            if local_idx >= len(indices):
                continue
            orig_idx = indices[local_idx]
            if orig_idx >= len(decisions):
                continue
            if decision.get("keep") is False:
                decisions[orig_idx]["keep"] = False
                stats["llm_dropped"] += 1
            if decision.get("reflow") is False:
                decisions[orig_idx]["reflow"] = False
                stats["llm_structured"] += 1

    cleaned_blocks: List[str] = []
    for block, dec in zip(blocks, decisions):
        if not dec["keep"]:
            continue
        if dec["reflow"]:
            merged = merge_lines(block)
            if merged:
                cleaned_blocks.append(merged)
        else:
            kept = "\n".join(block).strip()
            if kept:
                cleaned_blocks.append(kept)

    text = "\n\n".join(cleaned_blocks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip(), stats


def clean_text(raw: str, repeat_threshold: int, llm_decider=None, llm_max_blocks: int = 300) -> Tuple[str, Dict[str, int]]:
    """Run the cleaning pipeline on a single document."""
    text = unicodedata.normalize("NFKC", raw.replace("\r", "\n"))
    text = text.replace("\ufeff", "").replace("\xa0", " ").replace("\u00ad", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    lines = [line.rstrip() for line in text.split("\n")]
    lines = drop_noise_lines(lines)
    lines = strip_gutenberg_boilerplate(lines)
    lines = drop_repeated_short_lines(lines, repeat_threshold)
    blocks = blockify(lines)
    merged, stats = normalize_blocks(blocks, llm_decider=llm_decider, llm_max_blocks=llm_max_blocks)
    return merged, stats


LLM_SYSTEM_PROMPT = (
    "You are a data cleaner preparing text for language-model pretraining. "
    "For each block of text, answer with JSON: "
    '{"keep": bool, "reflow": bool, "reason": "short note"}. '
    '"keep" should be false for boilerplate (licenses, disclaimers, table of contents), '
    "OCR gibberish, back-of-book index entries (lists of names/places with page numbers), "
    "and bibliographic/reference lists (citations like journal/volume/pages), "
    "and clearly unsafe/harmful content. "
    '"reflow" should be false when line breaks must be preserved (poetry, plays, code, bullet lists). '
    "Never rewrite the text—only classify it. Keep the JSON minimal."
)


def make_llm_decider(
    *,
    model: str,
    api_base: str,
    api_key: str,
    timeout: float,
    block_char_limit: int,
):
    """Create a callable that classifies blocks via an OpenAI-compatible endpoint."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "The 'openai' package is required for --use_llm.\nInstall with `pip install openai` (>=1.0)."
        ) from exc

    client = OpenAI(base_url=api_base, api_key=api_key)
    cache: Dict[str, Dict[str, bool]] = {}

    def decide(blocks: Sequence[Sequence[str]]) -> Dict[int, Dict[str, bool]]:
        results: Dict[int, Dict[str, bool]] = {}
        for idx, block in enumerate(blocks):
            text = "\n".join(block).strip()
            if not text:
                continue
            key = hashlib.blake2s(text.encode("utf-8")).hexdigest()
            if key in cache:
                results[idx] = cache[key]
                continue

            snippet = text[:block_char_limit]
            try:
                logger.info("LLM classify block %d (%d chars)", idx, len(text))
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": LLM_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Text block:\n```\n{snippet}\n```",
                        },
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=128,
                    temperature=0,
                    timeout=timeout,
                )
                content = completion.choices[0].message.content or "{}"
                data = json.loads(content)
                decision = {
                    "keep": bool(data.get("keep", True)),
                    "reflow": bool(data.get("reflow", True)),
                }
            except Exception as exc:
                logger.warning("LLM classification failed on block %d: %s", idx, exc)
                # Fall back to default heuristic decision on API failure/parsing issues.
                decision = {"keep": True, "reflow": True}

            cache[key] = decision
            results[idx] = decision
        return results

    return decide


def iter_txt_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.txt")):
        if path.is_file():
            yield path


def main():
    ap = argparse.ArgumentParser(description="Clean a directory of converted book .txt files for pretraining.")
    ap.add_argument("input_dir", help="Directory containing raw .txt files (recursed).")
    ap.add_argument("--output_dir", default="data/books_clean", help="Where to write cleaned files (mirrors input tree).")
    ap.add_argument("--aggregate_out", default="data/books_corpus.txt", help="Concatenate cleaned docs into this file.")
    ap.add_argument("--min_chars", type=int, default=1500, help="Drop documents shorter than this after cleaning.")
    ap.add_argument("--repeat_threshold", type=int, default=4, help="Lines repeated at least this many times are treated as headers/footers.")
    ap.add_argument("--no_dedup", action="store_true", help="Keep duplicate docs instead of hashing for deduplication.")
    ap.add_argument("--use_llm", action="store_true", help="Use an OpenAI-compatible endpoint to classify/drop blocks.")
    ap.add_argument("--llm_model", default=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"), help="Model id for the LLM endpoint.")
    ap.add_argument("--llm_api_base", default=os.getenv("LLM_API_BASE", "http://localhost:8000/v1"), help="Base URL for the LLM endpoint.")
    ap.add_argument("--llm_api_key", default=os.getenv("LLM_API_KEY", "dummy"), help="API key for the LLM endpoint.")
    ap.add_argument("--llm_timeout", type=float, default=30.0, help="Timeout (s) for each LLM call.")
    ap.add_argument("--llm_block_char_limit", type=int, default=2000, help="Truncate block text to this many chars before sending to LLM.")
    ap.add_argument("--llm_max_blocks", type=int, default=300, help="Only the first N blocks are sent to the LLM to cap cost.")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging for debugging.")
    ap.add_argument("--progress_every", type=int, default=50, help="Log progress every N files when verbose.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    aggregate_out = Path(args.aggregate_out).expanduser() if args.aggregate_out else None

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if aggregate_out:
        aggregate_out.parent.mkdir(parents=True, exist_ok=True)

    llm_decider = None
    if args.use_llm:
        llm_decider = make_llm_decider(
            model=args.llm_model,
            api_base=args.llm_api_base,
            api_key=args.llm_api_key,
            timeout=args.llm_timeout,
            block_char_limit=args.llm_block_char_limit,
        )

    seen_hashes: Set[str] = set()
    stats = {
        "files": 0,
        "binary": 0,
        "empty": 0,
        "too_short": 0,
        "dedup": 0,
        "written": 0,
        "errors": 0,
        "llm_dropped": 0,
        "llm_structured": 0,
    }

    agg_f = aggregate_out.open("w", encoding="utf-8") if aggregate_out else None
    try:
        for path in iter_txt_files(input_dir):
            stats["files"] += 1
            if is_binary_file(path):
                stats["binary"] += 1
                continue
            try:
                raw = read_text(path)
            except Exception as exc:
                print(f"[SKIP] Failed to read {path}: {exc}")
                stats["errors"] += 1
                continue
            if not raw.strip():
                stats["empty"] += 1
                continue

            cleaned, llm_stats = clean_text(
                raw,
                args.repeat_threshold,
                llm_decider=llm_decider,
                llm_max_blocks=args.llm_max_blocks,
            )
            if args.verbose and any(llm_stats.values()):
                logger.info(
                    "LLM stats for %s: dropped=%d structured=%d",
                    path,
                    llm_stats.get("llm_dropped", 0),
                    llm_stats.get("llm_structured", 0),
                )
            stats["llm_dropped"] += llm_stats.get("llm_dropped", 0)
            stats["llm_structured"] += llm_stats.get("llm_structured", 0)
            if len(cleaned) < args.min_chars:
                stats["too_short"] += 1
                continue

            doc_hash = hashlib.blake2s(cleaned.encode("utf-8")).hexdigest()
            if not args.no_dedup:
                if doc_hash in seen_hashes:
                    stats["dedup"] += 1
                    continue
                seen_hashes.add(doc_hash)

            out_path = output_dir / path.relative_to(input_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(cleaned + "\n", encoding="utf-8")

            if agg_f:
                if stats["written"] > 0:
                    agg_f.write("\n\n")
                agg_f.write(cleaned)

            stats["written"] += 1
            if args.verbose and stats["files"] % max(1, args.progress_every) == 0:
                logger.info(
                    "Progress: files=%d written=%d dedup=%d too_short=%d",
                    stats["files"],
                    stats["written"],
                    stats["dedup"],
                    stats["too_short"],
                )
    finally:
        if agg_f:
            agg_f.close()

    print("=== clean_books summary ===")
    print(f"Input files       : {stats['files']}")
    print(f"Written (kept)    : {stats['written']}")
    print(f"Deduped           : {stats['dedup']}")
    print(f"Skipped binary    : {stats['binary']}")
    print(f"Skipped empty     : {stats['empty']}")
    print(f"Skipped too short : {stats['too_short']} (<{args.min_chars} chars)")
    print(f"Read errors       : {stats['errors']}")
    if args.use_llm:
        print(f"LLM dropped blocks: {stats['llm_dropped']}")
        print(f"LLM kept structured blocks: {stats['llm_structured']}")
    if aggregate_out:
        print(f"Aggregate corpus  : {aggregate_out}")
    print(f"Per-book output   : {output_dir}")


if __name__ == "__main__":
    main()
