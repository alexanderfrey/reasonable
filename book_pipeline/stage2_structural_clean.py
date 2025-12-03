#!/usr/bin/env python3
"""
Stage 2: Structural cleaning
- Remove page numbers, headers/footers, boilerplate lines (incl. Gutenberg).
- Fix line breaks, hyphenation, and collapse paragraphs.
"""
import argparse
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Set

from book_pipeline.common import auto_read_text, ensure_parent_dir, is_binary_file, iter_txt_files


def strip_gutenberg_license(lines: List[str]) -> List[str]:
    """Drop Project Gutenberg boilerplate if present."""
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if "start of this project gutenberg ebook" in low or "start of the project gutenberg ebook" in low:
            start_idx = i + 1
        if "end of this project gutenberg ebook" in low or "end of the project gutenberg ebook" in low:
            end_idx = i
            break
    if start_idx is not None:
        lines = lines[start_idx:]
    if end_idx is not None:
        lines = lines[:end_idx]
    return lines


def drop_noise_lines(lines: List[str]) -> List[str]:
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
        if "project gutenberg" in low or "gutenberg.org" in low:
            continue
        if low.startswith("produced by ") or low.startswith("created by "):
            continue
        if re.fullmatch(r"[-_*#=]{3,}", stripped):
            continue
        if re.match(r"(chapter|section)\s+[\divxlc0-9]+", low) and len(stripped) <= 60:
            continue
        if re.fullmatch(r"[ivxlcdm]+", low) and len(stripped) <= 5:
            continue
        if len(stripped) <= 4 and sum(c.isalpha() for c in stripped) <= 1:
            continue
        alpha = sum(c.isalpha() for c in stripped)
        upper = sum(c.isupper() for c in stripped)
        if alpha > 0 and upper / alpha > 0.85 and re.search(r"\d", stripped) and len(stripped) <= 70:
            continue
        cleaned.append(stripped)
    return cleaned


def drop_repeated_short_lines(lines: List[str], min_repeats: int) -> List[str]:
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
        if len(line) <= 8 or alpha_ratio < 0.35 or count >= 12:
            to_remove.add(line)

    return [line for line in lines if (line and line not in to_remove) or line == ""]


def merge_lines_to_paragraphs(lines: List[str]) -> str:
    paragraphs: List[str] = []
    buffer: List[str] = []
    for line in lines:
        if not line:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            paragraphs.append("")
            continue
        if buffer and buffer[-1].endswith("-"):
            buffer[-1] = buffer[-1][:-1] + line.lstrip()
        else:
            buffer.append(line)
    if buffer:
        paragraphs.append(" ".join(buffer))

    compact: List[str] = []
    for para in paragraphs:
        para = re.sub(r"[ \t]+", " ", para).strip()
        if not para:
            if compact and compact[-1] != "":
                compact.append("")
            continue
        compact.append(para)

    if compact and compact[-1] == "":
        compact.pop()
    return "\n\n".join(compact)


def clean_structural(raw: str, repeat_threshold: int) -> str:
    text = unicodedata.normalize("NFKC", raw.replace("\r", "\n"))
    text = text.replace("\ufeff", "").replace("\xa0", " ")
    lines = [line.rstrip() for line in text.split("\n")]
    lines = strip_gutenberg_license(lines)
    lines = drop_noise_lines(lines)
    lines = drop_repeated_short_lines(lines, repeat_threshold)
    text = merge_lines_to_paragraphs(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def process_file(path: Path, input_root: Path, output_root: Path, repeat_threshold: int, min_chars: int, overwrite: bool) -> str:
    rel = path.relative_to(input_root)
    out_path = output_root / rel
    if not overwrite and out_path.exists():
        return "skipped_existing"
    if is_binary_file(path):
        return "binary"
    try:
        raw = auto_read_text(path)
    except Exception as exc:
        return f"error:{exc}"
    cleaned = clean_structural(raw, repeat_threshold)
    if len(cleaned) < min_chars:
        return "too_short"
    ensure_parent_dir(out_path)
    out_path.write_text(cleaned + "\n", encoding="utf-8")
    return "written"


def main():
    ap = argparse.ArgumentParser(description="Stage 2: structural cleaning of normalized book .txt files.")
    ap.add_argument("input_dir", help="Directory with stage1 outputs.")
    ap.add_argument("--output_dir", default="data/books_stage2_struct", help="Where to write cleaned files.")
    ap.add_argument("--repeat_threshold", type=int, default=4, help="Lines repeated at least this many times are treated as headers/footers.")
    ap.add_argument("--min_chars", type=int, default=1200, help="Drop documents shorter than this after cleaning.")
    ap.add_argument("--workers", type=int, default=8, help="Thread pool workers.")
    ap.add_argument("--overwrite", action="store_true", help="Rewrite files even if output exists.")
    args = ap.parse_args()

    input_root = Path(args.input_dir).expanduser()
    output_root = Path(args.output_dir).expanduser()
    if not input_root.exists():
        raise SystemExit(f"Input directory does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    stats = {"written": 0, "skipped_existing": 0, "binary": 0, "too_short": 0, "errors": 0}

    def task(p: Path) -> None:
        result = process_file(p, input_root, output_root, args.repeat_threshold, args.min_chars, args.overwrite)
        if result.startswith("error:"):
            stats["errors"] += 1
            print(f"[SKIP] {p}: {result}")
        elif result == "written":
            stats["written"] += 1
        else:
            stats[result] = stats.get(result, 0) + 1

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(task, iter_txt_files([input_root]), chunksize=32))

    print("=== Stage 2 summary ===")
    print(f"Written          : {stats['written']}")
    print(f"Skipped existing : {stats['skipped_existing']}")
    print(f"Skipped binary   : {stats['binary']}")
    print(f"Skipped too short: {stats['too_short']} (<{args.min_chars})")
    print(f"Errors           : {stats['errors']}")
    print(f"Output dir       : {output_root}")


if __name__ == "__main__":
    main()
