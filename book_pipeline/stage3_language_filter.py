#!/usr/bin/env python3
"""
Stage 3: Language and quality filtering
- Fast language detection (langdetect if installed) with allowlist.
- Heuristics: min chars, letter ratio, weird-symbol ratio, token/char sanity.
"""
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Set
import unicodedata

from book_pipeline.common import auto_read_text, ensure_parent_dir, is_binary_file, iter_txt_files

try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
except Exception:  # pragma: no cover - optional dependency
    detect = None


def detect_lang(text: str, max_chars: int = 4000) -> Optional[str]:
    if detect is None:
        return None
    sample = text[:max_chars]
    try:
        return detect(sample)
    except Exception:
        return None


def quality_checks(text: str, min_chars: int, min_letter_ratio: float, max_weird_ratio: float, min_char_per_token: float) -> Optional[str]:
    if len(text) < min_chars:
        return "too_short"
    total = len(text)
    letters = 0
    weird = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
        cat = unicodedata.category(ch)
        is_punct = cat.startswith("P")
        if not (ch.isalnum() or ch.isspace() or is_punct):
            weird += 1
    letter_ratio = letters / total
    weird_ratio = weird / total
    if letter_ratio < min_letter_ratio:
        return "low_letter_ratio"
    if weird_ratio > max_weird_ratio:
        return "high_weird_ratio"
    tokens = re.findall(r"\w+", text.lower())
    if tokens:
        chars_per_token = total / len(tokens)
        if chars_per_token < min_char_per_token:
            return "too_many_tokens_vs_chars"
    return None


def process_file(
    path: Path,
    input_root: Path,
    output_root: Path,
    allowed_langs: Set[str],
    min_chars: int,
    min_letter_ratio: float,
    max_weird_ratio: float,
    min_char_per_token: float,
    overwrite: bool,
) -> str:
    rel = path.relative_to(input_root)
    out_path = output_root / rel
    if not overwrite and out_path.exists():
        return "skipped_existing"
    if is_binary_file(path):
        return "binary"
    try:
        text = auto_read_text(path)
    except Exception as exc:
        return f"error:{exc}"

    reason = quality_checks(text, min_chars, min_letter_ratio, max_weird_ratio, min_char_per_token)
    if reason:
        return reason

    if allowed_langs:
        lang = detect_lang(text)
        if lang is not None and lang not in allowed_langs:
            return f"lang_{lang}"

    ensure_parent_dir(out_path)
    out_path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return "written"


def main():
    ap = argparse.ArgumentParser(description="Stage 3: language + quality filtering for cleaned book texts.")
    ap.add_argument("input_dir", help="Directory with stage2 outputs.")
    ap.add_argument("--output_dir", default="data/books_stage3_langfiltered", help="Where to write filtered files.")
    ap.add_argument("--langs", default="en", help="Comma-separated allowed languages (ISO-639-1). Empty to skip language filtering.")
    ap.add_argument("--min_chars", type=int, default=1200, help="Drop docs shorter than this.")
    ap.add_argument("--min_letter_ratio", type=float, default=0.65, help="Min letters/total chars ratio.")
    ap.add_argument("--max_weird_ratio", type=float, default=0.06, help="Max share of weird symbols.")
    ap.add_argument("--min_char_per_token", type=float, default=2.5, help="Drop if chars/token is below this, indicating OCR garbage.")
    ap.add_argument("--workers", type=int, default=8, help="Thread pool workers.")
    ap.add_argument("--overwrite", action="store_true", help="Rewrite files even if output exists.")
    args = ap.parse_args()

    input_root = Path(args.input_dir).expanduser()
    output_root = Path(args.output_dir).expanduser()
    if not input_root.exists():
        raise SystemExit(f"Input directory does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    allowed_langs = {lang.strip() for lang in args.langs.split(",") if lang.strip()} if args.langs else set()
    if allowed_langs and detect is None:
        raise SystemExit("Language filtering requested via --langs but 'langdetect' is not installed. Install with `pip install langdetect` or pass --langs='' to disable.")

    stats = {
        "written": 0,
        "skipped_existing": 0,
        "binary": 0,
        "too_short": 0,
        "low_letter_ratio": 0,
        "high_weird_ratio": 0,
        "too_many_tokens_vs_chars": 0,
        "lang_filtered": 0,
        "errors": 0,
    }

    def task(p: Path) -> None:
        result = process_file(
            p,
            input_root,
            output_root,
            allowed_langs,
            args.min_chars,
            args.min_letter_ratio,
            args.max_weird_ratio,
            args.min_char_per_token,
            args.overwrite,
        )
        if result.startswith("error:"):
            stats["errors"] += 1
            print(f"[SKIP] {p}: {result}")
        elif result.startswith("lang_"):
            stats["lang_filtered"] += 1
        elif result == "written":
            stats["written"] += 1
        else:
            stats[result] = stats.get(result, 0) + 1

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(task, iter_txt_files([input_root]), chunksize=32))

    print("=== Stage 3 summary ===")
    print(f"Written              : {stats['written']}")
    print(f"Skipped existing     : {stats['skipped_existing']}")
    print(f"Skipped binary       : {stats['binary']}")
    print(f"Too short            : {stats['too_short']}")
    print(f"Low letter ratio     : {stats['low_letter_ratio']}")
    print(f"Weird symbol ratio   : {stats['high_weird_ratio']}")
    print(f"Token/char mismatch  : {stats['too_many_tokens_vs_chars']}")
    print(f"Lang filtered        : {stats['lang_filtered']}")
    print(f"Errors               : {stats['errors']}")
    print(f"Output dir           : {output_root}")


if __name__ == "__main__":
    main()
