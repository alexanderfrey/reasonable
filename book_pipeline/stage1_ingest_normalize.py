#!/usr/bin/env python3
"""
Stage 1: Ingest & normalize
- Convert to UTF-8, normalize Unicode (NFKC), normalize line endings.
- Strip control characters, BOMs, excess whitespace.
- Writes a mirrored tree to --output_dir.
"""
import argparse
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from book_pipeline.common import auto_read_text, ensure_parent_dir, is_binary_file, iter_txt_files


def normalize_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\xa0", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def process_file(path: Path, input_root: Path, output_root: Path, overwrite: bool) -> str:
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
    if not raw.strip():
        return "empty"
    ensure_parent_dir(out_path)
    out_path.write_text(normalize_text(raw), encoding="utf-8")
    return "written"


def main():
    ap = argparse.ArgumentParser(description="Stage 1: ingest and normalize book .txt files.")
    ap.add_argument("input_dir", help="Directory with raw converted .txt files.")
    ap.add_argument("--output_dir", default="data/books_stage1_norm", help="Where to write normalized files.")
    ap.add_argument("--workers", type=int, default=8, help="Thread pool workers.")
    ap.add_argument("--overwrite", action="store_true", help="Rewrite files even if output exists.")
    args = ap.parse_args()

    input_root = Path(args.input_dir).expanduser()
    output_root = Path(args.output_dir).expanduser()
    if not input_root.exists():
        raise SystemExit(f"Input directory does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    stats = {"written": 0, "skipped_existing": 0, "binary": 0, "empty": 0, "errors": 0}

    def task(p: Path) -> None:
        result = process_file(p, input_root, output_root, args.overwrite)
        if result.startswith("error:"):
            stats["errors"] += 1
            print(f"[SKIP] {p}: {result}")
        elif result == "written":
            stats["written"] += 1
        else:
            stats[result] = stats.get(result, 0) + 1

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(task, iter_txt_files([input_root]), chunksize=32))

    print("=== Stage 1 summary ===")
    print(f"Written          : {stats['written']}")
    print(f"Skipped existing : {stats['skipped_existing']}")
    print(f"Skipped binary   : {stats['binary']}")
    print(f"Skipped empty    : {stats['empty']}")
    print(f"Errors           : {stats['errors']}")
    print(f"Output dir       : {output_root}")


if __name__ == "__main__":
    main()
