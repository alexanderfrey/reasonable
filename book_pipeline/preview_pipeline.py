#!/usr/bin/env python3
"""
Preview run of the book cleaning pipeline on a small sample.
- Copies the first N *.txt files from an input dir into a sandbox.
- Runs stages 1-4 on that subset (using the existing stage scripts).
- Collects before/after stats and writes a report + outputs into a result folder.
"""
import argparse
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from book_pipeline.common import auto_read_text, ensure_parent_dir, iter_txt_files


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    values = sorted(values)
    k = (len(values) - 1) * q
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def file_stats(path: Path) -> Dict[str, float]:
    text = auto_read_text(path)
    chars = len(text)
    lines = text.count("\n") + (0 if text.endswith("\n") or chars == 0 else 1)
    words = len(text.split())
    if chars == 0:
        letter_ratio = 0.0
        weird_ratio = 0.0
    else:
        letters = sum(ch.isalpha() for ch in text)
        weird = sum(ch not in ("\n", " ", "\t") and not ch.isalnum() and ch not in ".,;:!?-'\"()[]{}" for ch in text)
        letter_ratio = letters / chars
        weird_ratio = weird / chars
    return {
        "chars": chars,
        "lines": lines,
        "words": words,
        "letter_ratio": letter_ratio,
        "weird_ratio": weird_ratio,
    }


def dir_summary(root: Path) -> Dict[str, float]:
    entries = [file_stats(p) for p in iter_txt_files([root])]
    if not entries:
        return {
            "files": 0,
            "chars_total": 0,
            "chars_mean": 0,
            "chars_median": 0,
            "chars_p95": 0,
            "lines_mean": 0,
            "words_mean": 0,
            "letter_ratio_mean": 0,
            "weird_ratio_mean": 0,
        }
    chars = [e["chars"] for e in entries]
    lines = [e["lines"] for e in entries]
    words = [e["words"] for e in entries]
    letter_ratio = [e["letter_ratio"] for e in entries]
    weird_ratio = [e["weird_ratio"] for e in entries]
    return {
        "files": len(entries),
        "chars_total": sum(chars),
        "chars_mean": statistics.mean(chars),
        "chars_median": quantile(chars, 0.5),
        "chars_p95": quantile(chars, 0.95),
        "lines_mean": statistics.mean(lines),
        "words_mean": statistics.mean(words),
        "letter_ratio_mean": statistics.mean(letter_ratio),
        "weird_ratio_mean": statistics.mean(weird_ratio),
    }


def copy_sample(input_dir: Path, sample_dir: Path, max_files: int, seed: int) -> List[Tuple[Path, Path]]:
    ensure_parent_dir(sample_dir / "dummy")
    all_txt = list(iter_txt_files([input_dir]))
    if not all_txt:
        return []
    rng = random.Random(seed)
    rng.shuffle(all_txt)
    chosen = all_txt[:max_files]
    copied = []
    for src in chosen:
        rel = src.relative_to(input_dir)
        dst = sample_dir / rel
        ensure_parent_dir(dst)
        shutil.copy2(src, dst)
        copied.append((src, dst))
    return copied


def parse_stage_summary(log_path: Path) -> Dict[str, int]:
    """Parse simple 'Key : value' lines from stage logs."""
    summary: Dict[str, int] = {}
    if not log_path.exists():
        return summary
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower().replace(" ", "_").replace("/", "_")
        if not key:
            continue
        try:
            summary[key] = int(val.strip())
        except ValueError:
            continue
    return summary


def run_stage(cmd: List[str], log_path: Path) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(f"Stage failed, see log: {log_path}")


def main():
    ap = argparse.ArgumentParser(description="Preview the book pipeline on a small subset.")
    ap.add_argument("input_dir", help="Directory containing raw books (.txt).")
    ap.add_argument("--max_files", type=int, default=10, help="Number of files to sample.")
    ap.add_argument("--result_dir", default="data/books_preview", help="Where to place the sample, stage outputs, and report.")
    ap.add_argument("--sample_seed", type=int, default=0, help="Seed for random sampling of files.")
    ap.add_argument("--langs", default="en", help="Comma-separated languages for stage3 (empty to disable lang filter).")
    ap.add_argument("--dup_chunk_fraction", type=float, default=0.4, help="Stage4 drop threshold for duplicate/near-duplicate chunks.")
    ap.add_argument("--chunk_size", type=int, default=4000, help="Stage4 chunk size.")
    ap.add_argument("--chunk_overlap", type=int, default=400, help="Stage4 chunk overlap.")
    ap.add_argument("--enable_lsh", action="store_true", help="Use MinHash+LSH for near-duplicate chunk detection (requires datasketch).")
    ap.add_argument("--lsh_threshold", type=float, default=0.8, help="LSH Jaccard threshold.")
    ap.add_argument("--num_perm", type=int, default=128, help="MinHash permutations.")
    ap.add_argument("--shingle_size", type=int, default=5, help="Character shingle size for MinHash.")
    ap.add_argument("--workers", type=int, default=4, help="Thread workers passed to stages.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing result_dir.")
    ap.add_argument("--run_stage5_llm", action="store_true", help="Run stage5 LLM boilerplate trimming.")
    ap.add_argument("--llm_model", default="openai/gpt-oss-20b", help="Stage5 LLM model id.")
    ap.add_argument("--llm_api_base", default="http://localhost:8000/v1", help="Stage5 LLM API base URL.")
    ap.add_argument("--llm_api_key", default="dummy", help="Stage5 LLM API key.")
    ap.add_argument("--llm_timeout", type=float, default=30.0, help="Stage5 LLM timeout (s).")
    ap.add_argument("--llm_head_chars", type=int, default=4000, help="Stage5 chars from start to check.")
    ap.add_argument("--llm_tail_chars", type=int, default=4000, help="Stage5 chars from end to check.")
    ap.add_argument("--llm_min_chars", type=int, default=800, help="Drop docs below this after stage5 trim.")
    ap.add_argument("--llm_max_tokens", type=int, default=64, help="Max tokens for stage5 LLM response.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    result_dir = Path(args.result_dir).expanduser()
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if result_dir.exists() and not args.overwrite:
        raise SystemExit(f"Result dir already exists: {result_dir}. Use --overwrite to replace.")
    if result_dir.exists() and args.overwrite:
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = result_dir / "sample_raw"
    stage1_dir = result_dir / "stage1_norm"
    stage2_dir = result_dir / "stage2_struct"
    stage3_dir = result_dir / "stage3_langfilter"
    stage4_dir = result_dir / "stage4_dedup"
    stage5_dir = result_dir / "stage5_llm"
    aggregate_out = result_dir / "aggregate.txt"

    print(f"Sampling first {args.max_files} .txt files from {input_dir} ...")
    copied = copy_sample(input_dir, sample_dir, args.max_files, args.sample_seed)
    if not copied:
        raise SystemExit("No .txt files found to sample.")
    print(f"Copied {len(copied)} files into {sample_dir} (seed={args.sample_seed})")

    base = Path(__file__).resolve().parent
    logs_dir = result_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_stage(
        [
            sys.executable,
            str(base / "stage1_ingest_normalize.py"),
            str(sample_dir),
            "--output_dir",
            str(stage1_dir),
            "--workers",
            str(args.workers),
            "--overwrite",
        ],
        logs_dir / "stage1.log",
    )
    run_stage(
        [
            sys.executable,
            str(base / "stage2_structural_clean.py"),
            str(stage1_dir),
            "--output_dir",
            str(stage2_dir),
            "--workers",
            str(args.workers),
            "--overwrite",
        ],
        logs_dir / "stage2.log",
    )
    run_stage(
        [
            sys.executable,
            str(base / "stage3_language_filter.py"),
            str(stage2_dir),
            "--output_dir",
            str(stage3_dir),
            "--langs",
            args.langs,
            "--workers",
            str(args.workers),
            "--overwrite",
        ],
        logs_dir / "stage3.log",
    )

    stage4_cmd = [
        sys.executable,
        str(base / "stage4_dedup.py"),
        str(stage3_dir),
        "--output_dir",
        str(stage4_dir),
        "--aggregate_out",
        str(aggregate_out),
        "--chunk_size",
        str(args.chunk_size),
        "--chunk_overlap",
        str(args.chunk_overlap),
        "--dup_chunk_fraction",
        str(args.dup_chunk_fraction),
        "--workers",
        str(args.workers),
        "--overwrite",
    ]
    if args.enable_lsh:
        stage4_cmd += [
            "--enable_lsh",
            "--lsh_threshold",
            str(args.lsh_threshold),
            "--num_perm",
            str(args.num_perm),
            "--shingle_size",
            str(args.shingle_size),
        ]
    run_stage(stage4_cmd, logs_dir / "stage4.log")

    final_dir = stage4_dir
    if args.run_stage5_llm:
        run_stage(
            [
                sys.executable,
                str(base / "stage5_llm_boilerplate.py"),
                str(stage4_dir),
                "--output_dir",
                str(stage5_dir),
                "--head_chars",
                str(args.llm_head_chars),
                "--tail_chars",
                str(args.llm_tail_chars),
                "--min_chars",
                str(args.llm_min_chars),
                "--llm_model",
                args.llm_model,
                "--llm_api_base",
                args.llm_api_base,
                "--llm_api_key",
                args.llm_api_key,
                "--llm_timeout",
                str(args.llm_timeout),
                "--llm_max_tokens",
                str(args.llm_max_tokens),
                "--workers",
                str(args.workers),
                "--overwrite",
            ],
            logs_dir / "stage5.log",
        )
        final_dir = stage5_dir

    stats_raw = dir_summary(sample_dir)
    stats_s1 = dir_summary(stage1_dir)
    stats_s2 = dir_summary(stage2_dir)
    stats_s3 = dir_summary(stage3_dir)
    stats_s4 = dir_summary(stage4_dir)
    stats_s5 = dir_summary(stage5_dir) if args.run_stage5_llm else None
    stage1_summary = parse_stage_summary(logs_dir / "stage1.log")
    stage2_summary = parse_stage_summary(logs_dir / "stage2.log")
    stage3_summary = parse_stage_summary(logs_dir / "stage3.log")
    stage4_summary = parse_stage_summary(logs_dir / "stage4.log")
    stage5_summary = parse_stage_summary(logs_dir / "stage5.log") if args.run_stage5_llm else {}

    # Build per-file comparison raw -> final
    comparison_lines = ["raw_path\tfinal_path\traw_chars\tfinal_chars\traw_lines\tfinal_lines"]
    final_lookup: Dict[str, Path] = {
        str(p.relative_to(final_dir)): p for p in iter_txt_files([final_dir])
    }
    for _, raw_copy in copied:
        rel = str(raw_copy.relative_to(sample_dir))
        raw_info = file_stats(raw_copy)
        final_path = final_lookup.get(rel)
        if final_path:
            final_info = file_stats(final_path)
            comparison_lines.append(
                f"{raw_copy}\t{final_path}\t{raw_info['chars']}\t{final_info['chars']}\t{raw_info['lines']}\t{final_info['lines']}"
            )
        else:
            comparison_lines.append(
                f"{raw_copy}\t<dropped>\t{raw_info['chars']}\t0\t{raw_info['lines']}\t0"
            )
    (result_dir / "file_comparison.tsv").write_text("\n".join(comparison_lines), encoding="utf-8")
    (result_dir / "sample_paths.tsv").write_text(
        "src\tdst\n" + "\n".join(f"{src}\t{dst}" for src, dst in copied), encoding="utf-8"
    )

    report_lines = [
        "=== Preview Report ===",
        f"Input dir            : {input_dir}",
        f"Sampled files        : {len(copied)} -> {sample_dir}",
        f"Stage1 output        : {stage1_dir}",
        f"Stage2 output        : {stage2_dir}",
        f"Stage3 output        : {stage3_dir}",
        f"Stage4 output        : {stage4_dir}",
        f"Stage5 output        : {stage5_dir if args.run_stage5_llm else '<skipped>'}",
        f"Aggregate corpus     : {aggregate_out}",
        "",
        "Counts (files):",
        f"  Raw     : {int(stats_raw['files'])}",
        f"  Stage1  : {int(stats_s1['files'])}",
        f"  Stage2  : {int(stats_s2['files'])}",
        f"  Stage3  : {int(stats_s3['files'])}",
        f"  Stage4  : {int(stats_s4['files'])}",
        *( [f"  Stage5  : {int(stats_s5['files'])}"] if args.run_stage5_llm and stats_s5 else [] ),
        "",
        "Chars total:",
        f"  Raw     : {int(stats_raw['chars_total'])}",
        f"  Stage2  : {int(stats_s2['chars_total'])}",
        f"  Stage4  : {int(stats_s4['chars_total'])}",
        *( [f"  Stage5  : {int(stats_s5['chars_total'])}"] if args.run_stage5_llm and stats_s5 else [] ),
        "",
        "Chars mean / median / p95:",
        f"  Raw     : {stats_raw['chars_mean']:.1f} / {stats_raw['chars_median']:.1f} / {stats_raw['chars_p95']:.1f}",
        f"  Stage2  : {stats_s2['chars_mean']:.1f} / {stats_s2['chars_median']:.1f} / {stats_s2['chars_p95']:.1f}",
        f"  Stage4  : {stats_s4['chars_mean']:.1f} / {stats_s4['chars_median']:.1f} / {stats_s4['chars_p95']:.1f}",
        *(
            [f"  Stage5  : {stats_s5['chars_mean']:.1f} / {stats_s5['chars_median']:.1f} / {stats_s5['chars_p95']:.1f}"]
            if args.run_stage5_llm and stats_s5
            else []
        ),
        "",
        "Letter ratio mean:",
        f"  Raw     : {stats_raw['letter_ratio_mean']:.3f}",
        f"  Stage2  : {stats_s2['letter_ratio_mean']:.3f}",
        f"  Stage4  : {stats_s4['letter_ratio_mean']:.3f}",
        *( [f"  Stage5  : {stats_s5['letter_ratio_mean']:.3f}"] if args.run_stage5_llm and stats_s5 else [] ),
        "",
        "Weird ratio mean:",
        f"  Raw     : {stats_raw['weird_ratio_mean']:.4f}",
        f"  Stage2  : {stats_s2['weird_ratio_mean']:.4f}",
        f"  Stage4  : {stats_s4['weird_ratio_mean']:.4f}",
        *( [f"  Stage5  : {stats_s5['weird_ratio_mean']:.4f}"] if args.run_stage5_llm and stats_s5 else [] ),
        "",
        "Stage drop reasons (from logs):",
        f"  Stage1: {stage1_summary}",
        f"  Stage2: {stage2_summary}",
        f"  Stage3: {stage3_summary}",
        f"  Stage4: {stage4_summary}",
        f"  Stage5: {stage5_summary if args.run_stage5_llm else '<skipped>'}",
        "",
        "See file_comparison.tsv for per-file before/after sizes.",
        "Sample list saved to sample_paths.tsv.",
        "Stage logs are under logs/ (stage1.log ... stage5.log).",
    ]
    (result_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Done. Report: {result_dir / 'report.txt'}")
    print(f"Per-file comparison: {result_dir / 'file_comparison.tsv'}")
    print(f"Stage outputs under: {result_dir}")


if __name__ == "__main__":
    main()
