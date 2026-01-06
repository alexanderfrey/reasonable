#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

EVAL_PATTERNS = {
    "eval": re.compile(
        r"Eval \(eval split\): lm_loss=(?P<lm_loss>[0-9.]+), "
        r"ret_benefit=(?P<ret_benefit>[+-]?[0-9.]+), "
        r"ret_rate=(?P<ret_rate>[0-9.]+)%?, "
        r"acc=(?P<acc>[0-9.]+), "
        r"memory=(?P<memory>[0-9]+)"
    ),
    "train": re.compile(
        r"Eval \(train split\): lm_loss=(?P<lm_loss>[0-9.]+), "
        r"ret_benefit=(?P<ret_benefit>[+-]?[0-9.]+), "
        r"ret_rate=(?P<ret_rate>[0-9.]+)%?, "
        r"acc=(?P<acc>[0-9.]+), "
        r"memory=(?P<memory>[0-9]+)"
    ),
    "fresh": re.compile(
        r"Eval \(fresh memory\): lm_loss=(?P<lm_loss>[0-9.]+), "
        r"ret_benefit=(?P<ret_benefit>[+-]?[0-9.]+), "
        r"ret_rate=(?P<ret_rate>[0-9.]+)%?, "
        r"acc=(?P<acc>[0-9.]+), "
        r"memory=(?P<memory>[0-9]+)"
    ),
}

SPLIT_ORDER = ["eval", "train", "fresh"]


def parse_condition_seed(log_path: Path):
    parent = log_path.parent.name
    match = re.match(r"(?P<cond>.+)_seed(?P<seed>\d+)$", parent)
    if match:
        return match.group("cond"), match.group("seed")
    return parent, ""


def parse_log_file(log_path: Path):
    latest = {}
    with log_path.open("r", errors="ignore") as handle:
        for line in handle:
            for split, pattern in EVAL_PATTERNS.items():
                match = pattern.search(line)
                if not match:
                    continue
                metrics = match.groupdict()
                latest[split] = {
                    "lm_loss": float(metrics["lm_loss"]),
                    "ret_benefit": float(metrics["ret_benefit"]),
                    "ret_rate_pct": float(metrics["ret_rate"]),
                    "acc": float(metrics["acc"]),
                    "memory": int(metrics["memory"]),
                }
    return latest


def collect_logs(paths, log_name):
    log_paths = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            log_paths.extend(sorted(p.rglob(log_name)))
        elif p.is_file():
            log_paths.append(p)
    return log_paths


def format_table(rows):
    headers = ["condition", "seed", "split", "lm_loss", "ret_benefit", "ret_rate_%", "acc", "memory"]
    values = [headers]
    for row in rows:
        values.append([
            row["condition"],
            row["seed"],
            row["split"],
            f"{row['lm_loss']:.4f}",
            f"{row['ret_benefit']:+.4f}",
            f"{row['ret_rate_pct']:.2f}",
            f"{row['acc']:.4f}",
            str(row["memory"]),
        ])

    widths = [max(len(v[i]) for v in values) for i in range(len(headers))]
    lines = []
    for idx, row in enumerate(values):
        line = "  ".join(val.ljust(widths[i]) for i, val in enumerate(row))
        if idx == 0:
            lines.append(line)
            lines.append("  ".join("-" * w for w in widths))
        else:
            lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate ablation metrics from training logs.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["memory_book_output_kv/ablation"],
        help="Log files or directories to scan (default: memory_book_output_kv/ablation)",
    )
    parser.add_argument(
        "--log-name",
        default="train.log",
        help="Log filename to search for inside directories (default: train.log)",
    )
    args = parser.parse_args()

    log_paths = collect_logs(args.paths, args.log_name)
    if not log_paths:
        raise SystemExit("No logs found. Check paths or --log-name.")

    rows = []
    for log_path in sorted(log_paths):
        cond, seed = parse_condition_seed(log_path)
        metrics_by_split = parse_log_file(log_path)
        for split in SPLIT_ORDER:
            metrics = metrics_by_split.get(split)
            if not metrics:
                continue
            rows.append({
                "condition": cond,
                "seed": seed,
                "split": split,
                **metrics,
            })

    rows.sort(key=lambda r: (r["condition"], r["seed"], SPLIT_ORDER.index(r["split"])))
    print(format_table(rows))


if __name__ == "__main__":
    main()
