#!/usr/bin/env python3
"""
Stage 5: LLM-assisted boilerplate trimming
- Sends head/tail excerpts of each book to an OpenAI-compatible endpoint.
- If the LLM flags boilerplate, trims the corresponding prefix/suffix.
- Writes kept files to --output_dir; drops docs that end up too short.
"""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple

from book_pipeline.common import auto_read_text, ensure_parent_dir, is_binary_file, iter_txt_files

LLM_PROMPT = (
    "You are filtering book text for pretraining. "
    "Given an excerpt from the START or END of a book, decide if it is boilerplate (licenses, copyright notices, "
    "disclaimers, table of contents, editor notes, metadata, ads, or Project Gutenberg front/back matter). "
    'Respond with JSON: {"boilerplate": bool, "reason": "short note"}. '
    "Be conservative: only mark boilerplate when it is clearly not main book content."
)


def make_llm_client(api_base: str, api_key: str):
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "The 'openai' package is required for stage5 LLM filtering.\nInstall with `pip install openai` (>=1.0)."
        ) from exc
    return OpenAI(base_url=api_base, api_key=api_key)


def make_llm_classifier(client, model: str, timeout: float, max_tokens: int, client_lock: Lock):
    cache: Dict[str, bool] = {}

    def classify(window: str, position: str) -> bool:
        if not window.strip():
            return False
        key = f"{position}:{hashlib.blake2s(window.encode('utf-8')).hexdigest()}"
        if key in cache:
            return cache[key]
        snippet = window.strip()
        try:
            with client_lock:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": LLM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Position: {position.upper()}\nExcerpt:\n```\n{snippet}\n```",
                        },
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    temperature=0,
                    timeout=timeout,
                )
            content = completion.choices[0].message.content or "{}"
            data = json.loads(content)
            result = bool(data.get("boilerplate", False))
        except Exception:
            result = False
        cache[key] = result
        return result

    return classify


def trim_boilerplate(text: str, head_flag: bool, tail_flag: bool, head_chars: int, tail_chars: int) -> str:
    trimmed = text
    if head_flag:
        cut = min(len(trimmed), head_chars)
        next_para = trimmed.find("\n\n", cut)
        if next_para != -1:
            cut = next_para + 2
        trimmed = trimmed[cut:]
    if tail_flag and trimmed:
        cut_from = max(0, len(trimmed) - tail_chars)
        prev_para = trimmed.rfind("\n\n", 0, cut_from)
        if prev_para != -1:
            cut_from = prev_para
        trimmed = trimmed[:cut_from]
    return trimmed.strip() + ("\n" if trimmed.strip() else "")


def process_file(
    path: Path,
    input_root: Path,
    output_root: Path,
    classify_fn,
    head_chars: int,
    tail_chars: int,
    min_chars: int,
    overwrite: bool,
) -> Tuple[str, str]:
    rel = path.relative_to(input_root)
    out_path = output_root / rel
    if not overwrite and out_path.exists():
        return ("skipped_existing", "")
    if is_binary_file(path):
        return ("binary", "")
    try:
        text = auto_read_text(path)
    except Exception as exc:
        return (f"error:{exc}", "")
    if len(text) < min_chars:
        return ("too_short", "")

    head_excerpt = text[:head_chars]
    tail_excerpt = text[-tail_chars:] if tail_chars > 0 else ""
    head_boiler = classify_fn(head_excerpt, "start") if head_excerpt else False
    tail_boiler = classify_fn(tail_excerpt, "end") if tail_excerpt else False

    trimmed = trim_boilerplate(text, head_boiler, tail_boiler, head_chars, tail_chars)
    if len(trimmed.strip()) < min_chars:
        return ("too_short", "")

    ensure_parent_dir(out_path)
    out_path.write_text(trimmed, encoding="utf-8")
    reason = []
    if head_boiler:
        reason.append("head_boilerplate")
    if tail_boiler:
        reason.append("tail_boilerplate")
    return ("written", "|".join(reason))


def main():
    ap = argparse.ArgumentParser(description="Stage 5: LLM-assisted boilerplate trimming (head/tail excerpts).")
    ap.add_argument("input_dir", help="Directory with stage4 outputs.")
    ap.add_argument("--output_dir", default="data/books_stage5_llmfilter", help="Where to write LLM-trimmed files.")
    ap.add_argument("--head_chars", type=int, default=4000, help="Characters from the start to send to the LLM.")
    ap.add_argument("--tail_chars", type=int, default=4000, help="Characters from the end to send to the LLM.")
    ap.add_argument("--min_chars", type=int, default=800, help="Drop documents below this after trimming.")
    ap.add_argument("--llm_model", default="openai/gpt-oss-20b", help="Model id for the LLM endpoint.")
    ap.add_argument("--llm_api_base", default="http://localhost:8000/v1", help="Base URL for the LLM endpoint.")
    ap.add_argument("--llm_api_key", default="dummy", help="API key for the LLM endpoint.")
    ap.add_argument("--llm_timeout", type=float, default=30.0, help="Timeout (s) for each LLM call.")
    ap.add_argument("--llm_max_tokens", type=int, default=64, help="Max tokens in LLM JSON response.")
    ap.add_argument("--workers", type=int, default=4, help="Thread pool workers.")
    ap.add_argument("--overwrite", action="store_true", help="Rewrite files even if output exists.")
    args = ap.parse_args()

    input_root = Path(args.input_dir).expanduser()
    output_root = Path(args.output_dir).expanduser()
    if not input_root.exists():
        raise SystemExit(f"Input directory does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    client = make_llm_client(args.llm_api_base, args.llm_api_key)
    client_lock = Lock()
    classify_fn = make_llm_classifier(client, args.llm_model, args.llm_timeout, args.llm_max_tokens, client_lock)

    stats = {
        "written": 0,
        "skipped_existing": 0,
        "binary": 0,
        "too_short": 0,
        "errors": 0,
        "head_boilerplate": 0,
        "tail_boilerplate": 0,
    }

    def task(p: Path) -> Tuple[str, str]:
        return process_file(
            p,
            input_root,
            output_root,
            classify_fn,
            args.head_chars,
            args.tail_chars,
            args.min_chars,
            args.overwrite,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for status, detail in ex.map(task, iter_txt_files([input_root]), chunksize=8):
            if status.startswith("error:"):
                stats["errors"] += 1
                print(f"[SKIP] {status}")
            elif status == "written":
                stats["written"] += 1
                if "head_boilerplate" in detail:
                    stats["head_boilerplate"] += 1
                if "tail_boilerplate" in detail:
                    stats["tail_boilerplate"] += 1
            else:
                stats[status] = stats.get(status, 0) + 1

    print("=== Stage 5 summary ===")
    print(f"Written               : {stats['written']}")
    print(f"Skipped existing      : {stats['skipped_existing']}")
    print(f"Skipped binary        : {stats['binary']}")
    print(f"Too short             : {stats['too_short']} (<{args.min_chars})")
    print(f"Errors                : {stats['errors']}")
    print(f"Head boilerplate trims: {stats['head_boilerplate']}")
    print(f"Tail boilerplate trims: {stats['tail_boilerplate']}")
    print(f"Output dir            : {output_root}")


if __name__ == "__main__":
    main()
