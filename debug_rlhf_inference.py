#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility script that mirrors the RLHF training dataloader + sampling path so we can
reproduce the exact generations (e.g. 2 samples per prompt for GRPO) without
running the full training loop. The goal is to inspect the decoded text, token
counts, and per-token log-probs in the same order that the trainer sees them.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import textwrap
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from model import GPT
from rlhf_train_grpo_dapo import (
    PromptDataset,
    apply_lora_adapters,
    compute_default_n_kv_head,
    generate_and_logprobs,
    load_base,
    load_tokenizer,
    prompt_collate_fn,
    _load_checkpoint_config,
    _maybe_load_lora_adapters,
    _sync_args_with_config,
    logger as train_logger,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(pref: str) -> torch.device:
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested via --device but not available.")
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_policy(args, tok, device: torch.device) -> GPT:
    base_cfg = _load_checkpoint_config(args.base_checkpoint)
    ref_cfg = _load_checkpoint_config(args.ref_checkpoint) if args.ref_checkpoint else None
    if base_cfg:
        _sync_args_with_config(args, base_cfg)
    elif ref_cfg:
        train_logger.info("Falling back to reference checkpoint config for model dims.")
        _sync_args_with_config(args, ref_cfg)

    if args.n_kv_head is None:
        if base_cfg and base_cfg.get("n_kv_head"):
            args.n_kv_head = base_cfg["n_kv_head"]
        else:
            args.n_kv_head = compute_default_n_kv_head(args.n_head)
            train_logger.info("Auto-selected n_kv_head=%d", args.n_kv_head)
    if args.d_ff is None:
        args.d_ff = args.d_model * 4

    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    model_cfg = {
        "vocab_size": len(tok),
        "d_model": args.d_model,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
        "n_kv_head": args.n_kv_head,
        "max_seq_len": args.max_seq_len,
        "dropout": args.dropout,
        "pad_idx": pad_id,
        "d_ff": args.d_ff,
    }

    lora_adapters = _maybe_load_lora_adapters(args.base_checkpoint)
    policy = GPT(**model_cfg)
    policy_base_path = args.base_checkpoint
    if lora_adapters:
        policy_base_path = args.policy_base_checkpoint or args.ref_checkpoint
        if not policy_base_path:
            raise SystemExit(
                "LoRA adapter checkpoint detected but no --policy_base_checkpoint/--ref_checkpoint provided."
            )
        if policy_base_path == args.ref_checkpoint:
            train_logger.info("Using --ref_checkpoint as the base weights for the policy.")

    load_base(policy, policy_base_path, device)
    replaced = apply_lora_adapters(
        policy,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        include_regex=args.lora_include,
        exclude_regex=args.lora_exclude or r"(token_embedding|lm_head)",
    )
    train_logger.info("LoRA wrappers attached to %d linear modules.", len(replaced))

    if lora_adapters:
        missing, unexpected = policy.load_state_dict(lora_adapters, strict=False)
        if missing:
            train_logger.warning("[load-adapters] missing keys: %s", missing[:8])
        if unexpected:
            train_logger.warning("[load-adapters] unexpected keys: %s", unexpected[:8])

    policy.eval()
    return policy


def iter_manual_prompts(prompts: Sequence[str], batch_size: int) -> Iterable[Dict[str, List[Optional[str]]]]:
    buf: List[str] = []
    for prompt in prompts:
        if not prompt:
            continue
        buf.append(prompt)
        if len(buf) == batch_size:
            yield {"prompt_text": buf[:], "ref_response": [None] * len(buf)}
            buf.clear()
    if buf:
        yield {"prompt_text": buf[:], "ref_response": [None] * len(buf)}


def resolve_prompt_iterator(args) -> Iterable[Dict[str, List[Optional[str]]]]:
    manual_prompts: List[str] = []
    if args.prompt:
        manual_prompts.extend(args.prompt)
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    manual_prompts.append(line)

    if manual_prompts:
        train_logger.info("Using %d manual prompts.", len(manual_prompts))
        return iter_manual_prompts(manual_prompts, args.batch_size)

    prompt_source = args.training_dir or args.train_jsonl
    if not prompt_source:
        raise SystemExit(
            "Provide --prompt, --prompt_file, --training_dir, or --train_jsonl so we know what to decode."
        )

    ds = PromptDataset(
        prompt_source,
        prompt_template=args.prompt_template,
        strip_think=args.strip_think,
        system_prompt=args.system_prompt,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=prompt_collate_fn,
    )


def expand_prompts(prompts: Sequence[str], group_size: int, algo: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    mapping: List[Tuple[int, int]] = []
    expanded: List[str] = []
    for idx, prompt in enumerate(prompts):
        repeat = group_size if algo == "grpo" else 1
        for slot in range(repeat):
            expanded.append(prompt)
            mapping.append((idx, slot))
    return expanded, mapping


def regroup_generations(
    mapping: Sequence[Tuple[int, int]], outputs: Sequence[str], token_ids: Sequence[Sequence[int]], logps
) -> Dict[int, List[Dict[str, object]]]:
    grouped: Dict[int, List[Dict[str, object]]] = {}
    for gen_idx, (prompt_idx, slot_idx) in enumerate(mapping):
        stats = {
            "slot": slot_idx,
            "global_index": gen_idx,
            "token_count": len(token_ids[gen_idx]),
            "avg_logprob": None,
            "sum_logprob": None,
            "text": outputs[gen_idx],
            "token_ids": token_ids[gen_idx],
        }
        if logps and gen_idx < len(logps) and logps[gen_idx].numel() > 0:
            lp = logps[gen_idx]
            stats["avg_logprob"] = float(lp.mean().item())
            stats["sum_logprob"] = float(lp.sum().item())
        grouped.setdefault(prompt_idx, []).append(stats)
    for entries in grouped.values():
        entries.sort(key=lambda item: item["slot"])
    return grouped


def preview(text: str, limit: int = 240) -> str:
    clean = text.replace("\n", " ").strip()
    return textwrap.shorten(clean, width=limit, placeholder="…")


def dump_records(handle, batch_idx: int, global_prompt_idx: int, prompt_text: str, ref_text: Optional[str], records):
    for entry in records:
        obj = {
            "batch_index": batch_idx,
            "prompt_index": global_prompt_idx,
            "prompt_text": prompt_text,
            "ref_response": ref_text,
            "slot": entry["slot"],
            "token_count": entry["token_count"],
            "avg_logprob": entry["avg_logprob"],
            "sum_logprob": entry["sum_logprob"],
            "token_ids": entry["token_ids"],
            "output_text": entry["text"],
        }
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
        handle.flush()


def run_debug_loop(args):
    device = choose_device(args.device)
    train_logger.info("Using device: %s", device)
    set_seed(args.seed)

    tok = load_tokenizer(args.tokenizer_name)
    policy = build_policy(args, tok, device)

    prompt_iter = resolve_prompt_iterator(args)
    dump_handle = open(args.dump_jsonl, "a", encoding="utf-8") if args.dump_jsonl else None

    total_prompts = 0
    total_generations = 0
    batch_idx = 0
    try:
        for batch in prompt_iter:
            prompts = batch["prompt_text"]
            refs = batch.get("ref_response") or [None] * len(prompts)
            if not prompts:
                continue

            if args.max_prompts is not None:
                remaining = args.max_prompts - total_prompts
                if remaining <= 0:
                    break
                if len(prompts) > remaining:
                    prompts = prompts[:remaining]
                    refs = refs[:remaining]

            expanded_prompts, mapping = expand_prompts(prompts, args.group_size, args.algo)
            gen_texts, gen_ids, logps = generate_and_logprobs(
                policy,
                tok,
                device,
                expanded_prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                temperature_floor=args.temperature_floor,
                suppress_eos_steps=args.suppress_eos_steps,
                eos_id=tok.eos_token_id,
                logprob_chunk_size=args.logprob_chunk_size,
                amp_enabled=args.use_amp,
                amp_dtype=torch.bfloat16,
                backprop_context_len=args.backprop_context_len,
            )
            grouped = regroup_generations(mapping, gen_texts, gen_ids, logps)

            print("=" * 80)
            print(f"Batch {batch_idx} | prompts={len(prompts)} | expanded_generations={len(expanded_prompts)}")
            for local_idx, prompt in enumerate(prompts):
                global_idx = total_prompts + local_idx
                ref_text = refs[local_idx] if refs else None
                print("-" * 80)
                print(f"[Prompt {global_idx}]")
                print(f"Prompt preview : {preview(prompt)}")
                if ref_text:
                    print(f"Reference      : {preview(ref_text)}")
                records = grouped.get(local_idx, [])
                if not records:
                    print("No generations recorded for this prompt.")
                    continue
                for entry in records:
                    avg = entry["avg_logprob"]
                    avg_str = f"{avg:.4f}" if avg is not None else "n/a"
                    print(
                        f"  -> Gen slot {entry['slot']} | tokens={entry['token_count']} | avg_logprob={avg_str} | "
                        f"global_idx={entry['global_index']}"
                    )
                    print(textwrap.indent(preview(entry["text"], limit=args.output_preview_chars), prefix="     "))
                if dump_handle:
                    dump_records(dump_handle, batch_idx, global_idx, prompt, ref_text, records)

            total_prompts += len(prompts)
            total_generations += len(expanded_prompts)
            batch_idx += 1

            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
        print("=" * 80)
        print(
            f"Completed debug run | prompts={total_prompts} | generations={total_generations} | batches={batch_idx}"
        )
    finally:
        if dump_handle:
            dump_handle.close()


def parse_args():
    ap = argparse.ArgumentParser(
        description="Mirror the RLHF trainer's sampling loop to debug multi-sample generations."
    )
    # Prompt sources
    ap.add_argument("--training_dir", type=str, default=None, help="Directory of *.jsonl files used during training.")
    ap.add_argument("--train_jsonl", type=str, default=None, help="Single JSONL file with instructions.")
    ap.add_argument("--prompt", action="append", help="Literal prompt text (final form, template not applied).")
    ap.add_argument("--prompt_file", type=str, default=None, help="Plain text file (one prompt per line).")
    ap.add_argument("--prompt_template", type=str, default="<|user|>\n{prompt}\n<|assistant|>\n")
    ap.add_argument("--system_prompt", type=str, default=None)
    ap.add_argument("--strip_think", action="store_true", help="Drop <think> blocks from ref responses.")

    # Model / tokenizer
    ap.add_argument("--base_checkpoint", type=str, required=True)
    ap.add_argument("--policy_base_checkpoint", type=str, default=None)
    ap.add_argument("--ref_checkpoint", type=str, default=None)
    ap.add_argument("--tokenizer_name", type=str, default="gpt2")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_head", type=int, default=16)
    ap.add_argument("--n_layer", type=int, default=24)
    ap.add_argument("--n_kv_head", type=int, default=None)
    ap.add_argument("--d_ff", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or explicit cuda device.")

    # LoRA config
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_include", type=str, default=r"(attn|Attention|q_proj|k_proj|v_proj|o_proj|mlp|fc)")
    ap.add_argument("--lora_exclude", type=str, default=None)

    # Sampling mirror
    ap.add_argument("--algo", type=str, choices=["dapo", "grpo"], default="grpo",
                    help="Use 'grpo' to duplicate prompts group_size times like the trainer.")
    ap.add_argument("--group_size", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--frequency_penalty", type=float, default=0.0)
    ap.add_argument("--presence_penalty", type=float, default=0.0)
    ap.add_argument("--temperature_floor", type=float, default=0.2)
    ap.add_argument("--suppress_eos_steps", type=int, default=0)
    ap.add_argument("--logprob_chunk_size", type=int, default=1)
    ap.add_argument("--backprop_context_len", type=int, default=0)
    ap.add_argument("--use_amp", action="store_true")

    # Runtime / logging
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--max_prompts", type=int, default=None, help="Hard stop after this many prompts.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dump_jsonl", type=str, default=None, help="Optional path to write JSONL debug records.")
    ap.add_argument("--output_preview_chars", type=int, default=240)

    args = ap.parse_args()
    if args.group_size < 1:
        raise SystemExit("--group_size must be >= 1.")
    if args.algo == "dapo" and args.group_size != 1:
        train_logger.warning("DAPO selected but group_size != 1; prompts will not be duplicated.")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    run_debug_loop(parse_args())


if __name__ == "__main__":
    main()
