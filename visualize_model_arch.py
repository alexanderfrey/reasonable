#!/usr/bin/env python
"""
Create a Graphviz diagram that summarises the architecture defined in model.py.

Usage examples:
    python visualize_model_arch.py --checkpoint ./gpt_pretrain_output/model_step_1000.pt
    python visualize_model_arch.py --config_json ./config.json --format png

The script only depends on the model configuration (no model instantiation), so it
works even if optional runtime dependencies such as flash-attn are unavailable.
"""

import argparse
import json
import os
import shutil
from typing import Dict, Any

try:
    from graphviz import Digraph
    _GRAPHVIZ_ERROR = None
except Exception as exc:
    Digraph = None
    _GRAPHVIZ_ERROR = exc

import torch


def compute_default_n_kv_head(n_head: int) -> int:
    if n_head <= 0:
        return n_head
    target = max(1, n_head // 4)
    while target > 1 and n_head % target != 0:
        target -= 1
    return max(1, target)


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any]

    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    elif args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = ckpt.get("config")
        if cfg is None:
            raise SystemExit(
                f"Checkpoint '{args.checkpoint}' does not contain a 'config' entry."
            )
    else:
        cfg = {
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
        }

    cfg.setdefault("max_seq_len", args.max_seq_len)
    cfg.setdefault("dropout", 0.0)
    cfg.setdefault("pad_idx", 0)
    cfg.setdefault("d_ff", cfg.get("d_model", 0) * 4)
    cfg.setdefault("rope_theta", 10000.0)
    cfg.setdefault("max_kv_len", None)

    n_head = int(cfg.get("n_head", args.n_head))
    n_kv_head = cfg.get("n_kv_head")
    if n_kv_head is None or n_head % n_kv_head != 0:
        cfg["n_kv_head"] = compute_default_n_kv_head(n_head)
    cfg["n_head"] = n_head
    return cfg


def describe_attention(cfg: Dict[str, Any]) -> str:
    n_head = cfg["n_head"]
    n_kv_head = cfg["n_kv_head"]
    d_model = cfg["d_model"]
    d_k = d_model // max(1, n_head)
    max_kv_len = cfg.get("max_kv_len")
    rope_theta = cfg.get("rope_theta")

    lines = [
        f"MultiHeadAttention",
        f"Q heads: {n_head}",
        f"KV heads: {n_kv_head}",
        f"Head dim: {d_k}",
    ]
    if max_kv_len:
        lines.append(f"KV cache window: {max_kv_len}")
    lines.append(f"RoPE theta: {rope_theta}")
    return "\\n".join(lines)


def describe_ffn(cfg: Dict[str, Any]) -> str:
    return f"SwiGLU FFN\\nHidden dim: {cfg['d_ff']}"


def build_graph(cfg: Dict[str, Any], args: argparse.Namespace) -> Digraph:
    if Digraph is None:
        raise SystemExit(
            "The 'graphviz' Python package is required. Install it with "
            "'pip install graphviz' (and ensure Graphviz binaries are on PATH)."
        ) from _GRAPHVIZ_ERROR

    dot = Digraph("GPT_Architecture", format=args.format)
    dot.attr(rankdir="LR", concentrate="true", splines="spline", fontsize="12")
    dot.attr("node", shape="record", style="filled", fillcolor="#F4F6F8")

    meta = (
        f"GPT Architecture\\n"
        f"d_model={cfg['d_model']} | n_layer={cfg['n_layer']} | "
        f"n_head={cfg['n_head']} | n_kv_head={cfg['n_kv_head']}\\n"
        f"max_seq_len={cfg['max_seq_len']} | vocab={cfg['vocab_size']}"
    )
    dot.attr(label=meta, labelloc="t", fontsize="14", fontname="Helvetica-Bold")

    dot.node("input", f"Input Tokens\\n(max_seq_len {cfg['max_seq_len']})", shape="box", fillcolor="#D8E6F3")
    dot.node("embed", f"TokenEmbedding\\n(vocab {cfg['vocab_size']} → d_model {cfg['d_model']})")
    dot.edge("input", "embed")

    prev = "embed"
    for layer_idx in range(cfg["n_layer"]):
        cluster_name = f"cluster_layer_{layer_idx}"
        with dot.subgraph(name=cluster_name) as sub:
            sub.attr(label=f"TransformerBlock {layer_idx}", fontsize="12", style="filled", color="#E5E9F2")
            sub.attr("node", shape="record", style="filled", fillcolor="#FFFFFF")

            norm_attn = f"norm_attn_{layer_idx}"
            attn = f"attn_{layer_idx}"
            drop_attn = f"drop_attn_{layer_idx}"
            norm_ffn = f"norm_ffn_{layer_idx}"
            ffn = f"ffn_{layer_idx}"
            drop_ffn = f"drop_ffn_{layer_idx}"

            sub.node(norm_attn, "RMSNorm")
            sub.node(attn, describe_attention(cfg))
            sub.node(drop_attn, "Dropout")
            sub.node(norm_ffn, "RMSNorm")
            sub.node(ffn, describe_ffn(cfg))
            sub.node(drop_ffn, "Dropout")

            sub.edge(norm_attn, attn)
            sub.edge(attn, drop_attn)
            sub.edge(drop_attn, norm_ffn)
            sub.edge(norm_ffn, ffn)
            sub.edge(ffn, drop_ffn)

        dot.edge(prev, norm_attn, label="residual", fontsize="10")
        dot.edge(drop_attn, norm_ffn, constraint="true")
        dot.edge(drop_ffn, f"residual_out_{layer_idx}", style="invis")  # keep rank
        prev = drop_ffn

    dot.node("final_norm", "RMSNorm", fillcolor="#FFFFFF")
    dot.node(
        "lm_head",
        f"lm_head (Linear)\\n(d_model {cfg['d_model']} → vocab {cfg['vocab_size']})",
    )
    dot.node("output", "Logits / Softmax", shape="box", fillcolor="#D8E6F3")

    dot.edge(prev, "final_norm")
    dot.edge("final_norm", "lm_head")
    dot.edge("lm_head", "output")

    return dot


def main():
    parser = argparse.ArgumentParser(description="Visualise the GPT architecture defined in model.py.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--checkpoint", type=str, help="Path to a .pt checkpoint containing a 'config' entry.")
    src.add_argument("--config_json", type=str, help="Path to a JSON file with the model configuration.")

    parser.add_argument("--output", type=str, default="gpt_architecture", help="Output file stem (without extension).")
    parser.add_argument("--format", type=str, default="pdf", help="Graphviz output format (pdf, png, svg, etc.).")
    parser.add_argument("--view", action="store_true", help="Open the rendered file with the default viewer.")

    # fallback config values if neither checkpoint nor config JSON supplied
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=2048)

    args = parser.parse_args()
    cfg = load_config(args)

    dot = build_graph(cfg, args)
    if shutil.which("dot") is None:
        raise SystemExit(
            "Graphviz executable 'dot' not found on PATH. Install Graphviz from "
            "https://graphviz.org/download/ and ensure the 'dot' binary is accessible."
        )
    out_path = dot.render(filename=args.output, cleanup=True, view=args.view)
    print(f"[visualize_model_arch] wrote diagram to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
