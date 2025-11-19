#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick inference for your custom GPT + optional LoRA adapters.

Key additions:
- Mixed precision (bf16 if supported, else fp16). Override via --amp_dtype.
- Optional weight casting to AMP dtype (default on) for lower VRAM.
- Token streaming enabled by default; disable with --no_stream.

Examples:

# One-shot (base only), stream to console
python infer_llm.py \
  --base_checkpoint ./gpt_pretrain_output_hf/model_final_slim.pt \
  --tokenizer_name gpt2 \
  --prompt "Explain transformers to a 12-year-old." \
  --max_new_tokens 256 --temperature 0.7 --top_p 0.95

# Base + LoRA adapters, non-streamed output
python infer_llm.py \
  --base_checkpoint ./gpt_pretrain_output_hf/model_final_slim.pt \
  --adapters ./gpt_lora_sft/lora_adapters_step5000_final.pt \
  --tokenizer_name gpt2 \
  --prompt "Write a short Python function that reverses a list." \
  --max_new_tokens 200 --temperature 0.6 --top_p 0.9 --no_stream
"""

import os, re, json, argparse, logging
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# --- Project-local model
try:
    from model import GPT
except Exception as e:
    raise SystemExit(f"model.py not found or failed to import: {e}")

# Torch perf
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("infer_llm")


def compute_default_n_kv_head(n_head: int, kv_ratio: int = 4) -> int:
    if n_head <= 0:
        return n_head
    target = max(1, n_head // max(1, kv_ratio))
    while target > 1 and n_head % target != 0:
        target -= 1
    return max(1, target)


# ---------------- LoRA wrapper (must match training) ----------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = base.bias
        self.weight = base.weight
        self.r = r
        self.lora_scaling = alpha / r if r > 0 else 1.0
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)
        # freeze base
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.r <= 0:
            return base_out
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.lora_scaling
        return base_out + lora_out

def _wrap_linear_with_lora(module: nn.Module, name: str,
                           include_regex: Optional[str], exclude_regex: Optional[str],
                           r: int, alpha: int, dropout: float, replaced: List[str]):
    for child_name, child in list(module.named_children()):
        full = f"{name}.{child_name}" if name else child_name
        _wrap_linear_with_lora(child, full, include_regex, exclude_regex, r, alpha, dropout, replaced)
        if isinstance(child, nn.Linear):
            ok = True
            if include_regex and not re.search(include_regex, full):
                ok = False
            if exclude_regex and re.search(exclude_regex, full):
                ok = False
            if ok:
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced.append(full)

def apply_lora_skeleton(model: nn.Module, cfg: Dict) -> List[str]:
    include_regex = cfg.get("include") or cfg.get("lora_include") or r"(attn|Attention|q_proj|k_proj|v_proj|o_proj|mlp|fc)"
    exclude_regex = cfg.get("exclude") or cfg.get("lora_exclude")
    r = int(cfg.get("lora_r", 8))
    alpha = int(cfg.get("lora_alpha", 16))
    dropout = float(cfg.get("lora_dropout", 0.0))
    replaced: List[str] = []
    _wrap_linear_with_lora(model, "", include_regex, exclude_regex, r, alpha, dropout, replaced)
    return replaced

def load_lora_adapters(model: nn.Module, adapters_path: str, device: torch.device):
    ckpt = torch.load(adapters_path, map_location="cpu")
    if "adapters_state_dict" in ckpt:
        ad_cfg = ckpt.get("config", {})
        replaced = apply_lora_skeleton(model, ad_cfg)
        log.info(f"Applied LoRA skeleton to {len(replaced)} Linear modules.")
        sd = ckpt["adapters_state_dict"]
        model.to(device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # Base checkpoint params aren't included in the adapters file, so ignore
        # those entries and only warn if actual LoRA tensors are absent.
        missing_lora = [k for k in missing if ".lora_" in k]
        unexpected_lora = [k for k in unexpected if ".lora_" in k]
        if missing_lora:
            log.warning(f"[LoRA load] Missing keys: {missing_lora[:10]}{'...' if len(missing_lora)>10 else ''}")
        if unexpected_lora:
            log.warning(f"[LoRA load] Unexpected keys: {unexpected_lora[:10]}{'...' if len(unexpected_lora)>10 else ''}")
        for p in model.parameters():
            p.requires_grad = False
    else:
        raise ValueError("Adapters file does not contain 'adapters_state_dict' (unexpected format).")

# ---------------- Tokenizer ----------------
def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token": "[EOS]"})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        tok.model_max_length = int(1e12)
    except Exception:
        pass
    return tok

# ---------------- Model load ----------------
def build_model(config: Dict) -> nn.Module:
    model = GPT(**config)
    if hasattr(model, "lm_head") and hasattr(model, "token_embedding"):
        try:
            model.lm_head.weight = model.token_embedding.weight
        except Exception:
            pass
    return model

def load_base_or_merged(args, vocab_size: int, pad_idx: int, device: torch.device) -> Tuple[nn.Module, Dict]:
    if args.merged_full_model:
        log.info(f"Loading merged full model from {args.merged_full_model}")
        ckpt = torch.load(args.merged_full_model, map_location="cpu")
        config = ckpt.get("config", None)
        if config is None:
            raise RuntimeError("Merged full model missing 'config' field.")
        config["vocab_size"] = vocab_size
        config["pad_idx"] = pad_idx
        if "n_kv_head" not in config or config["n_kv_head"] is None:
            kv_ratio = 4
            if config.get("n_kv_head"):
                kv_ratio = max(1, config.get("n_head", args.n_head) // config["n_kv_head"])
            config["n_kv_head"] = compute_default_n_kv_head(config.get("n_head", args.n_head), kv_ratio=kv_ratio)
        model = build_model(config)
        state = ckpt.get("model_state_dict", ckpt)
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning(f"[merged] Missing keys: {missing[:10]}{'...' if len(missing)>10 else ''}")
        if unexpected:
            log.warning(f"[merged] Unexpected keys: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
        model.to(device)
        for p in model.parameters(): p.requires_grad = False
        return model, config

    if args.base_checkpoint is None:
        raise RuntimeError("Provide either --merged_full_model or --base_checkpoint.")
    log.info(f"Loading base checkpoint from {args.base_checkpoint}")
    base_ckpt = torch.load(args.base_checkpoint, map_location="cpu")
    base_state = base_ckpt.get("model_state_dict", base_ckpt)
    base_config = base_ckpt.get("config", None)

    if base_config is None:
        base_config = dict(
            vocab_size=vocab_size,
            d_model=args.d_model, n_head=args.n_head,
            n_kv_head=args.n_kv_head,
            n_layer=args.n_layer, max_seq_len=args.max_seq_len,
            dropout=args.dropout, pad_idx=pad_idx,
            d_ff=args.d_ff or args.d_model * 4,
        )
    else:
        base_config["vocab_size"] = vocab_size
        base_config["pad_idx"] = pad_idx
        base_config.setdefault("max_seq_len", args.max_seq_len)
        base_config.setdefault("d_ff", base_config.get("d_model", 0) * 4)
        if base_config.get("n_kv_head") is None:
            kv_ratio = 4
            if base_config.get("n_kv_head"):
                kv_ratio = max(1, base_config.get("n_head", args.n_head) // base_config["n_kv_head"])
            n_head_cfg = base_config.get("n_head", args.n_head)
            base_config["n_kv_head"] = compute_default_n_kv_head(n_head_cfg, kv_ratio=kv_ratio)

    model = build_model(base_config)
    if any(k.startswith("_orig_mod.") for k in base_state):
        base_state = {k.replace("_orig_mod.", ""): v for k, v in base_state.items()}
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if missing:
        log.warning(f"[base] Missing keys: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        log.warning(f"[base] Unexpected keys: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
    model.to(device)
    for p in model.parameters(): p.requires_grad = False
    return model, base_config

# ---------------- Prompting ----------------
def build_chat_prompt(user_text: str, system_prompt: Optional[str], template: str) -> str:
    if system_prompt and system_prompt.strip():
        ctx = system_prompt.strip() + "\n\n" + user_text.strip()
    else:
        ctx = user_text.strip()
    return template.format(prompt=ctx)

# ---------------- Sampling (with streaming) ----------------

@torch.inference_mode()
def debug_style_generate(
    model: nn.Module,
    tok: AutoTokenizer,
    device: torch.device,
    prompt_text: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    eos_id: Optional[int] = None,
    repetition_penalty: float = 1.1,
    use_amp: bool = True,
    amp_dtype_str: str = "auto",  # "bf16", "fp16", "auto"
    stream: bool = False,
    on_token=None,                # callback(str) -> None
    suppress_eos_steps: int = 0,  # mask EOS for first N steps if needed
) -> str:
    was_training = model.training
    model.eval()

    # choose AMP dtype
    if device.type == "cuda":
        bf16_ok = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if (amp_dtype_str == "bf16" or (amp_dtype_str == "auto" and bf16_ok)) else torch.float16
    else:
        amp_dtype = torch.float32

    input_ids = tok.encode(prompt_text, add_special_tokens=False)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    printed_text = ""
    def _default_on_token(s: str):
        nonlocal printed_text
        print(s, end="", flush=True)
        printed_text += s
    on_token = on_token or _default_on_token

    try:
        for step in range(max_new_tokens):
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=amp_dtype,
                enabled=(use_amp and (device.type in ("cuda", "cpu")) and amp_dtype != torch.float32),
            ):
                logits, _ = model(ids)           # (1, T, V)
                next_logits = logits[:, -1, :]   # (1, V)

            # repetition penalty on full context
            if repetition_penalty and repetition_penalty > 1.0:
                used = ids[0].unique()
                used_vals = next_logits[0, used]
                penalized = torch.where(
                    used_vals < 0, used_vals * repetition_penalty, used_vals / repetition_penalty
                )
                next_logits = next_logits.clone()
                next_logits[0, used] = penalized

            # (optional) suppress EOS for a few steps
            if eos_id is not None and step < suppress_eos_steps:
                next_logits[0, eos_id] = float("-inf")

            # sample
            if temperature <= 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits / max(1e-6, temperature), dim=-1)
                if 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_idx = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_idx.gather(-1, next_idx)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_token], dim=1)
            tok_id = next_token.item()

            # stream chunk if requested
            if stream:
                chunk = tok.decode([tok_id], skip_special_tokens=False)
                if chunk:
                    on_token(chunk)

            if eos_id is not None and tok_id == eos_id:
                break

        # decode continuation
        full = ids[0].tolist()
        return tok.decode(full[len(input_ids):], skip_special_tokens=False)
    finally:
        if was_training:
            model.train()
            
            
@torch.inference_mode()
def sample_generate(
    model,
    tok,
    prompt_text: str,
    device,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    eos_token_id: Optional[int] = None,
    stop_seqs: Optional[List[str]] = None,
    use_amp: bool = True,
    amp_dtype_str: str = "auto",  # "bf16", "fp16", or "auto"
    temperature_floor: float = 0.2,
    stream: bool = True,
    on_token=None,  # callback(new_text_chunk: str) -> None
) -> str:
    model.eval()

    # AMP dtype selection
    if device.type == "cuda":
        bf16_ok = torch.cuda.is_bf16_supported()
        if amp_dtype_str == "bf16" or (amp_dtype_str == "auto" and bf16_ok):
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    # Prepare input
    input_ids = tok.encode(prompt_text, add_special_tokens=False)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    out_ids: List[int] = []
    stop_seqs = stop_seqs or []
    stop_tok = [tok.encode(s, add_special_tokens=False) for s in stop_seqs if s]

    freq: Dict[int, int] = {}

    def violates_no_repeat_ngram(cand_id: int, seq: List[int], n: int) -> bool:
        if n <= 0 or len(seq) < n - 1:
            return False
        if n == 1:
            return False
        prefix = seq[-(n - 1):] if n > 1 else []
        for i in range(len(seq) - (n - 1)):
            if seq[i:i + (n - 1)] == prefix:
                if i + (n - 1) < len(seq) and seq[i + (n - 1)] == cand_id:
                    return True
        return False

    def apply_repetition_and_openai_penalties(logits: torch.Tensor, generated: List[int]):
        if repetition_penalty != 1.0 and generated:
            unique_ids = list(set(generated))
            idx = torch.tensor(unique_ids, device=logits.device, dtype=torch.long)
            vals = logits.index_select(-1, idx)
            penalized = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
            logits.scatter_(-1, idx.unsqueeze(0), penalized)

        if (frequency_penalty > 0.0 or presence_penalty > 0.0) and freq:
            ids = torch.tensor(list(freq.keys()), device=logits.device, dtype=torch.long)
            counts = torch.tensor([freq[i] for i in ids.tolist()], device=logits.device, dtype=logits.dtype)
            penalty = presence_penalty * (counts > 0).to(logits.dtype) + frequency_penalty * counts
            logits[0, ids] -= penalty

    # streaming state
    printed_chars = 0
    def default_on_token(chunk: str):
        print(chunk, end="", flush=True)

    on_token = on_token or default_on_token

    for t in range(max_new_tokens):
        with torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=amp_dtype,
            enabled=(use_amp and (device.type in ("cuda", "cpu")) and amp_dtype != torch.float32),
        ):
            logits, _ = model(x)             # [B, T, V]
            next_logits = logits[:, -1, :]   # [1, V]

        apply_repetition_and_openai_penalties(next_logits, out_ids)

        temp = max(temperature_floor, temperature) if t > 64 else temperature
        if temp <= 0:
            next_id = int(torch.argmax(next_logits, dim=-1).item())
        else:
            probs = torch.softmax(next_logits / max(1e-8, temp), dim=-1).squeeze(0)

            if top_k and top_k > 0 and top_k < probs.numel():
                topk_vals, topk_idx = torch.topk(probs, k=top_k)
                mask = torch.ones_like(probs, dtype=torch.bool)
                mask.scatter_(0, topk_idx, False)
                probs = probs.masked_fill(mask, 0)

            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                cutoff_len = int((cdf <= top_p).sum().item())
                if cutoff_len > 0:
                    keep_idx = sorted_idx[:cutoff_len]
                    mask = torch.ones_like(probs, dtype=torch.bool)
                    mask.scatter_(0, keep_idx, False)
                    probs = probs.masked_fill(mask, 0)

            if 0.0 < min_p < 1.0:
                p_max = torch.max(probs)
                floor = min_p * p_max
                probs = torch.where(probs >= floor, probs, torch.zeros_like(probs))

            if probs.sum() <= 0:
                probs = torch.softmax(next_logits, dim=-1).squeeze(0)

            next_id = None
            tries = 0
            while next_id is None and tries < 20:
                cand = int(torch.multinomial(probs / probs.sum(), 1).item())
                if no_repeat_ngram_size > 0 and violates_no_repeat_ngram(cand, input_ids + out_ids, no_repeat_ngram_size):
                    probs[cand] = 0.0
                    if probs.sum() <= 0:
                        next_id = cand
                    tries += 1
                    continue
                next_id = cand

        out_ids.append(next_id)
        freq[next_id] = freq.get(next_id, 0) + 1
        x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

        # EOS
        if eos_token_id is not None and next_id == eos_token_id:
            break

        # Stop sequence check
        if stop_tok:
            for seq in stop_tok:
                L = len(seq)
                if L > 0 and len(out_ids) >= L and out_ids[-L:] == seq:
                    out_ids = out_ids[:-L]
                    # stream final chunk before returning
                    if stream:
                        text_now = tok.decode(out_ids, skip_special_tokens=False)
                        chunk = text_now[printed_chars:]
                        if chunk:
                            on_token(chunk)
                            printed_chars = len(text_now)
                    return tok.decode(out_ids, skip_special_tokens=False)

        # Stream newly formed text
        if stream:
            text_now = tok.decode(out_ids, skip_special_tokens=False)
            chunk = text_now[printed_chars:]
            if chunk:
                on_token(chunk)
                printed_chars = len(text_now)

    # Final flush for non-stream mode or if last chunk wasn't printed
    return tok.decode(out_ids, skip_special_tokens=False)

def main():
    ap = argparse.ArgumentParser(description="Inference for custom GPT with optional LoRA adapters.")
    # Model sources
    ap.add_argument("--base_checkpoint", type=str, default=None, help="Path to base model .pt")
    ap.add_argument("--merged_full_model", type=str, default=None, help="Path to merged full model .pt")
    ap.add_argument(
        "--adapters",
        type=str,
        default=None,
        help="Path to LoRA adapters .pt (from SFT/RLHF).",
    )
    ap.add_argument(
        "--lora_adapters",
        type=str,
        default=None,
        help="Alias for --adapters (useful for checkpoints produced by lora_finetune.py).",
    )

    # Tokenizer / templating
    ap.add_argument("--tokenizer_name", type=str, default="gpt2")
    ap.add_argument("--prompt_template", type=str, default="<|user|>\n{prompt}\n<|assistant|>\n")
    ap.add_argument("--system_prompt", type=str, default=None)

    # Architecture fallbacks (used if base ckpt lacks config)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_head", type=int, default=16)
    ap.add_argument("--n_layer", type=int, default=24)
    ap.add_argument("--d_ff", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--n_kv_head", type=int, default=None)

    # Decoding
    ap.add_argument("--prompt", type=str, default=None, help="One-shot prompt (omit for --interactive)")
    ap.add_argument("--interactive", action="store_true", help="Start interactive REPL")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--stop", type=str, nargs="*", default=[], help="Stop sequences")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true", help="Disable AMP and weight casting on CUDA/CPU")
    ap.add_argument("--amp_dtype", type=str, choices=["auto", "bf16", "fp16"], default="auto",
                    help="AMP dtype (auto picks bf16 if supported else fp16)")
    ap.add_argument("--cast_weights", action="store_true", default=True,
                    help="(default: on) Cast model weights to AMP dtype to save VRAM")
    ap.add_argument("--no_cast_weights", action="store_false", dest="cast_weights",
                    help="Disable weight casting (keep weights in fp32)")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--min_p", type=float, default=0.0, help="Min-p nucleus variant (0=off).")
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0, help="e.g., 3 or 4; 0=off")
    ap.add_argument("--frequency_penalty", type=float, default=0.0)
    ap.add_argument("--presence_penalty", type=float, default=0.0)
    ap.add_argument("--temperature_floor", type=float, default=0.2, help="Lower bound if we decay temp on long generations.")

    # Streaming
    ap.add_argument("--no_stream", action="store_true", help="Disable token-by-token streaming")

    args = ap.parse_args()

    if args.n_kv_head is None and args.base_checkpoint:
        try:
            ckpt = torch.load(args.base_checkpoint, map_location="cpu")
            cfg = ckpt.get("config")
            if cfg and cfg.get("n_kv_head"):
                args.n_kv_head = cfg["n_kv_head"]
                log.info(f"Setting n_kv_head={args.n_kv_head} from base checkpoint config.")
        except Exception as exc:
            log.warning(f"Could not read base checkpoint config to infer n_kv_head: {exc}")
    if args.n_kv_head is None:
        args.n_kv_head = compute_default_n_kv_head(args.n_head)
        log.info(f"Auto-selected n_kv_head={args.n_kv_head} (n_head={args.n_head}).")
    elif args.n_head % args.n_kv_head != 0:
        adjusted = compute_default_n_kv_head(args.n_head)
        log.warning(
            f"n_head ({args.n_head}) is not divisible by requested n_kv_head ({args.n_kv_head}); "
            f"adjusting to {adjusted}."
        )
        args.n_kv_head = adjusted

    # Device & seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Tokenizer
    tok = load_tokenizer(args.tokenizer_name)
    vocab_size = len(tok)
    pad_idx = tok.pad_token_id
    eos_id = tok.eos_token_id

    # Load model (base or merged)
    model, _cfg = load_base_or_merged(args, vocab_size, pad_idx, device)

    # Apply LoRA adapters if provided
    adapter_path = args.adapters or args.lora_adapters
    if adapter_path:
        load_lora_adapters(model, adapter_path, device)
        log.info("LoRA adapters loaded.")

    # AMP setup: optionally cast weights to reduce VRAM
    amp_enabled = (not args.no_amp)
    if device.type == "cuda" and amp_enabled and args.cast_weights:
        bf16_ok = torch.cuda.is_bf16_supported()
        if args.amp_dtype == "bf16" or (args.amp_dtype == "auto" and bf16_ok):
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float16
        try:
            model.to(dtype=target_dtype, device=device)
            log.info(f"Casted model weights to {str(target_dtype).replace('torch.','')}.")
        except Exception as e:
            log.warning(f"Weight casting failed ({e}); continuing with original dtype.")

    # Build runner
    def run_once(user_text: str, stream: bool = True):
        chat_prompt = build_chat_prompt(user_text, args.system_prompt, args.prompt_template)

        # stream flag from CLI
        do_stream = stream and (not args.no_stream)

        streamed_any = False
        def _stream_cb(chunk: str):
            nonlocal streamed_any
            if chunk:
                streamed_any = True
                print(chunk, end="", flush=True)
        stream_cb = _stream_cb if do_stream else None

        generated = debug_style_generate(
            model=model,
            tok=tok,
            device=device,
            prompt_text=chat_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_id=eos_id,
            repetition_penalty=args.repetition_penalty,
            use_amp=amp_enabled,
            amp_dtype_str=args.amp_dtype,
            stream=do_stream,
            on_token=stream_cb,
            suppress_eos_steps=3,  # <- helps avoid immediate EOS; tune or set 0
        )

        # If we streamed tokens, ensure newline and also print final (in case buffer was empty)
        if do_stream:
            if not streamed_any and generated:
                # streaming path requested but nothing was emitted (e.g., immediate EOS)
                print(generated, end="")
            print()

        else:
            # non-stream mode: print the whole thing
            print(generated)

        # Trim at any role tag
        for tag in ["<|user|>", "<|system|>", "<|assistant|>"]:
            pos = generated.find(tag)
            if pos > 0:
                generated = generated[:pos]
        return generated.strip()

    if args.interactive:
        print("=== Interactive mode (Ctrl+C to quit) ===")
        while True:
            try:
                user_text = input("\nYou: ").strip()
                if not user_text:
                    continue
                print("Assistant: ", end="", flush=True)
                reply = run_once(user_text, stream=True)
                if args.no_stream:
                    print(reply)
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
    else:
        if not args.prompt:
            raise SystemExit("Provide --prompt for one-shot or use --interactive.")
        if args.no_stream:
            out = run_once(args.prompt, stream=False)
            print(out)
        else:
            # streamed by default
            run_once(args.prompt, stream=True)

if __name__ == "__main__":
    main()
