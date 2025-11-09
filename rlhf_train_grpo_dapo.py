#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLHF-style fine-tuning for a custom GPT model with LoRA:
- Algorithms: DAPO (Direct Advantage PO) and GRPO (Group Relative PO)
- Supports external reward model via HTTP endpoint (batched with retries)
- Weights & Biases logging (per-step and per-epoch), optional artifacts
"""

import os, re, json, glob, math, logging, argparse, random, time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import amp
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# -------- Optional deps (graceful fallbacks) ----------
try:
    import requests
except Exception:
    requests = None

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

# ----------------------- Project-local model -----------------------
try:
    from model import GPT
except Exception as e:
    raise SystemExit("model.py not found or failed to import: %s" % e)

# ----------------------- Torch perf knobs -------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# ----------------------- Logging ---------------------------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rlhf")
logger.setLevel(logging.DEBUG)

IGNORE_INDEX = -100

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


def compute_default_n_kv_head(n_head: int, kv_ratio: int = 4) -> int:
    if n_head <= 0:
        return n_head
    target = max(1, n_head // max(1, kv_ratio))
    while target > 1 and n_head % target != 0:
        target -= 1
    return max(1, target)


_OVERRIDE_KEYS = ["d_model", "n_head", "n_kv_head", "n_layer", "d_ff", "max_seq_len", "dropout"]


def _load_checkpoint_config(path: str) -> Optional[dict]:
    try:
        blob = torch.load(path, map_location="cpu")
    except Exception as exc:
        logger.warning(f"Could not read checkpoint config from {path}: {exc}")
        return None
    cfg = blob.get("config")
    if not isinstance(cfg, dict):
        logger.warning(f"No config dict found in checkpoint {path}.")
        return None
    if not any(k in cfg for k in _OVERRIDE_KEYS):
        return None
    return cfg


def _sync_args_with_config(args, cfg: dict):
    for key in _OVERRIDE_KEYS:
        if key in cfg:
            new_val = cfg[key]
            old_val = getattr(args, key, None)
            if old_val != new_val:
                logger.info(f"Setting args.{key}={new_val} to match checkpoint config (was {old_val}).")
                setattr(args, key, new_val)


def _maybe_load_lora_adapters(path: str) -> Optional[dict]:
    try:
        blob = torch.load(path, map_location="cpu")
    except Exception as exc:
        logger.warning(f"Could not read checkpoint from {path}: {exc}")
        return None
    adapters = blob.get("adapters_state_dict")
    if adapters:
        return adapters
    return None

# ===================== LoRA modules =====================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
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
        dev, dt = self.weight.device, self.weight.dtype
        self.lora_A.to(device=dev, dtype=dt); self.lora_B.to(device=dev, dtype=dt)
        self.weight.requires_grad = False
        if self.bias is not None: self.bias.requires_grad = False

    def forward(self, x: torch.Tensor):
        if self.lora_A.weight.device != x.device:
            self.lora_A.to(device=x.device); self.lora_B.to(device=x.device)
        base_out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.r <= 0: return base_out
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
            if include_regex and not re.search(include_regex, full): ok = False
            if exclude_regex and re.search(exclude_regex, full): ok = False
            if ok:
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced.append(full)

def apply_lora_adapters(model: nn.Module, r: int, alpha: int, dropout: float,
                        include_regex: Optional[str], exclude_regex: Optional[str]) -> List[str]:
    replaced: List[str] = []
    _wrap_linear_with_lora(model, "", include_regex, exclude_regex, r, alpha, dropout, replaced)
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_A.weight" in n) or ("lora_B.weight" in n)
    return replaced

def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}

def merge_lora_into_base(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            with torch.no_grad():
                delta = (m.lora_B.weight @ m.lora_A.weight) * m.lora_scaling
                m.weight.data += delta
    for p in model.parameters(): p.requires_grad = False

# ===================== Tokenizer + prompt helpers =====================
def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.eos_token is None: tok.add_special_tokens({"eos_token": "[EOS]"})
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    try: tok.model_max_length = int(1e12)
    except Exception: pass
    return tok

def build_chat_prompt(user_text: str, system_prompt: Optional[str], template: str) -> str:
    user_text = user_text.strip()
    if system_prompt and system_prompt.strip():
        ctx = system_prompt.strip() + "\n\n" + user_text
    else:
        ctx = user_text
    return template.format(prompt=ctx)


def build_prompt(instr: str, ctx: Optional[str], template: str, system_prompt: Optional[str]) -> str:
    prompt_body = instr.strip()
    if ctx not in (None, "", []):
        prompt_body = f"{ctx}\n\n{prompt_body}"
    return build_chat_prompt(prompt_body, system_prompt, template)

# ===================== Data =====================
class PromptDataset(Dataset):
    def __init__(self, paths_or_dir: str, prompt_template: str, strip_think: bool = False,
                 system_prompt: Optional[str] = None):
        if os.path.isdir(paths_or_dir):
            files = sorted(glob.glob(os.path.join(paths_or_dir, "**", "*.jsonl"), recursive=True))
        else:
            files = glob.glob(paths_or_dir) if any(ch in paths_or_dir for ch in "*?[]") else [paths_or_dir]
        if not files: raise FileNotFoundError(f"No JSONL files found for: {paths_or_dir}")
        rows = []
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    obj = json.loads(line)
                    instr = (obj.get("instruction") or "").strip()
                    if not instr: continue
                    resp  = (obj.get("response") or "").strip()
                    if strip_think and resp:
                        resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
                    rows.append({"instruction": instr, "context": obj.get("context", None), "ref_response": resp or None})
        if not rows: raise RuntimeError("No valid instruction rows.")
        self.rows = rows; self.template = prompt_template; self.system_prompt = system_prompt
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        return {"prompt_text": build_prompt(r["instruction"], r["context"], self.template, self.system_prompt),
                "ref_response": r["ref_response"]}


def prompt_collate_fn(batch: List[dict]) -> Dict[str, List[Optional[str]]]:
    return {
        "prompt_text": [item["prompt_text"] for item in batch],
        "ref_response": [item.get("ref_response") for item in batch],
    }

def _violates_no_repeat_ngram(cand: int, seq: List[int], n: int) -> bool:
    if n <= 0 or len(seq) < n - 1:
        return False
    prefix = seq[-(n - 1):] if n > 1 else []
    for i in range(len(seq) - (n - 1)):
        if seq[i:i + (n - 1)] == prefix:
            if i + (n - 1) < len(seq) and seq[i + (n - 1)] == cand:
                return True
    return False


def generate_and_logprobs(model: nn.Module, tok, device, prompt_texts: List[str],
                          max_new_tokens: int = 192, temperature: float = 0.7, top_p: float = 0.9,
                          top_k: int = 0, min_p: float = 0.0,
                          repetition_penalty: float = 1.1,
                          no_repeat_ngram_size: int = 0,
                          frequency_penalty: float = 0.0,
                          presence_penalty: float = 0.0,
                          temperature_floor: float = 0.2,
                          suppress_eos_steps: int = 0,
                          eos_id: Optional[int] = None, logprob_chunk_size: int = 1,
                          amp_enabled: bool = False, amp_dtype: torch.dtype = torch.bfloat16,
                          backprop_context_len: int = 0):
    # Remember original mode to restore at the end
    _prev_training = model.training
    amp_enabled = bool(amp_enabled and device.type == "cuda")

    B = len(prompt_texts)
    enc = tok(prompt_texts, add_special_tokens=False, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask")
    if attn_mask is None:
        pad_id_tmp = tok.pad_token_id
        if pad_id_tmp is None:
            pad_id_tmp = tok.eos_token_id if tok.eos_token_id is not None else 0
        attn_mask = (input_ids != pad_id_tmp).long()
    attn_mask = attn_mask.to(device)
    prompt_lengths = attn_mask.sum(dim=1).to(torch.long)
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    # ------- Sampling loop (eval + no_grad) -------
    generated = [[] for _ in range(B)]
    freq_counters = [dict() for _ in range(B)]
    cur_ids = input_ids.clone()
    cur_mask = attn_mask.clone()

    model.eval()
    with torch.no_grad():
        autocast_ctx = amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled)
        with autocast_ctx:
            for step in range(max_new_tokens):
                logits, _ = model(cur_ids.detach(), attention_mask=cur_mask.detach())
                next_logits = logits[:, -1, :]

                sampled_tokens: List[int] = []
                for b in range(B):
                    logits_b = next_logits[b].clone()
                    ctx_len = int(cur_mask[b].sum().item())
                    if ctx_len > 0:
                        ctx_tokens = cur_ids[b, :ctx_len]
                        if repetition_penalty and repetition_penalty != 1.0:
                            unique_tokens = torch.unique(ctx_tokens)
                            if unique_tokens.numel() > 0:
                                vals = logits_b[unique_tokens]
                                penalized = torch.where(
                                    vals > 0, vals / repetition_penalty, vals * repetition_penalty
                                )
                                logits_b[unique_tokens] = penalized
                    if (frequency_penalty > 0.0 or presence_penalty > 0.0) and freq_counters[b]:
                        ids = torch.tensor(list(freq_counters[b].keys()), device=device, dtype=torch.long)
                        counts = torch.tensor(
                            [freq_counters[b][int(i)] for i in ids.tolist()],
                            device=device,
                            dtype=logits_b.dtype,
                        )
                        penalty = presence_penalty * (counts > 0).to(logits_b.dtype) + frequency_penalty * counts
                        logits_b.scatter_sub_(0, ids, penalty)
                    if eos_id is not None and suppress_eos_steps > 0 and step < suppress_eos_steps:
                        logits_b[eos_id] = float("-inf")

                    temp = temperature
                    if step > 64:
                        temp = max(temperature_floor, temperature)
                    if temp <= 0:
                        token_id = int(torch.argmax(logits_b).item())
                    else:
                        probs = torch.softmax(logits_b / max(1e-8, temp), dim=-1)
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
                            probs = torch.softmax(logits_b, dim=-1)

                        tries = 0
                        token_id = None
                        ctx_list = cur_ids[b, :ctx_len].tolist() if ctx_len > 0 else []
                        while token_id is None and tries < 20:
                            cand = int(torch.multinomial(probs / probs.sum(), 1).item())
                            if no_repeat_ngram_size > 0 and _violates_no_repeat_ngram(
                                cand, ctx_list + generated[b], no_repeat_ngram_size
                            ):
                                probs[cand] = 0.0
                                if probs.sum() <= 0:
                                    token_id = cand
                                tries += 1
                                continue
                            token_id = cand

                    sampled_tokens.append(token_id)
                    freq_counters[b][token_id] = freq_counters[b].get(token_id, 0) + 1
                    generated[b].append(token_id)

                next_token = torch.tensor(sampled_tokens, device=device, dtype=torch.long).unsqueeze(1)
                cur_ids = torch.cat([cur_ids, next_token], dim=1)
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((B, 1), dtype=cur_mask.dtype, device=device)],
                    dim=1,
                )
                if eos_id is not None and all((seq and seq[-1] == eos_id) for seq in generated):
                    break

    out_texts = [tok.decode(seq, skip_special_tokens=False) for seq in generated]

    # Prepare packed full sequences for log-prob computation
    max_total_len = 0
    total_lengths = []
    for b in range(B):
        total_len = int(prompt_lengths[b].item()) + len(generated[b])
        total_lengths.append(total_len)
        max_total_len = max(max_total_len, total_len)

    if max_total_len == 0:
        out_logps = [torch.empty(0, device=device) for _ in range(B)]
        # Restore original mode before returning
        model.train(_prev_training)
        if not _prev_training:
            model.eval()
        return out_texts, generated, out_logps

    full_ids = torch.full((B, max_total_len), pad_id, dtype=torch.long, device=device)
    full_mask = torch.zeros((B, max_total_len), dtype=torch.long, device=device)
    ctx_starts = [0 for _ in range(B)]
    for b in range(B):
        plen = int(prompt_lengths[b].item())
        if plen > 0:
            full_ids[b, :plen] = input_ids[b, :plen]
        if generated[b]:
            cont = torch.tensor(generated[b], device=device, dtype=torch.long)
            full_ids[b, plen:plen + len(generated[b])] = cont
        full_mask[b, :total_lengths[b]] = 1
        if backprop_context_len > 0:
            ctx_starts[b] = max(0, plen - backprop_context_len)

    if not getattr(generate_and_logprobs, "_logged_trainable_count", False):
        trainable = sum(p.requires_grad for p in model.parameters())
        logger.info("[generate_and_logprobs] trainable parameter tensors=%d", trainable)
        generate_and_logprobs._logged_trainable_count = True
    if not getattr(generate_and_logprobs, "_logged_grad_state", False):
        logger.info("[generate_and_logprobs] grad enabled before logprob pass? %s", torch.is_grad_enabled())
        generate_and_logprobs._logged_grad_state = True

    # ------- Log-prob pass (force train + grad) -------
    out_logps: List[torch.Tensor] = [torch.empty(0, device=device) for _ in range(B)]
    model.train()  # ensure any train/eval-gated autograd is ON
    try:
        with torch.set_grad_enabled(True):
            chunk = logprob_chunk_size if (logprob_chunk_size and logprob_chunk_size > 0) else B
            chunk = max(1, min(chunk, B))
            for start in range(0, B, chunk):
                end = min(B, start + chunk)
                chunk_bs = list(range(start, end))
                if backprop_context_len > 0:
                    padded_ids = []
                    padded_mask = []
                    max_len = 0
                    for b in chunk_bs:
                        ctx_start = ctx_starts[b]
                        seq_len = total_lengths[b] - ctx_start
                        seq = full_ids[b, ctx_start:ctx_start + seq_len]
                        mask = full_mask[b, ctx_start:ctx_start + seq_len]
                        padded_ids.append(seq)
                        padded_mask.append(mask)
                        max_len = max(max_len, seq_len)
                    chunk_ids = torch.full((len(chunk_bs), max_len), pad_id, dtype=torch.long, device=device)
                    chunk_mask = torch.zeros((len(chunk_bs), max_len), dtype=torch.long, device=device)
                    for i, (seq, mask) in enumerate(zip(padded_ids, padded_mask)):
                        L = seq.size(0)
                        if L > 0:
                            chunk_ids[i, :L] = seq
                            chunk_mask[i, :L] = mask
                else:
                    chunk_ids = full_ids[start:end]
                    chunk_mask = full_mask[start:end]
                with amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                    logits, _ = model(chunk_ids, attention_mask=chunk_mask)
                if not getattr(generate_and_logprobs, "_logged_logits_grad", False):
                    logger.info("[generate_and_logprobs] logits.requires_grad=%s", logits.requires_grad)
                    generate_and_logprobs._logged_logits_grad = True
                for local_idx, b in enumerate(chunk_bs):
                    gen_len = len(generated[b])
                    if gen_len == 0:
                        continue
                    start_pos = int(prompt_lengths[b].item())
                    ctx_start = ctx_starts[b] if backprop_context_len > 0 else 0
                    positions = torch.arange(gen_len, device=device, dtype=torch.long) + (start_pos - ctx_start)
                    tokens = torch.tensor(generated[b], device=device, dtype=torch.long)
                    selected_logits = logits[local_idx, positions, :]
                    nll = F.cross_entropy(selected_logits, tokens, reduction='none')
                    out_logps[b] = -nll
    finally:
        # restore original mode
        model.train(_prev_training)
        if not _prev_training:
            model.eval()

    return out_texts, generated, out_logps

@torch.no_grad()
def compute_ref_logprobs(model_ref: nn.Module, tok, device, prompt_texts: List[str], continuations: List[List[int]]):
    model_ref.eval()
    enc = tok(prompt_texts, add_special_tokens=False, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    logps = []
    for b in range(len(prompt_texts)):
        ctx = input_ids[b:b+1]
        seq_logps = []
        for t in continuations[b]:
            logits, _ = model_ref(ctx)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            token = torch.tensor([[t]], device=device, dtype=torch.long)
            lp = torch.log(torch.gather(probs, -1, token))[0, 0]
            seq_logps.append(lp)
            ctx = torch.cat([ctx, token], dim=1)
        logps.append(torch.stack(seq_logps, dim=0) if seq_logps else torch.tensor([], device=device))
    return logps

# ===================== Rewards (built-ins) =====================
def reward_length(text: str) -> float:
    n = max(1, len(text.strip().split()))
    return min(n / 64.0, 1.0)

def reward_anti_repeat(text: str) -> float:
    toks = text.strip().split()
    if len(toks) < 6: return 0.3
    grams = set(); reps = 0
    for i in range(len(toks)-2):
        g = (toks[i], toks[i+1], toks[i+2])
        if g in grams: reps += 1
        grams.add(g)
    ratio = reps / max(1, len(toks)-2)
    return max(0.0, 1.0 - 2.0 * ratio)

def reward_ref_match(generated: str, ref: Optional[str]) -> float:
    if not ref: return 0.0
    g = set(generated.lower().split()); r = set(ref.lower().split())
    if not g or not r: return 0.0
    return len(g & r) / max(1, len(g | r))


def reward_think_prefix(text: str) -> float:
    """
    Validator reward: encourage outputs that begin with a <think> block.
    Returns 1.0 if the trimmed text starts with '<think', otherwise 0.0.
    """
    stripped = text.lstrip()
    if not stripped:
        return 0.0
    return 1.0 if stripped.lower().startswith("<think") else 0.0


def _preview_sequence(values, *, max_items: int = 4) -> str:
    """Compact string preview for logging."""
    if not values:
        return "[]"
    vals = list(values)
    shown = []
    for v in vals[:max_items]:
        if isinstance(v, float):
            shown.append(f"{float(v):.4f}")
        else:
            shown.append(str(v))
    if len(vals) > max_items:
        shown.append(f"...(+{len(vals) - max_items} more)")
    return "[" + ", ".join(shown) + "]"


def _truncate_tokens_for_log(text: str, max_tokens: int = 256) -> str:
    """Return text limited to max_tokens (whitespace tokens) for log output."""
    toks = text.strip().split()
    if len(toks) <= max_tokens:
        return " ".join(toks).replace("\n", " ")
    return (" ".join(toks[:max_tokens]) + " ...").replace("\n", " ")

def _preview_text(text: str, limit: int = 100) -> str:
    """Single-line preview capped at `limit` characters for debug logging."""
    clean = text.replace("\n", " ")
    return clean[:limit] + ("…" if len(clean) > limit else "")

# ===================== Reward Model Stub (HTTP) =====================
def http_reward_scores(
    prompts: List[str],
    outputs: List[str],
    endpoint: str,
    timeout: float = 10.0,
    max_retries: int = 3,
    backoff: float = 1.5,
    extra_payload: Optional[dict] = None,
) -> List[float]:
    """
    Call your reward model server.
    Expected JSON response:
      { "scores": [float, float, ...] } aligned with inputs length.
    Request JSON (example):
      { "prompts": [...], "outputs": [...], "extra": {...} }
    """
    if (requests is None) or (not endpoint):
        return [0.0] * len(outputs)

    payload = {"prompts": prompts, "outputs": outputs}
    if extra_payload: payload["extra"] = extra_payload

    last_err = None
    for attempt in range(max_retries):
        try:
            start = time.time()
            resp = requests.post(endpoint, json=payload, timeout=timeout)
            elapsed = (time.time() - start) * 1000.0
            logger.info(
                "[reward_model] POST %s attempt=%d | outputs=%d | status=%d | %.1f ms",
                endpoint,
                attempt + 1,
                len(outputs),
                resp.status_code,
                elapsed,
            )
            if resp.status_code == 200:
                obj = resp.json()
                scores = obj.get("scores", [])
                orders = obj.get("orders")
                logger.info(
                    "[reward_model] response scores=%s | orders=%s",
                    _preview_sequence(scores),
                    json.dumps(orders[:2]) if isinstance(orders, list) else "None",
                )
                if not isinstance(scores, list):
                    logger.warning("[reward_model] 'scores' not a list; returning zeros.")
                    return [0.0] * len(outputs)
                # Align length; pad/truncate conservatively
                if len(scores) < len(outputs):
                    logger.warning("[reward_model] fewer scores than outputs; padding zeros.")
                    scores = scores + [0.0] * (len(outputs) - len(scores))
                elif len(scores) > len(outputs):
                    scores = scores[:len(outputs)]
                return [float(s) for s in scores]
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(backoff ** (attempt + 1))
    logger.warning(f"[reward_model] failed after retries: {last_err}")
    return [0.0] * len(outputs)

def composite_reward(
    generated_texts: List[str],
    ref_texts: List[Optional[str]],
    weights: Dict[str, float],
    prompts: Optional[List[str]] = None,
    reward_model_url: Optional[str] = None,
    reward_model_timeout: float = 10.0,
    reward_model_retries: int = 3,
    reward_model_backoff: float = 1.5,
    reward_model_extra: Optional[dict] = None,
    reward_group_size: int = 1,
    shuffle_reward_groups: bool = True,
) -> List[float]:
    base = []
    use_rm = (weights.get("reward_model", 0.0) != 0.0) and bool(reward_model_url)
    rm_scores = []
    if use_rm:
        prompts_for_rm = prompts or [""] * len(generated_texts)
        idx_order = list(range(len(generated_texts)))
        request_indices = []
        if shuffle_reward_groups and reward_group_size > 1:
            for start in range(0, len(idx_order), reward_group_size):
                group = idx_order[start:start + reward_group_size]
                random.shuffle(group)
                request_indices.extend(group)
        else:
            request_indices = idx_order

        shuffled_prompts = [prompts_for_rm[i] for i in request_indices]
        shuffled_outputs = [generated_texts[i] for i in request_indices]
        shuffled_scores = http_reward_scores(
            prompts=shuffled_prompts,
            outputs=shuffled_outputs,
            endpoint=reward_model_url,
            timeout=reward_model_timeout,
            max_retries=reward_model_retries,
            backoff=reward_model_backoff,
            extra_payload=reward_model_extra,
        )
        rm_scores = [0.0] * len(generated_texts)
        for pos, original_idx in enumerate(request_indices):
            if pos < len(shuffled_scores):
                rm_scores[original_idx] = float(shuffled_scores[pos])
        if reward_group_size > 0 and shuffled_scores:
            per_group = []
            for start in range(0, len(request_indices), reward_group_size):
                group = request_indices[start:start + reward_group_size]
                group_scores = shuffled_scores[start:start + len(group)]
                if not group_scores:
                    continue
                best_local = max(range(len(group_scores)), key=lambda j: group_scores[j])
                best_idx = group[best_local]
                snippet = _truncate_tokens_for_log(generated_texts[best_idx]) if generated_texts else ""
                per_group.append({
                    "orig_idx": best_idx,
                    "score": group_scores[best_local],
                    "snippet": snippet,
                })
            if per_group:
                msg = []
                for item in per_group:
                    msg.append(f"{item['orig_idx']}:{item['score']:.4f} | {item['snippet']}")
                # logger.info("[reward_model] best per group (orig_idx:score | text): %s", " || ".join(msg))

    for i, (g, r) in enumerate(zip(generated_texts, ref_texts)):
        s = 0.0
        if weights.get("length", 0) != 0:       s += weights["length"] * reward_length(g)
        if weights.get("anti_repeat", 0) != 0:  s += weights["anti_repeat"] * reward_anti_repeat(g)
        if weights.get("ref_match", 0) != 0:    s += weights["ref_match"] * reward_ref_match(g, r)
        if weights.get("think_prefix", 0) != 0:
            s += weights["think_prefix"] * reward_think_prefix(g)
        if use_rm:
            s += weights["reward_model"] * float(rm_scores[i])
        base.append(float(s))
    return base

def log_reward_traces(
    trace_file,
    *,
    epoch: int,
    step: int,
    algo: str,
    group_size: int,
    prompts: List[str],
    ref_texts: List[Optional[str]],
    generated_texts: List[str],
    rewards: List[float],
    kls: List[torch.Tensor],
    advantages: List[float],
) -> None:
    """
    Append JSONL entries describing each prompt's generations plus the preferred sample.
    """
    if trace_file is None:
        return
    group = group_size if algo == "grpo" else 1
    sample_idx = 0
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    for prompt, ref in zip(prompts, ref_texts):
        if sample_idx >= len(generated_texts):
            break
        completions = []
        best_idx = 0
        best_reward = -float("inf")
        best_global_idx = None
        for rel in range(group):
            if sample_idx >= len(generated_texts):
                break
            current_idx = sample_idx
            reward_val = float(rewards[sample_idx])
            kl_val = float(kls[sample_idx].item()) if sample_idx < len(kls) else None
            adv_val = float(advantages[sample_idx]) if sample_idx < len(advantages) else None
            text_val = generated_texts[sample_idx]
            completions.append({
                "text": text_val,
                "reward": reward_val,
                "kl": kl_val,
                "advantage": adv_val,
            })
            if reward_val > best_reward:
                best_reward = reward_val
                best_idx = rel
                best_global_idx = current_idx
            sample_idx += 1
        entry = {
            "timestamp": timestamp,
            "epoch": epoch,
            "step": step,
            "algo": algo,
            "prompt": prompt,
            "ref_response": ref,
            "completions": completions,
            "preferred_index": best_idx if completions else None,
            "preferred_global_index": best_global_idx,
            "preferred_text": completions[best_idx]["text"] if completions else None,
            "preferred_reward": completions[best_idx]["reward"] if completions else None,
        }
        trace_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    trace_file.flush()

# ===================== KL approx (per-token) =====================
def approx_kl_from_logps(logps_pol: torch.Tensor, logps_ref: torch.Tensor) -> torch.Tensor:
    if logps_pol.numel() == 0: return torch.tensor(0.0, device=logps_pol.device)
    return (logps_pol - logps_ref).mean()

# ===================== Advantage shaping =====================
class MovingBaseline:
    def __init__(self, momentum=0.9): self.m = momentum; self.mean = None
    def update(self, x: float) -> float:
        if self.mean is None: self.mean = x
        else: self.mean = self.m * self.mean + (1 - self.m) * x
        return self.mean

def ranks_to_advantages(rewards: List[float]) -> List[float]:
    K = len(rewards)
    order = sorted(range(K), key=lambda i: rewards[i])
    ranks = [0]*K
    for r,i in enumerate(order): ranks[i] = r
    mu = (K-1)/2.0
    denom = max(1e-6, math.sqrt((K**2-1)/12.0))
    return [(rk - mu)/denom for rk in ranks]

def zscore_within_group(rewards: List[float]) -> List[float]:
    mu = sum(rewards)/max(1,len(rewards))
    var = sum((x-mu)**2 for x in rewards)/max(1,len(rewards))
    sd = math.sqrt(max(1e-8, var))
    return [(x-mu)/sd for x in rewards]

# ===================== Train utils =====================
def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_only = sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and ("lora_" in n))
    pct = 100.0 * trainable / max(1,total)
    return dict(total_params=total, trainable_params=trainable, lora_params=lora_only, trainable_pct=pct)

def load_base(model: nn.Module, ckpt_path: str, device: torch.device):
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.",""): v for k,v in state.items()}
    miss, unexp = model.load_state_dict(state, strict=False)
    if miss:  logger.warning(f"[load] missing keys: {miss[:10]}{'...' if len(miss)>10 else ''}")
    if unexp: logger.warning(f"[load] unexpected keys: {unexp[:10]}{'...' if len(unexp)>10 else ''}")
    model.to(device)

def save_adapters(output_dir: str, tag: str, model: nn.Module, cfg: dict, args, log_to_wandb: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"rl_adapters_{tag}.pt")
    torch.save({"adapters_state_dict": lora_state_dict(model),
                "config": cfg, "args": vars(args), "timestamp": int(time.time())}, path)
    logger.info(f"Saved LoRA adapters: {path}")
    if log_to_wandb and wandb is not None and wandb.run is not None:
        artifact = wandb.Artifact(name=f"adapters-{tag}", type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)

def parse_reward_components(spec: str) -> Dict[str,float]:
    # e.g. 'length:0.05,anti_repeat:0.35,ref_match:0.30,reward_model:0.30'
    if not spec: return {}
    out = {}
    for part in spec.split(","):
        if not part.strip(): continue
        k,v = part.split(":")
        out[k.strip()] = float(v)
    return out

# ===================== Main train =====================
def train(args):
    trace_file = None
    try:
        # --------- wandb init ---------
        if args.wandb_mode != "disabled" and _WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=args.wandb_run_name or None,
                mode=args.wandb_mode,  # "online", "offline"
                config=vars(args),
            )
            logger.info("[wandb] Logging enabled.")
        elif args.wandb_mode != "disabled" and not _WANDB_AVAILABLE:
            logger.warning("[wandb] not installed; set --wandb_mode disabled or install wandb.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("CUDA expected for efficiency.")

        base_cfg = _load_checkpoint_config(args.base_checkpoint)
        ref_cfg = _load_checkpoint_config(args.ref_checkpoint)
        if base_cfg:
            _sync_args_with_config(args, base_cfg)
        elif ref_cfg:
            logger.info("Using reference checkpoint config to set model dims.")
            _sync_args_with_config(args, ref_cfg)

        if ref_cfg and base_cfg and ref_cfg != base_cfg:
            logger.warning("Reference checkpoint config differs from base; proceeding with base config values.")

        if args.n_kv_head is None and base_cfg:
            inferred = base_cfg.get("n_kv_head")
            if inferred:
                args.n_kv_head = inferred
                logger.info(f"Setting n_kv_head={inferred} to match base checkpoint config.")
        if args.n_kv_head is None:
            kv_ratio = 4
            if base_cfg and base_cfg.get("n_kv_head"):
                kv_ratio = max(1, base_cfg.get("n_head", args.n_head) // base_cfg["n_kv_head"])
            args.n_kv_head = compute_default_n_kv_head(args.n_head, kv_ratio=kv_ratio)
            logger.info(f"Auto-selected n_kv_head={args.n_kv_head} (n_head={args.n_head}).")
        elif args.n_head % args.n_kv_head != 0:
            adjusted = compute_default_n_kv_head(args.n_head)
            logger.warning(
                f"n_head ({args.n_head}) is not divisible by requested n_kv_head ({args.n_kv_head}); adjusting to {adjusted}."
            )
            args.n_kv_head = adjusted

        tok = load_tokenizer(args.tokenizer_name)
        pad_id = tok.pad_token_id; vocab_size = len(tok)

        trace_path = args.reward_trace_file or os.path.join(args.output_dir, "reward_traces.jsonl")
        trace_dir = os.path.dirname(os.path.abspath(trace_path))
        os.makedirs(trace_dir, exist_ok=True)
        trace_file = open(trace_path, "a", encoding="utf-8")
        logger.info(f"[trace] Reward traces will be appended to {trace_path}")

        ds = PromptDataset(args.training_dir if args.training_dir else args.train_jsonl,
                           prompt_template=args.prompt_template, strip_think=args.strip_think,
                           system_prompt=args.system_prompt)
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=prompt_collate_fn,
        )

        model_cfg = {
            "vocab_size": vocab_size, "d_model": args.d_model, "n_head": args.n_head, "n_layer": args.n_layer,
            "n_kv_head": args.n_kv_head, "max_seq_len": args.max_seq_len, "dropout": args.dropout,
            "pad_idx": pad_id, "d_ff": args.d_ff or args.d_model*4
        }
        lora_adapters = _maybe_load_lora_adapters(args.base_checkpoint)
        policy = GPT(**model_cfg)
        policy_base_path = args.base_checkpoint
        if lora_adapters:
            policy_base_path = args.policy_base_checkpoint or args.ref_checkpoint
            if policy_base_path is None:
                raise SystemExit("LoRA checkpoint provided but no base weights specified. Use --policy_base_checkpoint.")
            if policy_base_path == args.ref_checkpoint:
                logger.info("Policy base weights will be loaded from --ref_checkpoint.")
        load_base(policy, policy_base_path, device)
        replaced = apply_lora_adapters(policy, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
                                       include_regex=args.lora_include,
                                       exclude_regex=(args.lora_exclude or r"(token_embedding|lm_head)"))
        logger.info(f"LoRA applied to {len(replaced)} modules, e.g. {replaced[:8]}")
        if lora_adapters:
            missing, unexpected = policy.load_state_dict(lora_adapters, strict=False)
            if missing:
                logger.warning(f"[load-adapters] missing keys: {missing[:10]}{'...' if len(missing)>10 else ''}")
            if unexpected:
                logger.warning(f"[load-adapters] unexpected keys: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")

        ref = GPT(**model_cfg); load_base(ref, args.ref_checkpoint, device)
        for p in ref.parameters(): p.requires_grad = False
        ref.eval()

        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(
                trainable_params,
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
            logger.info("Using bitsandbytes AdamW8bit optimizer.")
        except Exception as exc:
            logger.warning(f"Falling back to torch AdamW (8-bit optimizer unavailable: {exc})")
            opt = optim.AdamW(trainable_params,
                              lr=args.learning_rate, weight_decay=args.weight_decay)
        scaler = amp.GradScaler(enabled=False)

        counts = count_params(policy)
        logger.info("Params: total={:,}, trainable={:,} (LoRA={:,}) | {:.2f}% trainable".format(
            counts["total_params"], counts["trainable_params"], counts["lora_params"], counts["trainable_pct"]
        ))
        if wandb and wandb.run: wandb.log({"params/total": counts["total_params"],
                                           "params/trainable": counts["trainable_params"],
                                           "params/lora": counts["lora_params"],
                                           "params/trainable_pct": counts["trainable_pct"]})

        rweights = parse_reward_components(args.reward_components)
        logger.info(f"Reward components: {rweights}")

        baseline = MovingBaseline(momentum=0.95) if args.algo == "dapo" else None
        global_step = 0

        for epoch in range(args.epochs):
            logger.info(f"=== Epoch {epoch+1}/{args.epochs} ===")
            for batch in dl:
                prompts: List[str] = batch["prompt_text"]
                ref_texts_raw: List[Optional[str]] = batch["ref_response"]
                ref_texts: List[Optional[str]] = [r if r else None for r in ref_texts_raw]

                if args.algo == "grpo":
                    expanded_prompts, expanded_refs = [], []
                    for p, r in zip(prompts, ref_texts):
                        for _ in range(args.group_size):
                            expanded_prompts.append(p)
                            expanded_refs.append(r)
                    prompts_to_gen, refs_to_use = expanded_prompts, expanded_refs
                else:
                    prompts_to_gen, refs_to_use = prompts, ref_texts
                group_size = args.group_size if args.algo == "grpo" else 1

                gen_texts, gen_ids, logps_pol = generate_and_logprobs(
                    policy, tok, device, prompts_to_gen,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
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
                # if gen_texts:
                #     logger.debug("[generate] sample output[0]: %s", _preview_text(gen_texts[0], limit=100))
                with torch.no_grad():
                    logps_ref = compute_ref_logprobs(ref, tok, device, prompts_to_gen, gen_ids)

                # rewards (built-ins + reward model)
                rewards = composite_reward(
                    gen_texts, refs_to_use, rweights,
                    prompts=prompts_to_gen,
                    reward_model_url=args.reward_model_url,
                    reward_model_timeout=args.reward_model_timeout,
                    reward_model_retries=args.reward_model_retries,
                    reward_model_backoff=args.reward_model_backoff,
                    reward_model_extra=json.loads(args.reward_model_extra) if args.reward_model_extra else None,
                    reward_group_size=args.group_size if args.algo == "grpo" else 1,
                )

                if logger.isEnabledFor(logging.INFO) and rewards:
                    for group_idx, start in enumerate(range(0, len(rewards), group_size)):
                        grp_rewards = rewards[start:start + group_size]
                        if not grp_rewards:
                            continue
                        prompt_preview = _preview_text(prompts_to_gen[start], limit=160) if start < len(prompts_to_gen) else ""
                        entries = []
                        for offset, reward_val in enumerate(grp_rewards):
                            global_idx = start + offset
                            snippet = _truncate_tokens_for_log(gen_texts[global_idx]) if global_idx < len(gen_texts) else ""
                            entries.append(f"\n\n{global_idx}:{reward_val:.4f}:\n\n{snippet}")
                        logger.info("[reward_detail] group %d prompt=\"%s\"", group_idx, prompt_preview)
                        logger.info("[reward_detail]   samples: %s", " \n\n ".join(entries))

                # KL per sample
                kls = []
                for lp_pol, lp_ref in zip(logps_pol, logps_ref):
                    T = min(lp_pol.numel(), lp_ref.numel())
                    kls.append(approx_kl_from_logps(lp_pol[:T], lp_ref[:T]) if T > 0 else torch.tensor(0.0, device=device))

                # advantages
                if args.algo == "dapo":
                    raw_adv = [float(r) - args.beta_kl * float(k.item()) for r, k in zip(rewards, kls)]
                    b = baseline.update(sum(raw_adv)/max(1,len(raw_adv)))
                    adv = [ra - b for ra in raw_adv]
                else:
                    adv = []
                    K = args.group_size
                    assert len(rewards) % K == 0, "group_size must divide batch*group count"
                    for i in range(0, len(rewards), K):
                        grp = rewards[i:i+K]
                        if args.beta_kl != 0.0:
                            grp = [grp[j] - args.beta_kl * float(kls[i+j].item()) for j in range(K)]
                        grp_adv = ranks_to_advantages(grp) if args.grpo_use_ranks else zscore_within_group(grp)
                        adv.extend(grp_adv)

                log_reward_traces(
                    trace_file,
                    epoch=epoch + 1,
                    step=global_step,
                    algo=args.algo,
                    group_size=args.group_size,
                    prompts=prompts,
                    ref_texts=ref_texts,
                    generated_texts=gen_texts,
                    rewards=rewards,
                    kls=kls,
                    advantages=adv,
                )

                winner_msgs = []
                for start in range(0, len(rewards), group_size):
                    grp = rewards[start:start + group_size]
                    if not grp:
                        continue
                    best_local = max(range(len(grp)), key=lambda j: grp[j])
                    global_idx = start + best_local
                    snippet = _truncate_tokens_for_log(gen_texts[global_idx])
                    winner_msgs.append(f"{global_idx}:{grp[best_local]:.4f} | {snippet}")
                # if winner_msgs:
                #     logger.info("[reward_total] best per group (global_idx:reward | text): %s", " || ".join(winner_msgs))

                # policy gradient
                losses = []
                seq_lengths = []
                if global_step == 0 and logps_pol:
                    grad_flags = [lp.requires_grad for lp in logps_pol[:min(4, len(logps_pol))]]
                    logger.info("[debug] logps_pol grad flags (first few): %s", grad_flags)
                for lp_seq, a in zip(logps_pol, adv):
                    if lp_seq.numel() == 0:
                        continue
                    seq_lengths.append(lp_seq.numel())
                    losses.append(-float(a) * lp_seq.mean())
                if not losses:
                    continue

                loss = torch.stack(losses).mean()
                loss_scaled = loss / args.grad_accum

                if scaler.is_enabled(): scaler.scale(loss_scaled).backward()
                else: loss_scaled.backward()

                if (global_step + 1) % args.grad_accum == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    if scaler.is_enabled(): scaler.step(opt); scaler.update()
                    else: opt.step()
                    opt.zero_grad(set_to_none=True)

                # logging
                avg_r = sum(rewards)/max(1,len(rewards))
                avg_kl = sum(float(k.item()) for k in kls)/max(1,len(kls))
                avg_len = sum(seq_lengths)/max(1,len(seq_lengths))
                if (global_step % args.log_interval) == 0:
                    logger.info(f"step {global_step} | algo={args.algo} | loss={loss.item():.4f} "
                                f"| avg_reward={avg_r:.3f} | avg_kl={avg_kl:.4f} | avg_len={avg_len:.1f} | beta={args.beta_kl}")
                if wandb and wandb.run:
                    wandb.log({
                        "train/loss": float(loss.item()),
                        "train/avg_reward": float(avg_r),
                        "train/avg_kl": float(avg_kl),
                        "train/avg_len": float(avg_len),
                        "train/beta_kl": float(args.beta_kl),
                        "meta/step": int(global_step),
                        "meta/epoch": int(epoch+1),
                        "meta/algo": 0 if args.algo=="dapo" else 1,
                    }, step=global_step)

                if (args.debug_infer_interval > 0) and (global_step % args.debug_infer_interval == 0):
                    d_idx = 0
                    logger.info(f"[sample] prompt: {prompts_to_gen[d_idx][:200].replace('\\n',' ')}")
                    logger.info(f"[sample] output: {gen_texts[d_idx][:200].replace('\\n',' ')} ...")

                if (args.save_interval > 0) and (global_step % args.save_interval == 0) and (global_step > 0):
                    save_adapters(args.output_dir, f"step{global_step}", policy, model_cfg, args, log_to_wandb=True)

                global_step += 1

        # epoch save
        save_adapters(args.output_dir, f"epoch{epoch+1}", policy, model_cfg, args, log_to_wandb=True)

        save_adapters(args.output_dir, "final", policy, model_cfg, args, log_to_wandb=True)
        if args.merge_and_save:
            logger.info("Merging LoRA into base and saving full model...")
            merge_lora_into_base(policy)
            full = os.path.join(args.output_dir, "merged_full_model_final.pt")
            torch.save({"model_state_dict": policy.state_dict(), "config": model_cfg}, full)
            logger.info(f"Saved merged model: {full}")
            if wandb and wandb.run:
                art = wandb.Artifact(name="merged-full-model-final", type="model")
                art.add_file(full); wandb.log_artifact(art)
    finally:
        if trace_file is not None:
            trace_file.close()

# ===================== CLI =====================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RLHF (DAPO/GRPO) for custom GPT with LoRA + reward model + wandb")

    # Data
    ap.add_argument("--training_dir", type=str, default=None)
    ap.add_argument("--train_jsonl", type=str, default=None)
    ap.add_argument("--prompt_template", type=str, default="<|user|>\n{prompt}\n<|assistant|>\n")
    ap.add_argument("--system_prompt", type=str, default=None)
    ap.add_argument("--strip_think", action="store_true")

    # Model / arch
    ap.add_argument("--base_checkpoint", type=str, required=True)
    ap.add_argument("--policy_base_checkpoint", type=str, default=None,
                    help="Full-weights checkpoint to initialize the policy when --base_checkpoint only contains LoRA adapters.")
    ap.add_argument("--ref_checkpoint", type=str, required=True)
    ap.add_argument("--tokenizer_name", type=str, default="gpt2")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_head", type=int, default=16)
    ap.add_argument("--n_layer", type=int, default=24)
    ap.add_argument("--n_kv_head", type=int, default=None,
                    help="Group-query KV heads (defaults to ~n_head/4, adjusted to divide n_head)")
    ap.add_argument("--d_ff", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.0)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_include", type=str, default=r"(attn|Attention|q_proj|k_proj|v_proj|o_proj|mlp|fc)")
    ap.add_argument("--lora_exclude", type=str, default=None)
    ap.add_argument("--merge_and_save", action="store_true")

    # RL algo + sampling
    ap.add_argument("--algo", type=str, choices=["dapo","grpo"], default="dapo")
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--grpo_use_ranks", action="store_true")
    ap.add_argument("--beta_kl", type=float, default=0.05)
    ap.add_argument("--max_new_tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--temperature_floor", type=float, default=0.2,
                    help="Lower bound applied to temperature after 64 decoding steps.")
    ap.add_argument("--suppress_eos_steps", type=int, default=0,
                    help="If >0, mask EOS for the first N decoding steps.")
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--frequency_penalty", type=float, default=0.0)
    ap.add_argument("--presence_penalty", type=float, default=0.0)

    # Rewards
    ap.add_argument("--reward_components", type=str,
                    default="length:0.05,anti_repeat:0.35,ref_match:0.00,reward_model:0.50,think_prefix:0.10",
                    help="Comma list: name:weight,...; names: length,anti_repeat,ref_match,reward_model,think_prefix")
    ap.add_argument("--reward_model_url", type=str, default=None,
                    help="HTTP endpoint that returns {'scores':[...]} for prompts/outputs")
    ap.add_argument("--reward_model_timeout", type=float, default=10.0)
    ap.add_argument("--reward_model_retries", type=int, default=3)
    ap.add_argument("--reward_model_backoff", type=float, default=1.5)
    ap.add_argument("--reward_model_extra", type=str, default=None,
                    help="JSON string with extra payload to send to RM endpoint")
    ap.add_argument("--reward_trace_file", type=str, default=None,
                    help="Optional path for JSONL reward traces (defaults to <output_dir>/reward_traces.jsonl).")

    # Train
    ap.add_argument("--output_dir", type=str, default="./gpt_lora_rlhf")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--logprob_chunk_size", type=int, default=1,
                    help="Chunk size for the log-prob recompute pass (set 0 to process full batch).")
    ap.add_argument("--backprop_context_len", type=int, default=0,
                    help="If >0, only backprop through the last K prompt tokens plus the generated continuation.")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--save_interval", type=int, default=100)
    ap.add_argument("--debug_infer_interval", type=int, default=200)

    # wandb
    ap.add_argument("--wandb_project", type=str, default="rlhf-custom-gpt")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, choices=["online","offline","disabled"], default="online")

    args = ap.parse_args()
    if not args.training_dir and not args.train_jsonl:
        raise SystemExit("Provide --training_dir (recommended) or --train_jsonl.")
    train(args)
