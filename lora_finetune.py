#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA Supervised Fine-Tuning (SFT) for your custom GPT model.
- Works with your existing `model.py` (expects GPT forward: logits, _ = model(input_ids))
- Consumes JSONL with keys: instruction, response, (optional) context
- Masks loss on the prompt; learns only on the response span (next-token prediction)
- Saves LoRA adapter weights separately; can also optionally MERGE and save a full model

NEW:
- If --training_dir is provided (e.g. ./training_data), we will:
  * scan all *.jsonl files in that directory (recursively),
  * read and combine examples,
  * deduplicate by (instruction, response),
  * shuffle (seeded),
  * split 85/15 into train/eval,
  * write compiled files into --output_dir and use them for training/eval.

Example usage (new):
python lora_finetune.py \
  --training_dir ./training_data \
  --output_dir ./gpt_lora_sft \
  --tokenizer_name gpt2 \
  --base_checkpoint ./gpt_pretrain_output_hf/model_final_slim.pt \
  --max_seq_len 2048 \
  --batch_size 4 \
  --epochs 2 \
  --learning_rate 1e-4 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --strip_think
"""

import os, re, json, glob, math, logging, argparse, random, csv
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import amp

from transformers import AutoTokenizer

import hashlib
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None
try:
    import orjson as fastjson  # optional speedup
except Exception:
    fastjson = None

# --- Project-local model
try:
    from model import GPT
except Exception as e:
    raise SystemExit("model.py not found or failed to import: %s" % e)

# --- Torch perf knobs ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# ----------------------- Logging ------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("lora_sft")

IGNORE_INDEX = -100

import time


def compute_default_n_kv_head(n_head: int) -> int:
    if n_head <= 0:
        return n_head
    target = max(1, n_head // 4)
    while target > 1 and n_head % target != 0:
        target -= 1
    return max(1, target)

def count_params(model: nn.Module):
    total = 0
    trainable = 0
    lora_only = 0
    for n, p in model.named_parameters():
        n_params = p.numel()
        total += n_params
        if p.requires_grad:
            trainable += n_params
        if "lora_A" in n or "lora_B" in n:
            lora_only += n_params
    pct = (trainable / total * 100.0) if total > 0 else 0.0
    return {
        "total_params": total,
        "trainable_params": trainable,
        "lora_params": lora_only,
        "trainable_pct": pct,
    }

def estimate_train_mem_bytes(trainable_params: int, dtype_bytes: int = 4, optimizer: str = "adamw"):
    """
    Very rough upper bound for training-time memory per trainable parameter:
      params + grads + optimizer states.
    For AdamW: params(4B) + grad(4B) + m(4B) + v(4B) ≈ 16B/param in fp32.
    """
    if optimizer.lower() == "adamw":
        return trainable_params * (dtype_bytes * 4)
    return trainable_params * (dtype_bytes * 2)

def _format_prompt(prompt_template: str, user_text: str) -> str:
    return prompt_template.format(prompt=user_text)


def _load_checkpoint_config(ckpt_path: str) -> Optional[dict]:
    """Returns the `config` block stored in a base checkpoint, if present."""
    try:
        blob = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        logger.warning(f"Could not read checkpoint config from {ckpt_path}: {e}")
        return None
    config = blob.get("config")
    if not isinstance(config, dict):
        logger.warning(f"No config dict found in checkpoint {ckpt_path}.")
        return None
    return config


def _sync_args_with_checkpoint_config(args: Namespace, ckpt_config: dict) -> None:
    """
    Override architectural CLI args with the checkpoint's saved config to avoid
    dimension mismatches (e.g., d_model, n_kv_head).
    """
    override_keys = [
        "d_model",
        "n_head",
        "n_kv_head",
        "n_layer",
        "d_ff",
        "max_seq_len",
        "dropout",
    ]
    for key in override_keys:
        if key in ckpt_config:
            new_val = ckpt_config[key]
            old_val = getattr(args, key, None)
            if old_val != new_val:
                logger.info(
                    f"Setting args.{key}={new_val} to match checkpoint (was {old_val})."
                )
                setattr(args, key, new_val)

@torch.no_grad()
def _debug_generate(
    model: nn.Module,
    tok: AutoTokenizer,
    device: torch.device,
    prompt_text: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    eos_id: Optional[int] = None,
    repetition_penalty: float = 1.0
) -> str:
    """Minimal autoregressive sampler using your model's forward (logits, _)."""
    was_training = model.training
    model.eval()
    try:
        input_ids = tok.encode(prompt_text, add_special_tokens=False)
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits, _ = model(ids)               # (B, T, V)
            next_logits = logits[:, -1, :]       # (B, V)
            
            # --- Repetition penalty (CTRL/GPT-NeoX style) ---
            if repetition_penalty and repetition_penalty > 1.0:
                # tokens generated so far (including the prompt context)
                used = ids[0].unique()
                # gather logits for used tokens
                used_logits = next_logits[0, used]
                # rule: if logit < 0 → multiply by penalty; else → divide by penalty
                penalized = torch.where(
                    used_logits < 0,
                    used_logits * repetition_penalty,
                    used_logits / repetition_penalty,
                )
                next_logits = next_logits.clone()  # avoid in-place on view
                next_logits[0, used] = penalized

            if temperature <= 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits / max(1e-6, temperature), dim=-1)
                if 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    # keep smallest prefix with mass >= top_p
                    mask = cumsum - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_idx = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_idx.gather(-1, next_idx)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_token], dim=1)
            if eos_id is not None and next_token.item() == eos_id:
                break

        out_ids = ids[0].tolist()
        # decode ONLY the freshly generated continuation (not the prompt)
        gen_text = tok.decode(out_ids[len(input_ids):], skip_special_tokens=False)
        return gen_text
    finally:
        if was_training:
            model.train()

# ===================== LoRA modules =====================
class LoRALinear(nn.Module):
    """
    LoRA wrapper for a Linear layer: y = x W^T + x (B A)^T * (alpha/r)
    - base W is frozen
    - only A/B are trainable
    - keeps A/B on the same device/dtype as the base weight
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = base.bias             # pointer (not trainable)
        self.weight = base.weight         # pointer to base weight (frozen)

        # LoRA params
        self.r = r
        self.lora_scaling = alpha / r if r > 0 else 1.0
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Init: A ~ N(0, 0.02), B = 0  (so initial delta is ~0)
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

        # Match device & dtype to base weight immediately
        dev = self.weight.device
        dt  = self.weight.dtype
        self.lora_A.to(device=dev, dtype=dt)
        self.lora_B.to(device=dev, dtype=dt)

        # Freeze base params
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure LoRA params live on the same device/dtype as inputs
        if self.lora_A.weight.device != x.device:
            self.lora_A.to(device=x.device)
            self.lora_B.to(device=x.device)
        base_out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.r <= 0:
            return base_out
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.lora_scaling
        return base_out + lora_out


def _wrap_linear_with_lora(module: nn.Module, name: str,
                           include_regex: Optional[str], exclude_regex: Optional[str],
                           r: int, alpha: int, dropout: float,
                           replaced: List[str]):
    """Recursively replace eligible nn.Linear submodules with LoRALinear wrappers."""
    for child_name, child in list(module.named_children()):
        full_name = f"{name}.{child_name}" if name else child_name
        # Recurse first
        _wrap_linear_with_lora(child, full_name, include_regex, exclude_regex, r, alpha, dropout, replaced)
        # Replace eligible linears
        if isinstance(child, nn.Linear):
            ok = True
            if include_regex and not re.search(include_regex, full_name):
                ok = False
            if exclude_regex and re.search(exclude_regex, full_name):
                ok = False
            if ok:
                wrapped = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, child_name, wrapped)
                replaced.append(full_name)


def apply_lora_adapters(model: nn.Module, r: int, alpha: int, dropout: float,
                        include_regex: Optional[str], exclude_regex: Optional[str]) -> List[str]:
    replaced: List[str] = []
    _wrap_linear_with_lora(model, name="", include_regex=include_regex, exclude_regex=exclude_regex,
                           r=r, alpha=alpha, dropout=dropout, replaced=replaced)
    # Unfreeze only LoRA params
    for n, p in model.named_parameters():
        if ("lora_A.weight" in n) or ("lora_B.weight" in n):
            p.requires_grad = True
        else:
            p.requires_grad = False
    return replaced


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Collect only LoRA adapter weights into a flat state_dict."""
    sd = {}
    for n, p in model.named_parameters():
        if p.requires_grad:  # after apply_lora_adapters only LoRA params are trainable
            sd[n] = p.detach().cpu()
    return sd


def merge_lora_into_base(model: nn.Module):
    """Merge LoRA weights into base linears in-place, then remove LoRA modules."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                delta_w = module.lora_B.weight @ module.lora_A.weight
                delta_w = delta_w * module.lora_scaling
                module.weight.data += delta_w
    for p in model.parameters():
        p.requires_grad = False

# ===================== Data pipeline =====================
import hashlib

def _hash_file_contents(path: str, chunk_bytes: int = 1024 * 1024) -> str:
    """Stable content hash (sha256) without using mtime; ~1MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b: break
            h.update(b)
    return h.hexdigest()[:16]

def _cache_key_for_compiled_jsonl(
    jsonl_path: str,
    tokenizer_name: str,
    max_seq_len: int,
    template: str,
    strip_think: bool,
    add_eos: bool,
) -> str:
    # Content hash makes the key independent of mtime rewrites.
    src_digest = _hash_file_contents(jsonl_path)
    key = f"{os.path.abspath(jsonl_path)}|sha={src_digest}|tok={tokenizer_name}|L={max_seq_len}|tpl={template}|strip={strip_think}|addeos={add_eos}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _read_jsonl_fast(path: str):
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            if fastjson:
                obj = fastjson.loads(line)
            else:
                obj = json.loads(line.decode("utf-8"))
            yield obj

def build_cache_from_jsonl(
    jsonl_path: str,
    tok: AutoTokenizer,
    max_seq_len: int,
    template: str,
    add_eos_to_response: bool,
    strip_think: bool,
    out_dir: str,
    tok_batch_size: int = 2048,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cache_key = _cache_key_for_compiled_jsonl(
        jsonl_path, tok.name_or_path, max_seq_len, template, strip_think, add_eos_to_response
    )
    cache_path = os.path.join(out_dir, f"cache_{os.path.basename(jsonl_path)}.{cache_key}.pt")
    if os.path.exists(cache_path):
        logger.info(f"[cache] Using existing token cache: {cache_path}")
        return cache_path

    # 1) Load raw rows
    instructions, responses, contexts = [], [], []
    for obj in _read_jsonl_fast(jsonl_path):
        instr = (obj.get("instruction") or "").strip()
        resp  = (obj.get("response") or "").strip()
        ctx   = obj.get("context", None)
        if not instr or not resp:
            continue
        if strip_think:
            resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
        instructions.append(instr)
        responses.append(resp)
        contexts.append(ctx)

    n = len(instructions)
    if n == 0:
        raise RuntimeError(f"No valid rows in {jsonl_path}")

    # 2) Build prompts text (vectorized)
    prompts = []
    for i in range(n):
        prompt = instructions[i] if contexts[i] in (None, "", []) else f"{contexts[i]}\n\n{instructions[i]}"
        prompts.append(template.format(prompt=prompt))

    # 3) Batched tokenization for prompts and responses (fast Rust tokenizer)
    all_input_ids = []
    all_labels = []

    for i in tqdm(range(0, n, tok_batch_size), desc=f"[cache] tokenizing {os.path.basename(jsonl_path)}"):
        sl = slice(i, min(i + tok_batch_size, n))
        enc_prompts = tok(prompts[sl], add_special_tokens=False)
        enc_resps   = tok(responses[sl], add_special_tokens=False)

        for p_ids, r_ids in zip(enc_prompts["input_ids"], enc_resps["input_ids"]):
            if add_eos_to_response and tok.eos_token_id is not None:
                r_ids = list(r_ids) + [tok.eos_token_id]

            ids = list(p_ids) + list(r_ids)
            if len(ids) > max_seq_len:
                # Left truncate
                ids = ids[-max_seq_len:]
                prompt_len = max(0, len(ids) - len(r_ids))
            else:
                prompt_len = len(p_ids)

            # Build labels (next-token, prompt masked)
            labels = [IGNORE_INDEX] * len(ids)
            for t in range(prompt_len, len(ids) - 1):
                labels[t] = ids[t + 1]

            all_input_ids.append(torch.tensor(ids, dtype=torch.long))
            all_labels.append(torch.tensor(labels, dtype=torch.long))

    # Save compactly
    torch.save(
        {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "meta": {
                "jsonl": os.path.abspath(jsonl_path),
                "tokenizer": tok.name_or_path,
                "max_seq_len": max_seq_len,
                "template": template,
                "strip_think": strip_think,
                "add_eos": add_eos_to_response,
            },
        },
        cache_path,
    )
    logger.info(f"[cache] Wrote token cache: {cache_path} (samples={len(all_input_ids)})")
    return cache_path


@dataclass
class Sample:
    input_ids: List[int]
    labels: List[int]
    
class CachedSFTDataset(Dataset):
    """Loads pre-tokenized tensors from a cache .pt file built by build_cache_from_jsonl."""
    def __init__(self, cache_path: str, pad_id: int, max_len: int):
        blob = torch.load(cache_path, map_location="cpu")
        self.input_ids_list = blob["input_ids"]
        self.labels_list = blob["labels"]
        self.pad_id = pad_id
        self.max_len = max_len

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids_list[idx],
            "labels": self.labels_list[idx],
        }

class JSONLSFTDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_seq_len: int,
                 template: str = "<|user|>\n{prompt}\n<|assistant|>\n",
                 add_eos_to_response: bool = True,
                 strip_think: bool = False):
        self.samples: List[Sample] = []
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_id = tokenizer.eos_token_id
        self.template = template
        self.add_eos = add_eos_to_response
        self.strip_think = strip_think

        files = glob.glob(path) if any(ch in path for ch in "*?[]") else [path]
        if not files:
            raise FileNotFoundError(f"No JSONL files match: {path}")
        rows = 0
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    instruction = obj.get("instruction", "").strip()
                    response = obj.get("response", "").strip()
                    context = obj.get("context", None)

                    if self.strip_think:
                        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

                    prompt = instruction if context in (None, "", []) else f"{context}\n\n{instruction}"
                    prompt_text = self.template.format(prompt=prompt)

                    prompt_ids = self.tok.encode(prompt_text, add_special_tokens=False)
                    resp_ids = self.tok.encode(response, add_special_tokens=False)
                    if self.add_eos and self.eos_id is not None:
                        resp_ids = resp_ids + [self.eos_id]

                    # concat + left-truncate if needed
                    ids = prompt_ids + resp_ids
                    if len(ids) > max_seq_len:
                        ids = ids[-max_seq_len:]
                        prompt_len = max(0, len(ids) - len(resp_ids))
                    else:
                        prompt_len = len(prompt_ids)

                    # Next-token labels (predict ids[t+1] at position t), prompt masked
                    labels = [IGNORE_INDEX] * len(ids)
                    for t in range(prompt_len, len(ids) - 1):
                        labels[t] = ids[t + 1]
                    assert len(labels) == len(ids)

                    self.samples.append(Sample(ids, labels))
                    rows += 1
        logger.info(f"Loaded {rows} rows → {len(self.samples)} SFT samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {"input_ids": torch.tensor(s.input_ids, dtype=torch.long),
                "labels": torch.tensor(s.labels, dtype=torch.long)}


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_id: int, max_len: int):
    maxL = min(max(len(x["input_ids"]) for x in batch), max_len)
    B = len(batch)
    input_ids = torch.full((B, maxL), pad_id, dtype=torch.long)
    labels = torch.full((B, maxL), IGNORE_INDEX, dtype=torch.long)
    for i, row in enumerate(batch):
        L = min(len(row["input_ids"]), maxL)
        input_ids[i, :L] = row["input_ids"][:L]
        labels[i, :L] = row["labels"][:L]
    return {"input_ids": input_ids, "labels": labels}

# ===================== Tokenizer helpers =====================

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

# ===================== Checkpoint load/save =====================

def build_model_from_args(args: Namespace, vocab_size: int, pad_idx: int) -> Tuple[nn.Module, dict]:
    n_kv_head = getattr(args, 'n_kv_head', None)
    if n_kv_head is None:
        n_kv_head = compute_default_n_kv_head(args.n_head)
        setattr(args, 'n_kv_head', n_kv_head)
        logger.info(f"Auto-selected n_kv_head={n_kv_head} (n_head={args.n_head}).")
    elif args.n_head % n_kv_head != 0:
        adjusted = compute_default_n_kv_head(args.n_head)
        logger.warning(
            f"n_head ({args.n_head}) is not divisible by requested n_kv_head ({n_kv_head}); adjusting to {adjusted}."
        )
        n_kv_head = adjusted
        setattr(args, 'n_kv_head', n_kv_head)

    model_config = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "n_head": args.n_head,
        "n_kv_head": n_kv_head,
        "n_layer": args.n_layer,
        "max_seq_len": args.max_seq_len,
        "dropout": args.dropout,
        "pad_idx": pad_idx,
        "d_ff": args.d_ff or args.d_model * 4,
    }
    model = GPT(**model_config)
    # tie weights if present
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        try:
            model.lm_head.weight = model.token_embedding.weight
        except Exception:
            pass
    return model, model_config


def load_base_weights(model: nn.Module, ckpt_path: str, device: torch.device):
    logger.info(f"Loading base checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    # Handle compiled checkpoints that prefix parameters with '_orig_mod.'
    if any(k.startswith('_orig_mod.') for k in state.keys()):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
    model.to(device)

# ===================== Data compilation helpers =====================


def _scan_jsonl_files(root_dir: str) -> List[str]:
    pattern = os.path.join(os.path.abspath(root_dir), "**", "*.jsonl")
    files = glob.glob(pattern, recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under: {root_dir}")
    return sorted(files)

def compile_dedup_shuffle_split(training_dir: str, eval_fraction: float = 0.15, shuffle_seed: int = 42
) -> Tuple[str, str, int, int, int]:
    files = _scan_jsonl_files(training_dir)
    logger.info(f"Found {len(files)} JSONL files under {training_dir}")

    seen = set()
    examples: List[dict] = []
    dropped_missing = 0
    dropped_dupes = 0

    for pth in tqdm(files, desc="[compile] reading files"):
        with open(pth, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                instr = (obj.get("instruction") or "").strip()
                resp  = (obj.get("response") or "").strip()
                if not instr or not resp:
                    dropped_missing += 1
                    continue
                key = (instr, resp)
                if key in seen:
                    dropped_dupes += 1
                    continue
                seen.add(key)
                examples.append({"instruction": instr, "response": resp, "context": obj.get("context", None)})

    total = len(examples)
    if total == 0:
        raise RuntimeError("No valid examples with both 'instruction' and 'response' after dedup.")

    random.Random(shuffle_seed).shuffle(examples)
    n_eval = max(1, int(round(eval_fraction * total)))
    n_train = max(1, total - n_eval)
    eval_examples = examples[:n_eval]
    train_examples = examples[n_eval:]

    logger.info(f"Compiled {total} unique examples "
                f"(dropped_missing={dropped_missing}, dropped_dupes={dropped_dupes}) "
                f"→ train={n_train}, eval={n_eval}")
    return train_examples, eval_examples, total, n_train, n_eval

def _write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ===================== Train / Eval =====================

def evaluate(model: nn.Module, dl: DataLoader, criterion, device, use_amp: bool, vocab_size: int) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc="[eval]"):
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            with amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, _ = model(x)
                logits = logits[:, :-1, :]
                y_shift = y[:, :-1]
                loss = criterion(logits.reshape(-1, vocab_size), y_shift.reshape(-1))
            non_ignored = (y_shift != IGNORE_INDEX).sum().item()
            if non_ignored > 0:
                total_loss += loss.item() * non_ignored
                total_tokens += non_ignored
    model.train()
    if total_tokens == 0:
        return float('inf'), float('inf')
    avg = total_loss / total_tokens
    try:
        ppl = math.exp(avg)
    except Exception:
        ppl = float('inf')
    return avg, ppl


def train(args: Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script expects CUDA for efficiency.")

    ckpt_config = _load_checkpoint_config(args.base_checkpoint)
    if ckpt_config:
        _sync_args_with_checkpoint_config(args, ckpt_config)

    wandb_run = None
    if getattr(args, "disable_wandb", False):
        logger.info("Weights & Biases logging is disabled via --disable_wandb.")
    elif wandb is None:
        logger.warning("wandb package not installed; disabling Weights & Biases logging.")
        args.disable_wandb = True
    else:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
                resume="allow",
                id=args.wandb_run_id,
            )
            args.wandb_run_id = wandb_run.id
            logger.info(f"Weights & Biases initialized. Run URL: {wandb_run.get_url()}")
        except Exception as e:
            logger.warning(f"Failed to initialize Weights & Biases: {e}. Disabling logging.")
            args.disable_wandb = True
            wandb_run = None

    # Tokenizer
    tok = load_tokenizer(args.tokenizer_name)
    vocab_size = len(tok)
    pad_id = tok.pad_token_id

    # Debug inference prompt + log path
    debug_prompt_text = _format_prompt(args.prompt_template, args.debug_infer_prompt)
    debug_log_path = os.path.join(args.output_dir, "debug_generations.txt")

    # Cache dir
    cache_dir = args.dataset_cache_dir or args.output_dir
    os.makedirs(cache_dir, exist_ok=True)

    # Compile (once) when using --training_dir
    compiled_train = os.path.join(args.output_dir, "compiled_train.jsonl")
    compiled_eval  = os.path.join(args.output_dir, "compiled_eval.jsonl")

    if args.training_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.recompile or not (os.path.exists(compiled_train) and os.path.exists(compiled_eval)):
            train_rows, eval_rows, total, n_train, n_eval = compile_dedup_shuffle_split(
                args.training_dir, eval_fraction=args.eval_fraction, shuffle_seed=args.shuffle_seed
            )
            _write_jsonl(compiled_train, train_rows)
            _write_jsonl(compiled_eval,  eval_rows)
            logger.info(f"Wrote compiled datasets → {compiled_train} ({n_train}), {compiled_eval} ({n_eval})")
        else:
            logger.info(f"Reusing compiled datasets → {compiled_train}, {compiled_eval}")
        train_jsonl_path, eval_jsonl_path = compiled_train, compiled_eval
    else:
        train_jsonl_path, eval_jsonl_path = args.train_jsonl, args.eval_jsonl

    # Build/reuse token caches
    train_cache = build_cache_from_jsonl(
        train_jsonl_path, tok, args.max_seq_len, args.prompt_template,
        add_eos_to_response=not args.no_add_eos, strip_think=args.strip_think,
        out_dir=cache_dir, tok_batch_size=args.tok_batch_size
    )
    eval_cache = None
    if eval_jsonl_path:
        eval_cache = build_cache_from_jsonl(
            eval_jsonl_path, tok, args.max_seq_len, args.prompt_template,
            add_eos_to_response=not args.no_add_eos, strip_think=args.strip_think,
            out_dir=cache_dir, tok_batch_size=args.tok_batch_size
        )

    # Model
    model, model_config = build_model_from_args(args, vocab_size, pad_id)
    load_base_weights(model, args.base_checkpoint, device)

    # Apply LoRA
    include_regex = args.lora_include
    exclude_regex = args.lora_exclude or r"(token_embedding|lm_head)"
    replaced = apply_lora_adapters(
        model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
        include_regex=include_regex, exclude_regex=exclude_regex
    )
    logger.info(f"LoRA applied to {len(replaced)} Linear modules. Examples: {replaced[:8]}")

    # --- Parameter accounting ---
    counts = count_params(model)
    mem_bytes = estimate_train_mem_bytes(counts["trainable_params"], dtype_bytes=4, optimizer="adamw")
    mem_gib = mem_bytes / (1024**3)
    logger.info(
        "Parameter counts: total={:,} | trainable={:,} (LoRA={:,}) | {:.4f}% trainable | ~{:.2f} GiB training-time memory (est.)"
        .format(counts["total_params"], counts["trainable_params"], counts["lora_params"], counts["trainable_pct"], mem_gib)
    )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "param_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": int(time.time()),
            "counts": counts,
            "estimate_bytes": mem_bytes,
            "estimate_gib": mem_gib,
            "optimizer": "adamw",
            "dtype_bytes_assumed": 4,
            "wandb_run_id": getattr(args, "wandb_run_id", None),
        }, f, ensure_ascii=False, indent=2)
    if wandb_run:
        wandb.summary["total_params"] = counts["total_params"]
        wandb.summary["trainable_params"] = counts["trainable_params"]
        wandb.summary["lora_params"] = counts["lora_params"]

    # Datasets (cached)
    train_ds = CachedSFTDataset(train_cache, pad_id, args.max_seq_len)
    eval_ds = CachedSFTDataset(eval_cache, pad_id, args.max_seq_len) if eval_cache else None

    collate = lambda batch: collate_fn(batch, pad_id, args.max_seq_len)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, collate_fn=collate
    )
    eval_dl = DataLoader(
        eval_ds, batch_size=args.eval_batch_size or args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False, collate_fn=collate
    ) if eval_ds else None

    # Optimizer / Loss / AMP
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = amp.GradScaler(enabled=False)  # bf16 autocast; no scaling

    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        bar = tqdm(total=len(train_dl), desc=f"[train] epoch {epoch+1}/{args.epochs}", leave=True)
        running = 0.0; count = 0

        for it, batch in enumerate(train_dl):
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            with amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.use_amp):
                logits, _ = model(x)
                logits = logits[:, :-1, :]
                y_shift = y[:, :-1]
                loss = criterion(logits.reshape(-1, vocab_size), y_shift.reshape(-1))
                loss_scaled = loss / args.grad_accum

            # One-time debug sample of token ids
            if global_step == 0 and it == 0:
                logger.info(f"ex input[:40]={x[0].tolist()[:40]}")
                logger.info(f"ex targ [:40]={y_shift[0].tolist()[:40]}")

            # Backward
            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # Optimizer step on grad accumulation boundary
            if (it + 1) % args.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    opt.step()
                opt.zero_grad(set_to_none=True)

                global_step += 1

                # --- Debug inference EXACTLY once per optimizer step ---
                if args.debug_infer_interval > 0 and (global_step % args.debug_infer_interval == 0):
                    dbg = _debug_generate(
                        model=model,
                        tok=tok,
                        device=device,
                        prompt_text=debug_prompt_text,
                        max_new_tokens=args.debug_infer_max_new_tokens,
                        temperature=args.debug_temp,
                        top_p=args.debug_top_p,
                        eos_id=tok.eos_token_id,
                        repetition_penalty=args.debug_repetition_penalty,
                    )
                    preview = dbg.replace("\n", " ")[:200]
                    logger.info(
                        f"[debug-gen] step {global_step} | prompt='{args.debug_infer_prompt}' → "
                        f"{preview}{'...' if len(dbg) > 200 else ''}"
                    )
                    os.makedirs(args.output_dir, exist_ok=True)
                    with open(debug_log_path, "a", encoding="utf-8") as f:
                        f.write(f"STEP {global_step}\nPROMPT: {args.debug_infer_prompt}\nOUTPUT:\n{dbg}\n")
                        f.write("-" * 80 + "\n")
                    csv_path = os.path.join(args.output_dir, "debug_generations.csv")
                    append_header = not os.path.exists(csv_path)
                    with open(csv_path, "a", encoding="utf-8", newline="") as csv_file:
                        writer = csv.writer(csv_file)
                        if append_header:
                            writer.writerow(["timestamp", "epoch", "global_step", "prompt", "completion"])
                        writer.writerow([
                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            epoch + 1,
                            global_step,
                            args.debug_infer_prompt,
                            dbg,
                        ])
                    if wandb_run:
                        wandb.log({
                            "debug/prompt": args.debug_infer_prompt,
                            "debug/generated_text": dbg[:1024],
                        }, step=global_step)

                # --- Eval EXACTLY once per optimizer step ---
                if eval_dl and args.eval_interval > 0 and (global_step % args.eval_interval == 0):
                    eval_loss, eval_ppl = evaluate(model, eval_dl, criterion, device, args.use_amp, vocab_size)
                    logger.info(f"[eval] step {global_step}: loss {eval_loss:.4f} | ppl {eval_ppl:.2f}")
                    if wandb_run:
                        wandb.log({
                            "eval/loss": eval_loss,
                            "eval/perplexity": eval_ppl,
                            "epoch": epoch + 1,
                        }, step=global_step)
                        best_loss = wandb.summary.get("best_eval_loss", float("inf"))
                        if eval_loss < best_loss:
                            wandb.summary["best_eval_loss"] = eval_loss
                            wandb.summary["best_eval_perplexity"] = eval_ppl
                            wandb.summary["best_eval_step"] = global_step
                    save_adapters(args, model, model_config, global_step, tag="eval")

            # Progress bar + running stats
            running += loss.item(); count += 1
            avg = running / max(1, count)
            try:
                ppl = math.exp(avg)
            except Exception:
                ppl = float('inf')
            bar.set_postfix(loss=f"{avg:.4f}", ppl=f"{ppl:.2f}", step=global_step)
            bar.update(1)

            # Periodic console log (avoid step 0 spam)
            if global_step > 0 and (global_step % args.log_interval == 0) and count > 0:
                logger.info(f"step {global_step}: train loss {avg:.4f} | ppl {ppl:.2f}")
                if wandb_run:
                    wandb.log({
                        "train/loss": avg,
                        "train/perplexity": ppl,
                        "train/learning_rate": opt.param_groups[0]["lr"],
                        "epoch": epoch + 1,
                        "processed_samples": global_step * args.batch_size * args.grad_accum,
                    }, step=global_step)
                running, count = 0.0, 0

        bar.close()
        save_adapters(args, model, model_config, global_step, tag=f"epoch{epoch+1}")

    save_adapters(args, model, model_config, global_step, tag="final", merge=args.merge_and_save)
    logger.info("Training complete.")
    if wandb_run:
        wandb.finish()


def save_adapters(args: Namespace, model: nn.Module, model_config: dict, step: int, tag: str,
                  merge: bool = False):
    os.makedirs(args.output_dir, exist_ok=True)

    # Save LoRA only
    ad_sd = lora_state_dict(model)
    ad_path = os.path.join(args.output_dir, f"lora_adapters_step{step}_{tag}.pt")
    torch.save({
        "adapters_state_dict": ad_sd,
        "config": {
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
            "include": args.lora_include, "exclude": args.lora_exclude,
            "tokenizer_name": args.tokenizer_name,
        }
    }, ad_path)
    logger.info(f"Saved LoRA adapters to {ad_path}")

    if merge:
        logger.info("Merging adapters into base weights and saving full model...")
        merge_lora_into_base(model)
        full_path = os.path.join(args.output_dir, f"merged_full_model_step{step}_{tag}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": model_config,
            "args": {
                "tokenizer_name": args.tokenizer_name,
                "d_model": args.d_model,
                "n_head": args.n_head,
                "n_layer": args.n_layer,
                "d_ff": args.d_ff or args.d_model*4,
                "max_seq_len": args.max_seq_len,
            }
        }, full_path)
        logger.info(f"Saved merged full model to {full_path}")

# ===================== CLI =====================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LoRA SFT for custom GPT on JSONL instruction data")

    # Data
    p.add_argument("--training_dir", type=str, default=None,
                   help="Directory containing many JSONL files to combine (takes precedence if set)")
    p.add_argument("--train_jsonl", type=str, default=None,
                   help="Path or glob to training JSONL (used only if --training_dir is NOT set)")
    p.add_argument("--eval_jsonl", type=str, default=None,
                   help="Optional eval JSONL path or glob (ignored if --training_dir is set)")
    p.add_argument("--eval_fraction", type=float, default=0.15,
                   help="Fraction for eval split when using --training_dir")
    p.add_argument("--shuffle_seed", type=int, default=42,
                   help="Shuffle seed used when compiling data from --training_dir")
    p.add_argument("--prompt_template", type=str, default="<|user|>\n{prompt}\n<|assistant|>\n",
                   help="Python format string; available field: {prompt}")
    p.add_argument("--strip_think", action="store_true",
                   help="Remove <think>...</think> from responses before training")
    p.add_argument("--no_add_eos", action="store_true", help="Do not append EOS to responses")

    # Model
    p.add_argument("--base_checkpoint", type=str, required=True, help="Path to pretrained base checkpoint (.pt)")
    p.add_argument("--tokenizer_name", type=str, default="gpt2")
    p.add_argument("--max_seq_len", type=int, default=4096)

    # Architecture (must match your base model)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--n_head", type=int, default=16)
    p.add_argument("--n_layer", type=int, default=24)
    p.add_argument("--n_kv_head", type=int, default=None,
                   help="Group-query KV heads (defaults to ~n_head/4, adjusted to divide n_head)")
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0)

    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_include", type=str, default=r"(attn|Attention|q_proj|k_proj|v_proj|o_proj|mlp|fc)",
                   help="Regex of module names to include")
    p.add_argument("--lora_exclude", type=str, default=None, help="Regex of module names to exclude")
    p.add_argument("--merge_and_save", action="store_true", help="Also save a merged full model at the end")
    p.add_argument("--dataset_cache_dir", type=str, default=None,
               help="Where to store tokenized dataset caches (.pt). Defaults to --output_dir")
    p.add_argument("--tok_batch_size", type=int, default=2048,
                help="Batch size for fast batched tokenization during caching")

    # Train
    p.add_argument("--output_dir", type=str, default="./gpt_lora_sft")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=5000)
    p.add_argument("--recompile", action="store_true",
               help="Force recompile of train/eval JSONL from --training_dir even if outputs already exist.")
    
    p.add_argument("--debug_infer_interval", type=int, default=50,
               help="Every N optimizer steps, run a quick debug generation.")
    p.add_argument("--debug_infer_prompt", type=str, default="What is love ?",
                help="Prompt to use for periodic debug generations.")
    p.add_argument("--debug_infer_max_new_tokens", type=int, default=512,
                help="Max new tokens to generate during debug inference.")
    p.add_argument("--debug_temp", type=float, default=0.7,
                help="Sampling temperature for debug inference (0 = greedy).")
    p.add_argument("--debug_top_p", type=float, default=0.9,
                help="Nucleus sampling top_p for debug inference (ignored if temp<=0).")
    p.add_argument(
        "--debug_repetition_penalty",
        type=float,
        default=1.1,  # >1.0 discourages repeats
        help="Repetition penalty for debug inference (CTRL/GPT-NeoX style).",
    )

    # Weights & Biases
    p.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default="gpt_lora_sft", help="Weights & Biases project name.")
    p.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (team or username).")
    p.add_argument("--wandb_run_name", type=str, default=None, help="Optional custom W&B run name.")
    p.add_argument("--wandb_run_id", type=str, default=None, help="Resume a specific W&B run ID.")
    
    args = p.parse_args()
    

    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    # Backward-compat guard: require some data source
    if not args.training_dir and not args.train_jsonl:
        raise SystemExit("Provide --training_dir (recommended) or --train_jsonl.")

    train(args)
