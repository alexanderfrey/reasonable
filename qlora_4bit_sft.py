#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA (4-bit) Supervised Fine-Tuning (SFT) for your custom GPT model.

- Quantizes ALL nn.Linear layers to 4-bit (nf4) via bitsandbytes
- Adds LoRA adapters on top (learnable) while 4-bit base weights stay frozen
- Trains only LoRA params (QLoRA)
- Consumes JSONL with keys: instruction, response, (optional) context
- Masks loss on the prompt; learns only on the response span
- Saves LoRA adapter weights; can optionally MERGE (dequantize+fold) and save a full model

Example:

python qlora_4bit_sft.py \
  --train_jsonl /data/train.jsonl \
  --eval_jsonl /data/val.jsonl \
  --output_dir ./gpt_qlora_sft \
  --tokenizer_name gpt2 \
  --base_checkpoint ./gpt_pretrain_output_4096/model_best_eval.pt \
  --max_seq_len 4096 \
  --batch_size 4 \
  --epochs 2 \
  --learning_rate 1e-4 \
  --lora_r 64 --lora_alpha 16 --lora_dropout 0.05 \
  --use_bf16 \
  --strip_think
"""

import os, re, json, glob, math, time, logging, argparse
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import amp

from transformers import AutoTokenizer

# ---- bitsandbytes (QLoRA) ----
import bitsandbytes as bnb

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
logger = logging.getLogger("qlora_sft")

IGNORE_INDEX = -100

# ===================== 4-bit Q helpers =====================

def _compute_dtype(use_bf16: bool):
    if use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    return torch.float16

def _to_linear8bit_from_linear(linear: nn.Linear, device: torch.device) -> nn.Module:
    """
    Convert torch.nn.Linear -> bitsandbytes Linear8bitLt using from_float().
    Keeps base frozen (QLoRA-style).
    """
    if not hasattr(bnb.nn.Linear8bitLt, "from_float"):
        raise RuntimeError("Your bitsandbytes build lacks Linear8bitLt.from_float(). Please upgrade bnb.")
    has_bias = linear.bias is not None
    qlin = bnb.nn.Linear8bitLt(
        linear.in_features,
        linear.out_features,
        bias=has_bias,
    ).to(device)
    qlin = qlin.from_float(linear.to(device))
    for p in qlin.parameters():
        p.requires_grad = False
    return qlin

def _to_linear4bit_from_linear(linear: nn.Linear,
                               compute_dtype: torch.dtype,
                               quant_type: str = "nf4",
                               double_quant: bool = True,
                               device: torch.device = torch.device("cuda")) -> nn.Module:
    """
    Convert torch.nn.Linear -> bitsandbytes Linear4bit using the OFFICIAL from_float path.
    We do NOT touch .weight manually. We also move to CUDA immediately to init quant state.
    """
    has_bias = linear.bias is not None

    if not hasattr(bnb.nn.Linear4bit, "from_float"):
        raise RuntimeError(
            "Your bitsandbytes build lacks Linear4bit.from_float(). "
            "Please upgrade to bitsandbytes >= 0.43 (recommended). "
            "Alternatively, change --bnb_quant_type to 8bit and use Linear8bitLt."
        )

    qlin = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=has_bias,
        compute_dtype=compute_dtype,
        quant_type=quant_type,
        compress_statistics=double_quant,
    )
    # Move to CUDA BEFORE from_float so quant state is set up on the right device
    qlin = qlin.to(device)

    # Proper quantization path (lets bnb pack weights & set quant_state)
    qlin = qlin.from_float(linear.to(device))

    # Freeze base params (QLoRA)
    for p in qlin.parameters():
        p.requires_grad = False

    return qlin


def quantize_model_to_4bit(model: nn.Module,
                           include_regex: Optional[str],
                           exclude_regex: Optional[str],
                           compute_dtype: torch.dtype,
                           quant_type: str = "nf4",
                           double_quant: bool = True,
                           device: torch.device = torch.device("cuda")) -> List[str]:
    """
    Replace eligible nn.Linear modules with proper Linear4bit via from_float(), on CUDA.
    """
    replaced: List[str] = []

    def _recurse(module: nn.Module, prefix: str):
        for child_name, child in list(module.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            _recurse(child, full)

            if isinstance(child, nn.Linear):
                ok = True
                if include_regex and not re.search(include_regex, full):
                    ok = False
                if exclude_regex and re.search(exclude_regex, full):
                    ok = False
                if ok:
                    qlin = _to_linear4bit_from_linear(
                        child, compute_dtype=compute_dtype,
                        quant_type=quant_type, double_quant=double_quant, device=device
                    )
                    setattr(module, child_name, qlin)
                    replaced.append(full)

    _recurse(model, "")
    return replaced

def quantize_model_to_bits(model: nn.Module,
                           include_regex: Optional[str],
                           exclude_regex: Optional[str],
                           compute_dtype: torch.dtype,
                           quant_type: str,              # "8bit", "nf4", or "fp4"
                           double_quant: bool,
                           device: torch.device) -> List[str]:
    """
    Replace eligible nn.Linear modules with bnb quantized modules:
      - 8bit  -> Linear8bitLt
      - nf4/fp4 -> Linear4bit
    """
    replaced: List[str] = []

    def _recurse(module: nn.Module, prefix: str):
        for child_name, child in list(module.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            _recurse(child, full)

            if isinstance(child, nn.Linear):
                ok = True
                if include_regex and not re.search(include_regex, full):
                    ok = False
                if exclude_regex and re.search(exclude_regex, full):
                    ok = False
                if not ok:
                    continue

                if quant_type.lower() == "8bit":
                    qlin = _to_linear8bit_from_linear(child, device=device)
                else:
                    # 4-bit path (nf4/fp4)
                    qlin = _to_linear4bit_from_linear(
                        child,
                        compute_dtype=compute_dtype,
                        quant_type=quant_type.lower(),
                        double_quant=double_quant,
                        device=device,
                    )
                setattr(module, child_name, qlin)
                replaced.append(full)

    _recurse(model, "")
    return replaced

# ===================== LoRA (on top of 4-bit) =====================

class LoRALinearOn4bit(nn.Module):
    """
    LoRA head that augments a frozen base Linear4bit (or any Linear-like) module.
    - base_module: module with .forward(x) and attributes .in_features/.out_features/.bias
    - Only A,B (and optional dropout) are trainable
    """
    def __init__(self, base_module: nn.Module, r: int, alpha: int, dropout: float, dtype=torch.float16):
        super().__init__()
        self.base = base_module
        # infer dims
        in_f = getattr(base_module, "in_features")
        out_f = getattr(base_module, "out_features")
        self.r = r
        self.scaling = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # LoRA A,B in higher precision (fp16/bf16)
        self.lora_A = nn.Linear(in_f, r, bias=False, dtype=dtype)
        self.lora_B = nn.Linear(r, out_f, bias=False, dtype=dtype)
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)
        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r <= 0:
            return out
        delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        # Cast if needed to match base dtype
        if delta.dtype != out.dtype:
            delta = delta.to(out.dtype)
        return out + delta


def apply_lora_on_4bit(model: nn.Module, r: int, alpha: int, dropout: float,
                       include_regex: Optional[str], exclude_regex: Optional[str],
                       lora_dtype: torch.dtype) -> List[str]:
    """
    Wrap eligible Linear4bit (or Linear) leaves with LoRA augmentation modules.
    """
    wrapped: List[str] = []

    def _recurse(module: nn.Module, prefix: str):
        for child_name, child in list(module.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            _recurse(child, full)

            # We attach LoRA on top of (quantized) linear modules
            is_candidate = isinstance(child, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, nn.Linear))
            if not is_candidate:
                continue

            ok = True
            if include_regex and not re.search(include_regex, full):
                ok = False
            if exclude_regex and re.search(exclude_regex, full):
                ok = False
            if ok:
                lora_wrap = LoRALinearOn4bit(child, r=r, alpha=alpha, dropout=dropout, dtype=lora_dtype)
                setattr(module, child_name, lora_wrap)
                # Enable only LoRA params
                for n, p in lora_wrap.named_parameters():
                    p.requires_grad = ("lora_A" in n) or ("lora_B" in n)
                wrapped.append(full)

    _recurse(model, "")
    return wrapped


def collect_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Collect only LoRA params (A/B) into a flat state dict.
    """
    sd = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            sd[n] = p.detach().cpu()
    return sd


def merge_lora_into_base_inplace(model: nn.Module):
    """
    Fold LoRA weights into the *dequantized* view of each base Linear (export path).
    Note: For 4-bit bases, we compute deltaW in higher precision and add to a float copy.
    This is intended only for saving a merged float checkpoint.
    """
    for module in model.modules():
        if isinstance(module, LoRALinearOn4bit):
            base = module.base
            with torch.no_grad():
                # Construct a float weight view: deltaW = B @ A * scaling
                delta = (module.lora_B.weight @ module.lora_A.weight) * module.scaling  # [out,in]
                # Try to access base weight in float
                try:
                    # Many bnb modules expose .weight as quantized params; use a float copy for merge-export
                    if hasattr(base, "weight") and hasattr(base.weight, "to"):
                        Wf = base.weight.float().detach().cpu().clone()
                    else:
                        # Fallback: create a zeros weight
                        in_f = getattr(base, "in_features")
                        out_f = getattr(base, "out_features")
                        Wf = torch.zeros((out_f, in_f), dtype=torch.float32)
                    Wf += delta.detach().cpu().float()
                    # Stash merged weight back on module for export path
                    base._merged_full_weight_for_export = Wf  # custom hook attribute
                except Exception:
                    pass
    # Freeze everything post-merge
    for p in model.parameters():
        p.requires_grad = False

# ===================== Data pipeline (unchanged) =====================
@dataclass
class Sample:
    input_ids: List[int]
    labels: List[int]

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

                    ids = prompt_ids + resp_ids
                    if len(ids) > max_seq_len:
                        ids = ids[-max_seq_len:]
                        prompt_len = max(0, len(ids) - len(resp_ids))
                    else:
                        prompt_len = len(prompt_ids)

                    labels = [IGNORE_INDEX] * prompt_len + ids[prompt_len:]
                    assert len(labels) == len(ids)

                    self.samples.append(Sample(ids, labels))
                    rows += 1
        logger.info(f"Loaded {rows} rows → {len(self.samples)} SFT samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids": torch.tensor(s.input_ids, dtype=torch.long),
            "labels": torch.tensor(s.labels, dtype=torch.long),
        }

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
    try: tok.model_max_length = int(1e12)
    except Exception: pass
    return tok

# ===================== Checkpoint load/save =====================

def build_model_from_args(args: Namespace, vocab_size: int, pad_idx: int) -> Tuple[nn.Module, dict]:
    model_config = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "n_head": args.n_head,
        "n_kv_head": getattr(args, 'n_kv_head', 8),
        "n_layer": args.n_layer,
        "max_seq_len": args.max_seq_len,
        "dropout": args.dropout,
        "pad_idx": pad_idx,
        "d_ff": args.d_ff or args.d_model * 4,
    }
    model = GPT(**model_config)
    # Tie
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
    if any(k.startswith('_orig_mod.') for k in state.keys()):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
    model.to(device)

# ===================== Train / Eval =====================

def evaluate(model: nn.Module, dl: DataLoader, criterion, device, use_amp: bool, vocab_size: int) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dl:
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            with amp.autocast("cuda", enabled=use_amp):
                logits, _ = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            non_ignored = (y != IGNORE_INDEX).sum().item()
            if non_ignored > 0:
                total_loss += loss.item() * non_ignored
                total_tokens += non_ignored
    model.train()
    if total_tokens == 0:
        return float('inf'), float('inf')
    avg = total_loss / total_tokens
    try: ppl = math.exp(avg)
    except Exception: ppl = float('inf')
    return avg, ppl


def train(args: Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA expects CUDA.")
    device = torch.device("cuda")

    # Dtypes
    compute_dtype = _compute_dtype(args.use_bf16)
    lora_dtype = compute_dtype

    # Tokenizer
    tok = load_tokenizer(args.tokenizer_name)
    vocab_size = len(tok)
    pad_id = tok.pad_token_id

    # Build & load float base
    model, model_config = build_model_from_args(args, vocab_size, pad_id)
    load_base_weights(model, args.base_checkpoint, device)
    
    model.to(device)

    # 1) Quantize all Linear → 4-bit (nf4)
    include_regex_q = args.quant_include
    exclude_regex_q = args.quant_exclude or r"(token_embedding|lm_head)"
    q_replaced = quantize_model_to_bits(
        model,
        include_regex_q,
        exclude_regex_q,
        compute_dtype=compute_dtype,
        quant_type=args.bnb_quant_type,           # now supports "8bit" | "nf4" | "fp4"
        double_quant=not args.no_double_quant,    # ignored by 8bit
        device=device,
    )
    logger.info(f"Quantized {len(q_replaced)} Linear modules to 4-bit. Examples: {q_replaced[:8]}")

    # 2) Attach LoRA heads on top of (quantized) linears
    include_regex_l = args.lora_include
    exclude_regex_l = args.lora_exclude or r"(token_embedding|lm_head)"
    lora_wrapped = apply_lora_on_4bit(
        model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
        include_regex=include_regex_l, exclude_regex=exclude_regex_l, lora_dtype=lora_dtype
    )
    logger.info(f"LoRA attached to {len(lora_wrapped)} modules. Examples: {lora_wrapped[:8]}")
    model.to(device)
    # Ensure only LoRA trains
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "No trainable LoRA params found."

    # Datasets / Loaders
    train_ds = JSONLSFTDataset(args.train_jsonl, tok, args.max_seq_len, template=args.prompt_template,
                               add_eos_to_response=not args.no_add_eos, strip_think=args.strip_think)
    eval_ds = JSONLSFTDataset(args.eval_jsonl, tok, args.max_seq_len, template=args.prompt_template,
                              add_eos_to_response=not args.no_add_eos, strip_think=args.strip_think) if args.eval_jsonl else None

    collate = lambda batch: collate_fn(batch, pad_id, args.max_seq_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          pin_memory=True, drop_last=True, collate_fn=collate)
    eval_dl = DataLoader(eval_ds, batch_size=args.eval_batch_size or args.batch_size, shuffle=False, num_workers=args.num_workers,
                         pin_memory=True, drop_last=False, collate_fn=collate) if eval_ds else None

    # Optimizer: PagedAdamW32 (bnb) for LoRA params
    opt = bnb.optim.PagedAdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = amp.GradScaler("cuda", enabled=args.use_amp)

    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        running = 0.0
        count = 0
        for it, batch in enumerate(train_dl):
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            with amp.autocast("cuda", enabled=args.use_amp):
                logits, _ = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                loss_scaled = loss / args.grad_accum

            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (it + 1) % args.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            running += loss.item()
            count += 1
            if global_step % args.log_interval == 0 and count > 0:
                avg = running / count
                try: ppl = math.exp(avg)
                except Exception: ppl = float('inf')
                logger.info(f"step {global_step}: train loss {avg:.4f} | ppl {ppl:.2f}")
                running, count = 0.0, 0

            if eval_dl and args.eval_interval > 0 and global_step > 0 and (global_step % args.eval_interval == 0):
                eval_loss, eval_ppl = evaluate(model, eval_dl, criterion, device, args.use_amp, vocab_size)
                logger.info(f"[eval] step {global_step}: loss {eval_loss:.4f} | ppl {eval_ppl:.2f}")
                save_adapters(args, model, model_config, global_step, tag="eval")

        save_adapters(args, model, model_config, global_step, tag=f"epoch{epoch+1}")

    save_adapters(args, model, model_config, global_step, tag="final", merge=args.merge_and_save)
    logger.info("Training complete.")


def _export_full_state_dict_with_merge(model: nn.Module, model_config: dict, tokenizer_name: str) -> dict:
    """
    Build a float32 export state dict after merge.
    For each LoRA-wrapped Linear, we try to grab the merged full weight set earlier.
    """
    export_sd = {}
    for name, mod in model.named_modules():
        # If this is a LoRA wrapper, dive to base
        if isinstance(mod, LoRALinearOn4bit):
            base = mod.base
            w_key = f"{name}.base.weight"
            # Use merged weight if present
            if hasattr(base, "_merged_full_weight_for_export"):
                export_sd[w_key] = base._merged_full_weight_for_export
            # Bias
            if getattr(base, "bias", None) is not None:
                export_sd[f"{name}.base.bias"] = base.bias.detach().cpu().float()
        elif isinstance(mod, (bnb.nn.Linear4bit, nn.Linear)):
            # Non-LoRA linears (rare if excluded)
            if hasattr(mod, "weight"):
                export_sd[f"{name}.weight"] = mod.weight.detach().cpu().float()
            if getattr(mod, "bias", None) is not None:
                export_sd[f"{name}.bias"] = mod.bias.detach().cpu().float()

    # Fallback: also include the model.state_dict() where possible
    # (won't overwrite merged weights we already set)
    base_sd = model.state_dict()
    for k, v in base_sd.items():
        if k not in export_sd:
            try:
                export_sd[k] = v.detach().cpu().float()
            except Exception:
                pass

    return {
        "model_state_dict": export_sd,
        "config": model_config,
        "args": {
            "tokenizer_name": tokenizer_name,
            "d_model": model_config["d_model"],
            "n_head": model_config["n_head"],
            "n_layer": model_config["n_layer"],
            "d_ff": model_config["d_ff"],
            "max_seq_len": model_config["max_seq_len"],
        }
    }


def save_adapters(args: Namespace, model: nn.Module, model_config: dict, step: int, tag: str,
                  merge: bool = False):
    os.makedirs(args.output_dir, exist_ok=True)

    # Save LoRA only (trainable params)
    ad_sd = collect_lora_state_dict(model)
    ad_path = os.path.join(args.output_dir, f"qlora_adapters_step{step}_{tag}.pt")
    torch.save({
        "adapters_state_dict": ad_sd,
        "config": {
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
            "include": args.lora_include, "exclude": args.lora_exclude,
            "tokenizer_name": args.tokenizer_name,
            "bnb_quant_type": args.bnb_quant_type,
            "double_quant": not args.no_double_quant,
        }
    }, ad_path)
    logger.info(f"Saved QLoRA adapters to {ad_path}")

    if merge:
        logger.info("Merging LoRA into base (exporting float checkpoint)...")
        merge_lora_into_base_inplace(model)
        full_path = os.path.join(args.output_dir, f"merged_full_model_step{step}_{tag}.pt")
        payload = _export_full_state_dict_with_merge(model, model_config, args.tokenizer_name)
        torch.save(payload, full_path)
        logger.info(f"Saved merged full model to {full_path}")

# ===================== CLI =====================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="QLoRA 4-bit SFT for custom GPT on JSONL instruction data")

    # Data
    p.add_argument("--train_jsonl", type=str, required=True, help="Path or glob to training JSONL")
    p.add_argument("--eval_jsonl", type=str, default=None, help="Optional eval JSONL path or glob")
    p.add_argument("--prompt_template", type=str, default="<|user|>\n{prompt}\n<|assistant|>\n",
                   help="Python format string; available field: {prompt}")
    p.add_argument("--strip_think", action="store_true", help="Remove <think>...</think> from responses before training")
    p.add_argument("--no_add_eos", action="store_true", help="Do not append EOS to responses")

    # Model
    p.add_argument("--base_checkpoint", type=str, required=True, help="Path to pretrained base checkpoint (.pt)")
    p.add_argument("--tokenizer_name", type=str, default="gpt2")
    p.add_argument("--max_seq_len", type=int, default=4096)

    # Architecture (must match your base model)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--n_head", type=int, default=16)
    p.add_argument("--n_layer", type=int, default=24)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0)

    # Quantization (bnb 4-bit)
    p.add_argument("--no_double_quant", action="store_true", help="Disable double quantization (compress stats)")
    p.add_argument("--use_bf16", action="store_true", help="Use bfloat16 compute if available (Hopper/Ampere)")
    p.add_argument("--bnb_quant_type", type=str, default="8bit", choices=["nf4", "fp4", "8bit"])
    
    # Which modules to quantize (and/or LoRA)
    p.add_argument("--quant_include", type=str,
                   default=r"(attn|Attention|q_proj|k_proj|v_proj|o_proj|mlp|fc|proj|down_proj|up_proj)")
    p.add_argument("--quant_exclude", type=str, default=r"(token_embedding|lm_head)")
    p.add_argument("--lora_include", type=str,
                   default=r"(attn|Attention|q_proj|k_proj|v_proj|o_proj|mlp|fc|proj|down_proj|up_proj)",
                   help="Regex of module names to attach LoRA")
    p.add_argument("--lora_exclude", type=str, default=None, help="Regex of module names to exclude from LoRA")

    # LoRA
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--merge_and_save", action="store_true", help="Also save a merged float model at the end")

    # Train
    p.add_argument("--output_dir", type=str, default="./gpt_qlora_sft")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=500)

    args = p.parse_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    train(args)