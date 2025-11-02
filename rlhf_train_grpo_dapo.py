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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rlhf")

IGNORE_INDEX = -100


def compute_default_n_kv_head(n_head: int) -> int:
    if n_head <= 0:
        return n_head
    target = max(1, n_head // 4)
    while target > 1 and n_head % target != 0:
        target -= 1
    return max(1, target)

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

def build_prompt(instr: str, ctx: Optional[str], template: str) -> str:
    prompt = instr if ctx in (None, "", []) else f"{ctx}\n\n{instr}"
    return template.format(prompt=prompt)

# ===================== Data =====================
class PromptDataset(Dataset):
    def __init__(self, paths_or_dir: str, prompt_template: str, strip_think: bool = False):
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
        self.rows = rows; self.template = prompt_template
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        return {"prompt_text": build_prompt(r["instruction"], r["context"], self.template),
                "ref_response": r["ref_response"]}

# ===================== Generation + logprob utilities =====================
@torch.no_grad()
def generate_and_logprobs(model: nn.Module, tok, device, prompt_texts: List[str],
                          max_new_tokens: int = 192, temperature: float = 0.7, top_p: float = 0.9,
                          eos_id: Optional[int] = None):
    model.eval()
    B = len(prompt_texts)
    enc = tok(prompt_texts, add_special_tokens=False, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)

    generated = [[] for _ in range(B)]
    logprobs  = [[] for _ in range(B)]
    cur_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        logits, _ = model(cur_ids)
        next_logits = logits[:, -1, :]
        if temperature > 0:
            probs = torch.softmax(next_logits / max(1e-6, temperature), dim=-1)
            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_idx = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_idx.gather(-1, next_idx)
                next_logp = torch.log(torch.gather(probs, -1, next_token))
            else:
                next_token = torch.multinomial(probs, num_samples=1)
                next_logp  = torch.log(torch.gather(probs, -1, next_token))
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            probs = torch.softmax(next_logits, dim=-1)
            next_logp = torch.log(torch.gather(probs, -1, next_token))

        cur_ids = torch.cat([cur_ids, next_token], dim=1)
        for b in range(B):
            t = next_token[b, 0].item()
            generated[b].append(t)
            logprobs[b].append(next_logp[b, 0].item())

        if eos_id is not None and all((seq and seq[-1] == eos_id) for seq in generated):
            break

    out_texts = [tok.decode(seq, skip_special_tokens=False) for seq in generated]
    out_logps = [torch.tensor(v, dtype=torch.float32, device=device) for v in logprobs]
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
            resp = requests.post(endpoint, json=payload, timeout=timeout)
            if resp.status_code == 200:
                obj = resp.json()
                scores = obj.get("scores", [])
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
) -> List[float]:
    base = []
    use_rm = (weights.get("reward_model", 0.0) != 0.0) and bool(reward_model_url)
    rm_scores = []
    if use_rm:
        rm_scores = http_reward_scores(
            prompts=prompts or [""] * len(generated_texts),
            outputs=generated_texts,
            endpoint=reward_model_url,
            timeout=reward_model_timeout,
            max_retries=reward_model_retries,
            backoff=reward_model_backoff,
            extra_payload=reward_model_extra,
        )

    for i, (g, r) in enumerate(zip(generated_texts, ref_texts)):
        s = 0.0
        if weights.get("length", 0) != 0:       s += weights["length"] * reward_length(g)
        if weights.get("anti_repeat", 0) != 0:  s += weights["anti_repeat"] * reward_anti_repeat(g)
        if weights.get("ref_match", 0) != 0:    s += weights["ref_match"] * reward_ref_match(g, r)
        if use_rm:
            s += weights["reward_model"] * float(rm_scores[i])
        base.append(float(s))
    return base

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

    if args.n_kv_head is None:
        args.n_kv_head = compute_default_n_kv_head(args.n_head)
        logger.info(f"Auto-selected n_kv_head={args.n_kv_head} (n_head={args.n_head}).")
    elif args.n_head % args.n_kv_head != 0:
        adjusted = compute_default_n_kv_head(args.n_head)
        logger.warning(
            f"n_head ({args.n_head}) is not divisible by requested n_kv_head ({args.n_kv_head}); adjusting to {adjusted}."
        )
        args.n_kv_head = adjusted

    tok = load_tokenizer(args.tokenizer_name)
    pad_id = tok.pad_token_id; vocab_size = len(tok)

    ds = PromptDataset(args.training_dir if args.training_dir else args.train_jsonl,
                       prompt_template=args.prompt_template, strip_think=args.strip_think)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model_cfg = {
        "vocab_size": vocab_size, "d_model": args.d_model, "n_head": args.n_head, "n_layer": args.n_layer,
        "n_kv_head": args.n_kv_head, "max_seq_len": args.max_seq_len, "dropout": args.dropout,
        "pad_idx": pad_id, "d_ff": args.d_ff or args.d_model*4
    }
    policy = GPT(**model_cfg); load_base(policy, args.base_checkpoint, device)
    replaced = apply_lora_adapters(policy, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
                                   include_regex=args.lora_include,
                                   exclude_regex=(args.lora_exclude or r"(token_embedding|lm_head)"))
    logger.info(f"LoRA applied to {len(replaced)} modules, e.g. {replaced[:8]}")

    ref = GPT(**model_cfg); load_base(ref, args.ref_checkpoint, device)
    for p in ref.parameters(): p.requires_grad = False
    ref.eval()

    opt = optim.AdamW([p for p in policy.parameters() if p.requires_grad],
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
            ref_texts: List[Optional[str]] = batch["ref_response"]

            if args.algo == "grpo":
                expanded_prompts, expanded_refs = [], []
                for p, r in zip(prompts, ref_texts):
                    for _ in range(args.group_size):
                        expanded_prompts.append(p); expanded_refs.append(r)
                prompts_to_gen, refs_to_use = expanded_prompts, expanded_refs
            else:
                prompts_to_gen, refs_to_use = prompts, ref_texts

            with amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.use_amp):
                gen_texts, gen_ids, logps_pol = generate_and_logprobs(
                    policy, tok, device, prompts_to_gen,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    top_p=args.top_p, eos_id=tok.eos_token_id
                )
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
            )

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

            # policy gradient
            losses = []
            seq_lengths = []
            for lp_seq, a in zip(logps_pol, adv):
                if lp_seq.numel() == 0: continue
                seq_lengths.append(lp_seq.numel())
                losses.append(- float(a) * lp_seq.mean())
            if not losses: continue

            loss = torch.stack([torch.tensor(x, device=device) for x in losses]).mean()
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

# ===================== CLI =====================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RLHF (DAPO/GRPO) for custom GPT with LoRA + reward model + wandb")

    # Data
    ap.add_argument("--training_dir", type=str, default=None)
    ap.add_argument("--train_jsonl", type=str, default=None)
    ap.add_argument("--prompt_template", type=str, default="<|user|>\n{prompt}\n<|assistant|>\n")
    ap.add_argument("--strip_think", action="store_true")

    # Model / arch
    ap.add_argument("--base_checkpoint", type=str, required=True)
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

    # Rewards
    ap.add_argument("--reward_components", type=str,
                    default="length:0.05,anti_repeat:0.35,ref_match:0.30,reward_model:0.30",
                    help="Comma list: name:weight,...; names: length,anti_repeat,ref_match,reward_model")
    ap.add_argument("--reward_model_url", type=str, default=None,
                    help="HTTP endpoint that returns {'scores':[...]} for prompts/outputs")
    ap.add_argument("--reward_model_timeout", type=float, default=10.0)
    ap.add_argument("--reward_model_retries", type=int, default=3)
    ap.add_argument("--reward_model_backoff", type=float, default=1.5)
    ap.add_argument("--reward_model_extra", type=str, default=None,
                    help="JSON string with extra payload to send to RM endpoint")

    # Train
    ap.add_argument("--output_dir", type=str, default="./gpt_lora_rlhf")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--save_interval", type=int, default=1000)
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
