import math
from collections import OrderedDict
from typing import Optional, Tuple, List, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

# FlashAttention imports (required)
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input


# --- RoPE helpers ---

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute RoPE freqs as complex numbers: cos(m*theta_i) + i*sin(m*theta_i).
    Returns shape [end, dim // 2] (complex64).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def _reshape_freqs_for_broadcast(freqs_cis: torch.Tensor, x_complex: torch.Tensor) -> torch.Tensor:
    """
    x_complex: [B, S, H, D/2] (complex)
    freqs_cis: [S, D/2] -> [1, S, 1, D/2] for broadcasting.
    """
    assert freqs_cis.shape == (x_complex.shape[1], x_complex.shape[-1]), (
        f"RoPE shape mismatch: freqs_cis={freqs_cis.shape} vs "
        f"expected=({x_complex.shape[1]}, {x_complex.shape[-1]})"
    )
    return freqs_cis.unsqueeze(0).unsqueeze(2)


def apply_rotary_pos_emb(
    xq: torch.Tensor,  # [B, S, H, D]
    xk: torch.Tensor,  # [B, S, H, D]
    freqs_cis: torch.Tensor,  # [S, D/2] (complex)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE via complex multiply on paired last-dim.
    """
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    fc = _reshape_freqs_for_broadcast(freqs_cis, xq_c)
    xq_rot = torch.view_as_real(xq_c * fc).flatten(3).type_as(xq)
    xk_rot = torch.view_as_real(xk_c * fc).flatten(3).type_as(xk)
    return xq_rot, xk_rot


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        # keep LN in fp32 numerics implicitly
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt().to(x.dtype)
        return self.weight * (x * norm)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat K/V heads to match Q heads (GQA).
    x: [B, S, Hkv, D] -> [B, S, Hq, D] where Hq = Hkv * n_rep
    """
    if n_rep == 1:
        return x
    B, S, H, D = x.shape
    return x.unsqueeze(3).expand(B, S, H, n_rep, D).reshape(B, S, H * n_rep, D)

# --- Core blocks ---

class MultiHeadAttention(nn.Module):
    """
    FlashAttention(v1) + RoPE + KV caching + Sliding Window, now with GQA:
      - Hq query heads
      - Hkv key/value heads (Hq must be a multiple of Hkv)
      - Cache stores K/V in Hkv heads; we repeat to Hq only for attention.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,                    # Hq
        dropout: float = 0.1,
        max_kv_len: Optional[int] = None,
        n_kv_head: Optional[int] = None # Hkv (defaults to Hq => MHA)
    ):
        super().__init__()
        Hq = n_head
        Hkv = n_kv_head if n_kv_head is not None else Hq

        assert d_model % Hq == 0, "d_model must be divisible by n_head (Hq)"
        d_k = d_model // Hq
        assert d_k % 2 == 0, "RoPE needs even head_dim"
        assert Hq % Hkv == 0, "n_head (Hq) must be a multiple of n_kv_head (Hkv)"

        self.d_model = d_model
        self.n_head = Hq
        self.n_kv_head = Hkv
        self.d_k = d_k
        self.n_rep = Hq // Hkv
        self.max_kv_len = max_kv_len

        # Projections: Q in Hq heads, K/V in Hkv heads
        self.w_q = nn.Linear(d_model, Hq * d_k, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * Hkv * d_k, bias=False)

        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                        # [B, S_q, D]
        freqs_cis: torch.Tensor,                # [S_q, d_k/2] (complex)
        mask: Optional[torch.Tensor] = None,    # [B, S_kv] (1=attend, 0=pad) over full KV (past+current)
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,  # (k_past, v_past, cache_pos)
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, None, Optional[Tuple[torch.Tensor, torch.Tensor, int]]]:

        B, S_q, _ = x.size()

        cache_position = 0
        k_past = v_past = None
        if layer_past is not None:
            if len(layer_past) == 3:
                k_past, v_past, past_position = layer_past
                if isinstance(past_position, torch.Tensor):
                    cache_position = int(past_position.item())
                else:
                    cache_position = int(past_position)
            else:
                k_past, v_past = layer_past
                cache_position = k_past.size(1)

        # Projections
        q = self.w_q(x).view(B, S_q, self.n_head, self.d_k)                       # [B, S_q, Hq, Dk]
        kv = self.w_kv(x).view(B, S_q, 2, self.n_kv_head, self.d_k)               # [B, S_q, 2, Hkv, Dk]
        k, v = kv.unbind(dim=2)                                                   # each [B, S_q, Hkv, Dk]

        # RoPE on Q and K (K has fewer heads)
        q, k_cur = apply_rotary_pos_emb(q, k, freqs_cis=freqs_cis)                # q:[B,S,Hq,D], k_cur:[B,S,Hkv,D]

        # KV cache kept in Hkv heads
        if layer_past is not None:
            k_full = torch.cat((k_past, k_cur), dim=1)                            # [B, S_past+S_q, Hkv, D]
            v_full = torch.cat((v_past, v),     dim=1)                            # [B, S_past+S_q, Hkv, D]
        else:
            k_full = k_cur
            v_full = v

        # Sliding window truncation
        if use_cache and (self.max_kv_len is not None) and (k_full.size(1) > self.max_kv_len):
            k_full = k_full[:, -self.max_kv_len:]
            v_full = v_full[:, -self.max_kv_len:]
            if mask is not None and mask.size(1) != k_full.size(1):
                mask = mask[:, -k_full.size(1):]

        next_cache_position = cache_position + S_q
        present = (k_full, v_full, next_cache_position) if use_cache else None

        # Repeat K/V heads to match Hq for attention computation
        k_attn = repeat_kv(k_full, self.n_rep)                                    # [B, S_kv, Hq, D]
        v_attn = repeat_kv(v_full, self.n_rep)                                    # [B, S_kv, Hq, D]

        # FlashAttention requirements
        if not (q.is_cuda and k_attn.is_cuda and v_attn.is_cuda):
            raise AssertionError("FlashAttention requires CUDA tensors. Move model and inputs to GPU.")
        allowed_dtypes = (torch.float16, torch.bfloat16)

        orig_dtype = q.dtype
        if q.dtype not in allowed_dtypes:
            q      = q.to(torch.float16)
            k_attn = k_attn.to(torch.float16)
            v_attn = v_attn.to(torch.float16)

        q = q.contiguous()
        k_attn = k_attn.contiguous()
        v_attn = v_attn.contiguous()

        # --- No-padding (fused) path ---
        if mask is None:
            if k_attn.size(1) != S_q:
                raise AssertionError(
                    f"flash_attn_func expects q_len==kv_len; got q={S_q}, kv={k_attn.size(1)}. "
                    "Provide a padding mask to use varlen with cache."
                )
            context = flash_attn_func(
                q, k_attn, v_attn,
                dropout_p=self.dropout_layer.p if self.training else 0.0,
                causal=True
            )

        # --- Varlen path (with mask / cache) ---
        else:
            if mask.size(1) == S_q:
                q_mask = mask.bool()
            elif mask.size(1) > S_q:
                q_mask = mask[:, -S_q:].bool()
            else:
                q_mask = torch.ones(B, S_q, dtype=torch.bool, device=mask.device)
            m_bool = mask.bool()

            q_unpad, q_idx, cu_q, max_q, *_ = unpad_input(q,      q_mask)
            k_unpad, _,     cu_k, max_k, *_ = unpad_input(k_attn, m_bool)
            v_unpad, *_                 = unpad_input(v_attn, m_bool)
            q_idx = q_idx.to(torch.int32)

            context_unpad = flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad,
                cu_q, cu_k, max_q, max_k,
                dropout_p=self.dropout_layer.p if self.training else 0.0,
                causal=True
            )
            try:
                context = pad_input(context_unpad, q_idx, B, S_q)
            except TypeError:
                context = pad_input(context_unpad, q_idx, S_q)

        if context.dtype != orig_dtype:
            context = context.to(orig_dtype)

        out = context.contiguous().view(B, S_q, self.d_model)
        out = self.fc(out)
        return out, None, present


class PositionwiseFeedForward(nn.Module):
    """ SwiGLU FFN """
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1, multiple_of: int = 256):
        super().__init__()
        if d_ff is None:
            d_ff = int(2 / 3 * 4 * d_model)
            d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.activation(self.w1(x))
        value = self.w3(x)
        x = self.dropout(gate * value)
        return self.w2(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1, max_kv_len=None, n_kv_head=None):
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.norm_ffn = RMSNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model, n_head, dropout,
            max_kv_len=max_kv_len,
            n_kv_head=n_kv_head
        )
        self.ffn  = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn  = nn.Dropout(dropout)

    def forward(self, x, freqs_cis, mask, layer_past=None, use_cache=False):
        h = self.norm_attn(x)
        attn_out, _, present = self.attn(h, freqs_cis=freqs_cis, mask=mask,
                                         layer_past=layer_past, use_cache=use_cache)
        x = x + self.drop_attn(attn_out)
        ffn_out = self.ffn(self.norm_ffn(x))
        x = x + self.drop_ffn(ffn_out)
        return x, present


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_head: int,             # Hq
        n_layer: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        pad_idx: int = 0,
        rope_theta: float = 10000.0,
        n_kv_head: Optional[int] = None,    # NEW: Hkv (defaults to Hq if None)
        max_kv_len: Optional[int] = None,   # optionally expose sliding-window limit
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.rope_theta = rope_theta

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, n_head, d_ff,
                max_kv_len=max_kv_len,
                dropout=dropout,
                n_kv_head=n_kv_head
            )
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        precomp_len = max_seq_len * 2
        freqs = precompute_freqs_cis(self.d_k, precomp_len, theta=self.rope_theta)
        self.register_buffer("precomputed_freqs_cis", freqs, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _slice_or_extend_rope(self, start_pos: int, seq_len: int, device: torch.device) -> torch.Tensor:
        end_pos = start_pos + seq_len
        # Extend on-the-fly if needed (without mutating the buffer)
        if end_pos > self.precomputed_freqs_cis.size(0):
            freqs = precompute_freqs_cis(self.d_k, end_pos, theta=self.rope_theta).to(device)
            return freqs[start_pos:end_pos]
        return self.precomputed_freqs_cis[start_pos:end_pos].to(device)

    def forward(
        self,
        input_ids: torch.Tensor,                       # [B, S_q]
        attention_mask: Optional[torch.Tensor] = None, # [B, S_q], 1=attend, 0=pad
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]]]:

        B, S_q = input_ids.size()
        device = input_ids.device

        past_len = 0
        start_pos = 0
        if past_key_values is not None and past_key_values[0] is not None:
            layer0 = past_key_values[0]
            past_len = layer0[0].size(1)
            if len(layer0) == 3:
                cache_position = layer0[2]
                start_pos = int(cache_position.item()) if isinstance(cache_position, torch.Tensor) else int(cache_position)
            else:
                start_pos = past_len

        x = self.dropout(self.token_embedding(input_ids))

        # RoPE for *current* tokens
        freqs_cis = self._slice_or_extend_rope(start_pos, S_q, device=device)

        # Build full KV mask:
        # - If attention_mask given, concat past-ones with it (so tail aligns with current queries).
        # - If no attention_mask but past exists, concat past-ones with current-ones (so varlen has a proper q-mask tail).
        kv_mask = None
        if attention_mask is not None:
            kv_mask = attention_mask if past_len == 0 else torch.cat(
                (torch.ones(B, past_len, dtype=attention_mask.dtype, device=device), attention_mask),
                dim=1
            )
        elif past_len > 0:
            kv_mask = torch.cat(
                (
                    torch.ones(B, past_len, dtype=torch.long, device=device),
                    torch.ones(B, S_q, dtype=torch.long, device=device),
                ),
                dim=1,
            )

        presents = [] if use_cache else None
        h = x
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            h, present = layer(h, freqs_cis=freqs_cis, mask=kv_mask, layer_past=layer_past, use_cache=use_cache)
            if use_cache:
                presents.append(present)

        h = self.final_norm(h)
        logits = self.lm_head(h)
        return logits, presents

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,      # [B, S_prompt]
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        tokenizer=None,
        pad_token_id: Optional[int] = None,
        streamer=None,
    ) -> List[List[int]]:
        self.eval()
        device = next(self.parameters()).device
        B, _ = input_ids.shape

        generated_ids = input_ids.clone()
        sequences = input_ids.tolist()
        past_key_values = None
        attn_mask = (input_ids != pad_token_id).long() if pad_token_id is not None else None
        cur_ids = input_ids

        for _ in range(max_new_tokens):
            logits, past_key_values = self(
                input_ids=cur_ids,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_logits = logits[:, -1, :]  # [B, V]

            if repetition_penalty != 1.0:
                # Penalize tokens already generated in each sequence
                for b in range(B):
                    prev = torch.tensor(sequences[b], device=next_logits.device, dtype=torch.long).unsqueeze(0)
                    scores = next_logits[b:b+1].gather(1, prev)
                    scores = torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)
                    next_logits[b:b+1].scatter_(1, prev, scores)

            if temperature <= 0:
                next_token = torch.argmax(next_logits, dim=-1)
            else:
                if temperature != 1.0:
                    next_logits = next_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            if streamer is not None:
                streamer.put(next_token.unsqueeze(0))

            cur_ids = next_token.unsqueeze(1)
            attn_mask = None  # KV cache handles positions; varlen path will synthesize q-mask as needed
            generated_ids = torch.cat([generated_ids, cur_ids], dim=1)

            all_finished = True
            for b in range(B):
                tid = next_token[b].item()
                sequences[b].append(tid)
                is_eos = tokenizer and hasattr(tokenizer, "eos_token_id") and tid == tokenizer.eos_token_id
                if not is_eos:
                    all_finished = False
            if all_finished:
                break

        if streamer is not None:
            streamer.end()

        self.train()
        return sequences


def upgrade_state_dict_for_block_norms(
    state_dict: Mapping[str, torch.Tensor]
) -> Mapping[str, torch.Tensor]:
    """
    Upgrade checkpoints trained with the old single RMSNorm per block to the new
    dual-norm layout (attn + ffn). Copies the old weights into both new norms and
    drops stale bias entries. Final RMSNorm weights are preserved; its old bias is discarded.
    """
    if any(".norm_attn.weight" in k for k in state_dict.keys()):
        return state_dict

    needs_upgrade = any(
        k.startswith("layers.") and ".norm." in k
        for k in state_dict.keys()
    )
    if not needs_upgrade:
        return state_dict

    upgraded = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("layers.") and ".norm." in key:
            if key.endswith(".weight"):
                prefix = key.replace(".norm.weight", "")
                upgraded[f"{prefix}.norm_attn.weight"] = value.clone()
                upgraded[f"{prefix}.norm_ffn.weight"] = value.clone()
            # Drop old bias/extra state for the obsolete norm module
            continue
        if key.startswith("final_norm.") and key.endswith(".bias"):
            # New RMSNorm has no bias.
            continue
        upgraded[key] = value
    return upgraded
