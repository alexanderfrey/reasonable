import math
from typing import Optional, Tuple, List

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


# --- Core blocks ---

class MultiHeadAttention(nn.Module):
    """
    MHA that *always* uses FlashAttention (v1) with RoPE and optional varlen path for padding.
    Expects:
      - x: [B, S_q, D_model]
      - mask: key padding mask over *full KV* (past + current), shape [B, S_kv], 1=attend, 0=pad (or None)
      - layer_past: optional (k,v) with shapes [B, S_past, H, Dk]
    Returns:
      - output: [B, S_q, D_model]
      - attn_weights: None (FA doesn't return weights)
      - present: (k_full, v_full) if use_cache else None
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,                # [S_q, Dk/2] (complex) or your real-ROPE tables
        mask: Optional[torch.Tensor] = None,    # [B, S_kv] (1=attend, 0=pad)
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, None, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        B, S_q, _ = x.size()

        q = self.w_q(x).view(B, S_q, self.n_head, self.d_k)
        k = self.w_k(x).view(B, S_q, self.n_head, self.d_k)
        v = self.w_v(x).view(B, S_q, self.n_head, self.d_k)

        q, k_cur = apply_rotary_pos_emb(q, k, freqs_cis=freqs_cis)

        if layer_past is not None:
            k_full = torch.cat((layer_past[0], k_cur), dim=1)    # [B, S_past+S_q, H, Dk]
            v_full = torch.cat((layer_past[1], v), dim=1)        # [B, S_past+S_q, H, Dk]
        else:
            k_full = k_cur
            v_full = v

        present = (k_full, v_full) if use_cache else None

        # FlashAttention requires CUDA + fp16/bf16
        if not (q.is_cuda and k_full.is_cuda and v_full.is_cuda):
            raise AssertionError("FlashAttention requires CUDA tensors. Move model and inputs to GPU.")
        allowed_dtypes = (torch.float16, torch.bfloat16)

        orig_dtype = q.dtype
        q = q.contiguous()
        k_full = k_full.contiguous()
        v_full = v_full.contiguous()

        # -------- No-padding (fused) path --------
        if mask is None:
            # q attends over full k_full/v_full causally; lengths must match (i.e., no cache).
            if k_full.size(1) != S_q:
                raise AssertionError(
                    f"flash_attn_func expects q_len==kv_len; got q={S_q}, kv={k_full.size(1)}. "
                    "Provide a padding mask to use varlen with cache."
                )
            if q.dtype not in allowed_dtypes:
                q = q.to(torch.float16)
                k_full = k_full.to(torch.float16)
                v_full = v_full.to(torch.float16)

            context = flash_attn_func(
                q, k_full, v_full,
                dropout_p=self.dropout_layer.p if self.training else 0.0,
                causal=True
            )

        # -------- Varlen path (with mask / cache) --------
        else:
            # Query mask for just the current S_q tokens at tail of KV
            if mask.size(1) == S_q:
                q_mask = mask.bool()
            elif mask.size(1) > S_q:
                q_mask = mask[:, -S_q:].bool()
            else:
                q_mask = torch.ones(B, S_q, dtype=torch.bool, device=mask.device)

            m_bool = mask.bool()

            if q.dtype not in allowed_dtypes:
                q = q.to(torch.float16)
                k_full = k_full.to(torch.float16)
                v_full = v_full.to(torch.float16)

            # Newer FA returns 5 values; older returns 4 → use starred unpack
            q_unpad, q_idx, cu_q, max_q, *_ = unpad_input(q, q_mask)
            k_unpad, _,      cu_k, max_k, *_ = unpad_input(k_full, m_bool)
            v_unpad, *_ = unpad_input(v_full, m_bool)

            # Some FA builds want int32 indices
            q_idx = q_idx.to(torch.int32)

            context_unpad = flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad,
                cu_q, cu_k, max_q, max_k,
                dropout_p=self.dropout_layer.p if self.training else 0.0,
                causal=True
            )

            # pad_input signature differs by version:
            # - Newer: pad_input(tensor, indices, B, S)
            # - Older: pad_input(tensor, indices, S)
            try:
                context = pad_input(context_unpad, q_idx, B, S_q)   # [B, S_q, H, Dk]
            except TypeError:
                context = pad_input(context_unpad, q_idx, S_q)      # [B, S_q, H, Dk]

        # Cast back to original dtype for the output projection
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
    """ Pre-LN block with FA+RoPE """
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        residual = x
        x = self.norm1(x)
        attn_out, _, present = self.attn(x, freqs_cis=freqs_cis, mask=mask, layer_past=layer_past, use_cache=use_cache)
        x = residual + self.dropout1(attn_out)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.ffn(x))
        return x, present


class GPT(nn.Module):
    """ GPT with FlashAttention + RoPE + KV caching (weights tied) """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_head: int,
        n_layer: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        pad_idx: int = 0,
        rope_theta: float = 10000.0,
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
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.token_embedding.weight

        # Precompute a safe baseline of RoPE freqs and register as buffer
        precomp_len = max_seq_len * 2
        freqs = precompute_freqs_cis(self.d_k, precomp_len, theta=self.rope_theta)
        self.register_buffer("precomputed_freqs_cis", freqs, persistent=False)  # non-persistent to avoid bloat

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
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:

        B, S_q = input_ids.size()
        device = input_ids.device

        past_len = past_key_values[0][0].size(1) if (past_key_values is not None and past_key_values[0] is not None) else 0
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