"""
Full GPT Model Implementation with FlashAttention-2, RoPE, and GQA.
Contains critical fixes for generation masking, float32 precision, and performance.
"""

import math
from collections import OrderedDict
from typing import Optional, Tuple, List, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# --- FlashAttention Imports & Guard ---
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("WARNING: flash_attn not installed. This model requires CUDA and FlashAttention to run.")


# --- RoPE Helpers ---

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute RoPE freqs. 
    Calculation is done in float32 to prevent precision issues in long contexts.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # torch.polar requires inputs to be the same dtype
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_pos_emb(
    xq: torch.Tensor, 
    xk: torch.Tensor, 
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE. Performs complex multiplication in float32.
    """
    # Reshape freqs for broadcasting: [S, D/2] -> [1, S, 1, D/2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    # View as complex, ensure float32 for stability
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Apply rotation
    xq_rot = torch.view_as_real(xq_c * freqs_cis).flatten(3)
    xk_rot = torch.view_as_real(xk_c * freqs_cis).flatten(3)
    
    # Cast back to original dtype
    return xq_rot.type_as(xq), xk_rot.type_as(xk)


# --- Components ---

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        # Calculate norm in float32
        x_f = x.float()
        norm = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x_f * norm).to(x.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat K/V heads to match Q heads (GQA)."""
    if n_rep == 1:
        return x
    B, S, H, D = x.shape
    return x.unsqueeze(3).expand(B, S, H, n_rep, D).reshape(B, S, H * n_rep, D)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.1,
        max_kv_len: Optional[int] = None,
        n_kv_head: Optional[int] = None
    ):
        super().__init__()
        Hq = n_head
        Hkv = n_kv_head if n_kv_head is not None else Hq
        
        self.n_head = Hq
        self.n_kv_head = Hkv
        self.d_k = d_model // Hq
        self.n_rep = Hq // Hkv
        self.max_kv_len = max_kv_len
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, Hq * self.d_k, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * Hkv * self.d_k, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, None, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        B, S_q, _ = x.size()

        # 1. Projections
        q = self.w_q(x).view(B, S_q, self.n_head, self.d_k)
        kv = self.w_kv(x).view(B, S_q, 2, self.n_kv_head, self.d_k)
        k, v = kv.unbind(dim=2)

        # 2. RoPE
        q, k = apply_rotary_pos_emb(q, k, freqs_cis=freqs_cis)

        # 3. KV Cache Management
        if layer_past is not None:
            k_past, v_past = layer_past
            k = torch.cat((k_past, k), dim=1)
            v = torch.cat((v_past, v), dim=1)
        
        # Sliding window (simple truncation)
        if use_cache and (self.max_kv_len is not None) and (k.size(1) > self.max_kv_len):
            k = k[:, -self.max_kv_len:]
            v = v[:, -self.max_kv_len:]
            # Note: If strict mask alignment is needed for sliding window training, 
            # the mask should be sliced by the caller.

        present = (k, v) if use_cache else None

        # 4. GQA Repeat
        k_attn = repeat_kv(k, self.n_rep)
        v_attn = repeat_kv(v, self.n_rep)

        # 5. FlashAttention
        # Inputs must be contiguous
        q = q.contiguous()
        k_attn = k_attn.contiguous()
        v_attn = v_attn.contiguous()

        # Cast to supported dtypes for Flash
        orig_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k_attn = k_attn.to(torch.bfloat16)
            v_attn = v_attn.to(torch.bfloat16)

        # Selection: Fused (Fast) vs Varlen (Masked)
        if mask is None:
            # Fast path: Standard FlashAttention
            flash_causal = True
            if layer_past is not None and S_q == 1 and k_attn.size(1) > 1:
                # Decoding a single token against a longer KV cache: allow full attention
                flash_causal = False

            context = flash_attn_func(
                q, k_attn, v_attn,
                dropout_p=self.dropout_layer.p if self.training else 0.0,
                causal=flash_causal
            )
        else:
            # Slow path: Varlen for padded batches
            # Construct Q mask
            varlen_causal = True
            if mask.size(1) == S_q:
                q_mask = mask.bool()
            else:
                # Decoding: Q spans only new tokens, let it see the entire history
                q_mask = torch.ones(B, S_q, dtype=torch.bool, device=mask.device)
                varlen_causal = False
            
            k_mask = mask.bool()

            # Unpad
            q_unpad, q_idx, cu_q, max_q, _ = unpad_input(q, q_mask)
            k_unpad, _,     cu_k, max_k, _ = unpad_input(k_attn, k_mask)
            v_unpad, *_                 = unpad_input(v_attn, k_mask)

            if q_unpad.numel() == 0:
                context = torch.zeros_like(q)
            else:
                context_unpad = flash_attn_varlen_func(
                    q_unpad, k_unpad, v_unpad,
                    cu_q.int(), cu_k.int(), max_q, max_k,
                    dropout_p=self.dropout_layer.p if self.training else 0.0,
                    causal=varlen_causal
                )
                context = pad_input(context_unpad, q_idx, B, S_q)

        # Restore dtype
        if context.dtype != orig_dtype:
            context = context.to(orig_dtype)

        # Output projection
        out = self.fc(context.contiguous().view(B, S_q, self.d_model))
        out = self.dropout_layer(out)
        return out, None, present


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            # SwiGLU default sizing
            d_ff = int(2 / 3 * 4 * d_model)
            d_ff = 256 * ((d_ff + 256 - 1) // 256)
            
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1, max_kv_len=None, n_kv_head=None):
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.norm_ffn = RMSNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model, n_head, dropout, max_kv_len, n_kv_head
        )
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, freqs_cis, mask, layer_past=None, use_cache=False):
        h = self.norm_attn(x)
        attn_out, _, present = self.attn(h, freqs_cis, mask, layer_past, use_cache)
        x = x + attn_out
        
        h = self.norm_ffn(x)
        ffn_out = self.ffn(h)
        x = x + ffn_out
        return x, present


# --- Main GPT Model ---

class GPT(nn.Module):
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
        n_kv_head: Optional[int] = None,
        max_kv_len: Optional[int] = None,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.rope_theta = rope_theta
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, n_head, d_ff,
                dropout=dropout,
                max_kv_len=max_kv_len,
                n_kv_head=n_kv_head
            )
            for _ in range(n_layer)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(dropout)

        # Precompute RoPE frequencies
        precomp_len = max_seq_len * 2 
        freqs = precompute_freqs_cis(self.d_k, precomp_len, theta=rope_theta)
        self.register_buffer("precomputed_freqs_cis", freqs, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        
        # Re-scaling of residual projections (GPT-2/Llama style)
        for name, p in module.named_parameters():
            if name.endswith("fc.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * len(self.layers)))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, # [B, S]
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:

        B, S_q = input_ids.size()
        device = input_ids.device

        # 1. RoPE Slicing
        start_pos = 0
        if past_key_values is not None and past_key_values[0] is not None:
            start_pos = past_key_values[0][0].size(1)

        end_pos = start_pos + S_q
        if end_pos > self.precomputed_freqs_cis.size(0):
            freqs = precompute_freqs_cis(self.d_k, end_pos + 1024, self.rope_theta).to(device)
            self.register_buffer("precomputed_freqs_cis", freqs, persistent=False)
            
        freqs_cis = self.precomputed_freqs_cis[start_pos:end_pos].to(device)

        # 2. Mask Logic
        kv_mask = None
        if attention_mask is not None:
            mask_len = attention_mask.size(1)
            expected_len = start_pos + S_q

            if mask_len == expected_len:
                # HuggingFace-style: mask already covers KV cache + current tokens
                kv_mask = attention_mask
            elif start_pos > 0 and mask_len == start_pos:
                # Legacy path: mask only contains cached positions, append new tokens
                curr_mask = torch.ones(B, S_q, dtype=attention_mask.dtype, device=device)
                kv_mask = torch.cat([attention_mask, curr_mask], dim=1)
            else:
                raise ValueError(
                    f"attention_mask has invalid shape {mask_len}, expected {expected_len} "
                    "or cached length only."
                )

            # Optimization: Use fast fused kernel if mask is full
            if torch.all(kv_mask):
                kv_mask = None

        # 3. Forward Pass
        x = self.dropout(self.token_embedding(input_ids))
        presents = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values else None
            
            if self.use_gradient_checkpointing and self.training and not use_cache:
                # Checkpoint wrapper
                x = checkpoint(
                    layer, x, freqs_cis, kv_mask, None, False, 
                    use_reentrant=False
                )[0] # unpack tuple
                present = None
            else:
                x, present = layer(x, freqs_cis, kv_mask, layer_past, use_cache)
            
            if use_cache:
                presents.append(present)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, presents

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        pad_token_id: Optional[int] = 0,
        eos_token_id: Optional[int] = None,
        streamer=None,
    ) -> List[List[int]]:
        self.eval()
        B, _ = input_ids.shape
        device = input_ids.device

        # Create initial attention mask (1 for real, 0 for pad)
        if pad_token_id is not None:
            attn_mask = (input_ids != pad_token_id).long()
        else:
            attn_mask = torch.ones_like(input_ids)

        generated_ids = input_ids.clone()
        past_key_values = None
        curr_ids = input_ids
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits, past_key_values = self(
                input_ids=curr_ids,
                attention_mask=attn_mask, # Pass accumulated mask
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_logits = logits[:, -1, :]

            # Repetition Penalty (GPU-based)
            if repetition_penalty != 1.0:
                score = torch.gather(next_logits, 1, generated_ids)
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                next_logits.scatter_(1, generated_ids, score)

            # Sampling
            if temperature <= 0:
                next_token = torch.argmax(next_logits, dim=-1)
            else:
                next_logits = next_logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float('inf')
                elif top_p is not None and 0.0 < top_p < 1.0:
                    # Nucleus sampling: keep smallest set whose cumulative prob >= top_p
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    cutoff_mask = cumulative_probs > top_p
                    # Shift mask right to always keep at least one token
                    cutoff_mask[..., 1:] = cutoff_mask[..., :-1].clone()
                    cutoff_mask[..., 0] = False
                    sorted_logits = torch.where(cutoff_mask, torch.full_like(sorted_logits, -float("inf")), sorted_logits)
                    next_logits = torch.empty_like(next_logits).scatter(1, sorted_indices, sorted_logits)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            if streamer:
                streamer.put(next_token.unsqueeze(0))

            # Update tracking
            curr_ids = next_token.unsqueeze(1)
            generated_ids = torch.cat([generated_ids, curr_ids], dim=1)
            
            # IMPORTANT: Extend the mask for the next step
            attn_mask = torch.cat([attn_mask, torch.ones(B, 1, dtype=torch.long, device=device)], dim=1)

            if eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)
                if finished.all():
                    break

        if streamer:
            streamer.end()
            
        self.train()
        return generated_ids.tolist()


# --- Utilities ---

def upgrade_state_dict_for_block_norms(state_dict: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    """
    Utility to upgrade old checkpoints (single norm) to new dual-norm layout.
    """
    if any(".norm_attn.weight" in k for k in state_dict.keys()):
        return state_dict

    upgraded = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("layers.") and ".norm." in key:
            if key.endswith(".weight"):
                prefix = key.replace(".norm.weight", "")
                upgraded[f"{prefix}.norm_attn.weight"] = value.clone()
                upgraded[f"{prefix}.norm_ffn.weight"] = value.clone()
            continue
        if key.startswith("final_norm.") and key.endswith(".bias"):
            continue
        upgraded[key] = value
    return upgraded


if __name__ == "__main__":
    # Simple Sanity Check
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
    else:
        print("Running sanity check on CUDA...")
        model = GPT(
            vocab_size=1000, d_model=256, n_head=4, n_layer=2, d_ff=512, 
            max_seq_len=128, n_kv_head=2 # Test GQA
        ).cuda()
        
        x = torch.randint(0, 1000, (2, 10)).cuda()
        
        # Forward
        logits, _ = model(x)
        print(f"Forward pass output: {logits.shape}") # [2, 10, 1000]
        
        # Generate
        out = model.generate(x, max_new_tokens=5)
        print(f"Generated length: {len(out[0])}") # Should be 15
        print("Test passed.")
