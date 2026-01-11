"""
Continuous Thought Machine (CTM) - Faithful Implementation.

Based on the original CTM paper:
- Synapse: z^t + o^t → a^t (pre-activations)
- NLM: processes pre-activation history → z^{t+1} (post-activations)
- Sync: dot products of post-activation histories between neuron pairs
- Sync projected to attention queries/outputs
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from flash_attn import flash_attn_func
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    raise ImportError("flash_attn is required. Install with: pip install flash-attn --no-build-isolation")

from model import OptimizedRMSNorm


@dataclass
class CTMConfig:
    """Configuration for CTM."""
    vocab_size: int
    d_model: int = 512
    n_head: int = 8
    n_kv_head: Optional[int] = None
    max_seq_len: int = 2048

    # CTM-specific
    num_ticks: int = 8
    nlm_hidden: int = 64
    nlm_depth: int = 2
    sync_pairs: int = 512

    # Training
    dropout: float = 0.0
    rope_theta: float = 500000.0

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head


class SynapseModel(nn.Module):
    """
    Synapse: combines post-activations z^t and attention output o^t
    to produce pre-activations a^t.

    U-Net style MLP with skip connections.
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()

        # Input: z + o (both d_model)
        input_dim = d_model * 2

        # Encoder
        self.enc1 = nn.Linear(input_dim, d_model, bias=False)
        self.enc2 = nn.Linear(d_model, d_model // 2, bias=False)

        # Decoder with skip
        self.dec2 = nn.Linear(d_model // 2, d_model, bias=False)
        self.dec1 = nn.Linear(d_model * 2, d_model, bias=False)  # skip connection

        self.norm1 = OptimizedRMSNorm(d_model)
        self.norm2 = OptimizedRMSNorm(d_model // 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, z: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, S, D) post-activations
            o: (B, S, D) attention output
        Returns:
            a: (B, S, D) pre-activations
        """
        x = torch.cat([z, o], dim=-1)

        # Encoder
        h1 = F.gelu(self.enc1(x))
        h1 = self.norm1(h1)
        h2 = F.gelu(self.enc2(h1))
        h2 = self.norm2(h2)

        # Decoder with skip
        d2 = F.gelu(self.dec2(h2))
        d2 = self.dropout(d2)
        d1 = torch.cat([d2, h1], dim=-1)
        out = self.dec1(d1)

        return out


class NeuronLevelModel(nn.Module):
    """
    NLM: Each neuron independently processes its pre-activation history
    to produce the next post-activation.

    D independent MLPs, efficiently batched.
    Input: (B, S, T, D) - history of pre-activations per neuron
    Output: (B, S, D) - post-activations
    """

    def __init__(self, d_model: int, nlm_hidden: int, nlm_depth: int, max_ticks: int):
        super().__init__()
        self.d_model = d_model
        self.nlm_hidden = nlm_hidden
        self.max_ticks = max_ticks

        # Per-neuron input projection: (D, max_ticks, nlm_hidden)
        self.w_in = nn.Parameter(torch.empty(d_model, max_ticks, nlm_hidden))
        self.b_in = nn.Parameter(torch.zeros(d_model, nlm_hidden))

        # Hidden layers
        self.w_hidden = nn.ParameterList()
        self.b_hidden = nn.ParameterList()
        for _ in range(nlm_depth - 2):
            self.w_hidden.append(nn.Parameter(torch.empty(d_model, nlm_hidden, nlm_hidden)))
            self.b_hidden.append(nn.Parameter(torch.zeros(d_model, nlm_hidden)))

        # Output projection: (D, nlm_hidden, 1)
        self.w_out = nn.Parameter(torch.empty(d_model, nlm_hidden, 1))
        self.b_out = nn.Parameter(torch.zeros(d_model, 1))

        self._init_weights()

    def _init_weights(self):
        """Initialize with diversity across neurons."""
        std = 0.02
        # Different scales per neuron for diverse dynamics
        scales = torch.exp(torch.linspace(-0.7, 0.7, self.d_model))
        scales = scales[torch.randperm(self.d_model)]

        nn.init.normal_(self.w_in, std=std)
        nn.init.normal_(self.w_out, std=std)
        with torch.no_grad():
            self.w_in.mul_(scales.view(-1, 1, 1))
            self.w_out.mul_(scales.view(-1, 1, 1))

        for w in self.w_hidden:
            nn.init.normal_(w, std=std)
            with torch.no_grad():
                w.mul_(scales.view(-1, 1, 1))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (B, S, T, D) pre-activation history
        Returns:
            z: (B, S, D) post-activations
        """
        B, S, T, D = history.shape

        # Pad to max_ticks if needed
        if T < self.max_ticks:
            pad = torch.zeros(B, S, self.max_ticks - T, D,
                            device=history.device, dtype=history.dtype)
            history = torch.cat([pad, history], dim=2)
        else:
            history = history[:, :, -self.max_ticks:]

        # (B, S, D, max_ticks)
        h = history.to(self.w_in.dtype).transpose(-1, -2)

        # Input projection
        h = torch.einsum('bsdt,dth->bsdh', h, self.w_in) + self.b_in
        h = F.gelu(h)

        # Hidden layers
        for w, b in zip(self.w_hidden, self.b_hidden):
            h = torch.einsum('bsdh,dhk->bsdk', h, w) + b
            h = F.gelu(h)

        # Output
        out = torch.einsum('bsdh,dho->bsdo', h, self.w_out) + self.b_out
        return out.squeeze(-1)


class SynchronizationModule(nn.Module):
    """
    Sync: Computes synchronization as dot products of post-activation
    histories between neuron pairs.

    S_{ij} = sum_t z_i^t * z_j^t (with decay weighting)
    """

    def __init__(self, d_model: int, sync_pairs: int):
        super().__init__()
        self.d_model = d_model
        self.sync_pairs = sync_pairs

        # Sample fixed neuron pairs
        idx1 = torch.randint(0, d_model, (sync_pairs,))
        idx2 = torch.randint(0, d_model, (sync_pairs,))
        self.register_buffer('idx1', idx1)
        self.register_buffer('idx2', idx2)

        # Learnable decay
        self.decay_raw = nn.Parameter(torch.zeros(sync_pairs))

    @property
    def decay(self):
        return F.softplus(self.decay_raw)

    def forward(self, z_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_history: (B, S, T, D) post-activation history
        Returns:
            sync: (B, S, sync_pairs)
        """
        B, S, T, D = z_history.shape
        dtype = self.decay_raw.dtype
        h = z_history.to(dtype)

        # Extract neuron pairs
        h1 = h[..., self.idx1]  # (B, S, T, P)
        h2 = h[..., self.idx2]  # (B, S, T, P)

        # Decay weights
        t_idx = torch.arange(T, device=h.device, dtype=dtype)
        weights = torch.exp(-self.decay.unsqueeze(0) * (T - 1 - t_idx).unsqueeze(1))
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)

        # Weighted dot product over time
        sync = (h1 * h2 * weights.unsqueeze(0).unsqueeze(0)).sum(dim=2)

        return sync


class CTMAttention(nn.Module):
    """
    Attention module where sync is projected to form the query.
    Queries the static input KV.
    """

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.d_model // config.n_head
        self.sync_pairs = config.sync_pairs

        # Sync → Query projection
        self.sync_to_q = nn.Linear(config.sync_pairs, config.n_head * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.d_model, bias=False)

        self.dropout = config.dropout

    def forward(
        self,
        sync: torch.Tensor,      # (B, S, sync_pairs)
        static_k: torch.Tensor,  # (B, S, n_kv_head, head_dim)
        static_v: torch.Tensor,  # (B, S, n_kv_head, head_dim)
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sync is projected to query, which attends to static KV.
        Returns attention output o.
        """
        B, S, _ = sync.shape

        # Project sync to query
        q = self.sync_to_q(sync).view(B, S, self.n_head, self.head_dim)

        # Ensure compatible dtype
        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        q = q.to(target_dtype)
        k = static_k.to(target_dtype)
        v = static_v.to(target_dtype)

        # Apply RoPE to query
        rotary_dim = cos.shape[-1] * 2
        if rotary_dim > self.head_dim:
            cos = cos[..., :self.head_dim // 2]
            sin = sin[..., :self.head_dim // 2]
        cos = cos.to(target_dtype)
        sin = sin.to(target_dtype)

        q = apply_rotary_emb(q, cos, sin, interleaved=False)

        # Flash attention
        attn_dropout = self.dropout if self.training else 0.0
        out = flash_attn_func(q, k, v, dropout_p=attn_dropout, causal=True)

        # Project output
        out = out.reshape(B, S, self.n_head * self.head_dim)
        out = out.to(self.o_proj.weight.dtype)

        return self.o_proj(out)


class InputEncoder(nn.Module):
    """
    Encodes input tokens to produce static KV for CTM attention.
    """

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.d_model // config.n_head

        # Simple projection to KV (could add transformer layers here)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_head * self.head_dim, bias=False)
        self.norm = OptimizedRMSNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            encoded: (B, S, D)
            static_k: (B, S, n_kv_head, head_dim) with RoPE
            static_v: (B, S, n_kv_head, head_dim)
        """
        B, S, D = x.shape
        x = self.norm(x)

        k = self.k_proj(x).view(B, S, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_head, self.head_dim)

        # Apply RoPE to keys
        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        k = k.to(target_dtype)

        rotary_dim = cos.shape[-1] * 2
        if rotary_dim > self.head_dim:
            cos = cos[..., :self.head_dim // 2]
            sin = sin[..., :self.head_dim // 2]
        cos = cos.to(target_dtype)
        sin = sin.to(target_dtype)

        k = apply_rotary_emb(k, cos, sin, interleaved=False)

        return x, k, v.to(target_dtype)


class CTMCore(nn.Module):
    """
    The CTM iterative core - faithful to original paper.

    Per tick t:
        1. Synapse(z^t, o^t) → a^t (pre-activations)
        2. NLM(a_history) → z^{t+1} (post-activations)
        3. Sync(z_history) → synchronization matrix
        4. Attention(sync → query, static_KV) → o^{t+1}
    """

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.num_ticks = config.num_ticks
        self.d_model = config.d_model

        # Synapse: z + o → a
        self.synapse = SynapseModel(config.d_model, config.dropout)

        # NLM: a_history → z
        self.nlm = NeuronLevelModel(
            config.d_model,
            config.nlm_hidden,
            config.nlm_depth,
            config.num_ticks
        )

        # Sync: z_history → sync matrix
        self.sync = SynchronizationModule(config.d_model, config.sync_pairs)

        # Attention: sync → query → attend to data
        self.attention = CTMAttention(config)

        # Normalization
        self.z_norm = OptimizedRMSNorm(config.d_model)

        # Tick embeddings for temporal identity
        self.tick_embed = nn.Embedding(config.num_ticks, config.d_model)

    def forward(
        self,
        initial_z: torch.Tensor,   # (B, S, D) initial post-activations
        static_k: torch.Tensor,    # (B, S, n_kv_head, head_dim)
        static_v: torch.Tensor,    # (B, S, n_kv_head, head_dim)
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_ticks: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Run CTM for num_ticks iterations.

        Returns:
            final_z: (B, S, D) final post-activations
            final_sync: (B, S, sync_pairs) final synchronization
            all_syncs: List of sync at each tick
        """
        B, S, D = initial_z.shape
        num_ticks = num_ticks or self.num_ticks
        device = initial_z.device
        dtype = initial_z.dtype

        # Initialize
        z = initial_z  # post-activations
        o = torch.zeros_like(z)  # attention output starts at zero

        a_history: List[torch.Tensor] = []  # pre-activation history
        z_history: List[torch.Tensor] = [z]  # post-activation history (includes initial)
        all_syncs: List[torch.Tensor] = []

        for tick in range(num_ticks):
            # Get tick embedding
            tick_idx = torch.tensor(tick, device=device)
            tick_emb = self.tick_embed(tick_idx)

            # 1. Synapse: z^t + o^t → a^t
            a = self.synapse(z, o)
            a = a + tick_emb  # add tick identity
            a_history.append(a)

            # 2. NLM: process pre-activation history → z^{t+1}
            a_hist_tensor = torch.stack(a_history, dim=2)  # (B, S, T, D)
            z = self.nlm(a_hist_tensor)
            z = self.z_norm(z)
            z_history.append(z)

            # 3. Sync: compute from post-activation history
            z_hist_tensor = torch.stack(z_history, dim=2)  # (B, S, T+1, D)
            sync_out = self.sync(z_hist_tensor)
            all_syncs.append(sync_out)

            # 4. Attention: sync → query → o^{t+1}
            o = self.attention(sync_out, static_k, static_v, cos, sin)

        return z, all_syncs[-1], all_syncs


class CTMLanguageModel(nn.Module):
    """
    Complete CTM for language modeling.

    Architecture:
        1. Token embedding
        2. Input encoder (produces static KV)
        3. CTM iterative core
        4. Output from sync (per CTM paper, sync IS the representation)
    """

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Input encoder
        self.input_encoder = InputEncoder(config)

        # CTM core
        self.ctm_core = CTMCore(config)

        # Output: sync → logits
        self.sync_norm = OptimizedRMSNorm(config.sync_pairs)
        self.output_head = nn.Linear(config.sync_pairs, config.vocab_size, bias=False)

        # RoPE
        self.head_dim = config.d_model // config.n_head
        self._init_rope()
        self._init_weights()

    def _init_rope(self):
        theta = self.config.rope_theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(self.config.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos().to(torch.bfloat16), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().to(torch.bfloat16), persistent=False)

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        return_all_ticks: bool = False,
        num_ticks: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            input_ids: (B, S) token IDs
            return_all_ticks: if True, return logits for all ticks

        Returns:
            logits: (B, S, V)
            all_logits: List of logits per tick (if return_all_ticks)
        """
        B, S = input_ids.shape

        # RoPE
        if input_pos is None:
            input_pos = torch.arange(S, device=input_ids.device)
        cos = self.cos_cached[input_pos]
        sin = self.sin_cached[input_pos]

        # Embed tokens
        x = self.token_embedding(input_ids)

        # Encode to static KV
        encoded, static_k, static_v = self.input_encoder(x, cos, sin)

        # Run CTM (initial z = encoded input)
        final_z, final_sync, all_syncs = self.ctm_core(
            encoded, static_k, static_v, cos, sin, num_ticks
        )

        # Output from sync
        if return_all_ticks:
            all_logits = [self.output_head(self.sync_norm(s)) for s in all_syncs]
            return all_logits[-1], all_logits
        else:
            logits = self.output_head(self.sync_norm(final_sync))
            return logits, None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        num_ticks: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple autoregressive generation."""
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_seq_len:]
            logits, _ = self(context, num_ticks=num_ticks)
            next_logits = logits[:, -1, :]

            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                if top_k is not None:
                    v, _ = torch.topk(probs, top_k)
                    probs[probs < v[:, [-1]]] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

        return generated
