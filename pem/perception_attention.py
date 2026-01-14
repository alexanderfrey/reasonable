"""
Perception Attention Module for PEM.

Enables selective attention to perception (Qwen) based on personality,
intention, and neural synchronization patterns.

Core idea:
- Qwen provides K, V (rich perceptual features, cached once)
- Query is built by SYNCHRONIZING oscillations across neurons
- Each tick refines what to attend to based on current mental state

Architecture:
    Qwen → KV Cache (static)
              ↓
    [Personality, Intention, Sync] → Query Builder → Q
              ↓
    CrossAttention(Q, K, V) → observation
              ↓
    Synapse(state, observation) → updated state
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptionOutput(NamedTuple):
    """Output from PerceptionAttention forward pass."""
    observation: torch.Tensor      # (B, S, D) attended perception
    attention_weights: torch.Tensor  # (B, H, S, S_kv) attention pattern


@dataclass
class PerceptionConfig:
    """Configuration for perception attention module."""

    # Dimensions
    d_model: int = 512              # Internal model dimension
    d_perception: int = 1536        # Qwen hidden dimension
    n_heads: int = 8                # Number of attention heads
    sync_pairs: int = 512           # Sync dimension (from SyncModule)

    # Query building
    num_oscillators: int = 32       # Oscillators for query building
    min_period: int = 1             # Min oscillation period (ticks)
    max_period: int = 64            # Max oscillation period (ticks)

    # Attention
    dropout: float = 0.0
    use_flash_attention: bool = True  # Use flash attention if available

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class PerceptionKVCache(nn.Module):
    """
    Caches Key-Value projections from perception (Qwen) AND imagination.

    Both real perception and imagination go through the SAME K, V projections,
    ensuring they compete in the same attention space. The attention mechanism
    naturally learns when to attend to real vs imagined content based on
    personality, intention, and context signals.

    Architecture:
        Real Perception (d_perception)     Imagination (d_model)
                    │                              │
                    ▼                              │
          perception_to_unified                   │
            (d_perception → d_model)              │
                    │                              │
                    └──────────┬───────────────────┘
                               │
                               ▼
                       Unified d_model space
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
               k_proj (shared)     v_proj (shared)
                     │                   │
                     ▼                   ▼
                  K_unified           V_unified
    """

    def __init__(
        self,
        d_perception: int,  # Qwen output dim
        d_model: int,       # Internal dim
        n_heads: int,
    ):
        super().__init__()
        self.d_perception = d_perception
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # === Perception path: d_perception → d_model ===
        self.perception_norm = RMSNorm(d_perception)
        self.perception_to_unified = nn.Linear(d_perception, d_model, bias=False)

        # === Imagination path: already d_model, just normalize ===
        self.imagination_norm = RMSNorm(d_model)

        # === SHARED K, V projections (both sources use these) ===
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.perception_to_unified.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)

    def _reshape_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reshape K, V for multi-head attention."""
        B, S, _ = k.shape
        k = k.view(B, S, self.n_heads, self.head_dim)
        v = v.view(B, S, self.n_heads, self.head_dim)
        return k, v

    def forward(
        self,
        perception_features: torch.Tensor,  # (B, S, d_perception) from Qwen
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project perception features to K, V for caching.
        Alias for project_perception() for backward compatibility.
        """
        return self.project_perception(perception_features)

    def project_perception(
        self,
        perception_features: torch.Tensor,  # (B, S, d_perception) from Qwen
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project real perception features to K, V.

        Args:
            perception_features: Hidden states from Qwen

        Returns:
            k: (B, S, n_heads, head_dim) keys
            v: (B, S, n_heads, head_dim) values
        """
        # Normalize in perception space
        x = self.perception_norm(perception_features)

        # Project to unified d_model space
        x = self.perception_to_unified(x)  # (B, S, d_model)

        # Project through SHARED K, V
        k = self.k_proj(x)  # (B, S, d_model)
        v = self.v_proj(x)  # (B, S, d_model)

        return self._reshape_kv(k, v)

    def project_imagination(
        self,
        imagined_features: torch.Tensor,  # (B, S, d_model) from ImaginationModule
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project imagined features to K, V using SAME shared projection.

        This ensures imagination competes in the same attention space as
        real perception. The query (shaped by personality, intention, etc.)
        decides what to attend to.

        Args:
            imagined_features: Features from ImaginationModule

        Returns:
            k: (B, S, n_heads, head_dim) keys
            v: (B, S, n_heads, head_dim) values
        """
        # Normalize in d_model space (imagination is already d_model)
        x = self.imagination_norm(imagined_features)

        # Project through SAME SHARED K, V as perception
        k = self.k_proj(x)  # Same weights as perception!
        v = self.v_proj(x)  # Same weights as perception!

        return self._reshape_kv(k, v)


class OscillationQueryBuilder(nn.Module):
    """
    Builds attention query by synchronizing oscillations across neurons.

    The query emerges from the coherent pattern of:
    - Personality: WHAT to look for (stable embedding)
    - Intention: HOW MUCH to pursue it (oscillating intensity)
    - Sync: Coordination patterns across neurons
    - Surprise: WHAT was unexpected (closes the experience loop)

    Each neuron contributes an oscillation; the query is the
    synchronized combination of these oscillations.

    The key insight for closing the experience loop:
    Surprise should influence WHERE we look next. If something surprised us,
    we should attend to it more. This creates the feedback:
        Perception → Prediction → Surprise → Attention → Perception
    """

    def __init__(
        self,
        d_model: int,
        sync_pairs: int,
        num_oscillators: int = 32,
        min_period: int = 1,
        max_period: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.sync_pairs = sync_pairs
        self.num_oscillators = num_oscillators

        # === 1. Query Oscillator Bank ===
        # Each oscillator has learned frequency, amplitude, phase
        # These oscillators modulate the query building process
        init_periods = torch.logspace(
            math.log10(max(min_period, 1)),
            math.log10(max_period),
            num_oscillators
        )
        init_frequencies = 1.0 / init_periods

        self.frequencies = nn.Parameter(init_frequencies)
        self.amplitudes = nn.Parameter(torch.ones(num_oscillators) / num_oscillators)
        self.phases = nn.Parameter(torch.rand(num_oscillators) * 2 * math.pi)

        # === 2. Personality → Query Bias ===
        # Personality embedding defines the base "what to look for"
        # This is a learned direction in query space
        self.personality_to_query = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # === 3. Intention → Query Magnitude ===
        # Intention modulates how strongly to pursue personality-defined query
        # Higher intention = more focused/sharper attention
        self.intention_to_magnitude = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # === 4. Sync → Query Modulation ===
        # Sync patterns adjust the query direction based on current neural state
        # "Given how neurons are coordinating, where should I look?"
        self.sync_to_query = nn.Sequential(
            nn.Linear(sync_pairs, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # === 5. Oscillation Integration ===
        # Combines oscillation state with other signals
        # Use larger hidden dim to preserve oscillation signal
        self.oscillation_integration = nn.Sequential(
            nn.Linear(num_oscillators, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Scale factor for oscillation contribution
        self.oscillation_scale = nn.Parameter(torch.tensor(1.0))

        # === 6. Surprise → Query (closes the experience loop) ===
        # Surprise direction tells us WHAT was unexpected
        # Surprise magnitude tells us HOW unexpected it was
        # Together they steer attention toward surprising things
        self.surprise_direction_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Surprise magnitude modulates query strength
        # Higher surprise = stronger/more focused attention
        self.surprise_magnitude_scale = nn.Parameter(torch.tensor(0.5))

        # === 7. Valence → Query Focus (affective modulation) ===
        # Positive valence → sharper/more focused query (exploit)
        # Negative valence → broader/more diffuse query (explore)
        self.valence_focus_scale = nn.Parameter(torch.tensor(0.3))

        # === 8. Curiosity → Query Exploration (epistemic drive) ===
        # Curiosity drives exploration of uncertain/novel regions
        # exploration_bonus from CuriosityModule adds to query
        self.curiosity_scale = nn.Parameter(torch.tensor(0.5))

        # === 9. Final Query Projection ===
        # Combines all signals into final query
        # Input: personality_query + intention_scaled + sync_modulation + oscillation + surprise + curiosity
        self.query_combiner = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 2),  # Now 6 inputs (added curiosity)
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.norm = RMSNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for module in [
            self.personality_to_query,
            self.intention_to_magnitude,
            self.sync_to_query,
            self.oscillation_integration,
            self.surprise_direction_proj,
            self.query_combiner,
        ]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        personality_signal: torch.Tensor,  # (B, S, D) from PersonalityModule
        intention_signal: torch.Tensor,    # (B, S, D) from IntentionModule
        sync: torch.Tensor,                # (B, S, sync_pairs) from SyncModule
        tick: int,                         # Current tick for oscillation phase
        context: Optional[torch.Tensor] = None,  # (B, S, D) optional context
        surprise_magnitude: Optional[torch.Tensor] = None,  # (B, S, 1) or (B, S)
        surprise_direction: Optional[torch.Tensor] = None,  # (B, S, D) what was unexpected
        valence: Optional[torch.Tensor] = None,  # (B, S, 1) or (B, S) good/bad
        exploration_bonus: Optional[torch.Tensor] = None,  # (B, S, D) from CuriosityModule
    ) -> torch.Tensor:
        """
        Build attention query by synchronizing all signals.

        The query represents "what I want to perceive right now" based on:
        - Who I am (personality)
        - How much I want it (intention)
        - How my neurons are coordinating (sync)
        - Where I am in the oscillation cycle (tick)
        - What surprised me (surprise - closes the experience loop)
        - Was it good or bad (valence - affective modulation)
        - What am I curious about (exploration_bonus - epistemic drive)

        Args:
            personality_signal: Personality-modulated context
            intention_signal: Intention modulation
            sync: Neural synchronization patterns
            tick: Current tick index
            context: Optional additional context
            surprise_magnitude: How surprising (scalar per position)
            surprise_direction: What was unexpected (direction in feature space)
            valence: Was the surprise good (+1) or bad (-1)
            exploration_bonus: Curiosity-driven exploration signal

        Returns:
            query: (B, S, D) attention query
        """
        B, S, D = personality_signal.shape
        device = personality_signal.device
        dtype = personality_signal.dtype

        # 1. Compute oscillation state at current tick
        tick_tensor = torch.tensor(tick, device=device, dtype=dtype)
        freqs = self.frequencies
        amps = F.softmax(self.amplitudes, dim=0)
        phases = self.phases

        # oscillations: (num_oscillators,) values at this tick
        oscillations = amps * torch.sin(2 * math.pi * freqs * tick_tensor + phases)

        # 2. Personality → base query direction
        personality_query = self.personality_to_query(personality_signal)  # (B, S, D)

        # 3. Intention → query magnitude/focus
        intention_magnitude = self.intention_to_magnitude(intention_signal)  # (B, S, 1)
        # Scale personality query by intention (stronger intention = more focused)
        intention_scaled = personality_query * (0.5 + intention_magnitude)  # (B, S, D)

        # 4. Sync → query modulation
        sync_modulation = self.sync_to_query(sync)  # (B, S, D)

        # 5. Oscillation → temporal modulation
        oscillation_signal = self.oscillation_integration(oscillations)  # (D,)
        oscillation_signal = oscillation_signal * self.oscillation_scale  # Scale for impact
        oscillation_signal = oscillation_signal.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        oscillation_signal = oscillation_signal.expand(B, S, -1)  # (B, S, D)

        # 6. Surprise → attention steering (closes the experience loop!)
        # If something surprised us, we should look at it more closely
        if surprise_direction is not None:
            surprise_query = self.surprise_direction_proj(surprise_direction)  # (B, S, D)

            # Scale surprise contribution by magnitude (if provided)
            if surprise_magnitude is not None:
                if surprise_magnitude.dim() == 2:
                    surprise_magnitude = surprise_magnitude.unsqueeze(-1)  # (B, S, 1)
                # Higher magnitude = more attention to surprising things
                surprise_scale = 1.0 + self.surprise_magnitude_scale * surprise_magnitude
                surprise_query = surprise_query * surprise_scale
        else:
            # No surprise: use zeros
            surprise_query = torch.zeros_like(personality_query)

        # 7. Process curiosity exploration bonus
        if exploration_bonus is not None:
            curiosity_query = self.curiosity_scale * exploration_bonus
        else:
            curiosity_query = torch.zeros_like(personality_query)

        # 8. Combine all signals (personality, intention, sync, oscillation, surprise, curiosity)
        combined = torch.cat([
            personality_query,
            intention_scaled,
            sync_modulation,
            oscillation_signal,
            surprise_query,
            curiosity_query,
        ], dim=-1)  # (B, S, D*6)

        query = self.query_combiner(combined)  # (B, S, D)
        query = self.norm(query)

        # 9. Valence → query focus modulation
        # Positive valence → sharper query (amplify magnitude)
        # Negative valence → softer query (reduce magnitude)
        # This affects attention: sharper queries → more focused attention
        if valence is not None:
            if valence.dim() == 2:
                valence = valence.unsqueeze(-1)  # (B, S, 1)
            # Map valence [-1, +1] to focus multiplier [0.7, 1.3]
            # Positive → amplify, Negative → dampen
            focus_mult = 1.0 + self.valence_focus_scale * valence
            query = query * focus_mult

        return query


class PerceptionCrossAttention(nn.Module):
    """
    Cross-attention from query (built from oscillation sync) to
    perception KV cache (from Qwen).

    This is where "what I want to see" meets "what is there to see".
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention

        # Query projection (from OscillationQueryBuilder output)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Check for flash attention
        self._has_flash_attn = False
        if use_flash_attention:
            try:
                from flash_attn import flash_attn_func
                self._has_flash_attn = True
                self._flash_attn_func = flash_attn_func
            except ImportError:
                pass

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02)

    def forward(
        self,
        query: torch.Tensor,    # (B, S, D) from OscillationQueryBuilder
        key: torch.Tensor,      # (B, S_kv, n_heads, head_dim) from KVCache
        value: torch.Tensor,    # (B, S_kv, n_heads, head_dim) from KVCache
        causal: bool = True,    # Use causal masking
        attention_temperature: Optional[torch.Tensor] = None,  # (B, S, 1) from ActivationModule
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attend to perception features.

        Args:
            query: Query built from oscillation synchronization
            key: Cached keys from Qwen
            value: Cached values from Qwen
            causal: Whether to use causal attention
            attention_temperature: Temperature for attention softmax (from ActivationModule).
                                   Lower temperature = sharper focus (high arousal)
                                   Higher temperature = broader attention (low arousal)

        Returns:
            output: (B, S, D) attended perception
            attn_weights: (B, H, S, S_kv) attention weights (for visualization)
        """
        B, S, D = query.shape
        S_kv = key.shape[1]

        # Project query
        q = self.q_proj(query)
        q = q.view(B, S, self.n_heads, self.head_dim)

        # Use flash attention if available and on CUDA
        if self._has_flash_attn and self.training and query.is_cuda:
            # Flash attention requires specific dtype
            target_dtype = q.dtype
            if target_dtype not in (torch.float16, torch.bfloat16):
                target_dtype = torch.bfloat16

            q = q.to(dtype=target_dtype)
            k = key.to(dtype=target_dtype)
            v = value.to(dtype=target_dtype)

            output = self._flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=causal,
            )

            output = output.reshape(B, S, D)
            output = output.to(self.o_proj.weight.dtype)
            output = self.o_proj(output)

            # No attention weights with flash attention
            attn_weights = torch.zeros(B, self.n_heads, S, S_kv, device=query.device)

        else:
            # Standard attention
            # q: (B, S, H, head_dim) -> (B, H, S, head_dim)
            # k: (B, S_kv, H, head_dim) -> (B, H, S_kv, head_dim)
            q = q.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)

            # Attention scores
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, S, S_kv)

            # Apply attention temperature (from arousal)
            # Lower temperature = sharper attention (high arousal, focused)
            # Higher temperature = broader attention (low arousal, relaxed)
            if attention_temperature is not None:
                # attention_temperature: (B, S, 1) -> (B, 1, S, 1) for broadcasting
                temp = attention_temperature.unsqueeze(1)  # (B, 1, S, 1)
                scores = scores / temp

            # Causal mask
            if causal:
                mask = torch.triu(
                    torch.ones(S, S_kv, device=query.device, dtype=torch.bool),
                    diagonal=1
                )
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            # Softmax
            attn_weights = F.softmax(scores, dim=-1)
            if self.dropout > 0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout)

            # Attend
            output = torch.matmul(attn_weights, v)  # (B, H, S, head_dim)
            output = output.transpose(1, 2).reshape(B, S, D)
            output = self.o_proj(output)

        return output, attn_weights


class PerceptionSynapse(nn.Module):
    """
    Integrates current state with perception observation.

    Similar to CTM's SynapseModel but specialized for perception integration.
    Uses U-Net style architecture with skip connections.
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()

        # Input: state + observation
        input_dim = d_model * 2

        # Encoder
        self.enc1 = nn.Linear(input_dim, d_model, bias=False)
        self.enc2 = nn.Linear(d_model, d_model // 2, bias=False)

        # Decoder with skip
        self.dec2 = nn.Linear(d_model // 2, d_model, bias=False)
        self.dec1 = nn.Linear(d_model * 2, d_model, bias=False)  # *2 for skip

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model // 2)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in [self.enc1, self.enc2, self.dec2, self.dec1]:
            nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        state: torch.Tensor,        # (B, S, D) current state
        observation: torch.Tensor,  # (B, S, D) from PerceptionCrossAttention
    ) -> torch.Tensor:
        """
        Integrate state with perception observation.

        Args:
            state: Current hidden state
            observation: Attended perception features

        Returns:
            integrated: (B, S, D) updated state
        """
        # Concatenate
        x = torch.cat([state, observation], dim=-1)

        # Encoder
        h1 = F.gelu(self.enc1(x))
        h1 = self.norm1(h1)
        h2 = F.gelu(self.enc2(h1))
        h2 = self.norm2(h2)

        # Decoder with skip
        d2 = F.gelu(self.dec2(h2))
        d2 = self.dropout(d2)
        d1 = torch.cat([d2, h1], dim=-1)  # Skip connection
        out = self.dec1(d1)

        return out


class PerceptionAttention(nn.Module):
    """
    Complete perception attention module with unified real + imagined attention pool.

    Combines:
    1. KV Cache from Qwen (real perception features)
    2. KV Cache from Imagination (imagined features) - SAME K,V projections
    3. Query Builder (from personality, intention, sync oscillations)
    4. Cross-Attention (query attends to unified [real + imagined] pool)
    5. Synapse (integrates observation with state)

    The key insight: both real and imagined features go through the SAME K, V
    projections, so they compete in the same attention space. The query
    (shaped by personality, intention, surprise, etc.) naturally learns when
    to attend to real perception vs imagination.

    Usage:
        # Initialize
        perception = PerceptionAttention(config)

        # Cache Qwen features (once at start)
        perception.cache_perception(qwen_features)

        # Each tick: optionally add imagination, then attend
        perception.add_imagination(imagined_features)  # optional
        observation = perception(
            state, personality_signal, intention_signal, sync, tick
        )
    """

    def __init__(self, config: PerceptionConfig):
        super().__init__()
        self.config = config

        # KV Cache (handles both perception and imagination)
        self.kv_cache = PerceptionKVCache(
            d_perception=config.d_perception,
            d_model=config.d_model,
            n_heads=config.n_heads,
        )

        # Query Builder
        self.query_builder = OscillationQueryBuilder(
            d_model=config.d_model,
            sync_pairs=config.sync_pairs,
            num_oscillators=config.num_oscillators,
            min_period=config.min_period,
            max_period=config.max_period,
        )

        # Cross-Attention
        self.cross_attention = PerceptionCrossAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_flash_attention=config.use_flash_attention,
        )

        # Synapse
        self.synapse = PerceptionSynapse(
            d_model=config.d_model,
            dropout=config.dropout,
        )

        # Cached KV for REAL perception (populated by cache_perception)
        self._cached_k_real: Optional[torch.Tensor] = None
        self._cached_v_real: Optional[torch.Tensor] = None

        # Cached KV for IMAGINATION (populated by add_imagination)
        self._cached_k_imag: Optional[torch.Tensor] = None
        self._cached_v_imag: Optional[torch.Tensor] = None

    def cache_perception(
        self,
        perception_features: torch.Tensor,  # (B, S, d_perception) from Qwen
    ) -> None:
        """
        Cache real perception KV at the start. Called once per input.

        Args:
            perception_features: Hidden states from Qwen feature extractor
        """
        self._cached_k_real, self._cached_v_real = self.kv_cache.project_perception(
            perception_features
        )

    def add_imagination(
        self,
        imagined_features: torch.Tensor,  # (B, S_imag, d_model) from ImaginationModule
    ) -> None:
        """
        Add imagined features to the attention pool.

        Uses the SAME K, V projections as real perception, ensuring imagination
        competes in the same attention space. Can be called each tick with
        updated imagination.

        Args:
            imagined_features: Features from ImaginationModule
        """
        self._cached_k_imag, self._cached_v_imag = self.kv_cache.project_imagination(
            imagined_features
        )

    def clear_imagination(self) -> None:
        """Clear only the imagination cache (keep real perception)."""
        self._cached_k_imag = None
        self._cached_v_imag = None

    def clear_cache(self) -> None:
        """Clear all KV caches (both real and imagined)."""
        self._cached_k_real = None
        self._cached_v_real = None
        self._cached_k_imag = None
        self._cached_v_imag = None

    @property
    def has_cache(self) -> bool:
        """Check if real perception is cached."""
        return self._cached_k_real is not None

    @property
    def has_imagination(self) -> bool:
        """Check if imagination is in the pool."""
        return self._cached_k_imag is not None

    def _get_unified_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get unified K, V by concatenating real + imagined.

        Returns:
            k: (B, S_real + S_imag, n_heads, head_dim)
            v: (B, S_real + S_imag, n_heads, head_dim)
        """
        if self._cached_k_imag is not None:
            # Concatenate real + imagined along sequence dimension
            k = torch.cat([self._cached_k_real, self._cached_k_imag], dim=1)
            v = torch.cat([self._cached_v_real, self._cached_v_imag], dim=1)
        else:
            k = self._cached_k_real
            v = self._cached_v_real

        return k, v

    def forward(
        self,
        state: torch.Tensor,              # (B, S, D) current CTM state
        personality_signal: torch.Tensor, # (B, S, D) from PersonalityModule
        intention_signal: torch.Tensor,   # (B, S, D) from IntentionModule
        sync: torch.Tensor,               # (B, S, sync_pairs) from SyncModule
        tick: int,                        # Current tick
        causal: bool = False,             # Causal masking (default False for unified pool)
        surprise_magnitude: Optional[torch.Tensor] = None,  # (B, S, 1) or (B, S)
        surprise_direction: Optional[torch.Tensor] = None,  # (B, S, D)
        valence: Optional[torch.Tensor] = None,  # (B, S, 1) or (B, S) good/bad
        exploration_bonus: Optional[torch.Tensor] = None,  # (B, S, D) from CuriosityModule
        attention_temperature: Optional[torch.Tensor] = None,  # (B, S, 1) from ActivationModule
    ) -> PerceptionOutput:
        """
        Attend to unified [real + imagined] perception pool.

        This is the core of the experience loop:
        - Personality defines WHAT to look for
        - Intention defines HOW MUCH to pursue it
        - Sync captures HOW neurons are coordinating
        - Surprise steers attention to WHAT was unexpected (closes the loop!)
        - Valence modulates attention FOCUS (positive→sharp, negative→broad)
        - Curiosity drives EXPLORATION of uncertain/novel regions
        - Arousal controls attention SHARPNESS via temperature
        - Imagination provides WHAT COULD BE for attention to choose from

        The attention pool contains both real perception (from Qwen) and
        imagined features (from ImaginationModule). Both use the same K, V
        projections, so the query naturally learns when to attend to each.

        Args:
            state: Current hidden state
            personality_signal: Personality modulation
            intention_signal: Intention modulation
            sync: Neural synchronization patterns
            tick: Current tick for oscillation phase
            causal: Whether to use causal attention (default False for unified pool)
            surprise_magnitude: How surprising each position was
            surprise_direction: What was unexpected (unit vector in feature space)
            valence: Was the surprise good (+1) or bad (-1)
            exploration_bonus: Curiosity-driven exploration signal
            attention_temperature: Softmax temperature from ActivationModule.
                                   Lower = sharper focus (high arousal)
                                   Higher = broader attention (low arousal)

        Returns:
            PerceptionOutput with:
                - observation: (B, S, D) perceived features (from real + imagined)
                - attention_weights: (B, H, S, S_kv) attention pattern over unified pool
        """
        if not self.has_cache:
            raise RuntimeError(
                "Perception KV not cached. Call cache_perception() first."
            )

        # 1. Build query from all signals: personality, intention, sync, surprise, valence, curiosity
        query = self.query_builder(
            personality_signal=personality_signal,
            intention_signal=intention_signal,
            sync=sync,
            tick=tick,
            surprise_magnitude=surprise_magnitude,
            surprise_direction=surprise_direction,
            valence=valence,
            exploration_bonus=exploration_bonus,
        )

        # 2. Get unified KV pool (real + imagined)
        k, v = self._get_unified_kv()

        # 3. Cross-attend to unified pool (with arousal-modulated temperature)
        attended, attn_weights = self.cross_attention(
            query=query,
            key=k,
            value=v,
            causal=causal,
            attention_temperature=attention_temperature,
        )

        # 4. Integrate with current state via synapse
        observation = self.synapse(state, attended)

        return PerceptionOutput(
            observation=observation,
            attention_weights=attn_weights,
        )


def create_perception_attention(
    d_model: int = 512,
    d_perception: int = 1536,
    n_heads: int = 8,
    sync_pairs: int = 512,
    **kwargs,
) -> PerceptionAttention:
    """Factory function to create a PerceptionAttention module."""
    config = PerceptionConfig(
        d_model=d_model,
        d_perception=d_perception,
        n_heads=n_heads,
        sync_pairs=sync_pairs,
        **kwargs,
    )
    return PerceptionAttention(config)
