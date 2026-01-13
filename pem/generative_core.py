"""
GenerativeCore - Shared generative model for Prediction and Imagination.

Key insight: Both prediction and imagination are GENERATIVE operations that
produce features which don't exist yet. They differ only in conditioning:
- Prediction: "What comes next given context?"
- Imagination: "What could be given context + mode?"

By sharing the generative weights, we:
1. Learn a unified "world model" for generating plausible features
2. Allow imagination to improve predictions (imagined scenarios inform expectations)
3. Allow predictions to ground imagination (predictions constrain what's plausible)

Architecture:
                        ┌─────────────────────────┐
                        │     GenerativeCore      │
                        │                         │
                        │  ┌─────────────────┐    │
    context ──────────▶ │  │    Encoder      │    │
                        │  │  (shared)       │    │
                        │  └────────┬────────┘    │
                        │           │             │
    mode ──────────────▶│  ┌────────▼────────┐    │
    (predict/scene/     │  │  Mode Fusion    │    │
     mind/counter)      │  │                 │    │
                        │  └────────┬────────┘    │
                        │           │             │
                        │  ┌────────▼────────┐    │
                        │  │    Decoder      │    │ ──────▶ generated
                        │  │  (shared)       │    │         features
                        │  └─────────────────┘    │
                        │                         │
                        └─────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math


@dataclass
class GenerativeCoreConfig:
    """Configuration for the shared generative model."""

    d_model: int = 1536              # Feature dimension
    hidden_dim: Optional[int] = None  # Latent space dimension
    n_layers: int = 2                 # Encoder/decoder depth
    n_heads: int = 8                  # Attention heads for cross-attention

    # Mode embeddings for different generation types
    num_modes: int = 5  # predict_next, scene, mind, counterfactual, prospection

    # Architecture options
    dropout: float = 0.0
    use_cross_attention: bool = True  # Cross-attend to context
    use_residual: bool = True         # Residual connections

    def __post_init__(self):
        if self.hidden_dim is None:
            self.hidden_dim = self.d_model


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class GenerativeEncoder(nn.Module):
    """
    Shared encoder that maps context to latent generative space.

    This encoder is used by both prediction and imagination, ensuring
    they build on the same understanding of context.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(d_model, hidden_dim)

        # Encoder layers
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)

        self.norm = RMSNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.input_proj.weight, std=0.02)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Encode context to latent space.

        Args:
            context: (B, S, D) input context

        Returns:
            latent: (B, S, hidden_dim) encoded representation
        """
        x = self.input_proj(context)
        x = self.encoder(x)
        return self.norm(x)


class GenerativeDecoder(nn.Module):
    """
    Shared decoder that generates features from latent + mode.

    This decoder is used by both prediction and imagination, ensuring
    they generate features in the same space.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Decoder layers
        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*layers) if layers else nn.Identity()

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, d_model)

        self.norm = RMSNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output_proj.weight, std=0.02)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to output features.

        Args:
            latent: (B, S, hidden_dim) latent representation

        Returns:
            output: (B, S, D) generated features
        """
        x = self.decoder(latent)
        x = self.output_proj(x)
        return self.norm(x)


class ModeFusion(nn.Module):
    """
    Fuses mode embedding with latent representation.

    Different modes (prediction, scene imagination, mind modeling, etc.)
    condition the generation differently.
    """

    def __init__(self, hidden_dim: int, num_modes: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes

        # Mode embeddings
        self.mode_embeddings = nn.Embedding(num_modes, hidden_dim)

        # Mode gating: how much should mode influence generation?
        self.mode_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Mode-conditioned transform
        self.mode_transform = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mode_embeddings.weight, std=0.02)
        for module in [self.mode_gate, self.mode_transform]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        latent: torch.Tensor,  # (B, S, hidden_dim)
        mode: int,             # Mode index
    ) -> torch.Tensor:
        """
        Fuse mode with latent representation.

        Args:
            latent: Encoded context
            mode: Generation mode index
                0 = predict_next (prediction)
                1 = scene (scene imagination)
                2 = mind (theory of mind)
                3 = counterfactual
                4 = prospection (future simulation)

        Returns:
            fused: (B, S, hidden_dim) mode-conditioned latent
        """
        B, S, D = latent.shape

        # Get mode embedding and expand to match latent
        mode_idx = torch.tensor(mode, device=latent.device)
        mode_embed = self.mode_embeddings(mode_idx)  # (hidden_dim,)
        mode_embed = mode_embed.unsqueeze(0).unsqueeze(0).expand(B, S, -1)  # (B, S, hidden_dim)

        # Concatenate latent and mode
        combined = torch.cat([latent, mode_embed], dim=-1)  # (B, S, 2*hidden_dim)

        # Compute gate (how much mode influence)
        gate = self.mode_gate(combined)  # (B, S, hidden_dim)

        # Compute mode-conditioned transform
        transform = self.mode_transform(combined)  # (B, S, hidden_dim)

        # Apply gated fusion: latent + gate * transform
        fused = latent + gate * transform

        return fused


class GenerativeCrossAttention(nn.Module):
    """
    Optional cross-attention to context for richer generation.

    Allows the generator to attend back to the original context,
    picking up relevant details for generation.
    """

    def __init__(
        self,
        hidden_dim: int,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        assert hidden_dim % n_heads == 0

        # Q from latent, K/V from context
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.v_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(proj.weight, std=0.02)

    def forward(
        self,
        latent: torch.Tensor,   # (B, S, hidden_dim)
        context: torch.Tensor,  # (B, S, d_model)
    ) -> torch.Tensor:
        """
        Cross-attend from latent to context.

        Args:
            latent: Mode-fused latent representation
            context: Original context features

        Returns:
            attended: (B, S, hidden_dim) context-enriched latent
        """
        B, S, _ = latent.shape
        S_kv = context.shape[1]

        # Project
        q = self.q_proj(latent).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(context).view(B, S_kv, self.n_heads, self.head_dim)
        v = self.v_proj(context).view(B, S_kv, self.n_heads, self.head_dim)

        # Transpose for attention: (B, H, S, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Attend
        out = torch.matmul(attn, v)  # (B, H, S, head_dim)
        out = out.transpose(1, 2).reshape(B, S, self.hidden_dim)

        return self.o_proj(out)


class GenerativeCore(nn.Module):
    """
    Shared generative model for both Prediction and Imagination.

    This is the core world model that learns to generate plausible features.
    Different modes (prediction, scene, mind, counterfactual) condition the
    generation to produce different types of outputs, but they all share
    the same underlying generative capability.

    Usage:
        core = GenerativeCore(config)

        # For prediction
        pred_features = core(context, mode="predict")

        # For scene imagination
        scene_features = core(context, mode="scene")

        # For theory of mind
        mind_features = core(context, mode="mind")

        # For counterfactuals
        counter_features = core(context, mode="counter")
    """

    # Mode name to index mapping
    MODE_MAP = {
        "predict": 0,
        "predict_next": 0,
        "scene": 1,
        "mind": 2,
        "counterfactual": 3,
        "counter": 3,
        "prospection": 4,
        "future": 4,
    }

    def __init__(self, config: GenerativeCoreConfig):
        super().__init__()
        self.config = config

        # Shared encoder
        self.encoder = GenerativeEncoder(
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )

        # Mode fusion
        self.mode_fusion = ModeFusion(
            hidden_dim=config.hidden_dim,
            num_modes=config.num_modes,
        )

        # Optional cross-attention to context
        if config.use_cross_attention:
            self.cross_attention = GenerativeCrossAttention(
                hidden_dim=config.hidden_dim,
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout,
            )
            self.cross_attn_norm = RMSNorm(config.hidden_dim)
        else:
            self.cross_attention = None

        # Shared decoder
        self.decoder = GenerativeDecoder(
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )

        self.use_residual = config.use_residual

    def forward(
        self,
        context: torch.Tensor,  # (B, S, D) input context
        mode: str = "predict",  # Generation mode
        memory: Optional[torch.Tensor] = None,  # (B, S, D) optional memory
        personality: Optional[torch.Tensor] = None,  # (D,) optional personality
    ) -> torch.Tensor:
        """
        Generate features conditioned on mode.

        Args:
            context: Input context features
            mode: Generation mode - one of:
                - "predict" / "predict_next": Prediction (what comes next)
                - "scene": Scene imagination (mental imagery)
                - "mind": Mind modeling (theory of mind)
                - "counter" / "counterfactual": Counterfactual generation
                - "prospection" / "future": Future simulation

            memory: Optional memory to enrich generation
            personality: Optional personality to color generation

        Returns:
            generated: (B, S, D) generated features
        """
        # Get mode index
        mode_idx = self.MODE_MAP.get(mode, 0)

        # Optionally combine with memory
        if memory is not None:
            enriched_context = context + 0.3 * memory
        else:
            enriched_context = context

        # 1. Encode context
        latent = self.encoder(enriched_context)

        # 2. Fuse with mode
        latent = self.mode_fusion(latent, mode_idx)

        # 3. Optional cross-attention back to context
        if self.cross_attention is not None:
            attn_out = self.cross_attention(latent, context)
            latent = self.cross_attn_norm(latent + attn_out)

        # 4. Apply personality bias if available
        if personality is not None:
            personality_factor = torch.tanh(personality.mean()) * 0.1
            latent = latent * (1.0 + personality_factor)

        # 5. Decode to output features
        generated = self.decoder(latent)

        # 6. Optional residual connection to context
        if self.use_residual:
            generated = generated + 0.1 * context

        return generated

    def generate_multiple(
        self,
        context: torch.Tensor,
        modes: list,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate features for multiple modes efficiently.

        Args:
            context: Input context
            modes: List of mode names
            **kwargs: Additional arguments passed to forward

        Returns:
            Dict mapping mode name to generated features
        """
        results = {}
        for mode in modes:
            results[mode] = self(context, mode=mode, **kwargs)
        return results


def create_generative_core(
    d_model: int = 1536,
    hidden_dim: Optional[int] = None,
    n_layers: int = 2,
    n_heads: int = 8,
    **kwargs,
) -> GenerativeCore:
    """Factory function to create a GenerativeCore."""
    config = GenerativeCoreConfig(
        d_model=d_model,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        **kwargs,
    )
    return GenerativeCore(config)
