"""
Prediction Module for PEM.

The system constantly asks: "Given my current understanding (sync),
what do I expect to see next?"

Generates multi-scale predictions:
- Immediate: next token features
- Short-term: next ~64 tokens (phrase/sentence level)
- Long-term: rest of document (topic/trajectory)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PredictionConfig:
    """Configuration for the prediction module."""

    # Input dimensions
    sync_pairs: int = 512       # Dimension of sync from CTM
    d_model: int = 1536         # Dimension of features from extractor

    # Attention configuration
    n_head: int = 8
    dropout: float = 0.0

    # Prediction horizons (all in tokens)
    immediate_horizon: int = 8    # ~2-3 words - next phrase fragment
    shortterm_horizon: int = 64   # ~1-2 sentences
    longterm_horizon: Optional[int] = 256  # ~few paragraphs
                                           # None = rest of sequence

    # Architecture options
    use_flash_attn: bool = True
    use_rope: bool = False       # RoPE for prediction attention

    # Shared generative core (optional)
    # When True, uses shared GenerativeCore instead of separate prediction heads
    # This allows prediction and imagination to share weights
    use_shared_generative_core: bool = False


class CausalCrossAttention(nn.Module):
    """
    Cross-attention with causal masking.

    Query attends to keys/values, but only to positions <= query position.
    This ensures predictions at position t only use context from positions <= t.
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0, use_flash: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout = dropout
        self.use_flash = use_flash

        assert d_model % n_head == 0, f"d_model ({d_model}) must be divisible by n_head ({n_head})"

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02 / math.sqrt(2))

    def forward(
        self,
        query: torch.Tensor,    # (B, S_q, D)
        key: torch.Tensor,      # (B, S_kv, D)
        value: torch.Tensor,    # (B, S_kv, D)
    ) -> torch.Tensor:
        """
        Causal cross-attention.

        Position i in query can only attend to positions 0..i in key/value.
        """
        B, S_q, D = query.shape
        S_kv = key.shape[1]

        # Project
        q = self.q_proj(query).view(B, S_q, self.n_head, self.head_dim)
        k = self.k_proj(key).view(B, S_kv, self.n_head, self.head_dim)
        v = self.v_proj(value).view(B, S_kv, self.n_head, self.head_dim)

        # Store original dtype for output
        orig_dtype = query.dtype

        # Try flash attention if available and enabled
        if self.use_flash and S_q == S_kv:
            try:
                from flash_attn import flash_attn_func

                # Flash attention expects (B, S, H, D) in fp16/bf16
                target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                q_flash = q.to(target_dtype)
                k_flash = k.to(target_dtype)
                v_flash = v.to(target_dtype)

                attn_dropout = self.dropout if self.training else 0.0
                out = flash_attn_func(q_flash, k_flash, v_flash, dropout_p=attn_dropout, causal=True)
                out = out.reshape(B, S_q, D)
                # Cast back to original dtype for o_proj
                out = out.to(orig_dtype)
                return self.o_proj(out)
            except (ImportError, RuntimeError):
                pass  # Fall back to manual implementation

        # Manual implementation with causal mask
        # (B, S, H, D) -> (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, S_q, S_kv)

        # Causal mask: position i can only attend to positions <= i
        # This works for cross-attention where S_q == S_kv (same sequence)
        if S_q == S_kv:
            causal_mask = torch.triu(
                torch.ones(S_q, S_kv, dtype=torch.bool, device=scores.device),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        else:
            # For different lengths, create appropriate mask
            # Query pos i can attend to key pos j if j <= i
            q_pos = torch.arange(S_q, device=scores.device).unsqueeze(1)
            k_pos = torch.arange(S_kv, device=scores.device).unsqueeze(0)
            causal_mask = k_pos > q_pos  # True where we should mask
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v)  # (B, H, S_q, D)
        out = out.transpose(1, 2).reshape(B, S_q, D)

        return self.o_proj(out)


class PredictionHead(nn.Module):
    """
    Single prediction head with optional MLP.

    Projects from internal representation to feature space prediction.
    """

    def __init__(self, d_model: int, use_mlp: bool = True):
        super().__init__()
        self.d_model = d_model

        if use_mlp:
            # Two-layer MLP with GELU
            self.net = nn.Sequential(
                nn.Linear(d_model, d_model * 2, bias=False),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model, bias=False),
            )
        else:
            self.net = nn.Linear(d_model, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionModule(nn.Module):
    """
    Sync state → predicted features at multiple time scales.

    Architecture:
        1. Project sync to query space (sync_pairs → d_model)
        2. Cross-attend to context features (causal)
        3. Apply scale-specific prediction heads (or shared GenerativeCore)

    Multi-scale predictions:
        - immediate: Next token features (t+1)
        - shortterm: Average of next N tokens (t+1 to t+N)
        - longterm:  Average of remaining document

    Supports optional shared GenerativeCore:
        When `use_shared_generative_core=True` or a GenerativeCore is set via
        `set_generative_core()`, the prediction uses shared weights with
        imagination. This allows predictions to benefit from imagination and
        vice versa.
    """

    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config

        # Shared generative core (optional - can be set later)
        self._generative_core = None
        self._use_shared_core = config.use_shared_generative_core

        # If using shared core, create it
        if config.use_shared_generative_core:
            from .generative_core import GenerativeCore, GenerativeCoreConfig
            core_config = GenerativeCoreConfig(
                d_model=config.d_model,
                hidden_dim=config.d_model,
                n_layers=2,
                n_heads=config.n_head,
                dropout=config.dropout,
            )
            self._generative_core = GenerativeCore(core_config)

        # Sync → query projection
        self.sync_to_query = nn.Sequential(
            nn.Linear(config.sync_pairs, config.d_model, bias=False),
            nn.LayerNorm(config.d_model),
        )

        # Cross-attention: query (from sync) attends to context (features)
        self.cross_attn = CausalCrossAttention(
            d_model=config.d_model,
            n_head=config.n_head,
            dropout=config.dropout,
            use_flash=config.use_flash_attn,
        )

        # Layer norm after attention
        self.attn_norm = nn.LayerNorm(config.d_model)

        # Scale-specific prediction heads (used when not using shared core)
        if not config.use_shared_generative_core:
            self.immediate_head = PredictionHead(config.d_model, use_mlp=True)
            self.shortterm_head = PredictionHead(config.d_model, use_mlp=True)
            self.longterm_head = PredictionHead(config.d_model, use_mlp=True)
        else:
            self.immediate_head = None
            self.shortterm_head = None
            self.longterm_head = None

        self._init_weights()

    def _init_weights(self):
        # Initialize sync_to_query
        for module in self.sync_to_query.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

    def set_generative_core(self, core: 'GenerativeCore') -> None:
        """
        Set a shared GenerativeCore for prediction.

        This allows prediction and imagination to share weights.
        The same core can be passed to ImaginationModule.

        Args:
            core: GenerativeCore instance to use
        """
        self._generative_core = core
        self._use_shared_core = True

    @property
    def generative_core(self):
        """Get the generative core (if any)."""
        return self._generative_core

    @property
    def uses_shared_core(self) -> bool:
        """Check if using shared generative core."""
        return self._use_shared_core and self._generative_core is not None

    def forward(
        self,
        sync: torch.Tensor,      # (B, S, sync_pairs) from CTM
        context: torch.Tensor,   # (B, S, d_model) from FeatureExtractor
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions at multiple time scales.

        Args:
            sync: Synchronization state from CTM
            context: Feature context from FeatureExtractor

        Returns:
            Dict with keys:
                'immediate': (B, S, D) - predicted features for position t+1
                'shortterm': (B, S, D) - predicted avg features for t+1 to t+horizon
                'longterm':  (B, S, D) - predicted avg features for rest of sequence
                'basis':     (B, S, D) - intermediate representation (for analysis)
        """
        B, S, _ = sync.shape

        # 1. Project sync to query space
        query = self.sync_to_query(sync)  # (B, S, d_model)

        # 2. Cross-attend to context (causal: position i attends to 0..i)
        attn_out = self.cross_attn(query, context, context)  # (B, S, d_model)

        # Residual + norm
        pred_basis = self.attn_norm(query + attn_out)  # (B, S, d_model)

        # 3. Apply prediction heads (or shared core)
        if self.uses_shared_core:
            # Use shared GenerativeCore for all predictions
            # The "predict" mode generates next-token predictions
            generated = self._generative_core(pred_basis, mode="predict")
            predictions = {
                'immediate': generated,   # (B, S, D)
                'shortterm': generated,   # Same base, different target
                'longterm': generated,    # Same base, different target
                'basis': pred_basis,
            }
        else:
            # Use dedicated prediction heads
            # Each head predicts from every position (during training)
            # During inference, typically only use last position
            predictions = {
                'immediate': self.immediate_head(pred_basis),   # (B, S, D)
                'shortterm': self.shortterm_head(pred_basis),   # (B, S, D)
                'longterm': self.longterm_head(pred_basis),     # (B, S, D)
                'basis': pred_basis,                            # For analysis/debugging
            }

        return predictions

    def predict_at_position(
        self,
        sync: torch.Tensor,      # (B, S, sync_pairs)
        context: torch.Tensor,   # (B, S, d_model)
        position: int = -1,      # Which position to predict from
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions from a specific position.

        Convenience method for inference - returns predictions only for
        the specified position (default: last).

        Returns:
            Dict with (B, 1, D) tensors for each scale
        """
        full_preds = self.forward(sync, context)

        return {
            'immediate': full_preds['immediate'][:, position:position+1 if position != -1 else None, :],
            'shortterm': full_preds['shortterm'][:, position:position+1 if position != -1 else None, :],
            'longterm': full_preds['longterm'][:, position:position+1 if position != -1 else None, :],
        }


class PredictionTargets:
    """
    Compute prediction targets from actual features.

    Used during training to create the ground truth that predictions
    should match.

    Targets are single tokens at specific offsets (not averaged windows).
    This preserves full variance at all horizons.
    """

    def __init__(
        self,
        immediate_horizon: int = 1,
        shortterm_horizon: int = 32,
        longterm_horizon: Optional[int] = None,
    ):
        """
        Args:
            immediate_horizon: Exact token offset for immediate target (default: 1 = next token)
            shortterm_horizon: Exact token offset for short-term target (default: 32)
            longterm_horizon: Exact token offset for long-term target (default: None = end of sequence)
        """
        self.immediate_horizon = immediate_horizon
        self.shortterm_horizon = shortterm_horizon
        self.longterm_horizon = longterm_horizon

    def compute_targets(
        self,
        features: torch.Tensor,  # (B, S, D) actual features from extractor
        padding_mask: Optional[torch.Tensor] = None,  # (B, S) True for valid positions
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale prediction targets using endpoint prediction.

        For position t, compute:
            - immediate: features[t + immediate_horizon] (e.g., next token)
            - shortterm: features[t + shortterm_horizon] (e.g., 32 tokens ahead)
            - longterm:  features[t + longterm_horizon] (e.g., 128 tokens ahead)

        Each target is a single token, preserving full variance at all horizons.

        Args:
            features: Actual features from feature extractor
            padding_mask: Optional mask for valid positions (True=valid)

        Returns:
            Dict with target tensors, each (B, S, D)
            Positions where targets can't be computed are zeroed.
        """
        B, S, D = features.shape
        device = features.device

        # Initialize targets
        immediate_targets = torch.zeros_like(features)
        shortterm_targets = torch.zeros_like(features)
        longterm_targets = torch.zeros_like(features)

        # Validity masks: which positions have valid targets
        immediate_valid = torch.zeros(B, S, dtype=torch.bool, device=device)
        shortterm_valid = torch.zeros(B, S, dtype=torch.bool, device=device)
        longterm_valid = torch.zeros(B, S, dtype=torch.bool, device=device)

        # Compute sequence lengths (accounting for padding)
        if padding_mask is not None:
            seq_lengths = padding_mask.sum(dim=1)  # (B,)
        else:
            seq_lengths = torch.full((B,), S, device=device)

        for b in range(B):
            seq_len = int(seq_lengths[b].item())

            # Immediate: single token at t + immediate_horizon
            valid_end = seq_len - self.immediate_horizon
            if valid_end > 0:
                immediate_targets[b, :valid_end] = features[b, self.immediate_horizon:seq_len]
                immediate_valid[b, :valid_end] = True
                if padding_mask is not None:
                    immediate_valid[b, :valid_end] &= padding_mask[b, :valid_end]

            # Shortterm: single token at t + shortterm_horizon
            valid_end = seq_len - self.shortterm_horizon
            if valid_end > 0:
                shortterm_targets[b, :valid_end] = features[b, self.shortterm_horizon:seq_len]
                shortterm_valid[b, :valid_end] = True
                if padding_mask is not None:
                    shortterm_valid[b, :valid_end] &= padding_mask[b, :valid_end]

            # Longterm: single token at t + longterm_horizon (or last token if None)
            longterm_horizon = self.longterm_horizon if self.longterm_horizon is not None else (seq_len - 1)
            valid_end = seq_len - longterm_horizon
            if valid_end > 0:
                longterm_targets[b, :valid_end] = features[b, longterm_horizon:seq_len]
                longterm_valid[b, :valid_end] = True
                if padding_mask is not None:
                    longterm_valid[b, :valid_end] &= padding_mask[b, :valid_end]

        return {
            'immediate': immediate_targets,
            'shortterm': shortterm_targets,
            'longterm': longterm_targets,
            'immediate_valid': immediate_valid,
            'shortterm_valid': shortterm_valid,
            'longterm_valid': longterm_valid,
        }

    def compute_targets_efficient(
        self,
        features: torch.Tensor,  # (B, S, D)
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Vectorized endpoint target computation (faster for training).

        For position t, targets are single tokens at exact offsets:
            - immediate: features[t + immediate_horizon]
            - shortterm: features[t + shortterm_horizon]
            - longterm:  features[t + longterm_horizon]
        """
        B, S, D = features.shape
        device = features.device

        if padding_mask is not None:
            padding_mask = padding_mask.to(device=device).bool()

        # Helper to compute endpoint targets for a given horizon
        def endpoint_targets(horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
            targets = torch.zeros_like(features)
            valid = torch.zeros(B, S, dtype=torch.bool, device=device)

            if horizon < S:
                # targets[:, t] = features[:, t + horizon] for valid t
                valid_positions = S - horizon
                targets[:, :valid_positions] = features[:, horizon:]
                valid[:, :valid_positions] = True

                if padding_mask is not None:
                    # Source position must be valid AND target position must be valid
                    target_valid = padding_mask[:, horizon:]  # (B, S-horizon)
                    source_valid = padding_mask[:, :valid_positions]  # (B, S-horizon)
                    combined_valid = target_valid & source_valid
                    valid[:, :valid_positions] = combined_valid
                    # Zero out invalid targets
                    targets[:, :valid_positions] = targets[:, :valid_positions] * combined_valid.unsqueeze(-1)

            return targets, valid

        immediate_targets, immediate_valid = endpoint_targets(self.immediate_horizon)
        shortterm_targets, shortterm_valid = endpoint_targets(self.shortterm_horizon)
        # Handle None longterm_horizon: predict the last token (S-1) for all positions
        longterm_horizon = self.longterm_horizon if self.longterm_horizon is not None else (S - 1)
        longterm_targets, longterm_valid = endpoint_targets(longterm_horizon)

        # Apply padding mask if provided
        if padding_mask is not None:
            immediate_valid = immediate_valid & padding_mask
            shortterm_valid = shortterm_valid & padding_mask
            longterm_valid = longterm_valid & padding_mask

        return {
            'immediate': immediate_targets,
            'shortterm': shortterm_targets,
            'longterm': longterm_targets,
            'immediate_valid': immediate_valid,
            'shortterm_valid': shortterm_valid,
            'longterm_valid': longterm_valid,
        }


class PredictionLoss(nn.Module):
    """
    Compute prediction loss across all scales.

    Uses cosine similarity as the primary metric (feature vectors
    should point in the same direction), with optional MSE component.
    """

    def __init__(
        self,
        immediate_weight: float = 1.0,
        shortterm_weight: float = 0.5,
        longterm_weight: float = 0.3,
        use_cosine: bool = True,
        use_mse: bool = False,
        mse_weight: float = 0.1,
    ):
        super().__init__()
        self.immediate_weight = immediate_weight
        self.shortterm_weight = shortterm_weight
        self.longterm_weight = longterm_weight
        self.use_cosine = use_cosine
        self.use_mse = use_mse
        self.mse_weight = mse_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted prediction loss.

        Args:
            predictions: Dict with 'immediate', 'shortterm', 'longterm' predictions
            targets: Dict with targets and validity masks

        Returns:
            total_loss: Scalar loss
            loss_dict: Breakdown by scale (for logging)
        """
        loss_dict = {}
        total_loss = 0.0

        scales = [
            ('immediate', self.immediate_weight),
            ('shortterm', self.shortterm_weight),
            ('longterm', self.longterm_weight),
        ]

        for scale_name, weight in scales:
            pred = predictions[scale_name]
            target = targets[scale_name]
            valid = targets[f'{scale_name}_valid']

            if not valid.any():
                loss_dict[f'{scale_name}_loss'] = torch.tensor(0.0, device=pred.device)
                continue

            # Mask out invalid positions
            pred_valid = pred[valid]      # (N, D)
            target_valid = target[valid]  # (N, D)

            scale_loss = 0.0

            if self.use_cosine:
                # Cosine similarity loss: 1 - cos_sim
                # cos_sim in [-1, 1], so loss in [0, 2]
                cos_sim = F.cosine_similarity(pred_valid, target_valid, dim=-1)
                cosine_loss = (1 - cos_sim).mean()
                scale_loss = scale_loss + cosine_loss
                loss_dict[f'{scale_name}_cosine'] = cosine_loss.detach()

            if self.use_mse:
                # Normalize before MSE to focus on direction
                pred_norm = F.normalize(pred_valid, dim=-1)
                target_norm = F.normalize(target_valid, dim=-1)
                mse_loss = F.mse_loss(pred_norm, target_norm)
                scale_loss = scale_loss + self.mse_weight * mse_loss
                loss_dict[f'{scale_name}_mse'] = mse_loss.detach()

            loss_dict[f'{scale_name}_loss'] = scale_loss.detach()
            total_loss = total_loss + weight * scale_loss

        return total_loss, loss_dict


def create_prediction_module(
    sync_pairs: int = 512,
    d_model: int = 1536,
    n_head: int = 8,
    immediate_horizon: int = 8,
    shortterm_horizon: int = 64,
    longterm_horizon: Optional[int] = 256,
    **kwargs,
) -> PredictionModule:
    """Factory function to create a prediction module.

    Args:
        sync_pairs: Dimension of sync from CTM
        d_model: Dimension of features from extractor
        n_head: Number of attention heads
        immediate_horizon: Tokens ahead for immediate prediction (default: 8, ~2-3 words)
        shortterm_horizon: Tokens ahead for short-term prediction (default: 64, ~1-2 sentences)
        longterm_horizon: Tokens ahead for long-term prediction (default: 256, ~paragraphs)
                         Set to None for "rest of sequence" behavior
    """
    config = PredictionConfig(
        sync_pairs=sync_pairs,
        d_model=d_model,
        n_head=n_head,
        immediate_horizon=immediate_horizon,
        shortterm_horizon=shortterm_horizon,
        longterm_horizon=longterm_horizon,
        **kwargs,
    )
    return PredictionModule(config)
