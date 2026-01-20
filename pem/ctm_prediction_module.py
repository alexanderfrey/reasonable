"""
CTM (Continuous Thought Machine) Prediction Module for PEM.

Faithful implementation of the CTM architecture:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    CTM Internal Tick Loop                           │
    │                                                                     │
    │  for t in 1..T:                                                    │
    │      ┌──────────────────────────────────────────────────────┐      │
    │      │  a_t = Synapse(concat(z_t, o_t))   [pre-activations] │      │
    │      └──────────────────┬───────────────────────────────────┘      │
    │                         │                                          │
    │                         ▼                                          │
    │      ┌──────────────────────────────────────────────────────┐      │
    │      │  A_t = [a_{t-M+1}, ..., a_t]   [pre-activation hist] │      │
    │      └──────────────────┬───────────────────────────────────┘      │
    │                         │                                          │
    │           ┌─────────────┼─────────────┐                            │
    │           ▼             ▼             ▼                            │
    │      ┌─────────┐   ┌─────────┐   ┌─────────┐                       │
    │      │ NLM_1   │   │ NLM_2   │...│ NLM_D   │  [per-neuron MLPs]   │
    │      │g_θ1(A_1)│   │g_θ2(A_2)│   │g_θD(A_D)│                       │
    │      └────┬────┘   └────┬────┘   └────┬────┘                       │
    │           └─────────────┼─────────────┘                            │
    │                         ▼                                          │
    │      ┌──────────────────────────────────────────────────────┐      │
    │      │  z_{t+1} = concat([z_1, z_2, ..., z_D])              │      │
    │      │  Z_t = [z_1, z_2, ..., z_t]   [post-activation hist] │      │
    │      └──────────────────┬───────────────────────────────────┘      │
    │                         │                                          │
    │                         ▼                                          │
    │      ┌──────────────────────────────────────────────────────┐      │
    │      │  S_t = Z_t · Z_t^T            [synchronization]      │      │
    │      │  S_action, S_out = subsample(S_t)                    │      │
    │      └──────────────────┬───────────────────────────────────┘      │
    │                         │                                          │
    │              ┌──────────┴──────────┐                               │
    │              ▼                      ▼                               │
    │      ┌─────────────┐        ┌─────────────┐                        │
    │      │q_t = W_in·S │        │y_t = W_out·S│                        │
    │      │   action    │        │     out     │                        │
    │      └──────┬──────┘        └─────────────┘                        │
    │             │                                                       │
    │             ▼                                                       │
    │      ┌──────────────────────────────────────────────────────┐      │
    │      │  o_t = Attention(Q=q_t, KV=features)                 │      │
    │      └──────────────────┬───────────────────────────────────┘      │
    │                         │                                          │
    │                         └──────────► next tick                     │
    └─────────────────────────────────────────────────────────────────────┘

Key concepts:
- Pre-activations (a_t): Output of synapse, collected into history A_t
- Post-activations (z_t): Output of per-neuron MLPs, form the "thought state"
- Synchronization (S_t): Z_t · Z_t^T captures temporal correlations between neurons
- Internal ticks: The model "thinks" for T steps before producing final output
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, NamedTuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTMPredictionOutput(NamedTuple):
    """Output from CTMPrediction forward pass."""
    predictions: Dict[str, torch.Tensor]  # immediate/shortterm/longterm at final tick
    all_outputs: List[torch.Tensor]       # y_t at each internal tick
    sync_matrix: torch.Tensor             # Final synchronization matrix
    post_activation_history: torch.Tensor  # Z_T for analysis


@dataclass
class CTMPredictionConfig:
    """Configuration for CTM prediction module."""

    # Dimensions
    d_model: int = 1536          # Feature dimension (from backbone)
    d_neurons: int = 512         # Number of neurons (D in the paper)
    d_sync_out: int = 256        # Number of sync pairs for output
    d_sync_action: int = 256     # Number of sync pairs for attention

    # History lengths
    M: int = 16                  # Pre-activation history length per neuron

    # Internal ticks
    T: int = 8                   # Number of internal thinking steps

    # Architecture
    synapse_hidden: int = 1024   # Hidden dim in synapse U-NET
    nlm_hidden: int = 64         # Hidden dim in per-neuron MLPs
    n_attention_heads: int = 8   # Heads for cross-attention

    # Prediction horizons (for output projection)
    immediate_horizon: int = 8
    shortterm_horizon: int = 64
    longterm_horizon: int = 256

    dropout: float = 0.0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class SynapseModel(nn.Module):
    """
    Synapse model: U-NET style MLP that produces pre-activations.

    a_t = f_θsyn(concat(z_t, o_t))

    Takes concatenation of current post-activations and attention output,
    produces pre-activations that feed into neuron-level models.
    """

    def __init__(
        self,
        d_neurons: int,      # D - number of neurons
        d_attention: int,    # Dimension of attention output
        d_hidden: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_neurons = d_neurons
        d_input = d_neurons + d_attention  # concat(z_t, o_t)

        # U-NET style: down -> bottleneck -> up with skip connections
        self.down1 = nn.Linear(d_input, d_hidden)
        self.down2 = nn.Linear(d_hidden, d_hidden // 2)
        self.bottleneck = nn.Linear(d_hidden // 2, d_hidden // 4)
        self.up1 = nn.Linear(d_hidden // 4 + d_hidden // 2, d_hidden // 2)  # skip
        self.up2 = nn.Linear(d_hidden // 2 + d_hidden, d_hidden)            # skip
        self.out = nn.Linear(d_hidden, d_neurons)

        self.norm1 = RMSNorm(d_hidden)
        self.norm2 = RMSNorm(d_hidden // 2)
        self.norm3 = RMSNorm(d_hidden // 4)
        self.norm_out = RMSNorm(d_neurons)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(
        self,
        z_t: torch.Tensor,    # (B, S, d_neurons) post-activations
        o_t: torch.Tensor,    # (B, S, d_attention) attention output
    ) -> torch.Tensor:
        """
        Produce pre-activations from post-activations and attention.

        Returns:
            a_t: (B, S, d_neurons) pre-activations
        """
        # Concatenate inputs
        x = torch.cat([z_t, o_t], dim=-1)  # (B, S, d_neurons + d_attention)

        # U-NET forward with skip connections
        h1 = self.act(self.norm1(self.down1(x)))
        h1 = self.dropout(h1)

        h2 = self.act(self.norm2(self.down2(h1)))
        h2 = self.dropout(h2)

        h3 = self.act(self.norm3(self.bottleneck(h2)))

        # Up path with skip connections
        h4 = self.act(self.up1(torch.cat([h3, h2], dim=-1)))
        h4 = self.dropout(h4)

        h5 = self.act(self.up2(torch.cat([h4, h1], dim=-1)))
        h5 = self.dropout(h5)

        # Output pre-activations
        a_t = self.norm_out(self.out(h5))

        return a_t


class NeuronLevelModels(nn.Module):
    """
    Per-neuron MLPs that process pre-activation histories.

    Each neuron d has its own privately parameterized MLP:
    z_d^{t+1} = g_θd(A_d^t)

    where A_d^t is the M-dimensional time series of pre-activations for neuron d.
    """

    def __init__(
        self,
        d_neurons: int,      # D - number of neurons
        M: int,              # History length
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_neurons = d_neurons
        self.M = M
        self.hidden_dim = hidden_dim

        # Each neuron has its own MLP: M -> hidden -> 1
        # We implement this efficiently using grouped convolutions or
        # separate weight tensors

        # Input projection: (D, M) -> (D, hidden)
        # Each neuron has its own weights
        self.w1 = nn.Parameter(torch.randn(d_neurons, M, hidden_dim) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(d_neurons, hidden_dim))

        # Output projection: (D, hidden) -> (D, 1)
        self.w2 = nn.Parameter(torch.randn(d_neurons, hidden_dim, 1) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(d_neurons, 1))

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        A_t: torch.Tensor,  # (B, S, D, M) pre-activation history per neuron
    ) -> torch.Tensor:
        """
        Each neuron processes its own history to produce post-activation.

        Returns:
            z_t: (B, S, D) post-activations
        """
        B, S, D, M = A_t.shape
        assert D == self.d_neurons and M == self.M

        # Reshape for batched matrix multiply
        # A_t: (B, S, D, M) -> (B*S, D, M)
        A_flat = A_t.reshape(B * S, D, M)

        # Per-neuron linear: (B*S, D, M) @ (D, M, hidden) -> (B*S, D, hidden)
        # Use einsum for per-neuron operation
        h = torch.einsum('bdm,dmh->bdh', A_flat, self.w1) + self.b1

        h = self.act(h)
        h = self.dropout(h)

        # Output: (B*S, D, hidden) @ (D, hidden, 1) -> (B*S, D, 1)
        z = torch.einsum('bdh,dho->bdo', h, self.w2) + self.b2
        z = z.squeeze(-1)  # (B*S, D)

        # Reshape back
        z_t = z.reshape(B, S, D)

        return z_t


class SynchronizationModule(nn.Module):
    """
    Computes neural synchronization from post-activation history.

    S_t = Z_t · (Z_t)^T ∈ R^{D×D}

    Then subsamples (i,j) pairs to get S_out and S_action.
    """

    def __init__(
        self,
        d_neurons: int,       # D
        d_sync_out: int,      # Number of pairs for output
        d_sync_action: int,   # Number of pairs for attention
    ):
        super().__init__()
        self.d_neurons = d_neurons
        self.d_sync_out = d_sync_out
        self.d_sync_action = d_sync_action

        # Register fixed random (i,j) pairs for subsampling
        # These define which neuron pairs to use for output and action
        total_pairs = (d_neurons * (d_neurons + 1)) // 2

        # Sample indices for output sync
        out_indices = self._sample_pairs(d_neurons, d_sync_out)
        self.register_buffer('out_i', out_indices[:, 0])
        self.register_buffer('out_j', out_indices[:, 1])

        # Sample indices for action sync
        action_indices = self._sample_pairs(d_neurons, d_sync_action)
        self.register_buffer('action_i', action_indices[:, 0])
        self.register_buffer('action_j', action_indices[:, 1])

    def _sample_pairs(self, D: int, n_pairs: int) -> torch.Tensor:
        """Sample n_pairs (i,j) index pairs from upper triangle of DxD matrix."""
        # Generate all unique pairs
        pairs = []
        for i in range(D):
            for j in range(i, D):
                pairs.append((i, j))

        # Sample randomly
        indices = torch.randperm(len(pairs))[:n_pairs]
        sampled = [pairs[i] for i in indices]

        return torch.tensor(sampled, dtype=torch.long)

    def forward(
        self,
        Z_t: torch.Tensor,  # (B, S, D, t) post-activation history
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute synchronization and subsample.

        Returns:
            S_full: (B, S, D, D) full sync matrix
            S_out: (B, S, d_sync_out) subsampled for output
            S_action: (B, S, d_sync_action) subsampled for attention
        """
        B, S, D, T = Z_t.shape

        # Compute S_t = Z_t · Z_t^T
        # Z_t: (B, S, D, T) -> S: (B, S, D, D)
        S_full = torch.matmul(Z_t, Z_t.transpose(-1, -2))  # (B, S, D, D)

        # Normalize by history length for stability
        S_full = S_full / math.sqrt(T)

        # Subsample for output: gather (i,j) pairs
        # S_full[..., i, j] for each pair
        S_out = S_full[:, :, self.out_i, self.out_j]  # (B, S, d_sync_out)

        # Subsample for action
        S_action = S_full[:, :, self.action_i, self.action_j]  # (B, S, d_sync_action)

        return S_full, S_out, S_action


class CTMPrediction(nn.Module):
    """
    CTM (Continuous Thought Machine) Prediction Module.

    Implements the full CTM architecture with:
    - Synapse model producing pre-activations
    - Per-neuron MLPs producing post-activations
    - Synchronization matrix for attention and output
    - Internal tick loop for "thinking"

    The model processes input features over T internal ticks,
    with synchronization driving attention to features and
    producing outputs at each tick.
    """

    def __init__(self, config: CTMPredictionConfig):
        super().__init__()
        self.config = config

        # 1. Project input features to attention KV space
        self.feature_proj = nn.Linear(config.d_model, config.d_model)

        # 2. Synapse model: concat(z_t, o_t) -> a_t
        self.synapse = SynapseModel(
            d_neurons=config.d_neurons,
            d_attention=config.d_model,  # o_t has same dim as features
            d_hidden=config.synapse_hidden,
            dropout=config.dropout,
        )

        # 3. Per-neuron models: A_d^t -> z_d^{t+1}
        self.nlm = NeuronLevelModels(
            d_neurons=config.d_neurons,
            M=config.M,
            hidden_dim=config.nlm_hidden,
            dropout=config.dropout,
        )

        # 4. Synchronization module
        self.sync = SynchronizationModule(
            d_neurons=config.d_neurons,
            d_sync_out=config.d_sync_out,
            d_sync_action=config.d_sync_action,
        )

        # 5. Sync -> attention query projection
        self.W_in = nn.Linear(config.d_sync_action, config.d_model)

        # 6. Cross-attention: q from sync, kv from features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_attention_heads,
            batch_first=True,
            dropout=config.dropout,
        )

        # 7. Sync -> output projection (to d_model for prediction)
        self.W_out = nn.Linear(config.d_sync_out, config.d_model)

        # 8. Prediction readout for different horizons
        self.readout_immediate = nn.Linear(config.d_model, config.d_model)
        self.readout_shortterm = nn.Linear(config.d_model, config.d_model)
        self.readout_longterm = nn.Linear(config.d_model, config.d_model)

        # Initial state projections
        self.init_z = nn.Linear(config.d_model, config.d_neurons)
        self.init_o = nn.Linear(config.d_model, config.d_model)

        # Norms
        self.norm_out = RMSNorm(config.d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        features: torch.Tensor,            # (B, S, d_model) from backbone
        return_all_ticks: bool = False,    # Return outputs at all ticks
    ) -> CTMPredictionOutput:
        """
        Run CTM internal tick loop over features.

        Args:
            features: Input features from backbone (e.g., Janus Pro)
            return_all_ticks: Whether to return outputs at all internal ticks

        Returns:
            CTMPredictionOutput with predictions and internal states
        """
        B, S, D = features.shape
        device = features.device
        T = self.config.T
        M = self.config.M
        d_neurons = self.config.d_neurons

        # Project features for cross-attention KV
        kv_features = self.feature_proj(features)  # (B, S, d_model)

        # Initialize states from features (pooled)
        features_pooled = features.mean(dim=1, keepdim=True).expand(B, S, D)
        z_t = self.init_z(features_pooled)  # (B, S, d_neurons) initial post-activations
        o_t = self.init_o(features_pooled)  # (B, S, d_model) initial attention output

        # Initialize histories
        # Pre-activation history: (B, S, d_neurons, M)
        A_history = torch.zeros(B, S, d_neurons, M, device=device)

        # Post-activation history: (B, S, d_neurons, 0) - grows with ticks
        Z_history = []

        # Store outputs at each tick
        all_outputs = []

        # ===== Internal Tick Loop =====
        for t in range(T):
            # 1. Synapse: produce pre-activations
            a_t = self.synapse(z_t, o_t)  # (B, S, d_neurons)

            # 2. Update pre-activation history (shift and append)
            if t < M:
                A_history[:, :, :, t] = a_t
            else:
                A_history = torch.cat([A_history[:, :, :, 1:], a_t.unsqueeze(-1)], dim=-1)

            # 3. NLM: each neuron processes its history -> post-activations
            # Use available history (pad with zeros if t < M)
            if t < M - 1:
                # Pad history for early ticks
                A_input = A_history.clone()
            else:
                A_input = A_history

            z_t_new = self.nlm(A_input)  # (B, S, d_neurons)

            # 4. Update post-activation history
            Z_history.append(z_t_new)

            # Stack into tensor: (B, S, d_neurons, t+1)
            Z_t = torch.stack(Z_history, dim=-1)

            # 5. Compute synchronization
            S_full, S_out, S_action = self.sync(Z_t)
            # S_out: (B, S, d_sync_out)
            # S_action: (B, S, d_sync_action)

            # 6. Generate attention query from sync
            q_t = self.W_in(S_action)  # (B, S, d_model)

            # 7. Cross-attention to features
            o_t_new, _ = self.cross_attn(q_t, kv_features, kv_features)
            # o_t: (B, S, d_model)

            # 8. Generate output from sync
            y_t = self.W_out(S_out)  # (B, S, d_model)
            y_t = self.norm_out(y_t)

            all_outputs.append(y_t)

            # Update for next tick
            z_t = z_t_new
            o_t = o_t_new

        # ===== Generate final predictions =====
        # Use last tick's output for predictions
        y_final = all_outputs[-1]

        predictions = {
            'immediate': self.readout_immediate(y_final),
            'shortterm': self.readout_shortterm(y_final),
            'longterm': self.readout_longterm(y_final),
        }

        return CTMPredictionOutput(
            predictions=predictions,
            all_outputs=all_outputs if return_all_ticks else [y_final],
            sync_matrix=S_full,
            post_activation_history=Z_t,
        )

    def get_sync(self, features: torch.Tensor) -> torch.Tensor:
        """Get sync signal for compatibility with other modules."""
        output = self.forward(features)
        # Return flattened sync matrix as "sync"
        B, S, D, _ = output.sync_matrix.shape
        return output.sync_matrix.reshape(B, S, -1)[:, :, :self.config.d_neurons]


def create_ctm_prediction(
    d_model: int = 1536,
    d_neurons: int = 512,
    M: int = 16,
    T: int = 8,
    **kwargs,
) -> CTMPrediction:
    """Factory function to create CTM prediction module."""
    config = CTMPredictionConfig(
        d_model=d_model,
        d_neurons=d_neurons,
        M=M,
        T=T,
        **kwargs,
    )
    return CTMPrediction(config)


class CTMLoss(nn.Module):
    """
    CTM Loss Function following the paper.

    The CTM produces outputs at each internal tick t. Instead of only using
    the last tick, we dynamically aggregate from:
    - t_1 = argmin(L) : tick with minimum loss
    - t_2 = argmax(C) : tick with maximum certainty

    Final loss = (L_{t_1} + L_{t_2}) / 2

    For feature prediction (vs classification), we adapt certainty:
    - Paper uses: C_t = 1 - normalized_entropy(logits)
    - We use: C_t = prediction_confidence (based on prediction consistency)
    """

    def __init__(
        self,
        immediate_weight: float = 1.0,
        shortterm_weight: float = 0.5,
        longterm_weight: float = 0.3,
        use_cosine: bool = True,
        use_mse: bool = True,
        mse_weight: float = 0.1,
    ):
        super().__init__()
        self.immediate_weight = immediate_weight
        self.shortterm_weight = shortterm_weight
        self.longterm_weight = longterm_weight
        self.use_cosine = use_cosine
        self.use_mse = use_mse
        self.mse_weight = mse_weight

    def compute_scale_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for a single scale."""
        if not valid.any():
            return torch.tensor(0.0, device=pred.device)

        pred_valid = pred[valid]
        target_valid = target[valid]

        loss = 0.0

        if self.use_cosine:
            cos_sim = F.cosine_similarity(pred_valid, target_valid, dim=-1)
            loss = loss + (1 - cos_sim).mean()

        if self.use_mse:
            pred_norm = F.normalize(pred_valid, dim=-1)
            target_norm = F.normalize(target_valid, dim=-1)
            loss = loss + self.mse_weight * F.mse_loss(pred_norm, target_norm)

        return loss

    def compute_tick_loss(
        self,
        y_t: torch.Tensor,  # (B, S, d_model) output at tick t
        targets: Dict[str, torch.Tensor],
        readout_immediate: nn.Module,
        readout_shortterm: nn.Module,
        readout_longterm: nn.Module,
        return_breakdown: bool = False,
    ) -> torch.Tensor:
        """Compute loss at a single internal tick."""
        # Normalize y_t before readout to ensure all ticks have similar input magnitude.
        # Without this, early ticks (small y_t) dominate learning because:
        # 1. Readout heads learn to work only with small inputs
        # 2. Early ticks always have lowest loss, blocking later tick learning
        y_t_norm = F.normalize(y_t, dim=-1)

        # Generate predictions from NORMALIZED tick output
        pred_immediate = readout_immediate(y_t_norm)
        pred_shortterm = readout_shortterm(y_t_norm)
        pred_longterm = readout_longterm(y_t_norm)

        # Compute loss for each scale
        loss_immediate = self.compute_scale_loss(
            pred_immediate, targets['immediate'], targets['immediate_valid']
        )
        loss_shortterm = self.compute_scale_loss(
            pred_shortterm, targets['shortterm'], targets['shortterm_valid']
        )
        loss_longterm = self.compute_scale_loss(
            pred_longterm, targets['longterm'], targets['longterm_valid']
        )

        # Weighted sum
        total = (
            self.immediate_weight * loss_immediate +
            self.shortterm_weight * loss_shortterm +
            self.longterm_weight * loss_longterm
        )

        if return_breakdown:
            return total, {
                'immediate_loss': loss_immediate,
                'shortterm_loss': loss_shortterm,
                'longterm_loss': loss_longterm,
            }
        return total

    def compute_certainty(
        self,
        y_t: torch.Tensor,  # (B, S, d_model)
        all_outputs_so_far: List[torch.Tensor],  # All outputs up to and including y_t
    ) -> torch.Tensor:
        """
        Compute certainty for feature predictions.

        For classification, certainty = 1 - normalized_entropy.
        For feature prediction, we measure how much the predictions have "settled":
        - Low relative variance across recent ticks = high certainty
        - Low relative change = high certainty (converged)

        Uses RELATIVE metrics (normalized by magnitude) to avoid bias toward
        early ticks when output magnitudes grow over time.
        """
        n_outputs = len(all_outputs_so_far)

        if n_outputs < 2:
            # First tick: low certainty (haven't explored yet)
            return torch.tensor(0.1, device=y_t.device)

        # Use last few ticks to measure stability
        window = min(n_outputs, 4)  # Look at last 4 ticks
        recent = torch.stack(all_outputs_so_far[-window:], dim=0)  # (window, B, S, d_model)

        # Compute RELATIVE variance (coefficient of variation)
        # This normalizes by magnitude so growing outputs don't artificially inflate variance
        mean_output = recent.mean(dim=0)  # (B, S, d_model)
        mean_magnitude = mean_output.abs().mean() + 1e-8
        variance = ((recent - mean_output) ** 2).mean()
        relative_variance = variance / (mean_magnitude ** 2)  # Normalize by squared magnitude

        # Compute RELATIVE change (change / current magnitude)
        # This ensures early ticks (small magnitude) and late ticks are treated fairly
        if n_outputs >= 2:
            y_prev = all_outputs_so_far[-2]
            change = (y_t - y_prev).norm(dim=-1).mean()
            magnitude = y_t.norm(dim=-1).mean() + 1e-8
            relative_change = change / magnitude
        else:
            relative_change = torch.tensor(1.0, device=y_t.device)

        # Combine: low relative variance + low relative change = high certainty
        # Scale factors tuned for relative metrics (typically 0-2 range)
        variance_certainty = torch.exp(-relative_variance * 5.0)
        change_certainty = torch.exp(-relative_change * 3.0)

        certainty = 0.5 * variance_certainty + 0.5 * change_certainty

        return certainty

    def forward(
        self,
        all_outputs: List[torch.Tensor],  # [y_1, y_2, ..., y_T] each (B, S, d_model)
        targets: Dict[str, torch.Tensor],
        readout_immediate: nn.Module,
        readout_shortterm: nn.Module,
        readout_longterm: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute CTM loss across all internal ticks.

        Returns:
            total_loss: Scalar loss
            loss_dict: Breakdown for logging
        """
        T = len(all_outputs)
        device = all_outputs[0].device

        # Compute loss and certainty at each tick
        losses = []
        certainties = []

        for t, y_t in enumerate(all_outputs):
            # Loss at tick t
            L_t = self.compute_tick_loss(
                y_t, targets,
                readout_immediate, readout_shortterm, readout_longterm
            )
            losses.append(L_t)

            # Certainty at tick t (based on all outputs so far)
            outputs_so_far = all_outputs[:t + 1]
            C_t = self.compute_certainty(y_t, outputs_so_far)
            certainties.append(C_t)

        losses = torch.stack(losses)  # (T,)
        certainties = torch.stack(certainties)  # (T,)

        # Find t_1 = argmin(L) and t_2 = argmax(C)
        t1 = losses.argmin()
        t2 = certainties.argmax()

        # Final loss = (L_{t1} + L_{t2}) / 2
        L_t1 = losses[t1]
        L_t2 = losses[t2]
        total_loss = (L_t1 + L_t2) / 2

        # Also compute final tick loss for comparison
        L_final = losses[-1]

        # Get per-scale breakdown for final tick (for logging)
        _, scale_breakdown = self.compute_tick_loss(
            all_outputs[-1], targets,
            readout_immediate, readout_shortterm, readout_longterm,
            return_breakdown=True,
        )

        # Build loss dict for logging
        loss_dict = {
            'loss': total_loss.detach(),
            'loss_t1': L_t1.detach(),
            'loss_t2': L_t2.detach(),
            'loss_final': L_final.detach(),
            't1': t1.detach(),
            't2': t2.detach(),
            'certainty_mean': certainties.mean().detach(),
            'certainty_final': certainties[-1].detach(),
            # Per-scale losses (from final tick)
            'immediate_loss': scale_breakdown['immediate_loss'].detach(),
            'shortterm_loss': scale_breakdown['shortterm_loss'].detach(),
            'longterm_loss': scale_breakdown['longterm_loss'].detach(),
        }

        # Add per-tick losses for analysis
        for t in range(T):
            loss_dict[f'loss_tick_{t}'] = losses[t].detach()

        return total_loss, loss_dict


class CTMLossSimple(nn.Module):
    """
    Simplified CTM loss that works with final predictions dict.

    Uses the paper's min-loss / max-certainty approach but computes
    from the final predictions dictionary (not all tick outputs).

    For training when you don't need full tick-by-tick analysis.
    """

    def __init__(
        self,
        immediate_weight: float = 1.0,
        shortterm_weight: float = 0.5,
        longterm_weight: float = 0.3,
        use_cosine: bool = True,
        use_mse: bool = True,
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
        Compute prediction loss from final predictions.

        This is the simple version that just uses the final tick's predictions.
        For full CTM loss with all ticks, use CTMLoss instead.
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

            pred_valid = pred[valid]
            target_valid = target[valid]

            scale_loss = 0.0

            if self.use_cosine:
                cos_sim = F.cosine_similarity(pred_valid, target_valid, dim=-1)
                cosine_loss = (1 - cos_sim).mean()
                scale_loss = scale_loss + cosine_loss
                loss_dict[f'{scale_name}_cosine'] = cosine_loss.detach()

            if self.use_mse:
                pred_norm = F.normalize(pred_valid, dim=-1)
                target_norm = F.normalize(target_valid, dim=-1)
                mse_loss = F.mse_loss(pred_norm, target_norm)
                scale_loss = scale_loss + self.mse_weight * mse_loss
                loss_dict[f'{scale_name}_mse'] = mse_loss.detach()

            loss_dict[f'{scale_name}_loss'] = scale_loss.detach()
            total_loss = total_loss + weight * scale_loss

        return total_loss, loss_dict
