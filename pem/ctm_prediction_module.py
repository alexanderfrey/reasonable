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
