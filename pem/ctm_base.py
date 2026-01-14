"""
CTM Base Module - Shared infrastructure for all CTM-based modules.

Every specialized module (Prediction, Surprise, Valence, etc.) uses the same
core CTM architecture:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    CTM Core (shared by all modules)                 │
    │                                                                     │
    │  for t in 1..T:                                                    │
    │      a_t = Synapse(concat(z_t, input_t))     [pre-activations]    │
    │      A_t = update_history(A_{t-1}, a_t)      [history buffer]     │
    │      z_t = NLM(A_t)                          [post-activations]   │
    │      S_t = Sync(Z_t)                         [synchronization]    │
    │                                                                     │
    │  Output: z_T (post-activations), S_T (sync matrix)                 │
    └─────────────────────────────────────────────────────────────────────┘

Each specialized module:
- Has its own input projection
- Shares the core CTM loop
- Has its own output readout
- Exposes post-activations for global sync
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, NamedTuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTMModuleOutput(NamedTuple):
    """Output from any CTM-based module."""
    result: torch.Tensor                # Module-specific output
    post_activations: torch.Tensor      # (B, S, D_neurons) final NLM state for global sync
    sync_matrix: torch.Tensor           # (B, S, D_n, D_n) internal sync matrix
    all_tick_outputs: List[torch.Tensor]  # Outputs at each tick (for CTM loss)
    certainty: torch.Tensor             # (B,) confidence in result
    all_tick_activations: List[torch.Tensor]  # (B, S, D_neurons) NLM activations at each tick


@dataclass
class CTMBaseConfig:
    """Base configuration shared by all CTM modules."""

    # Dimensions
    d_input: int = 1536          # Input feature dimension
    d_neurons: int = 256         # Number of neurons (D in the paper)
    d_output: int = 1536         # Output dimension

    # Sync dimensions
    d_sync_out: int = 128        # Number of sync pairs for output
    d_sync_internal: int = 128   # Number of sync pairs for internal use

    # History lengths
    M: int = 8                   # Pre-activation history length per neuron

    # Internal ticks
    T: int = 4                   # Number of internal thinking steps

    # Architecture
    synapse_hidden: int = 512    # Hidden dim in synapse U-NET
    nlm_hidden: int = 32         # Hidden dim in per-neuron MLPs

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

    a_t = f_θsyn(concat(z_t, input_t))

    Takes concatenation of current post-activations and input,
    produces pre-activations that feed into neuron-level models.
    """

    def __init__(
        self,
        d_neurons: int,
        d_input: int,
        d_hidden: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_neurons = d_neurons
        d_total = d_neurons + d_input

        # U-NET style: down -> bottleneck -> up with skip connections
        self.down1 = nn.Linear(d_total, d_hidden)
        self.down2 = nn.Linear(d_hidden, d_hidden // 2)
        self.bottleneck = nn.Linear(d_hidden // 2, d_hidden // 4)
        self.up1 = nn.Linear(d_hidden // 4 + d_hidden // 2, d_hidden // 2)
        self.up2 = nn.Linear(d_hidden // 2 + d_hidden, d_hidden)
        self.out = nn.Linear(d_hidden, d_neurons)

        self.norm1 = RMSNorm(d_hidden)
        self.norm2 = RMSNorm(d_hidden // 2)
        self.norm3 = RMSNorm(d_hidden // 4)
        self.norm_out = RMSNorm(d_neurons)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(
        self,
        z_t: torch.Tensor,      # (B, S, d_neurons) post-activations
        input_t: torch.Tensor,  # (B, S, d_input) input signal
    ) -> torch.Tensor:
        """
        Produce pre-activations from post-activations and input.

        Returns:
            a_t: (B, S, d_neurons) pre-activations
        """
        x = torch.cat([z_t, input_t], dim=-1)

        # U-NET forward
        h1 = self.act(self.norm1(self.down1(x)))
        h1 = self.dropout(h1)

        h2 = self.act(self.norm2(self.down2(h1)))
        h2 = self.dropout(h2)

        h3 = self.act(self.norm3(self.bottleneck(h2)))

        h4 = self.act(self.up1(torch.cat([h3, h2], dim=-1)))
        h4 = self.dropout(h4)

        h5 = self.act(self.up2(torch.cat([h4, h1], dim=-1)))
        h5 = self.dropout(h5)

        a_t = self.norm_out(self.out(h5))

        return a_t


class NeuronLevelModels(nn.Module):
    """
    Per-neuron MLPs that process pre-activation histories.

    Each neuron d has its own privately parameterized MLP:
    z_d^{t+1} = g_θd(A_d^t)
    """

    def __init__(
        self,
        d_neurons: int,
        M: int,
        hidden_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_neurons = d_neurons
        self.M = M
        self.hidden_dim = hidden_dim

        # Each neuron has its own MLP: M -> hidden -> 1
        self.w1 = nn.Parameter(torch.randn(d_neurons, M, hidden_dim) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(d_neurons, hidden_dim))

        self.w2 = nn.Parameter(torch.randn(d_neurons, hidden_dim, 1) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(d_neurons, 1))

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, A_t: torch.Tensor) -> torch.Tensor:
        """
        Each neuron processes its own history to produce post-activation.

        Args:
            A_t: (B, S, D, M) pre-activation history per neuron

        Returns:
            z_t: (B, S, D) post-activations
        """
        B, S, D, M = A_t.shape
        assert D == self.d_neurons and M == self.M

        A_flat = A_t.reshape(B * S, D, M)

        # Per-neuron linear
        h = torch.einsum('bdm,dmh->bdh', A_flat, self.w1) + self.b1
        h = self.act(h)
        h = self.dropout(h)

        z = torch.einsum('bdh,dho->bdo', h, self.w2) + self.b2
        z = z.squeeze(-1)

        z_t = z.reshape(B, S, D)

        return z_t


class SynchronizationModule(nn.Module):
    """
    Computes neural synchronization from post-activation history.

    S_t = Z_t · (Z_t)^T ∈ R^{D×D}

    Then subsamples (i,j) pairs to get S_out and S_internal.
    """

    def __init__(
        self,
        d_neurons: int,
        d_sync_out: int,
        d_sync_internal: int,
    ):
        super().__init__()
        self.d_neurons = d_neurons
        self.d_sync_out = d_sync_out
        self.d_sync_internal = d_sync_internal

        # Register fixed random (i,j) pairs for subsampling
        out_indices = self._sample_pairs(d_neurons, d_sync_out)
        self.register_buffer('out_i', out_indices[:, 0])
        self.register_buffer('out_j', out_indices[:, 1])

        internal_indices = self._sample_pairs(d_neurons, d_sync_internal)
        self.register_buffer('internal_i', internal_indices[:, 0])
        self.register_buffer('internal_j', internal_indices[:, 1])

    def _sample_pairs(self, D: int, n_pairs: int) -> torch.Tensor:
        """Sample n_pairs (i,j) index pairs from upper triangle of DxD matrix."""
        pairs = []
        for i in range(D):
            for j in range(i, D):
                pairs.append((i, j))

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
            S_internal: (B, S, d_sync_internal) subsampled for internal use
        """
        B, S, D, T = Z_t.shape

        S_full = torch.matmul(Z_t, Z_t.transpose(-1, -2))
        S_full = S_full / math.sqrt(T)

        S_out = S_full[:, :, self.out_i, self.out_j]
        S_internal = S_full[:, :, self.internal_i, self.internal_j]

        return S_full, S_out, S_internal


class CTMCore(nn.Module):
    """
    Core CTM loop shared by all specialized modules.

    This encapsulates the Synapse -> NLM -> Sync loop that runs for T ticks.
    Specialized modules wrap this with their own input/output projections.
    """

    def __init__(self, config: CTMBaseConfig):
        super().__init__()
        self.config = config

        # Core components
        self.synapse = SynapseModel(
            d_neurons=config.d_neurons,
            d_input=config.d_input,
            d_hidden=config.synapse_hidden,
            dropout=config.dropout,
        )

        self.nlm = NeuronLevelModels(
            d_neurons=config.d_neurons,
            M=config.M,
            hidden_dim=config.nlm_hidden,
            dropout=config.dropout,
        )

        self.sync = SynchronizationModule(
            d_neurons=config.d_neurons,
            d_sync_out=config.d_sync_out,
            d_sync_internal=config.d_sync_internal,
        )

        # Sync -> output projection
        self.sync_to_output = nn.Linear(config.d_sync_out, config.d_output)

        # Initial state
        self.init_z = nn.Linear(config.d_input, config.d_neurons)

        self.norm_out = RMSNorm(config.d_output)

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
        input_features: torch.Tensor,  # (B, S, d_input)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Run the CTM core loop.

        Args:
            input_features: Input to process

        Returns:
            post_activations: (B, S, d_neurons) final post-activations
            sync_matrix: (B, S, d_neurons, d_neurons) final sync matrix
            output: (B, S, d_output) final tick output
            all_outputs: List of outputs at each tick
            all_activations: List of NLM post-activations at each tick
        """
        B, S, D = input_features.shape
        device = input_features.device
        T = self.config.T
        M = self.config.M
        d_neurons = self.config.d_neurons

        # Initialize post-activations from input
        z_t = self.init_z(input_features)  # (B, S, d_neurons)

        # Pre-activation history as list (avoids in-place ops)
        A_history_list: List[torch.Tensor] = []

        # Post-activation history (grows with ticks)
        Z_history = []

        # Outputs at each tick
        all_outputs = []

        # ===== CTM Tick Loop =====
        for t in range(T):
            # 1. Synapse: produce pre-activations
            a_t = self.synapse(z_t, input_features)

            # 2. Update pre-activation history (no in-place ops)
            A_history_list.append(a_t)
            if len(A_history_list) > M:
                A_history_list = A_history_list[-M:]

            # Stack and pad to M if needed
            if len(A_history_list) < M:
                # Pad with zeros at the beginning
                padding = [torch.zeros_like(a_t) for _ in range(M - len(A_history_list))]
                A_history = torch.stack(padding + A_history_list, dim=-1)
            else:
                A_history = torch.stack(A_history_list, dim=-1)

            # 3. NLM: process history -> post-activations
            z_t_new = self.nlm(A_history)

            # 4. Update post-activation history
            Z_history.append(z_t_new)
            Z_t = torch.stack(Z_history, dim=-1)

            # 5. Compute synchronization
            S_full, S_out, S_internal = self.sync(Z_t)

            # 6. Generate output from sync
            y_t = self.sync_to_output(S_out)
            y_t = self.norm_out(y_t)

            all_outputs.append(y_t)

            # Update for next tick
            z_t = z_t_new

        return z_t, S_full, all_outputs[-1], all_outputs, Z_history

    def compute_certainty(
        self,
        all_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute certainty based on output stability across ticks.
        """
        n_outputs = len(all_outputs)

        if n_outputs < 2:
            return torch.tensor(0.1, device=all_outputs[0].device)

        window = min(n_outputs, 4)
        recent = torch.stack(all_outputs[-window:], dim=0)

        mean_output = recent.mean(dim=0)
        variance = ((recent - mean_output) ** 2).mean()

        if n_outputs >= 2:
            change = (all_outputs[-1] - all_outputs[-2]).norm(dim=-1).mean()
        else:
            change = torch.tensor(1.0, device=all_outputs[0].device)

        variance_certainty = torch.exp(-variance * 10.0)
        change_certainty = torch.exp(-change * 5.0)

        return 0.5 * variance_certainty + 0.5 * change_certainty


class CTMModule(nn.Module):
    """
    Base class for all CTM-based specialized modules.

    Subclasses should:
    1. Override input_projection() to preprocess their specific inputs
    2. Override output_projection() to produce their specific outputs
    3. Call super().__init__() with appropriate config

    The core CTM loop (Synapse -> NLM -> Sync) is shared.
    """

    def __init__(self, config: CTMBaseConfig):
        super().__init__()
        self.config = config
        self.core = CTMCore(config)

    def input_projection(self, *args, **kwargs) -> torch.Tensor:
        """
        Project module-specific inputs to d_input space.

        Override in subclass.
        """
        raise NotImplementedError("Subclass must implement input_projection")

    def output_projection(self, core_output: torch.Tensor) -> torch.Tensor:
        """
        Project core output to module-specific output.

        Override in subclass.
        """
        raise NotImplementedError("Subclass must implement output_projection")

    def forward(self, *args, **kwargs) -> CTMModuleOutput:
        """
        Run the CTM module.

        1. Project inputs
        2. Run core CTM loop
        3. Project outputs
        4. Return CTMModuleOutput with post-activations for global sync
        """
        # 1. Input projection (module-specific)
        input_features = self.input_projection(*args, **kwargs)

        # 2. Run core CTM loop
        post_activations, sync_matrix, output, all_outputs, all_activations = self.core(input_features)

        # 3. Output projection (module-specific)
        result = self.output_projection(output)

        # 4. Compute certainty
        certainty = self.core.compute_certainty(all_outputs)

        return CTMModuleOutput(
            result=result,
            post_activations=post_activations,
            sync_matrix=sync_matrix,
            all_tick_outputs=all_outputs,
            certainty=certainty,
            all_tick_activations=all_activations,
        )
