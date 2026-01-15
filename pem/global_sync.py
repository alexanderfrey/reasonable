"""
Global Sync Module - Cross-module synchronization for PEM.

Follows the CTM paper's "Synchronization as a representation" blueprint:

    S_t = Z_t · (Z_t)^T ∈ R^{D×D}

Each module computes its synchronization matrix from post-activation history,
then cross-module sync is computed by comparing these sync patterns.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         GLOBAL SYNC MODULE                              │
    │                                                                         │
    │   Module Z_histories (post-activation histories):                       │
    │   [Z_pred, Z_surp, ...]  where Z = [z_1, z_2, ..., z_T]                │
    │         │       │                                                       │
    │         ▼       ▼                                                       │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │  Sync Computation: S = Z · Z^T  (per module)                    │  │
    │   │  Then subsample (i,j) pairs to get S_sync                       │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                          │
    │                              ▼                                          │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │  Cross-Module Sync: compare sync patterns between modules       │  │
    │   │  Which modules have similar synchronization dynamics?           │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                          │
    │                              ▼                                          │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │  Sync-Derived Queries: q = W_in · S_action                      │  │
    │   │  Cross-attention between modules using sync as queries          │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                          │
    │                              ▼                                          │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │  Global Sync State                                              │  │
    │   │  (B, S, sync_pairs) - drives attention to features              │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────┘
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalSyncOutput(NamedTuple):
    """Output from GlobalSyncModule."""
    sync: torch.Tensor                 # (B, S, sync_pairs) global sync state
    cross_module_sync: torch.Tensor    # (num_modules, num_modules, B, S) sync matrix
    module_contributions: torch.Tensor # (B, S, num_modules) how much each module contributes


@dataclass
class GlobalSyncConfig:
    """Configuration for Global Sync Module."""

    # Sync computation
    d_sync_space: int = 128        # Subsampled sync pairs per module

    # Cross-module attention
    n_heads: int = 4
    dropout: float = 0.0
    attention_temperature: float = 1.0  # Higher = softer attention (1.0 = standard)

    # Output
    sync_pairs: int = 256          # Output sync dimension

    # Learnable module embeddings
    use_module_embeddings: bool = True


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class ModuleSyncComputer(nn.Module):
    """
    Computes synchronization from a module's post-activation history.

    Following the paper:
        Z_t = [z_1, z_2, ..., z_t] ∈ R^{D×t}
        S_t = Z_t · (Z_t)^T ∈ R^{D×D}
        S_sync = subsample(S_t, n_pairs)
    """

    def __init__(
        self,
        d_neurons: int,
        d_sync_space: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_neurons = d_neurons
        self.d_sync_space = d_sync_space

        # Register fixed random (i,j) pairs for subsampling sync matrix
        # Sample from upper triangle of D×D matrix
        sync_indices = self._sample_pairs(d_neurons, d_sync_space)
        self.register_buffer('sync_i', sync_indices[:, 0])
        self.register_buffer('sync_j', sync_indices[:, 1])

        # Project subsampled sync to common space
        self.sync_proj = nn.Sequential(
            nn.Linear(d_sync_space, d_sync_space),
            nn.GELU(),
            nn.Linear(d_sync_space, d_sync_space),
        )
        self.norm = RMSNorm(d_sync_space)

        self._init_weights()

    def _sample_pairs(self, D: int, n_pairs: int) -> torch.Tensor:
        """Sample n_pairs (i,j) index pairs from upper triangle of DxD matrix."""
        pairs = []
        for i in range(D):
            for j in range(i, D):
                pairs.append((i, j))

        # Randomly sample pairs
        indices = torch.randperm(len(pairs))[:n_pairs]
        sampled = [pairs[idx] for idx in indices]

        return torch.tensor(sampled, dtype=torch.long)

    def _init_weights(self):
        for m in self.sync_proj.modules():
            if isinstance(m, nn.Linear):
                # Xavier init for proper gradient flow
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        Z_history: List[torch.Tensor],  # List of (B, S, D) post-activations per tick
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute synchronization from post-activation history.

        Args:
            Z_history: List of post-activations [z_1, z_2, ..., z_T]
                      Each z_t has shape (B, S, D)

        Returns:
            S_sync: (B, S, d_sync_space) subsampled sync representation
            S_full: (B, S, D, D) full sync matrix (for analysis)
        """
        # Stack: (B, S, D, T)
        Z_t = torch.stack(Z_history, dim=-1)
        B, S, D, T = Z_t.shape

        # Compute sync matrix: S = Z · Z^T / sqrt(T)
        # (B, S, D, T) @ (B, S, T, D) -> (B, S, D, D)
        S_full = torch.matmul(Z_t, Z_t.transpose(-1, -2))
        S_full = S_full / math.sqrt(T)

        # Subsample (i,j) pairs
        S_sync = S_full[:, :, self.sync_i, self.sync_j]  # (B, S, d_sync_space)

        # Project to common space
        S_sync = self.sync_proj(S_sync)
        S_sync = self.norm(S_sync)

        return S_sync, S_full


class SyncCrossModuleAttention(nn.Module):
    """
    Cross-module attention using sync-derived queries.

    Following the paper:
        q_t = W_in · S_action

    Each module's sync representation generates queries that attend to
    other modules' sync representations.
    """

    def __init__(
        self,
        d_sync_space: int,
        num_modules: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_sync_space = d_sync_space
        self.num_modules = num_modules
        self.n_heads = n_heads
        self.head_dim = d_sync_space // n_heads
        self.temperature = temperature

        assert d_sync_space % n_heads == 0

        # Sync -> Query projection (paper: W_in)
        self.sync_to_query = nn.Sequential(
            nn.Linear(d_sync_space, d_sync_space),
            nn.GELU(),
            nn.Linear(d_sync_space, d_sync_space),
        )

        # K, V projections for cross-module attention
        self.k_proj = nn.Linear(d_sync_space, d_sync_space, bias=False)
        self.v_proj = nn.Linear(d_sync_space, d_sync_space, bias=False)
        self.o_proj = nn.Linear(d_sync_space, d_sync_space, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = RMSNorm(d_sync_space)

        self._init_weights()

    def _init_weights(self):
        for m in self.sync_to_query.modules():
            if isinstance(m, nn.Linear):
                # Xavier init for proper gradient flow
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for proj in [self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def forward(
        self,
        module_syncs: torch.Tensor,  # (num_modules, B, S, d_sync_space)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-module attention using sync-derived queries.

        Args:
            module_syncs: Stacked sync representations from all modules

        Returns:
            attended: (num_modules, B, S, d_sync_space) updated sync states
            cross_module_sync: (num_modules, num_modules, B, S) attention weights
        """
        num_modules, B, S, D = module_syncs.shape

        # Reshape: (num_modules, B, S, D) -> (B*S, num_modules, D)
        syncs_flat = module_syncs.permute(1, 2, 0, 3).reshape(B * S, num_modules, D)

        # Generate queries from sync (paper: q_t = W_in · S_action)
        q = self.sync_to_query(syncs_flat)  # (B*S, num_modules, D)

        # K, V from sync representations
        k = self.k_proj(syncs_flat)
        v = self.v_proj(syncs_flat)

        # Reshape for multi-head attention
        q = q.view(B * S, num_modules, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * S, num_modules, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * S, num_modules, self.n_heads, self.head_dim).transpose(1, 2)
        # Now: (B*S, n_heads, num_modules, head_dim)

        # Attention with temperature scaling
        # Higher temperature = softer attention (more uniform)
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores / self.temperature  # Temperature scaling
        attn_weights = F.softmax(scores, dim=-1)  # (B*S, n_heads, num_modules, num_modules)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attended = torch.matmul(attn_weights, v)  # (B*S, n_heads, num_modules, head_dim)
        attended = attended.transpose(1, 2).reshape(B * S, num_modules, D)

        # Output projection with residual
        attended = self.o_proj(attended)
        attended = attended.view(B, S, num_modules, D).permute(2, 0, 1, 3)
        attended = self.norm(attended + module_syncs)  # Residual connection

        # Get cross-module sync matrix (average over heads)
        cross_module_sync = attn_weights.mean(dim=1)  # (B*S, num_modules, num_modules)
        cross_module_sync = cross_module_sync.view(B, S, num_modules, num_modules)
        cross_module_sync = cross_module_sync.permute(2, 3, 0, 1)  # (num_modules, num_modules, B, S)

        return attended, cross_module_sync


class SyncIntegrator(nn.Module):
    """
    Integrate cross-module sync states into global sync output.
    """

    def __init__(
        self,
        d_sync_space: int,
        num_modules: int,
        sync_pairs: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Input: concatenated module sync states
        d_input = d_sync_space * num_modules

        self.integrator = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_input, sync_pairs),
        )

        # Module contribution weights (learned)
        self.module_weights = nn.Linear(d_sync_space, 1)

        self.norm = RMSNorm(sync_pairs)

        self._init_weights()

    def _init_weights(self):
        for m in self.integrator.modules():
            if isinstance(m, nn.Linear):
                # Xavier init for proper gradient flow
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.module_weights.weight)

    def forward(
        self,
        attended_syncs: torch.Tensor,  # (num_modules, B, S, d_sync_space)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate module sync states into global sync.

        Returns:
            sync: (B, S, sync_pairs) global sync state
            contributions: (B, S, num_modules) module contribution weights
        """
        num_modules, B, S, D = attended_syncs.shape

        # Compute module contributions
        contributions = self.module_weights(attended_syncs)  # (num_modules, B, S, 1)
        contributions = F.softmax(contributions.squeeze(-1).permute(1, 2, 0), dim=-1)  # (B, S, num_modules)

        # Concatenate all module sync states
        # (num_modules, B, S, D) -> (B, S, num_modules * D)
        concat = attended_syncs.permute(1, 2, 0, 3).reshape(B, S, -1)

        # Integrate
        sync = self.integrator(concat)  # (B, S, sync_pairs)
        sync = self.norm(sync)

        return sync, contributions


class GlobalSyncModule(nn.Module):
    """
    Global Sync Module - combines post-activation histories from all CTM modules.

    Following the paper's "Synchronization as a representation" blueprint:
    1. Each module provides its Z_history (post-activations at each tick)
    2. Compute S = Z · Z^T for each module (true neural synchronization)
    3. Subsample sync pairs from each module
    4. Use sync-derived queries for cross-module attention
    5. Integrate into global sync state

    Usage:
        global_sync = GlobalSyncModule(config)

        # Register module sync computers (once at init)
        global_sync.register_module('prediction', d_neurons=256)
        global_sync.register_module('surprise', d_neurons=128)

        # Forward pass - now takes Z_history instead of just z_T
        output = global_sync({
            'prediction': pred_ctm.all_tick_activations,  # List[(B, S, 256)]
            'surprise': surp_ctm.all_tick_activations,    # List[(B, S, 128)]
        })

        # Use global sync for attention
        sync = output.sync  # (B, S, sync_pairs)
    """

    def __init__(self, config: GlobalSyncConfig):
        super().__init__()
        self.config = config

        # Module sync computers (added dynamically via register_module)
        self.module_sync_computers = nn.ModuleDict()
        self.module_names: List[str] = []

        # Cross-module attention (created after modules registered)
        self._cross_module_attn = None
        self._sync_integrator = None

        # Module embeddings (learnable identity for each module)
        if config.use_module_embeddings:
            self.module_embeddings = nn.ParameterDict()
        else:
            self.module_embeddings = None

    def register_module(self, name: str, d_neurons: int) -> None:
        """
        Register a CTM module for global sync.

        Args:
            name: Module name (e.g., 'prediction', 'surprise')
            d_neurons: Number of neurons in that module's post-activations
        """
        self.module_sync_computers[name] = ModuleSyncComputer(
            d_neurons=d_neurons,
            d_sync_space=self.config.d_sync_space,
            dropout=self.config.dropout,
        )
        self.module_names.append(name)

        if self.module_embeddings is not None:
            self.module_embeddings[name] = nn.Parameter(
                torch.randn(self.config.d_sync_space) * 0.02
            )

        # Recreate cross-module attention with updated num_modules
        self._create_attention_layers()

    def _create_attention_layers(self):
        """Create/recreate attention layers based on registered modules."""
        num_modules = len(self.module_names)
        if num_modules < 1:
            return

        self._cross_module_attn = SyncCrossModuleAttention(
            d_sync_space=self.config.d_sync_space,
            num_modules=num_modules,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout,
            temperature=self.config.attention_temperature,
        )

        self._sync_integrator = SyncIntegrator(
            d_sync_space=self.config.d_sync_space,
            num_modules=num_modules,
            sync_pairs=self.config.sync_pairs,
            dropout=self.config.dropout,
        )

    def forward(
        self,
        module_activations: Dict[str, List[torch.Tensor]],  # {name: List[(B, S, d_neurons)]}
    ) -> GlobalSyncOutput:
        """
        Compute global sync from module post-activation histories.

        Args:
            module_activations: Dict mapping module name to Z_history
                               (list of post-activations at each tick)

        Returns:
            GlobalSyncOutput with sync state and cross-module analysis
        """
        if len(self.module_names) == 0:
            raise RuntimeError("No modules registered. Call register_module() first.")

        # Verify all modules provided
        for name in self.module_names:
            if name not in module_activations:
                raise ValueError(f"Missing activations for module: {name}")

        # 1. Compute sync representation for each module (S = Z · Z^T, subsampled)
        module_syncs = []
        for name in self.module_names:
            Z_history = module_activations[name]
            sync_computer = self.module_sync_computers[name]

            S_sync, _ = sync_computer(Z_history)  # (B, S, d_sync_space)

            # Add module embedding if enabled
            if self.module_embeddings is not None:
                emb = self.module_embeddings[name]  # (d_sync_space,)
                S_sync = S_sync + emb.unsqueeze(0).unsqueeze(0)

            module_syncs.append(S_sync)

        # 2. Stack: (num_modules, B, S, d_sync_space)
        stacked = torch.stack(module_syncs, dim=0)

        # 3. Cross-module attention using sync-derived queries
        attended, cross_module_sync = self._cross_module_attn(stacked)

        # 4. Integrate into global sync
        sync, contributions = self._sync_integrator(attended)

        return GlobalSyncOutput(
            sync=sync,
            cross_module_sync=cross_module_sync,
            module_contributions=contributions,
        )


def create_global_sync(
    d_sync_space: int = 128,
    sync_pairs: int = 256,
    n_heads: int = 4,
    **kwargs,
) -> GlobalSyncModule:
    """Factory function to create GlobalSyncModule."""
    config = GlobalSyncConfig(
        d_sync_space=d_sync_space,
        sync_pairs=sync_pairs,
        n_heads=n_heads,
        **kwargs,
    )
    return GlobalSyncModule(config)
