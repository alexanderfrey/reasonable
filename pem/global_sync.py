"""
Global Sync Module - Cross-module synchronization for PEM.

Combines post-activations from all CTM-based modules into a global sync state.
This is the "global workspace" where all specialized processors meet.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         GLOBAL SYNC MODULE                              │
    │                                                                         │
    │   Module Post-Activations:                                              │
    │   [h_pred, h_surp, h_val, h_cur, h_act, h_imag, ...]                   │
    │         │      │      │      │      │      │                           │
    │         ▼      ▼      ▼      ▼      ▼      ▼                           │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │  Module Projections (each module -> common sync space)          │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                          │
    │                              ▼                                          │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │  Cross-Module Sync Matrix: H @ H.T                              │  │
    │   │  Which modules are "in sync"?                                   │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                          │
    │                              ▼                                          │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │  Cross-Module Attention                                         │  │
    │   │  Modules attend to each other based on sync                     │  │
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

    # Module configuration
    num_modules: int = 2           # Number of CTM modules (start with 2: pred, surprise)
    d_neurons: int = 256           # Neurons per module (can vary per module)
    d_sync_space: int = 128        # Common sync space dimension

    # Cross-module attention
    n_heads: int = 4
    dropout: float = 0.0

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


class ModuleProjection(nn.Module):
    """
    Project a single module's post-activations to common sync space.

    Each module may have different d_neurons, so we need separate projections.
    """

    def __init__(
        self,
        d_neurons: int,
        d_sync_space: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_neurons, d_sync_space),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_sync_space, d_sync_space),
        )
        self.norm = RMSNorm(d_sync_space)
        self._init_weights()

    def _init_weights(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: (B, S, d_neurons) module post-activations

        Returns:
            projected: (B, S, d_sync_space)
        """
        return self.norm(self.proj(activations))


class CrossModuleAttention(nn.Module):
    """
    Attention between modules in the global sync space.

    Each module attends to other modules based on sync patterns.
    This implements the "global workspace" where modules share information.
    """

    def __init__(
        self,
        d_sync_space: int,
        num_modules: int,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_sync_space = d_sync_space
        self.num_modules = num_modules
        self.n_heads = n_heads
        self.head_dim = d_sync_space // n_heads

        self.q_proj = nn.Linear(d_sync_space, d_sync_space, bias=False)
        self.k_proj = nn.Linear(d_sync_space, d_sync_space, bias=False)
        self.v_proj = nn.Linear(d_sync_space, d_sync_space, bias=False)
        self.o_proj = nn.Linear(d_sync_space, d_sync_space, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(proj.weight, std=0.02)

    def forward(
        self,
        module_states: torch.Tensor,  # (num_modules, B, S, d_sync_space)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-module attention.

        Args:
            module_states: Stacked projected module activations

        Returns:
            attended: (num_modules, B, S, d_sync_space) updated module states
            attn_weights: (num_modules, num_modules, B, S) cross-module attention
        """
        num_modules, B, S, D = module_states.shape

        # Reshape: (num_modules, B, S, D) -> (B*S, num_modules, D)
        states_flat = module_states.permute(1, 2, 0, 3).reshape(B * S, num_modules, D)

        # Project to Q, K, V
        q = self.q_proj(states_flat)  # (B*S, num_modules, D)
        k = self.k_proj(states_flat)
        v = self.v_proj(states_flat)

        # Reshape for multi-head attention
        q = q.view(B * S, num_modules, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * S, num_modules, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * S, num_modules, self.n_heads, self.head_dim).transpose(1, 2)
        # Now: (B*S, n_heads, num_modules, head_dim)

        # Attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)  # (B*S, n_heads, num_modules, num_modules)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attended = torch.matmul(attn_weights, v)  # (B*S, n_heads, num_modules, head_dim)
        attended = attended.transpose(1, 2).reshape(B * S, num_modules, D)

        # Output projection
        attended = self.o_proj(attended)  # (B*S, num_modules, D)

        # Reshape back: (B*S, num_modules, D) -> (num_modules, B, S, D)
        attended = attended.view(B, S, num_modules, D).permute(2, 0, 1, 3)

        # Get cross-module sync matrix (average over heads)
        # attn_weights: (B*S, n_heads, num_modules, num_modules)
        cross_module_sync = attn_weights.mean(dim=1)  # (B*S, num_modules, num_modules)
        cross_module_sync = cross_module_sync.view(B, S, num_modules, num_modules)
        cross_module_sync = cross_module_sync.permute(2, 3, 0, 1)  # (num_modules, num_modules, B, S)

        return attended, cross_module_sync


class SyncIntegrator(nn.Module):
    """
    Integrate cross-module attended states into global sync output.
    """

    def __init__(
        self,
        d_sync_space: int,
        num_modules: int,
        sync_pairs: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Input: concatenated module states
        d_input = d_sync_space * num_modules

        self.integrator = nn.Sequential(
            nn.Linear(d_input, d_input // 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_input // 2, sync_pairs),
        )

        # Module contribution weights (learned)
        self.module_weights = nn.Linear(d_sync_space, 1)

        self.norm = RMSNorm(sync_pairs)

        self._init_weights()

    def _init_weights(self):
        for m in self.integrator.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.module_weights.weight, std=0.02)

    def forward(
        self,
        attended_states: torch.Tensor,  # (num_modules, B, S, d_sync_space)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate module states into global sync.

        Returns:
            sync: (B, S, sync_pairs) global sync state
            contributions: (B, S, num_modules) module contribution weights
        """
        num_modules, B, S, D = attended_states.shape

        # Compute module contributions
        contributions = self.module_weights(attended_states)  # (num_modules, B, S, 1)
        contributions = F.softmax(contributions.squeeze(-1).permute(1, 2, 0), dim=-1)  # (B, S, num_modules)

        # Concatenate all module states
        # (num_modules, B, S, D) -> (B, S, num_modules * D)
        concat = attended_states.permute(1, 2, 0, 3).reshape(B, S, -1)

        # Integrate
        sync = self.integrator(concat)  # (B, S, sync_pairs)
        sync = self.norm(sync)

        return sync, contributions


class GlobalSyncModule(nn.Module):
    """
    Global Sync Module - combines post-activations from all CTM modules.

    This is the "global workspace" where specialized processors meet:
    - Each module contributes its post-activations
    - Cross-module attention determines which modules are in sync
    - The global sync state drives attention to features

    Usage:
        global_sync = GlobalSyncModule(config)

        # Register module projections (once at init)
        global_sync.register_module('prediction', d_neurons=256)
        global_sync.register_module('surprise', d_neurons=128)

        # Forward pass
        output = global_sync({
            'prediction': pred_ctm.post_activations,
            'surprise': surp_ctm.post_activations,
        })

        # Use global sync for attention
        sync = output.sync  # (B, S, sync_pairs)
    """

    def __init__(self, config: GlobalSyncConfig):
        super().__init__()
        self.config = config

        # Module projections (added dynamically via register_module)
        self.module_projections = nn.ModuleDict()
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
        self.module_projections[name] = ModuleProjection(
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

        self._cross_module_attn = CrossModuleAttention(
            d_sync_space=self.config.d_sync_space,
            num_modules=num_modules,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout,
        )

        self._sync_integrator = SyncIntegrator(
            d_sync_space=self.config.d_sync_space,
            num_modules=num_modules,
            sync_pairs=self.config.sync_pairs,
            dropout=self.config.dropout,
        )

    def forward(
        self,
        module_activations: Dict[str, torch.Tensor],  # {name: (B, S, d_neurons)}
    ) -> GlobalSyncOutput:
        """
        Compute global sync from module post-activations.

        Args:
            module_activations: Dict mapping module name to post-activations

        Returns:
            GlobalSyncOutput with sync state and cross-module analysis
        """
        if len(self.module_names) == 0:
            raise RuntimeError("No modules registered. Call register_module() first.")

        # Verify all modules provided
        for name in self.module_names:
            if name not in module_activations:
                raise ValueError(f"Missing activations for module: {name}")

        # 1. Project each module to common sync space
        projected = []
        for name in self.module_names:
            act = module_activations[name]
            proj = self.module_projections[name](act)  # (B, S, d_sync_space)

            # Add module embedding if enabled
            if self.module_embeddings is not None:
                emb = self.module_embeddings[name]  # (d_sync_space,)
                proj = proj + emb.unsqueeze(0).unsqueeze(0)

            projected.append(proj)

        # 2. Stack: (num_modules, B, S, d_sync_space)
        stacked = torch.stack(projected, dim=0)

        # 3. Cross-module attention
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
