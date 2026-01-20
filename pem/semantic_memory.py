"""
Semantic Memory Module for PEM.

A simple episodic memory that:
1. Stores representations when surprise is HIGH
2. Retrieves from memory to IMPROVE predictions
3. Uses importance-based eviction when full

Memory is PERSISTENT - saved with checkpoints, never cleared automatically.
Uses registered buffers so state_dict() includes memory contents.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SemanticMemoryConfig:
    """Configuration for semantic memory."""
    num_slots: int = 2048           # Memory capacity
    key_dim: int = 256              # Compressed key dimension
    value_dim: int = 1536           # Value dimension (now matches feature_dim for target features)
    feature_dim: int = 1536         # Input feature dimension
    context_window: int = 8         # Positions on each side for key encoding (total 2*8+1=17)
    retrieval_temperature: float = 0.1  # Softmax temperature for retrieval
    importance_decay: float = 0.8   # Per-batch importance decay (fast to evict stale memories)
    write_threshold: float = 0.1    # Minimum surprise to write to memory


class MemoryReadOutput(NamedTuple):
    """Output from memory read operation."""
    retrieved: torch.Tensor          # (B, S, value_dim) retrieved values
    attention_weights: torch.Tensor  # (B, S, num_slots) attention over memory
    max_attention: torch.Tensor      # (B, S) max attention weight per position


class MemoryTopKOutput(NamedTuple):
    """Output from top-K memory read operation."""
    values: torch.Tensor             # (B, S, K, value_dim) top-K retrieved values
    attention_weights: torch.Tensor  # (B, S, K) attention weights for top-K
    indices: torch.Tensor            # (B, S, K) indices of top-K slots
    max_attention: torch.Tensor      # (B, S) max attention weight per position


class SemanticMemory(nn.Module):
    """
    Simple episodic memory with importance-based management.

    Key insight: Memory is read BEFORE prediction to augment features,
    and written AFTER with importance weighted by surprise.

    Storage:
    - keys: (num_slots, key_dim) - compressed feature keys
    - values: (num_slots, value_dim) - CTM post-activations
    - importance: (num_slots,) - surprise-based importance scores
    - occupied: (num_slots,) - boolean mask of which slots have data

    All storage is registered as buffers for checkpoint persistence.
    """

    def __init__(self, config: SemanticMemoryConfig):
        super().__init__()
        self.config = config

        # Key encoder with local context window
        # Input: features from (2*context_window+1) positions concatenated
        window_size = 2 * config.context_window + 1  # e.g., 17 for context_window=8
        self.key_encoder = nn.Sequential(
            nn.Linear(config.feature_dim * window_size, config.key_dim * 4),
            nn.GELU(),
            nn.Linear(config.key_dim * 4, config.key_dim),
        )

        # Value decoder: retrieved value -> augmentation signal (TRAINED via read path)
        # Now 1536 -> 1536 (values are target features, same dim as input features)
        self.value_decoder = nn.Linear(config.value_dim, config.feature_dim)

        # Register memory as buffers (saved with model state)
        self.register_buffer('keys', torch.zeros(config.num_slots, config.key_dim))
        self.register_buffer('values', torch.zeros(config.num_slots, config.value_dim))
        self.register_buffer('importance', torch.zeros(config.num_slots))
        self.register_buffer('occupied', torch.zeros(config.num_slots, dtype=torch.bool))
        self.register_buffer('write_head', torch.tensor(0, dtype=torch.long))

        # Store windowed features for key refresh (re-encode with updated key_encoder)
        self.register_buffer('original_features',
            torch.zeros(config.num_slots, config.feature_dim * window_size))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def num_occupied(self) -> int:
        """Number of occupied memory slots."""
        return self.occupied.sum().item()

    @property
    def utilization(self) -> float:
        """Fraction of memory slots occupied."""
        return self.num_occupied / self.config.num_slots

    def _extract_window(self, features: torch.Tensor) -> torch.Tensor:
        """Extract local context window around each position.

        Args:
            features: (B, S, D) input features

        Returns:
            windowed: (B, S, D * window_size) features with local context
        """
        B, S, D = features.shape
        w = self.config.context_window

        # Pad sequence on both ends using replication
        padded = F.pad(features, (0, 0, w, w), mode='replicate')  # (B, S+2w, D)

        # Extract windows by gathering slices
        windows = []
        for i in range(2 * w + 1):
            windows.append(padded[:, i:i+S, :])

        return torch.cat(windows, dim=-1)  # (B, S, D * window_size)

    def read(
        self,
        features: torch.Tensor,  # (B, S, feature_dim)
    ) -> MemoryReadOutput:
        """
        Read from memory using soft attention.

        Args:
            features: Input features to use as query

        Returns:
            MemoryReadOutput with retrieved values and attention weights
        """
        B, S, _ = features.shape
        device = features.device

        # If memory is empty, return zeros
        if not self.occupied.any():
            retrieved = torch.zeros(B, S, self.config.feature_dim, device=device)
            attention = torch.zeros(B, S, self.config.num_slots, device=device)
            max_attn = torch.zeros(B, S, device=device)
            return MemoryReadOutput(retrieved, attention, max_attn)

        # Encode query keys WITH context window
        windowed_features = self._extract_window(features)  # (B, S, D * window_size)
        query_keys = self.key_encoder(windowed_features)    # (B, S, key_dim)

        # Get occupied keys and values
        occupied_mask = self.occupied  # (num_slots,)
        memory_keys = self.keys  # (num_slots, key_dim)
        memory_values = self.values  # (num_slots, value_dim)

        # Compute attention scores (dot product similarity)
        # query_keys: (B, S, key_dim)
        # memory_keys: (num_slots, key_dim)
        scores = torch.einsum('bsd,nd->bsn', query_keys, memory_keys)  # (B, S, num_slots)
        scores = scores / (self.config.key_dim ** 0.5)  # Scale by sqrt(dim)

        # Mask out unoccupied slots
        mask = ~occupied_mask.unsqueeze(0).unsqueeze(0).expand(B, S, -1)  # (B, S, num_slots)
        scores = scores.masked_fill(mask, float('-inf'))

        # Softmax with temperature
        attention = F.softmax(scores / self.config.retrieval_temperature, dim=-1)  # (B, S, num_slots)

        # Handle case where all slots are masked (returns uniform zeros)
        attention = torch.nan_to_num(attention, nan=0.0)

        # Retrieve values via soft attention
        # attention: (B, S, num_slots)
        # memory_values: (num_slots, value_dim)
        retrieved_values = torch.einsum('bsn,nv->bsv', attention, memory_values)  # (B, S, value_dim)

        # Decode to feature space
        retrieved = self.value_decoder(retrieved_values)  # (B, S, feature_dim)

        # Max attention for each position
        max_attn = attention.max(dim=-1)[0]  # (B, S)

        return MemoryReadOutput(retrieved, attention, max_attn)

    def read_top_k(
        self,
        features: torch.Tensor,  # (B, S, feature_dim)
        k: int = 4,
    ) -> MemoryTopKOutput:
        """
        Read top-K memories for each position.

        Instead of weighted average, returns the K highest-attention memories
        separately. These can be fed to cross-attention for the model to decide
        how to use them.

        Args:
            features: Input features to use as query
            k: Number of top memories to retrieve per position

        Returns:
            MemoryTopKOutput with top-K values, attention weights, and indices
        """
        B, S, _ = features.shape
        device = features.device

        # If memory is empty, return zeros
        if not self.occupied.any():
            values = torch.zeros(B, S, k, self.config.feature_dim, device=device)
            attention = torch.zeros(B, S, k, device=device)
            indices = torch.zeros(B, S, k, dtype=torch.long, device=device)
            max_attn = torch.zeros(B, S, device=device)
            return MemoryTopKOutput(values, attention, indices, max_attn)

        # Encode query keys WITH context window
        windowed_features = self._extract_window(features)  # (B, S, D * window_size)
        query_keys = self.key_encoder(windowed_features)    # (B, S, key_dim)

        # Get occupied keys and values
        occupied_mask = self.occupied  # (num_slots,)
        memory_keys = self.keys  # (num_slots, key_dim)
        memory_values = self.values  # (num_slots, value_dim)

        # Compute attention scores
        scores = torch.einsum('bsd,nd->bsn', query_keys, memory_keys)  # (B, S, num_slots)
        scores = scores / (self.config.key_dim ** 0.5)

        # Mask out unoccupied slots
        mask = ~occupied_mask.unsqueeze(0).unsqueeze(0).expand(B, S, -1)
        scores = scores.masked_fill(mask, float('-inf'))

        # Softmax with temperature
        attention_full = F.softmax(scores / self.config.retrieval_temperature, dim=-1)
        attention_full = torch.nan_to_num(attention_full, nan=0.0)

        # Get top-K indices and attention weights
        num_occupied = occupied_mask.sum().item()
        actual_k = min(k, num_occupied)

        if actual_k == 0:
            values = torch.zeros(B, S, k, self.config.feature_dim, device=device)
            attention = torch.zeros(B, S, k, device=device)
            indices = torch.zeros(B, S, k, dtype=torch.long, device=device)
            max_attn = torch.zeros(B, S, device=device)
            return MemoryTopKOutput(values, attention, indices, max_attn)

        # Top-K selection
        top_attn, top_indices = attention_full.topk(actual_k, dim=-1)  # (B, S, actual_k)

        # Pad to k if necessary
        if actual_k < k:
            pad_attn = torch.zeros(B, S, k - actual_k, device=device)
            pad_indices = torch.zeros(B, S, k - actual_k, dtype=torch.long, device=device)
            top_attn = torch.cat([top_attn, pad_attn], dim=-1)
            top_indices = torch.cat([top_indices, pad_indices], dim=-1)

        # Gather top-K values
        # memory_values: (num_slots, value_dim)
        # top_indices: (B, S, K)
        top_values = memory_values[top_indices]  # (B, S, K, value_dim)

        # Decode to feature space
        top_values_decoded = self.value_decoder(top_values)  # (B, S, K, feature_dim)

        # Max attention
        max_attn = top_attn[:, :, 0]  # (B, S) - top-1 attention

        return MemoryTopKOutput(top_values_decoded, top_attn, top_indices, max_attn)

    @torch.no_grad()
    def write(
        self,
        features: torch.Tensor,           # (B, S, feature_dim) - context for keys
        values: torch.Tensor,             # (B, S, value_dim) - target features (what came next)
        importance_scores: torch.Tensor,  # (B, S) - surprise magnitude
    ) -> int:
        """
        Write to memory with importance = surprise magnitude.

        Only writes positions where importance > threshold.
        Uses importance-based eviction when memory is full.

        Args:
            features: Features to use as keys (will be windowed for local context)
            values: Target features to store (what actually came next)
            importance_scores: Surprise magnitude (determines write importance)

        Returns:
            Number of entries written
        """
        B, S, _ = features.shape
        device = features.device

        # Extract windowed features for key encoding
        windowed_features = self._extract_window(features)  # (B, S, D * window_size)

        # Flatten batch and sequence dimensions
        windowed_flat = windowed_features.reshape(-1, windowed_features.shape[-1])  # (B*S, D*window_size)
        values_flat = values.reshape(-1, self.config.value_dim)  # (B*S, value_dim)
        importance_flat = importance_scores.reshape(-1)  # (B*S,)

        # Filter by importance threshold
        write_mask = importance_flat > self.config.write_threshold
        if not write_mask.any():
            return 0

        windowed_to_write = windowed_flat[write_mask]  # (N, D * window_size)
        values_to_write = values_flat[write_mask]      # (N, value_dim)
        importance_to_write = importance_flat[write_mask]  # (N,)

        N = windowed_to_write.shape[0]

        # Encode keys from windowed context (same encoder as queries - learns from read path)
        keys_to_write = self.key_encoder(windowed_to_write)  # (N, key_dim)

        # Store raw values directly (no encoding - value_decoder learns to use them)
        # This ensures all trainable params get gradients via the read path

        # Write entries one by one (simple implementation)
        written = 0
        for i in range(N):
            slot = self._get_write_slot(importance_to_write[i].item())

            self.keys[slot] = keys_to_write[i]
            self.values[slot] = values_to_write[i]  # Target features (what came next)
            self.original_features[slot] = windowed_to_write[i]  # Windowed context for key refresh
            self.importance[slot] = importance_to_write[i]
            self.occupied[slot] = True

            written += 1

        return written

    def _get_write_slot(self, new_importance: float) -> int:
        """
        Get slot index for writing.

        Strategy:
        1. If memory not full, use next empty slot
        2. If memory full, evict lowest-importance entry
        """
        # Check for empty slots
        empty_slots = ~self.occupied
        if empty_slots.any():
            # Use first empty slot
            slot = empty_slots.nonzero()[0].item()
            return slot

        # Memory full - evict lowest importance
        min_importance_idx = self.importance.argmin().item()
        min_importance = self.importance[min_importance_idx].item()

        # Only evict if new entry is more important
        if new_importance > min_importance:
            return min_importance_idx
        else:
            # Evict anyway (FIFO fallback)
            slot = self.write_head.item()
            self.write_head = (self.write_head + 1) % self.config.num_slots
            return slot

    @torch.no_grad()
    def decay_importance(self):
        """
        Decay importance of all entries (aging mechanism).

        Called once per batch to age memories.
        """
        if self.occupied.any():
            self.importance = self.importance * self.config.importance_decay

    @torch.no_grad()
    def reset(self):
        """Clear all memory contents."""
        self.keys.zero_()
        self.values.zero_()
        self.importance.zero_()
        self.occupied.fill_(False)
        self.write_head.fill_(0)
        self.original_features.zero_()

    @torch.no_grad()
    def refresh_keys(self):
        """
        Re-encode all stored keys with current key_encoder weights.

        Call periodically during training to update old memories with
        the improved key_encoder. This helps old memories stay retrievable
        as the encoder learns better representations.

        Note: original_features now stores windowed features (D * window_size dim),
        which can be directly fed to the key_encoder.
        """
        if not self.occupied.any():
            return 0

        # Get occupied slot indices
        occupied_indices = self.occupied.nonzero().squeeze(-1)

        # Re-encode keys from windowed original features
        for idx in occupied_indices:
            windowed_feat = self.original_features[idx].unsqueeze(0)  # (1, D * window_size)
            new_key = self.key_encoder(windowed_feat).squeeze(0)  # (key_dim,)
            self.keys[idx] = new_key

        return len(occupied_indices)

    def get_stats(self) -> dict:
        """Get memory statistics for logging."""
        stats = {
            'utilization': self.utilization,
            'num_occupied': self.num_occupied,
        }

        if self.occupied.any():
            occupied_importance = self.importance[self.occupied]
            stats['avg_importance'] = occupied_importance.mean().item()
            stats['max_importance'] = occupied_importance.max().item()
            stats['min_importance'] = occupied_importance.min().item()
        else:
            stats['avg_importance'] = 0.0
            stats['max_importance'] = 0.0
            stats['min_importance'] = 0.0

        return stats


class MemoryIntegrator(nn.Module):
    """
    Integrates retrieved memory content with input features.

    Uses a gated mechanism to blend retrieved values with original features.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        hidden_dim = hidden_dim or feature_dim // 2

        # Gate: decides how much to use memory vs original features
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

        # Transform: combines features and retrieved memory
        self.transform = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        features: torch.Tensor,    # (B, S, feature_dim)
        retrieved: torch.Tensor,   # (B, S, feature_dim)
        memory_strength: Optional[torch.Tensor] = None,  # (B, S) - max attention weight
    ) -> torch.Tensor:
        """
        Integrate retrieved memory with original features.

        Args:
            features: Original input features
            retrieved: Retrieved memory content
            memory_strength: Optional attention strength to modulate gate

        Returns:
            augmented: (B, S, feature_dim) memory-augmented features
        """
        combined = torch.cat([features, retrieved], dim=-1)  # (B, S, feature_dim*2)

        # Compute gate value
        gate = self.gate(combined)  # (B, S, feature_dim)

        # If memory_strength provided, scale gate by it
        # (weaker memory match = less memory influence)
        if memory_strength is not None:
            gate = gate * memory_strength.unsqueeze(-1)  # (B, S, feature_dim)

        # Transform combined representation
        transformed = self.transform(combined)  # (B, S, feature_dim)

        # Gated blend: features + gate * (transformed - features)
        # When gate=0: output = features (no memory influence)
        # When gate=1: output = transformed (full memory influence)
        augmented = features + gate * (transformed - features)

        return augmented


def create_semantic_memory(
    num_slots: int = 2048,
    key_dim: int = 256,
    value_dim: int = 1536,
    feature_dim: int = 1536,
    context_window: int = 8,
    **kwargs,
) -> SemanticMemory:
    """Factory function to create semantic memory."""
    config = SemanticMemoryConfig(
        num_slots=num_slots,
        key_dim=key_dim,
        value_dim=value_dim,
        feature_dim=feature_dim,
        context_window=context_window,
        **kwargs,
    )
    return SemanticMemory(config)


def create_memory_integrator(
    feature_dim: int = 1536,
    hidden_dim: Optional[int] = None,
) -> MemoryIntegrator:
    """Factory function to create memory integrator."""
    return MemoryIntegrator(feature_dim, hidden_dim)
