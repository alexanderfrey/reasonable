"""
Sync Module for PEM.

Memory-augmented, personality-colored synchronization module that replaces
the correlation-based EnhancedSynchronizationModule.

Core equation:
    sync = f(personality, context_state, memory_state)

Components:
    - ContextEncoder: Encodes history into context representation
    - ChangePointDetector: Detects semantic boundaries via embedding similarity
    - MemoryBank: Stores chunks with importance-weighted eviction
    - PersonalityModule: Learned objective embedding
    - SyncIntegrator: Combines all signals into sync output
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SyncModuleOutput(NamedTuple):
    """Output from SyncModule forward pass."""
    sync: torch.Tensor           # (B, S, sync_pairs) - main sync output
    kl_loss: torch.Tensor        # Scalar - KL divergence from personality prior
    boundary_probs: torch.Tensor # (B, S) - detected semantic boundaries


@dataclass
class SyncModuleConfig:
    """Configuration for the sync module."""

    # Core dimensions
    d_model: int = 512          # Input feature dimension
    sync_pairs: int = 512       # Output sync dimension
    n_heads: int = 8            # Attention heads

    # Memory bank
    memory_slots: int = 100     # Number of memory slots
    memory_dim: Optional[int] = None  # Memory embedding dimension (defaults to d_model)

    # Change-point detection
    cpd_window_size: int = 8    # Rolling window size for similarity
    cpd_threshold: float = 0.3  # Cosine similarity drop threshold
    cpd_min_chunk_size: int = 4 # Minimum tokens between boundaries

    # Personality
    personality_dim: Optional[int] = None  # Personality embedding dimension (defaults to d_model)
    kl_temperature: float = 1.0  # Temperature for KL divergence (lower = stronger constraint)

    # Intention (oscillating goal-directed drive)
    use_intention: bool = False           # Enable intention module
    intention_oscillators: int = 16       # Number of frequency components
    intention_min_period: int = 8         # Minimum oscillation period (~phrase)
    intention_max_period: int = 2048      # Maximum oscillation period (~document)
    intention_context_gating: bool = True # Gate oscillators by context relevance
    intention_modulate_kl: bool = True    # Intention affects KL constraint strength
    intention_kl_range: Tuple[float, float] = (0.5, 2.0)  # KL temp multiplier range

    # General
    dropout: float = 0.0

    def __post_init__(self):
        if self.memory_dim is None:
            self.memory_dim = self.d_model
        if self.personality_dim is None:
            self.personality_dim = self.d_model


@dataclass
class IntentionConfig:
    """Configuration for the intention module (oscillating goal-directed drive)."""

    d_model: int = 512                    # Feature dimension
    num_oscillators: int = 16             # Number of frequency components
    min_period: int = 8                   # Minimum oscillation period (tokens)
    max_period: int = 2048                # Maximum oscillation period (tokens)
    use_context_gating: bool = True       # Gate oscillators by context relevance
    modulate_kl: bool = True              # Intention affects KL constraint strength
    kl_modulation_range: Tuple[float, float] = (0.5, 2.0)  # (min, max) KL temp multiplier


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class ContextEncoder(nn.Module):
    """
    Encode history (B, S, T, D) -> context (B, S, D).

    Uses learned temporal attention to aggregate history into a single
    context vector per position.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Learned query for temporal aggregation
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Key/value projections for history
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (B, S, T, D) neural activation history

        Returns:
            context: (B, S, D) aggregated context per position
        """
        B, S, T, D = history.shape

        # Reshape: treat each position independently
        h = history.view(B * S, T, D)  # (B*S, T, D)

        # Project to keys and values
        k = self.k_proj(h)  # (B*S, T, D)
        v = self.v_proj(h)  # (B*S, T, D)

        # Expand query for all positions
        q = self.query.expand(B * S, 1, D)  # (B*S, 1, D)

        # Reshape for multi-head attention
        q = q.view(B * S, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * S, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * S, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B*S, H, 1, T)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Aggregate
        out = torch.matmul(attn, v)  # (B*S, H, 1, head_dim)
        out = out.transpose(1, 2).reshape(B * S, 1, D)  # (B*S, 1, D)
        out = self.o_proj(out)

        # Reshape back
        context = out.view(B, S, D)  # (B, S, D)

        return self.norm(context)


class ChangePointDetector(nn.Module):
    """
    Detect semantic boundaries via embedding-based change-point detection.

    Algorithm:
    1. Compute rolling window embeddings (mean of past N tokens)
    2. Compute cosine similarity between adjacent windows
    3. When similarity drops below threshold -> potential boundary
    4. Learned MLP refines raw detections

    Output:
    - boundary_probs: (B, S) probability of boundary at each position
    - chunk_ids: (B, S) integer chunk assignment per position
    """

    def __init__(
        self,
        d_model: int,
        window_size: int = 8,
        threshold: float = 0.3,
        min_chunk_size: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size

        # Window projection (optional learned transformation)
        self.window_proj = nn.Linear(d_model, d_model, bias=False)

        # Boundary refinement MLP
        # Input: [prev_window, curr_window, raw_sim_score]
        self.boundary_refiner = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.eye_(self.window_proj.weight)  # Start as identity
        for module in self.boundary_refiner.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _compute_window_embeddings(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute rolling mean window embeddings using cumsum trick.

        Args:
            features: (B, S, D)

        Returns:
            window_embeds: (B, S, D) where position t is mean of features[max(0,t-window_size):t+1]
        """
        B, S, D = features.shape

        # Cumulative sum for efficient rolling mean
        cumsum = torch.cumsum(features, dim=1)  # (B, S, D)

        # Pad for indexing
        cumsum_padded = F.pad(cumsum, (0, 0, self.window_size, 0))  # (B, S+window_size, D)

        # Rolling sum: cumsum[t] - cumsum[t - window_size]
        # But clamp indices to valid range
        window_sums = cumsum_padded[:, self.window_size:, :] - cumsum_padded[:, :-self.window_size, :]

        # Actual window sizes (handles beginning of sequence)
        positions = torch.arange(S, device=features.device).float() + 1
        actual_window_sizes = torch.clamp(positions, max=self.window_size)  # (S,)
        actual_window_sizes = actual_window_sizes.view(1, S, 1)  # (1, S, 1)

        # Mean
        window_embeds = window_sums / actual_window_sizes

        return self.window_proj(window_embeds)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, S, D) encoded features

        Returns:
            boundary_probs: (B, S) probability of boundary at each position
            chunk_ids: (B, S) integer chunk assignment per position
        """
        B, S, D = features.shape
        device = features.device

        # 1. Compute rolling window embeddings
        window_embeds = self._compute_window_embeddings(features)  # (B, S, D)

        # 2. Cosine similarity between adjacent windows
        # sim[t] = cos_sim(window[t-1], window[t])
        prev_windows = F.pad(window_embeds[:, :-1, :], (0, 0, 1, 0))  # (B, S, D)
        cos_sims = F.cosine_similarity(prev_windows, window_embeds, dim=-1)  # (B, S)

        # First position has no predecessor, set similarity to 1 (no boundary)
        cos_sims[:, 0] = 1.0

        # 3. Raw boundary detection: where similarity drops below threshold
        raw_boundary_signal = (1 - cos_sims).unsqueeze(-1)  # (B, S, 1) higher = more likely boundary

        # 4. Learned refinement
        refine_input = torch.cat([
            prev_windows,
            window_embeds,
            raw_boundary_signal,
        ], dim=-1)  # (B, S, 2D + 1)

        boundary_probs = self.boundary_refiner(refine_input).squeeze(-1)  # (B, S)

        # 5. Convert to chunk IDs
        # Apply minimum chunk size constraint using a soft version
        chunk_ids = self._probs_to_chunk_ids(boundary_probs)  # (B, S)

        return boundary_probs, chunk_ids

    def _probs_to_chunk_ids(self, boundary_probs: torch.Tensor) -> torch.Tensor:
        """
        Convert boundary probabilities to hard chunk assignments.

        Uses cumulative sum of thresholded boundaries to assign chunk IDs.
        """
        B, S = boundary_probs.shape

        # Threshold to get hard boundaries (during inference)
        # During training, use soft boundaries via straight-through estimator
        if self.training:
            # Straight-through: hard forward, soft backward
            hard_boundaries = (boundary_probs > 0.5).float()
            hard_boundaries = hard_boundaries - boundary_probs.detach() + boundary_probs
        else:
            hard_boundaries = (boundary_probs > 0.5).float()

        # First position is always start of chunk 0
        hard_boundaries[:, 0] = 0

        # Enforce minimum chunk size by zeroing boundaries too close together
        # This is a simple heuristic; could be made learnable
        if self.min_chunk_size > 1:
            for i in range(1, self.min_chunk_size):
                if i < S:
                    # Zero out boundaries within min_chunk_size of previous boundary
                    prev_boundary_mask = F.pad(hard_boundaries[:, :-i], (i, 0))
                    hard_boundaries = hard_boundaries * (1 - prev_boundary_mask * 0.5)

        # Cumsum to get chunk IDs
        chunk_ids = torch.cumsum(hard_boundaries, dim=1).long()  # (B, S)

        return chunk_ids


class MemoryCrossAttention(nn.Module):
    """Cross-attention for memory retrieval."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(proj.weight, std=0.02)

    def forward(
        self,
        query: torch.Tensor,    # (B, S, D)
        key: torch.Tensor,      # (B, M, D) memory
        value: torch.Tensor,    # (B, M, D) memory
    ) -> torch.Tensor:
        """Cross-attend from query to memory."""
        B, S, D = query.shape
        M = key.shape[1]

        q = self.q_proj(query).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, S, D)

        return self.o_proj(out)


class MemoryBank(nn.Module):
    """
    Fixed-size memory bank storing semantic chunk embeddings.

    Features:
    - Stores embeddings of semantic chunks (not individual tokens)
    - Importance-weighted eviction: surprising chunks persist longer
    - Cross-attention retrieval based on current context
    """

    def __init__(
        self,
        num_slots: int = 100,
        memory_dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.memory_dim = memory_dim

        # Memory storage (registered as buffers for persistence)
        self.register_buffer('memory', torch.zeros(num_slots, memory_dim))
        self.register_buffer('importance', torch.zeros(num_slots))
        self.register_buffer('age', torch.zeros(num_slots))
        self.register_buffer('occupied', torch.zeros(num_slots, dtype=torch.bool))
        self.register_buffer('write_ptr', torch.tensor(0))

        # Chunk encoder: aggregates tokens within a chunk
        self.chunk_encoder = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, memory_dim),
        )

        # Cross-attention for retrieval
        self.retrieval_attn = MemoryCrossAttention(
            d_model=memory_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Importance estimation (maps surprise to importance)
        self.importance_scale = nn.Parameter(torch.tensor(1.0))
        self.importance_bias = nn.Parameter(torch.tensor(0.5))

        self._init_weights()

    def _init_weights(self):
        for module in self.chunk_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _find_eviction_slot(self, new_importance: float) -> int:
        """
        Find a slot for new memory entry using importance-weighted eviction.

        Priority:
        1. Empty slot (not occupied)
        2. Lowest effective_importance slot

        effective_importance = importance * decay(age)
        High importance decays slower (surprising memories persist).
        """
        # Check for empty slots
        if not self.occupied.all():
            empty_idx = (~self.occupied).nonzero(as_tuple=True)[0][0].item()
            return empty_idx

        # Compute effective importance with age decay
        # Higher importance = slower decay
        decay_rate = 0.1 / (self.importance + 0.1)  # (num_slots,)
        effective_importance = self.importance * torch.exp(-decay_rate * self.age)

        # Find slot with lowest effective importance
        min_idx = effective_importance.argmin().item()

        return min_idx

    def write(
        self,
        features: torch.Tensor,      # (B, S, D) features to potentially store
        chunk_ids: torch.Tensor,     # (B, S) chunk assignment per position
        surprise: Optional[torch.Tensor] = None,  # (B, S, 1) or (B, S) surprise magnitude
    ) -> None:
        """
        Write new chunks to memory with importance-weighted eviction.

        For each unique chunk in the batch:
        1. Compute chunk embedding (mean of tokens in chunk)
        2. Compute importance from surprise
        3. Store in available or evicted slot
        """
        B, S, D = features.shape
        device = features.device

        # Handle surprise shape
        if surprise is not None:
            if surprise.dim() == 3:
                surprise = surprise.squeeze(-1)  # (B, S)

        # Process each batch item
        for b in range(B):
            unique_chunks = chunk_ids[b].unique()

            for chunk_id in unique_chunks:
                chunk_id_val = chunk_id.item()
                if chunk_id_val < 0:  # Invalid chunk marker
                    continue

                # Get tokens belonging to this chunk
                mask = (chunk_ids[b] == chunk_id)
                if not mask.any():
                    continue

                chunk_features = features[b, mask]  # (num_tokens, D)

                # Encode chunk as single embedding
                chunk_mean = chunk_features.mean(dim=0, keepdim=True)  # (1, D)
                chunk_embed = self.chunk_encoder(chunk_mean).squeeze(0)  # (D,)

                # Compute importance from surprise
                if surprise is not None:
                    chunk_surprise = surprise[b, mask].mean().item()
                    importance = self.importance_scale * chunk_surprise + self.importance_bias
                    importance = max(0.0, min(1.0, importance.item()))  # Clamp to [0, 1]
                else:
                    importance = 0.5  # Default importance

                # Find slot to write to
                slot_idx = self._find_eviction_slot(importance)

                # Write to memory
                self.memory[slot_idx] = chunk_embed.detach()
                self.importance[slot_idx] = importance
                self.age[slot_idx] = 0
                self.occupied[slot_idx] = True

        # Age all occupied slots
        self.age[self.occupied] += 1

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve from memory via cross-attention.

        Args:
            query: (B, S, D) current context as query

        Returns:
            retrieved: (B, S, D) memory-augmented representation
        """
        B, S, D = query.shape

        # Check if memory has any occupied slots
        if not self.occupied.any():
            return torch.zeros_like(query)

        # Get occupied memory slots
        occupied_mask = self.occupied
        memory_kv = self.memory[occupied_mask]  # (num_occupied, D)

        # Expand for batch
        num_occupied = memory_kv.shape[0]
        memory_kv = memory_kv.unsqueeze(0).expand(B, num_occupied, D)  # (B, num_occupied, D)

        # Cross-attention retrieval
        retrieved = self.retrieval_attn(query, memory_kv, memory_kv)  # (B, S, D)

        return retrieved

    def reset(self):
        """Clear all memory slots."""
        self.memory.zero_()
        self.importance.zero_()
        self.age.zero_()
        self.occupied.fill_(False)
        self.write_ptr.fill_(0)


class PersonalityModule(nn.Module):
    """
    Learned embedding that defines an "objective" for the model.

    Features:
    - Learnable base personality embedding
    - Can be seeded from instruction text
    - Colors interpretation through context modulation
    - Provides a prior distribution for KL divergence regularization
    """

    def __init__(
        self,
        d_model: int = 512,
        hidden_dim: int = 256,
        sync_pairs: int = 512,
        kl_temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.sync_pairs = sync_pairs
        self.kl_temperature = kl_temperature

        # Learnable base personality (the "soul" of the model)
        self.base_personality = nn.Parameter(torch.randn(d_model) * 0.02)

        # Text-to-personality projection (for instruction seeding)
        self.text_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

        # Personality-context interaction
        # How does the personality color the current context?
        self.context_modulator = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),  # personality + context
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

        # Prior projection: personality -> sync space for KL divergence
        # This defines what the "ideal" sync distribution looks like
        # given the personality
        self.prior_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, sync_pairs),
        )

        self.norm = RMSNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for module in [self.text_proj, self.context_modulator, self.prior_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def seed_from_text(self, text_embedding: torch.Tensor) -> None:
        """
        Seed personality from instruction text embedding.

        Args:
            text_embedding: (D,) or (N, D) from text encoder
                           e.g., encoded "Always be truthful and kind"
        """
        if text_embedding.dim() == 2:
            text_embedding = text_embedding.mean(dim=0)  # Average if multiple

        # Project text to personality space
        text_personality = self.text_proj(text_embedding)

        # Blend into base personality
        with torch.no_grad():
            self.base_personality.add_(text_personality * 0.5)

    def forward(
        self,
        context: torch.Tensor,  # (B, S, D) current context
        text_embedding: Optional[torch.Tensor] = None,  # Optional runtime instruction
    ) -> torch.Tensor:
        """
        Apply personality to context.

        Args:
            context: Current context state
            text_embedding: Optional runtime instruction override

        Returns:
            personality_signal: (B, S, D) personality-modulated signal
        """
        B, S, D = context.shape

        # Get current personality
        personality = self.base_personality  # (D,)

        # Optionally blend in runtime instruction
        if text_embedding is not None:
            if text_embedding.dim() == 2:
                text_embedding = text_embedding.mean(dim=0)
            runtime_personality = self.text_proj(text_embedding)  # (D,)
            personality = personality + 0.3 * runtime_personality

        personality = self.norm(personality)

        # Expand to match context shape
        personality_expanded = personality.unsqueeze(0).unsqueeze(0).expand(B, S, D)

        # Modulate: how does this context relate to the personality?
        combined = torch.cat([personality_expanded, context], dim=-1)  # (B, S, 2D)
        personality_signal = self.context_modulator(combined)  # (B, S, D)

        return personality_signal

    def get_prior_logits(
        self,
        text_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the personality prior in sync space (as logits).

        Args:
            text_embedding: Optional runtime instruction to blend in

        Returns:
            prior_logits: (sync_pairs,) logits defining personality prior
        """
        personality = self.base_personality  # (D,)

        if text_embedding is not None:
            if text_embedding.dim() == 2:
                text_embedding = text_embedding.mean(dim=0)
            runtime_personality = self.text_proj(text_embedding)
            personality = personality + 0.3 * runtime_personality

        personality = self.norm(personality)

        # Project to sync space
        prior_logits = self.prior_proj(personality)  # (sync_pairs,)
        return prior_logits

    def compute_kl_divergence(
        self,
        sync: torch.Tensor,  # (B, S, sync_pairs) actual sync output
        text_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute KL divergence between sync distribution and personality prior.

        KL(sync || prior) - penalizes sync for deviating from personality.

        Args:
            sync: Actual sync output from the model
            text_embedding: Optional runtime instruction

        Returns:
            kl_loss: Scalar KL divergence (averaged over batch and sequence)
        """
        # Get prior logits
        prior_logits = self.get_prior_logits(text_embedding)  # (sync_pairs,)

        # Convert to distributions with temperature
        # Lower temperature = sharper distributions = stronger constraint
        sync_log_probs = F.log_softmax(sync / self.kl_temperature, dim=-1)  # (B, S, sync_pairs)
        prior_probs = F.softmax(prior_logits / self.kl_temperature, dim=-1)  # (sync_pairs,)

        # KL divergence: sum over sync_pairs, mean over batch and sequence
        # KL(q || p) = sum(q * log(q/p)) = sum(q * log_q) - sum(q * log_p)
        # Since we have log_softmax(sync), we need:
        # KL = sum(sync_probs * sync_log_probs) - sum(sync_probs * log(prior_probs))

        sync_probs = F.softmax(sync / self.kl_temperature, dim=-1)
        prior_log_probs = F.log_softmax(prior_logits / self.kl_temperature, dim=-1)

        # Expand prior to match sync shape
        prior_log_probs = prior_log_probs.unsqueeze(0).unsqueeze(0)  # (1, 1, sync_pairs)

        # KL(sync || prior) per position
        kl_per_pos = (sync_probs * (sync_log_probs - prior_log_probs)).sum(dim=-1)  # (B, S)

        # Average over all positions
        kl_loss = kl_per_pos.mean()

        return kl_loss


class IntentionModule(nn.Module):
    """
    Goal-directed drive modeled as oscillating intensity.

    Intention = "wanting to achieve X" where X is defined by personality.
    The oscillation modulates HOW MUCH to pursue personality-defined goals,
    not WHAT those goals are. It's the intensity dial.

    Key features:
    - Multiple learned frequencies spanning phrase-to-document timescales
    - Context gating: which oscillators are relevant to current context
    - KL modulation: stronger intention = tighter constraint to personality
    """

    def __init__(self, config: IntentionConfig):
        super().__init__()
        self.config = config
        self.num_oscillators = config.num_oscillators

        # === 1. Learned Frequency Bank ===
        # Initialize frequencies to cover diverse timescales (log-uniform)
        init_periods = torch.logspace(
            math.log10(config.min_period),
            math.log10(config.max_period),
            config.num_oscillators
        )
        init_frequencies = 1.0 / init_periods  # Convert periods to frequencies

        self.frequencies = nn.Parameter(init_frequencies)
        self.amplitudes = nn.Parameter(
            torch.ones(config.num_oscillators) / config.num_oscillators
        )
        self.phases = nn.Parameter(
            torch.rand(config.num_oscillators) * 2 * math.pi
        )

        # === 2. Context-Dependent Gating ===
        # Allows context to modulate which oscillators are active
        if config.use_context_gating:
            self.context_gate = nn.Sequential(
                nn.Linear(config.d_model, config.num_oscillators * 2),
                nn.GELU(),
                nn.Linear(config.num_oscillators * 2, config.num_oscillators),
                nn.Sigmoid()
            )
        else:
            self.context_gate = None

        # === 3. Oscillation to Intention ===
        # Maps oscillation values to d_model intention vector
        self.oscillation_to_intention = nn.Sequential(
            nn.Linear(config.num_oscillators, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, config.d_model),
        )

        # === 4. KL Modulation (optional) ===
        # Maps oscillation to KL temperature multiplier
        if config.modulate_kl:
            self.kl_modulator = nn.Sequential(
                nn.Linear(config.num_oscillators, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # Output in [0, 1], scaled to kl_range
            )
        else:
            self.kl_modulator = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in [self.oscillation_to_intention]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        if self.context_gate is not None:
            for m in self.context_gate.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        if self.kl_modulator is not None:
            for m in self.kl_modulator.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        context: torch.Tensor,      # (B, S, D) current context
        positions: Optional[torch.Tensor] = None,  # (S,) position indices
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute intention signal from position and context.

        Args:
            context: Current context from ContextEncoder
            positions: Position indices (defaults to 0, 1, 2, ...)

        Returns:
            intention_signal: (B, S, D) intention modulation signal
            kl_temperature_mult: (B, S) KL temperature multiplier per position
        """
        B, S, D = context.shape
        device = context.device
        dtype = context.dtype

        # 1. Generate position indices if not provided
        if positions is None:
            positions = torch.arange(S, device=device, dtype=dtype)
        else:
            positions = positions.to(dtype)

        # 2. Compute raw oscillations: A * sin(2*pi*f*pos + phi)
        # positions: (S,) -> (S, 1)
        # frequencies: (num_osc,) -> (1, num_osc)
        pos = positions.unsqueeze(1)  # (S, 1)
        freqs = self.frequencies.unsqueeze(0)  # (1, num_osc)
        amps = F.softmax(self.amplitudes, dim=0).unsqueeze(0)  # (1, num_osc) normalized
        phases = self.phases.unsqueeze(0)  # (1, num_osc)

        # Raw oscillations: (S, num_osc)
        oscillations = amps * torch.sin(2 * math.pi * freqs * pos + phases)

        # 3. Context-dependent gating (optional)
        if self.context_gate is not None:
            # Use mean context across sequence for global gating decision
            context_summary = context.mean(dim=1)  # (B, D)
            gates = self.context_gate(context_summary)  # (B, num_osc)
            # Gate oscillators: (B, S, num_osc)
            oscillations = oscillations.unsqueeze(0) * gates.unsqueeze(1)
        else:
            oscillations = oscillations.unsqueeze(0).expand(B, -1, -1)  # (B, S, num_osc)

        # 4. Convert oscillations to intention signal
        intention_signal = self.oscillation_to_intention(oscillations)  # (B, S, D)

        # 5. Compute KL temperature multiplier (optional)
        if self.kl_modulator is not None:
            # Mean oscillation state for KL modulation
            kl_raw = self.kl_modulator(oscillations).squeeze(-1)  # (B, S)
            min_mult, max_mult = self.config.kl_modulation_range
            kl_temperature_mult = min_mult + kl_raw * (max_mult - min_mult)
        else:
            kl_temperature_mult = torch.ones(B, S, device=device, dtype=dtype)

        return intention_signal, kl_temperature_mult

    def get_oscillation_state(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get raw oscillation values for visualization/analysis.

        Args:
            positions: (S,) position indices

        Returns:
            oscillations: (S, num_oscillators) raw oscillation values
        """
        pos = positions.unsqueeze(1).float()
        freqs = self.frequencies.unsqueeze(0)
        amps = F.softmax(self.amplitudes, dim=0).unsqueeze(0)
        phases = self.phases.unsqueeze(0)

        oscillations = amps * torch.sin(2 * math.pi * freqs * pos + phases)
        return oscillations


class SyncIntegrator(nn.Module):
    """
    Combine context, memory, and personality into sync output.

    Implements: sync = f(personality, context_state, memory_state)
    """

    def __init__(
        self,
        context_dim: int,
        memory_dim: int,
        personality_dim: int,
        output_dim: int,  # sync_pairs
        hidden_dim: Optional[int] = None,
        use_intention: bool = False,  # Include intention strength in integration
    ):
        super().__init__()
        self.use_intention = use_intention

        # +1 for boundary_prob, +1 for intention_strength (if used)
        extra_dims = 1 + (1 if use_intention else 0)
        total_input = context_dim + memory_dim + personality_dim + extra_dims
        if hidden_dim is None:
            hidden_dim = (total_input + output_dim) // 2

        self.integrator = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Residual connection from context (for stability)
        self.context_residual = nn.Linear(context_dim, output_dim, bias=False)
        self.residual_weight = nn.Parameter(torch.tensor(0.3))

        self._init_weights()

    def _init_weights(self):
        for module in self.integrator.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.context_residual.weight, std=0.02)

    def forward(
        self,
        context_state: torch.Tensor,      # (B, S, D)
        memory_state: torch.Tensor,       # (B, S, D)
        personality_signal: torch.Tensor, # (B, S, D)
        boundary_probs: torch.Tensor,     # (B, S)
        intention_strength: Optional[torch.Tensor] = None,  # (B, S) optional
    ) -> torch.Tensor:
        """
        Integrate all signals into sync output.

        Args:
            context_state: Encoded context from history
            memory_state: Retrieved memory embeddings
            personality_signal: Personality-modulated signal
            boundary_probs: Semantic boundary probabilities
            intention_strength: Optional intention intensity signal

        Returns:
            sync: (B, S, sync_pairs)
        """
        # Concatenate all inputs
        inputs = [
            context_state,
            memory_state,
            personality_signal,
            boundary_probs.unsqueeze(-1),  # (B, S, 1)
        ]
        if self.use_intention and intention_strength is not None:
            inputs.append(intention_strength.unsqueeze(-1))  # (B, S, 1)

        combined = torch.cat(inputs, dim=-1)

        # Main integration path
        sync = self.integrator(combined)

        # Add residual from context (weighted)
        residual = self.context_residual(context_state)
        sync = sync + torch.sigmoid(self.residual_weight) * residual

        return sync


class SyncModule(nn.Module):
    """
    Memory-augmented, personality-colored synchronization module.

    Drop-in replacement for EnhancedSynchronizationModule.

    Input: (B, S, T, D) neural history
    Output: (B, S, sync_pairs) sync features

    Components:
    1. ContextEncoder - aggregates temporal history
    2. ChangePointDetector - detects semantic boundaries
    3. MemoryBank - stores/retrieves chunk embeddings
    4. PersonalityModule - learned objective embedding
    5. SyncIntegrator - combines all signals
    """

    def __init__(self, config: SyncModuleConfig):
        super().__init__()
        self.config = config
        self.sync_pairs = config.sync_pairs

        # Context encoder: history -> context
        self.context_encoder = ContextEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

        # Change-point detection
        self.cpd = ChangePointDetector(
            d_model=config.d_model,
            window_size=config.cpd_window_size,
            threshold=config.cpd_threshold,
            min_chunk_size=config.cpd_min_chunk_size,
        )

        # Memory bank
        self.memory = MemoryBank(
            num_slots=config.memory_slots,
            memory_dim=config.memory_dim,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

        # Personality (with KL divergence support)
        self.personality = PersonalityModule(
            d_model=config.d_model,
            hidden_dim=config.personality_dim // 2,
            sync_pairs=config.sync_pairs,
            kl_temperature=config.kl_temperature,
        )

        # Intention (oscillating goal-directed drive) - optional
        if config.use_intention:
            intention_config = IntentionConfig(
                d_model=config.d_model,
                num_oscillators=config.intention_oscillators,
                min_period=config.intention_min_period,
                max_period=config.intention_max_period,
                use_context_gating=config.intention_context_gating,
                modulate_kl=config.intention_modulate_kl,
                kl_modulation_range=config.intention_kl_range,
            )
            self.intention = IntentionModule(intention_config)
        else:
            self.intention = None

        # Sync integrator
        self.sync_integrator = SyncIntegrator(
            context_dim=config.d_model,
            memory_dim=config.memory_dim,
            personality_dim=config.d_model,
            output_dim=config.sync_pairs,
            use_intention=config.use_intention,
        )

        # Final normalization (for compatibility)
        self.norm = RMSNorm(config.sync_pairs)

    def forward(
        self,
        history: torch.Tensor,  # (B, S, T, D)
        surprise: Optional[torch.Tensor] = None,  # (B, S, 1) from PEM
        personality_text: Optional[torch.Tensor] = None,  # Runtime instruction
        positions: Optional[torch.Tensor] = None,  # (S,) position indices for intention
    ) -> torch.Tensor:
        """
        Compute synchronization with memory and personality.

        Args:
            history: Neural activation history from CTM
            surprise: Optional surprise signal from PEM SurpriseModule
            personality_text: Optional text embedding for runtime objective
            positions: Optional position indices for intention oscillation phase

        Returns:
            sync: (B, S, sync_pairs) synchronization features

        Side effects:
            Sets self.last_kl_loss: KL divergence from personality prior
            Sets self.last_boundary_probs: Detected semantic boundaries
            Sets self.last_intention_strength: Intention intensity (if intention enabled)
        """
        B, S, T, D = history.shape

        # 1. Encode context from history (aggregate temporal dimension)
        context = self.context_encoder(history)  # (B, S, D)

        # 2. Detect change points (semantic boundaries)
        boundary_probs, chunk_ids = self.cpd(context)  # (B, S), (B, S)

        # 3. Update memory with new chunks (importance from surprise)
        self.memory.write(context, chunk_ids, surprise)

        # 4. Retrieve from memory
        memory_state = self.memory.read(context)  # (B, S, D)

        # 5. Apply personality
        personality_signal = self.personality(context, personality_text)  # (B, S, D)

        # 6. Apply intention modulation (if enabled)
        intention_strength = None
        kl_temp_mult = None
        if self.intention is not None:
            intention_signal, kl_temp_mult = self.intention(context, positions)

            # Modulate personality signal by intention
            # Intention scales HOW MUCH personality influences the sync
            # (1 + intention_signal) centers modulation around 1.0
            personality_signal = personality_signal * (1.0 + 0.5 * torch.tanh(intention_signal))

            # Compute intention strength for integrator
            intention_strength = intention_signal.norm(dim=-1)  # (B, S)
            self.last_intention_strength = intention_strength
            self.last_kl_temp_mult = kl_temp_mult

        # 7. Integrate all signals into sync output
        sync = self.sync_integrator(
            context_state=context,
            memory_state=memory_state,
            personality_signal=personality_signal,
            boundary_probs=boundary_probs,
            intention_strength=intention_strength,
        )  # (B, S, sync_pairs)

        # 8. Normalize
        sync = self.norm(sync)

        # 9. Compute KL divergence from personality prior (stored for training)
        # With intention, KL temperature varies by position (stronger intention = tighter)
        if kl_temp_mult is not None and self.config.intention_modulate_kl:
            self.last_kl_loss = self._compute_modulated_kl(
                sync, kl_temp_mult, personality_text
            )
        else:
            self.last_kl_loss = self.personality.compute_kl_divergence(
                sync, personality_text
            )
        self.last_boundary_probs = boundary_probs

        return sync

    def _compute_modulated_kl(
        self,
        sync: torch.Tensor,           # (B, S, sync_pairs)
        kl_temp_mult: torch.Tensor,   # (B, S) temperature multiplier per position
        personality_text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute KL divergence with position-varying temperature.

        Higher intention (higher kl_temp_mult) = tighter constraint to personality.
        Lower intention = more freedom to deviate.
        """
        prior_logits = self.personality.get_prior_logits(personality_text)
        base_temp = self.personality.kl_temperature

        # Per-position effective temperature
        # Higher mult = lower effective temp = sharper dist = stronger constraint
        effective_temp = base_temp / kl_temp_mult.unsqueeze(-1)  # (B, S, 1)

        # Compute KL with position-varying temperature
        sync_log_probs = F.log_softmax(sync / effective_temp, dim=-1)
        sync_probs = F.softmax(sync / effective_temp, dim=-1)

        # Prior with mean temperature (could also use position-varying)
        mean_temp = effective_temp.mean()
        prior_log_probs = F.log_softmax(prior_logits / mean_temp, dim=-1)

        # Expand prior to match sync shape
        prior_log_probs = prior_log_probs.unsqueeze(0).unsqueeze(0)

        # KL per position
        kl_per_pos = (sync_probs * (sync_log_probs - prior_log_probs)).sum(dim=-1)

        return kl_per_pos.mean()

    def get_kl_loss(self) -> torch.Tensor:
        """
        Get the KL divergence loss from the last forward pass.

        This loss penalizes sync from deviating too far from the
        personality-defined prior distribution.

        Usage in training:
            sync = sync_module(history)
            kl_loss = sync_module.get_kl_loss()
            total_loss = main_loss + kl_weight * kl_loss
        """
        if not hasattr(self, 'last_kl_loss'):
            raise RuntimeError("No KL loss available. Call forward() first.")
        return self.last_kl_loss

    def reset_memory(self):
        """Clear the memory bank."""
        self.memory.reset()

    def seed_personality(self, text_embedding: torch.Tensor):
        """Seed personality from instruction text."""
        self.personality.seed_from_text(text_embedding)


def create_sync_module(
    d_model: int = 512,
    sync_pairs: int = 512,
    memory_slots: int = 100,
    **kwargs,
) -> SyncModule:
    """Factory function to create a sync module."""
    config = SyncModuleConfig(
        d_model=d_model,
        sync_pairs=sync_pairs,
        memory_slots=memory_slots,
        **kwargs,
    )
    return SyncModule(config)
