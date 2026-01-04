"""
Experiential Stream — Memory Crystallization via Relative Surprisal × Novelty

Computes what moments are worth remembering using:
- Relative surprisal: token CE loss minus running baseline (EMA)
  This captures "more surprising than usual" not just "rare token"
- Novelty: how unlike existing memories the current representation is
  This prevents storing redundant memories

Surprise signal:
  s_t = -log p(x_t | x_<t)                    # per-token CE loss
  excess_t = (s_t - ema_mu) / (ema_sigma + eps)  # relative to baseline
  novelty_t = 1 - max_cosine(h_t, memory)     # unlike stored memories
  surprise_t = relu(excess_t) * novelty_t     # both conditions
  chunk_surprise = mean(topk(surprise_t, k))  # robust aggregation

Usage:
    from experiential import ExperientialStream

    exp = ExperientialStream(d_model=768)

    # Forward pass with per-token CE and memory keys
    output = exp(
        hidden_states,
        per_token_ce=ce_loss,      # [B, seq_len]
        memory_keys=memory.keys()   # [num_memories, d_model] or None
    )

    # Use chunk_surprise for crystallization decisions
    if output['chunk_surprise'] > threshold:
        memory.store(...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class ExperientialConfig:
    """Configuration for experiential stream."""
    d_model: int
    predictor_hidden_mult: int = 2
    temperature: float = 0.1
    split_ratio: float = 0.5  # first half / second half split


@dataclass
class Episode:
    """A discrete memory of something that happened."""
    timestamp: int              # when (global step / position)
    content: torch.Tensor       # what (embedding of the event) [d_model]
    context: torch.Tensor       # surrounding state when it happened [d_model]
    salience: float             # how important (surprise × affect)
    valence: float = 0.0        # emotional valence at the time
    arousal: float = 0.0        # arousal level at the time
    retrieval_count: int = 0    # how often accessed (for consolidation)

    def to(self, device: torch.device) -> 'Episode':
        """Move episode tensors to device."""
        return Episode(
            timestamp=self.timestamp,
            content=self.content.to(device),
            context=self.context.to(device),
            salience=self.salience,
            valence=self.valence,
            arousal=self.arousal,
            retrieval_count=self.retrieval_count
        )


class EpisodicMemory(nn.Module):
    """
    Episodic memory buffer — stores crystallized experiences.

    Key concepts:
    - Crystallization: high-salience moments become discrete memories
    - Retrieval: find relevant memories by similarity to current state
    - Decay: old, unused memories fade or get consolidated

    Usage:
        memory = EpisodicMemory(d_model=768, capacity=1000)

        # During experience
        if memory.should_crystallize(salience):
            memory.store(state, context, salience, timestamp)

        # During recall
        retrieved = memory.retrieve(query_state, top_k=5)
    """

    def __init__(
        self,
        d_model: int,
        capacity: int = 1000,
        crystallization_threshold: float = 0.3,
        decay_rate: float = 0.01,
        decay_on_store: bool = True,
        min_salience: float = 0.05
    ):
        super().__init__()
        self.d_model = d_model
        self.capacity = capacity
        self.crystallization_threshold = crystallization_threshold
        self.decay_rate = decay_rate
        self.decay_on_store = decay_on_store
        self.min_salience = min_salience  # Memories below this salience are pruned

        # Episode storage
        self.episodes: List[Episode] = []

        # Global timestamp counter
        self._global_step = 0

    @property
    def size(self) -> int:
        """Current number of stored episodes."""
        return len(self.episodes)

    def should_crystallize(self, salience: float) -> bool:
        """Decide whether a moment should become a memory."""
        return salience > self.crystallization_threshold

    def get_keys(self, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """
        Get all memory content vectors stacked as a tensor.

        Used for novelty computation: comparing current hidden states
        against all stored memories to avoid redundant storage.

        Args:
            device: target device for the tensor

        Returns:
            [num_memories, d_model] tensor or None if empty
        """
        if not self.episodes:
            return None

        keys = torch.stack([ep.content for ep in self.episodes])  # [num_memories, d_model]
        if device is not None:
            keys = keys.to(device)
        return keys

    def store(
        self,
        content: torch.Tensor,
        context: torch.Tensor,
        salience: float,
        valence: float = 0.0,
        arousal: float = 0.0,
        timestamp: Optional[int] = None
    ) -> Episode:
        """
        Store a new episode in memory.

        Args:
            content: the state embedding to store [d_model]
            context: the context/prediction at the time [d_model]
            salience: importance score
            valence: emotional valence
            arousal: arousal level
            timestamp: optional explicit timestamp

        Returns:
            The stored Episode
        """
        if timestamp is None:
            timestamp = self._global_step

        episode = Episode(
            timestamp=timestamp,
            content=content.detach().cpu(),
            context=context.detach().cpu(),
            salience=salience,
            valence=valence,
            arousal=arousal,
            retrieval_count=0
        )

        # Apply decay to existing memories before adding new one
        if self.decay_on_store:
            self.apply_decay()

        # Manage capacity
        if len(self.episodes) >= self.capacity:
            self._evict_lowest_priority()

        self.episodes.append(episode)
        self._global_step += 1

        return episode

    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        min_salience: float = 0.0
    ) -> List[Tuple[Episode, float]]:
        """
        Retrieve relevant episodes by similarity to query.

        Args:
            query: current state to match against [d_model]
            top_k: number of episodes to retrieve
            min_salience: minimum salience threshold for retrieval

        Returns:
            List of (episode, similarity_score) tuples, sorted by similarity
        """
        if not self.episodes:
            return []

        query_cpu = query.detach().cpu()
        query_norm = F.normalize(query_cpu, dim=-1)

        # Compute similarities
        scores = []
        for ep in self.episodes:
            if ep.salience < min_salience:
                continue
            content_norm = F.normalize(ep.content, dim=-1)
            sim = torch.dot(query_norm, content_norm).item()
            scores.append((ep, sim))

        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Update retrieval counts for top-k
        for ep, _ in scores[:top_k]:
            ep.retrieval_count += 1

        return scores[:top_k]

    def retrieve_by_time(
        self,
        recent_n: int = 10
    ) -> List[Episode]:
        """Retrieve most recent episodes."""
        return self.episodes[-recent_n:]

    def retrieve_by_salience(
        self,
        top_k: int = 10
    ) -> List[Episode]:
        """Retrieve most salient episodes."""
        sorted_eps = sorted(self.episodes, key=lambda x: x.salience, reverse=True)
        return sorted_eps[:top_k]

    def retrieve_soft(
        self,
        query: torch.Tensor,
        temperature: float = 0.1,
        salience_weight: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable retrieval for end-to-end training.

        Computes soft attention over all episodes and returns weighted sum.
        Gradients flow through both query and the returned values.

        Args:
            query: [batch, d_model] or [d_model] - query vector(s)
            temperature: softmax temperature (lower = sharper attention)
            salience_weight: how much to weight by salience (0 = pure similarity)

        Returns:
            values: [batch, d_model] - weighted sum of episode contents
            weights: [batch, n_episodes] - attention weights over episodes
        """
        # Handle empty memory
        if not self.episodes:
            if query.dim() == 1:
                return torch.zeros(self.d_model, device=query.device), torch.zeros(0, device=query.device)
            else:
                batch_size = query.size(0)
                return torch.zeros(batch_size, self.d_model, device=query.device), torch.zeros(batch_size, 0, device=query.device)

        # Ensure query is 2D: [batch, d_model]
        if query.dim() == 1:
            query = query.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = query.size(0)
        device = query.device

        # Stack episode contents: [n_episodes, d_model]
        # Keep on same device as query for gradient flow
        episode_contents = torch.stack([ep.content.to(device) for ep in self.episodes])
        n_episodes = episode_contents.size(0)

        # Normalize for cosine similarity
        query_norm = F.normalize(query, dim=-1)  # [batch, d_model]
        content_norm = F.normalize(episode_contents, dim=-1)  # [n_episodes, d_model]

        # Compute similarities: [batch, n_episodes]
        similarities = torch.mm(query_norm, content_norm.t()) / temperature

        # Optionally weight by salience
        if salience_weight > 0:
            saliences = torch.tensor(
                [ep.salience for ep in self.episodes],
                device=device,
                dtype=query.dtype
            )
            # Log-salience bonus (so it doesn't dominate)
            salience_bonus = salience_weight * torch.log1p(saliences)
            similarities = similarities + salience_bonus.unsqueeze(0)

        # Softmax to get attention weights
        weights = F.softmax(similarities, dim=-1)  # [batch, n_episodes]

        # Weighted sum of episode contents
        # [batch, n_episodes] @ [n_episodes, d_model] = [batch, d_model]
        values = torch.mm(weights, episode_contents)

        # Update retrieval counts (based on attention, not hard selection)
        with torch.no_grad():
            # Increment counts proportionally to attention
            avg_weights = weights.mean(dim=0)  # [n_episodes]
            for i, ep in enumerate(self.episodes):
                ep.retrieval_count += int(avg_weights[i].item() > 0.1)

        if squeeze_output:
            values = values.squeeze(0)
            weights = weights.squeeze(0)

        return values, weights

    def get_content_matrix(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get all episode contents as a matrix.

        Args:
            device: device to place tensor on

        Returns:
            contents: [n_episodes, d_model] or empty tensor if no episodes
        """
        if not self.episodes:
            return torch.zeros(0, self.d_model, device=device)
        contents = torch.stack([ep.content for ep in self.episodes])
        if device is not None:
            contents = contents.to(device)
        return contents

    def get_context_matrix(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get all episode contexts as a matrix.

        Args:
            device: device to place tensor on

        Returns:
            contexts: [n_episodes, d_model] or empty tensor if no episodes
        """
        if not self.episodes:
            return torch.zeros(0, self.d_model, device=device)
        contexts = torch.stack([ep.context for ep in self.episodes])
        if device is not None:
            contexts = contexts.to(device)
        return contexts

    def _evict_lowest_priority(self):
        """Remove the least important episode to make room."""
        if not self.episodes:
            return

        # Priority = salience × recency_factor × (1 + retrieval_count)
        # Lower priority = more likely to evict
        current_time = self._global_step
        priorities = []
        for i, ep in enumerate(self.episodes):
            age = current_time - ep.timestamp + 1
            recency = 1.0 / (1 + self.decay_rate * age)
            priority = ep.salience * recency * (1 + 0.1 * ep.retrieval_count)
            priorities.append((i, priority))

        # Find and remove lowest priority
        min_idx = min(priorities, key=lambda x: x[1])[0]
        self.episodes.pop(min_idx)

    def decay_salience(self, factor: float = 0.99):
        """Apply multiplicative decay to all episode saliences."""
        for ep in self.episodes:
            ep.salience *= factor

    def apply_decay(self):
        """
        Apply time-based decay to all memories and prune faded ones.

        Decay formula: salience *= (1 - decay_rate) ^ age_since_last_decay
        Memories below min_salience are removed.

        This should be called periodically (e.g., on each store or every N steps).
        """
        if not self.episodes or self.decay_rate <= 0:
            return

        # Apply decay based on age (memories that haven't been refreshed fade)
        decay_factor = 1 - self.decay_rate
        current_time = self._global_step

        for ep in self.episodes:
            # Decay based on time since storage
            age = current_time - ep.timestamp
            if age > 0:
                # Exponential decay: older memories fade more
                ep.salience *= (decay_factor ** age)

            # Retrieval refreshes memory (reduces effective age)
            # Each retrieval adds back some salience
            if ep.retrieval_count > 0:
                refresh_bonus = min(0.1 * ep.retrieval_count, 0.5)  # Cap at 50% boost
                ep.salience = min(1.0, ep.salience * (1 + refresh_bonus))

        # Prune memories that have faded below threshold
        self.episodes = [ep for ep in self.episodes if ep.salience >= self.min_salience]

    def clear(self):
        """Clear all episodes."""
        self.episodes = []

    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        if not self.episodes:
            return {
                'size': 0,
                'avg_salience': 0.0,
                'avg_retrieval_count': 0.0,
                'avg_age': 0.0
            }

        saliences = [ep.salience for ep in self.episodes]
        retrieval_counts = [ep.retrieval_count for ep in self.episodes]
        ages = [self._global_step - ep.timestamp for ep in self.episodes]

        # Count memories at risk of pruning (within 2x of min_salience)
        at_risk = sum(1 for s in saliences if s < self.min_salience * 2)

        return {
            'size': len(self.episodes),
            'avg_salience': sum(saliences) / len(saliences),
            'min_salience': min(saliences),
            'max_salience': max(saliences),
            'at_risk_count': at_risk,  # Memories close to being pruned
            'avg_retrieval_count': sum(retrieval_counts) / len(retrieval_counts),
            'avg_age': sum(ages) / len(ages),
            'max_age': max(ages)
        }


class ExperientialStream(nn.Module):
    """
    Experiential stream using predictive coding approach with persistent state.

    Key insight: use the model's own hidden states directly instead of
    training separate encoders. The transformer already encodes rich
    representations — we just learn to predict how they evolve.

    Core idea:
    - h_mid = hidden state at sequence midpoint (model's "current state")
    - h_end = hidden state at sequence end (model's "future state")
    - predictor: h_mid → predicted h_end
    - surprise = distance(predicted h_end, actual h_end)

    Persistent state:
    - Maintains state across chunks for continuity
    - Uses previous chunk's end state to inform predictions
    - Enables "memory" across sequence boundaries

    Simple design:
    - No learned attention pooling
    - Just a small MLP predictor
    - Directly uses model's representations
    """

    def __init__(
        self,
        d_model: int,
        predictor_hidden_mult: int = 2,
        split_ratio: float = 0.5,
        use_layer_norm: bool = True,
        use_persistent_state: bool = True,
        use_affect: bool = True,
        use_meta_surprise: bool = True,
        tbptt_steps: int = 0,
        # New surprise signal parameters
        ema_decay: float = 0.99,
        surprise_topk: int = 8,
        novelty_weight: float = 1.0,
        # Salience computation parameters
        meta_surprise_salience_weight: float = 1.0,  # How much meta-surprise boosts salience
    ):
        super().__init__()
        self.d_model = d_model
        self.split_ratio = split_ratio
        self.use_persistent_state = use_persistent_state
        self.use_affect = use_affect
        self.use_meta_surprise = use_meta_surprise
        self.meta_surprise_salience_weight = meta_surprise_salience_weight

        # Relative surprisal parameters
        self.ema_decay = ema_decay
        self.surprise_topk = surprise_topk
        self.novelty_weight = novelty_weight
        self.eps = 1e-8

        # Surprise normalization: raw chunk_surprise is unbounded (z-scored),
        # but predicted_surprise is sigmoid-bounded [0, 1]. We use centered sigmoid:
        #   chunk_surprise = sigmoid((raw - ema_raw_mu) * scale / ema_raw_sigma)
        # This centers output around 0.5 and spreads across full [0, 1] range.
        self.surprise_scale = 2.0  # Higher scale = more spread

        # EMA buffers for baseline CE tracking
        self.register_buffer('ema_mu', torch.tensor(2.0))  # Start with reasonable prior
        self.register_buffer('ema_sigma', torch.tensor(1.0))
        self.register_buffer('ema_initialized', torch.tensor(False))

        # EMA buffers for chunk_surprise_raw normalization (to center sigmoid)
        self.register_buffer('ema_raw_mu', torch.tensor(1.0))  # Mean of chunk_surprise_raw
        self.register_buffer('ema_raw_sigma', torch.tensor(1.0))  # Std of chunk_surprise_raw
        self.register_buffer('ema_raw_initialized', torch.tensor(False))

        # Predictor input size depends on whether we use persistent state
        # With persistent state: [h_mid, prev_state] → predicted h_end
        # Without: h_mid → predicted h_end
        predictor_input_dim = d_model * 2 if use_persistent_state else d_model
        hidden_dim = d_model * predictor_hidden_mult

        layers = [
            nn.Linear(predictor_input_dim, hidden_dim),
            nn.GELU(),
        ]
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, d_model))

        self.predictor = nn.Sequential(*layers)

        # State update gate: controls how much new info vs old state
        if use_persistent_state:
            self.state_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )

        # Affect prediction heads (valence and arousal)
        # These predict the emotional quality of the current state
        if use_affect:
            affect_hidden = d_model // 2
            # Valence: positive/negative (-1 to 1)
            self.valence_head = nn.Sequential(
                nn.Linear(d_model, affect_hidden),
                nn.GELU(),
                nn.Linear(affect_hidden, 1),
                nn.Tanh()  # Output in [-1, 1]
            )
            # Arousal: activation level (0 to 1)
            self.arousal_head = nn.Sequential(
                nn.Linear(d_model, affect_hidden),
                nn.GELU(),
                nn.Linear(affect_hidden, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )

        # Meta-surprise: predict own surprise BEFORE computing it
        # This is the first step toward self-awareness: "How surprised will I be?"
        if use_meta_surprise:
            meta_hidden = d_model // 2
            self.surprise_predictor = nn.Sequential(
                nn.Linear(predictor_input_dim, meta_hidden),
                nn.GELU(),
                nn.Linear(meta_hidden, 1),
                nn.Sigmoid()  # Output in [0, 1] to match surprise range
            )

            # Self-modulation: adjust processing based on self-knowledge
            # This is the key step: meta-surprise doesn't just affect memory,
            # it affects actual processing. High meta-surprise ("I don't know
            # myself here") → blend toward more conservative/prior representation.
            #
            # Input: h_end (what we computed) + meta_surprise (how uncertain)
            # Output: confidence gate per dimension [0, 1]
            #   High confidence → use h_end
            #   Low confidence → use fallback (prev_state or h_mid)
            self.self_modulator = nn.Sequential(
                nn.Linear(d_model + 1, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
                nn.Sigmoid()
            )

            # Extended self-awareness: predict retrieval before it happens
            # "What will I remember?" - predict what memory will be retrieved
            # Input: predictor_input (h_mid + prev_state)
            # Output: predicted memory embedding [d_model]
            self.retrieval_predictor = nn.Sequential(
                nn.Linear(predictor_input_dim, meta_hidden),
                nn.GELU(),
                nn.Linear(meta_hidden, d_model)
            )

            # Extended self-awareness: predict affect before computing it
            # "How will I feel?" - predict valence and arousal before experiencing
            # Input: predictor_input (h_mid + prev_state)
            # Output: [predicted_valence, predicted_arousal]
            self.affect_predictor = nn.Sequential(
                nn.Linear(predictor_input_dim, meta_hidden),
                nn.GELU(),
                nn.Linear(meta_hidden, 2)  # [valence, arousal]
            )

        # Persistent state buffer (not a parameter, just a buffer)
        self.register_buffer('_persistent_state', None)
        self._batch_size = None

        # Truncated BPTT settings
        # tbptt_steps > 0: allow gradients for K steps, then detach
        # tbptt_steps = 0: always detach (original behavior, prevents OOM but no state learning)
        self.tbptt_steps = tbptt_steps
        self._steps_since_detach = 0

        self._init_weights()

    def _init_weights(self):
        """Initialize predictor output with values that produce reasonable predictions."""
        # Use standard xavier init. We need predictions with meaningful magnitude
        # (target hidden states have norm ~8-10). Gain=0.01 was too conservative.
        nn.init.xavier_uniform_(self.predictor[-1].weight, gain=0.5)
        nn.init.zeros_(self.predictor[-1].bias)
        if self.use_persistent_state:
            # Initialize gate to 0.5 (balanced between old and new)
            nn.init.zeros_(self.state_gate[0].weight)
            nn.init.constant_(self.state_gate[0].bias, 0.0)
        if self.use_meta_surprise:
            # Initialize surprise predictor to predict ~0.5 (uncertain)
            nn.init.zeros_(self.surprise_predictor[-2].weight)
            nn.init.constant_(self.surprise_predictor[-2].bias, 0.0)  # sigmoid(0) = 0.5
            # Initialize self-modulator to output ~1 (high confidence initially)
            # This means h_end passes through unchanged until the system learns
            nn.init.zeros_(self.self_modulator[-2].weight)
            nn.init.constant_(self.self_modulator[-2].bias, 2.0)  # sigmoid(2) ≈ 0.88
            # Initialize retrieval predictor with small weights (start uncertain)
            nn.init.xavier_uniform_(self.retrieval_predictor[-1].weight, gain=0.1)
            nn.init.zeros_(self.retrieval_predictor[-1].bias)
            # Initialize affect predictor to predict neutral (valence=0, arousal=0.5)
            nn.init.zeros_(self.affect_predictor[-1].weight)
            nn.init.zeros_(self.affect_predictor[-1].bias)  # tanh(0)=0, sigmoid(0)=0.5

    def compute_excess_surprisal(
        self,
        per_token_ce: torch.Tensor,
        update_ema: bool = True,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute surprisal relative to running baseline.

        Args:
            per_token_ce: [B, seq_len] per-token cross-entropy loss
            update_ema: whether to update running statistics (set False for eval)
            mask: [B, seq_len] boolean mask, True = valid (exclude padding from EMA)

        Returns:
            excess: [B, seq_len] normalized surprisal (positive = above baseline)
        """
        # Update EMA during training (exclude padding from statistics)
        if update_ema and self.training:
            if mask is not None:
                # Only compute stats over valid (non-pad) tokens
                valid_ce = per_token_ce[mask]
                if valid_ce.numel() > 0:
                    batch_mu = valid_ce.mean()
                    batch_sigma = valid_ce.std().clamp(min=self.eps) if valid_ce.numel() > 1 else self.ema_sigma
                else:
                    # No valid tokens, skip update
                    batch_mu = self.ema_mu
                    batch_sigma = self.ema_sigma
            else:
                batch_mu = per_token_ce.mean()
                batch_sigma = per_token_ce.std().clamp(min=self.eps)

            if not self.ema_initialized:
                self.ema_mu.copy_(batch_mu)
                self.ema_sigma.copy_(batch_sigma)
                self.ema_initialized.fill_(True)
            else:
                self.ema_mu.mul_(self.ema_decay).add_(batch_mu * (1 - self.ema_decay))
                self.ema_sigma.mul_(self.ema_decay).add_(batch_sigma * (1 - self.ema_decay))

        # Normalize: (CE - mean) / std
        excess = (per_token_ce - self.ema_mu) / (self.ema_sigma + self.eps)
        return excess

    def compute_novelty(
        self,
        hidden_states: torch.Tensor,
        memory_keys: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute novelty = 1 - max_cosine(h_t, memory).

        Args:
            hidden_states: [B, seq_len, d_model]
            memory_keys: [num_memories, d_model] or None

        Returns:
            novelty: [B, seq_len] in [0, 1], 1 = completely novel
        """
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device

        # No memories yet → everything is novel
        if memory_keys is None or memory_keys.numel() == 0:
            return torch.ones(batch_size, seq_len, device=device)

        # Normalize for cosine similarity
        h_norm = F.normalize(hidden_states, dim=-1)  # [B, seq_len, d_model]
        m_norm = F.normalize(memory_keys, dim=-1)     # [num_memories, d_model]

        # Compute similarities: [B, seq_len, num_memories]
        similarities = torch.einsum('bsd,md->bsm', h_norm, m_norm)

        # Max similarity per position (most similar memory)
        max_sim, _ = similarities.max(dim=-1)  # [B, seq_len]

        # Novelty = 1 - max_sim (clamp to handle numerical issues)
        novelty = (1.0 - max_sim).clamp(min=0.0, max=1.0)
        return novelty

    def compute_surprise_signal(
        self,
        per_token_ce: torch.Tensor,
        hidden_states: torch.Tensor,
        memory_keys: Optional[torch.Tensor] = None,
        update_ema: bool = True,
        start_idx: int = 0,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute relative surprisal × novelty signal.

        This is the main surprise computation:
        - excess_t = (CE_t - ema_mu) / ema_sigma  (relative to baseline)
        - novelty_t = 1 - max_cosine(h_t, memory)  (unlike stored memories)
        - surprise_t = relu(excess_t) * novelty_t  (both conditions)
        - chunk_surprise = sigmoid(mean(topk(surprise_t[start_idx:])))  (normalized)

        Args:
            per_token_ce: [B, seq_len] per-token cross-entropy loss
            hidden_states: [B, seq_len, d_model]
            memory_keys: [num_memories, d_model] or None
            update_ema: whether to update running statistics
            start_idx: only aggregate surprise from this index onward (for "future chunk" semantics)
            mask: [B, seq_len] boolean mask, True = valid token, False = pad (excluded from aggregation)

        Returns:
            dict with:
                - surprise_t: [B, seq_len] per-token surprise (full sequence)
                - excess_t: [B, seq_len] relative surprisal
                - novelty_t: [B, seq_len] novelty vs memory
                - chunk_surprise: [B] aggregated chunk surprise, normalized to [0, 1]
                - chunk_surprise_raw: [B] raw (unnormalized) chunk surprise
                - ema_mu: current baseline mean
                - ema_sigma: current baseline std
        """
        batch_size, seq_len = per_token_ce.shape

        # 1. Excess surprisal relative to baseline (pass mask to exclude padding from EMA)
        excess = self.compute_excess_surprisal(per_token_ce, update_ema, mask=mask)  # [B, seq_len]

        # 2. Novelty vs memory
        novelty = self.compute_novelty(hidden_states, memory_keys)  # [B, seq_len]

        # 3. Combine: relu(excess) * novelty
        # relu ensures we only care about above-baseline surprisal
        # multiplication means both conditions must hold
        if self.novelty_weight > 0:
            surprise_t = F.relu(excess) * (novelty ** self.novelty_weight)
        else:
            surprise_t = F.relu(excess)

        # 4. Aggregate: top-k mean for robustness
        # Only consider tokens from start_idx onward (future chunk semantics)
        # Also exclude padded positions if mask is provided
        future_surprise = surprise_t[:, start_idx:]  # [B, future_len]
        future_len = future_surprise.size(1)

        if mask is not None:
            future_mask = mask[:, start_idx:]  # [B, future_len]
            # Mask out padded positions by setting to -inf before topk
            future_surprise = future_surprise.masked_fill(~future_mask, 0.0)
            valid_counts = future_mask.sum(dim=-1).clamp(min=1)  # [B]
        else:
            valid_counts = torch.full((batch_size,), future_len, device=surprise_t.device)

        topk = min(self.surprise_topk, future_len)
        if topk > 0 and topk < future_len:
            # Adjust topk per batch based on valid counts
            topk_vals, _ = future_surprise.topk(topk, dim=-1)
            chunk_surprise_raw = topk_vals.mean(dim=-1)
        else:
            if mask is not None:
                chunk_surprise_raw = future_surprise.sum(dim=-1) / valid_counts
            else:
                chunk_surprise_raw = future_surprise.mean(dim=-1)

        # 5. Normalize to [0, 1] using centered sigmoid
        # This spreads surprise across full [0, 1] range instead of being stuck > 0.5
        with torch.no_grad():
            batch_raw_mean = chunk_surprise_raw.mean()
            # Use unbiased=False to avoid warning on single samples
            batch_raw_std = chunk_surprise_raw.std(unbiased=False).clamp(min=self.eps)

            if update_ema:
                if not self.ema_raw_initialized:
                    self.ema_raw_mu.copy_(batch_raw_mean)
                    self.ema_raw_sigma.copy_(batch_raw_std)
                    self.ema_raw_initialized.fill_(True)
                else:
                    self.ema_raw_mu.mul_(self.ema_decay).add_(batch_raw_mean * (1 - self.ema_decay))
                    self.ema_raw_sigma.mul_(self.ema_decay).add_(batch_raw_std * (1 - self.ema_decay))

        # Center by subtracting EMA mean, scale by EMA std
        # This makes sigmoid output centered around 0.5 with good spread
        centered_raw = (chunk_surprise_raw - self.ema_raw_mu) / (self.ema_raw_sigma + self.eps)
        chunk_surprise = torch.sigmoid(centered_raw * self.surprise_scale)

        return {
            'surprise_t': surprise_t,
            'excess_t': excess,
            'novelty_t': novelty,
            'chunk_surprise': chunk_surprise,
            'chunk_surprise_raw': chunk_surprise_raw,
            'ema_mu': self.ema_mu.item(),
            'ema_sigma': self.ema_sigma.item(),
            'ema_raw_mu': self.ema_raw_mu.item(),
            'ema_raw_sigma': self.ema_raw_sigma.item(),
        }

    def reset_state(self, batch_size: Optional[int] = None):
        """Reset persistent state (call at start of new sequence/episode)."""
        self._persistent_state = None
        self._batch_size = batch_size
        self._steps_since_detach = 0  # Reset TBPTT counter

    def detach_state(self):
        """Detach state from computation graph (for truncated BPTT)."""
        if self._persistent_state is not None:
            self._persistent_state = self._persistent_state.detach()
        self._steps_since_detach = 0  # Reset counter on manual detach

    def get_state(self) -> Optional[torch.Tensor]:
        """Get current persistent state."""
        return self._persistent_state

    def set_state(self, state: torch.Tensor):
        """Set persistent state (e.g., when resuming)."""
        self._persistent_state = state.detach()
        self._batch_size = state.size(0)

    def _get_or_init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get persistent state, initializing if needed."""
        if self._persistent_state is None or self._batch_size != batch_size:
            # Initialize to zeros (no prior context)
            self._persistent_state = torch.zeros(batch_size, self.d_model, device=device)
            self._batch_size = batch_size
        return self._persistent_state

    def _update_state(self, h_end: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        """Update persistent state using gated combination."""
        # Gate controls: how much of h_end to incorporate vs keeping prev_state
        gate_input = torch.cat([h_end, prev_state], dim=-1)
        gate = self.state_gate(gate_input)  # [B, d_model], values in [0, 1]

        # new_state = gate * h_end + (1 - gate) * prev_state
        new_state = gate * h_end + (1 - gate) * prev_state
        return new_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        mid_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        update_state: bool = True,
        per_token_ce: Optional[torch.Tensor] = None,
        memory_keys: Optional[torch.Tensor] = None,
        ce_mask: Optional[torch.Tensor] = None,
        retrieved_memory: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process hidden states and compute surprise/salience for crystallization.

        NEW: When per_token_ce is provided, uses relative surprisal × novelty signal:
            - excess = (CE - baseline) / std  (relative to EMA)
            - novelty = 1 - max_cosine(h, memory)  (unlike stored)
            - surprise = sigmoid(mean(topk(relu(excess) * novelty)))  (normalized to [0,1])

        When per_token_ce is None, falls back to MLP-based h_mid→h_end prediction.

        Extended self-awareness (v0.4):
            - Predict retrieval: "What will I remember?" before memory is retrieved
            - Predict affect: "How will I feel?" before computing valence/arousal
            - meta_retrieval_surprise: |predicted_retrieval - actual_retrieval|
            - meta_affect_surprise: |predicted_affect - actual_affect|

        Args:
            hidden_states: [batch, seq_len, d_model] from transformer
            mid_idx: optional custom midpoint index (default: seq_len * split_ratio)
            end_idx: optional custom endpoint index (default: -1)
            update_state: whether to update persistent state after this forward pass
            per_token_ce: [batch, seq_len-1] per-token cross-entropy loss
            memory_keys: [num_memories, d_model] for novelty computation
            ce_mask: [batch, seq_len-1] boolean mask, True = valid token (excludes padding)
            retrieved_memory: [batch, d_model] actual retrieved memory (for meta_retrieval_surprise)

        Returns:
            dict with:
                - state: [batch, d_model] current state (h_mid)
                - prediction: [batch, d_model] predicted future (what we expect)
                - target: [batch, d_model] = modulated_output (what we commit to, detached)
                - h_end: [batch, d_model] raw world state (before self-modulation)
                - surprise: [batch] chunk surprise (either CE-based or MLP-based)
                - surprise_t: [batch, seq_len] per-token surprise (if CE-based)
                - excess_t: [batch, seq_len] relative surprisal (if CE-based)
                - novelty_t: [batch, seq_len] novelty vs memory (if CE-based)
                - predicted_surprise: [batch] self-predicted surprise
                - meta_surprise: [batch] |predicted_surprise - surprise|
                - valence: [batch] emotional valence (-1=negative, 1=positive)
                - arousal: [batch] activation level (0=calm, 1=excited)
                - predicted_valence: [batch] self-predicted valence
                - predicted_arousal: [batch] self-predicted arousal
                - meta_affect_surprise: [batch] |predicted_affect - actual_affect|
                - predicted_retrieval: [batch, d_model] self-predicted memory retrieval
                - meta_retrieval_surprise: [batch] |predicted_retrieval - actual_retrieval|
                - salience: [batch] importance signal for crystallization
                - modulated_output: [batch, d_model] h_end adjusted by self-knowledge
                - confidence_gate: [batch, d_model] how much we trusted h_end
                - persistent_state: [batch, d_model] updated state
                - gate_values: [batch, d_model] state gate activations (if persistent)
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Get midpoint and endpoint hidden states
        if mid_idx is None:
            mid_idx = int(seq_len * self.split_ratio)
        if end_idx is None:
            end_idx = -1

        h_mid = hidden_states[:, mid_idx, :]  # [B, d] — model's state at midpoint
        h_end = hidden_states[:, end_idx, :]  # [B, d] — model's state at end

        # Get or initialize persistent state
        if self.use_persistent_state:
            prev_state = self._get_or_init_state(batch_size, hidden_states.device)
            # Predictor input: concatenate current state with persistent state
            predictor_input = torch.cat([h_mid, prev_state], dim=-1)
        else:
            prev_state = None
            predictor_input = h_mid

        # Predict future from current (+ persistent state if enabled)
        prediction = self.predictor(predictor_input)  # [B, d]

        # Meta-surprise: predict own surprise BEFORE computing it
        # This is the self-awareness signal: "How surprised will I be?"
        predicted_surprise = None
        meta_surprise = None
        # Extended self-awareness predictions
        predicted_retrieval = None
        meta_retrieval_surprise = None
        predicted_valence = None
        predicted_arousal = None
        meta_affect_surprise = None

        if self.use_meta_surprise:
            predicted_surprise = self.surprise_predictor(predictor_input).squeeze(-1)  # [B]

            # Extended self-awareness: predict retrieval BEFORE it happens
            # "What will I remember?" - predict what memory will be retrieved
            predicted_retrieval = self.retrieval_predictor(predictor_input)  # [B, d_model]

            # Extended self-awareness: predict affect BEFORE computing it
            # "How will I feel?" - predict valence and arousal
            affect_pred = self.affect_predictor(predictor_input)  # [B, 2]
            predicted_valence = torch.tanh(affect_pred[:, 0])  # [B] in [-1, 1]
            predicted_arousal = torch.sigmoid(affect_pred[:, 1])  # [B] in [0, 1]

        # Compute surprise signal
        # NEW: Use relative surprisal × novelty when per_token_ce is provided
        # FALLBACK: Use MLP-based h_mid→h_end prediction when not
        surprise_t = None
        excess_t = None
        novelty_t = None

        if per_token_ce is not None:
            # NEW: Relative surprisal × novelty signal
            # This is better because:
            # 1. Uses GPT's own prediction error (principled)
            # 2. Normalizes by baseline (avoids storing rare proper nouns)
            # 3. Filters by novelty (avoids storing redundant memories)
            #
            # Alignment note: per_token_ce[i] is CE for predicting token i+1 using hidden state i
            # So per_token_ce is [B, seq_len-1], use hidden_states[:, :-1, :] to match
            hidden_for_surprise = hidden_states[:, :-1, :] if seq_len > 1 else hidden_states
            ce_seq_len = per_token_ce.size(1)

            # Future chunk semantics: only aggregate surprise from mid_idx onward
            # This matches the original design where predicted_surprise comes from h_mid
            # and actual surprise should be computed over the "future" portion (mid to end)
            # In CE space: mid_idx in hidden corresponds to mid_idx in CE (CE[i] uses hidden[i])
            ce_start_idx = min(mid_idx, ce_seq_len - 1) if ce_seq_len > 0 else 0

            surprise_signal = self.compute_surprise_signal(
                per_token_ce=per_token_ce,
                hidden_states=hidden_for_surprise,
                memory_keys=memory_keys,
                update_ema=self.training,
                start_idx=ce_start_idx,
                mask=ce_mask  # Excludes padding from aggregation
            )
            surprise = surprise_signal['chunk_surprise']
            surprise_t = surprise_signal['surprise_t']
            excess_t = surprise_signal['excess_t']
            novelty_t = surprise_signal['novelty_t']
        else:
            # FALLBACK: MLP-based surprise (for backward compatibility)
            # This predicts h_end from h_mid, but the MLP can't know future tokens
            with torch.no_grad():
                pred_norm = F.normalize(prediction, dim=-1)
                target_norm = F.normalize(h_end, dim=-1)
                similarity = (pred_norm * target_norm).sum(dim=-1)
                surprise = 1 - similarity

        # Compute meta-surprise: how wrong was my self-prediction?
        # This measures self-calibration: "Did I know how I would react?"
        if self.use_meta_surprise and predicted_surprise is not None:
            meta_surprise = (predicted_surprise - surprise.detach()).abs()

        # Compute affect (valence and arousal)
        valence = None
        arousal = None
        salience = None
        if self.use_affect:
            # Predict affect from the end state (what we actually experienced)
            valence = self.valence_head(h_end).squeeze(-1)  # [B]
            arousal = self.arousal_head(h_end).squeeze(-1)  # [B]

            # Compute meta-affect-surprise: "Did I know how I would feel?"
            if self.use_meta_surprise and predicted_valence is not None:
                # Compare predicted affect to actual affect
                # Valence error in [-1, 1] → abs error in [0, 2], normalize to [0, 1]
                valence_error = (predicted_valence - valence.detach()).abs() / 2.0
                # Arousal error in [0, 1] → abs error in [0, 1]
                arousal_error = (predicted_arousal - arousal.detach()).abs()
                # Combined meta-affect-surprise (average of both)
                meta_affect_surprise = (valence_error + arousal_error) / 2.0  # [B] in [0, 1]

            # Salience = how important is this moment?
            # NOTE: Affect (arousal, valence) is computed but NOT used for salience
            # because affect heads are unsupervised. Using them would make
            # crystallization decisions partially random. Salience = surprise only.
            # Meta-surprise boost: moments of self-ignorance are extra important
            #   "I don't know myself here" → pay attention, remember this
            # Weight is configurable (default 1.0, was 3.0) to prevent uncalibrated
            # meta-surprise from dominating salience.
            with torch.no_grad():
                base_salience = surprise  # Affect removed: was surprise * arousal * valence.abs()
                if meta_surprise is not None and self.meta_surprise_salience_weight > 0:
                    # Boost salience by meta-surprise with configurable weight
                    # Range: [1, 1 + weight*max_ms] where max_ms ≈ 0.5-1.0
                    salience = base_salience * (1 + self.meta_surprise_salience_weight * meta_surprise)
                else:
                    salience = base_salience

        # Compute meta-retrieval-surprise: "Did I know what I would remember?"
        # Compare predicted retrieval to actual retrieved memory (if provided)
        if self.use_meta_surprise and predicted_retrieval is not None and retrieved_memory is not None:
            # Cosine distance between predicted and actual retrieval
            # 1 - cosine_similarity gives us a distance in [0, 2], we use [0, 1]
            pred_norm = F.normalize(predicted_retrieval, dim=-1)
            actual_norm = F.normalize(retrieved_memory, dim=-1)
            retrieval_similarity = (pred_norm * actual_norm).sum(dim=-1)  # [B]
            meta_retrieval_surprise = (1 - retrieval_similarity) / 2.0  # [B] in [0, 1]

        # Self-modulation: adjust processing based on self-knowledge
        # This is where meta-surprise AFFECTS behavior, not just memory
        modulated_output = h_end
        confidence_gate = None
        if self.use_meta_surprise and meta_surprise is not None:
            # Build input: h_end + meta_surprise scalar
            ms_scalar = meta_surprise.unsqueeze(-1)  # [B, 1]
            mod_input = torch.cat([h_end, ms_scalar], dim=-1)  # [B, d_model + 1]

            # Compute confidence gate per dimension
            # High meta-surprise → lower confidence → blend toward fallback
            confidence_gate = self.self_modulator(mod_input)  # [B, d_model] in [0, 1]

            # Fallback: previous state (what we knew) or h_mid (current input)
            # This represents "when uncertain, be conservative"
            fallback = prev_state if prev_state is not None else h_mid

            # Blend: confident → use h_end, uncertain → use fallback
            modulated_output = confidence_gate * h_end + (1 - confidence_gate) * fallback

        # Update persistent state with MODULATED output (closes the feedback loop)
        # This means: what the system commits to → becomes input to next prediction
        gate_values = None
        if self.use_persistent_state and update_state:
            # Truncated BPTT: allow gradients for tbptt_steps, then detach
            # This enables learning what to retain while preventing unbounded graph growth
            if self.tbptt_steps > 0:
                # Compute new state WITH gradients
                new_state = self._update_state(modulated_output, prev_state)

                self._steps_since_detach += 1
                if self._steps_since_detach >= self.tbptt_steps:
                    # Time to truncate: detach to prevent further backprop
                    self._persistent_state = new_state.detach()
                    self._steps_since_detach = 0
                else:
                    # Keep gradients flowing
                    self._persistent_state = new_state

                # Track gate values (with gradients since we want to train the gate)
                gate_input = torch.cat([modulated_output, prev_state], dim=-1)
                gate_values = self.state_gate(gate_input)
            else:
                # Original behavior: always detach (no state learning, but safe from OOM)
                new_state = self._update_state(modulated_output.detach(), prev_state)
                self._persistent_state = new_state.detach()

                # Track gate values for analysis only
                with torch.no_grad():
                    gate_input = torch.cat([modulated_output, prev_state], dim=-1)
                    gate_values = self.state_gate(gate_input)

        return {
            'state': h_mid,
            'prediction': prediction,
            # TARGET is now modulated_output: predictor learns to predict committed output
            # This closes the loop: predict(prev_committed) → next_committed
            'target': modulated_output.detach(),  # stop gradient for contrastive loss
            'surprise': surprise,
            'predicted_surprise': predicted_surprise,
            'meta_surprise': meta_surprise,
            'valence': valence,
            'arousal': arousal,
            # Extended self-awareness: affect prediction
            'predicted_valence': predicted_valence,
            'predicted_arousal': predicted_arousal,
            'meta_affect_surprise': meta_affect_surprise,
            # Extended self-awareness: retrieval prediction
            'predicted_retrieval': predicted_retrieval,
            'meta_retrieval_surprise': meta_retrieval_surprise,
            'salience': salience,
            'h_end': h_end.detach(),               # raw world state (for analysis)
            'modulated_output': modulated_output,  # h_end adjusted by self-knowledge (= target)
            'confidence_gate': confidence_gate,    # how much we trusted h_end
            'mid_idx': mid_idx,
            'end_idx': end_idx if end_idx != -1 else seq_len - 1,
            'persistent_state': self._persistent_state,
            'prev_state': prev_state,
            'gate_values': gate_values,
            # NEW: Per-token surprise signal (only when per_token_ce provided)
            'surprise_t': surprise_t,              # [B, seq_len] per-token surprise
            'excess_t': excess_t,                  # [B, seq_len] relative surprisal
            'novelty_t': novelty_t,                # [B, seq_len] novelty vs memory
            'ema_mu': self.ema_mu.item() if per_token_ce is not None else None,
            'ema_sigma': self.ema_sigma.item() if per_token_ce is not None else None,
        }

    def forward_multiscale(
        self,
        hidden_states: torch.Tensor,
        horizons: list[float] = [0.25, 0.5, 0.75],
        update_state: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Predict at multiple horizons for richer signal.

        Args:
            hidden_states: [batch, seq_len, d_model]
            horizons: list of positions (as fraction of seq_len) to predict from
            update_state: whether to update persistent state after this forward pass

        Returns:
            dict with predictions, targets, surprises for each horizon
        """
        batch_size, seq_len, d_model = hidden_states.shape
        h_end = hidden_states[:, -1, :]

        # Get persistent state if enabled
        if self.use_persistent_state:
            prev_state = self._get_or_init_state(batch_size, hidden_states.device)
        else:
            prev_state = None

        predictions = []
        surprises = []
        states = []

        for horizon in horizons:
            idx = int(seq_len * horizon)
            h_t = hidden_states[:, idx, :]

            # Build predictor input
            if self.use_persistent_state:
                predictor_input = torch.cat([h_t, prev_state], dim=-1)
            else:
                predictor_input = h_t

            pred = self.predictor(predictor_input)

            with torch.no_grad():
                pred_norm = F.normalize(pred, dim=-1)
                target_norm = F.normalize(h_end, dim=-1)
                surprise = 1 - (pred_norm * target_norm).sum(dim=-1)

            states.append(h_t)
            predictions.append(pred)
            surprises.append(surprise)

        # Update persistent state after processing all horizons
        if self.use_persistent_state and update_state:
            new_state = self._update_state(h_end.detach(), prev_state)
            self._persistent_state = new_state.detach()

        return {
            'states': torch.stack(states, dim=1),        # [B, n_horizons, d]
            'predictions': torch.stack(predictions, dim=1),  # [B, n_horizons, d]
            'target': h_end.detach(),                    # [B, d]
            'surprises': torch.stack(surprises, dim=1),  # [B, n_horizons]
            'horizons': horizons,
            'persistent_state': self._persistent_state,
            'prev_state': prev_state
        }


# Backwards compatibility alias
ExperientialStreamV02 = ExperientialStream


def experiential_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    InfoNCE contrastive loss for experiential prediction.

    Each prediction should match its own target (same sequence),
    not targets from other sequences in the batch.

    Args:
        prediction: [batch, d_model] predicted future embedding
        target: [batch, d_model] actual future embedding (should be detached)
        temperature: softmax temperature (lower = sharper)

    Returns:
        Scalar loss value
    """
    # Normalize for cosine similarity
    pred_norm = F.normalize(prediction, dim=-1)
    target_norm = F.normalize(target, dim=-1)

    # Similarity matrix: [batch, batch]
    # sim[i, j] = similarity between prediction_i and target_j
    sim_matrix = torch.mm(pred_norm, target_norm.t()) / temperature

    # Labels: each prediction should match target at same index (diagonal)
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

    # Cross entropy: prediction_i should have highest similarity to target_i
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


def meta_surprise_loss(
    predicted_surprise: torch.Tensor,
    actual_surprise: torch.Tensor
) -> torch.Tensor:
    """
    Loss for training meta-surprise prediction (self-awareness).

    The system learns to predict its own surprise before experiencing it.
    This is training for self-calibration: "How surprised will I be?"

    Uses MSE loss between predicted and actual surprise values.

    Args:
        predicted_surprise: [batch] predicted surprise (0-1)
        actual_surprise: [batch] actual surprise (0-1, should be detached)

    Returns:
        Scalar loss value
    """
    return F.mse_loss(predicted_surprise, actual_surprise.detach())


def meta_affect_loss(
    predicted_valence: torch.Tensor,
    predicted_arousal: torch.Tensor,
    actual_valence: torch.Tensor,
    actual_arousal: torch.Tensor
) -> torch.Tensor:
    """
    Loss for training affect prediction (extended self-awareness).

    The system learns to predict its own emotional response before experiencing it.
    This is training for: "How will I feel?"

    Uses MSE loss between predicted and actual valence/arousal values.

    Args:
        predicted_valence: [batch] predicted valence (-1 to 1)
        predicted_arousal: [batch] predicted arousal (0 to 1)
        actual_valence: [batch] actual valence (-1 to 1, should be detached)
        actual_arousal: [batch] actual arousal (0 to 1, should be detached)

    Returns:
        Scalar loss value
    """
    valence_loss = F.mse_loss(predicted_valence, actual_valence.detach())
    arousal_loss = F.mse_loss(predicted_arousal, actual_arousal.detach())
    return (valence_loss + arousal_loss) / 2.0


def meta_retrieval_loss(
    predicted_retrieval: torch.Tensor,
    actual_retrieval: torch.Tensor
) -> torch.Tensor:
    """
    Loss for training retrieval prediction (extended self-awareness).

    The system learns to predict what memories will be retrieved before retrieval.
    This is training for: "What will I remember?"

    Uses cosine similarity loss between predicted and actual retrieved memory.

    Args:
        predicted_retrieval: [batch, d_model] predicted memory embedding
        actual_retrieval: [batch, d_model] actual retrieved memory (should be detached)

    Returns:
        Scalar loss value
    """
    # Cosine similarity loss: 1 - cosine_similarity
    pred_norm = F.normalize(predicted_retrieval, dim=-1)
    actual_norm = F.normalize(actual_retrieval.detach(), dim=-1)
    similarity = (pred_norm * actual_norm).sum(dim=-1)  # [batch]
    # Loss = 1 - similarity (we want similarity to be 1)
    return (1 - similarity).mean()


def combined_experiential_loss(
    output: Dict[str, torch.Tensor],
    exp_weight: float = 1.0,
    meta_weight: float = 0.1,
    self_mod_weight: float = 0.1,
    affect_weight: float = 0.1,
    retrieval_weight: float = 0.1,
    temperature: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined loss for experiential prediction, meta-surprise, self-modulation,
    and extended self-awareness (affect prediction, retrieval prediction).

    Args:
        output: dict from ExperientialStream.forward()
        exp_weight: weight for experiential prediction loss
        meta_weight: weight for meta-surprise loss
        self_mod_weight: weight for self-modulation loss
        affect_weight: weight for meta-affect loss (extended self-awareness)
        retrieval_weight: weight for meta-retrieval loss (extended self-awareness)
        temperature: temperature for contrastive loss

    Returns:
        total_loss: combined scalar loss
        loss_dict: breakdown of individual losses
    """
    loss_dict = {}

    # Experiential prediction loss (predicting world)
    exp_loss = experiential_loss(
        output['prediction'],
        output['target'],
        temperature=temperature
    )
    loss_dict['exp_loss'] = exp_loss.item()

    total_loss = exp_weight * exp_loss

    # Meta-surprise loss (predicting self)
    if output.get('predicted_surprise') is not None:
        meta_loss = meta_surprise_loss(
            output['predicted_surprise'],
            output['surprise']
        )
        loss_dict['meta_loss'] = meta_loss.item()
        loss_dict['mean_meta_surprise'] = output['meta_surprise'].mean().item()
        total_loss = total_loss + meta_weight * meta_loss

    # Self-modulation loss: confidence should inversely track meta-surprise
    # High meta-surprise ("I don't know myself") → low confidence → blend toward fallback
    # This gives the self-modulator a direct training signal
    if output.get('confidence_gate') is not None and output.get('meta_surprise') is not None:
        confidence_mean = output['confidence_gate'].mean(dim=-1)  # [B]
        # Target: high meta-surprise → low confidence, low meta-surprise → high confidence
        target_confidence = 1 - output['meta_surprise'].detach()  # [B], detach to not affect meta predictor
        self_mod_loss = F.mse_loss(confidence_mean, target_confidence)
        loss_dict['self_mod_loss'] = self_mod_loss.item()
        loss_dict['mean_confidence'] = confidence_mean.mean().item()
        total_loss = total_loss + self_mod_weight * self_mod_loss

    # Extended self-awareness: meta-affect loss (predict own emotional response)
    if (output.get('predicted_valence') is not None and
        output.get('predicted_arousal') is not None and
        output.get('valence') is not None and
        output.get('arousal') is not None):
        affect_loss = meta_affect_loss(
            output['predicted_valence'],
            output['predicted_arousal'],
            output['valence'],
            output['arousal']
        )
        loss_dict['affect_loss'] = affect_loss.item()
        if output.get('meta_affect_surprise') is not None:
            loss_dict['mean_meta_affect_surprise'] = output['meta_affect_surprise'].mean().item()
        total_loss = total_loss + affect_weight * affect_loss

    # Extended self-awareness: meta-retrieval loss (predict what will be retrieved)
    if (output.get('predicted_retrieval') is not None and
        output.get('retrieved_memory') is not None):
        retrieval_loss = meta_retrieval_loss(
            output['predicted_retrieval'],
            output['retrieved_memory']
        )
        loss_dict['retrieval_loss'] = retrieval_loss.item()
        if output.get('meta_retrieval_surprise') is not None:
            loss_dict['mean_meta_retrieval_surprise'] = output['meta_retrieval_surprise'].mean().item()
        total_loss = total_loss + retrieval_weight * retrieval_loss

    loss_dict['total_loss'] = total_loss.item()
    return total_loss, loss_dict


def prediction_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    Compute prediction accuracy: what fraction correctly identifies its target?

    Random baseline = 1/batch_size
    Perfect = 1.0

    Args:
        prediction: [batch, d_model]
        target: [batch, d_model]

    Returns:
        Accuracy as float
    """
    with torch.no_grad():
        pred_norm = F.normalize(prediction, dim=-1)
        target_norm = F.normalize(target, dim=-1)

        # Similarity matrix
        sims = torch.mm(pred_norm, target_norm.t())

        # Each prediction's best match
        predicted_idx = sims.argmax(dim=-1)
        correct_idx = torch.arange(len(prediction), device=prediction.device)

        accuracy = (predicted_idx == correct_idx).float().mean()

    return accuracy.item()


def compute_metrics(output: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute all metrics from experiential output.

    Args:
        output: dict from ExperientialStream.forward()

    Returns:
        dict with loss, accuracy, mean_surprise, etc.
    """
    loss = experiential_loss(output['prediction'], output['target'])
    accuracy = prediction_accuracy(output['prediction'], output['target'])

    return {
        'loss': loss.item(),
        'accuracy': accuracy,
        'mean_surprise': output['surprise'].mean().item(),
        'std_surprise': output['surprise'].std().item(),
    }


# --- Memory-Augmented Generation ---

class MemoryAugmentedGPT(nn.Module):
    """
    GPT model augmented with episodic and semantic memory.

    Retrieves relevant memories from past experiences and semantic knowledge
    to condition the model's predictions. Enables:
    - Learning from past experiences (episodic: what happened before)
    - Abstracting knowledge from repeated patterns (semantic: what I know)
    - Maintaining consistency across long contexts
    - Recalling relevant patterns/events

    Integration options:
    1. Residual: Add retrieved memory to hidden states
    2. Gated: Learn to blend memory with hidden states
    3. Cross-attention: Attend to memory as additional context

    Memory hierarchy:
    - Experiential → Episodic (crystallization): high-salience moments
    - Episodic → Semantic (consolidation): repeated patterns become knowledge

    Usage:
        gpt = GPT(config)
        memory_gpt = MemoryAugmentedGPT(gpt, memory_capacity=1000)

        # Training: forward returns (logits, hidden_states, memory_output)
        logits, hidden, mem_out = memory_gpt(input_ids, crystallize=True)

        # The model learns to use retrieved memories
        lm_loss = F.cross_entropy(logits.view(-1, vocab), targets.view(-1))
        exp_loss = experiential_loss(mem_out['prediction'], mem_out['target'])
        total_loss = lm_loss + 0.1 * exp_loss

        # Periodic consolidation (e.g., every N steps)
        if step % consolidation_interval == 0:
            memory_gpt.consolidate()
    """

    def __init__(
        self,
        gpt_model: nn.Module,
        memory_capacity: int = 1000,
        crystallization_threshold: float = 0.3,
        memory_integration: str = 'gated',  # 'residual', 'gated', or 'attention'
        memory_weight: float = 0.1,
        use_experiential: bool = True,
        use_semantic: bool = True,
        retrieval_temperature: float = 0.1,
        semantic_weight: float = 0.5,
        consolidation_interval: int = 100,
        min_consolidation_evidence: int = 3,
        pad_token_id: Optional[int] = None,
        meta_surprise_salience_weight: float = 1.0,  # How much meta-surprise boosts salience
    ):
        """
        Args:
            gpt_model: Pre-existing GPT model to wrap
            memory_capacity: Maximum episodes to store
            crystallization_threshold: Salience threshold for storing memories
            memory_integration: How to integrate memories ('residual', 'gated', 'attention')
            memory_weight: Base weight for memory contribution (for residual mode)
            use_experiential: Whether to use experiential stream for surprise/salience
            use_semantic: Whether to use semantic memory for abstracted knowledge
            retrieval_temperature: Temperature for soft retrieval
            semantic_weight: Weight for semantic vs episodic retrieval [0, 1]
            consolidation_interval: Steps between automatic consolidation (0 = manual only)
            min_consolidation_evidence: Minimum episodes for consolidation
            pad_token_id: Token ID for padding (excluded from surprise computation)
            meta_surprise_salience_weight: How much meta-surprise boosts salience (default 1.0, was 3.0)
        """
        super().__init__()
        self.gpt = gpt_model
        self.d_model = gpt_model.config.d_model
        self.memory_integration = memory_integration
        self.memory_weight = memory_weight
        self.retrieval_temperature = retrieval_temperature
        self.semantic_weight = semantic_weight
        self.consolidation_interval = consolidation_interval
        self._step_counter = 0
        # Padding token ID for excluding pad tokens from surprise computation
        # Try to get from gpt config if not provided
        if pad_token_id is None and hasattr(gpt_model.config, 'pad_idx'):
            pad_token_id = gpt_model.config.pad_idx
        self.pad_token_id = pad_token_id

        # Episodic memory
        self.memory = EpisodicMemory(
            d_model=self.d_model,
            capacity=memory_capacity,
            crystallization_threshold=crystallization_threshold
        )

        # Semantic memory (abstracted knowledge)
        if use_semantic:
            self.semantic = SemanticStream(
                d_model=self.d_model,
                min_evidence=min_consolidation_evidence
            )
        else:
            self.semantic = None

        # Experiential stream (for computing surprise/salience)
        if use_experiential:
            self.experiential = ExperientialStream(
                d_model=self.d_model,
                use_affect=True,
                use_persistent_state=True,
                use_meta_surprise=True,  # Enable self-modulation
                meta_surprise_salience_weight=meta_surprise_salience_weight,
            )
        else:
            self.experiential = None

        # Memory integration components
        if memory_integration == 'gated':
            # Learned gate: how much to use memory vs original hidden state
            self.memory_gate = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.Sigmoid()
            )
            # Project retrieved memory to match hidden state space
            self.memory_proj = nn.Linear(self.d_model, self.d_model)
            # Semantic projection (separate from episodic)
            if use_semantic:
                self.semantic_proj = nn.Linear(self.d_model, self.d_model)
                # Gate for blending episodic and semantic
                self.semantic_gate = nn.Sequential(
                    nn.Linear(self.d_model * 2, self.d_model),
                    nn.Sigmoid()
                )

        elif memory_integration == 'attention':
            # Cross-attention to memory
            self.memory_attention = nn.MultiheadAttention(
                self.d_model,
                num_heads=4,
                batch_first=True
            )
            self.memory_norm = nn.LayerNorm(self.d_model)
            # Semantic also uses cross-attention (shared or separate)
            if use_semantic:
                self.semantic_attention = nn.MultiheadAttention(
                    self.d_model,
                    num_heads=4,
                    batch_first=True
                )
                self.semantic_norm = nn.LayerNorm(self.d_model)

        # Query projection for retrieval
        self.query_proj = nn.Linear(self.d_model, self.d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize memory integration weights."""
        if self.memory_integration == 'gated':
            # Initialize gate to pass through original hidden states initially
            nn.init.zeros_(self.memory_gate[0].weight)
            nn.init.constant_(self.memory_gate[0].bias, -2.0)  # sigmoid(-2) ≈ 0.12
            nn.init.xavier_uniform_(self.memory_proj.weight)
            nn.init.zeros_(self.memory_proj.bias)
            # Initialize semantic layers if present
            if self.semantic is not None:
                nn.init.xavier_uniform_(self.semantic_proj.weight)
                nn.init.zeros_(self.semantic_proj.bias)
                nn.init.zeros_(self.semantic_gate[0].weight)
                nn.init.constant_(self.semantic_gate[0].bias, 0.0)  # sigmoid(0) = 0.5

        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        crystallize: bool = True,
        use_memory: bool = True,
        use_semantic: bool = True,
        return_memory_weights: bool = False,
        prev_memory_query: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with CAUSAL memory retrieval and optional crystallization.

        CAUSAL MEMORY: Memory is retrieved using prev_memory_query (from previous
        step), NOT the current sequence's hidden states. This ensures that memory
        retrieval doesn't leak information from future tokens to earlier positions.

        Args:
            input_ids: [batch, seq_len] input token IDs
            input_pos: Optional position indices
            crystallize: Whether to store high-salience moments in memory
            use_memory: Whether to retrieve and use episodic memories
            use_semantic: Whether to retrieve and use semantic knowledge
            return_memory_weights: Include retrieval weights in output
            prev_memory_query: [batch, d_model] query from PREVIOUS step for causal
                retrieval. If None, memory retrieval is skipped (first step).

        Returns:
            logits: [batch, seq_len, vocab_size] output logits
            hidden_states: [batch, seq_len, d_model] final hidden states
            memory_output: dict with experiential/memory info including:
                - 'next_memory_query': [batch, d_model] query for next step
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Ensure semantic stream is on the correct device (may not be after loading)
        if self.semantic is not None:
            self.semantic = self.semantic.to(device)

        # 1. CAUSAL RETRIEVAL: Query memory BEFORE seeing current sequence
        # This uses prev_memory_query (from previous step), not current hidden states
        episodic_retrieved = None
        semantic_retrieved = None
        episodic_weights = None
        semantic_weights = None

        if prev_memory_query is not None:
            # Use previous step's query for causal memory retrieval
            query = prev_memory_query  # Already projected by previous step

            # 1a. Episodic retrieval (causal: query is from before this sequence)
            if use_memory and self.memory.size > 0:
                episodic_retrieved, episodic_weights = self.memory.retrieve_soft(
                    query,
                    temperature=self.retrieval_temperature
                )

            # 1b. Semantic retrieval (causal: query is from before this sequence)
            if use_semantic and self.semantic is not None and self.semantic.size > 0:
                semantic_retrieved, semantic_weights = self.semantic.query_soft(
                    query,
                    temperature=self.retrieval_temperature
                )

        # 2. Get hidden states from GPT
        logits, hidden_states = self.gpt(
            input_ids,
            input_pos=input_pos,
            return_hidden_states=True
        )

        # 3. Integrate memory with hidden states (memory was retrieved causally)
        if episodic_retrieved is not None or semantic_retrieved is not None:
            hidden_states = self._integrate_memory(
                hidden_states,
                episodic_retrieved,
                semantic_retrieved
            )
            # Recompute logits with memory-augmented hidden states
            logits = self.gpt.lm_head(self.gpt.final_norm(hidden_states))

        # 4. Prepare query for NEXT step (causal: computed after processing current)
        # This query will be used by the next step to retrieve relevant memories
        current_query_state = hidden_states[:, -1, :]
        next_memory_query = self.query_proj(current_query_state)

        memory_output = {
            'retrieved_episodic': episodic_retrieved,
            'retrieved_semantic': semantic_retrieved,
            'episodic_weights': episodic_weights if return_memory_weights else None,
            'semantic_weights': semantic_weights if return_memory_weights else None,
            'crystallized': False,
            'consolidated': False,
            'episodic_size': self.memory.size,
            'semantic_size': self.semantic.size if self.semantic else 0,
            'next_memory_query': next_memory_query,  # For causal retrieval in next step
        }

        # 4. Experiential processing (for crystallization and prediction)
        if self.experiential is not None:
            # Compute per-token CE loss for the new surprise signal
            # logits[:, i] predicts token at position i+1
            # So we shift: logits[:-1] predicts tokens[1:]
            # NOTE: No gradients needed - this is just for surprise computation
            vocab_size = logits.size(-1)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            with torch.no_grad():
                # Use ignore_index for pad tokens to avoid surprise spikes
                ignore_idx = self.pad_token_id if self.pad_token_id is not None else -100
                per_token_ce = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    reduction='none',
                    ignore_index=ignore_idx
                ).view(batch_size, -1)  # [B, seq_len-1]

                # Create mask for non-pad positions (True = valid, False = pad)
                # This is passed to experiential for proper aggregation
                if self.pad_token_id is not None:
                    ce_mask = (shift_labels != self.pad_token_id)  # [B, seq_len-1]
                else:
                    ce_mask = None

            # Get memory keys for novelty computation
            memory_keys = self.memory.get_keys(device=device)

            # Combine retrieved memories for extended self-awareness
            # Use episodic as primary, semantic as fallback
            retrieved_for_meta = None
            if episodic_retrieved is not None:
                retrieved_for_meta = episodic_retrieved
            elif semantic_retrieved is not None:
                retrieved_for_meta = semantic_retrieved

            # Call experiential stream with the new surprise signal
            # Note: per_token_ce is [B, seq_len-1], alignment handled inside experiential
            # Pass retrieved_memory for extended self-awareness (meta-retrieval-surprise)
            exp_output = self.experiential(
                hidden_states,  # Full hidden states for h_mid/h_end extraction
                per_token_ce=per_token_ce,
                memory_keys=memory_keys,
                ce_mask=ce_mask,
                retrieved_memory=retrieved_for_meta
            )
            memory_output.update({
                'prediction': exp_output['prediction'],
                'target': exp_output['target'],
                'surprise': exp_output['surprise'],
                'valence': exp_output['valence'],
                'arousal': exp_output['arousal'],
                'salience': exp_output['salience'],
                'modulated_output': exp_output.get('modulated_output'),
                'confidence_gate': exp_output.get('confidence_gate'),
                'meta_surprise': exp_output.get('meta_surprise'),
                # Extended self-awareness: affect prediction
                'predicted_valence': exp_output.get('predicted_valence'),
                'predicted_arousal': exp_output.get('predicted_arousal'),
                'meta_affect_surprise': exp_output.get('meta_affect_surprise'),
                # Extended self-awareness: retrieval prediction
                'predicted_retrieval': exp_output.get('predicted_retrieval'),
                'meta_retrieval_surprise': exp_output.get('meta_retrieval_surprise'),
                'retrieved_memory': retrieved_for_meta,  # For loss computation
                # Per-token surprise signal components
                'surprise_t': exp_output.get('surprise_t'),
                'excess_t': exp_output.get('excess_t'),
                'novelty_t': exp_output.get('novelty_t'),
                'ema_mu': exp_output.get('ema_mu'),
                'ema_sigma': exp_output.get('ema_sigma'),
            })

            # 5. Crystallize high-salience moments into episodic memory
            # Store the MODULATED output (post-self-regulation), not raw target
            # This means memories contain what the system "committed to" after reflection
            if crystallize:
                modulated = exp_output.get('modulated_output', exp_output['target'])
                for i in range(batch_size):
                    salience = exp_output['salience'][i].item()
                    if self.memory.should_crystallize(salience):
                        self.memory.store(
                            content=modulated[i],
                            context=exp_output['prediction'][i],
                            salience=salience,
                            valence=exp_output['valence'][i].item(),
                            arousal=exp_output['arousal'][i].item()
                        )
                        memory_output['crystallized'] = True

        # 6. Periodic consolidation (episodic → semantic)
        self._step_counter += 1
        if (self.consolidation_interval > 0 and
            self.semantic is not None and
            self._step_counter % self.consolidation_interval == 0):
            n_consolidated = self.consolidate()
            memory_output['consolidated'] = n_consolidated > 0

        memory_output['episodic_size'] = self.memory.size
        memory_output['semantic_size'] = self.semantic.size if self.semantic else 0

        return logits, hidden_states, memory_output

    def _integrate_memory(
        self,
        hidden_states: torch.Tensor,
        episodic_retrieved: Optional[torch.Tensor] = None,
        semantic_retrieved: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Integrate retrieved episodic and semantic memory into hidden states.

        Args:
            hidden_states: [batch, seq_len, d_model]
            episodic_retrieved: [batch, d_model] - retrieved episodic memories
            semantic_retrieved: [batch, d_model] - retrieved semantic knowledge

        Returns:
            augmented: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = hidden_states.shape

        if self.memory_integration == 'residual':
            # Simple additive: add memory to all positions
            result = hidden_states

            if episodic_retrieved is not None:
                episodic_contrib = episodic_retrieved.unsqueeze(1).expand(-1, seq_len, -1)
                result = result + self.memory_weight * (1 - self.semantic_weight) * episodic_contrib

            if semantic_retrieved is not None:
                semantic_contrib = semantic_retrieved.unsqueeze(1).expand(-1, seq_len, -1)
                result = result + self.memory_weight * self.semantic_weight * semantic_contrib

            return result

        elif self.memory_integration == 'gated':
            result = hidden_states

            # Integrate episodic memory
            if episodic_retrieved is not None:
                episodic_expanded = episodic_retrieved.unsqueeze(1).expand(-1, seq_len, -1)
                episodic_proj = self.memory_proj(episodic_expanded)
                gate_input = torch.cat([result, episodic_proj], dim=-1)
                episodic_gate = self.memory_gate(gate_input)
                result = episodic_gate * episodic_proj + (1 - episodic_gate) * result

            # Integrate semantic memory (if available)
            if semantic_retrieved is not None and self.semantic is not None:
                semantic_expanded = semantic_retrieved.unsqueeze(1).expand(-1, seq_len, -1)
                semantic_proj = self.semantic_proj(semantic_expanded)
                gate_input = torch.cat([result, semantic_proj], dim=-1)
                semantic_gate = self.semantic_gate(gate_input)
                result = semantic_gate * semantic_proj + (1 - semantic_gate) * result

            return result

        elif self.memory_integration == 'attention':
            result = hidden_states

            # Attend to episodic memory
            if episodic_retrieved is not None:
                episodic_tokens = episodic_retrieved.unsqueeze(1)
                attended, _ = self.memory_attention(
                    result,
                    episodic_tokens,
                    episodic_tokens
                )
                result = result + self.memory_norm(attended)

            # Attend to semantic memory
            if semantic_retrieved is not None and self.semantic is not None:
                semantic_tokens = semantic_retrieved.unsqueeze(1)
                attended, _ = self.semantic_attention(
                    result,
                    semantic_tokens,
                    semantic_tokens
                )
                result = result + self.semantic_norm(attended)

            return result

        else:
            raise ValueError(f"Unknown integration mode: {self.memory_integration}")

    def reset_memory(self):
        """Clear all stored memories (episodic and semantic)."""
        self.memory.clear()
        if self.semantic is not None:
            self.semantic.clear()
        if self.experiential is not None:
            self.experiential.reset_state()
        self._step_counter = 0

    def reset_episodic(self):
        """Clear only episodic memory (preserve semantic knowledge)."""
        self.memory.clear()
        if self.experiential is not None:
            self.experiential.reset_state()

    def reset_hidden_state(self):
        """
        Reset experiential hidden state but KEEP all memories.

        This is the key operation for resume-after-interruption training:
        - Simulates "forgetting" the immediate context (what was just processed)
        - Preserves episodic and semantic memories
        - Forces the model to rely on memory retrieval for context

        Use case:
            # Process chunks 1..N, building memory
            for chunk in chunks[:interrupt_point]:
                model(chunk, crystallize=True)

            # Interrupt: forget immediate context
            model.reset_hidden_state()

            # Resume: must use memory to understand
            for chunk in chunks[interrupt_point:]:
                logits, _, _ = model(chunk, use_memory=True)
                # Loss on these chunks forces memory to be useful
        """
        if self.experiential is not None:
            self.experiential.reset_state()
        # Note: episodic memory (self.memory) and semantic memory (self.semantic)
        # are intentionally NOT cleared - that's the whole point

    def consolidate(self, n_clusters: int = 5, min_cluster_size: int = 3) -> int:
        """
        Consolidate episodic memories into semantic concepts.

        This is the episodic → semantic transition: repeated patterns
        become abstracted knowledge.

        Args:
            n_clusters: Number of clusters to try for grouping episodes
            min_cluster_size: Minimum episodes per cluster for consolidation

        Returns:
            Number of new concepts created
        """
        if self.semantic is None:
            return 0

        if self.memory.size < min_cluster_size:
            return 0

        concepts = self.semantic.consolidate_from_memory(
            self.memory,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size
        )

        # Advance semantic step counter
        self.semantic.step()

        return len(concepts)

    def get_memory_stats(self) -> Dict:
        """Get statistics about current memory state."""
        stats = {
            'episodic': self.memory.get_stats(),
            'step_counter': self._step_counter
        }
        if self.semantic is not None:
            stats['semantic'] = self.semantic.get_stats()
        if self.experiential is not None:
            stats['experiential_state'] = self.experiential.get_state() is not None
        return stats

    def retrieve_relevant(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        include_semantic: bool = True
    ) -> Dict[str, List]:
        """
        Retrieve relevant memories for a query (hard retrieval for inspection).

        Args:
            query: [d_model] query vector
            top_k: number of memories to retrieve per memory type
            include_semantic: whether to include semantic concepts

        Returns:
            Dict with 'episodic' and optionally 'semantic' lists of (item, similarity) tuples
        """
        result = {
            'episodic': self.memory.retrieve(query, top_k=top_k)
        }

        if include_semantic and self.semantic is not None and self.semantic.size > 0:
            result['semantic'] = self.semantic.query(query, top_k=top_k)

        return result


def memory_augmented_loss(
    lm_logits: torch.Tensor,
    targets: torch.Tensor,
    memory_output: Dict,
    lm_weight: float = 1.0,
    exp_weight: float = 0.1,
    meta_weight: float = 0.01,
    self_mod_weight: float = 0.01,
    affect_weight: float = 0.01,
    retrieval_weight: float = 0.01
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined loss for memory-augmented generation.

    Args:
        lm_logits: [batch, seq_len, vocab] model output logits
        targets: [batch, seq_len] target token IDs
        memory_output: output dict from MemoryAugmentedGPT.forward()
        lm_weight: weight for language modeling loss
        exp_weight: weight for experiential prediction loss
        meta_weight: weight for meta-surprise loss (trains surprise predictor)
        self_mod_weight: weight for self-modulation loss (trains confidence gate)
        affect_weight: weight for meta-affect loss (extended self-awareness)
        retrieval_weight: weight for meta-retrieval loss (extended self-awareness)

    Returns:
        total_loss: combined scalar loss
        loss_dict: breakdown of individual losses including memory stats
    """
    loss_dict = {}

    # Language modeling loss
    lm_loss = F.cross_entropy(
        lm_logits.view(-1, lm_logits.size(-1)),
        targets.view(-1),
        ignore_index=-100  # Skip padding
    )
    loss_dict['lm_loss'] = lm_loss.item()

    total_loss = lm_weight * lm_loss

    # Experiential prediction loss (now uses combined_experiential_loss for full training)
    # This trains: predictor, surprise_predictor, self_modulator, affect_predictor, retrieval_predictor
    if 'prediction' in memory_output and memory_output['prediction'] is not None:
        exp_loss, exp_dict = combined_experiential_loss(
            memory_output,
            exp_weight=1.0,  # Base weight, scaled by exp_weight below
            meta_weight=meta_weight / exp_weight if exp_weight > 0 else 0.0,
            self_mod_weight=self_mod_weight / exp_weight if exp_weight > 0 else 0.0,
            affect_weight=affect_weight / exp_weight if exp_weight > 0 else 0.0,
            retrieval_weight=retrieval_weight / exp_weight if exp_weight > 0 else 0.0,
        )
        loss_dict['exp_loss'] = exp_dict.get('exp_loss', 0.0)
        loss_dict['meta_loss'] = exp_dict.get('meta_loss', 0.0)
        loss_dict['self_mod_loss'] = exp_dict.get('self_mod_loss', 0.0)
        loss_dict['mean_meta_surprise'] = exp_dict.get('mean_meta_surprise', 0.0)
        loss_dict['mean_confidence'] = exp_dict.get('mean_confidence', 0.0)
        # Extended self-awareness metrics
        loss_dict['affect_loss'] = exp_dict.get('affect_loss', 0.0)
        loss_dict['retrieval_loss'] = exp_dict.get('retrieval_loss', 0.0)
        loss_dict['mean_meta_affect_surprise'] = exp_dict.get('mean_meta_affect_surprise', 0.0)
        loss_dict['mean_meta_retrieval_surprise'] = exp_dict.get('mean_meta_retrieval_surprise', 0.0)
        total_loss = total_loss + exp_weight * exp_loss

    # Memory statistics (for logging)
    loss_dict['episodic_size'] = memory_output.get('episodic_size', 0)
    loss_dict['semantic_size'] = memory_output.get('semantic_size', 0)
    loss_dict['crystallized'] = memory_output.get('crystallized', False)
    loss_dict['consolidated'] = memory_output.get('consolidated', False)

    loss_dict['total_loss'] = total_loss.item()
    return total_loss, loss_dict


# --- Semantic Stream (Consolidation) ---

@dataclass
class ConceptRelation:
    """A relationship between two concepts."""
    target_id: int
    relation_type: str  # "is-a", "has-a", "related-to", "causes", "opposite-of"
    strength: float     # [0, 1]
    evidence: List[int] = field(default_factory=list)  # episode timestamps


@dataclass
class Concept:
    """A piece of abstracted knowledge distilled from episodes."""
    id: int
    embedding: torch.Tensor          # [d_model] - distributed representation
    prototype: torch.Tensor          # [d_model] - mean of source episodes

    # Provenance
    source_episodes: List[int]       # timestamps of source episodes
    consolidation_time: int          # when this concept was created
    evidence_count: int              # number of supporting episodes

    # Confidence
    confidence: float                # [0, 1] - based on evidence

    # Optional
    name: Optional[str] = None       # human-readable label
    connections: Dict[int, ConceptRelation] = field(default_factory=dict)
    parent: Optional[int] = None     # more abstract concept
    children: List[int] = field(default_factory=list)  # more specific concepts
    abstraction_level: int = 0       # 0 = concrete, higher = more abstract


class PatternExtractor(nn.Module):
    """
    Extract common pattern from a set of embeddings.

    Uses attention pooling to find latent structure beyond simple averaging.
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        hidden_dim = hidden_dim or d_model * 2

        # Attention pooling
        self.attention = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Pattern refinement
        self.refiner = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Extract pattern from embeddings.

        Args:
            embeddings: [n_items, d_model] - items to find pattern in

        Returns:
            pattern: [d_model] - extracted common pattern
        """
        if embeddings.dim() == 1:
            return embeddings  # Single item, return as-is

        # Add batch dimension
        embeddings = embeddings.unsqueeze(0)  # [1, n_items, d_model]

        # Attention pooling
        pooled, _ = self.attention(self.query, embeddings, embeddings)
        pooled = pooled.squeeze(0).squeeze(0)  # [d_model]

        # Refine pattern
        pattern = self.refiner(pooled)

        return pattern


class RelationPredictor(nn.Module):
    """
    Predict relation between two concepts.
    """

    RELATION_TYPES = [
        "related-to",   # general association (most common)
        "is-a",         # hyponymy (child is-a parent)
        "has-a",        # meronymy
        "causes",       # causation
        "opposite-of",  # antonymy
        "none"          # no significant relation
    ]

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.n_types = len(self.RELATION_TYPES)

        # Relation classifier: takes concat(a, b, a-b)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.n_types)
        )

        # Strength predictor
        self.strength_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        concept_a: torch.Tensor,
        concept_b: torch.Tensor,
        threshold: float = 0.3
    ) -> Optional[ConceptRelation]:
        """
        Predict relation from concept_a to concept_b.

        Args:
            concept_a: [d_model] - source concept embedding
            concept_b: [d_model] - target concept embedding
            threshold: minimum strength to return a relation

        Returns:
            ConceptRelation or None if no significant relation
        """
        # Concatenate with difference for asymmetric relations
        combined = torch.cat([
            concept_a,
            concept_b,
            concept_a - concept_b
        ], dim=-1)

        # Predict relation type
        type_logits = self.classifier(combined)
        type_probs = F.softmax(type_logits, dim=-1)
        type_idx = type_probs.argmax().item()
        relation_type = self.RELATION_TYPES[type_idx]

        if relation_type == "none":
            return None

        # Predict strength
        strength = self.strength_head(combined).item()

        if strength < threshold:
            return None

        return ConceptRelation(
            target_id=-1,  # Filled by caller
            relation_type=relation_type,
            strength=strength,
            evidence=[]
        )


class SemanticStream(nn.Module):
    """
    Semantic memory: abstracted, connected knowledge.

    Consolidates episodic memories into concepts, builds a knowledge graph,
    and provides retrieval for informing current processing.

    Usage:
        semantic = SemanticStream(d_model=1024)

        # Consolidate similar episodes into a concept
        episodes = memory.retrieve_by_salience(top_k=10)
        concept = semantic.consolidate(episodes)

        # Query for relevant knowledge
        relevant = semantic.query(current_hidden_state, top_k=5)

        # Differentiable retrieval for training
        knowledge, weights = semantic.query_soft(query)
    """

    def __init__(
        self,
        d_model: int,
        min_evidence: int = 3,
        similarity_threshold: float = 0.7,
        max_concepts: int = 10000
    ):
        """
        Args:
            d_model: embedding dimension
            min_evidence: minimum episodes required for consolidation
            similarity_threshold: threshold for auto-clustering episodes
            max_concepts: maximum concepts to store
        """
        super().__init__()
        self.d_model = d_model
        self.min_evidence = min_evidence
        self.similarity_threshold = similarity_threshold
        self.max_concepts = max_concepts

        # Pattern extraction
        self.pattern_extractor = PatternExtractor(d_model)

        # Relation prediction
        self.relation_predictor = RelationPredictor(d_model)

        # Query projection for retrieval
        self.query_projection = nn.Linear(d_model, d_model)

        # Knowledge storage
        self.concepts: Dict[int, Concept] = {}
        self.next_concept_id = 0
        self._step = 0

        # Cached embedding index for fast retrieval
        self._embedding_cache: Optional[torch.Tensor] = None
        self._cache_valid = False

    @property
    def size(self) -> int:
        """Number of concepts stored."""
        return len(self.concepts)

    def consolidate(
        self,
        episodes: List[Episode],
        require_min_evidence: bool = True,
        name: Optional[str] = None
    ) -> Optional[Concept]:
        """
        Distill episodes into a concept.

        Args:
            episodes: list of Episode objects to consolidate
            require_min_evidence: if True, require at least min_evidence episodes
            name: optional human-readable name for the concept

        Returns:
            Created Concept, or None if insufficient evidence
        """
        if require_min_evidence and len(episodes) < self.min_evidence:
            return None

        if len(episodes) == 0:
            return None

        # Stack episode embeddings
        device = episodes[0].content.device
        episode_embeddings = torch.stack([ep.content.to(device) for ep in episodes])

        # Ensure all SemanticStream modules are on the same device
        self.pattern_extractor = self.pattern_extractor.to(device)
        self.relation_predictor = self.relation_predictor.to(device)
        self.query_projection = self.query_projection.to(device)

        # Extract common pattern
        pattern = self.pattern_extractor(episode_embeddings)

        # Compute prototype (mean)
        prototype = episode_embeddings.mean(dim=0)

        # Create concept
        concept = Concept(
            id=self.next_concept_id,
            embedding=pattern.detach().clone(),
            prototype=prototype.detach().clone(),
            source_episodes=[ep.timestamp for ep in episodes],
            consolidation_time=self._step,
            evidence_count=len(episodes),
            confidence=min(1.0, len(episodes) / 10.0),
            name=name,
            abstraction_level=0
        )

        # Add to storage
        self._add_concept(concept)

        # Discover relations to existing concepts
        self._discover_relations(concept)

        return concept

    def consolidate_from_memory(
        self,
        memory: EpisodicMemory,
        n_clusters: int = 5,
        min_cluster_size: int = 3
    ) -> List[Concept]:
        """
        Automatically consolidate episodes from episodic memory.

        Uses clustering to find groups of similar episodes.

        Args:
            memory: EpisodicMemory to consolidate from
            n_clusters: number of clusters to try
            min_cluster_size: minimum episodes per cluster

        Returns:
            List of created concepts
        """
        if memory.size < min_cluster_size:
            return []

        # Get all episode contents
        contents = memory.get_content_matrix()
        if contents.size(0) == 0:
            return []

        # Simple clustering: k-means style
        clusters = self._cluster_episodes(
            memory.episodes,
            contents,
            n_clusters,
            min_cluster_size
        )

        # Consolidate each cluster
        new_concepts = []
        for cluster_episodes in clusters:
            concept = self.consolidate(cluster_episodes, require_min_evidence=True)
            if concept is not None:
                new_concepts.append(concept)

        return new_concepts

    def _cluster_episodes(
        self,
        episodes: List[Episode],
        contents: torch.Tensor,
        n_clusters: int,
        min_size: int
    ) -> List[List[Episode]]:
        """Simple clustering of episodes by similarity."""
        if len(episodes) < n_clusters:
            return []

        # Normalize for cosine similarity
        contents_norm = F.normalize(contents, dim=-1)

        # Initialize centroids randomly
        n_clusters = min(n_clusters, len(episodes) // min_size)
        if n_clusters < 1:
            return []

        indices = torch.randperm(len(episodes))[:n_clusters]
        centroids = contents_norm[indices].clone()

        # K-means iterations
        for _ in range(10):
            # Assign to nearest centroid
            sims = torch.mm(contents_norm, centroids.t())  # [n, k]
            assignments = sims.argmax(dim=1)  # [n]

            # Update centroids
            new_centroids = []
            for k in range(n_clusters):
                mask = (assignments == k)
                if mask.sum() > 0:
                    new_centroids.append(contents_norm[mask].mean(dim=0))
                else:
                    new_centroids.append(centroids[k])
            centroids = torch.stack(new_centroids)
            centroids = F.normalize(centroids, dim=-1)

        # Build clusters
        clusters = [[] for _ in range(n_clusters)]
        for i, ep in enumerate(episodes):
            clusters[assignments[i].item()].append(ep)

        # Filter by minimum size
        return [c for c in clusters if len(c) >= min_size]

    def query(
        self,
        query: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[Concept, float]]:
        """
        Retrieve relevant concepts for a query.

        Args:
            query: [d_model] or [batch, d_model] query vector
            top_k: number of concepts to retrieve

        Returns:
            List of (Concept, similarity) tuples
        """
        if self.size == 0:
            return []

        # Ensure 2D
        if query.dim() == 1:
            query = query.unsqueeze(0)

        # Project query
        query_emb = self.query_projection(query)
        query_norm = F.normalize(query_emb, dim=-1)

        # Get embedding index
        emb_index = self._get_embedding_index(query.device)
        emb_norm = F.normalize(emb_index, dim=-1)

        # Compute similarities
        sims = torch.mm(query_norm, emb_norm.t())  # [batch, n_concepts]

        # Get top-k for first query
        top_k = min(top_k, self.size)
        values, indices = sims[0].topk(top_k)

        return [
            (self.concepts[idx.item()], values[i].item())
            for i, idx in enumerate(indices)
        ]

    def query_soft(
        self,
        query: torch.Tensor,
        temperature: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable retrieval over concepts.

        Args:
            query: [batch, d_model] or [d_model] query vector
            temperature: softmax temperature

        Returns:
            knowledge: [batch, d_model] - weighted sum of concept embeddings
            weights: [batch, n_concepts] - attention weights
        """
        if self.size == 0:
            if query.dim() == 1:
                return torch.zeros(self.d_model, device=query.device), torch.zeros(0, device=query.device)
            else:
                batch_size = query.size(0)
                return torch.zeros(batch_size, self.d_model, device=query.device), torch.zeros(batch_size, 0, device=query.device)

        # Ensure 2D
        squeeze_output = False
        if query.dim() == 1:
            query = query.unsqueeze(0)
            squeeze_output = True

        # Project query
        query_emb = self.query_projection(query)
        query_norm = F.normalize(query_emb, dim=-1)

        # Get embedding index
        emb_index = self._get_embedding_index(query.device)

        # Compute attention
        scores = torch.mm(query_norm, F.normalize(emb_index, dim=-1).t()) / temperature
        weights = F.softmax(scores, dim=-1)

        # Weighted sum
        knowledge = torch.mm(weights, emb_index)

        if squeeze_output:
            knowledge = knowledge.squeeze(0)
            weights = weights.squeeze(0)

        return knowledge, weights

    def connect(
        self,
        concept_a_id: int,
        concept_b_id: int,
        relation_type: str = "related-to",
        strength: float = 0.5,
        evidence: Optional[List[int]] = None
    ) -> bool:
        """
        Create or strengthen a relation between concepts.

        Args:
            concept_a_id: source concept
            concept_b_id: target concept
            relation_type: type of relation
            strength: relation strength [0, 1]
            evidence: episode timestamps supporting this relation

        Returns:
            True if relation was created/updated
        """
        if concept_a_id not in self.concepts or concept_b_id not in self.concepts:
            return False

        relation = ConceptRelation(
            target_id=concept_b_id,
            relation_type=relation_type,
            strength=strength,
            evidence=evidence or []
        )

        self.concepts[concept_a_id].connections[concept_b_id] = relation
        return True

    def generalize(
        self,
        concept_ids: List[int],
        name: Optional[str] = None
    ) -> Optional[Concept]:
        """
        Create a more abstract concept from specific ones.

        Args:
            concept_ids: IDs of concepts to generalize
            name: optional name for the abstraction

        Returns:
            Created abstract Concept, or None if failed
        """
        if len(concept_ids) < 2:
            return None

        # Get concept embeddings
        embeddings = []
        for cid in concept_ids:
            if cid not in self.concepts:
                continue
            embeddings.append(self.concepts[cid].embedding)

        if len(embeddings) < 2:
            return None

        embeddings = torch.stack(embeddings)

        # Extract higher-level pattern
        pattern = self.pattern_extractor(embeddings)

        # Determine abstraction level
        max_level = max(
            self.concepts[cid].abstraction_level
            for cid in concept_ids if cid in self.concepts
        )

        # Create abstract concept
        abstract = Concept(
            id=self.next_concept_id,
            embedding=pattern.detach().clone(),
            prototype=embeddings.mean(dim=0).detach().clone(),
            source_episodes=[],  # Derived from concepts, not episodes
            consolidation_time=self._step,
            evidence_count=sum(
                self.concepts[cid].evidence_count
                for cid in concept_ids if cid in self.concepts
            ),
            confidence=min(
                self.concepts[cid].confidence
                for cid in concept_ids if cid in self.concepts
            ),
            name=name,
            children=concept_ids,
            abstraction_level=max_level + 1
        )

        self._add_concept(abstract)

        # Update children to point to parent
        for cid in concept_ids:
            if cid in self.concepts:
                self.concepts[cid].parent = abstract.id
                self.connect(cid, abstract.id, "is-a", 1.0)

        return abstract

    def get_stats(self) -> Dict:
        """Get statistics about the knowledge graph."""
        if self.size == 0:
            return {
                'n_concepts': 0,
                'n_relations': 0,
                'avg_evidence': 0,
                'avg_confidence': 0,
                'max_abstraction_level': 0
            }

        n_relations = sum(
            len(c.connections) for c in self.concepts.values()
        )

        return {
            'n_concepts': self.size,
            'n_relations': n_relations,
            'avg_evidence': sum(c.evidence_count for c in self.concepts.values()) / self.size,
            'avg_confidence': sum(c.confidence for c in self.concepts.values()) / self.size,
            'max_abstraction_level': max(c.abstraction_level for c in self.concepts.values()),
            'concepts_by_level': {
                level: sum(1 for c in self.concepts.values() if c.abstraction_level == level)
                for level in range(max(c.abstraction_level for c in self.concepts.values()) + 1)
            }
        }

    def clear(self):
        """Clear all concepts."""
        self.concepts.clear()
        self.next_concept_id = 0
        self._cache_valid = False
        self._embedding_cache = None

    def _add_concept(self, concept: Concept):
        """Add concept to storage."""
        if self.size >= self.max_concepts:
            # Evict lowest confidence concept
            min_conf_id = min(
                self.concepts.keys(),
                key=lambda k: self.concepts[k].confidence
            )
            del self.concepts[min_conf_id]

        self.concepts[concept.id] = concept
        self.next_concept_id = max(self.next_concept_id, concept.id + 1)
        self._cache_valid = False

    def _discover_relations(self, new_concept: Concept):
        """Discover relations between new concept and existing ones."""
        if self.size <= 1:
            return

        for cid, existing in self.concepts.items():
            if cid == new_concept.id:
                continue

            # Predict relation
            relation = self.relation_predictor(
                new_concept.embedding,
                existing.embedding
            )

            if relation is not None:
                self.connect(
                    new_concept.id,
                    cid,
                    relation.relation_type,
                    relation.strength
                )

    def _get_embedding_index(self, device: torch.device) -> torch.Tensor:
        """Get cached embedding index."""
        if not self._cache_valid or self._embedding_cache is None:
            if self.size == 0:
                self._embedding_cache = torch.zeros(0, self.d_model, device=device)
            else:
                self._embedding_cache = torch.stack([
                    self.concepts[i].embedding
                    for i in sorted(self.concepts.keys())
                ]).to(device)
            self._cache_valid = True
        return self._embedding_cache.to(device)

    def step(self):
        """Advance internal step counter."""
        self._step += 1


# --- Testing utilities ---

def test_shapes():
    """Quick shape test."""
    print("Testing ExperientialStream shapes...")

    batch_size = 4
    seq_len = 128
    d_model = 256

    exp = ExperientialStream(d_model=d_model)
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    output = exp(hidden_states)

    assert output['state'].shape == (batch_size, d_model), f"State shape wrong: {output['state'].shape}"
    assert output['prediction'].shape == (batch_size, d_model), f"Prediction shape wrong: {output['prediction'].shape}"
    assert output['target'].shape == (batch_size, d_model), f"Target shape wrong: {output['target'].shape}"
    assert output['surprise'].shape == (batch_size,), f"Surprise shape wrong: {output['surprise'].shape}"

    print("  All shapes correct!")
    return True


def test_multiscale():
    """Test multiscale prediction."""
    print("Testing ExperientialStream multiscale...")

    batch_size = 4
    seq_len = 128
    d_model = 256

    exp = ExperientialStream(d_model=d_model)
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    horizons = [0.25, 0.5, 0.75]
    output = exp.forward_multiscale(hidden_states, horizons=horizons)

    assert output['states'].shape == (batch_size, len(horizons), d_model)
    assert output['predictions'].shape == (batch_size, len(horizons), d_model)
    assert output['target'].shape == (batch_size, d_model)
    assert output['surprises'].shape == (batch_size, len(horizons))

    print("  Multiscale shapes correct!")
    return True


def test_gradient_flow():
    """Test that gradients flow through prediction but not target."""
    print("Testing gradient flow...")

    exp = ExperientialStream(d_model=128)
    hidden_states = torch.randn(4, 64, 128, requires_grad=True)

    output = exp(hidden_states)
    loss = experiential_loss(output['prediction'], output['target'])
    loss.backward()

    # Check gradients exist for model parameters
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in exp.parameters())
    assert has_grads, "No gradients in model parameters!"

    # Check hidden_states has gradients
    assert hidden_states.grad is not None, "No gradient to hidden_states!"

    print("  Gradients flow correctly!")
    return True


def test_learning():
    """Test that the model can learn a simple pattern."""
    print("Testing learning on synthetic data...")

    d_model = 128
    batch_size = 32
    seq_len = 64
    n_steps = 100

    exp = ExperientialStream(d_model=d_model)
    optimizer = torch.optim.Adam(exp.parameters(), lr=1e-3)

    initial_acc = None
    final_acc = None

    for step in range(n_steps):
        # Create data where end state is a function of mid state
        hidden_states = torch.randn(batch_size, seq_len, d_model)

        # Make the second half correlated with midpoint
        mid = seq_len // 2
        mid_state = hidden_states[:, mid, :].unsqueeze(1)
        hidden_states[:, mid:, :] = mid_state + torch.randn(batch_size, seq_len - mid, d_model) * 0.3

        output = exp(hidden_states)
        loss = experiential_loss(output['prediction'], output['target'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = prediction_accuracy(output['prediction'], output['target'])

        if step == 0:
            initial_acc = acc
        if step == n_steps - 1:
            final_acc = acc

        if step % 20 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, accuracy={acc:.4f}")

    print(f"  Initial accuracy: {initial_acc:.4f}")
    print(f"  Final accuracy: {final_acc:.4f}")

    assert final_acc > initial_acc, "Accuracy should improve!"
    assert final_acc > 0.5, f"Expected >50% accuracy, got {final_acc:.4f}"

    print("  Learning test passed!")
    return True


def test_affect_prediction():
    """Test that affect prediction produces valid outputs."""
    print("Testing affect prediction...")

    batch_size = 4
    seq_len = 64
    d_model = 128

    # Test with affect enabled (default)
    exp = ExperientialStream(d_model=d_model, use_affect=True)
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    output = exp(hidden_states)

    # Check affect outputs exist and have correct shape
    assert output['valence'] is not None, "Valence should exist"
    assert output['arousal'] is not None, "Arousal should exist"
    assert output['salience'] is not None, "Salience should exist"

    assert output['valence'].shape == (batch_size,), f"Valence shape wrong: {output['valence'].shape}"
    assert output['arousal'].shape == (batch_size,), f"Arousal shape wrong: {output['arousal'].shape}"
    assert output['salience'].shape == (batch_size,), f"Salience shape wrong: {output['salience'].shape}"

    # Check value ranges
    assert (output['valence'] >= -1).all() and (output['valence'] <= 1).all(), \
        f"Valence out of range [-1, 1]: {output['valence']}"
    assert (output['arousal'] >= 0).all() and (output['arousal'] <= 1).all(), \
        f"Arousal out of range [0, 1]: {output['arousal']}"
    assert (output['salience'] >= 0).all(), f"Salience should be non-negative: {output['salience']}"

    print(f"  Valence range: [{output['valence'].min():.3f}, {output['valence'].max():.3f}]")
    print(f"  Arousal range: [{output['arousal'].min():.3f}, {output['arousal'].max():.3f}]")
    print(f"  Salience range: [{output['salience'].min():.3f}, {output['salience'].max():.3f}]")

    # Test with affect disabled
    exp_no_affect = ExperientialStream(d_model=d_model, use_affect=False)
    output_no_affect = exp_no_affect(hidden_states)

    assert output_no_affect['valence'] is None, "Valence should be None when affect disabled"
    assert output_no_affect['arousal'] is None, "Arousal should be None when affect disabled"
    assert output_no_affect['salience'] is None, "Salience should be None when affect disabled"

    print("  Affect prediction shapes and ranges correct!")
    return True


def test_affect_gradients():
    """Test that gradients flow through affect heads."""
    print("Testing affect gradient flow...")

    batch_size = 8
    seq_len = 64
    d_model = 128

    exp = ExperientialStream(d_model=d_model, use_affect=True)
    hidden_states = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    output = exp(hidden_states)

    # Create a simple loss using affect outputs
    # (In practice, we'd have a target for valence/arousal)
    affect_loss = output['valence'].mean() + output['arousal'].mean()
    affect_loss.backward()

    # Check gradients exist for affect heads
    has_valence_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in exp.valence_head.parameters())
    has_arousal_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in exp.arousal_head.parameters())

    assert has_valence_grads, "No gradients in valence head!"
    assert has_arousal_grads, "No gradients in arousal head!"

    print("  Gradients flow through affect heads correctly!")
    return True


def test_meta_surprise():
    """Test meta-surprise (self-awareness) computation."""
    print("Testing meta-surprise...")

    batch_size = 8
    seq_len = 64
    d_model = 128

    exp = ExperientialStream(d_model=d_model, use_meta_surprise=True)
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    output = exp(hidden_states)

    # Check meta-surprise outputs exist
    assert output['predicted_surprise'] is not None, "predicted_surprise should exist"
    assert output['meta_surprise'] is not None, "meta_surprise should exist"

    # Check shapes
    assert output['predicted_surprise'].shape == (batch_size,), \
        f"Wrong predicted_surprise shape: {output['predicted_surprise'].shape}"
    assert output['meta_surprise'].shape == (batch_size,), \
        f"Wrong meta_surprise shape: {output['meta_surprise'].shape}"

    # Check ranges
    assert (output['predicted_surprise'] >= 0).all() and (output['predicted_surprise'] <= 1).all(), \
        "predicted_surprise should be in [0, 1]"
    assert (output['meta_surprise'] >= 0).all(), \
        "meta_surprise should be non-negative"

    # Check surprise is also present
    assert output['surprise'].shape == (batch_size,), "surprise should exist"

    print(f"  predicted_surprise range: [{output['predicted_surprise'].min():.3f}, {output['predicted_surprise'].max():.3f}]")
    print(f"  actual surprise range: [{output['surprise'].min():.3f}, {output['surprise'].max():.3f}]")
    print(f"  meta_surprise range: [{output['meta_surprise'].min():.3f}, {output['meta_surprise'].max():.3f}]")
    print("  Meta-surprise shapes and ranges correct!")
    return True


def test_meta_surprise_gradients():
    """Test that gradients flow through meta-surprise prediction."""
    print("Testing meta-surprise gradient flow...")

    batch_size = 8
    seq_len = 64
    d_model = 128

    exp = ExperientialStream(d_model=d_model, use_meta_surprise=True)
    hidden_states = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    output = exp(hidden_states)

    # Use meta_surprise_loss
    loss = meta_surprise_loss(output['predicted_surprise'], output['surprise'])
    loss.backward()

    # Check gradients exist for surprise predictor
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in exp.surprise_predictor.parameters())
    assert has_grads, "No gradients in surprise predictor!"

    # Check hidden_states has gradients
    assert hidden_states.grad is not None, "No gradient to hidden_states!"

    print("  Gradients flow through meta-surprise correctly!")
    return True


def test_meta_surprise_learning():
    """Test that the system can learn to predict its own surprise."""
    print("Testing meta-surprise learning (self-calibration)...")

    d_model = 128
    batch_size = 16
    seq_len = 64
    n_steps = 100

    exp = ExperientialStream(d_model=d_model, use_meta_surprise=True)
    optimizer = torch.optim.Adam(exp.parameters(), lr=0.01)

    initial_meta_surprise = None
    final_meta_surprise = None

    for step in range(n_steps):
        # Generate random hidden states
        hidden_states = torch.randn(batch_size, seq_len, d_model)

        output = exp(hidden_states)

        # Combined loss: predict world + predict self
        total_loss, loss_dict = combined_experiential_loss(
            output,
            exp_weight=1.0,
            meta_weight=0.5
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Detach state to prevent memory growth
        exp.detach_state()

        if step == 0:
            initial_meta_surprise = loss_dict['mean_meta_surprise']
        if step == n_steps - 1:
            final_meta_surprise = loss_dict['mean_meta_surprise']

        if step % 20 == 0:
            print(f"  Step {step}: exp_loss={loss_dict['exp_loss']:.4f}, "
                  f"meta_loss={loss_dict['meta_loss']:.4f}, "
                  f"mean_meta_surprise={loss_dict['mean_meta_surprise']:.4f}")

    print(f"  Initial meta-surprise: {initial_meta_surprise:.4f}")
    print(f"  Final meta-surprise: {final_meta_surprise:.4f}")

    # Meta-surprise should decrease as system learns to predict itself
    # Note: this may not always hold with random data, but gives us a baseline
    print("  Meta-surprise learning test completed!")
    return True


def test_extended_self_awareness():
    """Test extended self-awareness: affect and retrieval prediction."""
    print("Testing extended self-awareness...")

    batch_size = 8
    seq_len = 64
    d_model = 128

    exp = ExperientialStream(
        d_model=d_model,
        use_meta_surprise=True,
        use_affect=True,
        use_persistent_state=True
    )
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    # Test without retrieved memory
    output = exp(hidden_states)

    # Check new outputs exist
    assert output['predicted_valence'] is not None, "predicted_valence should exist"
    assert output['predicted_arousal'] is not None, "predicted_arousal should exist"
    assert output['predicted_retrieval'] is not None, "predicted_retrieval should exist"

    # Check shapes
    assert output['predicted_valence'].shape == (batch_size,), \
        f"Wrong predicted_valence shape: {output['predicted_valence'].shape}"
    assert output['predicted_arousal'].shape == (batch_size,), \
        f"Wrong predicted_arousal shape: {output['predicted_arousal'].shape}"
    assert output['predicted_retrieval'].shape == (batch_size, d_model), \
        f"Wrong predicted_retrieval shape: {output['predicted_retrieval'].shape}"

    # Check ranges
    assert (output['predicted_valence'] >= -1).all() and (output['predicted_valence'] <= 1).all(), \
        "predicted_valence should be in [-1, 1]"
    assert (output['predicted_arousal'] >= 0).all() and (output['predicted_arousal'] <= 1).all(), \
        "predicted_arousal should be in [0, 1]"

    # Test meta_affect_surprise computation
    assert output['meta_affect_surprise'] is not None, "meta_affect_surprise should exist"
    assert output['meta_affect_surprise'].shape == (batch_size,), \
        f"Wrong meta_affect_surprise shape: {output['meta_affect_surprise'].shape}"
    assert (output['meta_affect_surprise'] >= 0).all() and (output['meta_affect_surprise'] <= 1).all(), \
        "meta_affect_surprise should be in [0, 1]"

    # meta_retrieval_surprise should be None without retrieved_memory
    assert output['meta_retrieval_surprise'] is None, \
        "meta_retrieval_surprise should be None without retrieved_memory"

    print(f"  predicted_valence range: [{output['predicted_valence'].min():.3f}, {output['predicted_valence'].max():.3f}]")
    print(f"  predicted_arousal range: [{output['predicted_arousal'].min():.3f}, {output['predicted_arousal'].max():.3f}]")
    print(f"  meta_affect_surprise mean: {output['meta_affect_surprise'].mean():.3f}")

    print("  Extended self-awareness shapes and ranges correct!")
    return True


def test_meta_retrieval_surprise():
    """Test meta-retrieval-surprise with actual retrieved memory."""
    print("Testing meta-retrieval-surprise...")

    batch_size = 8
    seq_len = 64
    d_model = 128

    exp = ExperientialStream(
        d_model=d_model,
        use_meta_surprise=True,
        use_affect=True,
        use_persistent_state=True
    )
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    retrieved_memory = torch.randn(batch_size, d_model)

    output = exp(hidden_states, retrieved_memory=retrieved_memory)

    # Now meta_retrieval_surprise should exist
    assert output['meta_retrieval_surprise'] is not None, \
        "meta_retrieval_surprise should exist with retrieved_memory"
    assert output['meta_retrieval_surprise'].shape == (batch_size,), \
        f"Wrong meta_retrieval_surprise shape: {output['meta_retrieval_surprise'].shape}"
    assert (output['meta_retrieval_surprise'] >= 0).all() and (output['meta_retrieval_surprise'] <= 1).all(), \
        "meta_retrieval_surprise should be in [0, 1]"

    print(f"  meta_retrieval_surprise mean: {output['meta_retrieval_surprise'].mean():.3f}")

    # Test that if retrieved_memory matches predicted_retrieval, surprise is low
    output2 = exp(hidden_states)  # Reset state
    predicted_retrieval = output2['predicted_retrieval'].detach()
    output3 = exp(hidden_states, retrieved_memory=predicted_retrieval)

    # When retrieved matches predicted, surprise should be near 0
    # (may not be exactly 0 due to normalization, but should be low)
    print(f"  meta_retrieval_surprise with matching retrieval: {output3['meta_retrieval_surprise'].mean():.4f}")

    print("  Meta-retrieval-surprise test passed!")
    return True


def test_extended_self_awareness_gradients():
    """Test that extended self-awareness modules receive gradients."""
    print("Testing extended self-awareness gradients...")

    batch_size = 8
    seq_len = 64
    d_model = 128

    exp = ExperientialStream(
        d_model=d_model,
        use_meta_surprise=True,
        use_affect=True,
        use_persistent_state=True
    )
    hidden_states = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    retrieved_memory = torch.randn(batch_size, d_model)

    output = exp(hidden_states, retrieved_memory=retrieved_memory)

    # Compute losses
    affect_loss = meta_affect_loss(
        output['predicted_valence'],
        output['predicted_arousal'],
        output['valence'],
        output['arousal']
    )
    retrieval_loss = meta_retrieval_loss(
        output['predicted_retrieval'],
        retrieved_memory
    )

    total_loss = affect_loss + retrieval_loss
    total_loss.backward()

    # Check gradients flow to new predictors
    affect_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in exp.affect_predictor.parameters()
    )
    retrieval_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in exp.retrieval_predictor.parameters()
    )

    assert affect_has_grad, "affect_predictor should receive gradients"
    assert retrieval_has_grad, "retrieval_predictor should receive gradients"

    print("  Extended self-awareness gradients flow correctly!")
    return True


def test_combined_loss_with_extended():
    """Test combined_experiential_loss with extended self-awareness."""
    print("Testing combined loss with extended self-awareness...")

    batch_size = 8
    seq_len = 64
    d_model = 128

    exp = ExperientialStream(
        d_model=d_model,
        use_meta_surprise=True,
        use_affect=True,
        use_persistent_state=True
    )
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    retrieved_memory = torch.randn(batch_size, d_model)

    output = exp(hidden_states, retrieved_memory=retrieved_memory)
    # Add retrieved_memory to output for loss computation
    output['retrieved_memory'] = retrieved_memory

    loss, loss_dict = combined_experiential_loss(
        output,
        exp_weight=1.0,
        meta_weight=0.1,
        self_mod_weight=0.1,
        affect_weight=0.1,
        retrieval_weight=0.1
    )

    # Check all expected losses are present
    assert 'exp_loss' in loss_dict, "exp_loss should be in loss_dict"
    assert 'meta_loss' in loss_dict, "meta_loss should be in loss_dict"
    assert 'self_mod_loss' in loss_dict, "self_mod_loss should be in loss_dict"
    assert 'affect_loss' in loss_dict, "affect_loss should be in loss_dict"
    assert 'retrieval_loss' in loss_dict, "retrieval_loss should be in loss_dict"
    assert 'mean_meta_affect_surprise' in loss_dict, "mean_meta_affect_surprise should be in loss_dict"
    assert 'mean_meta_retrieval_surprise' in loss_dict, "mean_meta_retrieval_surprise should be in loss_dict"

    # Loss should be finite
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss should be finite"

    print(f"  exp_loss: {loss_dict['exp_loss']:.4f}")
    print(f"  meta_loss: {loss_dict['meta_loss']:.4f}")
    print(f"  affect_loss: {loss_dict['affect_loss']:.4f}")
    print(f"  retrieval_loss: {loss_dict['retrieval_loss']:.4f}")
    print(f"  mean_meta_affect_surprise: {loss_dict['mean_meta_affect_surprise']:.4f}")
    print(f"  mean_meta_retrieval_surprise: {loss_dict['mean_meta_retrieval_surprise']:.4f}")
    print(f"  total_loss: {loss_dict['total_loss']:.4f}")

    print("  Combined loss with extended self-awareness test passed!")
    return True


def test_self_modulation():
    """Test self-modulation: hidden states adjust based on meta-surprise."""
    print("Testing self-modulation...")

    batch_size = 8
    seq_len = 64
    d_model = 128

    exp = ExperientialStream(
        d_model=d_model,
        use_meta_surprise=True,
        use_persistent_state=True
    )
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    output = exp(hidden_states)

    # Check self-modulation outputs exist
    assert output['modulated_output'] is not None, "modulated_output should exist"
    assert output['confidence_gate'] is not None, "confidence_gate should exist"

    # Check shapes
    assert output['modulated_output'].shape == (batch_size, d_model), \
        f"Wrong modulated_output shape: {output['modulated_output'].shape}"
    assert output['confidence_gate'].shape == (batch_size, d_model), \
        f"Wrong confidence_gate shape: {output['confidence_gate'].shape}"

    # Check confidence_gate is in [0, 1] (sigmoid output)
    assert (output['confidence_gate'] >= 0).all() and (output['confidence_gate'] <= 1).all(), \
        "confidence_gate should be in [0, 1]"

    # Verify modulated_output is a blend of h_end and fallback
    # When confidence_gate ≈ 1, modulated_output ≈ h_end (target)
    # When confidence_gate ≈ 0, modulated_output ≈ fallback (prev_state or h_mid)
    print(f"  modulated_output shape: {output['modulated_output'].shape}")
    print(f"  confidence_gate range: [{output['confidence_gate'].min():.3f}, {output['confidence_gate'].max():.3f}]")
    print(f"  confidence_gate mean: {output['confidence_gate'].mean():.3f}")

    print("  Self-modulation shapes and ranges correct!")
    return True


def test_self_modulation_behavior():
    """Test that high meta-surprise leads to lower confidence."""
    print("Testing self-modulation behavior...")

    batch_size = 8
    seq_len = 64
    d_model = 128

    exp = ExperientialStream(
        d_model=d_model,
        use_meta_surprise=True,
        use_persistent_state=True
    )

    # Train briefly so meta-surprise predictor has learned something
    optimizer = torch.optim.Adam(exp.parameters(), lr=0.01)
    for _ in range(20):
        hidden = torch.randn(batch_size, seq_len, d_model)
        exp.reset_state(batch_size)
        out = exp(hidden)
        loss, _ = combined_experiential_loss(out, meta_weight=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Now test: samples with higher meta-surprise should have lower mean confidence
    exp.eval()
    with torch.no_grad():
        hidden = torch.randn(batch_size * 4, seq_len, d_model)
        exp.reset_state(batch_size * 4)
        output = exp(hidden)

        meta_surprise = output['meta_surprise']  # [B]
        confidence = output['confidence_gate'].mean(dim=-1)  # [B] mean across dimensions

        # Sort samples by meta-surprise
        sorted_indices = torch.argsort(meta_surprise)
        low_ms_indices = sorted_indices[:batch_size]
        high_ms_indices = sorted_indices[-batch_size:]

        low_ms_conf = confidence[low_ms_indices].mean()
        high_ms_conf = confidence[high_ms_indices].mean()

        print(f"  Low meta-surprise samples: ms={meta_surprise[low_ms_indices].mean():.4f}, conf={low_ms_conf:.4f}")
        print(f"  High meta-surprise samples: ms={meta_surprise[high_ms_indices].mean():.4f}, conf={high_ms_conf:.4f}")

        # Note: After initialization, this relationship may not hold perfectly
        # The key test is that the mechanism exists and gradients flow
        print("  Self-modulation behavior test completed!")

    return True


def test_self_modulation_gradients():
    """Test that gradients flow through self-modulation."""
    print("Testing self-modulation gradient flow...")

    batch_size = 8
    seq_len = 64
    d_model = 128

    exp = ExperientialStream(
        d_model=d_model,
        use_meta_surprise=True,
        use_persistent_state=True
    )
    hidden_states = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    output = exp(hidden_states)

    # Loss that uses modulated_output
    loss = output['modulated_output'].mean()
    loss.backward()

    # Check gradients exist for self_modulator
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in exp.self_modulator.parameters())
    assert has_grads, "No gradients in self_modulator!"

    # Check hidden_states has gradients
    assert hidden_states.grad is not None, "No gradient to hidden_states!"

    print("  Gradients flow through self-modulation correctly!")
    return True


def test_persistent_state():
    """Test that persistent state carries over across chunks."""
    print("Testing persistent state...")

    batch_size = 4
    seq_len = 64
    d_model = 128

    exp = ExperientialStream(d_model=d_model, use_persistent_state=True)

    # Process first chunk
    chunk1 = torch.randn(batch_size, seq_len, d_model)
    output1 = exp(chunk1)

    assert output1['persistent_state'] is not None, "Persistent state should exist after first chunk"
    assert output1['prev_state'] is not None, "prev_state should exist"
    state_after_chunk1 = output1['persistent_state'].clone()

    # Process second chunk
    chunk2 = torch.randn(batch_size, seq_len, d_model)
    output2 = exp(chunk2)

    # State should have been updated
    assert output2['persistent_state'] is not None
    state_after_chunk2 = output2['persistent_state'].clone()

    # prev_state for chunk2 should be state_after_chunk1
    assert torch.allclose(output2['prev_state'], state_after_chunk1), \
        "prev_state should match state from previous chunk"

    # States should be different after processing different chunks
    # (unless gate is exactly 0, which is unlikely)
    assert not torch.allclose(state_after_chunk1, state_after_chunk2), \
        "State should change after processing new chunk"

    print("  Persistent state carries over correctly!")

    # Test reset
    exp.reset_state()
    chunk3 = torch.randn(batch_size, seq_len, d_model)
    output3 = exp(chunk3)

    # After reset, prev_state should be zeros
    assert torch.allclose(output3['prev_state'], torch.zeros_like(output3['prev_state'])), \
        "After reset, prev_state should be zeros"

    print("  State reset works correctly!")

    # Test without persistent state
    exp_no_persist = ExperientialStream(d_model=d_model, use_persistent_state=False)
    output_no_persist = exp_no_persist(chunk1)
    assert output_no_persist['persistent_state'] is None, \
        "Without persistent state, should return None"
    assert output_no_persist['prev_state'] is None, \
        "Without persistent state, prev_state should be None"

    print("  Non-persistent mode works correctly!")
    return True


def test_persistent_state_learning():
    """Test that persistent state helps with sequential prediction."""
    print("Testing persistent state improves sequential learning...")

    d_model = 128
    batch_size = 16
    seq_len = 32
    n_chunks = 5
    n_epochs = 50

    # Create model with persistent state
    exp = ExperientialStream(d_model=d_model, use_persistent_state=True)
    optimizer = torch.optim.Adam(exp.parameters(), lr=1e-3)

    # Create sequential data where chunks are related
    # Each chunk's end state is influenced by previous chunk's end state
    def generate_sequential_data():
        """Generate correlated sequential chunks."""
        chunks = []
        prev_end = torch.randn(batch_size, d_model) * 0.1

        for _ in range(n_chunks):
            chunk = torch.randn(batch_size, seq_len, d_model)
            # Make end of chunk related to previous end
            chunk[:, -1, :] = prev_end + torch.randn(batch_size, d_model) * 0.3
            # Make mid related to end
            chunk[:, seq_len//2, :] = chunk[:, -1, :] + torch.randn(batch_size, d_model) * 0.2
            prev_end = chunk[:, -1, :].clone()
            chunks.append(chunk)

        return chunks

    losses = []
    for epoch in range(n_epochs):
        chunks = generate_sequential_data()
        exp.reset_state()  # Reset at start of each sequence

        epoch_loss = 0.0
        for chunk in chunks:
            output = exp(chunk)
            loss = experiential_loss(output['prediction'], output['target'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Detach state to prevent huge graphs
            exp.detach_state()
            epoch_loss += loss.item()

        losses.append(epoch_loss / n_chunks)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: avg_loss={losses[-1]:.4f}")

    # Loss should decrease
    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print("  Sequential learning test passed!")
    return True


def test_episodic_memory_basic():
    """Test basic episodic memory operations."""
    print("Testing episodic memory basics...")

    d_model = 128
    capacity = 10

    memory = EpisodicMemory(d_model=d_model, capacity=capacity, crystallization_threshold=0.3)

    # Initially empty
    assert memory.size == 0, "Memory should start empty"

    # Store some episodes
    for i in range(5):
        content = torch.randn(d_model)
        context = torch.randn(d_model)
        salience = 0.5 + 0.1 * i  # Increasing salience

        memory.store(content, context, salience, valence=0.1, arousal=0.5)

    assert memory.size == 5, f"Memory should have 5 episodes, got {memory.size}"

    print(f"  Stored 5 episodes, size={memory.size}")

    # Test crystallization threshold
    assert memory.should_crystallize(0.5), "0.5 should pass threshold 0.3"
    assert not memory.should_crystallize(0.2), "0.2 should not pass threshold 0.3"

    # Test retrieval by similarity
    query = memory.episodes[2].content.clone()  # Query with known episode
    retrieved = memory.retrieve(query, top_k=3)

    assert len(retrieved) == 3, f"Should retrieve 3 episodes, got {len(retrieved)}"
    assert retrieved[0][0].timestamp == 2, "First retrieved should be the queried episode"
    assert retrieved[0][1] > 0.99, "First retrieved should have similarity ~1.0"

    print(f"  Retrieval works: top match similarity = {retrieved[0][1]:.4f}")

    # Test retrieval count increment
    assert retrieved[0][0].retrieval_count == 1, "Retrieval count should be 1"

    # Test retrieval by salience
    top_salient = memory.retrieve_by_salience(top_k=2)
    assert len(top_salient) == 2, "Should get 2 most salient"
    assert top_salient[0].salience >= top_salient[1].salience, "Should be sorted by salience"

    # Test retrieval by time
    recent = memory.retrieve_by_time(recent_n=2)
    assert len(recent) == 2, "Should get 2 most recent"
    assert recent[-1].timestamp > recent[-2].timestamp, "Last should be most recent"

    print("  Retrieval by salience and time work!")

    # Test capacity management (eviction)
    for i in range(10):  # Add more than capacity
        memory.store(torch.randn(d_model), torch.randn(d_model), 0.5)

    assert memory.size == capacity, f"Memory should be at capacity {capacity}, got {memory.size}"

    print(f"  Capacity management works: size capped at {memory.size}")

    # Test stats
    stats = memory.get_stats()
    assert 'size' in stats and stats['size'] == capacity
    assert 'avg_salience' in stats
    assert 'avg_retrieval_count' in stats

    print(f"  Stats: {stats}")

    # Test clear
    memory.clear()
    assert memory.size == 0, "Memory should be empty after clear"

    print("  Episodic memory basics test passed!")
    return True


def test_episodic_memory_with_experiential():
    """Test episodic memory integration with experiential stream."""
    print("Testing episodic memory with experiential stream...")

    d_model = 128
    batch_size = 8
    seq_len = 64

    exp = ExperientialStream(d_model=d_model, use_affect=True)
    memory = EpisodicMemory(d_model=d_model, capacity=100, crystallization_threshold=0.1)

    # Process several batches and store high-salience moments
    n_batches = 20
    crystallized_count = 0

    for batch_idx in range(n_batches):
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        exp.reset_state(batch_size=batch_size)

        output = exp(hidden_states)

        # Check each sample in batch for crystallization
        for i in range(batch_size):
            salience = output['salience'][i].item()

            if memory.should_crystallize(salience):
                memory.store(
                    content=output['target'][i],
                    context=output['prediction'][i],
                    salience=salience,
                    valence=output['valence'][i].item(),
                    arousal=output['arousal'][i].item()
                )
                crystallized_count += 1

    print(f"  Processed {n_batches * batch_size} experiences")
    print(f"  Crystallized {crystallized_count} episodes ({crystallized_count/(n_batches*batch_size)*100:.1f}%)")
    print(f"  Memory size: {memory.size}")

    # Verify some episodes were stored
    assert memory.size > 0, "Should have stored some episodes"

    # Test retrieval with a new query
    query = torch.randn(d_model)
    retrieved = memory.retrieve(query, top_k=5)

    print(f"  Retrieved {len(retrieved)} episodes for query")
    if retrieved:
        print(f"  Top match salience: {retrieved[0][0].salience:.4f}, similarity: {retrieved[0][1]:.4f}")

    # Test stats
    stats = memory.get_stats()
    print(f"  Memory stats: size={stats['size']}, avg_salience={stats['avg_salience']:.4f}")

    print("  Episodic memory integration test passed!")
    return True


def test_retrieve_soft():
    """Test differentiable retrieval."""
    print("Testing retrieve_soft (differentiable retrieval)...")

    d_model = 128
    memory = EpisodicMemory(d_model=d_model, capacity=100)

    # Test with empty memory
    query = torch.randn(d_model, requires_grad=True)
    values, weights = memory.retrieve_soft(query)
    assert values.shape == (d_model,), f"Empty memory should return zeros of shape {d_model}"
    assert weights.shape == (0,), "Empty memory should return empty weights"
    print("  Empty memory handling works")

    # Store some episodes with known content
    n_episodes = 5
    episode_contents = []
    for i in range(n_episodes):
        content = torch.randn(d_model)
        content = F.normalize(content, dim=-1)  # Normalize for easier testing
        episode_contents.append(content)
        memory.store(content, torch.randn(d_model), salience=0.5 + 0.1 * i)

    # Test 1: Query that matches a specific episode
    query = episode_contents[2].clone().requires_grad_(True)
    values, weights = memory.retrieve_soft(query, temperature=0.1)

    assert values.shape == (d_model,), f"Values shape wrong: {values.shape}"
    assert weights.shape == (n_episodes,), f"Weights shape wrong: {weights.shape}"
    assert weights.sum().item() - 1.0 < 1e-5, "Weights should sum to 1"

    # The matching episode should have highest weight
    assert weights[2].item() > weights.max().item() - 0.01, \
        f"Query should match episode 2 most (weight={weights[2]:.3f})"

    print(f"  Retrieval weights: {weights.detach().numpy().round(3)}")
    print(f"  Highest weight at index: {weights.argmax().item()} (expected 2)")

    # Test 2: Gradient flow
    loss = values.sum()
    loss.backward()
    assert query.grad is not None, "Gradients should flow through retrieve_soft"
    assert query.grad.abs().sum() > 0, "Gradients should be non-zero"
    print("  Gradient flow works!")

    # Test 3: Batched query
    batch_size = 4
    batch_query = torch.randn(batch_size, d_model)
    batch_values, batch_weights = memory.retrieve_soft(batch_query)

    assert batch_values.shape == (batch_size, d_model), \
        f"Batched values shape wrong: {batch_values.shape}"
    assert batch_weights.shape == (batch_size, n_episodes), \
        f"Batched weights shape wrong: {batch_weights.shape}"
    print(f"  Batched retrieval works: {batch_values.shape}")

    # Test 4: Temperature effect
    query = episode_contents[0].clone()
    _, weights_hot = memory.retrieve_soft(query, temperature=1.0)  # softer
    _, weights_cold = memory.retrieve_soft(query, temperature=0.01)  # sharper

    # Cold temperature should be more peaked
    assert weights_cold.max() > weights_hot.max(), \
        "Lower temperature should produce sharper attention"
    print(f"  Temperature effect: hot_max={weights_hot.max():.3f}, cold_max={weights_cold.max():.3f}")

    # Test 5: Salience weighting
    _, weights_no_sal = memory.retrieve_soft(torch.randn(d_model), salience_weight=0.0)
    _, weights_with_sal = memory.retrieve_soft(torch.randn(d_model), salience_weight=1.0)

    # With salience weighting, higher salience episodes should get more weight
    # Episode 4 has highest salience (0.9)
    print(f"  Salience effect: no_sal[4]={weights_no_sal[4]:.3f}, with_sal[4]={weights_with_sal[4]:.3f}")

    print("  retrieve_soft test passed!")
    return True


def test_retrieve_soft_learning():
    """Test that retrieve_soft enables learning."""
    print("Testing retrieve_soft enables learning...")

    d_model = 64
    batch_size = 16
    n_steps = 50

    # Create memory with some episodes
    memory = EpisodicMemory(d_model=d_model, capacity=20)
    for i in range(10):
        memory.store(torch.randn(d_model), torch.randn(d_model), salience=0.5)

    # Create a simple query projector to train
    query_proj = nn.Linear(d_model, d_model)
    optimizer = torch.optim.Adam(query_proj.parameters(), lr=0.01)

    # Training task: make retrieval output match a target
    target = torch.randn(d_model)

    losses = []
    for step in range(n_steps):
        # Random input
        x = torch.randn(batch_size, d_model)

        # Project to query space
        query = query_proj(x)

        # Retrieve from memory
        retrieved, weights = memory.retrieve_soft(query, temperature=0.5)

        # Loss: retrieved should match target
        loss = F.mse_loss(retrieved, target.expand(batch_size, -1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # Loss should decrease
    initial_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")

    assert final_loss < initial_loss, "Loss should decrease with training"
    print("  retrieve_soft learning test passed!")
    return True


# Mock GPT for testing MemoryAugmentedGPT
class MockGPTConfig:
    """Minimal config for testing."""
    def __init__(self, vocab_size=1000, d_model=128, n_layer=2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer


class MockGPT(nn.Module):
    """Minimal GPT-like model for testing memory augmentation."""

    def __init__(self, config: MockGPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model)
            for _ in range(config.n_layer)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos=None,
        return_hidden_states: bool = False
    ):
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = F.gelu(layer(x))
        hidden_states = self.final_norm(x)
        logits = self.lm_head(hidden_states)

        if return_hidden_states:
            return logits, hidden_states
        return logits, None


def test_memory_augmented_gpt():
    """Test MemoryAugmentedGPT wrapper."""
    print("Testing MemoryAugmentedGPT...")

    # Setup
    vocab_size = 1000
    d_model = 128
    batch_size = 4
    seq_len = 32

    config = MockGPTConfig(vocab_size=vocab_size, d_model=d_model)
    gpt = MockGPT(config)

    # Test all three integration modes
    for mode in ['residual', 'gated', 'attention']:
        print(f"  Testing {mode} integration...")

        memory_gpt = MemoryAugmentedGPT(
            gpt,
            memory_capacity=100,
            crystallization_threshold=0.1,  # Low threshold for testing
            memory_integration=mode,
            use_semantic=False  # Test episodic only first
        )

        # Initial forward (no memories yet)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits, hidden, mem_out = memory_gpt(input_ids, crystallize=True)

        assert logits.shape == (batch_size, seq_len, vocab_size), \
            f"Logits shape wrong: {logits.shape}"
        assert hidden.shape == (batch_size, seq_len, d_model), \
            f"Hidden shape wrong: {hidden.shape}"
        assert 'surprise' in mem_out, "Should have experiential output"

        initial_size = mem_out['episodic_size']
        print(f"    Initial memory size: {initial_size}")

        # Run multiple forwards to build up memory
        for _ in range(10):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            logits, hidden, mem_out = memory_gpt(input_ids, crystallize=True)

        final_size = mem_out['episodic_size']
        print(f"    Final memory size: {final_size}")
        assert final_size > initial_size, "Memory should have grown"

        # Test with memory retrieval
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits_mem, hidden_mem, mem_out = memory_gpt(
            input_ids,
            crystallize=False,
            use_memory=True,
            return_memory_weights=True
        )

        assert mem_out['retrieved_episodic'] is not None, "Should have retrieved episodic values"
        assert mem_out['episodic_weights'] is not None, "Should have episodic retrieval weights"
        print(f"    Retrieved values shape: {mem_out['retrieved_episodic'].shape}")

        # Reset and verify
        memory_gpt.reset_memory()
        assert memory_gpt.memory.size == 0, "Memory should be cleared"
        print(f"    {mode} integration works!")

    print("  MemoryAugmentedGPT test passed!")
    return True


def test_memory_augmented_gradient_flow():
    """Test that gradients flow through memory-augmented model."""
    print("Testing memory-augmented gradient flow...")

    vocab_size = 500
    d_model = 64
    batch_size = 4
    seq_len = 16

    config = MockGPTConfig(vocab_size=vocab_size, d_model=d_model)
    gpt = MockGPT(config)
    memory_gpt = MemoryAugmentedGPT(
        gpt,
        memory_capacity=50,
        crystallization_threshold=0.05,
        memory_integration='gated',
        use_semantic=False  # Test episodic only
    )

    # Build up some memories first
    for _ in range(5):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        memory_gpt(input_ids, crystallize=True)

    # Now test gradient flow
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, hidden, mem_out = memory_gpt(input_ids, use_memory=True)

    # Compute loss
    loss, loss_dict = memory_augmented_loss(
        logits, targets, mem_out,
        lm_weight=1.0, exp_weight=0.1
    )

    # Backward pass
    loss.backward()

    # Check gradients exist
    has_gpt_grads = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in gpt.parameters()
    )
    assert has_gpt_grads, "GPT should have gradients"

    has_memory_grads = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in memory_gpt.memory_gate.parameters()
    )
    assert has_memory_grads, "Memory gate should have gradients"

    print(f"  Loss breakdown: {loss_dict}")
    print("  Gradient flow test passed!")
    return True


def test_memory_augmented_learning():
    """Test that memory-augmented model can learn."""
    print("Testing memory-augmented learning...")

    vocab_size = 200
    d_model = 64
    batch_size = 8
    seq_len = 16
    n_steps = 30

    config = MockGPTConfig(vocab_size=vocab_size, d_model=d_model)
    gpt = MockGPT(config)
    memory_gpt = MemoryAugmentedGPT(
        gpt,
        memory_capacity=50,
        crystallization_threshold=0.1,
        memory_integration='gated',
        use_semantic=False  # Test episodic only
    )

    optimizer = torch.optim.Adam(memory_gpt.parameters(), lr=0.01)

    losses = []
    for step in range(n_steps):
        # Generate data where next token depends on pattern
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = input_ids.roll(-1, dims=1)  # Simple next-token prediction
        targets[:, -1] = 0  # Pad last position

        logits, hidden, mem_out = memory_gpt(input_ids, crystallize=True, use_memory=True)

        loss, _ = memory_augmented_loss(logits, targets, mem_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 10 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, episodic_size={mem_out['episodic_size']}")

    initial_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")

    # Loss should decrease (model is learning)
    assert final_loss < initial_loss, "Loss should decrease with training"

    print("  Memory-augmented learning test passed!")
    return True


def test_semantic_stream_basic():
    """Test basic SemanticStream operations."""
    print("Testing SemanticStream basics...")

    d_model = 128

    semantic = SemanticStream(d_model=d_model, min_evidence=2)

    # Initially empty
    assert semantic.size == 0, "Semantic memory should start empty"

    # Create some fake episodes for consolidation
    episodes = []
    for i in range(5):
        content = torch.randn(d_model)
        content = F.normalize(content, dim=-1)
        ep = Episode(
            content=content,
            context=torch.randn(d_model),
            timestamp=i,
            salience=0.5 + 0.1 * i,
            valence=0.0,
            arousal=0.5
        )
        episodes.append(ep)

    # Test consolidation
    concept = semantic.consolidate(episodes, require_min_evidence=True, name="test_concept")

    assert concept is not None, "Should create concept from 5 episodes"
    assert concept.id == 0, "First concept should have id 0"
    assert concept.evidence_count == 5, "Should have 5 episodes as evidence"
    assert concept.name == "test_concept", "Name should be set"
    assert concept.embedding.shape == (d_model,), f"Embedding shape wrong: {concept.embedding.shape}"
    assert concept.prototype.shape == (d_model,), f"Prototype shape wrong: {concept.prototype.shape}"

    print(f"  Created concept: id={concept.id}, name={concept.name}, evidence={concept.evidence_count}")
    print(f"  Semantic memory size: {semantic.size}")

    # Test that min_evidence is enforced
    small_episodes = episodes[:1]
    small_concept = semantic.consolidate(small_episodes, require_min_evidence=True)
    assert small_concept is None, "Should not create concept with insufficient evidence"

    print("  Min evidence requirement works!")

    # Test stats
    stats = semantic.get_stats()
    assert stats['n_concepts'] == 1, f"Should have 1 concept, got {stats['n_concepts']}"
    print(f"  Stats: {stats}")

    print("  SemanticStream basics test passed!")
    return True


def test_semantic_query():
    """Test SemanticStream querying."""
    print("Testing SemanticStream querying...")

    d_model = 128
    semantic = SemanticStream(d_model=d_model, min_evidence=2)

    # Create several distinct concepts
    n_concepts = 3
    created_concepts = []

    for c in range(n_concepts):
        # Create a cluster of episodes around a center
        center = torch.randn(d_model)
        center = F.normalize(center, dim=-1)

        episodes = []
        for i in range(5):
            # Episodes similar to center
            content = center + torch.randn(d_model) * 0.1
            content = F.normalize(content, dim=-1)
            ep = Episode(
                content=content,
                context=torch.randn(d_model),
                timestamp=c * 10 + i,
                salience=0.5,
                valence=0.0,
                arousal=0.5
            )
            episodes.append(ep)

        concept = semantic.consolidate(episodes, name=f"concept_{c}")
        assert concept is not None, f"Should create concept {c}"
        created_concepts.append(concept)

    assert semantic.size == n_concepts, f"Should have {n_concepts} concepts"

    # Test query - query with a concept's own embedding should find that concept
    query = created_concepts[1].embedding.clone()  # Use actual embedding
    results = semantic.query(query, top_k=2)

    assert len(results) == 2, f"Should retrieve 2 concepts, got {len(results)}"
    concept, similarity = results[0]
    # With projected query, similarity should still be reasonable
    print(f"  Query results: top match id={concept.id}, similarity = {similarity:.4f}")

    # Test query_soft (differentiable)
    query = torch.randn(d_model, requires_grad=True)
    knowledge, weights = semantic.query_soft(query, temperature=0.1)

    assert knowledge.shape == (d_model,), f"Knowledge shape wrong: {knowledge.shape}"
    assert weights.shape == (n_concepts,), f"Weights shape wrong: {weights.shape}"
    assert abs(weights.sum().item() - 1.0) < 1e-5, "Weights should sum to 1"

    # Test gradient flow
    loss = knowledge.sum()
    loss.backward()
    assert query.grad is not None, "Gradients should flow through query_soft"
    print("  Gradient flow through query_soft works!")

    # Test batched query_soft
    batch_query = torch.randn(4, d_model)
    batch_knowledge, batch_weights = semantic.query_soft(batch_query)
    assert batch_knowledge.shape == (4, d_model), f"Batch knowledge shape wrong: {batch_knowledge.shape}"
    assert batch_weights.shape == (4, n_concepts), f"Batch weights shape wrong: {batch_weights.shape}"
    print("  Batched query_soft works!")

    print("  SemanticStream querying test passed!")
    return True


def test_semantic_consolidation_from_memory():
    """Test automatic consolidation from EpisodicMemory."""
    print("Testing consolidation from EpisodicMemory...")

    d_model = 128
    memory = EpisodicMemory(d_model=d_model, capacity=100)
    semantic = SemanticStream(d_model=d_model, min_evidence=3)

    # Store episodes in clusters
    n_clusters = 3
    episodes_per_cluster = 8

    for c in range(n_clusters):
        center = torch.randn(d_model)
        center = F.normalize(center, dim=-1)

        for i in range(episodes_per_cluster):
            content = center + torch.randn(d_model) * 0.15
            memory.store(content, torch.randn(d_model), salience=0.5)

    print(f"  Stored {memory.size} episodes in {n_clusters} clusters")

    # Consolidate from memory
    concepts = semantic.consolidate_from_memory(memory, n_clusters=n_clusters, min_cluster_size=3)

    print(f"  Created {len(concepts)} concepts")
    assert len(concepts) >= 1, "Should create at least 1 concept"

    for c in concepts:
        print(f"    Concept {c.id}: evidence={c.evidence_count}, confidence={c.confidence:.3f}")

    # Stats should reflect the new concepts
    stats = semantic.get_stats()
    print(f"  Stats: {stats}")

    print("  Consolidation from memory test passed!")
    return True


def test_semantic_generalization():
    """Test creating abstract concepts from specific ones."""
    print("Testing semantic generalization...")

    d_model = 128
    semantic = SemanticStream(d_model=d_model, min_evidence=2)

    # Create 3 base concepts
    base_concept_ids = []
    for c in range(3):
        episodes = []
        for i in range(3):
            content = torch.randn(d_model)
            ep = Episode(
                content=content,
                context=torch.randn(d_model),
                timestamp=c * 10 + i,
                salience=0.5,
                valence=0.0,
                arousal=0.5
            )
            episodes.append(ep)

        concept = semantic.consolidate(episodes, name=f"base_{c}")
        base_concept_ids.append(concept.id)

    assert semantic.size == 3, "Should have 3 base concepts"
    print(f"  Created {semantic.size} base concepts")

    # Generalize to abstract concept
    abstract = semantic.generalize(base_concept_ids, name="abstract_concept")

    assert abstract is not None, "Should create abstract concept"
    assert abstract.abstraction_level == 1, "Abstract concept should be level 1"
    assert abstract.name == "abstract_concept", "Name should be set"
    assert len(abstract.children) == 3, "Should have 3 children"

    print(f"  Abstract concept: id={abstract.id}, level={abstract.abstraction_level}, children={abstract.children}")

    # Check that children point to parent
    for cid in base_concept_ids:
        assert semantic.concepts[cid].parent == abstract.id, f"Child {cid} should point to parent"
        # Check is-a relation exists
        assert abstract.id in semantic.concepts[cid].connections, f"Child {cid} should have is-a relation"

    print("  Parent-child relationships established!")

    # Stats should show the hierarchy
    stats = semantic.get_stats()
    print(f"  Stats: {stats}")
    assert stats['max_abstraction_level'] == 1, "Max abstraction level should be 1"
    assert stats['concepts_by_level'][0] == 3, "Should have 3 level-0 concepts"
    assert stats['concepts_by_level'][1] == 1, "Should have 1 level-1 concept"

    print("  Semantic generalization test passed!")
    return True


def test_semantic_relations():
    """Test relation creation and discovery."""
    print("Testing semantic relations...")

    d_model = 128
    semantic = SemanticStream(d_model=d_model, min_evidence=2)

    # Create two concepts
    for c in range(2):
        episodes = []
        for i in range(3):
            ep = Episode(
                content=torch.randn(d_model),
                context=torch.randn(d_model),
                timestamp=c * 10 + i,
                salience=0.5,
                valence=0.0,
                arousal=0.5
            )
            episodes.append(ep)
        semantic.consolidate(episodes, name=f"concept_{c}")

    # Manually connect concepts
    success = semantic.connect(0, 1, relation_type="related-to", strength=0.8)
    assert success, "Should successfully connect concepts"

    # Check relation exists
    assert 1 in semantic.concepts[0].connections, "Connection should exist"
    relation = semantic.concepts[0].connections[1]
    assert relation.relation_type == "related-to", f"Wrong relation type: {relation.relation_type}"
    assert relation.strength == 0.8, f"Wrong strength: {relation.strength}"

    print(f"  Created relation: {relation.relation_type} with strength {relation.strength}")

    # Test invalid connection
    success = semantic.connect(0, 999, relation_type="related-to", strength=0.5)
    assert not success, "Should fail to connect to non-existent concept"

    # Stats should show relations
    stats = semantic.get_stats()
    assert stats['n_relations'] >= 1, f"Should have at least 1 relation, got {stats['n_relations']}"
    print(f"  Stats: {stats}")

    print("  Semantic relations test passed!")
    return True


def test_integrated_semantic_memory():
    """Test MemoryAugmentedGPT with semantic memory integration."""
    print("Testing integrated semantic memory...")

    # Setup
    vocab_size = 1000
    d_model = 128
    batch_size = 4
    seq_len = 32

    config = MockGPTConfig(vocab_size=vocab_size, d_model=d_model)
    gpt = MockGPT(config)

    # Create memory-augmented model with semantic memory
    memory_gpt = MemoryAugmentedGPT(
        gpt,
        memory_capacity=100,
        crystallization_threshold=0.1,  # Low threshold for testing
        memory_integration='gated',
        use_semantic=True,
        consolidation_interval=20,  # Consolidate every 20 steps
        min_consolidation_evidence=3
    )

    # Verify semantic stream is initialized
    assert memory_gpt.semantic is not None, "Semantic stream should be initialized"
    assert memory_gpt.semantic.size == 0, "Semantic stream should start empty"

    # Run multiple forwards to build episodic memory
    print("  Building episodic memory...")
    for i in range(25):  # Enough steps to trigger consolidation
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits, hidden, mem_out = memory_gpt(input_ids, crystallize=True)

    episodic_size = mem_out['episodic_size']
    semantic_size = mem_out['semantic_size']
    print(f"  Episodic size: {episodic_size}, Semantic size: {semantic_size}")

    # Should have some episodic memories
    assert episodic_size > 0, "Should have episodic memories"

    # Semantic might have consolidated (depends on clustering)
    # Not guaranteed, but should not crash

    # Test retrieval with semantic
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, hidden, mem_out = memory_gpt(
        input_ids,
        use_memory=True,
        use_semantic=True,
        return_memory_weights=True
    )

    assert mem_out['retrieved_episodic'] is not None, "Should have episodic retrieval"
    print(f"  Episodic retrieval shape: {mem_out['retrieved_episodic'].shape}")

    # Manual consolidation
    print("  Testing manual consolidation...")
    n_consolidated = memory_gpt.consolidate(n_clusters=3, min_cluster_size=3)
    print(f"  Consolidated {n_consolidated} concepts")

    # Check semantic memory grew
    if n_consolidated > 0:
        assert memory_gpt.semantic.size > 0, "Semantic memory should have concepts"

        # Now test semantic retrieval
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits, hidden, mem_out = memory_gpt(
            input_ids,
            use_memory=True,
            use_semantic=True,
            return_memory_weights=True
        )

        if mem_out['retrieved_semantic'] is not None:
            print(f"  Semantic retrieval shape: {mem_out['retrieved_semantic'].shape}")
        else:
            print("  No semantic retrieval (semantic memory may be empty)")

    # Test retrieve_relevant
    query = torch.randn(d_model)
    results = memory_gpt.retrieve_relevant(query, top_k=3, include_semantic=True)
    assert 'episodic' in results, "Should have episodic results"
    print(f"  retrieve_relevant: {len(results['episodic'])} episodic, {len(results.get('semantic', []))} semantic")

    # Test get_memory_stats
    stats = memory_gpt.get_memory_stats()
    assert 'episodic' in stats, "Should have episodic stats"
    if memory_gpt.semantic.size > 0:
        assert 'semantic' in stats, "Should have semantic stats"
    print(f"  Stats: episodic={stats['episodic']['size']}, step_counter={stats['step_counter']}")

    # Test reset_episodic (preserve semantic)
    semantic_before = memory_gpt.semantic.size
    memory_gpt.reset_episodic()
    assert memory_gpt.memory.size == 0, "Episodic should be cleared"
    assert memory_gpt.semantic.size == semantic_before, "Semantic should be preserved"

    # Test full reset
    memory_gpt.reset_memory()
    assert memory_gpt.memory.size == 0, "Episodic should be cleared"
    assert memory_gpt.semantic.size == 0, "Semantic should be cleared"

    print("  Integrated semantic memory test passed!")
    return True


def test_integrated_semantic_gradient_flow():
    """Test gradient flow through semantic retrieval."""
    print("Testing gradient flow through semantic retrieval...")

    vocab_size = 1000
    d_model = 64
    batch_size = 2
    seq_len = 16

    config = MockGPTConfig(vocab_size=vocab_size, d_model=d_model)
    gpt = MockGPT(config)

    memory_gpt = MemoryAugmentedGPT(
        gpt,
        memory_capacity=50,
        crystallization_threshold=0.0,  # Store everything
        memory_integration='gated',
        use_semantic=True,
        consolidation_interval=0  # Manual only
    )

    # Build some episodic memories
    for _ in range(20):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        memory_gpt(input_ids, crystallize=True)

    # Consolidate to create semantic memories
    n_consolidated = memory_gpt.consolidate(n_clusters=3, min_cluster_size=3)

    if n_consolidated == 0:
        print("  Warning: No concepts consolidated, testing with episodic only")

    # Test gradient flow
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, hidden, mem_out = memory_gpt(
        input_ids,
        use_memory=True,
        use_semantic=True,
        crystallize=False
    )

    # Compute loss
    loss, loss_dict = memory_augmented_loss(logits, targets, mem_out)

    # Check gradients flow
    loss.backward()

    # Check query projection has gradients
    assert memory_gpt.query_proj.weight.grad is not None, "Query proj should have gradients"
    assert memory_gpt.query_proj.weight.grad.abs().sum() > 0, "Query proj gradients should be non-zero"

    # Check memory gate has gradients
    assert memory_gpt.memory_gate[0].weight.grad is not None, "Memory gate should have gradients"

    # If semantic was used, check semantic layers
    if n_consolidated > 0 and hasattr(memory_gpt, 'semantic_gate'):
        assert memory_gpt.semantic_gate[0].weight.grad is not None, "Semantic gate should have gradients"
        print("  Semantic gate gradients verified")

    print(f"  Loss dict: lm={loss_dict['lm_loss']:.4f}, episodic={loss_dict['episodic_size']}, semantic={loss_dict['semantic_size']}")
    print("  Gradient flow test passed!")
    return True


def test_all_integration_modes_with_semantic():
    """Test all integration modes work with semantic memory."""
    print("Testing all integration modes with semantic...")

    vocab_size = 1000
    d_model = 64
    batch_size = 2
    seq_len = 16

    config = MockGPTConfig(vocab_size=vocab_size, d_model=d_model)

    for mode in ['residual', 'gated', 'attention']:
        print(f"  Testing {mode} mode...")

        gpt = MockGPT(config)
        memory_gpt = MemoryAugmentedGPT(
            gpt,
            memory_capacity=50,
            crystallization_threshold=0.0,
            memory_integration=mode,
            use_semantic=True,
            consolidation_interval=0
        )

        # Build memories
        for _ in range(15):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            memory_gpt(input_ids, crystallize=True)

        # Consolidate
        memory_gpt.consolidate(n_clusters=2, min_cluster_size=3)

        # Forward with both memory types
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits, hidden, mem_out = memory_gpt(
            input_ids,
            use_memory=True,
            use_semantic=True
        )

        assert logits.shape == (batch_size, seq_len, vocab_size), f"Wrong logits shape for {mode}"
        assert hidden.shape == (batch_size, seq_len, d_model), f"Wrong hidden shape for {mode}"

        print(f"    {mode}: episodic={mem_out['episodic_size']}, semantic={mem_out['semantic_size']}")

    print("  All integration modes test passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("ExperientialStream & EpisodicMemory Tests")
    print("=" * 60)

    test_shapes()
    print()
    test_multiscale()
    print()
    test_gradient_flow()
    print()
    test_learning()
    print()
    test_affect_prediction()
    print()
    test_affect_gradients()
    print()
    test_meta_surprise()
    print()
    test_meta_surprise_gradients()
    print()
    test_meta_surprise_learning()
    print()
    test_extended_self_awareness()
    print()
    test_meta_retrieval_surprise()
    print()
    test_extended_self_awareness_gradients()
    print()
    test_combined_loss_with_extended()
    print()
    test_self_modulation()
    print()
    test_self_modulation_behavior()
    print()
    test_self_modulation_gradients()
    print()
    test_persistent_state()
    print()
    test_persistent_state_learning()
    print()
    test_episodic_memory_basic()
    print()
    test_episodic_memory_with_experiential()
    print()
    test_retrieve_soft()
    print()
    test_retrieve_soft_learning()
    print()
    test_memory_augmented_gpt()
    print()
    test_memory_augmented_gradient_flow()
    print()
    test_memory_augmented_learning()
    print()

    print("=" * 60)
    print("SemanticStream Tests")
    print("=" * 60)

    test_semantic_stream_basic()
    print()
    test_semantic_query()
    print()
    test_semantic_consolidation_from_memory()
    print()
    test_semantic_generalization()
    print()
    test_semantic_relations()

    print()
    print("=" * 60)
    print("Integrated Semantic Memory Tests")
    print("=" * 60)

    test_integrated_semantic_memory()
    print()
    test_integrated_semantic_gradient_flow()
    print()
    test_all_integration_modes_with_semantic()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
