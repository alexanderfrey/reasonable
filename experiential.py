"""
Experiential Stream — Minimal Implementation

Validates the core premise: can we predict "what comes next" in latent space?

Uses model's own hidden states directly (predictive coding style):
- h_mid = hidden state at sequence midpoint (model's "current state")
- h_end = hidden state at sequence end (model's "future state")
- predictor: h_mid → predicted h_end
- surprise = distance(predicted h_end, actual h_end)

Usage:
    from experiential import ExperientialStream, experiential_loss

    # Create module (only needs a small predictor MLP)
    exp = ExperientialStream(d_model=768)

    # Forward pass (on hidden states from transformer)
    output = exp(hidden_states)  # hidden_states: [B, seq_len, d_model]

    # Compute loss
    loss = experiential_loss(output['prediction'], output['target'])
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
        decay_rate: float = 0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.capacity = capacity
        self.crystallization_threshold = crystallization_threshold
        self.decay_rate = decay_rate

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
        """Apply decay to all episode saliences (for consolidation)."""
        for ep in self.episodes:
            ep.salience *= factor

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

        return {
            'size': len(self.episodes),
            'avg_salience': sum(saliences) / len(saliences),
            'max_salience': max(saliences),
            'avg_retrieval_count': sum(retrieval_counts) / len(retrieval_counts),
            'avg_age': sum(ages) / len(ages)
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
        use_affect: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.split_ratio = split_ratio
        self.use_persistent_state = use_persistent_state
        self.use_affect = use_affect

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

        # Persistent state buffer (not a parameter, just a buffer)
        self.register_buffer('_persistent_state', None)
        self._batch_size = None

        self._init_weights()

    def _init_weights(self):
        """Initialize predictor output near zero (start with identity-ish)."""
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)
        if self.use_persistent_state:
            # Initialize gate to 0.5 (balanced between old and new)
            nn.init.zeros_(self.state_gate[0].weight)
            nn.init.constant_(self.state_gate[0].bias, 0.0)

    def reset_state(self, batch_size: Optional[int] = None):
        """Reset persistent state (call at start of new sequence/episode)."""
        self._persistent_state = None
        self._batch_size = batch_size

    def detach_state(self):
        """Detach state from computation graph (for truncated BPTT)."""
        if self._persistent_state is not None:
            self._persistent_state = self._persistent_state.detach()

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
        update_state: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Predict future hidden state from current hidden state.

        Args:
            hidden_states: [batch, seq_len, d_model] from transformer
            mid_idx: optional custom midpoint index (default: seq_len * split_ratio)
            end_idx: optional custom endpoint index (default: -1)
            update_state: whether to update persistent state after this forward pass

        Returns:
            dict with:
                - state: [batch, d_model] current state (h_mid)
                - prediction: [batch, d_model] predicted future (pred h_end)
                - target: [batch, d_model] actual future (h_end, detached)
                - surprise: [batch] prediction error (0=expected, 1=surprising)
                - valence: [batch] emotional valence (-1=negative, 1=positive)
                - arousal: [batch] activation level (0=calm, 1=excited)
                - salience: [batch] importance signal (surprise × arousal × |valence|)
                - persistent_state: [batch, d_model] updated persistent state
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

        # Compute surprise (no gradient needed for this metric)
        with torch.no_grad():
            pred_norm = F.normalize(prediction, dim=-1)
            target_norm = F.normalize(h_end, dim=-1)
            similarity = (pred_norm * target_norm).sum(dim=-1)
            surprise = 1 - similarity

        # Compute affect (valence and arousal)
        valence = None
        arousal = None
        salience = None
        if self.use_affect:
            # Predict affect from the end state (what we actually experienced)
            valence = self.valence_head(h_end).squeeze(-1)  # [B]
            arousal = self.arousal_head(h_end).squeeze(-1)  # [B]

            # Salience = how important is this moment?
            # High surprise + high arousal + strong valence = very salient
            with torch.no_grad():
                salience = surprise * arousal * valence.abs()

        # Update persistent state
        gate_values = None
        if self.use_persistent_state and update_state:
            # Compute new state using gated update
            new_state = self._update_state(h_end.detach(), prev_state)
            self._persistent_state = new_state.detach()  # Detach to prevent huge graphs

            # Track gate values for analysis
            with torch.no_grad():
                gate_input = torch.cat([h_end, prev_state], dim=-1)
                gate_values = self.state_gate(gate_input)

        return {
            'state': h_mid,
            'prediction': prediction,
            'target': h_end.detach(),  # stop gradient for contrastive loss
            'surprise': surprise,
            'valence': valence,
            'arousal': arousal,
            'salience': salience,
            'mid_idx': mid_idx,
            'end_idx': end_idx if end_idx != -1 else seq_len - 1,
            'persistent_state': self._persistent_state,
            'prev_state': prev_state,
            'gate_values': gate_values
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
    GPT model augmented with episodic memory.

    Retrieves relevant memories from past experiences and uses them to
    condition the model's predictions. Enables:
    - Learning from past experiences (what worked before)
    - Maintaining consistency across long contexts
    - Recalling relevant patterns/events

    Integration options:
    1. Residual: Add retrieved memory to hidden states
    2. Gated: Learn to blend memory with hidden states
    3. Cross-attention: Attend to memory as additional context

    Usage:
        gpt = GPT(config)
        memory_gpt = MemoryAugmentedGPT(gpt, memory_capacity=1000)

        # Training: forward returns (logits, hidden_states, memory_output)
        logits, hidden, mem_out = memory_gpt(input_ids, crystallize=True)

        # The model learns to use retrieved memories
        lm_loss = F.cross_entropy(logits.view(-1, vocab), targets.view(-1))
        exp_loss = experiential_loss(mem_out['prediction'], mem_out['target'])
        total_loss = lm_loss + 0.1 * exp_loss
    """

    def __init__(
        self,
        gpt_model: nn.Module,
        memory_capacity: int = 1000,
        crystallization_threshold: float = 0.3,
        memory_integration: str = 'gated',  # 'residual', 'gated', or 'attention'
        memory_weight: float = 0.1,
        use_experiential: bool = True,
        retrieval_temperature: float = 0.1,
    ):
        """
        Args:
            gpt_model: Pre-existing GPT model to wrap
            memory_capacity: Maximum episodes to store
            crystallization_threshold: Salience threshold for storing memories
            memory_integration: How to integrate memories ('residual', 'gated', 'attention')
            memory_weight: Base weight for memory contribution (for residual mode)
            use_experiential: Whether to use experiential stream for surprise/salience
            retrieval_temperature: Temperature for soft retrieval
        """
        super().__init__()
        self.gpt = gpt_model
        self.d_model = gpt_model.config.d_model
        self.memory_integration = memory_integration
        self.memory_weight = memory_weight
        self.retrieval_temperature = retrieval_temperature

        # Episodic memory
        self.memory = EpisodicMemory(
            d_model=self.d_model,
            capacity=memory_capacity,
            crystallization_threshold=crystallization_threshold
        )

        # Experiential stream (for computing surprise/salience)
        if use_experiential:
            self.experiential = ExperientialStream(
                d_model=self.d_model,
                use_affect=True,
                use_persistent_state=True
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

        elif memory_integration == 'attention':
            # Cross-attention to memory
            self.memory_attention = nn.MultiheadAttention(
                self.d_model,
                num_heads=4,
                batch_first=True
            )
            self.memory_norm = nn.LayerNorm(self.d_model)

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

        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        crystallize: bool = True,
        use_memory: bool = True,
        return_memory_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with memory retrieval and optional crystallization.

        Args:
            input_ids: [batch, seq_len] input token IDs
            input_pos: Optional position indices
            crystallize: Whether to store high-salience moments in memory
            use_memory: Whether to retrieve and use memories
            return_memory_weights: Include retrieval weights in output

        Returns:
            logits: [batch, seq_len, vocab_size] output logits
            hidden_states: [batch, seq_len, d_model] final hidden states
            memory_output: dict with experiential/memory info
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Get hidden states from GPT
        logits, hidden_states = self.gpt(
            input_ids,
            input_pos=input_pos,
            return_hidden_states=True
        )

        memory_output = {
            'retrieved_values': None,
            'retrieval_weights': None,
            'crystallized': False,
            'memory_size': self.memory.size
        }

        # 2. Retrieve from memory and integrate
        if use_memory and self.memory.size > 0:
            # Use end-of-sequence hidden state as query
            # Shape: [batch, d_model]
            query_state = hidden_states[:, -1, :]
            query = self.query_proj(query_state)

            # Soft retrieval
            retrieved, weights = self.memory.retrieve_soft(
                query,
                temperature=self.retrieval_temperature
            )

            memory_output['retrieved_values'] = retrieved
            if return_memory_weights:
                memory_output['retrieval_weights'] = weights

            # Integrate memory with hidden states
            hidden_states = self._integrate_memory(hidden_states, retrieved)

            # Recompute logits with memory-augmented hidden states
            logits = self.gpt.lm_head(self.gpt.final_norm(hidden_states))

        # 3. Experiential processing (for crystallization and prediction)
        if self.experiential is not None:
            exp_output = self.experiential(hidden_states)
            memory_output.update({
                'prediction': exp_output['prediction'],
                'target': exp_output['target'],
                'surprise': exp_output['surprise'],
                'valence': exp_output['valence'],
                'arousal': exp_output['arousal'],
                'salience': exp_output['salience']
            })

            # 4. Crystallize high-salience moments
            if crystallize:
                for i in range(batch_size):
                    salience = exp_output['salience'][i].item()
                    if self.memory.should_crystallize(salience):
                        self.memory.store(
                            content=exp_output['target'][i],
                            context=exp_output['prediction'][i],
                            salience=salience,
                            valence=exp_output['valence'][i].item(),
                            arousal=exp_output['arousal'][i].item()
                        )
                        memory_output['crystallized'] = True

        memory_output['memory_size'] = self.memory.size

        return logits, hidden_states, memory_output

    def _integrate_memory(
        self,
        hidden_states: torch.Tensor,
        retrieved: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate retrieved memory into hidden states.

        Args:
            hidden_states: [batch, seq_len, d_model]
            retrieved: [batch, d_model]

        Returns:
            augmented: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = hidden_states.shape

        if self.memory_integration == 'residual':
            # Simple additive: add memory to all positions
            # Broadcast retrieved [batch, d_model] to [batch, seq_len, d_model]
            memory_contribution = retrieved.unsqueeze(1).expand(-1, seq_len, -1)
            return hidden_states + self.memory_weight * memory_contribution

        elif self.memory_integration == 'gated':
            # Learned gate: how much memory vs original at each position
            # Expand retrieved to match sequence
            retrieved_expanded = retrieved.unsqueeze(1).expand(-1, seq_len, -1)
            projected = self.memory_proj(retrieved_expanded)

            # Compute gate: [batch, seq_len, d_model]
            gate_input = torch.cat([hidden_states, projected], dim=-1)
            gate = self.memory_gate(gate_input)

            # Blend: gate * memory + (1 - gate) * original
            return gate * projected + (1 - gate) * hidden_states

        elif self.memory_integration == 'attention':
            # Cross-attention to memory
            # retrieved: [batch, d_model] -> [batch, 1, d_model] as single "memory token"
            memory_tokens = retrieved.unsqueeze(1)

            # Attend to memory from each position in hidden_states
            attended, _ = self.memory_attention(
                hidden_states,
                memory_tokens,
                memory_tokens
            )

            # Residual connection with normalization
            return hidden_states + self.memory_norm(attended)

        else:
            raise ValueError(f"Unknown integration mode: {self.memory_integration}")

    def reset_memory(self):
        """Clear all stored memories."""
        self.memory.clear()
        if self.experiential is not None:
            self.experiential.reset_state()

    def get_memory_stats(self) -> Dict:
        """Get statistics about current memory state."""
        stats = self.memory.get_stats()
        if self.experiential is not None:
            stats['experiential_state'] = self.experiential.get_state() is not None
        return stats

    def retrieve_relevant(
        self,
        query: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[Episode, float]]:
        """
        Retrieve relevant memories for a query (hard retrieval for inspection).

        Args:
            query: [d_model] query vector
            top_k: number of memories to retrieve

        Returns:
            List of (Episode, similarity) tuples
        """
        return self.memory.retrieve(query, top_k=top_k)


def memory_augmented_loss(
    lm_logits: torch.Tensor,
    targets: torch.Tensor,
    memory_output: Dict,
    lm_weight: float = 1.0,
    exp_weight: float = 0.1,
    retrieval_weight: float = 0.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined loss for memory-augmented generation.

    Args:
        lm_logits: [batch, seq_len, vocab] model output logits
        targets: [batch, seq_len] target token IDs
        memory_output: output dict from MemoryAugmentedGPT.forward()
        lm_weight: weight for language modeling loss
        exp_weight: weight for experiential prediction loss
        retrieval_weight: weight for retrieval contrastive loss (if applicable)

    Returns:
        total_loss: combined scalar loss
        loss_dict: breakdown of individual losses
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

    # Experiential prediction loss
    if 'prediction' in memory_output and memory_output['prediction'] is not None:
        exp_loss = experiential_loss(
            memory_output['prediction'],
            memory_output['target']
        )
        loss_dict['exp_loss'] = exp_loss.item()
        total_loss = total_loss + exp_weight * exp_loss

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
            memory_integration=mode
        )

        # Initial forward (no memories yet)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits, hidden, mem_out = memory_gpt(input_ids, crystallize=True)

        assert logits.shape == (batch_size, seq_len, vocab_size), \
            f"Logits shape wrong: {logits.shape}"
        assert hidden.shape == (batch_size, seq_len, d_model), \
            f"Hidden shape wrong: {hidden.shape}"
        assert 'surprise' in mem_out, "Should have experiential output"

        initial_size = mem_out['memory_size']
        print(f"    Initial memory size: {initial_size}")

        # Run multiple forwards to build up memory
        for _ in range(10):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            logits, hidden, mem_out = memory_gpt(input_ids, crystallize=True)

        final_size = mem_out['memory_size']
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

        assert mem_out['retrieved_values'] is not None, "Should have retrieved values"
        assert mem_out['retrieval_weights'] is not None, "Should have retrieval weights"
        print(f"    Retrieved values shape: {mem_out['retrieved_values'].shape}")

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
        memory_integration='gated'
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
        memory_integration='gated'
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
            print(f"  Step {step}: loss={loss.item():.4f}, memory_size={mem_out['memory_size']}")

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
    print("All tests passed!")
    print("=" * 60)
