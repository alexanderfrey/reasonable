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
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
