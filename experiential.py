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
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ExperientialConfig:
    """Configuration for experiential stream."""
    d_model: int
    predictor_hidden_mult: int = 2
    temperature: float = 0.1
    split_ratio: float = 0.5  # first half / second half split


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
        use_persistent_state: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.split_ratio = split_ratio
        self.use_persistent_state = use_persistent_state

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
                - surprise: [batch] prediction error
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


if __name__ == "__main__":
    print("=" * 60)
    print("ExperientialStream Tests")
    print("=" * 60)

    test_shapes()
    print()
    test_multiscale()
    print()
    test_gradient_flow()
    print()
    test_learning()
    print()
    test_persistent_state()
    print()
    test_persistent_state_learning()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
