"""
Experiential Stream v0.1 — Minimal Implementation

Validates the core premise: can we predict "what comes next" in latent space?

Usage:
    from experiential import ExperientialStreamV01, experiential_loss

    # Create module
    exp = ExperientialStreamV01(d_model=768)

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
    n_heads: int = 4
    predictor_hidden_mult: int = 2
    temperature: float = 0.1
    split_ratio: float = 0.5  # first half / second half split


class ExperientialStreamV01(nn.Module):
    """
    Minimal experiential stream for validation.

    Core idea:
    - Encode first half of sequence into a "state" vector
    - Predict embedding of second half from that state
    - Use contrastive loss: prediction should match its target, not others

    This tests whether the model can capture "where things are going"
    from "where things are now."
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        predictor_hidden_mult: int = 2,
        split_ratio: float = 0.5
    ):
        super().__init__()
        self.d_model = d_model
        self.split_ratio = split_ratio

        # State encoder: attention-pool first half into single vector
        self.state_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.state_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.0
        )
        self.state_norm = nn.LayerNorm(d_model)

        # Predictor: state → predicted future embedding
        hidden_dim = d_model * predictor_hidden_mult
        self.predictor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model)
        )

        # Target encoder: attention-pool second half
        # Separate parameters to avoid shortcut learning
        self.target_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.target_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.0
        )
        self.target_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Predictor output near zero initially
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_halves: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process hidden states through experiential stream.

        Args:
            hidden_states: [batch, seq_len, d_model] from transformer
            return_halves: if True, also return the raw halves

        Returns:
            dict with:
                - state: [batch, d_model] encoded current state
                - prediction: [batch, d_model] predicted future
                - target: [batch, d_model] actual future (detached)
                - surprise: [batch] prediction error (0 = perfect, 1 = orthogonal)
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Split into first and second half
        split_idx = int(seq_len * self.split_ratio)
        first_half = hidden_states[:, :split_idx, :]
        second_half = hidden_states[:, split_idx:, :]

        # Encode first half into state
        state = self._encode_state(first_half)  # [B, d]

        # Predict future embedding
        prediction = self.predictor(state)  # [B, d]

        # Encode second half into target
        target = self._encode_target(second_half)  # [B, d]

        # Compute surprise (prediction error) — no gradient needed
        with torch.no_grad():
            pred_norm = F.normalize(prediction, dim=-1)
            tgt_norm = F.normalize(target, dim=-1)
            similarity = (pred_norm * tgt_norm).sum(dim=-1)
            surprise = 1 - similarity  # 0 = identical, 1 = orthogonal, 2 = opposite

        result = {
            'state': state,
            'prediction': prediction,
            'target': target.detach(),  # stop gradient for contrastive loss
            'surprise': surprise
        }

        if return_halves:
            result['first_half'] = first_half
            result['second_half'] = second_half

        return result

    def _encode_state(self, first_half: torch.Tensor) -> torch.Tensor:
        """Encode first half into state vector via attention pooling."""
        batch_size = first_half.size(0)
        query = self.state_query.expand(batch_size, -1, -1)

        state, _ = self.state_attn(query, first_half, first_half)
        state = self.state_norm(state.squeeze(1))

        return state

    def _encode_target(self, second_half: torch.Tensor) -> torch.Tensor:
        """Encode second half into target vector via attention pooling."""
        batch_size = second_half.size(0)
        query = self.target_query.expand(batch_size, -1, -1)

        target, _ = self.target_attn(query, second_half, second_half)
        target = self.target_norm(target.squeeze(1))

        return target

    def predict_from_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Generate prediction from a state vector.
        Useful for probing or generation.
        """
        return self.predictor(state)


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
        output: dict from ExperientialStreamV01.forward()

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
    print("Testing ExperientialStreamV01 shapes...")

    batch_size = 4
    seq_len = 128
    d_model = 256

    exp = ExperientialStreamV01(d_model=d_model)
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    output = exp(hidden_states)

    assert output['state'].shape == (batch_size, d_model), f"State shape wrong: {output['state'].shape}"
    assert output['prediction'].shape == (batch_size, d_model), f"Prediction shape wrong: {output['prediction'].shape}"
    assert output['target'].shape == (batch_size, d_model), f"Target shape wrong: {output['target'].shape}"
    assert output['surprise'].shape == (batch_size,), f"Surprise shape wrong: {output['surprise'].shape}"

    print("  All shapes correct!")
    return True


def test_gradient_flow():
    """Test that gradients flow through prediction but not target."""
    print("Testing gradient flow...")

    exp = ExperientialStreamV01(d_model=128)
    hidden_states = torch.randn(4, 64, 128, requires_grad=True)

    output = exp(hidden_states)
    loss = experiential_loss(output['prediction'], output['target'])
    loss.backward()

    # Check gradients exist for model parameters
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in exp.parameters())
    assert has_grads, "No gradients in model parameters!"

    # Check hidden_states has gradients (for full integration)
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

    exp = ExperientialStreamV01(d_model=d_model)
    optimizer = torch.optim.Adam(exp.parameters(), lr=1e-3)

    # Track metrics
    initial_acc = None
    final_acc = None

    for step in range(n_steps):
        # Create structured data: first half predicts second half
        # (simple pattern: second half = transform of first half)
        first = torch.randn(batch_size, seq_len // 2, d_model)
        second = first.mean(dim=1, keepdim=True).expand(-1, seq_len // 2, -1)
        second = second + torch.randn_like(second) * 0.1

        hidden_states = torch.cat([first, second], dim=1)

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


if __name__ == "__main__":
    print("=" * 60)
    print("ExperientialStreamV01 Tests")
    print("=" * 60)

    test_shapes()
    print()
    test_gradient_flow()
    print()
    test_learning()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
