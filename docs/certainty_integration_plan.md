# Certainty Integration Plan

## Current State Analysis

### What Already Exists

| Signal | Location | Implementation | Quality |
|--------|----------|----------------|---------|
| `certainty` | `self_state.py:748` | `1 - clamp(surprise, 0, 1)` | Simple, could be richer |
| `confidence_gate` | `experiential.py:1868` | Learned `[B, d_model]` | Rich, per-dimension |
| `self_confidence` | `self_state.py:244` | Self-model confidence | Learned, scalar |
| `meta_surprise` | `experiential.py:1308` | `|predicted - actual|` surprise | Strong training signal |
| `predicted_surprise` | `experiential.py` | Self-awareness of surprise | Trained via loss |

### Current Flow
```
ExperientialStream                    SelfState
      │                                   │
      ├─→ surprise ──────────────────────→│──→ certainty = 1 - surprise
      ├─→ meta_surprise ─────────────────→│──→ (not used directly)
      ├─→ confidence_gate ───────────────→│──→ (not used in soma)
      │                                   │
      │                                   ▼
      │                            SomaIntegrator
      │                                   │
      │                                   ▼
      └─────────────────────────────→ soma feedback
                                    (hidden gate, Q bias)
```

### Gap Analysis

1. **Certainty is too simple**: Just `1 - surprise`, ignores confidence_gate and self_confidence
2. **No certainty calibration**: No loss ensuring certainty matches actual correctness
3. **No thinking loop**: Can't iterate to increase certainty
4. **No certainty-driven halting**: No mechanism to "think more" when uncertain
5. **Questions not integrated**: Questioning system exists in docs but not in code

---

## Integration Plan

### Phase 1: Enhanced Certainty Signal (Low Risk, High Value)
**Goal**: Replace simple `1 - surprise` with richer certainty estimation

**Files to modify**:
- `self_state.py` - SelfState.extract_signals_from_experiential()

**Changes**:

```python
# In self_state.py, replace lines ~745-755

# BEFORE:
certainty = 1.0 - torch.clamp(surprise, 0, 1)

# AFTER:
def compute_certainty(
    self,
    surprise: torch.Tensor,           # [B]
    confidence_gate: torch.Tensor,    # [B, d_model] from experiential
    meta_surprise: torch.Tensor,      # [B]
) -> torch.Tensor:
    """
    Enhanced certainty combining multiple signals.

    - Low surprise → high certainty
    - High confidence_gate → high certainty
    - Low meta_surprise → high certainty (know myself well)
    """
    # Base certainty from surprise
    base_certainty = 1.0 - torch.clamp(surprise, 0, 1)

    # Confidence contribution (mean of per-dimension confidence)
    confidence_contribution = confidence_gate.mean(dim=-1)  # [B]

    # Meta-certainty: if I predicted my surprise well, I'm more certain
    meta_certainty = 1.0 - torch.clamp(meta_surprise, 0, 1)

    # Weighted combination (learnable weights could be added)
    certainty = (
        0.4 * base_certainty +
        0.4 * confidence_contribution +
        0.2 * meta_certainty
    )

    return certainty
```

**Integration point**:
- `experiential.py:3083-3120` where `extract_signals_from_experiential` is called
- Pass `confidence_gate` and `meta_surprise` from experiential output

**Testing**:
- Run existing tests
- Log certainty distribution during training
- Verify certainty correlates with actual prediction accuracy

---

### Phase 2: Certainty Head (Dedicated Module)
**Goal**: Add dedicated certainty estimation with calibration loss

> **Note on Estimation vs Training Strategy**:
> The CertaintyHead is an **estimation module** - it outputs a certainty score in a single forward pass.
> Simple calibration loss (BCE against correctness) is used here.
> **Dual-tick selection** (CTM-style min-loss + max-certainty) is a **training strategy** that
> requires multiple forward passes at different ticks - this belongs in Phase 4's CertaintyDrivenLoss.

**Files to create/modify**:
- Create `certainty.py` (new file)
- Modify `experiential.py` to use CertaintyHead
- Modify training loss in `experiential.py:memory_augmented_loss()`

**New file `certainty.py`**:

```python
"""
Certainty estimation module.

Provides calibrated certainty scores that can be trained to match
actual prediction correctness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CertaintyHead(nn.Module):
    """
    Estimates calibrated certainty from hidden states and other signals.
    """

    def __init__(self, d_model: int, d_soma: int = 64):
        super().__init__()

        # Main certainty network
        self.certainty_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Multi-signal certainty (optional, when soma available)
        self.multi_signal_net = nn.Sequential(
            nn.Linear(d_model + d_soma + 2, d_model // 2),  # +2 for surprise, meta_surprise
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden: torch.Tensor,                    # [B, seq_len, d_model] or [B, d_model]
        soma: torch.Tensor = None,               # [B, d_soma]
        surprise: torch.Tensor = None,           # [B]
        meta_surprise: torch.Tensor = None,      # [B]
    ) -> torch.Tensor:
        """
        Compute certainty score.

        Returns:
            certainty: [B] in [0, 1]
        """
        # Handle sequence input
        if hidden.dim() == 3:
            hidden = hidden[:, -1, :]  # Use last position [B, d_model]

        # If we have all signals, use multi-signal network
        if soma is not None and surprise is not None and meta_surprise is not None:
            combined = torch.cat([
                hidden,
                soma,
                surprise.unsqueeze(-1),
                meta_surprise.unsqueeze(-1),
            ], dim=-1)
            certainty = self.multi_signal_net(combined).squeeze(-1)
        else:
            certainty = self.certainty_net(hidden).squeeze(-1)

        return certainty


def certainty_calibration_loss(
    certainty: torch.Tensor,      # [B] predicted certainty
    was_correct: torch.Tensor,    # [B] binary, whether prediction was correct
) -> torch.Tensor:
    """
    Calibration loss: certainty should match actual correctness.

    Uses BCE so that:
    - High certainty + correct = low loss (good)
    - High certainty + wrong = high loss (overconfident)
    - Low certainty + wrong = low loss (honest uncertainty)
    - Low certainty + correct = medium loss (underconfident)
    """
    return F.binary_cross_entropy(certainty, was_correct.float())
```

**Integration in MemoryAugmentedGPT**:

```python
# In experiential.py MemoryAugmentedGPT.__init__
self.certainty_head = CertaintyHead(d_model, d_soma) if use_certainty else None

# In forward pass, after experiential processing
if self.certainty_head is not None:
    certainty = self.certainty_head(
        hidden=hidden_states,
        soma=self_state_output.get('soma'),
        surprise=exp_output.get('surprise'),
        meta_surprise=exp_output.get('meta_surprise'),
    )
    memory_output['certainty'] = certainty
```

**Integration in loss function**:

```python
# In memory_augmented_loss(), add:
if 'certainty' in memory_output and targets is not None:
    # Compute whether predictions were correct
    predictions = logits.argmax(dim=-1)  # [B, seq_len]
    was_correct = (predictions[:, :-1] == targets[:, 1:]).float().mean(dim=-1)  # [B]

    calibration_loss = certainty_calibration_loss(
        memory_output['certainty'],
        was_correct
    )
    loss_dict['certainty_calibration'] = calibration_loss
    total_loss = total_loss + certainty_calibration_weight * calibration_loss
```

**Testing**:
- Unit test CertaintyHead
- Verify calibration loss decreases during training
- Plot calibration curve (certainty vs accuracy)

---

### Phase 3: Certainty in Soma Integration
**Goal**: Make soma aware of certainty, modulate feedback strength

**Files to modify**:
- `self_state.py` - SomaIntegrator, SomaFeedback

**Changes to SomaIntegrator**:

```python
# Add certainty-weighted signal integration

def forward(self, prev_soma, signals, certainty=None):
    # ... existing signal gating ...

    # NEW: Modulate integration strength by certainty
    if certainty is not None:
        # When uncertain, be more conservative in state updates
        # certainty_gate: high certainty → normal update, low → smaller update
        certainty_gate = 0.5 + 0.5 * certainty.unsqueeze(-1)  # [B, 1] in [0.5, 1.0]
        signal_contribution = signal_contribution * certainty_gate

    new_soma = self.decay * prev_soma + (1 - self.decay) * signal_contribution
    # ... rest of method ...
```

**Changes to SomaFeedback**:

```python
# Modulate feedback strength by certainty

def forward(self, hidden_states, soma, certainty=None):
    # Compute gate and add as before
    gate = self.hidden_gate(soma).unsqueeze(1)
    add = self.hidden_add(soma).unsqueeze(1)

    # NEW: Scale feedback by certainty
    if certainty is not None:
        # When uncertain, reduce soma's influence on hidden states
        # This prevents uncertain internal state from corrupting processing
        feedback_strength = 0.3 + 0.7 * certainty.unsqueeze(-1).unsqueeze(-1)
        gate = 1 - feedback_strength * (1 - gate)  # Blend toward 1 (no gating)
        add = add * feedback_strength

    modulated = hidden_states * gate + 0.1 * add
    return modulated
```

**Rationale**: When the model is uncertain, it should:
1. Update soma more conservatively (don't overreact to uncertain signals)
2. Let soma influence processing less (don't let uncertain mood affect predictions)

---

### Phase 4: Thinking Steps (Mini CTM)
**Goal**: Add ability to "think more" when uncertain

**Files to create/modify**:
- Create `thinking.py` (new file)
- Modify `experiential.py` to optionally use thinking loop

**New file `thinking.py`**:

```python
"""
Thinking loop module.

Allows the model to iterate on its hidden state when uncertain,
inspired by Continuous Thought Machines (CTM).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class ThinkStep(nn.Module):
    """Single step of iterative refinement."""

    def __init__(self, d_model: int, n_heads: int = 8, d_soma: int = 64):
        super().__init__()

        # Self-attention for refinement
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )

        # Soma modulation
        self.soma_gate = nn.Sequential(
            nn.Linear(d_soma, d_model),
            nn.Sigmoid(),
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden: torch.Tensor,      # [B, seq_len, d_model]
        soma: torch.Tensor,        # [B, d_soma]
    ) -> torch.Tensor:
        """One thinking step."""

        # Self-attention
        residual = hidden
        hidden = self.norm1(hidden)
        hidden_attn, _ = self.self_attn(hidden, hidden, hidden)
        hidden = residual + hidden_attn

        # Soma modulation
        soma_gate = self.soma_gate(soma).unsqueeze(1)
        hidden = hidden * soma_gate

        # FFN
        residual = hidden
        hidden = self.norm2(hidden)
        hidden = residual + self.ffn(hidden)

        return hidden


class ThinkingLoop(nn.Module):
    """
    Iterative thinking with certainty-driven halting.

    Simple version: fixed number of steps, track certainty.
    Advanced version: halt when certain (CTM-style).
    """

    def __init__(
        self,
        d_model: int,
        d_soma: int = 64,
        max_ticks: int = 3,
        certainty_threshold: float = 0.8,
        n_heads: int = 8,
    ):
        super().__init__()

        self.max_ticks = max_ticks
        self.certainty_threshold = certainty_threshold

        # Thinking step (shared across ticks for efficiency)
        self.think_step = ThinkStep(d_model, n_heads, d_soma)

        # Certainty head (per-tick estimation)
        self.certainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden: torch.Tensor,       # [B, seq_len, d_model]
        soma: torch.Tensor,         # [B, d_soma]
        return_trajectory: bool = False,
    ) -> Dict[str, Any]:
        """
        Think until certain or max ticks.
        """
        B = hidden.shape[0]

        certainty_trajectory = []
        hidden_trajectory = [] if return_trajectory else None

        for tick in range(self.max_ticks):
            # One thinking step
            hidden = self.think_step(hidden, soma)

            # Compute certainty (from last position)
            certainty = self.certainty_head(hidden[:, -1, :]).squeeze(-1)  # [B]
            certainty_trajectory.append(certainty)

            if return_trajectory:
                hidden_trajectory.append(hidden.clone())

            # Early halt if all samples are certain
            if (certainty > self.certainty_threshold).all():
                break

        # Stack trajectories
        certainty_trajectory = torch.stack(certainty_trajectory, dim=1)  # [B, ticks]

        return {
            'hidden': hidden,
            'certainty': certainty_trajectory[:, -1],  # Final certainty
            'certainty_trajectory': certainty_trajectory,
            'hidden_trajectory': hidden_trajectory,
            'ticks_used': tick + 1,
        }
```

**Integration in MemoryAugmentedGPT**:

```python
# In __init__
self.thinking_loop = ThinkingLoop(d_model, d_soma, max_ticks=3) if use_thinking else None

# In forward, after experiential and self-state processing:
if self.thinking_loop is not None:
    thinking_output = self.thinking_loop(
        hidden=hidden_states,
        soma=self_state_output.get('soma', torch.zeros(B, self.d_soma, device=device)),
    )
    hidden_states = thinking_output['hidden']
    memory_output['thinking_certainty'] = thinking_output['certainty']
    memory_output['thinking_ticks'] = thinking_output['ticks_used']
```

**Config additions**:
```python
# In MemoryAugmentedGPT.__init__ parameters
use_thinking: bool = False,
thinking_max_ticks: int = 3,
thinking_certainty_threshold: float = 0.8,
```

#### Dual-Tick Selection Loss (CTM-Style Training)

This is where CTM's dual-tick selection strategy belongs. Unlike Phase 2's simple calibration loss
(which trains certainty at a single point), this loss operates across the tick trajectory.

```python
class CertaintyDrivenLoss(nn.Module):
    """
    CTM-inspired loss that selects two ticks per sample:
    - t_min_loss: tick with minimum task loss (best prediction)
    - t_max_certainty: tick with maximum certainty

    This trains the model to:
    1. Find good answers at some tick (correctness)
    2. Be confident when correct (calibration)
    3. Support adaptive compute at inference (efficiency)
    """

    def __init__(self, task_weight: float = 0.5, certainty_weight: float = 0.5):
        super().__init__()
        self.task_weight = task_weight
        self.certainty_weight = certainty_weight

    def forward(
        self,
        tick_losses: torch.Tensor,        # [B, T] task loss at each tick
        tick_certainties: torch.Tensor,   # [B, T] certainty at each tick
        was_correct: torch.Tensor,        # [B, T] whether prediction was correct
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CTM-style dual-tick loss.

        For each sample:
        1. Find t_min_loss = argmin(tick_losses[b, :])
        2. Find t_max_certainty = argmax(tick_certainties[b, :])
        3. Loss = 0.5 * loss[t_min_loss] + 0.5 * loss[t_max_certainty]
        """
        B, T = tick_losses.shape

        # Find best ticks per sample
        t_min_loss = tick_losses.argmin(dim=1)      # [B]
        t_max_certainty = tick_certainties.argmax(dim=1)  # [B]

        # Gather losses at selected ticks
        batch_idx = torch.arange(B, device=tick_losses.device)
        loss_at_min = tick_losses[batch_idx, t_min_loss]
        loss_at_max_cert = tick_losses[batch_idx, t_max_certainty]

        # Task loss: average of losses at both selected ticks
        task_loss = self.task_weight * loss_at_min + self.certainty_weight * loss_at_max_cert

        # Calibration at max-certainty tick: certainty should match correctness
        cert_at_max = tick_certainties[batch_idx, t_max_certainty]
        correct_at_max = was_correct[batch_idx, t_max_certainty]
        calibration_loss = F.binary_cross_entropy(cert_at_max, correct_at_max.float())

        return {
            'loss': task_loss.mean() + 0.1 * calibration_loss,
            'task_loss': task_loss.mean(),
            'calibration_loss': calibration_loss,
            't_min_loss': t_min_loss,
            't_max_certainty': t_max_certainty,
        }
```

**Why dual-tick selection lives here, not in CertaintyHead**:

| Component | Purpose | Requires |
|-----------|---------|----------|
| **CertaintyHead** (Phase 2) | Estimate certainty at a single state | Single forward pass |
| **CertaintyDrivenLoss** (Phase 4) | Train across tick trajectory | Multiple ticks from ThinkingLoop |

The CertaintyHead is called at each tick to produce `tick_certainties`. The CertaintyDrivenLoss
then selects which ticks to optimize against. This separation allows:
- Phase 2: Works without thinking loop (simple calibration)
- Phase 4: Full CTM-style adaptive compute training

---

### Phase 5: Certainty-Driven Question Triggering
**Goal**: Generate questions when certainty remains low after thinking

**Files to modify**:
- `thinking.py` - add question generation
- Create `question.py` for Question dataclass

**Add to ThinkingLoop**:

```python
def forward(self, hidden, soma, question_threshold=0.3, ...):
    # ... existing thinking loop ...

    final_certainty = certainty_trajectory[:, -1]

    # Identify samples that need questions
    needs_question = final_certainty < question_threshold

    # Generate question info for uncertain samples
    questions = None
    if needs_question.any():
        questions = self._generate_question_info(
            hidden[needs_question],
            final_certainty[needs_question],
        )

    return {
        # ... existing outputs ...
        'needs_question': needs_question,
        'questions': questions,
    }

def _generate_question_info(self, hidden, certainty):
    """
    Generate question metadata for uncertain samples.

    Full question generation would use a decoder.
    For now, return uncertainty info for the action module.
    """
    return {
        'hidden_state': hidden,
        'certainty': certainty,
        'uncertainty_magnitude': 1 - certainty,
    }
```

---

## Implementation Order

```
Phase 1: Enhanced Certainty Signal
    │
    ├── Modify self_state.py extract_signals
    ├── Pass confidence_gate, meta_surprise from experiential
    ├── Test: certainty distribution, correlation with accuracy
    │
    ▼
Phase 2: Certainty Head (Estimation Module)
    │
    ├── Create certainty.py with CertaintyHead
    ├── Add CertaintyHead to MemoryAugmentedGPT
    ├── Add SIMPLE calibration loss (BCE against correctness)
    ├── Note: No tick selection here - single forward pass only
    ├── Test: calibration curve
    │
    ▼
Phase 3: Certainty in Soma
    │
    ├── Modify SomaIntegrator (certainty-weighted updates)
    ├── Modify SomaFeedback (certainty-scaled feedback)
    ├── Test: behavior under high vs low certainty
    │
    ▼
Phase 4: Thinking Steps + Dual-Tick Loss
    │
    ├── Create thinking.py with ThinkStep, ThinkingLoop
    ├── Add ThinkingLoop to MemoryAugmentedGPT
    ├── Add CertaintyDrivenLoss (CTM-style dual-tick selection)
    ├── Dual-tick selects: t_min_loss + t_max_certainty
    ├── Add config flags
    ├── Test: certainty increases with ticks, dual-tick selection works
    │
    ▼
Phase 5: Question Triggering
    │
    ├── Add question generation to ThinkingLoop
    ├── Integrate with action module (future)
    ├── Test: questions generated for low-certainty samples
```

---

## Config Changes Summary

```python
# New config parameters for MemoryAugmentedGPT

# Phase 1
use_enhanced_certainty: bool = True

# Phase 2
use_certainty_head: bool = True
certainty_calibration_weight: float = 0.1

# Phase 3
certainty_modulates_soma: bool = True
certainty_modulates_feedback: bool = True

# Phase 4
use_thinking_loop: bool = False  # Off by default initially
thinking_max_ticks: int = 3
thinking_certainty_threshold: float = 0.8
use_dual_tick_loss: bool = True  # CTM-style training when thinking_loop enabled
dual_tick_task_weight: float = 0.5
dual_tick_certainty_weight: float = 0.5

# Phase 5
question_certainty_threshold: float = 0.3
```

---

## Testing Strategy

### Unit Tests
1. `test_certainty_head.py` - CertaintyHead forward, simple calibration loss
2. `test_thinking_loop.py` - ThinkingLoop halting, certainty trajectory
3. `test_certainty_soma.py` - Certainty modulation of soma
4. `test_dual_tick_loss.py` - CertaintyDrivenLoss tick selection, gradient flow

### Integration Tests
1. Run training with enhanced certainty, verify no regression
2. Add certainty calibration loss, verify calibration improves
3. Enable thinking loop, verify ticks adapt to difficulty
4. Enable dual-tick loss, verify t_min_loss and t_max_certainty are sensibly selected

### Metrics to Track
1. **Certainty calibration**: Plot certainty vs accuracy buckets
2. **Thinking efficiency**: Average ticks used vs accuracy
3. **Question rate**: % of samples with certainty < threshold
4. **Tick divergence**: How often t_min_loss ≠ t_max_certainty (measures calibration gap)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Training instability from new loss | Start with low weight (0.01), increase gradually |
| Thinking loop adds latency | Keep max_ticks low (3), disable for inference |
| Certainty not calibrated | Use temperature scaling post-training if needed |
| Regression on existing metrics | A/B test each phase, keep fallback path |

---

## Success Criteria

### Phase 1
- [ ] Certainty uses multiple signals
- [ ] Certainty correlates with accuracy (r > 0.3)

### Phase 2
- [ ] Calibration loss decreases during training
- [ ] Calibration error < 0.1 (bucket-wise)

### Phase 3
- [ ] Soma updates smaller when uncertain
- [ ] Hidden state modulation reduced when uncertain

### Phase 4
- [ ] Certainty increases with ticks on hard examples
- [ ] Easy examples halt early (fewer ticks)
- [ ] Dual-tick selection: t_min_loss ≠ t_max_certainty on some samples (learning is non-trivial)
- [ ] CertaintyDrivenLoss converges

### Phase 5
- [ ] Questions generated for low-certainty samples
- [ ] Question info includes uncertainty source
