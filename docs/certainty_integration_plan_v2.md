# Certainty Integration Plan v2

## Executive Summary

This plan integrates CTM-inspired certainty into your existing experiential architecture. The key insight: **your system already computes the raw materials for certainty** (surprise, meta_surprise, confidence_gate, self_confidence) but doesn't unify them into an actionable control signal.

We will:
1. Unify existing signals into a calibrated CertaintyHead
2. Add a ThinkingLoop for adaptive computation
3. Use certainty to control soma updates, memory crystallization, and question generation
4. Train with CTM-style dual-tick selection for calibrated uncertainty

---

## Current Architecture Analysis

### What You Already Have

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXISTING CERTAINTY-RELATED SIGNALS                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ExperientialStream (experiential.py)                                        │
│  ├── surprise [B]                    ← excess_surprisal × novelty            │
│  │   Location: compute_surprise_signal() lines 1487-1607                     │
│  │   Range: [0, 1] via centered sigmoid                                      │
│  │                                                                           │
│  ├── meta_surprise [B]               ← |predicted_surprise - actual|         │
│  │   Location: forward() line 1816                                           │
│  │   Range: [0, 1]                                                           │
│  │   Meaning: "How wrong was my self-prediction?"                            │
│  │                                                                           │
│  ├── confidence_gate [B, d_model]    ← sigmoid([h_end, meta_surprise])       │
│  │   Location: forward() lines 1864-1882                                     │
│  │   Range: [0, 1] per dimension                                             │
│  │   Usage: modulated_output = gate * h_end + (1-gate) * fallback            │
│  │                                                                           │
│  └── salience [B]                    ← surprise × (1 + meta_surprise_weight) │
│      Location: forward() lines 1837-1852                                     │
│      Usage: Memory crystallization priority                                  │
│                                                                              │
│  SelfState (self_state.py)                                                   │
│  ├── self_surprise [B]               ← ||predicted_delta - actual_delta||   │
│  │   Location: SelfModel.forward() lines 286-300                             │
│  │   Meaning: "How well do I know my own reactions?"                         │
│  │                                                                           │
│  ├── self_confidence [B]             ← sigmoid([soma, input])                │
│  │   Location: SelfModel lines 271-273                                       │
│  │   Usage: Weights self_surprise loss                                       │
│  │                                                                           │
│  └── certainty signal [B]            ← 1 - clamp(surprise, 0, 1)             │
│      Location: extract_signals() line 748                                    │
│      Usage: One of 6 soma integration signals                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Gap

These signals exist but aren't unified:
- `confidence_gate` modulates hidden states but isn't used for decision-making
- `certainty` in soma is just `1 - surprise`, ignoring meta_surprise and self_confidence
- No mechanism to "think more" when uncertain
- No trigger for questions based on persistent uncertainty
- No calibration loss ensuring certainty matches actual correctness

---

## Integration Architecture

### Target State

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED CERTAINTY ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         CertaintyHead                                │    │
│  │  Inputs:                                                             │    │
│  │    - hidden_states [B, seq, d_model]                                │    │
│  │    - surprise [B]                                                    │    │
│  │    - meta_surprise [B]                                               │    │
│  │    - confidence_gate [B, d_model]                                    │    │
│  │    - self_confidence [B] (optional)                                  │    │
│  │    - soma [B, d_soma] (optional)                                     │    │
│  │                                                                      │    │
│  │  Output:                                                             │    │
│  │    - certainty [B] ∈ [0, 1], calibrated                             │    │
│  │    - uncertainty_sources [B, n_sources] (diagnostic)                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         ThinkingLoop                                 │    │
│  │                                                                      │    │
│  │  for tick in range(max_ticks):                                      │    │
│  │      hidden = think_step(hidden, soma, memory)                      │    │
│  │      certainty = certainty_head(hidden, signals)                    │    │
│  │      trajectory.append(certainty)                                   │    │
│  │                                                                      │    │
│  │      if certainty > threshold:                                      │    │
│  │          break  # Confident enough                                  │    │
│  │                                                                      │    │
│  │  if certainty < question_threshold:                                 │    │
│  │      trigger_question()  # Still uncertain after max effort         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    CertaintyDrivenLoss                               │    │
│  │                                                                      │    │
│  │  # CTM-style dual-tick selection                                    │    │
│  │  t_min_loss = argmin(tick_losses)      # Best prediction            │    │
│  │  t_max_certainty = argmax(certainties) # Most confident             │    │
│  │                                                                      │    │
│  │  loss = 0.5 * loss[t_min_loss] + 0.5 * loss[t_max_certainty]       │    │
│  │  calibration = BCE(certainty[t_max], correct[t_max])               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Unified CertaintyHead

**Goal**: Combine existing signals into a single calibrated certainty score.

**Risk**: Low - additive change, existing logic unchanged.

### 1.1 Create certainty.py

**File**: `/home/alexander/Projects/reasonable/certainty.py` (new)

```python
"""
Unified certainty estimation module.

Combines multiple uncertainty signals into a single calibrated certainty score.
Inspired by CTM (Continuous Thought Machines) but adapted for experiential architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class CertaintyOutput:
    """Output from CertaintyHead."""
    certainty: torch.Tensor           # [B] in [0, 1], main output
    uncertainty_sources: torch.Tensor # [B, n_sources] contribution of each source
    raw_signals: Dict[str, torch.Tensor]  # Original signals for debugging


class CertaintyHead(nn.Module):
    """
    Estimates calibrated certainty from multiple uncertainty signals.

    This module unifies:
    - surprise: prediction error
    - meta_surprise: self-calibration error
    - confidence_gate: per-dimension confidence
    - self_confidence: confidence in self-model
    - soma: internal state context

    The output is trained to match actual prediction correctness (calibration).
    """

    def __init__(
        self,
        d_model: int,
        d_soma: int = 64,
        n_sources: int = 5,  # surprise, meta, confidence, self_conf, hidden
        use_soma: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_soma = d_soma
        self.n_sources = n_sources
        self.use_soma = use_soma

        # Source-specific processors
        # Each source gets its own small network to extract certainty contribution

        # 1. Surprise-based certainty
        self.surprise_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 2. Meta-surprise-based certainty
        self.meta_surprise_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 3. Confidence gate aggregation
        self.confidence_gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        # 4. Self-confidence processing
        self.self_confidence_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 5. Hidden state certainty (learned from representation)
        self.hidden_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        # Soma context (optional)
        if use_soma:
            self.soma_context = nn.Sequential(
                nn.Linear(d_soma, 32),
                nn.GELU(),
                nn.Linear(32, n_sources),  # Modulates source weights
            )

        # Final aggregation
        # Learns how to weight different sources
        self.source_weights = nn.Parameter(torch.ones(n_sources) / n_sources)

        # Output calibration
        self.output_calibration = nn.Sequential(
            nn.Linear(n_sources, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, seq_len, d_model] or [B, d_model]
        surprise: Optional[torch.Tensor] = None,        # [B]
        meta_surprise: Optional[torch.Tensor] = None,   # [B]
        confidence_gate: Optional[torch.Tensor] = None, # [B, d_model]
        self_confidence: Optional[torch.Tensor] = None, # [B]
        soma: Optional[torch.Tensor] = None,            # [B, d_soma]
        return_sources: bool = False,
    ) -> CertaintyOutput:
        """
        Compute unified certainty score.

        Args:
            hidden_states: Model hidden states
            surprise: Prediction surprise [0, 1]
            meta_surprise: Self-prediction error [0, 1]
            confidence_gate: Per-dimension confidence [0, 1]
            self_confidence: Confidence in self-model [0, 1]
            soma: Internal state vector
            return_sources: Whether to return per-source contributions

        Returns:
            CertaintyOutput with certainty score and diagnostics
        """
        # Handle sequence input - use last position
        if hidden_states.dim() == 3:
            hidden = hidden_states[:, -1, :]  # [B, d_model]
        else:
            hidden = hidden_states  # [B, d_model]

        B = hidden.shape[0]
        device = hidden.device

        # Compute per-source certainty contributions
        source_contributions = []
        raw_signals = {}

        # 1. Surprise → certainty (inverse relationship)
        if surprise is not None:
            raw_signals['surprise'] = surprise
            surprise_cert = self.surprise_net(surprise.unsqueeze(-1))  # [B, 1]
            # Low surprise = high certainty
            surprise_cert = 1 - torch.sigmoid(surprise_cert)
        else:
            surprise_cert = torch.full((B, 1), 0.5, device=device)
        source_contributions.append(surprise_cert)

        # 2. Meta-surprise → certainty (inverse: good self-knowledge = certain)
        if meta_surprise is not None:
            raw_signals['meta_surprise'] = meta_surprise
            meta_cert = self.meta_surprise_net(meta_surprise.unsqueeze(-1))
            meta_cert = 1 - torch.sigmoid(meta_cert)
        else:
            meta_cert = torch.full((B, 1), 0.5, device=device)
        source_contributions.append(meta_cert)

        # 3. Confidence gate → certainty (aggregate per-dimension confidence)
        if confidence_gate is not None:
            raw_signals['confidence_gate'] = confidence_gate
            conf_cert = self.confidence_gate_net(confidence_gate)  # [B, 1]
            conf_cert = torch.sigmoid(conf_cert)
        else:
            conf_cert = torch.full((B, 1), 0.5, device=device)
        source_contributions.append(conf_cert)

        # 4. Self-confidence → certainty (direct)
        if self_confidence is not None:
            raw_signals['self_confidence'] = self_confidence
            self_cert = self.self_confidence_net(self_confidence.unsqueeze(-1))
            self_cert = torch.sigmoid(self_cert)
        else:
            self_cert = torch.full((B, 1), 0.5, device=device)
        source_contributions.append(self_cert)

        # 5. Hidden state → certainty (learned pattern)
        hidden_cert = self.hidden_net(hidden)  # [B, 1]
        hidden_cert = torch.sigmoid(hidden_cert)
        raw_signals['hidden'] = hidden
        source_contributions.append(hidden_cert)

        # Stack sources: [B, n_sources]
        sources = torch.cat(source_contributions, dim=-1)

        # Apply soma-based context modulation if available
        if self.use_soma and soma is not None:
            raw_signals['soma'] = soma
            soma_modulation = torch.sigmoid(self.soma_context(soma))  # [B, n_sources]
            sources = sources * soma_modulation

        # Weighted aggregation
        weights = F.softmax(self.source_weights, dim=0)  # [n_sources]
        weighted_sources = sources * weights.unsqueeze(0)  # [B, n_sources]

        # Final calibrated output
        certainty = self.output_calibration(weighted_sources).squeeze(-1)  # [B]

        return CertaintyOutput(
            certainty=certainty,
            uncertainty_sources=sources if return_sources else weighted_sources,
            raw_signals=raw_signals if return_sources else {},
        )


def certainty_calibration_loss(
    certainty: torch.Tensor,      # [B] predicted certainty
    was_correct: torch.Tensor,    # [B] binary correctness
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Calibration loss: certainty should match actual correctness.

    Uses BCE so that:
    - High certainty + correct → low loss (appropriate confidence)
    - High certainty + wrong → high loss (overconfidence penalty)
    - Low certainty + wrong → low loss (honest uncertainty)
    - Low certainty + correct → medium loss (underconfidence)
    """
    return F.binary_cross_entropy(
        certainty,
        was_correct.float(),
        reduction=reduction,
    )


def expected_calibration_error(
    certainty: torch.Tensor,      # [N] predicted certainties
    was_correct: torch.Tensor,    # [N] binary correctness
    n_bins: int = 10,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures how well-calibrated the certainty estimates are:
    - Bucket samples by certainty
    - Compare mean certainty vs mean accuracy per bucket
    - Weighted average of |certainty - accuracy| per bucket

    Returns:
        ece: Scalar ECE value
        details: Per-bucket statistics for visualization
    """
    # Flatten
    certainty = certainty.flatten()
    was_correct = was_correct.flatten().float()

    # Create bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=certainty.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1, device=certainty.device)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (certainty > bin_lower) & (certainty <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if in_bin.sum() > 0:
            accuracy_in_bin = was_correct[in_bin].mean()
            confidence_in_bin = certainty[in_bin].mean()

            ece += prop_in_bin * torch.abs(accuracy_in_bin - confidence_in_bin)

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(torch.tensor(0.0, device=certainty.device))
            bin_confidences.append(torch.tensor(0.0, device=certainty.device))
            bin_counts.append(torch.tensor(0, device=certainty.device))

    return ece, {
        'bin_accuracies': torch.stack(bin_accuracies),
        'bin_confidences': torch.stack(bin_confidences),
        'bin_counts': torch.stack(bin_counts),
        'bin_boundaries': bin_boundaries,
    }
```

### 1.2 Integrate into ExperientialStream

**File**: `experiential.py`

**Modify `__init__`** (around line 1200):

```python
# Add import at top
from certainty import CertaintyHead, CertaintyOutput

# In __init__, after self.self_modulator:
self.certainty_head = CertaintyHead(
    d_model=d_model,
    d_soma=d_soma if hasattr(self, 'd_soma') else 64,
    use_soma=True,
) if use_certainty_head else None
```

**Modify `forward`** (after line 1882, after self-modulation):

```python
# Compute unified certainty
certainty_output = None
if self.certainty_head is not None:
    certainty_output = self.certainty_head(
        hidden_states=h_end,
        surprise=chunk_surprise,
        meta_surprise=meta_surprise,
        confidence_gate=confidence_gate,
        self_confidence=None,  # Will be added when SelfState is integrated
        soma=None,  # Will be passed when available
    )
```

**Add to return dict** (around line 1940):

```python
# In the return dictionary
'certainty': certainty_output.certainty if certainty_output else None,
'certainty_sources': certainty_output.uncertainty_sources if certainty_output else None,
```

### 1.3 Add Calibration Loss

**File**: `experiential.py`

**Modify `combined_experiential_loss`** (after line 2206):

```python
# Add certainty calibration loss
if 'certainty' in exp_output and exp_output['certainty'] is not None:
    # Compute correctness from language modeling
    # was_correct = whether top prediction matched target
    if targets is not None:
        predictions = logits.argmax(dim=-1)  # [B, seq_len]
        # Use mean correctness over sequence as target
        was_correct = (predictions[:, :-1] == targets[:, 1:]).float().mean(dim=-1)  # [B]

        from certainty import certainty_calibration_loss
        cert_loss = certainty_calibration_loss(
            exp_output['certainty'],
            was_correct,
        )
        loss_dict['certainty_calibration'] = cert_loss
        total_loss = total_loss + certainty_calibration_weight * cert_loss
```

**Add config parameter**:

```python
# In combined_experiential_loss signature
certainty_calibration_weight: float = 0.1,
```

---

## Phase 2: Certainty in Soma Integration

**Goal**: Make soma aware of calibrated certainty, use it to modulate integration.

**Risk**: Low-Medium - modifies soma behavior.

### 2.1 Update SelfState Signal Extraction

**File**: `self_state.py`

**Modify `extract_signals_from_experiential`** (around line 745):

```python
def extract_signals_from_experiential(
    self,
    exp_output: Dict[str, Any],
    hidden_states: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Extract signals for soma integration."""

    surprise = exp_output.get('surprise', torch.zeros(B))
    meta_surprise = exp_output.get('meta_surprise', torch.zeros(B))
    confidence_gate = exp_output.get('confidence_gate', None)

    # OLD: Simple certainty
    # certainty = 1.0 - torch.clamp(surprise, 0, 1)

    # NEW: Use unified certainty if available
    if 'certainty' in exp_output and exp_output['certainty'] is not None:
        certainty = exp_output['certainty']
    else:
        # Fallback: enhanced combination
        base_certainty = 1.0 - torch.clamp(surprise, 0, 1)
        meta_certainty = 1.0 - torch.clamp(meta_surprise, 0, 1)

        if confidence_gate is not None:
            conf_certainty = confidence_gate.mean(dim=-1)
            certainty = 0.4 * base_certainty + 0.4 * conf_certainty + 0.2 * meta_certainty
        else:
            certainty = 0.6 * base_certainty + 0.4 * meta_certainty

    return {
        'surprise': surprise,
        'arousal': exp_output.get('arousal', torch.zeros(B)),
        'valence': exp_output.get('valence', torch.zeros(B)),
        'novelty': exp_output.get('novelty', torch.zeros(B)),
        'certainty': certainty,  # Now using unified certainty
        'engagement': self._compute_engagement(hidden_states),
    }
```

### 2.2 Certainty-Modulated Soma Updates

**File**: `self_state.py`

**Modify `SomaIntegrator.forward`** (around line 175):

```python
def forward(
    self,
    prev_soma: torch.Tensor,
    signals: Dict[str, torch.Tensor],
    certainty: Optional[torch.Tensor] = None,  # NEW parameter
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Integrate signals into soma with optional certainty modulation.

    When uncertain, be more conservative in state updates.
    """
    # ... existing signal gating logic ...

    # Compute weighted signal contribution
    signal_contribution = self.signal_proj(weighted_signals)

    # NEW: Certainty-modulated integration
    if certainty is not None:
        # When uncertain, reduce update magnitude
        # certainty_gate ∈ [0.3, 1.0] - never fully stop updates
        certainty_gate = 0.3 + 0.7 * certainty.unsqueeze(-1)  # [B, 1]
        signal_contribution = signal_contribution * certainty_gate

    # Integrate with decay
    new_soma = self.decay * prev_soma + (1 - self.decay) * signal_contribution

    # ... rest of method ...
```

### 2.3 Certainty-Modulated Feedback

**File**: `self_state.py`

**Modify `SomaFeedback.forward`** (around line 600):

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    soma: torch.Tensor,
    certainty: Optional[torch.Tensor] = None,  # NEW parameter
) -> torch.Tensor:
    """
    Apply soma feedback to hidden states.

    When uncertain, reduce soma's influence on processing.
    """
    # Compute gate and add
    gate = self.hidden_gate(soma).unsqueeze(1)  # [B, 1, d_model]
    add = self.hidden_add(soma).unsqueeze(1)

    # NEW: Certainty-scaled feedback
    if certainty is not None:
        # When uncertain, let hidden states flow more freely
        # feedback_strength ∈ [0.2, 1.0]
        feedback_strength = 0.2 + 0.8 * certainty.unsqueeze(-1).unsqueeze(-1)

        # Blend gate toward 1 (no gating) when uncertain
        gate = 1 - feedback_strength * (1 - gate)
        # Reduce additive bias when uncertain
        add = add * feedback_strength

    modulated = hidden_states * gate + 0.1 * add
    return modulated
```

---

## Phase 3: ThinkingLoop with Adaptive Computation

**Goal**: Add ability to iterate when uncertain, halt when confident.

**Risk**: Medium - new component, affects inference flow.

### 3.1 Create thinking.py

**File**: `/home/alexander/Projects/reasonable/thinking.py` (new)

```python
"""
Thinking loop module for adaptive computation.

Allows the model to iterate on its hidden state when uncertain,
halting when certainty exceeds threshold or max ticks reached.

Inspired by CTM (Continuous Thought Machines).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from certainty import CertaintyHead, CertaintyOutput


@dataclass
class ThinkingOutput:
    """Output from ThinkingLoop."""
    hidden: torch.Tensor              # [B, seq_len, d_model] final hidden state
    certainty: torch.Tensor           # [B] final certainty
    certainty_trajectory: torch.Tensor  # [B, ticks_used] certainty at each tick
    hidden_trajectory: Optional[List[torch.Tensor]]  # Optional hidden states at each tick
    ticks_used: int                   # How many ticks were used
    halted_early: torch.Tensor        # [B] bool, whether each sample halted early
    needs_question: torch.Tensor      # [B] bool, whether certainty still low


class ThinkStep(nn.Module):
    """
    Single step of iterative refinement.

    Each step:
    1. Self-attention to refine representation
    2. Soma modulation to incorporate internal state
    3. Memory integration (optional)
    4. FFN for further processing
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_soma: int = 64,
        d_memory: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_memory = d_memory or d_model

        # Self-attention for iterative refinement
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Soma modulation
        self.soma_gate = nn.Sequential(
            nn.Linear(d_soma, d_model),
            nn.Sigmoid(),
        )
        self.soma_add = nn.Sequential(
            nn.Linear(d_soma, d_model),
            nn.Tanh(),
        )

        # Optional memory integration
        if d_memory:
            self.memory_attn = nn.MultiheadAttention(
                d_model, n_heads,
                kdim=d_memory,
                vdim=d_memory,
                dropout=dropout,
                batch_first=True,
            )
            self.memory_gate = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        if d_memory:
            self.norm_mem = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden: torch.Tensor,           # [B, seq_len, d_model]
        soma: torch.Tensor,             # [B, d_soma]
        memory: Optional[torch.Tensor] = None,  # [B, mem_len, d_memory]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """One thinking step."""

        step_info = {}

        # 1. Self-attention
        residual = hidden
        hidden_norm = self.norm1(hidden)
        hidden_attn, attn_weights = self.self_attn(
            hidden_norm, hidden_norm, hidden_norm,
            key_padding_mask=attention_mask,
        )
        hidden = residual + hidden_attn
        step_info['self_attn_weights'] = attn_weights

        # 2. Soma modulation
        soma_gate = self.soma_gate(soma).unsqueeze(1)  # [B, 1, d_model]
        soma_add = self.soma_add(soma).unsqueeze(1)
        hidden = hidden * soma_gate + 0.1 * soma_add

        # 3. Memory integration (optional)
        if memory is not None and hasattr(self, 'memory_attn'):
            residual = hidden
            hidden_norm = self.norm_mem(hidden)
            mem_attn, mem_weights = self.memory_attn(
                hidden_norm, memory, memory,
            )
            # Gated memory integration
            mem_gate = self.memory_gate(hidden_norm)  # [B, seq_len, 1]
            hidden = residual + mem_gate * mem_attn
            step_info['memory_attn_weights'] = mem_weights
            step_info['memory_gate'] = mem_gate

        # 4. FFN
        residual = hidden
        hidden = self.norm2(hidden)
        hidden = residual + self.ffn(hidden)

        # Final norm
        hidden = self.norm3(hidden)

        return hidden, step_info


class ThinkingLoop(nn.Module):
    """
    Iterative thinking with certainty-driven halting.

    The loop continues until:
    - Certainty exceeds threshold (confident), OR
    - Maximum ticks reached (effort exhausted)

    If certainty is still low after max ticks, flags for question generation.
    """

    def __init__(
        self,
        d_model: int,
        d_soma: int = 64,
        d_memory: int = None,
        max_ticks: int = 5,
        certainty_threshold: float = 0.8,
        question_threshold: float = 0.3,
        n_heads: int = 8,
        share_weights: bool = True,  # Share ThinkStep across ticks
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_ticks = max_ticks
        self.certainty_threshold = certainty_threshold
        self.question_threshold = question_threshold

        # Thinking steps
        if share_weights:
            # Single shared step (more parameter efficient)
            self.think_steps = nn.ModuleList([
                ThinkStep(d_model, n_heads, d_soma, d_memory, dropout)
            ])
            self.shared = True
        else:
            # Separate step per tick (more expressive)
            self.think_steps = nn.ModuleList([
                ThinkStep(d_model, n_heads, d_soma, d_memory, dropout)
                for _ in range(max_ticks)
            ])
            self.shared = False

        # Per-tick certainty estimation
        self.certainty_head = CertaintyHead(
            d_model=d_model,
            d_soma=d_soma,
            use_soma=True,
        )

        # Tick embedding (helps model know which iteration it's on)
        self.tick_embedding = nn.Embedding(max_ticks, d_model)

    def forward(
        self,
        hidden: torch.Tensor,           # [B, seq_len, d_model]
        soma: torch.Tensor,             # [B, d_soma]
        memory: Optional[torch.Tensor] = None,
        surprise: Optional[torch.Tensor] = None,
        meta_surprise: Optional[torch.Tensor] = None,
        confidence_gate: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        force_max_ticks: bool = False,  # For training, run all ticks
    ) -> ThinkingOutput:
        """
        Think until certain or max ticks.

        Args:
            hidden: Initial hidden states
            soma: Internal state
            memory: Optional retrieved memories
            surprise, meta_surprise, confidence_gate: Signals for certainty
            attention_mask: Padding mask
            return_trajectory: Store all hidden states (memory intensive)
            force_max_ticks: Always run max ticks (for training)

        Returns:
            ThinkingOutput with final state and trajectory
        """
        B, seq_len, _ = hidden.shape
        device = hidden.device

        certainty_trajectory = []
        hidden_trajectory = [] if return_trajectory else None
        step_infos = []

        # Track which samples have halted
        halted = torch.zeros(B, dtype=torch.bool, device=device)

        actual_ticks = 0

        for tick in range(self.max_ticks):
            actual_ticks = tick + 1

            # Get think step (shared or per-tick)
            think_step = self.think_steps[0] if self.shared else self.think_steps[tick]

            # Add tick embedding
            tick_emb = self.tick_embedding(
                torch.tensor([tick], device=device)
            ).unsqueeze(0).expand(B, seq_len, -1)
            hidden_with_tick = hidden + 0.1 * tick_emb

            # One thinking step
            hidden_new, step_info = think_step(
                hidden_with_tick, soma, memory, attention_mask
            )
            step_infos.append(step_info)

            # Only update non-halted samples
            if not force_max_ticks and halted.any():
                hidden = torch.where(
                    halted.unsqueeze(-1).unsqueeze(-1).expand_as(hidden),
                    hidden,  # Keep old for halted
                    hidden_new,  # Update for active
                )
            else:
                hidden = hidden_new

            # Compute certainty
            cert_output = self.certainty_head(
                hidden_states=hidden,
                surprise=surprise,
                meta_surprise=meta_surprise,
                confidence_gate=confidence_gate,
                soma=soma,
            )
            certainty = cert_output.certainty
            certainty_trajectory.append(certainty)

            if return_trajectory:
                hidden_trajectory.append(hidden.clone())

            # Check halting condition
            newly_halted = certainty > self.certainty_threshold
            halted = halted | newly_halted

            # Early exit if all halted (and not forcing max ticks)
            if not force_max_ticks and halted.all():
                break

        # Stack trajectory: [B, ticks_used]
        certainty_trajectory = torch.stack(certainty_trajectory, dim=1)
        final_certainty = certainty_trajectory[:, -1]

        # Flag samples that need questions
        needs_question = final_certainty < self.question_threshold

        return ThinkingOutput(
            hidden=hidden,
            certainty=final_certainty,
            certainty_trajectory=certainty_trajectory,
            hidden_trajectory=hidden_trajectory,
            ticks_used=actual_ticks,
            halted_early=halted & (actual_ticks < self.max_ticks),
            needs_question=needs_question,
        )


class CertaintyDrivenLoss(nn.Module):
    """
    CTM-inspired loss with dual-tick selection.

    For each sample, selects two ticks:
    - t_min_loss: tick with minimum task loss (best answer)
    - t_max_certainty: tick with maximum certainty (most confident)

    This trains the model to:
    1. Find correct answers at some tick (task performance)
    2. Be confident when correct (calibration)
    3. Use fewer ticks when possible (efficiency)
    """

    def __init__(
        self,
        task_weight: float = 0.5,
        certainty_weight: float = 0.5,
        calibration_weight: float = 0.1,
        efficiency_weight: float = 0.01,  # Encourage early halting
    ):
        super().__init__()

        self.task_weight = task_weight
        self.certainty_weight = certainty_weight
        self.calibration_weight = calibration_weight
        self.efficiency_weight = efficiency_weight

    def forward(
        self,
        tick_losses: torch.Tensor,        # [B, T] task loss at each tick
        tick_certainties: torch.Tensor,   # [B, T] certainty at each tick
        tick_correct: torch.Tensor,       # [B, T] correctness at each tick
    ) -> Dict[str, torch.Tensor]:
        """
        Compute dual-tick loss.

        Args:
            tick_losses: Task loss (e.g., CE) at each tick
            tick_certainties: Certainty scores at each tick
            tick_correct: Whether prediction was correct at each tick

        Returns:
            Dict with loss components and selected ticks
        """
        B, T = tick_losses.shape
        device = tick_losses.device

        # Select ticks
        t_min_loss = tick_losses.argmin(dim=1)           # [B]
        t_max_certainty = tick_certainties.argmax(dim=1) # [B]

        # Gather values at selected ticks
        batch_idx = torch.arange(B, device=device)

        loss_at_min = tick_losses[batch_idx, t_min_loss]
        loss_at_max_cert = tick_losses[batch_idx, t_max_certainty]

        cert_at_max = tick_certainties[batch_idx, t_max_certainty]
        correct_at_max = tick_correct[batch_idx, t_max_certainty]

        # Task loss: optimize at both selected ticks
        task_loss = (
            self.task_weight * loss_at_min +
            self.certainty_weight * loss_at_max_cert
        )

        # Calibration loss: certainty should match correctness at max-certainty tick
        calibration_loss = F.binary_cross_entropy(
            cert_at_max,
            correct_at_max.float(),
        )

        # Efficiency loss: encourage earlier ticks (optional)
        # Penalize using later ticks when earlier ones are good
        tick_indices = torch.arange(T, device=device, dtype=torch.float)
        selected_tick_mean = (t_min_loss.float() + t_max_certainty.float()) / 2
        efficiency_loss = selected_tick_mean.mean() / T  # Normalized [0, 1]

        # Combined loss
        total_loss = (
            task_loss.mean() +
            self.calibration_weight * calibration_loss +
            self.efficiency_weight * efficiency_loss
        )

        return {
            'loss': total_loss,
            'task_loss': task_loss.mean(),
            'calibration_loss': calibration_loss,
            'efficiency_loss': efficiency_loss,
            't_min_loss': t_min_loss,
            't_max_certainty': t_max_certainty,
            'tick_divergence': (t_min_loss != t_max_certainty).float().mean(),
        }
```

### 3.2 Integrate ThinkingLoop into MemoryAugmentedGPT

**File**: `experiential.py`

**Modify `__init__`** (around line 2450):

```python
# Add import
from thinking import ThinkingLoop, ThinkingOutput, CertaintyDrivenLoss

# In __init__
self.thinking_loop = ThinkingLoop(
    d_model=d_model,
    d_soma=d_soma,
    max_ticks=thinking_max_ticks,
    certainty_threshold=thinking_certainty_threshold,
    question_threshold=question_certainty_threshold,
    n_heads=n_head,
) if use_thinking_loop else None

self.certainty_driven_loss = CertaintyDrivenLoss() if use_thinking_loop else None
```

**Modify `forward`** (after experiential and self-state processing, around line 3200):

```python
# Apply thinking loop if enabled
thinking_output = None
if self.thinking_loop is not None:
    thinking_output = self.thinking_loop(
        hidden=hidden_states,
        soma=self_state_output.get('soma', torch.zeros(B, self.d_soma, device=device)),
        memory=memory_output.get('retrieved_memory'),
        surprise=exp_output.get('surprise'),
        meta_surprise=exp_output.get('meta_surprise'),
        confidence_gate=exp_output.get('confidence_gate'),
        force_max_ticks=self.training,  # Run all ticks during training
    )
    hidden_states = thinking_output.hidden
    memory_output['thinking'] = {
        'certainty': thinking_output.certainty,
        'certainty_trajectory': thinking_output.certainty_trajectory,
        'ticks_used': thinking_output.ticks_used,
        'needs_question': thinking_output.needs_question,
    }
```

### 3.3 Training with Dual-Tick Loss

**File**: `experiential.py`

**Modify `memory_augmented_loss`** (around line 3850):

```python
# Add thinking loop loss
if 'thinking' in memory_output and self.certainty_driven_loss is not None:
    # Need to compute task loss at each tick
    # This requires running the output head at each tick's hidden state

    trajectory = memory_output['thinking'].get('hidden_trajectory')
    if trajectory is not None:
        tick_losses = []
        tick_correct = []

        for tick_hidden in trajectory:
            tick_logits = self.lm_head(tick_hidden)
            tick_loss = F.cross_entropy(
                tick_logits[:, :-1].reshape(-1, vocab_size),
                targets[:, 1:].reshape(-1),
                reduction='none',
            ).view(B, -1).mean(dim=-1)  # [B]
            tick_losses.append(tick_loss)

            tick_pred = tick_logits[:, :-1].argmax(dim=-1)
            tick_acc = (tick_pred == targets[:, 1:]).float().mean(dim=-1)  # [B]
            tick_correct.append(tick_acc)

        tick_losses = torch.stack(tick_losses, dim=1)  # [B, T]
        tick_correct = torch.stack(tick_correct, dim=1)  # [B, T]

        thinking_loss_dict = self.certainty_driven_loss(
            tick_losses=tick_losses,
            tick_certainties=memory_output['thinking']['certainty_trajectory'],
            tick_correct=tick_correct,
        )

        loss_dict['thinking_loss'] = thinking_loss_dict['loss']
        loss_dict['tick_divergence'] = thinking_loss_dict['tick_divergence']
        total_loss = total_loss + thinking_weight * thinking_loss_dict['loss']
```

---

## Phase 4: Certainty-Driven Memory Crystallization

**Goal**: Use certainty to modulate what gets stored in memory.

**Risk**: Low - enhances existing mechanism.

### 4.1 Certainty-Weighted Salience

**File**: `experiential.py`

**Modify salience computation** (around line 1840):

```python
# Current: salience = surprise * (1 + meta_surprise_weight * meta_surprise)

# NEW: Add certainty component
# High certainty + high surprise = confident novel insight → crystallize
# Low certainty + high surprise = confused → maybe ask, don't crystallize
# High certainty + low surprise = routine → don't crystallize
# Low certainty + low surprise = bored but uncertain → don't crystallize

if certainty is not None:
    # Certainty gates surprise for crystallization
    # Only crystallize surprises we're certain about
    certainty_factor = 0.3 + 0.7 * certainty  # [0.3, 1.0]
    salience = surprise * certainty_factor * (1 + meta_surprise_weight * meta_surprise)
else:
    salience = surprise * (1 + meta_surprise_weight * meta_surprise)
```

**Rationale**: Uncertain surprises might be noise or confusion. Certain surprises are worth remembering.

---

## Phase 5: Question Triggering

**Goal**: Generate questions when certainty remains low after thinking.

**Risk**: Medium - new behavior.

### 5.1 Create question.py

**File**: `/home/alexander/Projects/reasonable/question.py` (new)

```python
"""
Question generation from persistent uncertainty.

When the model remains uncertain after maximum thinking effort,
it should formulate a question to resolve the uncertainty.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class QuestionType(Enum):
    """Types of questions based on uncertainty source."""
    CONTENT = "content"           # What does this mean?
    CLARIFICATION = "clarification"  # Can you clarify X?
    CONFIRMATION = "confirmation"    # Is X correct?
    ELABORATION = "elaboration"      # Tell me more about X
    FACTUAL = "factual"             # What is X?
    CAUSAL = "causal"               # Why does X happen?


@dataclass
class Question:
    """A generated question."""
    question_type: QuestionType
    hidden_state: torch.Tensor    # [d_model] state that triggered question
    certainty: float              # How uncertain we were
    uncertainty_sources: Dict[str, float]  # Which sources contributed
    focus_region: Optional[torch.Tensor] = None  # Attention over input


class QuestionGenerator(nn.Module):
    """
    Generates question metadata from uncertain states.

    This module identifies:
    1. What type of question to ask
    2. What to focus the question on
    3. Priority/urgency of the question

    Full question text generation would require a decoder.
    This provides the semantic content for question generation.
    """

    def __init__(
        self,
        d_model: int,
        n_question_types: int = 6,
    ):
        super().__init__()

        self.d_model = d_model

        # Question type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_question_types),
        )

        # Focus attention (what part of input to ask about)
        self.focus_query = nn.Linear(d_model, d_model)

        # Priority estimator
        self.priority = nn.Sequential(
            nn.Linear(d_model + 1, 64),  # +1 for certainty
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,      # [B, seq_len, d_model]
        certainty: torch.Tensor,          # [B]
        uncertainty_sources: torch.Tensor, # [B, n_sources]
        input_hidden: Optional[torch.Tensor] = None,  # [B, input_len, d_model]
    ) -> List[Question]:
        """
        Generate questions for uncertain samples.

        Args:
            hidden_states: Current model state
            certainty: Certainty scores
            uncertainty_sources: Per-source uncertainty
            input_hidden: Input representations for focus attention

        Returns:
            List of Question objects for samples with low certainty
        """
        B = hidden_states.shape[0]
        device = hidden_states.device

        # Use last position for question generation
        h = hidden_states[:, -1, :]  # [B, d_model]

        # Classify question type
        type_logits = self.type_classifier(h)  # [B, n_types]
        type_probs = torch.softmax(type_logits, dim=-1)
        predicted_types = type_logits.argmax(dim=-1)  # [B]

        # Compute focus attention if input available
        focus_regions = None
        if input_hidden is not None:
            query = self.focus_query(h).unsqueeze(1)  # [B, 1, d_model]
            focus_scores = torch.bmm(query, input_hidden.transpose(1, 2))  # [B, 1, input_len]
            focus_regions = torch.softmax(focus_scores.squeeze(1), dim=-1)  # [B, input_len]

        # Compute priority
        priority_input = torch.cat([h, certainty.unsqueeze(-1)], dim=-1)
        priorities = self.priority(priority_input).squeeze(-1)  # [B]

        # Build question objects
        questions = []
        question_types = list(QuestionType)

        for b in range(B):
            q = Question(
                question_type=question_types[predicted_types[b].item()],
                hidden_state=h[b],
                certainty=certainty[b].item(),
                uncertainty_sources={
                    f'source_{i}': uncertainty_sources[b, i].item()
                    for i in range(uncertainty_sources.shape[1])
                },
                focus_region=focus_regions[b] if focus_regions is not None else None,
            )
            questions.append(q)

        return questions


class QuestionBuffer:
    """
    Buffers questions until they can be addressed.

    Questions accumulate during forward passes and are
    processed when:
    - User input provides an answer
    - Model reasons through to an answer
    - Question expires (too old to be relevant)
    """

    def __init__(self, max_size: int = 10, max_age: int = 100):
        self.questions: List[Question] = []
        self.ages: List[int] = []
        self.max_size = max_size
        self.max_age = max_age

    def add(self, question: Question):
        """Add a question to the buffer."""
        if len(self.questions) >= self.max_size:
            # Remove oldest
            self.questions.pop(0)
            self.ages.pop(0)

        self.questions.append(question)
        self.ages.append(0)

    def step(self):
        """Age all questions by one step."""
        self.ages = [a + 1 for a in self.ages]

        # Remove expired questions
        valid = [(q, a) for q, a in zip(self.questions, self.ages) if a < self.max_age]
        if valid:
            self.questions, self.ages = zip(*valid)
            self.questions = list(self.questions)
            self.ages = list(self.ages)
        else:
            self.questions = []
            self.ages = []

    def get_pending(self) -> List[Question]:
        """Get all pending questions sorted by priority."""
        return sorted(
            self.questions,
            key=lambda q: (1 - q.certainty),  # Most uncertain first
            reverse=True,
        )

    def resolve(self, question: Question):
        """Mark a question as resolved."""
        if question in self.questions:
            idx = self.questions.index(question)
            self.questions.pop(idx)
            self.ages.pop(idx)
```

### 5.2 Integrate Question Generation

**File**: `experiential.py`

**Modify `__init__`** (around line 2460):

```python
from question import QuestionGenerator, QuestionBuffer

self.question_generator = QuestionGenerator(d_model) if use_questions else None
self.question_buffer = QuestionBuffer() if use_questions else None
```

**Modify `forward`** (after thinking loop, around line 3220):

```python
# Generate questions for uncertain samples
if self.question_generator is not None and thinking_output is not None:
    if thinking_output.needs_question.any():
        # Only generate for samples that need questions
        uncertain_mask = thinking_output.needs_question
        uncertain_hidden = hidden_states[uncertain_mask]
        uncertain_certainty = thinking_output.certainty[uncertain_mask]
        uncertain_sources = memory_output['thinking'].get('certainty_sources',
            torch.zeros(uncertain_mask.sum(), 5, device=device))

        questions = self.question_generator(
            hidden_states=uncertain_hidden.unsqueeze(1),  # Add seq dim
            certainty=uncertain_certainty,
            uncertainty_sources=uncertain_sources,
            input_hidden=hidden_states[uncertain_mask],  # Use current hidden as focus
        )

        # Add to buffer
        for q in questions:
            self.question_buffer.add(q)

        memory_output['questions'] = questions
        memory_output['n_questions_pending'] = len(self.question_buffer.get_pending())
```

---

## Phase 6: Configuration and Training Integration

### 6.1 Configuration Parameters

**File**: `experiential.py`

Add to `MemoryAugmentedGPT.__init__`:

```python
# Certainty configuration
use_certainty_head: bool = True,
certainty_calibration_weight: float = 0.1,

# Soma-certainty integration
certainty_modulates_soma: bool = True,
certainty_modulates_feedback: bool = True,

# Thinking loop configuration
use_thinking_loop: bool = False,  # Off by default (experimental)
thinking_max_ticks: int = 5,
thinking_certainty_threshold: float = 0.8,
thinking_weight: float = 0.1,

# Question configuration
use_questions: bool = False,  # Off by default
question_certainty_threshold: float = 0.3,
```

### 6.2 Training Script Updates

**File**: `train_memory_augmented.py`

Add logging for certainty metrics:

```python
# In training loop, after loss computation
if 'certainty_calibration' in loss_dict:
    wandb.log({
        'certainty/calibration_loss': loss_dict['certainty_calibration'].item(),
        'certainty/mean': memory_output.get('certainty', torch.tensor(0.5)).mean().item(),
    })

if 'thinking' in memory_output:
    wandb.log({
        'thinking/ticks_used': memory_output['thinking']['ticks_used'],
        'thinking/final_certainty': memory_output['thinking']['certainty'].mean().item(),
        'thinking/needs_question_pct': memory_output['thinking']['needs_question'].float().mean().item(),
    })

if 'tick_divergence' in loss_dict:
    wandb.log({
        'thinking/tick_divergence': loss_dict['tick_divergence'].item(),
    })
```

---

## Implementation Order

```
Week 1: Foundation
├── Create certainty.py with CertaintyHead
├── Unit tests for CertaintyHead
├── Integrate into ExperientialStream
└── Add calibration loss

Week 2: Soma Integration
├── Update signal extraction to use unified certainty
├── Add certainty modulation to SomaIntegrator
├── Add certainty modulation to SomaFeedback
└── Integration tests

Week 3: Thinking Loop
├── Create thinking.py with ThinkStep, ThinkingLoop
├── Unit tests for thinking components
├── Integrate into MemoryAugmentedGPT
├── Add CertaintyDrivenLoss
└── Training with thinking loop

Week 4: Questions and Polish
├── Create question.py
├── Integrate question generation
├── Certainty-weighted memory crystallization
├── End-to-end testing
└── Performance optimization
```

---

## Testing Strategy

### Unit Tests

```python
# test_certainty.py
def test_certainty_head_output_range():
    """Certainty should be in [0, 1]."""
    head = CertaintyHead(d_model=512)
    hidden = torch.randn(4, 128, 512)
    output = head(hidden)
    assert (output.certainty >= 0).all()
    assert (output.certainty <= 1).all()

def test_certainty_responds_to_surprise():
    """Higher surprise should reduce certainty."""
    head = CertaintyHead(d_model=512)
    hidden = torch.randn(4, 512)

    cert_low_surprise = head(hidden, surprise=torch.zeros(4))
    cert_high_surprise = head(hidden, surprise=torch.ones(4))

    assert cert_low_surprise.certainty.mean() > cert_high_surprise.certainty.mean()

# test_thinking.py
def test_thinking_loop_halts_when_certain():
    """Should halt before max_ticks when certain."""
    loop = ThinkingLoop(d_model=512, max_ticks=10, certainty_threshold=0.5)
    hidden = torch.randn(4, 128, 512)
    soma = torch.randn(4, 64)

    # Mock certainty head to return high certainty
    loop.certainty_head = lambda **kwargs: CertaintyOutput(
        certainty=torch.ones(4) * 0.9,
        uncertainty_sources=torch.zeros(4, 5),
        raw_signals={},
    )

    output = loop(hidden, soma)
    assert output.ticks_used < 10
    assert output.halted_early.all()

def test_dual_tick_loss():
    """Dual-tick loss should select different ticks."""
    loss_fn = CertaintyDrivenLoss()

    # Create scenario where best loss and best certainty are at different ticks
    tick_losses = torch.tensor([[0.5, 0.3, 0.4, 0.6]])  # Min at tick 1
    tick_certainties = torch.tensor([[0.3, 0.5, 0.8, 0.6]])  # Max at tick 2
    tick_correct = torch.tensor([[0.0, 1.0, 1.0, 0.0]])

    output = loss_fn(tick_losses, tick_certainties, tick_correct)

    assert output['t_min_loss'].item() == 1
    assert output['t_max_certainty'].item() == 2
    assert output['tick_divergence'].item() == 1.0
```

### Integration Tests

```python
def test_full_forward_with_certainty():
    """End-to-end test with all certainty components."""
    model = MemoryAugmentedGPT(
        use_certainty_head=True,
        use_thinking_loop=True,
        thinking_max_ticks=3,
    )

    input_ids = torch.randint(0, 1000, (2, 64))
    output = model(input_ids)

    assert 'certainty' in output
    assert 'thinking' in output
    assert output['thinking']['ticks_used'] <= 3
```

### Metrics to Track

1. **Calibration curve**: Plot certainty vs actual accuracy per bin
2. **ECE (Expected Calibration Error)**: Should decrease during training
3. **Thinking efficiency**: Ticks used vs task difficulty
4. **Tick divergence**: How often t_min_loss ≠ t_max_certainty
5. **Question rate**: % of samples with certainty < threshold

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Training instability | Medium | High | Start with low weights, gradual increase |
| Thinking loop latency | High | Medium | Keep max_ticks low (3-5), disable for inference |
| Certainty collapse (always 0.5) | Medium | Medium | Diversity loss, temperature scaling |
| Questions too frequent | Medium | Low | Tune threshold, add cooldown |
| Memory overhead from trajectories | Medium | Medium | Only store during training, subsample |

---

## Success Criteria

### Phase 1: CertaintyHead
- [ ] Certainty in [0, 1] for all inputs
- [ ] Higher surprise → lower certainty
- [ ] Calibration loss < 0.2 after training
- [ ] ECE < 0.1

### Phase 2: Soma Integration
- [ ] Soma updates scaled by certainty
- [ ] Hidden state feedback scaled by certainty
- [ ] No regression on language modeling loss

### Phase 3: ThinkingLoop
- [ ] Certainty increases with ticks (on average)
- [ ] Easy inputs halt early (< max_ticks)
- [ ] Hard inputs use more ticks
- [ ] Dual-tick loss converges

### Phase 4: Memory Crystallization
- [ ] High-certainty surprises stored more often
- [ ] Low-certainty surprises filtered

### Phase 5: Questions
- [ ] Questions generated for low-certainty samples
- [ ] Question types correlate with uncertainty sources
- [ ] Question buffer manages pending questions

---

## Files to Create/Modify

### New Files
1. `certainty.py` - CertaintyHead, calibration loss, ECE
2. `thinking.py` - ThinkStep, ThinkingLoop, CertaintyDrivenLoss
3. `question.py` - QuestionGenerator, QuestionBuffer

### Modified Files
1. `experiential.py`
   - Import new modules
   - Add CertaintyHead to ExperientialStream
   - Add ThinkingLoop to MemoryAugmentedGPT
   - Modify salience computation
   - Add question generation
   - Extend loss functions

2. `self_state.py`
   - Modify extract_signals_from_experiential
   - Add certainty modulation to SomaIntegrator
   - Add certainty modulation to SomaFeedback

3. `train_memory_augmented.py`
   - Add certainty metrics logging
   - Add thinking metrics logging
   - Add configuration flags

### Test Files
1. `test_certainty.py`
2. `test_thinking.py`
3. `test_questions.py`
