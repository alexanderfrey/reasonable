# Soma: Internal State System

## Architecture Status

### Core Components
- [x] **SomaIntegrator** — Accumulates signals into persistent state with decay
- [x] **SelfModel** — Predicts own soma changes, computes self-surprise
- [x] **SelfDescriptionHead** — Projects soma → text (lossy inspection)
- [x] **SelfInterpretationHead** — Reads text → soma update (narrative loop)
- [x] **IdealSelf** — Actual vs ideal discrepancy with regulation signal
- [x] **SomaFeedback** — Modulates hidden states based on soma
- [x] **SelfState** — Complete system combining all components

### Integration
- [x] MemoryAugmentedGPT integration (`use_self_state=True`)
- [x] Self-state losses in `memory_augmented_loss()`
- [x] Soma outputs in `memory_output` dict
- [x] Hidden state gating by soma
- [x] Attention Q bias from soma (`self_state_to_attention=True`)
- [ ] Full narrative loop during forward pass (currently manual)
- [ ] Logit bias from soma (implemented but not wired)

### Training
- [x] Self-model loss (minimize self-surprise)
- [x] Ideal discrepancy loss (move toward ideal self)
- [x] Regulation alignment loss (delta follows regulation signal)
- [ ] Curriculum for soma → ideal learning
- [ ] Narrative loop training (describe → interpret cycle)

### Experiments
- [ ] Personality emergence over long training
- [ ] Self-description quality evaluation
- [ ] Narrative self-modification effects
- [ ] Temperament stability analysis

---

## The Core Idea

Interoception for language models — an internal milieu that:
1. Integrates moment-to-moment signals into a persistent state
2. Contains a self-model (expectation of own reactions)
3. Feeds back to modulate perception/prediction
4. Emerges and differentiates through experience

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         SELF-MODEL              │
                    │  "How I expect to react"        │
                    │  predicted_delta = f(soma, h)   │
                    └───────────┬─────────────────────┘
                                │ prediction
                                ▼
┌─────────────┐    ┌─────────────────────────────────┐     ┌──────────────┐
│  Signals    │───▶│           SOMA                  │────▶│  IDEAL SELF  │
│  surprise   │    │  Integrated internal state      │     │  "Who I want │
│  arousal    │    │  [d_soma] latent vector         │     │   to be"     │
│  valence    │    │                                 │     └──────┬───────┘
│  novelty    │    └───────────┬─────────────────────┘            │
│  certainty  │                │                           discrepancy
│  engagement │   ┌────────────┼────────────┐                     │
└─────────────┘   ▼            ▼            ▼                     ▼
            ┌─────────┐  ┌──────────┐  ┌──────────┐      ┌────────────────┐
            │ DESCRIBE│  │ MODULATE │  │TEMPERAMENT│     │ REGULATION     │
            │ soma→txt│  │ hidden   │  │ slow EMA  │     │ signal toward  │
            └────┬────┘  │ states   │  │ baseline  │     │ ideal          │
                 │       └──────────┘  └───────────┘     └────────────────┘
                 ▼
            ┌─────────┐
            │INTERPRET│
            │ txt→soma│ (narrative self-modification)
            └─────────┘
```

## Implementation

### Files

| File | Description |
|------|-------------|
| `self_state.py` | Core SelfState module and all components |
| `experiential.py:2490-2514` | MemoryAugmentedGPT integration |
| `experiential.py:3083-3120` | Forward pass self-state processing |
| `experiential.py:3621-3690` | Helper methods (get_soma, describe_self, etc.) |
| `experiential.py:3908-3947` | Self-state losses in memory_augmented_loss |
| `test_self_state.py` | Comprehensive tests (10 tests) |

### Components

#### 1. SomaIntegrator (`self_state.py:60-130`)

Integrates raw signals into persistent soma with decay:

```python
# Signal gating: which signals matter depends on current state
gate_input = torch.cat([prev_soma, signals], dim=-1)
signal_weights = self.signal_gate(gate_input)  # [B, n_signals]

# Weight and project to soma space
weighted_signals = signals * signal_weights
signal_contribution = self.signal_proj(weighted_signals)

# Integrate with decay (EMA-style)
new_soma = decay * prev_soma + (1 - decay) * signal_contribution
```

**Timescales:**
- `soma_decay=0.9` — Mood inertia (persists across ~10 steps)
- `temperament_decay=0.999` — Personality baseline (persists across ~1000 steps)

#### 2. SelfModel (`self_state.py:133-200`)

Predicts how soma will change given current state and input:

```python
predicted_delta = self.predictor(torch.cat([soma, input_embedding], dim=-1))
confidence = self.confidence_head(...)

# After actual delta is computed:
self_surprise = ||predicted_delta - actual_delta||
```

**Self-surprise semantics:**
- High → "I didn't expect to react this way" → update self-model
- Low → "I know myself well" → stable identity

#### 3. SelfDescriptionHead (`self_state.py:203-280`)

Projects soma to text space (lossy inspection interface):

```python
hidden_seed = self.soma_to_hidden(soma)  # [B, d_model]
logits = self.description_head(hidden_seed)  # [B, vocab_size]
template_values = self.template_values(soma)  # [B, 5] structured output
```

Template values: `[intensity, valence, arousal, certainty, engagement]`

#### 4. SelfInterpretationHead (`self_state.py:283-345`)

Reads text and modifies soma (narrative self-modification):

```python
raw_delta = self.interpreter(text_embedding)
valence = self.interpretation_valence(torch.cat([text_embedding, soma], dim=-1))

# Valence-dependent effect:
# Positive → calming/integrating (naming tames)
# Negative → amplifying (rumination)
effective_delta = narrative_gain * raw_delta * valence
updated_soma = soma + effective_delta
```

#### 5. IdealSelf (`self_state.py:348-420`)

Maintains actual vs ideal self with discrepancy signal:

```python
# Ideal state (learnable parameter, optionally context-modulated)
ideal = self.ideal_state + context_modulator(context)

# Project soma to ideal space
actual = self.soma_to_ideal_space(soma)

# Discrepancy drives regulation
discrepancy = ideal - actual
regulation_signal = self.regulation_head(discrepancy)
```

#### 6. SomaFeedback (`self_state.py:423-480`)

Modulates hidden states based on soma:

```python
gate = self.hidden_gate(soma).unsqueeze(1)  # [B, 1, d_model]
add = self.hidden_add(soma).unsqueeze(1)

modulated = hidden_states * gate + 0.1 * add
```

Optional: attention bias, logit bias (implemented but not wired by default).

### Usage

```python
from experiential import MemoryAugmentedGPT

model = MemoryAugmentedGPT(
    gpt,
    use_self_state=True,
    self_state_d_soma=64,           # Soma dimension
    self_state_soma_decay=0.9,      # Mood inertia
    self_state_temperament_decay=0.999,  # Personality drift
    self_state_narrative_gain=0.1,  # Text → soma strength
    self_state_use_ideal_self=True, # Enable actual-ideal discrepancy
    self_state_gate_hidden=True,    # Soma modulates hidden states
)

# Forward pass
logits, hidden, mem_out = model(input_ids)

# Access soma state
soma = mem_out['soma']              # [B, d_soma]
temperament = mem_out['temperament']  # [B, d_soma]
self_surprise = mem_out['self_surprise']  # [B]

# Manual narrative loop
description = model.describe_self()
interpretation = model.interpret_self_description(description_embedding)
```

### Losses

```python
from experiential import memory_augmented_loss

loss, loss_dict = memory_augmented_loss(
    logits, targets, mem_out,
    self_state_weight=0.01,  # Enable self-state losses
)

# Loss components:
# - self_model_loss: minimize self-surprise
# - ideal_discrepancy_loss: move toward ideal self
# - regulation_alignment_loss: delta aligns with regulation signal
```

## Design Decisions

### Latent Primary, Text Secondary

The soma is the authoritative representation — a `[d_soma]` vector that captures internal state. Text is a lossy projection for:
- **Inspection**: We can read what the system thinks about itself
- **Modification**: Reading descriptions can change state (narrative loop)

But the text is not the state. Like how describing an emotion is not the emotion.

### Three Timescales

| Timescale | Representation | Decay | Semantics |
|-----------|----------------|-------|-----------|
| Moment | Raw signals | N/A | Surprise, arousal at this instant |
| Mood | Soma | 0.9 | Persists ~10 steps, inertia |
| Personality | Temperament | 0.999 | Persists ~1000 steps, baseline |

### Signal Gating

Which signals matter depends on current state:
- A "curious" soma might upweight novelty signals
- An "anxious" soma might upweight negative valence

This is learned via `signal_gate(soma, signals) → weights`.

### Self-Fulfilling Prophecy

The self-model predicts reactions. If it predicts "I will be bored," does that cause boredom? This loop exists but is bounded by:
1. Actual signals from text (surprise is grounded in prediction error)
2. Discrepancy with ideal self provides counter-pressure
3. Self-model trained to match reality, not to self-confirm

## Open Questions

1. **Curriculum**: Should ideal self be fixed, learned from positive outcomes, or specified externally?

2. **Narrative frequency**: How often should describe → interpret cycle run? Every step? Periodically? On high self-surprise?

3. **Grounding**: Current signals are `[surprise, arousal, valence, novelty, certainty, engagement]`. Are these the right primitives? Should there be more? Fewer?

4. **Feedback strength**: How much should soma modulate hidden states? Current: multiplicative gate + small additive. Too strong → runaway feedback. Too weak → no effect.

5. **Cross-document**: Should soma reset between documents? Currently: soma resets, temperament persists. Is this right?
