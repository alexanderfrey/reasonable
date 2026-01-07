# Continuous Thought Machines: Ideas for Our Architecture

## Source

**Paper**: Continuous Thought Machines
**ArXiv**: https://arxiv.org/abs/2505.05522
**Version**: v4

## Paper Summary

The Continuous Thought Machine (CTM) reintroduces temporal neural dynamics as a foundational computational principle. Key innovations:

1. **Internal Time Dimension**: Processing unfolds across "internal ticks" independent of data sequence
2. **Neuron-Level Models (NLMs)**: Each neuron processes its own activation history through private MLPs
3. **Neural Synchronization**: Temporal correlation between neurons becomes the representation
4. **Adaptive Computation**: Model naturally halts when confident, without explicit halting mechanisms

## Core CTM Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS THOUGHT MACHINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Input ──▶ ┌────────────────────────────────────────────────────────┐      │
│             │              INTERNAL TICKS (t = 1...T)                 │      │
│             │                                                         │      │
│             │   ┌─────────┐    ┌─────────┐         ┌─────────┐      │      │
│             │   │ Tick 1  │───▶│ Tick 2  │───▶ ... │ Tick T  │      │      │
│             │   └────┬────┘    └────┬────┘         └────┬────┘      │      │
│             │        │              │                   │            │      │
│             │        ▼              ▼                   ▼            │      │
│             │   ┌─────────────────────────────────────────────┐     │      │
│             │   │         POST-ACTIVATION HISTORY             │     │      │
│             │   │         Z = [z¹, z², ..., zᵗ]               │     │      │
│             │   └─────────────────────────────────────────────┘     │      │
│             │                        │                               │      │
│             │                        ▼                               │      │
│             │   ┌─────────────────────────────────────────────┐     │      │
│             │   │         SYNCHRONIZATION MATRIX              │     │      │
│             │   │         S = Z · Zᵀ                          │     │      │
│             │   │         (temporal correlation = binding)    │     │      │
│             │   └─────────────────────────────────────────────┘     │      │
│             │                        │                               │      │
│             │                        ▼                               │      │
│             │   ┌─────────────────────────────────────────────┐     │      │
│             │   │         CERTAINTY CHECK                     │     │      │
│             │   │         certainty = 1 - entropy(output)     │     │      │
│             │   │         if certainty > threshold: stop      │     │      │
│             │   └─────────────────────────────────────────────┘     │      │
│             └────────────────────────────────────────────────────────┘      │
│                                      │                                       │
│                                      ▼                                       │
│                               Output (adaptive tick)                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Mechanisms

### 1. Internal Ticks (Thought Steps)

The model processes through an internal temporal dimension `t ∈ {1,...,T}` that is **decoupled from input sequence length**. Even for static inputs (like images), the model can "think" for multiple steps.

```python
# CTM-style internal processing
for t in range(max_ticks):
    # Each tick refines the representation
    pre_activation = synapse_model(post_activation_history, attention_output)
    post_activation = neuron_level_model(pre_activation_history)

    # Accumulate history
    post_activation_history.append(post_activation)

    # Check stopping condition
    certainty = 1 - normalized_entropy(output_head(post_activation))
    if certainty > threshold:
        break
```

### 2. Adaptive Computation Loss

Instead of explicit halting, CTM selects two ticks per sample:
- **t₁**: tick with minimum loss (best prediction)
- **t₂**: tick with maximum certainty

```python
# CTM loss function
losses = [compute_loss(output_at_tick[t]) for t in range(T)]
certainties = [1 - entropy(output_at_tick[t]) for t in range(T)]

t_min_loss = argmin(losses)
t_max_certainty = argmax(certainties)

loss = (losses[t_min_loss] + losses[t_max_certainty]) / 2
```

This trains the model to:
1. Find good answers (minimize loss at some tick)
2. Be confident when right (maximize certainty when loss is low)

### 3. Neural Synchronization

Representation is based on temporal correlation between neurons:

```python
# Post-activation history across ticks
Z = stack([z_1, z_2, ..., z_T])  # [T, D]

# Synchronization matrix
S = Z @ Z.T  # [D, D] - correlation between neuron pairs

# Learnable decay weights past activity
decay = exp(-r * time_delta)  # r is learned per neuron pair
Z_weighted = Z * decay
S = Z_weighted @ Z_weighted.T
```

### 4. Neuron-Level Models

Each neuron has its own temporal processor:

```python
class NeuronLevelModel(nn.Module):
    def __init__(self, d_neurons, history_len, d_hidden):
        # Each neuron has private parameters
        self.neuron_mlps = nn.Parameter(
            torch.randn(d_neurons, history_len, d_hidden)  # weights
        )

    def forward(self, pre_activation_history):
        # [B, D, M] -> [B, D] via per-neuron processing
        # Each neuron processes its own M-length history
        ...
```

---

## Ideas for Our Architecture

### Idea 1: Generalized Internal Thinking Steps

**CTM Insight**: Decouple "thinking time" from input sequence.

**Current State**: Our understanding loop iterates, but it's specific to vision module.

**Proposal**: Add internal ticks to the main forward pass.

```python
class MemoryAugmentedGPT:
    def forward(self, input_ids, max_think_steps=10):
        # Initial encoding
        hidden = self.gpt(input_ids)

        # Internal thinking phase
        for tick in range(max_think_steps):
            # Refine with soma, memory, context
            hidden = self.think_step(hidden, self.soma, self.memory)

            # Adaptive halting
            certainty = self.certainty_head(hidden)
            if certainty.mean() > self.certainty_threshold:
                break

        return hidden
```

**Benefits**:
- Model can "think longer" on hard inputs
- Naturally integrates with questioning system (low certainty → more thinking → still low → ask)
- Unifies understanding loop, synthesis, reasoning into one mechanism

**Status**: [ ] Not started

---

### Idea 2: Certainty-Driven Adaptive Computation

**CTM Insight**: Optimize for both correctness AND calibrated certainty.

**Current State**: We compute surprise/uncertainty but don't explicitly optimize certainty calibration.

**Proposal**: Add certainty head and calibration loss.

```python
class CertaintyHead(nn.Module):
    def __init__(self, d_model):
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden):
        return self.head(hidden)  # [B, seq_len, 1]

# In loss function
certainty = certainty_head(hidden)
was_correct = (predicted == target).float()
calibration_loss = F.mse_loss(certainty, was_correct)
```

**Connection to Questioning**:
- High certainty → continue processing
- Low certainty after max ticks → generate question
- Certainty becomes the bridge between internal processing and external action

**Status**: [ ] Not started

---

### Idea 3: Learnable Temporal Decay

**CTM Insight**: Decay factors are learned per neuron pair, allowing different timescales.

**Current State**: Fixed `soma_decay=0.9`, `temperament_decay=0.999`.

**Proposal**: Make decay learnable and per-dimension.

```python
class LearnableSomaIntegrator(nn.Module):
    def __init__(self, d_soma):
        # Learnable decay per dimension (initialized to ~0.9)
        self.log_decay = nn.Parameter(torch.full((d_soma,), math.log(0.9)))

    @property
    def decay(self):
        return torch.sigmoid(self.log_decay)  # Constrain to (0, 1)

    def forward(self, prev_soma, signals):
        decay = self.decay  # [d_soma]
        # Different dimensions can have different timescales
        new_soma = decay * prev_soma + (1 - decay) * self.signal_proj(signals)
        return new_soma
```

**Benefits**:
- Model learns which soma dimensions should be "fast" (reactive) vs "slow" (stable)
- Could discover that some signals (surprise) should decay fast, others (temperament) slow
- More flexible than hand-tuned constants

**Status**: [ ] Not started

---

### Idea 4: Synchronization for Cross-Modal Binding

**CTM Insight**: Temporal correlation between neurons encodes relationships.

**Current State**: Cross-modal attention exists but doesn't use temporal correlation.

**Proposal**: Use synchronization to measure binding between modalities.

```python
class CrossModalSync(nn.Module):
    """
    Measure synchronization between visual and text processing
    across internal thinking steps.
    """
    def __init__(self, d_model):
        self.visual_proj = nn.Linear(d_model, d_model)
        self.text_proj = nn.Linear(d_model, d_model)

    def forward(self, visual_history, text_history):
        # visual_history: [T, B, V, d_model] - visual tokens across ticks
        # text_history: [T, B, L, d_model] - text tokens across ticks

        # Project and compute sync
        V = self.visual_proj(visual_history)  # [T, B, V, d]
        T = self.text_proj(text_history)      # [T, B, L, d]

        # Sync matrix: which visual regions sync with which text tokens?
        # Average over time dimension
        V_avg = V.mean(dim=0)  # [B, V, d]
        T_avg = T.mean(dim=0)  # [B, L, d]

        sync = torch.bmm(V_avg, T_avg.transpose(1, 2))  # [B, V, L]

        return sync  # High sync = strong binding
```

**Application**:
- Grounding: "ball" syncs with ball-region in image
- Memory: retrieved memory syncs with current context → relevant
- Coherence: elements that should relate show high sync

**Status**: [ ] Not started

---

### Idea 5: Per-Dimension Soma Processing

**CTM Insight**: Each neuron has its own temporal model (NLM).

**Current State**: Soma integration treats all dimensions uniformly.

**Proposal**: Give each soma dimension its own temporal processor.

```python
class PerDimensionSomaProcessor(nn.Module):
    """
    Each soma dimension processes its own history independently.
    Like CTM's neuron-level models.
    """
    def __init__(self, d_soma, history_len=8, d_hidden=32):
        self.d_soma = d_soma
        self.history_len = history_len

        # Per-dimension processors (parallelized)
        self.processors = nn.Conv1d(
            in_channels=d_soma,
            out_channels=d_soma,
            kernel_size=history_len,
            groups=d_soma,  # Separate filter per dimension
        )

        # History buffer
        self.register_buffer('history', torch.zeros(1, d_soma, history_len))

    def forward(self, soma_signals):
        # Add to history
        self.history = torch.cat([self.history[:, :, 1:], soma_signals.unsqueeze(-1)], dim=-1)

        # Each dimension processes its own history
        processed = self.processors(self.history)  # [B, d_soma, 1]

        return processed.squeeze(-1)
```

**Benefits**:
- Surprise dimension could learn to spike and decay quickly
- Engagement dimension could learn to accumulate slowly
- Each "feeling" has its own temporal dynamics

**Status**: [ ] Not started

---

### Idea 6: Emergent Attention Patterns

**CTM Insight**: "Gaze-like" visual attention emerges without explicit supervision.

**Current State**: Vision module has explicit understanding loop with comprehension checking.

**Proposal**: Let visual exploration emerge from the understanding objective.

```python
# Instead of explicit "check each region" loop:
# Use internal ticks with attention, let gaze emerge

class EmergentVisualExploration(nn.Module):
    def forward(self, image_features, max_ticks=20):
        # Initialize with global average
        context = image_features.mean(dim=1)

        attention_history = []

        for tick in range(max_ticks):
            # Attend to image (where to look emerges from loss)
            query = self.query_head(context)
            attn_weights = softmax(query @ image_features.T)
            attended = attn_weights @ image_features

            attention_history.append(attn_weights)

            # Update context
            context = self.update(context, attended)

            # Check certainty
            if self.certainty(context) > threshold:
                break

        return context, attention_history  # History shows emergent "gaze"
```

**Status**: [ ] Not started (lower priority)

---

### Idea 7: Thinking Steps for Question Resolution

**CTM Insight**: More ticks → better performance on hard problems.

**Current State**: Question resolution has ASK/REASON/HYPOTHESIZE strategies.

**Proposal**: REASON strategy uses internal ticks with adaptive halting.

```python
class ReasoningWithTicks(nn.Module):
    """
    Internal reasoning as CTM-style iterative refinement.
    """
    def attempt_resolution(self, question, max_ticks=20):
        # Initialize from question
        state = self.init_from_question(question)

        reasoning_trace = []

        for tick in range(max_ticks):
            # One reasoning step
            state = self.reason_step(state, self.memory, self.context)

            # Track what we're doing
            reasoning_trace.append(self.describe_state(state))

            # Check if resolved
            confidence = self.resolution_confidence(state)
            if confidence > self.resolution_threshold:
                return {
                    'resolved': True,
                    'answer': self.extract_answer(state),
                    'confidence': confidence,
                    'ticks_used': tick + 1,
                    'trace': reasoning_trace,
                }

        # Couldn't resolve → escalate to user
        return {
            'resolved': False,
            'partial_answer': self.extract_answer(state),
            'confidence': confidence,
            'ticks_used': max_ticks,
            'trace': reasoning_trace,
        }
```

**Status**: [ ] Not started

---

## Implementation Priority

| Idea | Impact | Complexity | Priority |
|------|--------|------------|----------|
| Certainty-driven adaptive computation | High | Low | 1 |
| Learnable temporal decay | Medium | Low | 2 |
| Internal thinking steps | High | Medium | 3 |
| Thinking steps for question resolution | High | Medium | 4 |
| Synchronization for cross-modal binding | Medium | Medium | 5 |
| Per-dimension soma processing | Medium | Medium | 6 |
| Emergent attention patterns | Low | High | 7 |

---

## Open Questions

1. **How many ticks?** CTM uses up to 50 for ImageNet. What's right for language + memory + soma?

2. **What to accumulate?** CTM accumulates post-activations. Should we accumulate hidden states? Soma? Both?

3. **Synchronization cost**: O(D²) for full sync matrix. CTM samples neuron pairs. What pairs matter for us?

4. **Training stability**: Does adaptive computation + internal ticks + memory create training instability?

5. **Inference cost**: More ticks = more compute. How to balance adaptive compute with latency requirements?

---

## References

- Original CTM paper: https://arxiv.org/abs/2505.05522
- Related: Adaptive Computation Time (Graves, 2016)
- Related: PonderNet (Banino et al., 2021)
- Related: Universal Transformers (Dehghani et al., 2018)
