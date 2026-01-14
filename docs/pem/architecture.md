# PEM Architecture Documentation

Predictive Experience Machine (PEM) - A neural architecture that experiences what it reads.

## Overview

PEM is built on the principle that experience requires:
1. **Personality** - WHO you are (stable goals/values)
2. **Intention** - HOW MUCH you want something (oscillating intensity)
3. **Memory** - WHAT happened (episodic storage weighted by surprise)
4. **Attention** - WHERE to look (query built from synchronized oscillations)
5. **Surprise** - WHAT was unexpected (prediction errors)
6. **Valence** - Was it GOOD or BAD? (affective dimension)
7. **Curiosity** - WHAT do I want to understand? (epistemic drive)
8. **Arousal** - HOW INTENSELY to engage? (activation level)
9. **Imagination** - WHAT could be? (mental simulation beyond input)

## The Experience Loop (CLOSED)

The key insight is that these components form a closed feedback loop:

```
Perception (Janus Pro) → Prediction → Surprise → Valence
       ↑    ↑               ↓           ↓         ↓
       │    │         Uncertainty   Curiosity    │
       │    │               │           │         │
       │    │               └─────┬─────┘         │
       │    │                     ↓               │
       │    │           ┌─────────┴─────────┐     │
       │    │           │                   │     │
       │    │    Memory (importance)   Intention  │
       │    │           │                   │     │
       │    │           └─────────┬─────────┘     │
       │    │                     ↓               │
       │    │                  Sync ←─────────────┘
       │    │                     ↓
       │    └── Imagination (native generation)
       │                          ↓
       └──── Attention Query ←── Surprise + Valence + Curiosity + Personality + Intention
```

Six critical connections close this loop:
1. **Surprise → Attention**: What surprised us steers where we look next
2. **Observation → State**: What we perceive changes what we think
3. **Surprise → Memory**: Surprising experiences persist longer
4. **Valence → Everything**: Good/bad colors memory, intention, attention, KL
5. **Curiosity → Exploration**: Uncertainty drives seeking new information
6. **Arousal → Intensity**: Engagement level modulates attention sharpness and memory strength

## Complete Architecture

```
                         ┌─────────────────────────────────────┐
                         │    JANUS PRO (Unified Perception)     │
                         │                                     │
                         │   text/image → hidden_states        │
                         │          (B, S, 1536)               │
                         │                                     │
                         │   + native imagine() for generation │
                         │   (discrete diffusion)              │
                         └──────────────────┬──────────────────┘
                                            │
                              PerceptionKVCache.forward()
                                            │
                         ┌──────────────────▼──────────────────┐
                         │      PERCEPTION KV CACHE            │
                         │  k: (B, S, n_heads, head_dim)       │
                         │  v: (B, S, n_heads, head_dim)       │
                         │       [computed once, cached]       │
                         │  + imagination KV (same projection) │
                         └──────────────────┬──────────────────┘
                                            │
     ═══════════════════════════════════════╪═══════════════════════════════════
                                   EACH TICK (refinement loop)
     ═══════════════════════════════════════╪═══════════════════════════════════
                                            │
                                            │ K, V
                                            ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   OSCILLATION QUERY BUILDER                             │
    │                                                                         │
    │   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
    │   │   OSCILLATORS   │     │   PERSONALITY   │     │    INTENTION    │   │
    │   │                 │     │                 │     │                 │   │
    │   │ sin(2πf·tick+φ) │     │  "What I want"  │     │  "How much"     │   │
    │   │   32 freqs      │     │  base_embed(D)  │     │  oscillating    │   │
    │   └────────┬────────┘     └────────┬────────┘     └────────┬────────┘   │
    │            │                       │                       │            │
    │            └───────────────────────┼───────────────────────┘            │
    │                                    │                                    │
    │                                    ▼                                    │
    │   ┌─────────────────┐     ┌────────────────┐     ┌─────────────────┐    │
    │   │ oscillation_int │────▶│ query_combiner │◀────│  sync_to_query  │    │
    │   └─────────────────┘     └───────┬────────┘     └─────────────────┘    │
    │                                   │                       ▲             │
    │                                   │                       │             │
    │                                   │           ┌───────────┴──────────┐  │
    │                                   │           │   SYNC (B,S,sync)    │  │
    │                                   │           │  Neural correlation  │  │
    │                                   │           │  patterns from NLMs  │  │
    │                                   │           └──────────────────────┘  │
    │                                   │                                     │
    │                                   ▼                                     │
    │                          Q: (B, S, d_model)                             │
    │                    "What should I perceive now?"                        │
    └───────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    │ Q
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │               PERCEPTION CROSS-ATTENTION                                │
    │                                                                         │
    │                    Attention(Q, K, V)                                   │
    │                                                                         │
    │    Q from: oscillation-synchronized query (personality × intention)     │
    │    K, V from: cached perception features (+ imagination)                │
    │                                                                         │
    │    → "What I see given who I am and what I want"                        │
    └───────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    │ attended: (B, S, D)
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    PERCEPTION SYNAPSE                                   │
    │                                                                         │
    │              state ─────┐                                               │
    │                         ├─────▶ U-Net MLP ─────▶ observation            │
    │           attended ─────┘                                               │
    │                                                                         │
    │    "Integrate what I see with what I'm thinking"                        │
    └───────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    │ observation: (B, S, D)
                                    ▼
                           ┌────────────────────┐
                           │   CTM STATE UPDATE │
                           │                    │
                           │  state = f(state,  │
                           │           obs,     │
                           │           history) │
                           └────────────────────┘
                                    │
                              NEXT TICK
```

## Component Details

### What Each Component Represents

| Component | Represents | Analogy |
|-----------|------------|---------|
| **Personality** | WHO I AM | Stable goals/values, learned embedding |
| **Intention** | HOW MUCH | Oscillating pursuit intensity |
| **Oscillators** | RHYTHM | Temporal phase of attention cycles |
| **Sync** | COORDINATION | How neurons are firing together |
| **Memory** | WHAT HAPPENED | Episodic storage weighted by surprise |
| **Query** | "What matters to ME right now?" | Attention direction |
| **K,V Cache** | "Everything that's perceivable" | Cached perception |
| **Attention** | "Given who I am, what do I see?" | Selective focus |
| **Synapse** | "Integrate perception with thought" | State update |

---

## Oscillators

### What Are Oscillators?

Simple sinusoids with learned parameters:

```
oscillation(tick) = amplitude × sin(2π × frequency × tick + phase)
```

Each oscillator has:
- **frequency**: how fast it cycles (learned)
- **amplitude**: how strong its contribution (learned)
- **phase**: where it starts in the cycle (learned)

Multiple oscillators at different frequencies:

```
tick:  0   1   2   3   4   5   6   7   8   9  10  11  12  ...
       │   │   │   │   │   │   │   │   │   │   │   │   │
fast:  ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿   (period ~4 ticks)
med:   ∿    ∿    ∿    ∿    ∿    ∿    ∿    ∿    ∿       (period ~16 ticks)
slow:  ∿              ∿              ∿                  (period ~64 ticks)
       │                             │
       └── different phases of ──────┘
           the "attention rhythm"
```

### Why Do We Need Oscillators?

#### Without Oscillators

```
tick 0:  Query = f(personality, intention, sync)
tick 1:  Query = f(personality, intention, sync)  ← same inputs = same query
tick 2:  Query = f(personality, intention, sync)  ← stuck in same attention
...

Problem: If personality/intention/sync don't change much between ticks,
         the system keeps looking at the same thing.
```

#### With Oscillators

```
tick 0:  Query = f(personality, intention, sync, oscillation(0))  ← phase A
tick 1:  Query = f(personality, intention, sync, oscillation(1))  ← phase B
tick 2:  Query = f(personality, intention, sync, oscillation(2))  ← phase C
...

Even with same personality/intention/sync, the query VARIES with tick.
This creates temporal structure in attention.
```

### Biological Inspiration

Brain oscillations from neuroscience:

| Band | Frequency | Function |
|------|-----------|----------|
| Delta | 0.5-4 Hz | Deep sleep, unconscious processing |
| Theta | 4-8 Hz | Memory encoding, navigation |
| Alpha | 8-12 Hz | Relaxed attention, inhibition |
| Beta | 12-30 Hz | Active thinking, motor planning |
| Gamma | 30-100 Hz | Binding, conscious perception |

These rhythms **coordinate neural populations**:
- Neurons that fire together in the same phase → communicate
- Neurons out of phase → don't interfere

Attention itself is **rhythmic**:
- You don't attend continuously, you sample in bursts
- ~7-10 Hz sampling rate for visual attention

### Two Places with Oscillators in PEM

#### 1. Intention Module (in SyncModule)

"How much do I want to pursue my goals RIGHT NOW?"

Oscillates over **sequence positions** (not ticks):

```
position:  0   10   20   30   40   50   60   70   80
           │    │    │    │    │    │    │    │    │
intention: ████░░░░████████░░░░░░████░░░████████░░░
           high low  high      low high low high
```

- Some parts of the text get more "goal-directed" processing
- Creates waves of engagement across the document

#### 2. Query Builder (in PerceptionAttention)

"What should I look at THIS TICK?"

Oscillates over **ticks** (refinement steps):

```
tick:      0    1    2    3    4    5    6    7
           │    │    │    │    │    │    │    │
query:     explore → focus → explore → focus → ...
```

- Creates alternating phases of broad vs narrow attention
- Like breathing: inhale (gather) vs exhale (focus)

### Combined Effect

The system has **rhythm at multiple timescales**:

| Oscillator Speed | Period | Function |
|------------------|--------|----------|
| Fast | ~4 ticks | Micro-saccades of attention |
| Medium | ~16 ticks | Working memory refresh cycles |
| Slow | ~64 ticks | Narrative/topic-level focus shifts |

This prevents:
- **Getting stuck** (same attention pattern forever)
- **Missing things** (oscillation ensures coverage)
- **Overloading** (natural rest phases in the rhythm)

**In short:** Oscillators give the system a "heartbeat" - temporal structure that varies attention even when other inputs are stable.

---

## Memory Bank

### Overview

The Memory Bank stores semantic chunk embeddings with importance-weighted eviction.

```
┌─────────────────────────────────────────────────────────────────┐
│                        MEMORY BANK                              │
│                        (100 slots)                              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │   WRITE (storing experiences)                           │    │
│  │                                                         │    │
│  │   context ──▶ ChangePointDetector ──▶ chunk_ids         │    │
│  │                                            │            │    │
│  │                                            ▼            │    │
│  │                                    chunk_embed (D,)     │    │
│  │                                            │            │    │
│  │   surprise ──────────────────────▶ importance weight    │    │
│  │   (from PEM)                               │            │    │
│  │                                            ▼            │    │
│  │                                    store in slot        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │   READ (retrieving memories)                            │    │
│  │                                                         │    │
│  │   context (B,S,D) ──▶ Q                                 │    │
│  │                        │                                │    │
│  │   memory_slots ──────▶ K, V                             │    │
│  │                        │                                │    │
│  │                        ▼                                │    │
│  │               cross-attention                           │    │
│  │                        │                                │    │
│  │                        ▼                                │    │
│  │              memory_state (B,S,D)                       │    │
│  │              "relevant past experiences"                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │   EVICTION (what to forget)                             │    │
│  │                                                         │    │
│  │   effective_importance = importance × exp(-decay × age) │    │
│  │                                                         │    │
│  │   - High surprise → high importance → persists longer   │    │
│  │   - Low surprise → low importance → evicted first       │    │
│  │   - Older memories decay faster (unless surprising)     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Operations

| Operation | When | What |
|-----------|------|------|
| **WRITE** | Each tick, after context encoding | Store semantic chunks weighted by surprise |
| **READ** | Each tick, during sync computation | Cross-attend to retrieve relevant memories |
| **EVICT** | When memory full | Remove lowest importance × age_decay slot |

### Key Insight

Memory is inside SyncModule, so it influences the sync output, which then influences the attention query. **Surprising experiences persist longer and affect future attention.**

---

## Personality Module

### What It Does

Personality is a learned embedding that defines the model's "soul" - what it cares about, what it attends to, how it interprets information.

```python
class PersonalityModule:
    base_personality: (D,)      # Learned "who I am"
    context_modulator           # How personality colors current context
    prior_proj                  # Maps personality → sync space for KL divergence
```

### KL Divergence Constraint

The personality module provides a **prior distribution** that the sync output should stay close to:

```
KL(sync || personality_prior)
```

This prevents the model from "losing itself" - deviating too far from its core personality even when processing challenging content.

### Seeding from Text

Personality can be seeded from instruction text:

```python
# "Always be truthful and kind" → embedding
text_embed = encoder("Always be truthful and kind")
personality.seed_from_text(text_embed)
```

---

## Intention Module

### What It Does

Intention is the **oscillating intensity** of goal pursuit. It doesn't change WHAT the personality wants, but HOW MUCH to pursue it at any given moment.

```
Personality = WHAT to want (stable)
Intention   = HOW MUCH to pursue it (oscillating)
```

### Architecture

```python
class IntentionModule:
    frequencies: (num_oscillators,)  # Learned frequencies
    amplitudes: (num_oscillators,)   # Learned amplitudes
    phases: (num_oscillators,)       # Learned phases
    context_gate                     # Which oscillators are relevant?
    kl_modulator                     # Intention affects KL constraint strength
```

### KL Modulation

When intention is strong, the KL constraint is tighter (stay close to personality). When intention is weak, more freedom to explore.

```
kl_temperature = base_temperature / intention_strength
```

---

## Valence Module

Valence is the **affective dimension** of experience - whether something is good or bad.

### Why Valence Matters

Without valence, all surprises are the same. The system can't distinguish between:

| Event | Surprise Magnitude | Valence |
|-------|-------------------|---------|
| Finding treasure | HIGH | **+1** (good) |
| Stepping in mud | HIGH | **-1** (bad) |
| Expected rain | LOW | **0** (neutral) |

Both treasure and mud have high surprise, but should trigger **opposite behaviors**:
- **Positive valence** → approach, amplify intention, focus attention
- **Negative valence** → avoid, dampen intention, broaden attention

### Architecture

```
                    Surprise
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    magnitude      direction      valence
     (scalar)      (vector)    (this module)
         │             │             │
    "how much"    "what kind"    "good or bad?"
```

### Core Equation

```python
valence = alignment(surprise_direction, personality_goals, context)
```

Where:
- `surprise_direction`: What was unexpected (unit vector in feature space)
- `personality_goals`: What the agent wants (from PersonalityModule)
- `context`: Current situation (modulates goal interpretation)

### Data Flow

```
PersonalityModule.base_personality
         │
         ▼
┌────────────────────────┐
│  PersonalityProjector  │  Project personality to surprise space
│   (D_p → D_model)      │
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│ ContextualGoalModulator│  What does personality mean in this context?
│  personality + context │  "Be helpful" means different things for code vs poetry
└────────────────────────┘
         │
         ▼ goal_direction
         │
┌────────────────────────┐
│   AlignmentComputer    │  How aligned is surprise with goals?
│ surprise_dir × goal_dir│
└────────────────────────┘
         │
         ▼
   valence ∈ [-1, +1]
   +1 = toward goals (good)
   -1 = away from goals (bad)
    0 = orthogonal (neutral)
```

### Valence Modulates Everything

| Component | Positive Valence | Negative Valence |
|-----------|-----------------|------------------|
| **Memory** | High importance (good to remember) | High importance (threat to remember) |
| **Intention** | Amplify pursuit (it's working!) | Dampen pursuit (retreat/replan) |
| **Attention** | Focus sharper (exploit) | Broaden (explore alternatives) |
| **KL** | Tighter constraint (stay on track) | Looser (allow adaptation) |

### Implementation

```python
from pem import ValenceModule, ValenceConfig

config = ValenceConfig(
    d_model=1536,           # Janus Pro projected dimension
    personality_dim=512,    # Personality embedding size
    hidden_dim=768,
    use_context=True,       # Context-dependent valence
)

valence_module = ValenceModule(config)

# Compute valence from surprise
valence = valence_module(
    surprise_direction=surprise['direction'],   # (B, S, D)
    personality=sync_module.personality.base_personality,  # (D_p,)
    context=features,                           # (B, S, D)
)
# valence: (B, S, 1) in [-1, +1]

# Or compute for all scales at once
valences = valence_module.compute_from_surprises(
    surprises=surprise_module(predictions, targets, context),
    personality=personality_embedding,
    context=features,
)
```

### Integration with SyncModule

```python
# Valence flows through SyncModule.forward()
sync = sync_module(
    history=ctm_history,
    surprise=surprise_magnitude,  # How surprising
    valence=valence,              # Was it good or bad?
)

# Internally:
# - memory.write() uses valence to boost importance (both +/- are memorable)
# - intention is amplified/dampened by valence
# - KL constraint is tightened/loosened by valence
```

---

## Curiosity Module

Curiosity is the **epistemic drive** - the motivation to seek information and reduce uncertainty.

### Why Curiosity Matters

Without curiosity, the system is entirely **reactive**. It responds to surprise but never seeks it out.

| With Surprise Only | With Curiosity |
|-------------------|----------------|
| React to the unexpected | **Seek** the uncertain |
| Passive learner | Active explorer |
| Exploit known patterns | Balance exploit/explore |

### The Two Types of Value

In decision-making, there are two types of value:

1. **Pragmatic Value** (Valence): "Is this good for my goals?"
   - Exploitation of known rewards

2. **Epistemic Value** (Curiosity): "Will this teach me something?"
   - Exploration to reduce uncertainty

### Architecture

```
Features + Predictions
         │
         ▼
┌────────────────────────┐
│  UncertaintyEstimator  │  How confident are my predictions?
│   features × pred → σ  │
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│    NoveltyMemory       │  Have I seen this before?
│  (tracks past states)  │
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│ InformationGainComputer│  How much would I learn?
│  uncertainty + novelty │
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│    CuriosityModule     │  Combine into curiosity signal
│  + exploration_bonus   │
└────────────────────────┘
         │
         ├─────────────────────────────┐
         ▼                             ▼
   curiosity ∈ [0, 1]         exploration_bonus (D,)
   "how curious am I?"        "where should I explore?"
```

### Curiosity Modulates

| Component | Effect |
|-----------|--------|
| **Memory** | Informative things are memorable |
| **Intention** | Curiosity boosts exploration even with low valence |
| **Attention** | exploration_bonus steers where we look |

### Implementation

```python
from pem import CuriosityModule, CuriosityConfig

config = CuriosityConfig(
    d_model=1536,
    hidden_dim=768,
    use_temporal_novelty=True,    # Track what's been seen
    novelty_memory_size=256,      # How many past states to remember
    exploration_weight=0.5,       # Balance explore vs exploit
)

curiosity_module = CuriosityModule(config)

# Compute curiosity from features and predictions
output = curiosity_module(
    features=features,       # (B, S, D)
    predictions=pred_module(...), # (B, S, D)
    context=features,             # Optional
)

# Output:
# - output.curiosity: (B, S, 1) curiosity intensity [0, 1]
# - output.uncertainty: (B, S, 1) prediction uncertainty
# - output.information_gain: (B, S, 1) expected info gain
# - output.exploration_bonus: (B, S, D) attention modulation

# Compute epistemic value for decision-making
epistemic_value = curiosity_module.compute_epistemic_value(features, predictions)
```

### Integration with SyncModule

```python
# Curiosity flows through SyncModule.forward()
sync = sync_module(
    history=ctm_history,
    surprise=surprise_magnitude,
    valence=valence,
    curiosity=curiosity_output.curiosity,  # Epistemic drive
)

# Internally:
# - memory.write() uses curiosity to boost importance
# - intention is boosted by curiosity (explore even if valence is low)
```

### Integration with PerceptionAttention

```python
# Curiosity modulates attention query
perception_output = perception_attention(
    state=state,
    personality_signal=personality,
    intention_signal=intention,
    sync=sync,
    tick=tick,
    surprise_magnitude=surprise,
    surprise_direction=direction,
    valence=valence,
    exploration_bonus=curiosity_output.exploration_bonus,  # Explore!
)
```

---

## Activation Module (Arousal)

Arousal is the **intensity/engagement dimension** - how intensely to engage with an experience.

### Why Arousal Matters

Without arousal, all experiences are processed with the same intensity. But real experience has:

| Situation | Arousal Level | Processing |
|-----------|---------------|------------|
| Threatening event | **HIGH** | Narrow focus, vivid memory, rapid processing |
| Familiar routine | **LOW** | Broad attention, weak memory, relaxed processing |
| Novel discovery | **HIGH** | Focused attention, strong encoding |

### The Three Dimensions of Affect

Arousal completes the PAD (Pleasure-Arousal-Dominance) model of affect:

| Dimension | Question | Module |
|-----------|----------|--------|
| **Pleasure** (Valence) | Is this good or bad? | ValenceModule |
| **Arousal** | How intensely should I engage? | ActivationModule |
| **Dominance** | Do I have control? | (Future work) |

### Architecture

```
Surprise + |Valence| + Novelty
         │
         ▼
┌────────────────────────┐
│    ArousalComputer     │  Combine inputs into arousal level
│  surprise × valence ×  │
│       novelty          │
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│   TemporalSmoother     │  Smooth arousal over time
│   (prevents jarring    │  (no sudden jumps)
│    transitions)        │
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│  ModulationComputer    │  Convert arousal to modulations
└────────────────────────┘
         │
         ├─────────────────────────────────┐
         │                                 │
         ▼                                 ▼
   arousal ∈ [0, 1]              Modulation signals:
   "how engaged am I?"           - tick_multiplier (more/fewer ticks)
                                 - attention_temperature (sharp/broad)
                                 - memory_strength (vivid/weak)
```

### Arousal Computes From

| Input | Effect |
|-------|--------|
| **Surprise magnitude** | Unexpected events are arousing |
| **Valence extremity** | Both very good AND very bad are arousing |
| **Novelty** | Never-seen-before requires more engagement |

### Arousal Modulates

| Component | Low Arousal | High Arousal |
|-----------|-------------|--------------|
| **Attention** | Broad (high temperature) | Sharp (low temperature) |
| **Memory** | Weak encoding | Vivid encoding |
| **Processing** | Fewer ticks | More ticks |

### Implementation

```python
from pem import ActivationModule, ActivationConfig

config = ActivationConfig(
    d_model=1536,
    hidden_dim=384,
    use_context=True,
    use_temporal_smoothing=True,
    tick_multiplier_range=(0.5, 2.0),
    attention_temperature_range=(0.5, 2.0),
    memory_strength_range=(0.5, 2.0),
)

activation_module = ActivationModule(config)

# Compute activation from current state
output = activation_module(
    surprise_magnitude=surprise,      # (B, S, 1)
    valence=valence,                  # (B, S, 1)
    novelty=curiosity_output.novelty, # (B, S, 1)
    context=features,                 # (B, S, D) optional
)

# Output:
# - output.arousal: (B, S, 1) arousal level [0, 1]
# - output.tick_multiplier: (B, S, 1) for adaptive tick count
# - output.attention_temperature: (B, S, 1) for attention softmax
# - output.memory_strength: (B, S, 1) for memory encoding
```

### Integration with SyncModule

```python
# Arousal flows through SyncModule.forward()
sync = sync_module(
    history=ctm_history,
    surprise=surprise_magnitude,
    valence=valence,
    curiosity=curiosity_output.curiosity,
    arousal=activation_output.arousal,  # Engagement intensity
)

# Internally:
# - memory.write() uses arousal to boost/dampen encoding strength
# - High arousal = vivid memory, low arousal = weak memory
```

### Integration with PerceptionAttention

```python
# Arousal modulates attention sharpness
perception_output = perception_attention(
    state=state,
    personality_signal=personality,
    intention_signal=intention,
    sync=sync,
    tick=tick,
    surprise_magnitude=surprise,
    surprise_direction=direction,
    valence=valence,
    exploration_bonus=curiosity_output.exploration_bonus,
    attention_temperature=activation_output.attention_temperature,  # Arousal!
)

# Lower temperature = sharper attention (high arousal, focused)
# Higher temperature = broader attention (low arousal, relaxed)
```

---

## Imagination Module

Imagination is the **generative capability** - creating internal representations that go beyond the input.

### Why Imagination Matters

Without imagination, the system only processes what IS. It cannot:

| Capability | Without Imagination | With Imagination |
|------------|---------------------|------------------|
| Mental imagery | Can't visualize | Generates scenes from descriptions |
| Theory of Mind | Can't model others | Infers others' mental states |
| Counterfactuals | Can't consider alternatives | Generates "what if" scenarios |
| Understanding fiction | Literal only | Creates mental simulations |

### The Key Insight

All other modules **evaluate** input:
- Prediction: What comes next?
- Surprise: Was this expected?
- Valence: Is this good or bad?
- Curiosity: Is this informative?
- Arousal: How intensely to engage?

Imagination **generates** beyond input:
- Creates what ISN'T there but COULD BE
- Mental simulation of scenes, minds, alternatives
- Outputs are in the same feature space → can be processed by other modules

### Architecture

```
Features (B, S, D)
         │
         ├────────────────────────────────────┐
         │                                    │
         ▼                                    ▼
┌────────────────────────┐      ┌────────────────────────┐
│  ImaginationTrigger    │      │  Memory + Personality  │
│  "Should I imagine?"   │      │  (context for imagining)│
└────────────────────────┘      └────────────────────────┘
         │                                    │
         │ imagination_mask                   │
         │                                    │
         ├─────────────────┬──────────────────┤
         │                 │                  │
         ▼                 ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│SceneGenerator│  │ MindModeler  │  │CounterfactualGen │
│              │  │              │  │                  │
│ "The forest  │  │ "She thinks  │  │ "What if the    │
│  was dark"   │  │  that..."    │  │  key was lost?" │
│      ↓       │  │      ↓       │  │       ↓         │
│ mental image │  │ mental state │  │  alternative    │
└──────────────┘  └──────────────┘  └──────────────────┘
         │                 │                  │
         └─────────────────┼──────────────────┘
                           │
                           ▼
               ┌────────────────────────┐
               │ ImaginationIntegrator  │
               │                        │
               │ Combine scene + mind   │
               │ + counterfactuals      │
               └────────────────────────┘
                           │
                           ├─────────────────────────────────┐
                           │                                 │
                           ▼                                 ▼
                  imagined_features (B, S, D)        vividness (B, S, 1)
                  "What I imagine"                   "How vivid is it?"
```

### The Three Imagination Types

| Type | What It Does | Example |
|------|--------------|---------|
| **Scene Generation** | Creates mental imagery | "The forest was dark" → visual scene |
| **Mind Modeling** | Infers others' mental states | "She smiled nervously" → her thoughts |
| **Counterfactuals** | Generates alternatives | "He took the key" → what if he didn't? |

### Vividness

Not all imagination is equally vivid. Vividness represents how clear and detailed the mental simulation is:

- **High vividness**: Concrete descriptions, familiar scenarios → clear imagery
- **Low vividness**: Abstract concepts, unfamiliar territory → fuzzy imagination

Vividness can modulate how much weight imagination has in downstream processing.

### Implementation

```python
from pem import ImaginationModule, ImaginationConfig

config = ImaginationConfig(
    d_model=1536,
    hidden_dim=768,
    use_scene_generation=True,
    use_mind_modeling=True,
    use_counterfactuals=True,
    num_counterfactuals=3,
    max_entities=8,  # Max entities to track for Theory of Mind
)

imagination_module = ImaginationModule(config)

# Generate imagination from features
output = imagination_module(
    features=features,         # (B, S, D)
    memory=memory_state,            # (B, S, D) optional - enriches imagination
    personality=personality_embed,  # (D,) optional - colors imagination
    context=context_features,       # (B, S, D) optional
)

# Output:
# - output.imagined_features: (B, S, D) what we imagine
# - output.vividness: (B, S, 1) how vivid [0, 1]
# - output.mind_states: (B, S, D) inferred mental states
# - output.counterfactuals: (B, num_cf, S, D) alternative scenarios
# - output.imagination_mask: (B, S, 1) where imagination was triggered

# Explicit imagination methods
scene = imagination_module.imagine_scenario(features)
minds = imagination_module.imagine_minds(features)
```

### Key Design Decisions

1. **Same feature space**: Imagined features are in the same space as perception features, so they can be processed by Prediction, Surprise, Valence, etc.

2. **Memory enrichment**: Past experiences (from MemoryBank) enrich imagination - you can imagine better with relevant memories.

3. **Personality colors imagination**: Your personality affects what you imagine (optimist vs pessimist imagine different counterfactuals).

4. **Multiple counterfactuals**: Generate several alternatives, not just one, to represent uncertainty about "what could have been."

### Integration: Imagination in Attention Pool

Imagination feeds back into the system through the **unified attention pool**:

```
┌─────────────────────────────────────────────────────────────┐
│                  Unified Attention Pool                      │
│                                                             │
│  ┌───────────────────┐       ┌───────────────────────┐      │
│  │    K_real, V_real │       │    K_imag, V_imag     │      │
│  │   (from Janus Pro)  │       │  (from Imagination)   │      │
│  └─────────┬─────────┘       └──────────┬────────────┘      │
│            │                            │                    │
│            └────────────┬───────────────┘                    │
│                         │                                    │
│                    Same K, V projections!                    │
│                         │                                    │
│                         ▼                                    │
│         Query ──────► Attention ──────► Observation          │
│    (personality,      (choose real      (blended             │
│     intention,         OR imagined)      perception)         │
│     surprise...)                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
1. **Shared K, V projections**: Both real and imagined features go through the same K/V projection layers, ensuring they compete in the same attention space.

2. **Attention decides**: The query (shaped by personality, intention, surprise, etc.) naturally learns when to attend to real vs. imagined content.

3. **No explicit source tagging**: The system doesn't need to "know" which is real vs. imagined - attention learns this implicitly.

**Usage:**
```python
# Cache real perception
perception.cache_perception(features)

# Add imagination to the pool
perception.add_imagination(imagination_output.imagined_features)

# Forward - attention can now choose from [real + imagined]
output = perception(state, personality, intention, sync, tick)

# The attention_weights now span both real and imagined positions
```

### Shared Generative Core

Imagination and Prediction can optionally share the same generative model:

```
                    ┌─────────────────────────┐
                    │     GenerativeCore      │
                    │                         │
                    │  ┌─────────────────┐    │
    context ──────▶ │  │    Encoder      │    │
                    │  │  (shared)       │    │
                    │  └────────┬────────┘    │
                    │           │             │
    mode ──────────▶│  ┌────────▼────────┐    │
    (predict/scene/ │  │  Mode Fusion    │    │
     mind/counter)  │  │                 │    │
                    │  └────────┬────────┘    │
                    │           │             │
                    │  ┌────────▼────────┐    │
                    │  │    Decoder      │    │ ──────▶ generated
                    │  │  (shared)       │    │         features
                    │  └─────────────────┘    │
                    │                         │
                    └─────────────────────────┘
```

**Modes:**
- `"predict"`: Prediction (what comes next)
- `"scene"`: Scene imagination (mental imagery)
- `"mind"`: Mind modeling (theory of mind)
- `"counter"`: Counterfactual generation

**Benefits of sharing:**
- Imagination improves predictions (imagined scenarios inform expectations)
- Predictions ground imagination (predictions constrain what's plausible)
- Unified world model for all generation

**Usage:**
```python
from pem import GenerativeCore, create_generative_core

# Create shared core
core = create_generative_core(d_model=1536)

# Set it on both modules
imagination_module.set_generative_core(core)
prediction_module.set_generative_core(core)

# Now both share the same generative weights!
```

---

## Files

| File | Description |
|------|-------------|
| `pem/feature_extractor.py` | Base classes and factory for feature extractors |
| `pem/janus_pro_feature_extractor.py` | **Janus Pro 1B unified feature extraction + generation** |
| `pem/prediction_module.py` | Multi-scale prediction heads |
| `pem/surprise_module.py` | Surprise and Valence modules (affective dimension) |
| `pem/curiosity_module.py` | Curiosity module (epistemic drive) |
| `pem/activation_module.py` | Activation/Arousal module (engagement intensity) |
| `pem/imagination_module.py` | Imagination module (mental simulation) |
| `pem/generative_core.py` | Shared generative model for prediction/imagination |
| `pem/sync_module.py` | SyncModule with Memory, Personality, Intention |
| `pem/perception_attention.py` | Perception attention with unified real+imagined pool |
| `pem/__init__.py` | Package exports |

---

## Janus Pro (Unified Backbone)

Janus Pro 1B is the **preferred backbone** for PEM because it provides both understanding AND generation in a unified, compact model.

### Why Janus Pro?

| Feature | Qwen3-VL (legacy) | Janus Pro 1B |
|---------|------------------|--------------|
| Understanding | ✓ | ✓ |
| Generation | ✗ (need separate model) | ✓ (CFG-based) |
| Image generation | ✗ | ✓ |
| Text-to-image | ✗ | ✓ |
| Model size | 2B | 1B/7B |
| Hidden dimension | 1536 | 2048 (projected to 1536) |

### Architecture

Janus Pro uses:
- **DeepSeek** architecture as the LLM backbone
- **Vision encoder** for image understanding
- **Autoregressive generation** with classifier-free guidance (CFG)
- **Discrete image tokens** for both understanding and generation

### Integration with Imagination

When Janus Pro is used, the ImaginationModule can leverage **native generation**:

```python
# Feature extractor with native generation
extractor = create_feature_extractor(
    model_name_or_path="deepseek-ai/Janus-Pro-1B"
)

# Imagination can use native generation
imagination_module.set_feature_extractor(extractor)

# Now imagination uses CFG-based generation!
output = imagination_module(features)
```

### Modes of Imagination

With Janus Pro, imagination has three modes (in order of preference):
1. **Native generation**: Janus Pro's CFG-based generation for true generative imagination
2. **Shared GenerativeCore**: Learned transformation shared with prediction
3. **Dedicated generators**: Independent scene/mind generators

---

## Usage Example

```python
from pem import (
    PerceptionAttention, PerceptionConfig,
    SyncModule, SyncModuleConfig,
    create_feature_extractor,
)

# Create feature extractor (Janus Pro by default)
extractor = create_feature_extractor()

# Create modules
sync_config = SyncModuleConfig(
    d_model=512,
    sync_pairs=512,
    use_intention=True,
)
sync_module = SyncModule(sync_config)

perception_config = PerceptionConfig(
    d_model=512,
    d_perception=1536,  # Janus Pro hidden size (projected)
    sync_pairs=512,
)
perception = PerceptionAttention(perception_config)

# Extract and cache features (once per input)
features = extractor(input_ids, pixel_values=images)
perception.cache_perception(features)

# Each tick: process with current mental state
for tick in range(num_ticks):
    # Get sync from neural history
    sync = sync_module(history)

    # Get personality and intention signals
    personality_signal = sync_module.personality(context)
    intention_signal, _ = sync_module.intention(context, tick)

    # Attend to perception based on mental state
    output = perception(
        state=ctm_state,
        personality_signal=personality_signal,
        intention_signal=intention_signal,
        sync=sync,
        tick=tick,
    )

    # Update state with observation
    ctm_state = update(ctm_state, output.observation)
```

---

## CTM Integration (Closed Experience Loop)

When using PEM with CTM, the experience loop is fully closed:

```python
from ctm_model import CTMLanguageModel, CTMConfig

# Enable PEM features
config = CTMConfig(
    vocab_size=32000,
    d_model=512,
    # PEM sync with memory and personality
    use_pem_sync=True,
    pem_memory_slots=100,
    pem_use_intention=True,
    # PEM perception (closes the loop)
    use_pem_perception=True,
    pem_perception_dim=1536,  # Janus Pro hidden size (projected)
    pem_perception_weight=0.5,
)

model = CTMLanguageModel(config)

# Cache perception features (once per input)
model.ctm_core.cache_perception(features)

# Forward pass with surprise (from prediction errors)
# Surprise steers attention and weights memory storage
final_state, all_states, final_sync, all_syncs = model.ctm_core(
    initial_state=initial_state,
    static_k=static_k,
    static_v=static_v,
    cos=cos,
    sin=sin,
    surprise_magnitude=surprise_magnitude,  # (B, S, 1)
    surprise_direction=surprise_direction,  # (B, S, D)
)

# The loop is now closed:
# 1. Surprise → Attention: surprise_direction steers the attention query
# 2. Observation → State: perception output blends into CTM state
# 3. Surprise → Memory: surprise_magnitude weights memory importance
```

### How It Works

At each tick boundary in CTM:

1. **Compute sync** from neural history (captures coordination patterns)
2. **Pass surprise to sync** → Memory stores chunks weighted by surprise
3. **Get personality/intention signals** from sync module
4. **Build attention query** from personality + intention + sync + **surprise**
5. **Cross-attend to perception** (cached perception features)
6. **Blend observation into state** → What we perceive changes what we think
7. **Next tick** uses updated state → Closed loop!
