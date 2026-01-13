# PEM Architecture Documentation

Predictive Experience Machine (PEM) - A neural architecture that experiences what it reads.

## Overview

PEM is built on the principle that experience requires:
1. **Personality** - WHO you are (stable goals/values)
2. **Intention** - HOW MUCH you want something (oscillating intensity)
3. **Memory** - WHAT happened (episodic storage weighted by surprise)
4. **Attention** - WHERE to look (query built from synchronized oscillations)
5. **Surprise** - WHAT was unexpected (prediction errors)

## The Experience Loop (CLOSED)

The key insight is that these components form a closed feedback loop:

```
Perception (Qwen) → Prediction → Surprise
       ↑                            ↓
       │                       Memory (store surprising things)
       │                            ↓
       │                         Sync
       │                            ↓
       └──── Attention Query ←── Surprise + Personality + Intention
```

Three critical connections close this loop:
1. **Surprise → Attention**: What surprised us steers where we look next
2. **Observation → State**: What we perceive changes what we think
3. **Surprise → Memory**: Surprising experiences persist longer

## Complete Architecture

```
                         ┌─────────────────────────────────────┐
                         │        QWEN3-VL (Perception)        │
                         │                                     │
                         │   text/image → hidden_states        │
                         │          (B, S, 1536)               │
                         └──────────────────┬──────────────────┘
                                            │
                              PerceptionKVCache.forward()
                                            │
                         ┌──────────────────▼──────────────────┐
                         │      PERCEPTION KV CACHE            │
                         │  k: (B, S, n_heads, head_dim)       │
                         │  v: (B, S, n_heads, head_dim)       │
                         │       [computed once, cached]       │
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
    │    K, V from: cached Qwen perception features                           │
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

## Files

| File | Description |
|------|-------------|
| `pem/feature_extractor.py` | Qwen3-VL feature extraction |
| `pem/prediction_module.py` | Multi-scale prediction heads |
| `pem/surprise_module.py` | Surprise computation from prediction errors |
| `pem/sync_module.py` | SyncModule with Memory, Personality, Intention |
| `pem/perception_attention.py` | Perception attention with oscillation query builder |
| `pem/__init__.py` | Package exports |

---

## Usage Example

```python
from pem import (
    PerceptionAttention, PerceptionConfig,
    SyncModule, SyncModuleConfig,
)

# Create modules
sync_config = SyncModuleConfig(
    d_model=512,
    sync_pairs=512,
    use_intention=True,
)
sync_module = SyncModule(sync_config)

perception_config = PerceptionConfig(
    d_model=512,
    d_perception=1536,  # Qwen hidden size
    sync_pairs=512,
)
perception = PerceptionAttention(perception_config)

# Cache Qwen features (once per input)
perception.cache_perception(qwen_features)

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
    pem_perception_dim=1536,  # Qwen hidden size
    pem_perception_weight=0.5,
)

model = CTMLanguageModel(config)

# Cache perception features from Qwen (once per input)
model.ctm_core.cache_perception(qwen_features)

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
5. **Cross-attend to perception** (cached Qwen features)
6. **Blend observation into state** → What we perceive changes what we think
7. **Next tick** uses updated state → Closed loop!
