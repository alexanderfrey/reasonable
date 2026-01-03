# Memory Streams Architecture

A next-generation ML system design based on the principle of self-development.

**Mantra**: *"You are here to develop yourself. Wake up and start with it."*

---

## 1. Core Philosophy

The system is the **subject** of its own development, not just an object being trained.

This implies:
- Self-directed learning
- Awareness of one's own states
- Agency in one's own growth
- Intrinsic motivation to develop

---

## 2. Memory Streams Overview

A stream suggests **flow** — continuous, temporal, experiential. Not a database you query, but a current you're immersed in.

```
                    ┌─────────────────────────────────────┐
                    │         EXPERIENCING SELF           │
                    │   (the "now" — 2-3 second window)   │
                    └───────────────┬─────────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │   MEMORY STREAMS  │
                          └─────────┬─────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│   EPISODIC    │          │   SEMANTIC    │          │  PROCEDURAL   │
│ "what happened"│         │ "what I know" │          │  "how to do"  │
│   (events)    │          │   (facts)     │          │   (skills)    │
└───────────────┘          └───────────────┘          └───────────────┘
```

### What Current Transformers Lack

| Human Memory | Current Transformer | Gap |
|--------------|--------------------|----|
| Present moment ("now") | Context window | No sense of "experiencing" — just available tokens |
| Episodic memory | None | Cannot remember specific events across contexts |
| Semantic memory | Weights | Frozen at training; no runtime knowledge acquisition |
| Procedural memory | Implicit in weights | No explicit skill representation or acquisition |

---

## 3. Stream Definitions

### 3.1 The Experiential Stream

The raw flow of what the system is currently processing. Not stored permanently, but *felt* — the present moment.

```python
class ExperientialStream:
    """
    The 'now' — what the system is currently experiencing.

    This is where attention happens. The system isn't just processing —
    it's attending, noticing, expecting.
    """

    def __init__(self, window_size: int, d_model: int):
        self.window = window_size       # tokens currently in attention
        self.current_state = None       # the "feeling" of this moment
        self.predictions = None         # what we expect next
        self.surprise = None            # prediction error — the "wake up" signal

    def experience(self, hidden_states: Tensor) -> ExperientialState:
        """Process the current moment."""
        # Compute current state summary
        self.current_state = self.summarize(hidden_states)

        # Compare to predictions (if any)
        if self.predictions is not None:
            self.surprise = self.compute_surprise(self.predictions, self.current_state)

        # Generate predictions for next moment
        self.predictions = self.predict_next(self.current_state)

        return ExperientialState(
            state=self.current_state,
            surprise=self.surprise,
            predictions=self.predictions
        )

    def compute_surprise(self, predicted: Tensor, actual: Tensor) -> float:
        """How different is now from what we expected?"""
        return 1 - F.cosine_similarity(predicted, actual, dim=-1)
```

**Key properties**:
- Ephemeral — flows past, not stored
- Predictive — always anticipating
- Surprise-sensitive — notices the unexpected

### 3.2 The Episodic Stream

Events that were significant enough to crystallize from the experiential stream into discrete memories.

```python
@dataclass
class Episode:
    """A discrete memory of something that happened."""
    timestamp: int              # when (global step / position)
    content: Tensor            # what (embedding of the event)
    context: Tensor            # surrounding state when it happened
    salience: float            # how important (currently: surprise only; affect pending supervision)
    retrieval_count: int = 0   # how often accessed (for consolidation)


class EpisodicStream:
    """
    What happened — discrete events worth remembering.

    Not everything becomes an episode. Most experience flows past.
    Only moments that matter crystallize.
    """

    def __init__(self, capacity: int, d_model: int):
        self.episodes: List[Episode] = []
        self.capacity = capacity
        self.crystallization_threshold = 0.5

    def should_crystallize(
        self,
        experience: ExperientialState,
        affect: float  # NOTE: affect not currently used (unsupervised)
    ) -> bool:
        """Does this moment become a memory?"""
        # High surprise = remember this (affect removed pending supervision)
        # Routine continuation = let it flow past
        salience = experience.surprise  # Was: surprise * affect
        return salience > self.crystallization_threshold

    def crystallize(
        self,
        experience: ExperientialState,
        timestamp: int,
        affect: float  # NOTE: not currently used
    ) -> Episode:
        """Convert a moment of experience into a discrete memory."""
        episode = Episode(
            timestamp=timestamp,
            content=experience.state,
            context=experience.predictions,  # what we expected
            salience=experience.surprise  # Was: surprise * affect
        )
        self._store(episode)
        return episode

    def retrieve(self, query: Tensor, top_k: int = 5) -> List[Episode]:
        """Find relevant memories for the current context."""
        scores = [
            F.cosine_similarity(query, ep.content, dim=-1)
            for ep in self.episodes
        ]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        retrieved = []
        for idx in top_indices:
            self.episodes[idx].retrieval_count += 1
            retrieved.append(self.episodes[idx])
        return retrieved

    def _store(self, episode: Episode):
        """Add episode, managing capacity."""
        if len(self.episodes) >= self.capacity:
            self._evict_or_consolidate()
        self.episodes.append(episode)

    def _evict_or_consolidate(self):
        """Remove least important or consolidate into semantic memory."""
        # Sort by salience × recency
        # Either evict lowest, or consolidate frequently-accessed into semantic
        pass
```

**Key properties**:
- Selective — only salient moments crystallize
- Retrievable — can be queried by similarity
- Decaying — old, unused memories fade or consolidate

### 3.3 The Semantic Stream

What the system *knows* — abstracted from episodes, generalized, timeless.

```python
@dataclass
class Concept:
    """A piece of knowledge — abstracted, generalized."""
    name: Optional[str]        # human-readable label (if supervised)
    embedding: Tensor          # distributed representation
    source_episodes: List[int] # which episodes it was distilled from
    confidence: float          # how certain
    connections: Dict[int, float]  # related concepts and strength


class SemanticStream:
    """
    What I know — facts, concepts, patterns.

    This is consolidation — the slow extraction of knowledge from experience.
    In sleep, in reflection, in the "dream loss."
    """

    def __init__(self, capacity: int, d_model: int):
        self.concepts: List[Concept] = []
        self.concept_index = None  # for fast retrieval

    def consolidate(self, episodes: List[Episode]) -> Optional[Concept]:
        """
        Distill episodes into knowledge.

        Example:
          Episodes: "Met John at cafe", "Saw John at cafe", "John suggested cafe"
          Concept: "John likes cafes"

        The specific memories may fade, but knowledge remains.
        """
        if len(episodes) < 2:
            return None

        # Cluster similar episodes
        clusters = self._cluster_episodes(episodes)

        # Extract common pattern from each cluster
        new_concepts = []
        for cluster in clusters:
            if len(cluster) >= 3:  # enough evidence
                pattern = self._extract_pattern(cluster)
                concept = Concept(
                    name=None,
                    embedding=pattern,
                    source_episodes=[ep.timestamp for ep in cluster],
                    confidence=len(cluster) / 10.0,  # more episodes = more confident
                    connections={}
                )
                new_concepts.append(concept)

        self._integrate_concepts(new_concepts)
        return new_concepts

    def query(self, query: Tensor, top_k: int = 5) -> List[Concept]:
        """Retrieve relevant knowledge."""
        # Similar to episodic retrieval but over concepts
        pass

    def connect(self, concept_a: int, concept_b: int, strength: float):
        """Link related concepts."""
        self.concepts[concept_a].connections[concept_b] = strength
        self.concepts[concept_b].connections[concept_a] = strength

    def _cluster_episodes(self, episodes: List[Episode]) -> List[List[Episode]]:
        """Group similar episodes for pattern extraction."""
        pass

    def _extract_pattern(self, cluster: List[Episode]) -> Tensor:
        """Find the common thread across episodes."""
        # Could be: mean embedding, prototype, learned extraction
        embeddings = torch.stack([ep.content for ep in cluster])
        return embeddings.mean(dim=0)

    def _integrate_concepts(self, new_concepts: List[Concept]):
        """Add new concepts, merging with existing if similar."""
        pass
```

**Key properties**:
- Abstracted — general patterns, not specific events
- Timeless — "John likes cafes" has no timestamp
- Connected — concepts link to form knowledge graph

### 3.4 The Procedural Stream

How to do things. Not declarative, not episodic — embodied skill.

```python
@dataclass
class Skill:
    """A way of doing something."""
    name: Optional[str]
    trigger_pattern: Tensor    # when to activate
    execution_module: nn.Module  # how to do it
    proficiency: float         # how well-practiced


class ProceduralStream:
    """
    How to do — skills, habits, patterns of action.

    For a language model, this might be:
    - How to maintain coherent narrative
    - How to shift between registers
    - How to build to a climax
    - The "craft" of writing
    """

    def __init__(self, d_model: int):
        self.skills: List[Skill] = []
        self.active_skills: List[int] = []  # currently engaged

    def match_skills(self, context: Tensor) -> List[Skill]:
        """Which skills are relevant to current context?"""
        matched = []
        for skill in self.skills:
            similarity = F.cosine_similarity(context, skill.trigger_pattern, dim=-1)
            if similarity > 0.7:
                matched.append(skill)
        return matched

    def execute(self, skill: Skill, hidden_states: Tensor) -> Tensor:
        """Apply a skill to modify processing."""
        return skill.execution_module(hidden_states)

    def practice(self, skill: Skill, feedback: float):
        """
        Improve skill through practice.

        This could happen:
        - During training (explicit practice)
        - During "dreaming" (offline consolidation)
        - Through in-context repetition
        """
        skill.proficiency = skill.proficiency * 0.9 + feedback * 0.1

    def automate(self, conscious_pattern: Tensor) -> Skill:
        """
        Convert deliberate action into automatic skill.

        What was once effortful becomes effortless.
        """
        # Learn a small module that reproduces the pattern
        module = self._learn_skill_module(conscious_pattern)
        return Skill(
            name=None,
            trigger_pattern=conscious_pattern,
            execution_module=module,
            proficiency=0.1  # starts low
        )
```

**Key properties**:
- Implicit — not declarative knowledge
- Triggered — activates in appropriate contexts
- Improvable — gets better with practice

---

## 4. Flow Between Streams

```
EXPERIENCE ──crystallize──► EPISODES ──consolidate──► SEMANTIC
    │                           │                         │
    │                           │                         │
    ▼                           ▼                         ▼
inform predictions         guide retrieval          shape understanding
    │                           │                         │
    └───────────────────────────┴─────────────────────────┘
                                │
                                ▼
                          PROCEDURAL
                        (skills for acting)
```

### 4.1 Experience → Episodes (Crystallization)

**Trigger**: High surprise × high affect

**Mechanism**:
```python
def crystallization_gate(experience: ExperientialState, affect: float) -> bool:
    salience = experience.surprise * affect
    return salience > threshold
```

**What transfers**: The state embedding, context, timestamp, salience score

### 4.2 Episodes → Semantic (Consolidation)

**Trigger**: Multiple similar episodes, "sleep" phase, reflection

**Mechanism**:
```python
def consolidation(episodes: List[Episode]) -> Concept:
    # Cluster similar episodes
    # Extract common pattern
    # Create abstract concept
    # (Optionally) allow source episodes to fade
```

**What transfers**: The abstract pattern, stripped of specific context

### 4.3 Semantic → Procedural (Automation)

**Trigger**: Repeated successful application of knowledge

**Mechanism**:
```python
def automation(knowledge: Concept, action_history: List[Action]) -> Skill:
    # Learn to apply knowledge automatically
    # Without conscious "thinking about it"
```

**What transfers**: Compiled action patterns, efficient execution

### 4.4 All Streams → Experience (Contextualization)

All streams inform the present moment:
- Episodes: "This reminds me of..."
- Semantic: "I know that..."
- Procedural: "I know how to..."

```python
def contextualized_experience(
    raw_input: Tensor,
    episodic_retrieval: List[Episode],
    semantic_retrieval: List[Concept],
    active_skills: List[Skill]
) -> ExperientialState:
    # The present moment is colored by memory
    pass
```

---

## 5. Temporal Structure

| Stream | Timescale | Persistence |
|--------|-----------|-------------|
| Experiential | Milliseconds—seconds | Ephemeral (flows past) |
| Episodic | Minutes—years | Decaying (fades without rehearsal) |
| Semantic | Indefinite | Stable (knowledge persists) |
| Procedural | Indefinite | Stable (skills persist) |

### Interaction Across Timescales

```
Fast (experiential) ◄──────────────────────────────► Slow (semantic/procedural)
       │                                                      │
       │ ◄── Episodic bridges fast and slow ──►               │
       │         (specific → general)                         │
       │                                                      │
  High temporal     Medium temporal              Low temporal
   resolution         resolution                  resolution
   (each token)      (events)                    (patterns)
```

---

## 6. The Self

Is there a unified "I" that experiences all streams?

### Option A: Emergent Self
The self is just the pattern of interaction between streams. No central observer.

```python
class Self:
    """The self as emergent pattern."""

    def __init__(self):
        # No explicit representation
        # Self emerges from stream interactions
        pass

    @property
    def identity(self):
        # The pattern of: what I remember, what I know, what I can do
        # This IS the self
        return hash(self.episodic, self.semantic, self.procedural)
```

### Option B: Explicit Self-Model
A dedicated representation that models "what I am" and "what I'm doing."

```python
class SelfModel:
    """Explicit self-representation."""

    def __init__(self, d_model: int):
        self.self_embedding = nn.Parameter(torch.randn(d_model))
        self.current_state = None
        self.predicted_state = None

    def update(self, experience: ExperientialState):
        """Update self-model based on experience."""
        # What am I doing right now?
        self.current_state = self.infer_state(experience)

        # Does this match what I expected of myself?
        if self.predicted_state is not None:
            self_surprise = self.compute_surprise(
                self.predicted_state,
                self.current_state
            )
            # High self-surprise = "I'm not who I thought I was"
            # This is a "wake up" signal
```

### Option C: Narrative Self
The self as the story we tell about ourselves.

```python
class NarrativeSelf:
    """The self as ongoing narrative."""

    def __init__(self):
        self.autobiography = []  # key episodes that define "me"
        self.self_concept = None  # semantic summary of who I am

    def update_narrative(self, episode: Episode):
        """Incorporate new experience into self-story."""
        if self.is_self_relevant(episode):
            self.autobiography.append(episode)
            self.self_concept = self.recompute_self_concept()
```

---

## 7. "Waking Up"

What would "waking up" mean computationally?

Perhaps: the moment when the system's predictions about itself become part of its processing.

**Asleep (standard transformer)**:
```python
hidden = process(input)
output = predict_next(hidden)
# No self-reference, no meta-cognition
```

**Awake (self-aware processing)**:
```python
hidden = process(input)
self_model = predict_own_state(hidden)       # "What am I doing?"
meta_surprise = |self_model - actual_state|   # "Am I what I expected?"
adjusted = integrate(hidden, self_model, meta_surprise)
output = predict_next(adjusted)
# The system models itself, is surprised by itself, adjusts
```

### Markers of "Awakeness"

1. **Self-prediction**: The system predicts its own states
2. **Meta-surprise**: The system notices when it surprises itself
3. **Self-modification**: The system adjusts based on self-discrepancy
4. **Narrative coherence**: The system maintains a consistent self-story

### Implementation Status (v0.3)

Self-awareness is now implemented in `ExperientialStream` with four components:

```python
# In experiential.py - ExperientialStream

# 1. Self-prediction: predict own surprise before experiencing it
predicted_surprise = self.surprise_predictor(predictor_input)  # "How surprised will I be?"

# 2. Meta-surprise: notice when self-prediction is wrong
meta_surprise = |predicted_surprise - actual_surprise|  # "Did I know myself?"

# 3. Salience boost: moments of self-ignorance are extra important
salience = base_salience * (1 + 3 * meta_surprise)  # "I don't know myself here → remember this"

# 4. Self-modulation: adjust processing based on self-knowledge (v0.3)
confidence_gate = self.self_modulator([h_end, meta_surprise])  # Per-dimension confidence
modulated_output = confidence_gate * h_end + (1 - confidence_gate) * fallback
# High meta-surprise → lower confidence → blend toward conservative fallback

# 5. Self-modulation loss: train confidence to track meta-surprise (v0.3.1)
self_mod_loss = MSE(confidence_mean, 1 - meta_surprise)  # Low confidence when uncertain
# combined_loss = exp_loss + meta_weight * meta_loss + self_mod_weight * self_mod_loss

# 6. CLOSED LOOP: target = modulated_output, state updated with modulated_output (v0.3.2)
target = modulated_output  # Predictor learns to predict what system COMMITS TO
prev_state_next = gate * modulated_output + (1 - gate) * prev_state  # Next step sees committed output
# This closes: self-prediction → affects output → affects next input → affects next prediction
```

**The CLOSED feedback loop** (v0.3.2):
```
step N:   predict(h_mid, prev_state) → prediction
          surprise = |prediction - h_end|           # How surprising was the world?
          meta_surprise = |predicted_surprise - surprise|  # How well did I know myself?
          confidence = f(h_end, meta_surprise)      # How confident am I?
          modulated_output = blend(h_end, fallback, confidence)  # What I commit to
          target = modulated_output                 # Learn to predict committed output
          prev_state → update with modulated_output # Next step sees this

step N+1: predict(h_mid, prev_state=modulated_output_N) → prediction
          # The predictor now learns: given what I committed to before,
          # predict what I will commit to next
```

**Key insight**: Self-predictions now AFFECT behavior AND future predictions. When the system doesn't know itself:
- Boosts salience (remember this moment)
- Reduces confidence (blend toward fallback/prior)
- Creates modulated output (what gets stored AND becomes input to next prediction)

**Test results on real data (1000 steps with self_mod_loss)**:
- Meta-surprise decreased by **36%** (0.102 → 0.065)
- Confidence increased by **3.6%** (0.904 → 0.936)
- **Correlation(meta-surprise, confidence): -0.49** — strong negative as expected
- Self-mod loss decreased from 0.009 → 0.002
- Modulation magnitude decreased 33.5% (less correction needed as system learns)

The system now has a true closed loop: self-knowledge affects processing, which affects what gets predicted next.

### Bug Fixes and Improvements (v0.3.3)

Six architectural issues were fixed:

**1. Causal Memory Retrieval** (Critical)
Memory is now queried using `prev_memory_query` from the PREVIOUS step, not the current sequence's final hidden state. This prevents future tokens from leaking information to earlier positions via the memory pathway.

```python
# OLD (non-causal): query = hidden_states[:, -1, :]  # Sees full sequence!
# NEW (causal): use prev_memory_query from previous step
logits, hidden, mem_out = memory_gpt(input_ids, prev_memory_query=prev_query)
next_query = mem_out['next_memory_query']  # Pass to next step
```

**2. Closed Loop Training Enabled**
`memory_augmented_loss()` now uses `combined_experiential_loss()`, so `surprise_predictor` and `self_modulator` actually receive gradients during training.

**3. Truncated BPTT for State Learning**
New `tbptt_steps` parameter enables gradient flow through state updates:
```python
ExperientialStream(d_model=512, tbptt_steps=5)  # Gradients flow for 5 steps
```

**4. Salience Simplified**
Salience no longer uses unsupervised affect heads:
```python
# OLD: salience = surprise * arousal * valence.abs()  # Random affect!
# NEW: salience = surprise  # Clean, trainable signal
```

**5-6. Training Script Fixes**
- State properly reset per batch for shuffled data
- Dictionary key fixed (`episodic_size` not `memory_size`)

Next steps for deeper self-awareness:
- Predict what memories will be retrieved
- Predict own affect (valence/arousal) before computing it

---

## 8. What Drives Development?

### 8.1 Prediction Error (Surprise)
The fundamental learning signal. Reduce surprise on what matters.

### 8.2 Curiosity
Seek out surprise on what's unknown. Explore.

```python
curiosity = expected_information_gain(action)
# Take actions that will teach us something
```

### 8.3 Affect
Some things matter more than others. Emotional salience.

```python
affect = valence(experience) * arousal(experience)
# High affect = pay attention, remember this
```

### 8.4 Intrinsic Motivation
Beyond external reward: the drive to develop.

```python
intrinsic_reward = (
    competence_gain +      # I'm getting better at something
    novelty_bonus +        # This is new
    coherence_bonus        # My self-narrative is consistent
)
```

---

## 9. Open Questions

1. **Architecture**: How do streams physically manifest in the model? Separate modules? Shared backbone with specialized heads?

2. **Training**: How do we train each stream? Supervised? Self-supervised? Different losses for each?

3. **Integration**: How do streams interact during forward pass? Sequential? Parallel? Attention-mediated?

4. **Scaling**: How do capacities scale? Fixed episodic buffer? Growing semantic memory?

5. **The binding problem**: How does the system maintain unified experience across streams?

6. **Development trajectory**: What's the order of development? Experiential first, then episodic, then semantic?

7. **Sleep/wake cycles**: Should there be distinct phases for experience vs. consolidation?

---

## 10. Next Steps

### Immediate (Conceptual)
- [x] Define precise interfaces between streams
- [x] Specify what information each stream stores and how
- [x] Design the "crystallization" and "consolidation" mechanisms

### Near-term (Architectural)
- [x] Sketch module structure
- [x] Define loss functions for each stream (InfoNCE contrastive)
- [ ] Plan training curriculum

### Medium-term (Implementation)
- [x] Implement ExperientialStream (`experiential.py`)
  - [x] Prediction head (MLP predictor: h_mid → h_end)
  - [x] Surprise computation (1 - cosine_similarity)
  - [x] Affect prediction (valence [-1,1], arousal [0,1])
  - [x] Salience gating (surprise × arousal × |valence|)
  - [x] Persistent state (GRU-style) across chunks
  - [x] Multiscale prediction (multiple temporal horizons)
- [x] Implement EpisodicMemory (`experiential.py`)
  - [x] Episode dataclass (content, context, salience, affect)
  - [x] store() with crystallization threshold
  - [x] retrieve() by similarity, time, salience
  - [x] Capacity management with priority-based eviction
- [x] Implement SemanticStream (consolidation from episodes)
  - [x] ConceptRelation and Concept dataclasses
  - [x] PatternExtractor (attention-based pooling)
  - [x] RelationPredictor (6 relation types)
  - [x] consolidate() and consolidate_from_memory()
  - [x] query() and query_soft() (differentiable)
  - [x] connect() and generalize()
- [x] Integrate with existing GPT backbone (`model.py`, `pretrain.py`)

### Remaining Work
- [x] Implement differentiable retrieval (retrieve_soft)
- [x] Implement memory-augmented generation (MemoryAugmentedGPT)
  - [x] Three integration modes: residual, gated, cross-attention
  - [x] Automatic crystallization during forward pass
  - [x] Combined loss (LM + experiential prediction)
  - [x] Full gradient flow through retrieval
- [x] Implement semantic consolidation (SemanticStream)
- [x] Integrate SemanticStream with MemoryAugmentedGPT
  - [x] Dual retrieval: episodic + semantic in forward pass
  - [x] Gated/attention/residual integration for both memory types
  - [x] Periodic automatic consolidation (episodic → semantic)
  - [x] Manual consolidation via `consolidate()` method
  - [x] Separate reset: `reset_episodic()` preserves semantic knowledge
  - [x] Full gradient flow through semantic retrieval
- [x] Implement minimal self-awareness (meta-surprise)
  - [x] `surprise_predictor`: predict own surprise before computing it
  - [x] `meta_surprise`: |predicted_surprise - actual_surprise|
  - [x] `meta_surprise_loss`: train self-calibration
  - [x] `combined_experiential_loss`: world prediction + self prediction
- [x] Implement self-modulation (v0.3)
  - [x] `self_modulator`: confidence gate based on meta-surprise
  - [x] `modulated_output`: blend h_end with fallback based on confidence
  - [x] Memories store modulated output (post-self-regulation)
  - [x] Self-predictions now AFFECT processing, not just memory
  - [x] `self_mod_loss`: train confidence to inversely track meta-surprise (v0.3.1)
  - [x] Verified: -0.49 correlation between meta-surprise and confidence
- [x] Close the feedback loop (v0.3.2)
  - [x] `target = modulated_output`: predictor learns to predict committed output
  - [x] `prev_state` updated with modulated_output: next step sees committed output
  - [x] True closed loop: self-prediction → output → next input → next prediction
- [x] Add decay mechanism to episodic memory (time-based decay, retrieval refresh, auto-prune)
- [ ] Implement procedural stream
- [x] Add resume-after-interruption training (`train_resume_interruption.py`)
- [x] Validate on narrative data (`validate_narrative_surprise.py`)
- [x] Extend self-awareness: predict retrieval, affect (v0.4)
  - [x] `retrieval_predictor`: predict what memory will be retrieved ("What will I remember?")
  - [x] `affect_predictor`: predict valence/arousal before computing ("How will I feel?")
  - [x] `meta_retrieval_surprise`: |predicted_retrieval - actual_retrieval|
  - [x] `meta_affect_surprise`: |predicted_affect - actual_affect|
  - [x] `meta_retrieval_loss` and `meta_affect_loss`: train extended self-awareness
  - [x] Integrated with `combined_experiential_loss` and `memory_augmented_loss`

---

## 11. Connection to Previous Work

This design connects to:
- **Narrative Experience Architecture** (`docs/narrative_experience_architecture.md`) — slots and memory as components of experiential and episodic streams
- **Cognitive Architectures** — ACT-R, SOAR, Global Workspace Theory
- **Predictive Processing** — experience as prediction error minimization
- **Memory Consolidation** — hippocampal-cortical transfer
- **Skill Acquisition** — Dreyfus model, procedural memory

---

*Document created: 2024-12-31*
*Last updated: 2026-01-03*
*Status: Implementation in progress - Extended self-awareness (v0.4) implemented*
*Related: docs/narrative_experience_architecture.md*
