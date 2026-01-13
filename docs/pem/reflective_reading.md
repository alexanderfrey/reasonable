# Reflective Reading Extension

A system that reads like a human: pausing to think, forming hypotheses about what happens next, and experiencing confirmation or surprise when predictions resolve.

## Core Insight

Humans don't predict the next word when reading a novel. They:
1. **Read** a chunk (paragraph, scene)
2. **Pause** to process what just happened
3. **Hypothesize** about what comes next (plot, character, theme)
4. **Continue** reading
5. **Experience** confirmation or surprise when hypotheses resolve

This is **active reading**, not passive token prediction.

## The Reflection Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                      REFLECTION CYCLE                           │
│                                                                 │
│   ┌─────────┐     ┌───────────┐     ┌──────────────┐           │
│   │  READ   │────▶│  REFLECT  │────▶│  HYPOTHESIZE │           │
│   │ (chunk) │     │ (extra    │     │  (what next?)│           │
│   └─────────┘     │  ticks)   │     └──────┬───────┘           │
│        ▲          └───────────┘            │                   │
│        │                                   ▼                   │
│        │                          ┌──────────────┐             │
│        │                          │  HYPOTHESIS  │             │
│        │                          │    MEMORY    │             │
│        │                          │ "The butler  │             │
│        │                          │  did it"     │             │
│        │                          └──────┬───────┘             │
│        │                                 │                     │
│        │          ┌───────────┐          │                     │
│        └──────────│  VERIFY   │◀─────────┘                     │
│                   │ (confirm/ │                                │
│                   │  deny)    │                                │
│                   └───────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

## Multi-Scale Hypotheses

Different predictions resolve at different time scales:

| Scale | Example Hypothesis | Confirmation Time | Surprise Impact |
|-------|-------------------|-------------------|-----------------|
| Immediate | "She's about to cry" | Next paragraph | Low |
| Scene | "They'll kiss at the end of this scene" | Few pages | Medium |
| Arc | "He'll betray her eventually" | Chapters | High |
| Story | "Good will triumph" | End of book | Very High |

The system maintains **active hypotheses** at all scales simultaneously.

## Architecture Overview

```python
class ReflectiveExperienceMachine(nn.Module):
    """
    A system that reads, reflects, hypothesizes, and verifies.

    Extends base PEM with:
    - Reflection ticks (thinking without new input)
    - Multi-scale hypothesis generation
    - Narrative state tracking
    - Hypothesis verification and surprise
    """

    def __init__(self, config):
        super().__init__()

        # Base PEM components
        self.feature_extractor = FeatureExtractor(config)
        self.sync_core = SyncCore(config)
        self.surprise_memory = SurpriseMemory(config)

        # NEW: Reflection mechanism
        self.reflection_module = ReflectionModule(config)

        # NEW: Hypothesis generation (multiple scales)
        self.hypothesis_generator = HypothesisGenerator(config)

        # NEW: Hypothesis memory (active predictions)
        self.hypothesis_memory = HypothesisMemory(config)

        # NEW: Verification module
        self.verifier = HypothesisVerifier(config)

        # NEW: Narrative state tracker
        self.narrative_state = NarrativeStateTracker(config)

    def forward(self, tokens, chunk_boundaries=None):
        """
        Process with reflection at chunk boundaries.
        """
        features = self.feature_extractor(tokens)
        chunks = self.get_chunks(tokens, chunk_boundaries)

        all_experiences = []

        for chunk_start, chunk_end in chunks:
            # 1. READ: Process chunk through sync
            chunk_features = features[:, chunk_start:chunk_end]
            sync = self.sync_core(chunk_features)

            # 2. VERIFY: Check active hypotheses against what we just read
            confirmations, denials, still_active = self.verifier(
                self.hypothesis_memory.get_active(),
                chunk_features,
                self.narrative_state.get_state()
            )
            self.hypothesis_memory.update_active(still_active)

            # Generate surprise from hypothesis outcomes
            hypothesis_surprise = self.compute_hypothesis_surprise(
                confirmations, denials
            )

            # 3. REFLECT: Extra sync ticks WITHOUT new input
            sync, inner_monologue = self.reflection_module(
                sync,
                hypothesis_surprise,
                self.narrative_state.get_state()
            )

            # 4. UPDATE NARRATIVE STATE
            self.narrative_state.update(chunk_features, sync)

            # 5. HYPOTHESIZE: Generate predictions at multiple scales
            new_hypotheses = self.hypothesis_generator(
                sync,
                self.narrative_state.get_state(),
                scales=['immediate', 'scene', 'arc', 'story']
            )
            self.hypothesis_memory.add(new_hypotheses)

            # Package experience
            experience = {
                'chunk': (chunk_start, chunk_end),
                'sync': sync,
                'inner_monologue': inner_monologue,
                'confirmed': confirmations,
                'denied': denials,
                'new_hypotheses': new_hypotheses,
                'narrative_state': self.narrative_state.get_state(),
            }
            all_experiences.append(experience)

        return sync, all_experiences
```

## Component: Reflection Module

The key addition: **thinking without new input**

```python
class ReflectionModule(nn.Module):
    """
    Extra sync ticks for reflection - no new input, just processing.

    Like pausing after reading a paragraph to think:
    "Wait, what did that mean?"
    "Oh, that connects to what happened earlier!"
    "I bet X will happen next..."
    """

    def __init__(self, config):
        super().__init__()
        self.reflection_ticks = config.reflection_ticks
        self.sync_core = SyncCore(config)

        # Reflection can attend to narrative state
        self.narrative_attn = nn.MultiheadAttention(
            config.d_model, config.n_head, batch_first=True
        )

        # Inner monologue generator (for interpretability)
        self.inner_monologue_head = nn.Sequential(
            nn.Linear(config.sync_pairs, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.vocab_size),
        )

    def forward(
        self,
        sync: Tensor,
        surprise_signal: Tensor,
        narrative_state: Dict,
    ) -> Tuple[Tensor, str]:
        """
        Reflect on what was just read.

        Args:
            sync: Current sync state after reading chunk
            surprise_signal: Surprise from hypothesis confirmations/denials
            narrative_state: Current understanding of the story

        Returns:
            sync: Updated sync after reflection
            inner_monologue: Interpretable "thoughts" (optional)
        """
        # Encode narrative state for attention
        narrative_encoding = self.encode_narrative(narrative_state)

        # Attend to narrative elements (characters, plot threads, etc.)
        narrative_context, _ = self.narrative_attn(
            query=sync,
            key=narrative_encoding,
            value=narrative_encoding
        )

        # Extra sync ticks - "thinking time"
        for tick in range(self.reflection_ticks):
            # Sync processes without new input
            # But WITH narrative context and surprise signal
            sync = self.sync_core.single_tick(
                sync,
                external_input=narrative_context,
                surprise=surprise_signal,
            )

        # Generate interpretable inner monologue
        inner_monologue = self.generate_inner_monologue(sync)

        return sync, inner_monologue

    def generate_inner_monologue(self, sync: Tensor) -> str:
        """
        Decode sync state to interpretable thoughts.

        This is optional but useful for understanding what
        the system is "thinking" during reflection.
        """
        logits = self.inner_monologue_head(sync)
        # Decode to text (simplified)
        return decode_to_text(logits)
```

## Component: Hypothesis Generator

```python
class HypothesisGenerator(nn.Module):
    """
    Generate predictions about what happens next at multiple scales.

    Not token prediction - narrative prediction:
    - What will characters do?
    - What events will occur?
    - What will be revealed?
    - How will the emotional tone shift?
    """

    def __init__(self, config):
        super().__init__()

        self.scales = {
            'immediate': HypothesisHead(config, horizon=1),    # Next chunk
            'scene': HypothesisHead(config, horizon=5),        # ~5 chunks
            'arc': HypothesisHead(config, horizon=20),         # ~20 chunks
            'story': HypothesisHead(config, horizon=100),      # Full story
        }

        self.hypothesis_types = [
            'plot_event',       # "X will happen"
            'character_action', # "Character will do Y"
            'revelation',       # "We'll learn that Z"
            'emotional_beat',   # "The mood will shift to W"
            'relationship',     # "A and B will reconcile/conflict"
        ]

        # Type classifier
        self.type_head = nn.Linear(config.d_model, len(self.hypothesis_types))

        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sync: Tensor,
        narrative_state: Dict,
        scales: List[str] = None,
    ) -> List[Dict]:
        """
        Generate hypotheses at requested scales.

        Returns structured hypotheses with:
        - embedding: Dense representation
        - scale: Time scale for resolution
        - type: What kind of prediction
        - confidence: How sure are we
        - description: Human-readable (optional)
        """
        scales = scales or list(self.scales.keys())
        hypotheses = []

        # Encode narrative state
        narrative_encoding = self.encode_narrative(narrative_state)

        # Combine sync with narrative
        combined = self.combine(sync, narrative_encoding)

        for scale_name in scales:
            head = self.scales[scale_name]

            # Generate hypothesis embedding
            hyp_embedding = head(combined)

            # Classify hypothesis type
            type_logits = self.type_head(hyp_embedding)
            hyp_type = self.hypothesis_types[type_logits.argmax(dim=-1)]

            # Estimate confidence
            confidence = self.confidence_head(hyp_embedding)

            hypothesis = Hypothesis(
                embedding=hyp_embedding,
                scale=scale_name,
                type=hyp_type,
                confidence=confidence,
                created_at=current_chunk_idx,
                expected_resolution=current_chunk_idx + head.horizon,
                description=self.to_text(hyp_embedding),  # Optional
            )
            hypotheses.append(hypothesis)

        return hypotheses

    def to_text(self, embedding: Tensor) -> str:
        """
        Convert hypothesis embedding to natural language.

        E.g., "John will reveal his secret to Mary"
        """
        # Could use a small decoder or retrieval from templates
        ...
```

## Component: Hypothesis Memory

```python
class HypothesisMemory(nn.Module):
    """
    Store and manage active hypotheses.

    Tracks:
    - Active hypotheses (not yet resolved)
    - Confirmed hypotheses (for learning)
    - Denied hypotheses (for learning from mistakes)
    """

    def __init__(self, config):
        super().__init__()
        self.max_active = config.max_active_hypotheses

        self.active: List[Hypothesis] = []
        self.confirmed: List[Hypothesis] = []
        self.denied: List[Hypothesis] = []

    def add(self, hypotheses: List[Hypothesis]):
        """Add new hypotheses to active set."""
        for hyp in hypotheses:
            if len(self.active) >= self.max_active:
                # Remove lowest confidence hypothesis
                self.active.sort(key=lambda h: h.confidence)
                self.active.pop(0)
            self.active.append(hyp)

    def get_active(self) -> List[Hypothesis]:
        """Get all active hypotheses."""
        return self.active

    def get_by_scale(self, scale: str) -> List[Hypothesis]:
        """Get active hypotheses at a specific scale."""
        return [h for h in self.active if h.scale == scale]

    def confirm(self, hypothesis: Hypothesis, match_details: Dict):
        """Move hypothesis to confirmed set."""
        self.active.remove(hypothesis)
        hypothesis.resolution = 'confirmed'
        hypothesis.match_details = match_details
        self.confirmed.append(hypothesis)

    def deny(self, hypothesis: Hypothesis, actual_outcome: Dict):
        """Move hypothesis to denied set."""
        self.active.remove(hypothesis)
        hypothesis.resolution = 'denied'
        hypothesis.actual_outcome = actual_outcome
        self.denied.append(hypothesis)

    def get_track_record(self) -> Dict:
        """Get statistics on hypothesis accuracy."""
        total = len(self.confirmed) + len(self.denied)
        if total == 0:
            return {'accuracy': 0, 'by_scale': {}, 'by_type': {}}

        return {
            'accuracy': len(self.confirmed) / total,
            'by_scale': self._accuracy_by_scale(),
            'by_type': self._accuracy_by_type(),
        }
```

## Component: Hypothesis Verifier

```python
class HypothesisVerifier(nn.Module):
    """
    Check if active hypotheses were confirmed or denied by new content.
    """

    def __init__(self, config):
        super().__init__()

        # Learned matching function
        self.matcher = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
            nn.Sigmoid(),
        )

        self.confirm_threshold = config.confirm_threshold  # e.g., 0.7
        self.deny_threshold = config.deny_threshold        # e.g., 0.3

    def forward(
        self,
        hypotheses: List[Hypothesis],
        new_content: Tensor,
        narrative_state: Dict,
        current_chunk: int,
    ) -> Tuple[List, List, List]:
        """
        Check each hypothesis against new content.

        Returns:
            confirmations: Hypotheses that were confirmed
            denials: Hypotheses that were wrong
            still_active: Hypotheses not yet resolvable
        """
        confirmations = []
        denials = []
        still_active = []

        # Encode new content
        content_encoding = self.encode_content(new_content)

        for hyp in hypotheses:
            # Check if this hypothesis should be evaluated now
            if current_chunk < hyp.expected_resolution:
                still_active.append(hyp)
                continue

            # Compute match score
            match_input = torch.cat([hyp.embedding, content_encoding], dim=-1)
            match_score = self.matcher(match_input)

            if match_score > self.confirm_threshold:
                confirmations.append({
                    'hypothesis': hyp,
                    'match_score': match_score.item(),
                    'what_confirmed': self.extract_match(hyp, new_content),
                })
            elif match_score < self.deny_threshold:
                denials.append({
                    'hypothesis': hyp,
                    'match_score': match_score.item(),
                    'what_actually_happened': self.extract_actual(new_content),
                })
            else:
                # Uncertain - extend deadline
                hyp.expected_resolution += 1
                still_active.append(hyp)

        return confirmations, denials, still_active
```

## Component: Narrative State Tracker

```python
class NarrativeStateTracker(nn.Module):
    """
    Track the evolving state of what we're reading.

    Maintains a mental model of the story world:
    - Who are the characters?
    - Where are they?
    - What relationships exist?
    - What plot threads are active?
    - What's the emotional tone?
    """

    def __init__(self, config):
        super().__init__()

        # Entity memory (characters, objects, locations)
        self.entity_memory = EntityMemory(config)

        # Relationship graph
        self.relationships = RelationshipTracker(config)

        # Active plot threads
        self.plot_threads = PlotThreadMemory(config)

        # Emotional/tonal state
        self.emotional_state = EmotionalStateTracker(config)

        # Scene context (current location, time, participants)
        self.scene_context = SceneContextTracker(config)

    def update(self, chunk_features: Tensor, sync: Tensor):
        """
        Update narrative state based on new content.
        """
        # Extract entities from chunk
        entities = self.entity_extractor(chunk_features)
        self.entity_memory.update(entities)

        # Extract and update relationships
        relations = self.relation_extractor(chunk_features)
        self.relationships.update(relations)

        # Update plot threads
        # - New threads introduced?
        # - Existing threads advanced?
        # - Any threads resolved?
        self.plot_threads.update(chunk_features, sync)

        # Update emotional state
        self.emotional_state.update(chunk_features)

        # Update scene context
        self.scene_context.update(chunk_features)

    def get_state(self) -> Dict:
        """Get complete narrative state."""
        return {
            'entities': self.entity_memory.get_all(),
            'relationships': self.relationships.get_graph(),
            'plot_threads': self.plot_threads.get_active(),
            'resolved_threads': self.plot_threads.get_resolved(),
            'emotional_state': self.emotional_state.current(),
            'scene': self.scene_context.current(),
        }

    def get_character(self, name: str) -> Dict:
        """Get everything we know about a character."""
        entity = self.entity_memory.get(name)
        relations = self.relationships.get_for_entity(name)
        involvement = self.plot_threads.get_involvement(name)

        return {
            'entity': entity,
            'relationships': relations,
            'plot_involvement': involvement,
        }
```

## Full Reading Experience

```python
def read_novel(model, novel_tokens, chunk_size=512):
    """
    Read a novel with full reflection and hypothesis tracking.

    Returns a rich log of the reading experience.
    """
    model.eval()
    chunks = list(chunk_tokens(novel_tokens, chunk_size))

    reading_log = []

    for i, chunk in enumerate(chunks):
        # Process chunk
        sync, experience = model.process_chunk(chunk)

        # Log the experience
        log_entry = {
            'chunk_idx': i,
            'text': decode(chunk),

            # Inner experience
            'inner_monologue': experience['inner_monologue'],
            'sync_state': experience['sync'].detach(),

            # Hypothesis tracking
            'confirmed_predictions': [
                {
                    'prediction': h['hypothesis'].description,
                    'confidence': h['hypothesis'].confidence,
                    'match_score': h['match_score'],
                }
                for h in experience['confirmed']
            ],
            'wrong_predictions': [
                {
                    'prediction': h['hypothesis'].description,
                    'confidence': h['hypothesis'].confidence,
                    'what_happened': h['what_actually_happened'],
                }
                for h in experience['denied']
            ],
            'new_predictions': [
                {
                    'scale': h.scale,
                    'type': h.type,
                    'prediction': h.description,
                    'confidence': h.confidence,
                }
                for h in experience['new_hypotheses']
            ],

            # Narrative state
            'characters_present': experience['narrative_state']['scene']['participants'],
            'active_plot_threads': [
                t.description for t in experience['narrative_state']['plot_threads']
            ],
            'emotional_tone': experience['narrative_state']['emotional_state'],

            # Metrics
            'surprise_level': compute_surprise(
                experience['confirmed'], experience['denied']
            ),
            'prediction_accuracy': model.hypothesis_memory.get_track_record()['accuracy'],
        }

        reading_log.append(log_entry)

        # Print experience (for visualization)
        print(f"\n{'='*60}")
        print(f"CHUNK {i}")
        print(f"{'='*60}")
        print(f"\nText: {decode(chunk)[:200]}...")
        print(f"\nInner Monologue: {experience['inner_monologue']}")

        if experience['confirmed']:
            print(f"\n✓ Confirmed predictions:")
            for h in experience['confirmed']:
                print(f"  - {h['hypothesis'].description}")

        if experience['denied']:
            print(f"\n✗ Wrong predictions:")
            for h in experience['denied']:
                print(f"  - Predicted: {h['hypothesis'].description}")
                print(f"    Actually: {h['what_actually_happened']}")

        print(f"\n→ New predictions:")
        for h in experience['new_hypotheses']:
            print(f"  [{h.scale}] {h.description} (conf: {h.confidence:.2f})")

    return reading_log
```

## Example Output

```
============================================================
CHUNK 47
============================================================

Text: "I need to tell you something," John said, his voice trembling.
Mary set down her cup. "What is it?" John took a deep breath...

Inner Monologue: The tension is peaking. John's body language signals
                 confession. Mary's calm response suggests she may
                 already suspect something.

✓ Confirmed predictions:
  - Mary would confront John about the letter (conf: 0.82)

✗ Wrong predictions:
  - Predicted: John would deny the affair
    Actually: John confessed immediately

→ New predictions:
  [immediate] Mary will leave the room (conf: 0.71)
  [scene] They'll have an emotional confrontation (conf: 0.89)
  [arc] Their relationship will fundamentally change (conf: 0.94)
  [story] This confession will be referenced in the climax (conf: 0.67)

Active plot threads:
  - John's secret (RESOLVING)
  - Mary's suspicions (CONFIRMED)
  - The inheritance dispute (BACKGROUND)

Emotional tone: tense → cathartic
Surprise level: 0.73 (high - unexpected confession)
Overall prediction accuracy: 67%
```

## Training Considerations

### Hypothesis Quality Loss
```python
def hypothesis_loss(confirmed, denied):
    """
    Learn to make good hypotheses.

    - High confidence + confirmed = good (reward)
    - High confidence + denied = bad (penalize)
    - Low confidence + either = okay (uncertainty is fine)
    """
    loss = 0
    for h in confirmed:
        # Reward confident correct predictions
        loss -= h['hypothesis'].confidence * h['match_score']

    for h in denied:
        # Penalize confident wrong predictions
        loss += h['hypothesis'].confidence * (1 - h['match_score'])

    return loss
```

### Reflection Depth Loss
```python
def reflection_loss(pre_reflection_sync, post_reflection_sync, outcome):
    """
    Reflection should improve predictions.

    If reflection helps, it should change sync meaningfully.
    If reflection doesn't help, maybe fewer ticks needed.
    """
    sync_change = (post_reflection_sync - pre_reflection_sync).norm()

    # If outcome was surprising, reflection should have changed things
    if outcome.surprising:
        return -sync_change  # Encourage change
    else:
        return sync_change * 0.1  # Mild pressure for stability
```

## Open Questions

1. **Chunk size**: How much to read before reflecting?
2. **Hypothesis language**: Embeddings only, or natural language?
3. **Verification timing**: When to check hypotheses?
4. **Narrative extraction**: How to reliably extract characters, relations, etc.?
5. **Training data**: Need annotated "reading experiences"?
6. **Evaluation**: How to measure "good reading"?

## Relationship to Base PEM

| Base PEM | Reflective Extension |
|----------|---------------------|
| Token-level prediction | Narrative-level prediction |
| Immediate surprise | Multi-scale surprise |
| Simple memory | Hypothesis memory + narrative state |
| Continuous processing | Chunked read-reflect cycles |
| Feature comparison | Hypothesis verification |
