# Memory Architecture

## Overview

The memory-augmented GPT system provides episodic memory that:
1. **Crystallizes** high-salience moments during training/inference
2. **Injects** relevant memories into GPT attention layers
3. **Modifies** memories based on retrieval feedback (refinement, correction, consolidation)

## 1. Saving Memories (Crystallization)

```
Input tokens → GPT forward → Hidden states → ExperientialStream
                                                    ↓
                                            Compute surprise/salience
                                                    ↓
                                            if salience > threshold:
                                                    ↓
                                            EpisodicMemory.store()
                                                    ↓
                                            Episode(content, context, salience, ...)
```

**Key code path** (`experiential.py`):
- `ExperientialStream.forward()` computes surprise from prediction error
- Salience = surprise × affect modulation
- `EpisodicMemory.store()` saves if `salience > crystallization_threshold`
- Content is the hidden state sequence `[n_tokens, d_model]`

### Episode Structure

```python
@dataclass
class Episode:
    timestamp: int              # when (global step)
    content: torch.Tensor       # [d_model] or [n_tokens, d_model]
    context: torch.Tensor       # surrounding state [d_model]
    salience: float             # importance (surprise × affect)
    valence: float              # emotional valence
    arousal: float              # arousal level
    retrieval_count: int        # access frequency
    text: Optional[str]         # human-readable text
    token_ids: Optional[List]   # raw token ids
    # Modification tracking
    harm_count: int             # times retrieval hurt
    benefit_count: int          # times retrieval helped
    cumulative_benefit: float   # sum of (weight × benefit)
    modification_count: int     # times content modified
    episode_id: int             # unique ID
    source_episode_ids: List    # merged-from tracking
```

## 2. Injecting into GPT (K/V Injection)

```
Query (from prev step) → retrieve_soft() → attention weights over all episodes
                                ↓
                        Top-k episodes selected
                                ↓
                        Episode content → K/V projection
                                ↓
                        Injected into GPT attention layers (e.g., layers n/4, n/2, 3n/4)
                                ↓
                        GPT attends to memory K/V alongside context K/V
```

**The injection happens DURING the forward pass**, not after. Memory K/V is concatenated with context K/V in selected transformer layers.

### Retrieval Mechanism

1. **Query projection**: Current hidden state → query vector
2. **Soft retrieval**: Cosine similarity + salience weighting → attention weights
3. **Gating**: Optional learned gate suppresses unhelpful retrievals
4. **Selection**: Top-k memories by attention weight
5. **Injection**: Memory content projected to K/V, concatenated in attention

## 3. Modifying During Run

```
Forward pass with memory → Compute loss_with_memory
Forward pass without memory → Compute loss_without_memory
                                    ↓
                        retrieval_benefit = loss_without - loss_with
                                    ↓
            ┌───────────────────────┼───────────────────────┐
            ↓                       ↓                       ↓
    benefit > 0              benefit < 0              periodic check
            ↓                       ↓                       ↓
    REFINE content          CORRECT content         CONSOLIDATE similar
    (EMA toward query)      (push away from query)  (merge into stronger)
```

### 3.1 Content Refinement (`--enable_content_refinement`)

When retrieval benefit > 0 (memory helped):
- Nudge episode content toward the query that found it useful
- Uses exponential moving average: `new = (1-α)·old + α·query`
- For sequences: shift all tokens uniformly to preserve structure
- Controlled by `--content_refinement_rate` (default 0.1)

### 3.2 Content Correction (`--enable_content_correction`)

When retrieval benefit < 0 (memory hurt):
- Push episode content away from the harmful query
- Track `harm_count` per episode
- Delete if `harm_count >= threshold` and `harm_count > 2×benefit_count`
- Controlled by `--content_correction_rate` (default 0.05)

### 3.3 Episodic Consolidation (`--enable_episodic_consolidation`)

Periodically merge similar, frequently-retrieved episodes:
- Trigger: cosine similarity > threshold AND both have enough retrievals
- Merge: weighted average by (salience × retrieval_count)
- Result: single stronger vector `[d_model]`
- Controlled by `--consolidation_similarity_threshold` (default 0.85)

## Full Loop Diagram

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
    [Input] → GPT + Memory Injection → [Logits] → Loss       │
                    │                              │          │
                    │                              ▼          │
                    │                    retrieval_benefit    │
                    │                              │          │
                    │         ┌────────────────────┼──────────┤
                    │         ▼                    ▼          │
                    │    if helpful:          if harmful:     │
                    │    refine content       correct content │
                    │         │                    │          │
                    │         └────────┬───────────┘          │
                    │                  ▼                      │
                    │         update salience                 │
                    │                  │                      │
                    ▼                  ▼                      │
            ExperientialStream → crystallize? ───────────────┘
                    │                  │
                    │                  ▼
                    │         EpisodicMemory.store()
                    │                  │
                    └──────────────────┘
```

## Key Files and Locations

| Component | File | Location |
|-----------|------|----------|
| Episode dataclass | `experiential.py` | line ~51 |
| Episode storage | `EpisodicMemory.store()` | line ~460 |
| Soft retrieval | `EpisodicMemory.retrieve_soft()` | line ~550 |
| K/V injection | `MemoryAugmentedGPT._inject_memory_kv()` | line ~2600 |
| Content refinement | `EpisodicMemory.apply_content_refinement()` | line ~803 |
| Content correction | `EpisodicMemory.apply_content_correction()` | line ~868 |
| Consolidation | `EpisodicMemory.consolidate_similar_episodes()` | line ~950 |
| Retrieval benefit | `apply_retrieval_benefit()` | line ~726 |

## CLI Parameters

### Memory Storage
```
--memory_capacity           Max episodes (default 1000)
--crystallization_threshold Salience threshold to store (default 0.2)
--decay_rate               Salience decay per step (default 0.01)
--min_salience             Prune below this (default 0.05)
--dedup_threshold          Reject if similarity > this (default 0.95)
```

### Memory Retrieval
```
--retrieval_temperature     Softmax temperature (default 0.1)
--retrieval_salience_weight Bias toward high-salience (default 0.0)
--cross_attention_top_k     Top-k memories to inject (default 8)
```

### Content Modification
```
--enable_content_refinement        Enable refinement on positive benefit
--content_refinement_rate          EMA rate (default 0.1)
--content_refinement_min_benefit   Min benefit to trigger (default 0.1)
--enable_content_correction        Enable correction on negative benefit
--content_correction_rate          Push-away rate (default 0.05)
--content_correction_harm_threshold Harm count for deletion (default 3)
--enable_episodic_consolidation    Enable merging similar episodes
--consolidation_similarity_threshold Cosine sim for merge (default 0.85)
--consolidation_min_retrievals     Min retrievals to merge (default 5)
--consolidation_check_interval     Steps between checks (default 100)
```

### Retrieval Benefit Training
```
--retrieval_benefit_weight         Train memory to help (default 0.1)
--retrieval_benefit_salience_weight Scale for salience updates (default 0.1)
--retrieval_gate_weight            Train retrieval gate (default 0.01)
--retrieval_gate_entropy_weight    Prevent gate collapse (default 0.01)
```
