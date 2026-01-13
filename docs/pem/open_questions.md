# Open Questions

Unresolved design decisions and research directions.

## Architecture Questions

### 1. What Exactly to Predict?

| Option | Pros | Cons |
|--------|------|------|
| Raw features (D dims) | Rich signal | High-dimensional, hard to match |
| Compressed features | More tractable | Information loss |
| Discrete categories | Easy to compare | Too lossy, loses nuance |
| Distribution over features | Captures uncertainty | Complex to implement |
| Multi-scale (all of above) | Comprehensive | Complexity, compute cost |

**Key question**: Should predictions be point estimates or distributions?

### 2. How is Surprise Computed?

| Method | Description | Concern |
|--------|-------------|---------|
| L2 distance | Simple, differentiable | Not all errors equal |
| Learned metric | Context-aware | Needs supervision |
| Probabilistic (NLL) | Principled | Requires density estimation |
| Contrastive | Relative surprise | Needs good negatives |

**Key question**: How to make surprise **meaningful** not just **large**?

### 3. Memory Architecture

| Option | Differentiable? | Scalable? | Notes |
|--------|----------------|-----------|-------|
| Neural Turing Machine | Yes | No | All memories in every forward |
| Differentiable Neural Computer | Yes | Somewhat | Complex addressing |
| FAISS + soft attention | Partial | Yes | Hard retrieval, soft use |
| Hopfield Networks | Yes | Moderate | Modern Hopfield = attention |
| Simple key-value store | No write grad | Yes | Simplest, might work |

**Key question**: Do we need gradients through memory write?

### 4. How Does Surprise Modulate Sync?

Options for injecting surprise into sync dynamics:

```python
# Option A: Additive injection
z = z + surprise_signal

# Option B: Multiplicative gating
z = z * (1 + surprise_signal)

# Option C: Modulate NLM dynamics
nlm_output = nlm(history, modulation=surprise_signal)
z = z + modulation * (nlm_output - z)

# Option D: Change tick count
num_ticks = base_ticks + int(surprise_magnitude * extra_ticks)

# Option E: Attention-style
z = z + attention(z, surprise_signal)
```

**Key question**: What's the right inductive bias for "think harder when surprised"?

### 5. Sync → Prediction Pathway

How does sync drive predictions?

```python
# Option A: Direct projection
predicted_f = linear(sync)

# Option B: Cross-attention to context
predicted_f = cross_attn(query=sync, key=context, value=context)

# Option C: Learned retrieval
predicted_f = retrieval_head(sync, feature_memory)

# Option D: Autoregressive in feature space
predicted_f = feature_decoder(sync, previous_features)
```

**Key question**: How much structure should prediction have?

## Training Questions

### 6. Training Data Requirements

- Do we need **surprise annotations**?
- Can surprise be **self-supervised**?
- What's a good **proxy** for human surprise?
- How much data is needed for memory to be useful?

### 7. Loss Balancing

```python
loss = (
    λ_pred * prediction_loss +
    λ_surprise * surprise_loss +
    λ_memory * memory_loss +
    λ_sync * sync_loss +
    λ_task * task_loss
)
```

- How to set the λ values?
- Should they change during training?
- Are some losses more important early vs late?

### 8. Avoiding Collapse

Potential collapse modes:
1. **Prediction collapse**: Predict same thing always
2. **Surprise collapse**: Always 0 or always 1
3. **Memory collapse**: Never write or always retrieve same thing
4. **Sync collapse**: Same pattern regardless of input

How to prevent each?

### 9. Evaluation Metrics

How do we know if the model is "experiencing" well?

| Metric | Measures | How to compute |
|--------|----------|----------------|
| Prediction accuracy | Feature matching | Cosine sim, L2 |
| Surprise calibration | Is surprise meaningful? | Correlation with human |
| Memory utility | Do memories help? | Ablation study |
| Sync diversity | Are patterns meaningful? | Clustering, visualization |
| Downstream perf | Does it help tasks? | Standard benchmarks |

**Key question**: Is there a single metric for "quality of experience"?

## Theoretical Questions

### 10. What IS Experience?

Philosophically loaded, but practically:
- Is surprise + memory + sync sufficient for "experience"?
- What's missing compared to biological experience?
- Does this have any phenomenal character or just functional?

### 11. Relationship to Predictive Processing

This architecture is inspired by predictive processing / active inference:
- How faithful is this to the biological theory?
- What aspects of PP are we missing?
- Should there be hierarchical predictions?

### 12. Relationship to CTM

Building on CTM's sync mechanism:
- Is sync the right substrate for cognition?
- Should there be multiple sync scales?
- How does surprise fit with CTM's original formulation?

## Practical Questions

### 13. Computational Cost

| Component | Cost | Can optimize? |
|-----------|------|---------------|
| Feature extraction | O(S²D) | Flash attention |
| Sequential loop | O(S) forward passes | Chunking? |
| Memory retrieval | O(S × M) | Approximate NN |
| Sync computation | O(T × D) per position | Sparse sync |

**Key question**: Can this be made efficient enough for real use?

### 14. Sequence Length Limitations

The sequential experience loop is inherently causal:
- Can we parallelize at all?
- Chunked processing?
- Is there a way to "batch" experiences?

### 15. Integration with Existing Systems

- Can this be added to existing LLMs?
- As a separate module? Fine-tuning?
- How to leverage pretrained weights?

## Research Directions

### Short-term Experiments

1. Implement minimal PEM on toy data
2. Visualize sync patterns and surprise
3. Test memory utility on simple recall tasks
4. Compare prediction quality: features vs tokens

### Medium-term Goals

1. Scale to real language data
2. Evaluate on benchmarks (with experience)
3. Study what sync patterns emerge
4. Analyze memory contents

### Long-term Vision

1. Truly experiential AI systems
2. Models that "remember" their mistakes
3. Adaptive processing based on content difficulty
4. Interpretable cognition through sync/surprise analysis
