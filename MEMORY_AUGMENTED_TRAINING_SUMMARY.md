# Memory-Augmented GPT Training Summary

## Overview

This document summarizes the training of the Memory-Augmented GPT model on the BookCorpus dataset, including validation of narrative surprise and extended self-awareness metrics.

**Date**: 2026-01-03
**Checkpoint**: `memory_augmented_books_v1/memory_gpt_epoch_1.pt`
**Base Model**: `tiny_pretrain_output/model_best_eval.pt`

---

## 1. Model Configuration

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 128,256 |
| Model Dimension | 1,024 |
| Attention Heads | 8 |
| KV Heads | 4 |
| Layers | 12 |
| Feed-Forward Dim | 4,096 |
| Max Sequence Length | 1,024 |
| Total Parameters | ~125M |

### Memory System Configuration

| Parameter | Value |
|-----------|-------|
| Memory Capacity | 2,000 |
| Crystallization Threshold | 0.3 |
| Memory Integration | Gated |

---

## 2. Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | BookCorpus |
| Training Examples | 136,071 |
| Training Tokens | ~70M |
| Batch Size | 8 |
| Sequence Length | 512 |
| Gradient Accumulation | 4 steps |
| Effective Batch Size | 32 |
| Learning Rate | 1e-4 |
| Max Steps | 2,000 |
| LM Loss Weight | 1.0 |
| Experiential Loss Weight | 0.1 |

---

## 3. Final Evaluation Metrics

| Metric | Value |
|--------|-------|
| **LM Loss** | 4.197 |
| **Experiential Loss** | 2.019 |
| **Token Accuracy** | 20.9% |
| **Episodic Memories** | 26 |

---

## 4. Narrative Surprise Validation

The model was validated on **"Lion of the Sky"** by Ritu Hemnani — a verse novel about the 1947 Partition of India, told from the perspective of a young boy named Raj.

**Book**: `data/books_clean/7e/_OceanofPDF.com_Lion_of_the_Sky_-_Ritu_Hemnani.txt`
**Characters Analyzed**: 100,000
**Tokens Analyzed**: 27,055

### 4.1 Overall Statistics

| Metric | Value |
|--------|-------|
| Chunks Analyzed | 421 |
| Surprise Mean | 0.624 ± 0.023 |
| Surprise Range | [0.567, 0.723] |
| Meta-Surprise Mean | 0.403 |
| Salience Mean | 1.377 |
| Crystallization Rate | 100% |
| Final Memory Size | 24 (capped by capacity) |

### 4.2 Structural Correlation Analysis

| Metric | Value |
|--------|-------|
| High-Surprise Moments | 105 |
| Structural Markers Found | 351 |
| High-Surprise Near Markers | 45 |
| **Correlation** | **42.9%** |

The 42.9% correlation indicates that high surprise occurs both at structural markers AND at semantically important moments that don't have explicit structural markers (emotional peaks within scenes).

### 4.3 Top Surprise Moments

| Rank | Surprise | Meta-Surprise | Narrative Moment |
|------|----------|---------------|------------------|
| 1 | 0.723 | 0.240 | **Book opening**: Dedication to family |
| 2 | 0.694 | 0.689 | **Violence**: Children hiding under sari folds |
| 3 | 0.690 | 0.654 | **Colonial critique**: Railways stealing resources |
| 4 | 0.689 | 0.653 | **Farewell**: Saying goodbye to Iqbal amid rubble |
| 5 | 0.688 | 0.683 | **Kite Festival**: Meeting Uncle Mitu, learning about Bhavna |
| 6 | 0.681 | 0.681 | **Kitchen scene**: Stealing dal, family intimacy |
| 7 | 0.678 | 0.028 | **The Line**: British drawing the Partition boundary |
| 8 | 0.675 | 0.602 | **Hope**: "They'll be gone soon enough, we'll be free" |
| 9 | 0.671 | 0.012 | **Fear**: "Should we leave Sindh?" |
| 10 | 0.671 | 0.629 | **Imagery**: Lines everywhere—knife, bullet, parents' eyes |

### 4.4 Analysis of High-Surprise Moments

The model identified emotionally and narratively significant moments:

1. **Violence and Partition** (positions 24768, 23744): Scenes of communal violence, children hiding, bruises and wounds — the traumatic core of the narrative.

2. **Colonial Critique** (position 8704): Discussion of British exploitation through railways — thematic exposition.

3. **Family Intimacy** (positions 3200, 5376, 5632): Kitchen scenes with Amma, cooking dal, stealing food — tender moments of normalcy.

4. **Boundary Lines** (positions 15040, 18496): The central metaphor of the book — lines drawn by the British that will tear apart communities.

5. **Farewells** (positions 23744, 20416): Saying goodbye to friends like Iqbal as the Partition approaches.

### 4.5 Memory Crystallization Pattern

The model crystallized 421 moments but retained only 24 in final memory (due to capacity limits). The highest-salience memories preserved were:
- Opening dedication and chapter structure
- Key emotional scenes
- Thematically significant passages about Partition

This selective retention mirrors how human memory prioritizes emotionally salient experiences.

---

## 5. Extended Self-Awareness Metrics

The extended self-awareness system adds two new self-prediction dimensions beyond the original meta-surprise:

### 5.1 Self-Prediction Dimensions

| Dimension | Question | Metric |
|-----------|----------|--------|
| **Meta-Surprise** | "How surprised will I be?" | Prediction error on own surprise |
| **Meta-Affect Surprise** | "How will I feel?" | Prediction error on valence/arousal |
| **Meta-Retrieval Surprise** | "What will I remember?" | Prediction error on retrieval content |

### 5.2 Module Status

| Module | Parameters | Status |
|--------|------------|--------|
| Retrieval Predictor | 1,574,400 | ✅ Initialized |
| Affect Predictor | 1,050,114 | ✅ Initialized |

**Note**: The extended self-awareness modules were added after this training run. They exist in the model architecture but contain randomly initialized weights. A new training run would optimize these weights to improve self-prediction accuracy.

### 5.3 Sample Inference Results

From a single forward pass on sample text:

| Metric | Value |
|--------|-------|
| Surprise | 0.780 |
| Meta-Surprise | 0.766 |
| Meta-Affect Surprise | 0.059 |
| Meta-Retrieval Surprise | N/A (no prior retrieval) |
| Valence | 0.233 |
| Arousal | 0.365 |
| Salience | 2.571 |

### 5.4 Output Keys Available

The forward pass now returns comprehensive experiential metrics:
- `surprise`, `meta_surprise` - Core surprise signals
- `valence`, `arousal`, `salience` - Affective dimensions
- `predicted_valence`, `predicted_arousal` - Self-predicted affect
- `meta_affect_surprise` - Affect prediction error
- `predicted_retrieval`, `meta_retrieval_surprise` - Retrieval prediction
- `surprise_t`, `excess_t`, `novelty_t` - Surprise components
- `ema_mu`, `ema_sigma` - Running surprise statistics

---

## 6. Memory System Status

### 6.1 Episodic Memory

After training, the model crystallized **26 episodic memories** from the corpus, representing high-salience moments that exceeded the crystallization threshold.

### 6.2 Semantic Memory

The model consolidated **73 semantic concepts** with **2,628 relations** through automatic extraction from episodic memories.

### 6.3 Memory Usage During Inference

- Episodic retrieval uses soft attention over stored memories
- Semantic retrieval provides conceptual context
- Memory integration uses gated mechanism to blend retrieved content with hidden states

---

## 7. Key Findings

1. **Surprise identifies narrative moments**: High surprise occurred at thematically significant passages — Partition violence, colonial critique, family intimacy, farewells.

2. **42.9% structural correlation**: High-surprise moments partially align with structural markers, but also occur at semantically important mid-scene moments without explicit markers.

3. **Extended self-awareness is architecturally complete**: The affect and retrieval predictors exist and produce outputs, but require training optimization.

---

## 7.1 Limitations & Caveats (Updated After Calibration)

**RESOLVED** ✓ Crystallization is now selective (25.2% retention with threshold=1.0, weight=1.0).

**Remaining issues:**

1. **Surprise is tightly compressed**: Range is only 0.578–0.777 (σ=0.023). Top 10 moments differ by ~0.09. This limits discriminative power.

2. **Crystallization driven by meta-surprise, not surprise**:
   - Crystallized moments: surprise=0.645, meta-surprise=0.628
   - Not crystallized: surprise=0.621, meta-surprise=0.329
   - Delta: surprise +0.02, **meta-surprise +0.30**
   - The model remembers "moments of self-ignorance" more than "narrative peaks"

3. **Structural correlation lacks baseline**: 37.1% correlation without null model is hard to interpret.

4. **Extended self-awareness modules are untrained**: The checkpoint predates the affect/retrieval predictors.

---

## 8. Next Steps

### Immediate Fixes

1. ✅ **Recalibrate crystallization threshold**: DONE. Set threshold=1.0, meta_surprise_weight=1.0 → 25.2% retention.

2. **Add baseline for structural correlation**: Compute correlation with shuffled surprise scores as null model.

3. **Train extended self-awareness**: Run training with meta-affect and meta-retrieval losses to calibrate the predictors.

### Evaluation Improvements

4. **Widen surprise dynamic range**: Investigate why surprise is compressed to 0.15 range. Consider temperature scaling or different surprise formulation.

5. **Evaluate retrieval quality**: Measure whether retrieved episodic memories improve prediction at later narrative points.

6. **Compare with baseline**: Benchmark against vanilla GPT on narrative understanding tasks.

### Calibration Parameters

For future runs, use these calibrated settings:
```bash
--threshold 1.0 --meta_surprise_weight 1.0
```

Or in code:
```python
MemoryAugmentedGPT(
    gpt,
    crystallization_threshold=1.0,
    meta_surprise_salience_weight=1.0,
)
```

---

## 9. Files Generated

| File | Description |
|------|-------------|
| `memory_augmented_books_v1/memory_gpt_epoch_1.pt` | Trained checkpoint |
| `validation_lion_of_the_sky.json` | Narrative surprise validation on "Lion of the Sky" |
| `book_corpus_output/training_book_corpus_metadata.json` | Training dataset metadata |
| `book_corpus_output/evaluation_book_corpus_metadata.json` | Evaluation dataset metadata |
| `MEMORY_AUGMENTED_TRAINING_SUMMARY.md` | This summary document |

---

## 10. How to Run

### Training
```bash
python train_memory_augmented.py \
  --checkpoint tiny_pretrain_output/model_best_eval.pt \
  --data_dir book_corpus_output \
  --output_dir memory_augmented_books_v2 \
  --batch_size 8 \
  --max_steps 5000 \
  --memory_capacity 2000 \
  --crystallization_threshold 0.3
```

### Validation
```bash
python validate_narrative_surprise.py \
  --checkpoint memory_augmented_books_v1/memory_gpt_epoch_1.pt \
  --narrative /path/to/book.txt \
  --output validation_results.json
```

### Inference
```python
from experiential import MemoryAugmentedGPT

# Load model
memory_gpt = MemoryAugmentedGPT(gpt, memory_capacity=2000)
memory_gpt.load_state_dict(checkpoint['memory_gpt_state_dict'])

# Forward pass with memory
logits, hidden, mem_out = memory_gpt(
    tokens,
    crystallize=True,
    use_memory=True
)

# Access metrics
surprise = mem_out['surprise']
salience = mem_out['salience']
meta_surprise = mem_out['meta_surprise']
```
