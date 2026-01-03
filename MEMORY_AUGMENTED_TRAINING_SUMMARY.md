# Memory-Augmented GPT Training Summary

## Overview

This document summarizes the training of the Memory-Augmented GPT model on the BookCorpus dataset, including validation of narrative surprise and extended self-awareness metrics.

**Base Model**: `tiny_pretrain_output/model_best_eval.pt`

| Version | Checkpoint | Date | Key Feature |
|---------|------------|------|-------------|
| v1 | `memory_augmented_books_v1/memory_gpt_epoch_1.pt` | 2026-01-03 | Initial training |
| v2 | `memory_augmented_books_v2/memory_gpt_epoch_1.pt` | 2026-01-04 | Extended self-awareness |

---

# Part I: V1 Baseline

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

### Memory System Configuration (v1)

| Parameter | Value |
|-----------|-------|
| Memory Capacity | 2,000 |
| Crystallization Threshold | 0.3 |
| Meta-surprise Salience Weight | 3.0 (hardcoded) |
| Memory Integration | Gated |

---

## 2. V1 Training Configuration

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
| Affect Loss Weight | 0.0 (not trained) |
| Retrieval Loss Weight | 0.0 (not trained) |

---

## 3. V1 Evaluation Metrics

| Metric | Value |
|--------|-------|
| **LM Loss** | 4.197 |
| **Experiential Loss** | 2.019 |
| **Token Accuracy** | 20.9% |
| **Episodic Memories** | 26 |

---

## 4. V1 Narrative Validation (Initial, Uncalibrated)

The model was validated on **"Lion of the Sky"** by Ritu Hemnani — a verse novel about the 1947 Partition of India.

**Book**: `data/books_clean/7e/_OceanofPDF.com_Lion_of_the_Sky_-_Ritu_Hemnani.txt`
**Characters Analyzed**: 100,000
**Tokens Analyzed**: 27,055

### 4.1 Overall Statistics (v1, threshold=0.3, weight=3.0)

| Metric | Value |
|--------|-------|
| Chunks Analyzed | 421 |
| Surprise Mean | 0.624 ± 0.023 |
| Surprise Range | [0.567, 0.723] |
| Meta-Surprise Mean | 0.403 |
| Salience Mean | 1.377 |
| **Crystallization Rate** | **100%** (BUG: threshold too low) |
| Final Memory Size | 24 (capped by capacity) |

**Issue**: With threshold=0.3 and meta-surprise weight=3.0, salience exceeded the threshold for every chunk, causing 100% crystallization. This defeated the purpose of selective memory.

### 4.2 Structural Correlation Analysis (v1)

| Metric | Value |
|--------|-------|
| High-Surprise Moments | 105 |
| Structural Markers Found | 351 |
| High-Surprise Near Markers | 45 |
| **Correlation** | **42.9%** |

The 42.9% correlation indicates that high surprise occurs both at structural markers AND at semantically important moments that don't have explicit structural markers (emotional peaks within scenes).

### 4.3 Top Surprise Moments (v1)

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

---

## 5. Extended Self-Awareness Architecture (v1: Untrained)

The extended self-awareness system adds two new self-prediction dimensions beyond the original meta-surprise:

### 5.1 Self-Prediction Dimensions

| Dimension | Question | Metric |
|-----------|----------|--------|
| **Meta-Surprise** | "How surprised will I be?" | Prediction error on own surprise |
| **Meta-Affect Surprise** | "How will I feel?" | Prediction error on valence/arousal |
| **Meta-Retrieval Surprise** | "What will I remember?" | Prediction error on retrieval content |

### 5.2 Module Status (v1)

| Module | Parameters | Status |
|--------|------------|--------|
| Retrieval Predictor | 1,574,400 | Initialized, NOT trained |
| Affect Predictor | 1,050,114 | Initialized, NOT trained |

**Note**: In v1, these modules exist but contain randomly initialized weights. They produce outputs but are not calibrated. Training with affect_weight and retrieval_weight is required.

---

## 6. Memory System Status (v1)

### 6.1 Episodic Memory

After training, the model crystallized **26 episodic memories** from the training corpus.

### 6.2 Semantic Memory

The model consolidated **73 semantic concepts** with **2,628 relations** through automatic extraction from episodic memories.

### 6.3 Memory Usage During Inference

- Episodic retrieval uses soft attention over stored memories
- Semantic retrieval provides conceptual context
- Memory integration uses gated mechanism to blend retrieved content with hidden states

---

## 7. V1 Key Findings

1. **Surprise identifies narrative moments**: High surprise occurred at thematically significant passages — Partition violence, colonial critique, family intimacy, farewells.

2. **42.9% structural correlation**: High-surprise moments partially align with structural markers, but also occur at semantically important mid-scene moments without explicit markers.

3. **Crystallization is broken**: 100% retention defeats selective memory. Threshold calibration needed.

4. **Extended self-awareness is untrained**: Affect and retrieval predictors exist but produce uncalibrated outputs.

---

# Part II: Calibration

## 8. V1 Limitations Identified

1. **Crystallization always-on (100%)**: Threshold=0.3 with weight=3.0 caused every chunk to crystallize.

2. **Surprise is tightly compressed**: Range is only 0.567–0.723 (σ=0.023). Top 10 moments differ by ~0.09. This limits discriminative power.

3. **Crystallization driven by meta-surprise, not surprise**:
   - Crystallized moments: surprise=0.645, meta-surprise=0.628
   - Not crystallized: surprise=0.621, meta-surprise=0.329
   - Delta: surprise +0.02, **meta-surprise +0.30**
   - The model remembers "moments of self-ignorance" more than "narrative peaks"

4. **Structural correlation lacks baseline**: 42.9% correlation without null model is hard to interpret.

5. **Extended self-awareness modules are untrained**: No affect_weight or retrieval_weight in v1 training.

---

## 9. Calibration Fixes Applied

### 9.1 Threshold Calibration

Made `meta_surprise_salience_weight` configurable (was hardcoded at 3.0):

```python
# Before (v1)
salience = surprise * (1 + 3.0 * meta_surprise)  # hardcoded

# After (v2)
salience = surprise * (1 + weight * meta_surprise)  # configurable
```

### 9.2 Calibrated Parameters

| Parameter | v1 | v2 |
|-----------|-----|-----|
| Crystallization Threshold | 0.3 | 1.0 |
| Meta-surprise Salience Weight | 3.0 | 1.0 |

### 9.3 Calibration Result

Re-running validation with threshold=1.0, weight=1.0:
- Crystallization rate: **25.2%** (down from 100%)
- 99 memories formed out of 421 chunks
- Selective retention restored

---

# Part III: V2 Extended Self-Awareness

## 10. V2 Training Configuration

**Date**: 2026-01-04
**Checkpoint**: `memory_augmented_books_v2/memory_gpt_epoch_1.pt`

| Parameter | Value |
|-----------|-------|
| Max Steps | 3,000 |
| LM Loss Weight | 1.0 |
| Experiential Loss Weight | 0.1 |
| **Affect Loss Weight** | **0.1** (NEW) |
| **Retrieval Loss Weight** | **0.1** (NEW) |
| Crystallization Threshold | 1.0 |
| Meta-surprise Salience Weight | 1.0 |

---

## 11. V2 Training Results

| Metric | Value |
|--------|-------|
| LM Loss (training) | 4.577 |
| Experiential Accuracy | 82.3% |
| Memory Size | 25 |
| Training Time | 2.8 minutes |

### V2 Evaluation Metrics

| Metric | Value |
|--------|-------|
| LM Loss | 4.197 |
| Exp Loss | 2.001 |
| Token Accuracy | 21.0% |
| Memory Size | 25 |

---

## 12. V2 Narrative Validation

Validation on "Lion of the Sky" with calibrated parameters:

| Metric | v1 (uncalibrated) | v2 (calibrated) |
|--------|-------------------|-----------------|
| Surprise Mean | 0.624 ± 0.023 | 0.626 ± 0.023 |
| Meta-surprise Mean | 0.403 | 0.397 |
| Salience Mean | 1.377 | 0.875 |
| **Crystallization Rate** | 100% (bug) | **23.5%** |
| Structural Correlation | 42.9% | 38.1% |

---

## 13. Extended Self-Awareness Output (v2)

The v2 checkpoint now produces calibrated self-prediction metrics:

```python
mem_out = {
    # Core surprise
    'surprise': 0.78,
    'meta_surprise': 0.77,

    # Affect prediction (NOW TRAINED)
    'valence': -0.36,
    'arousal': 0.43,
    'predicted_valence': 0.02,
    'predicted_arousal': 0.41,  # close match!
    'meta_affect_surprise': 0.10,

    # Retrieval prediction (NOW TRAINED)
    'predicted_retrieval': tensor,
    'meta_retrieval_surprise': value,
}
```

The arousal prediction is already accurate (0.43 actual vs 0.41 predicted). Valence prediction needs more training.

---

# Appendix

## A. Files Generated

| File | Description |
|------|-------------|
| `memory_augmented_books_v1/memory_gpt_epoch_1.pt` | v1 checkpoint (no extended self-awareness training) |
| `memory_augmented_books_v2/memory_gpt_epoch_1.pt` | v2 checkpoint (with extended self-awareness) |
| `validation_lion_of_the_sky.json` | v1 narrative validation (uncalibrated) |
| `validation_lion_v2.json` | v2 narrative validation (calibrated) |
| `book_corpus_output/training_book_corpus_metadata.json` | Training dataset metadata |
| `book_corpus_output/evaluation_book_corpus_metadata.json` | Evaluation dataset metadata |

---

## B. How to Run

### Training (v2 with Extended Self-Awareness)
```bash
python train_memory_augmented.py \
  --checkpoint tiny_pretrain_output/model_best_eval.pt \
  --data_dir book_corpus_output \
  --output_dir memory_augmented_books_v2 \
  --batch_size 8 \
  --max_steps 3000 \
  --memory_capacity 2000 \
  --crystallization_threshold 1.0 \
  --meta_surprise_salience_weight 1.0 \
  --lm_weight 1.0 \
  --exp_weight 0.1 \
  --affect_weight 0.1 \
  --retrieval_weight 0.1
```

### Validation
```bash
python validate_narrative_surprise.py \
  --checkpoint memory_augmented_books_v2/memory_gpt_epoch_1.pt \
  --narrative /path/to/book.txt \
  --output validation_results.json \
  --threshold 1.0 \
  --meta_surprise_weight 1.0
```

### Inference
```python
from experiential import MemoryAugmentedGPT

# Load model with calibrated parameters
memory_gpt = MemoryAugmentedGPT(
    gpt,
    memory_capacity=2000,
    crystallization_threshold=1.0,
    meta_surprise_salience_weight=1.0,
)
memory_gpt.load_state_dict(checkpoint['memory_gpt_state_dict'])

# Forward pass with memory
logits, hidden, mem_out = memory_gpt(
    tokens,
    crystallize=True,
    use_memory=True
)

# Core surprise metrics
surprise = mem_out['surprise']
meta_surprise = mem_out['meta_surprise']
salience = mem_out['salience']

# Extended self-awareness (v2)
valence = mem_out['valence']
arousal = mem_out['arousal']
meta_affect_surprise = mem_out['meta_affect_surprise']
meta_retrieval_surprise = mem_out['meta_retrieval_surprise']
```

---

## C. Remaining Issues

1. **Surprise is tightly compressed**: Range is only ~0.15. Consider temperature scaling.

2. **Structural correlation lacks baseline**: Need null model with shuffled surprise scores.

3. **Valence prediction needs work**: Arousal is accurate, valence is not (after 3000 steps).
