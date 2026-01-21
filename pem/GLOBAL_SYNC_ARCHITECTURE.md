# PEM Global Sync Architecture with Oscillatory World Model

## Current Implementation (as of 2026-01-21)

This document provides a complete architectural overview of the PEM system with the oscillatory world model.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              PEM LOOP (Iterative Processing)                            │
│                                                                                         │
│   Input: Token Embeddings (B, S, d_model)                                               │
│                    │                                                                    │
│                    ▼                                                                    │
│   ┌────────────────────────────────────────────────────────────────────────────────┐   │
│   │                        ITERATION i (i = 1..num_iterations)                      │   │
│   │                                                                                 │   │
│   │   State: observation, cumulative_sync, world_state                              │   │
│   │                    │                                                            │   │
│   │   ┌────────────────┼────────────────┐                                           │   │
│   │   │                │                │                                           │   │
│   │   ▼                ▼                ▼                                           │   │
│   │ ┌─────────┐  ┌─────────────┐  ┌─────────────┐                                   │   │
│   │ │Prediction│  │  Surprise   │  │   Memory    │                                   │   │
│   │ │   CTM   │  │    CTM      │  │    CTM      │                                   │   │
│   │ └────┬────┘  └──────┬──────┘  └──────┬──────┘                                   │   │
│   │      │              │                │                                          │   │
│   │      │   Post-Activations (Z_history for each module)                           │   │
│   │      │              │                │                                          │   │
│   │      └──────────────┼────────────────┘                                          │   │
│   │                     ▼                                                           │   │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐   │   │
│   │   │                      GLOBAL SYNC MODULE                                 │   │   │
│   │   │                                                                         │   │   │
│   │   │   ┌─────────────────────────────────────────────────────────────────┐   │   │   │
│   │   │   │  1. Sync Computation: S = Z · Z^T (per module)                  │   │   │   │
│   │   │   │     - Captures neural synchronization patterns                  │   │   │   │
│   │   │   │     - Subsamples to sync_pairs dimensions                       │   │   │   │
│   │   │   └─────────────────────────────────────────────────────────────────┘   │   │   │
│   │   │                              │                                          │   │   │
│   │   │                              ▼                                          │   │   │
│   │   │   ┌─────────────────────────────────────────────────────────────────┐   │   │   │
│   │   │   │  2. Cross-Module Attention                                      │   │   │   │
│   │   │   │     - Which modules have similar sync dynamics?                 │   │   │   │
│   │   │   │     - Produces attended_syncs per module                        │   │   │   │
│   │   │   └─────────────────────────────────────────────────────────────────┘   │   │   │
│   │   │                              │                                          │   │   │
│   │   │                              ▼                                          │   │   │
│   │   │   ┌─────────────────────────────────────────────────────────────────┐   │   │   │
│   │   │   │  3. Sync Integration                                            │   │   │   │
│   │   │   │     - Combines module syncs into global sync (B, S, sync_pairs) │   │   │   │
│   │   │   └─────────────────────────────────────────────────────────────────┘   │   │   │
│   │   │                              │                                          │   │   │
│   │   │         ┌────────────────────┴────────────────────┐                     │   │   │
│   │   │         │                                         │                     │   │   │
│   │   │         ▼                                         ▼                     │   │   │
│   │   │   ┌───────────┐                       ┌───────────────────────────┐     │   │   │
│   │   │   │   Sync    │                       │  OSCILLATORY WORLD MODEL  │     │   │   │
│   │   │   │  Output   │                       │  (Content-based Memory)   │     │   │   │
│   │   │   └───────────┘                       │                           │     │   │   │
│   │   │                                       │  See detailed diagram     │     │   │   │
│   │   │                                       │  below                    │     │   │   │
│   │   │                                       └─────────────┬─────────────┘     │   │   │
│   │   │                                                     │                   │   │   │
│   │   │                                                     ▼                   │   │   │
│   │   │                                       ┌───────────────────────────┐     │   │   │
│   │   │                                       │  World State (B, S, d)    │     │   │   │
│   │   │                                       │  Position-specific ctx    │     │   │   │
│   │   │                                       └───────────────────────────┘     │   │   │
│   │   └─────────────────────────────────────────────────────────────────────────┘   │   │
│   │                     │                                   │                       │   │
│   │                     ▼                                   │                       │   │
│   │   ┌─────────────────────────────────────────────────────┼───────────────────┐   │   │
│   │   │              PERCEPTION ATTENTION                   │                   │   │   │
│   │   │   sync → Query → Attend to Features → observation   │                   │   │   │
│   │   └─────────────────────────────────────────────────────┼───────────────────┘   │   │
│   │                     │                                   │                       │   │
│   │                     ▼                                   │                       │   │
│   │              New observation ◄──────────────────────────┘                       │   │
│   │                     │         (world_state biases CTM z_0)                      │   │
│   │                     │                                                           │   │
│   └─────────────────────┼───────────────────────────────────────────────────────────┘   │
│                         │                                                               │
│                         ▼                                                               │
│                   Next iteration                                                        │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Oscillatory World Model - Detailed Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        OSCILLATORY WORLD MODEL (Content-based Memory)                   │
│                                                                                         │
│  KEY INSIGHT: Store WHAT happened (content), gated by WHAT mattered (surprise),         │
│               retrieved by WHAT's relevant now (sync)                                   │
│                                                                                         │
│  ═══════════════════════════════════════════════════════════════════════════════════   │
│                                                                                         │
│                              ┌─────────────────────┐                                    │
│                              │      INPUTS         │                                    │
│                              └──────────┬──────────┘                                    │
│                                         │                                               │
│           ┌─────────────────────────────┼─────────────────────────────┐                 │
│           │                             │                             │                 │
│           ▼                             ▼                             ▼                 │
│   ┌───────────────┐            ┌───────────────┐            ┌───────────────┐          │
│   │   Features    │            │    Surprise   │            │     Sync      │          │
│   │ (B,S,d_model) │            │  (B,S,1) mag  │            │(B,S,sync_pairs)│          │
│   │   Content to  │            │  Importance   │            │  Query for    │          │
│   │   remember    │            │  gate         │            │  retrieval    │          │
│   └───────┬───────┘            └───────┬───────┘            └───────┬───────┘          │
│           │                             │                             │                 │
│           ▼                             │                             │                 │
│   ┌───────────────┐                     │                             │                 │
│   │    Mean +     │                     │                             │                 │
│   │  Compress     │                     │                             │                 │
│   │ (d_feature)   │                     │                             │                 │
│   └───────┬───────┘                     │                             │                 │
│           │                             │                             │                 │
│  ═════════╪═════════════════════════════╪═════════════════════════════╪═════════════   │
│           │        WRITE PATH           │                             │                 │
│           │                             │                             │                 │
│           ▼                             ▼                             │                 │
│   ┌───────────────────────────────────────────────────────┐           │                 │
│   │              AMPLITUDE MODULATION                     │           │                 │
│   │                                                       │           │                 │
│   │   features ──► amp_modulator ──► amp_mod (N_osc,)     │           │                 │
│   │                    │                                  │           │                 │
│   │   surprise ──► sigmoid(bias + scale*surp) ──► gate    │           │                 │
│   │                    │                                  │           │                 │
│   │              gated_amp_mod = amp_mod * gate           │           │                 │
│   │                    │                                  │           │                 │
│   │   base_amplitudes * (1 + gated_amp_mod * max_mod)     │           │                 │
│   │                    │                                  │           │                 │
│   │                    ▼                                  │           │                 │
│   │            modulated_amplitudes (N_osc,)              │           │                 │
│   └───────────────────────┬───────────────────────────────┘           │                 │
│                           │                                           │                 │
│                           ▼                                           │                 │
│   ┌───────────────────────────────────────────────────────┐           │                 │
│   │              PHASE MODULATION                         │           │                 │
│   │                                                       │           │                 │
│   │   features ──► phase_modulator ──► phase_shift        │           │                 │
│   │                                                       │           │                 │
│   │   effective_phase = phases + freq_contrib + shift     │           │                 │
│   │                     ▲                                 │           │                 │
│   │                     │                                 │           │                 │
│   │   phases (buffer) ──┘  (detached, carries state)      │           │                 │
│   └───────────────────────┬───────────────────────────────┘           │                 │
│                           │                                           │                 │
│                           ▼                                           │                 │
│   ┌───────────────────────────────────────────────────────┐           │                 │
│   │              OSCILLATOR OUTPUT                        │           │                 │
│   │                                                       │           │                 │
│   │   memory_states = modulated_amps * sin(eff_phases)    │           │                 │
│   │                           │                           │           │                 │
│   │                           │ (num_oscillators,)        │           │                 │
│   │                           │                           │           │                 │
│   │                           ▼                           │           │                 │
│   │                 ┌─────────────────┐                   │           │                 │
│   │                 │  Memory Bank    │                   │           │                 │
│   │                 │  64 oscillators │                   │           │                 │
│   │                 │  periods: 8-4096│                   │           │                 │
│   │                 └────────┬────────┘                   │           │                 │
│   └──────────────────────────┼────────────────────────────┘           │                 │
│                              │                                        │                 │
│  ════════════════════════════╪════════════════════════════════════════╪═════════════   │
│                              │        READ PATH                       │                 │
│                              │                                        │                 │
│                              ▼                                        ▼                 │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │                        CROSS-ATTENTION READOUT                                  │  │
│   │                                                                                 │  │
│   │   Sync (B,S,sync_pairs) ──► osc_query_proj ──► Query (B,S,d_world)              │  │
│   │                                                      │                          │  │
│   │   Memory (num_osc,) ──► osc_key_proj ──► Key (B,1,d_world)                      │  │
│   │                     └──► osc_value_proj ─► Value (B,1,d_world)                  │  │
│   │                                                      │                          │  │
│   │                              ┌───────────────────────┘                          │  │
│   │                              ▼                                                  │  │
│   │                     MultiheadAttention                                          │  │
│   │                              │                                                  │  │
│   │                              ▼                                                  │  │
│   │                  attn_out (B,S,d_world)                                         │  │
│   │                              │                                                  │  │
│   │                              ▼                                                  │  │
│   │              world_state = RMSNorm(attn_out + Query)                            │  │
│   │                              │                                                  │  │
│   │                              │  (B, S, d_world_output)                          │  │
│   │                              │  Position-specific context!                      │  │
│   │                              │                                                  │  │
│   └──────────────────────────────┼──────────────────────────────────────────────────┘  │
│                                  │                                                     │
└──────────────────────────────────┼─────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────┐
                    │    OUTPUT: world_state    │
                    │    (B, S, d_world_output) │
                    │                           │
                    │    Each position has its  │
                    │    own retrieved context  │
                    │    from oscillator memory │
                    └───────────────────────────┘
```

---

## CTM Integration - How World State Affects Processing

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          CTM CORE (Prediction, Surprise, Memory)                        │
│                                                                                         │
│   Inputs:                                                                               │
│     - observation: (B, S, d_neurons)                                                    │
│     - world_state: (B, S, d_world) position-specific  OR  (d_world,) broadcast          │
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         NLM INITIALIZATION (t=0)                                │   │
│   │                                                                                 │   │
│   │   x_embed ──► x_to_z0 ──► z_0 (B, S, d_neurons)                                 │   │
│   │                              │                                                  │   │
│   │   world_state ──► world_to_z0 ──► z_world                                       │   │
│   │                                      │                                          │   │
│   │                                      │  if world_state.dim() == 1:              │   │
│   │                                      │      broadcast to (B, S, d_neurons)      │   │
│   │                                      │  else:                                   │   │
│   │                                      │      project (B, S, d_world) → (B,S,d_n) │   │
│   │                                      │                                          │   │
│   │                              z_t = z_0 + z_world  ◄────────────────────────────┘   │
│   │                                      │                                          │   │
│   │                                      │  World context biases initial state!     │   │
│   │                                      │                                          │   │
│   └──────────────────────────────────────┼──────────────────────────────────────────┘   │
│                                          │                                              │
│                                          ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         NLM DYNAMICS (t=1..T)                                   │   │
│   │                                                                                 │   │
│   │   for t in range(T_ticks):                                                      │   │
│   │       z_t = NLM.forward(z_t)                                                    │   │
│   │       y_t = z_to_y(z_t)                                                         │   │
│   │       all_tick_activations.append(z_t)                                          │   │
│   │       all_tick_outputs.append(y_t)                                              │   │
│   │                                                                                 │   │
│   │   # Final tick output is the prediction                                         │   │
│   │   predictions = y_T                                                             │   │
│   │                                                                                 │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Gradient Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              GRADIENT FLOW DIAGRAM                                      │
│                                                                                         │
│   Loss (prediction error)                                                               │
│         │                                                                               │
│         ▼                                                                               │
│   CTM outputs ◄─── z_to_y                                                               │
│         │                                                                               │
│         ▼                                                                               │
│   z_t (NLM dynamics) ◄─── NLM.forward()                                                 │
│         │                                                                               │
│         ▼                                                                               │
│   z_0 = z_init + z_world                                                                │
│         │              │                                                                │
│         │              ▼                                                                │
│         │      world_to_z0 ◄─── LEARNABLE                                               │
│         │              │                                                                │
│         │              ▼                                                                │
│         │      world_state (from cross-attention)                                       │
│         │              │                                                                │
│         │              ▼                                                                │
│         │      osc_cross_attn, osc_query_proj, etc. ◄─── LEARNABLE                      │
│         │              │                                                                │
│         │              ▼                                                                │
│         │      memory_states = amps * sin(phases)                                       │
│         │              │                                                                │
│         │              ▼                                                                │
│         │      modulated_amps ◄─── base_amplitudes ◄─── LEARNABLE                       │
│         │              │                                                                │
│         │              ▼                                                                │
│         │      amp_modulator, phase_modulator ◄─── LEARNABLE                            │
│         │              │                                                                │
│         │              ▼                                                                │
│         │      feature_compressor ◄─── LEARNABLE                                        │
│         │              │                                                                │
│         ▼              ▼                                                                │
│   x_to_z0 ◄─── LEARNABLE                                                                │
│                                                                                         │
│   NOTE: phases buffer is DETACHED - carries state but doesn't need gradients            │
│         frequencies get gradients through effective_phase computation                   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Tensor Shapes Summary

| Tensor | Shape | Description |
|--------|-------|-------------|
| `features` | `(B, S, d_model)` | Token embeddings, content for memory |
| `surprise` | `(B, S, 1)` | Surprise magnitude, gates memory writes |
| `sync` | `(B, S, sync_pairs)` | Global sync state, queries memory |
| `features_compressed` | `(d_feature_input,)` | Compressed features for writing |
| `surprise_scalar` | `scalar` | Compressed surprise for gating |
| `memory_states` | `(num_oscillators,)` | Oscillator outputs (64 values) |
| `world_state` | `(B, S, d_world_output)` | Position-specific retrieved context |
| `z_t` | `(B, S, d_neurons)` | NLM post-activations |
| `all_tick_activations` | `List[(B, S, d_neurons)]` | History for sync computation |

---

## Key Design Decisions

### 1. Content-based Memory (vs Sync-based)
- **Original**: Sync patterns → Oscillators → World state
- **Problem**: Sync captures HOW neurons fire together, not WHAT content
- **Fix**: Features (content) → Oscillators, Sync → Query readout

### 2. Surprise Gating
- High surprise = unexpected = important to remember
- `write_gate = sigmoid(bias + surprise * scale)`
- Prevents writing mundane content, prioritizes novelty
- **Best signal**: `attention_entropy` (per ablation tests)

### 3. Position-Specific Readout (vs Broadcast)
- **Original**: Single world_state vector broadcast to all positions
- **Problem**: No position-specific context
- **Fix**: Cross-attention where each position queries memory with its sync

### 4. Oscillator Timescale Hierarchy
```
Oscillator 0:  period=8     → phrase-level memory (~8 tokens)
Oscillator 31: period=512   → paragraph-level memory
Oscillator 63: period=4096  → document-level memory (~4096 tokens)
```

---

## File Locations

| Component | File | Key Functions |
|-----------|------|---------------|
| PEM Loop | `pem/pem_loop_global.py` | `PEMLoopGlobal.forward()` |
| Global Sync | `pem/global_sync.py` | `GlobalSyncModule.forward()` |
| Oscillatory World | `pem/oscillatory_world.py` | `OscillatoryWorldState.forward()` |
| CTM Core | `pem/ctm_base.py` | `CTMCore.forward()` |
| Prediction CTM | `pem/prediction_ctm.py` | `PredictionCTM.forward()` |
| Surprise CTM | `pem/surprise_ctm.py` | `SurpriseCTM.forward()` |
| Training | `pem/train_pem_global.py` | Training loop |
| Ablation Tests | `pem/run_ablation_tests.py` | Ablation experiments |

---

## Ablation Test Results (2026-01-21)

### Results Table (sorted by loss, best to worst)

| Configuration | Loss | Certainty | Loop Improvement | World State Benefit |
|--------------|------|-----------|------------------|---------------------|
| **no_aux_pred** | **0.0271** | 0.540 | **+0.0443** | +10.1% |
| no_world_model | 0.0292 | 0.590 | +0.0165 | +0.0% |
| **surprise_attn_entropy** | 0.0560 | **0.714** | +0.0182 | +64.6% |
| surprise_pred_error | 0.0625 | 0.664 | +0.0171 | +86.0% |
| multi_tick_injection | 0.0790 | 0.174 | +0.0046 | +28.3% |
| baseline | 0.0805 | 0.624 | +0.0164 | +80.3% |
| combined_best | 0.1102 | 0.020 | +0.0070 | +47.6% |
| aux_pred_high_weight | 0.1711 | 0.553 | +0.0221 | +86.2% |

### Key Findings

1. **Auxiliary Prediction Loss - HURTS Performance**
   - Disabling it achieved lowest loss (0.0271) - nearly 3x better than baseline
   - **Recommendation: Keep disabled (default)**

2. **Surprise Signal Types**
   - `attention_entropy`: Best certainty (0.714), good loss
   - `prediction_error`: Good balance
   - **Recommendation: Use `attention_entropy` (now default)**

3. **Multi-tick World Injection - Not Beneficial**
   - Very low certainty (0.174) vs baseline (0.624)
   - **Recommendation: Keep disabled (default)**

4. **Combined Features - Worst Overall**
   - Combining all features resulted in worst performance
   - **Recommendation: Don't stack features**

### Optimal Configuration

```bash
python -m pem.train_pem_global \
    --dataset local \
    --data_dir /path/to/texts \
    --surprise_signal_type attention_entropy  # Default
    # auxiliary prediction disabled by default
    # multi-tick injection disabled by default
```

---

## Configuration Options

### Oscillatory World Model
```python
num_oscillators: int = 64           # Number of oscillators (memory slots)
min_period: int = 8                 # Fastest oscillator (phrase-level)
max_period: int = 4096              # Slowest oscillator (document-level)
d_world_output: int = 256           # Output dimension of world state
surprise_gate_bias: float = 0.5     # Base write strength
surprise_gate_scale: float = 1.0    # Surprise sensitivity
```

### Surprise Signal
```python
surprise_signal_type: str = "attention_entropy"  # Best per ablation
# Options: "ctm", "prediction_error", "attention_entropy"
```

### Auxiliary Prediction (disabled by default)
```python
use_auxiliary_prediction: bool = False  # Hurts performance per ablation
auxiliary_prediction_horizon: int = 8
auxiliary_prediction_weight: float = 0.1
```

### Multi-tick World Injection (disabled by default)
```python
multi_tick_world_injection: bool = False  # Low certainty per ablation
```

---

## Future Directions

1. **Hyperparameter tuning** - Optimize oscillator count and frequency range
2. **Longer training** - More epochs/documents for better convergence
3. **Alternative architectures** - Different readout mechanisms
4. **Multi-document learning** - Cross-document knowledge transfer
