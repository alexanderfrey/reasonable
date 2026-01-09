# CTM (Continuous Thought Machine) Architecture

## Overview

This document describes the Transformer-CTM hybrid architecture for language modeling.
The implementation preserves the core CTM principles while adapting them for autoregressive
text generation.

## Core Principles

### 1. Neural Synchronization as Representation

The correlation structure between neurons over time IS the latent representation:

```
S^t = correlation(Z^{0:t})
```

Where Z contains the history of post-activations. This decouples the representation
from the neural dynamics, allowing richer temporal patterns.

### 2. Global Thought Stream

The synchronization is computed from the FULL history spanning ALL layers and ticks,
preserving the continuous thought:

```
WRONG (fragmented):
  Layer 1: z^{1,0} → z^{1,T}  |  S¹ captures only this
           [RESET]            |
  Layer 2: z^{2,0} → z^{2,T}  |  S² knows nothing about Layer 1

CORRECT (continuous):
  Full stream: z^{1,0} → ... → z^{1,T} → z^{2,0} → ... → z^{2,T}
               |_______________________________________________|
                                     ↓
                     S^t computed from ALL of this
```

### 3. Data as Static Key-Values

Input is encoded once and stored as static keys/values. The model actively queries
this data each tick rather than having data flow through layers.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CTMLanguageModel                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐                                                          │
│   │ Input Tokens │  (B, S)                                                  │
│   └──────┬───────┘                                                          │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────────┐                                                      │
│   │ Token Embedding  │  (B, S) → (B, S, D)                                  │
│   └──────┬───────────┘                                                      │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────────────────────────────────────────────┐                  │
│   │              Input Encoder (2 Transformer Blocks)     │                  │
│   │  ┌─────────────┐    ┌─────────────┐                  │                  │
│   │  │ Block 1     │ →  │ Block 2     │                  │                  │
│   │  │ (Attn+FFN)  │    │ (Attn+FFN)  │                  │                  │
│   │  └─────────────┘    └─────────────┘                  │                  │
│   └──────────────────────┬───────────────────────────────┘                  │
│                          │                                                   │
│          ┌───────────────┼───────────────┐                                  │
│          │               │               │                                   │
│          ▼               ▼               ▼                                   │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐                            │
│   │ encoded    │  │ K_proj     │  │ V_proj     │                            │
│   │ (B,S,D)    │  │ (B,S,H,d)  │  │ (B,S,H,d)  │                            │
│   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                            │
│         │               │               │                                    │
│         │               └───────┬───────┘                                    │
│         │                       │                                            │
│         │            ┌──────────┴──────────┐                                │
│         │            │   Static KV Cache   │  (reused every tick)           │
│         │            │   ┌─────┐ ┌─────┐   │                                │
│         │            │   │  K  │ │  V  │   │                                │
│         │            │   └─────┘ └─────┘   │                                │
│         │            └──────────┬──────────┘                                │
│         │                       │                                            │
│         ▼                       ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                         CTM Core                                 │       │
│   │  (see detailed diagram below)                                    │       │
│   └───────────────────────────┬─────────────────────────────────────┘       │
│                               │                                              │
│                               ▼                                              │
│                    ┌─────────────────────┐                                  │
│                    │ all_states (list)   │                                  │
│                    │ [s₀, s₁, ..., s_T]  │  T = n_layer × num_ticks         │
│                    └──────────┬──────────┘                                  │
│                               │                                              │
│                    ┌──────────┴──────────┐                                  │
│                    ▼                      ▼                                  │
│             ┌────────────┐         ┌────────────┐                           │
│             │ final_norm │         │ final_norm │  (for each tick)          │
│             └─────┬──────┘         └─────┬──────┘                           │
│                   │                      │                                   │
│                   ▼                      ▼                                   │
│             ┌────────────┐         ┌────────────┐                           │
│             │  lm_head   │         │  lm_head   │                           │
│             └─────┬──────┘         └─────┬──────┘                           │
│                   │                      │                                   │
│                   ▼                      ▼                                   │
│             ┌────────────┐         ┌────────────┐                           │
│             │final_logits│         │ all_logits │  → CTMLoss (tick select) │
│             │ (B, S, V)  │         │ [T × ...]  │                           │
│             └────────────┘         └────────────┘                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## CTM Core: The Continuous Thought Engine

The CTM Core is where the "thinking" happens. It maintains a global history
of post-activations and computes synchronization from the entire thought stream.

### Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CTM Core                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SHARED MODULES (operate on global history):                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Global NLM          │  Global Sync Module                          │   │
│  │  (processes full     │  (computes correlations from                 │   │
│  │   temporal history)  │   full thought stream)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  LAYERS (each processes num_ticks iterations):                              │
│  ┌─────────────────┐  ┌─────────────────┐       ┌─────────────────┐        │
│  │    CTMLayer 0   │  │    CTMLayer 1   │  ...  │  CTMLayer N-1   │        │
│  │  (num_ticks×)   │  │  (num_ticks×)   │       │  (num_ticks×)   │        │
│  └─────────────────┘  └─────────────────┘       └─────────────────┘        │
│                                                                              │
│  GLOBAL HISTORY BUFFER:                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  (B, S, n_layer × num_ticks, D)                                      │   │
│  │                                                                       │   │
│  │  Layer 0, tick 0 │ Layer 0, tick 1 │ ... │ Layer N-1, tick T-1       │   │
│  │       ↓                ↓                           ↓                  │   │
│  │    post_act₀        post_act₁        ...     post_act_{N×T-1}        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Execution Flow

```
global_step = 0
global_history = zeros(B, S, n_layer × num_ticks, D)
state = initial_state (from encoder)

for layer_idx in [0, 1, ..., n_layer-1]:
    for tick in [0, 1, ..., num_ticks-1]:

        ┌─────────────────────────────────────────────────────────────┐
        │ Step: layer={layer_idx}, tick={tick}, global_step={g_step}  │
        └─────────────────────────────────────────────────────────────┘

        1. GET CURRENT HISTORY
           ┌──────────────────────────────────────────────────────────┐
           │  current_history = global_history[:, :, :global_step]    │
           │                                                          │
           │  Example at layer=1, tick=2 (global_step=6):             │
           │  ┌───┬───┬───┬───┬───┬───┬───┬───┐                      │
           │  │L0 │L0 │L0 │L0 │L1 │L1 │   │   │                      │
           │  │t0 │t1 │t2 │t3 │t0 │t1 │ - │ - │                      │
           │  └───┴───┴───┴───┴───┴───┴───┴───┘                      │
           │   ↑───────────────────────↑                              │
           │   current_history (6 steps of thought)                   │
           └──────────────────────────────────────────────────────────┘

        2. COMPUTE GLOBAL SYNC
           ┌──────────────────────────────────────────────────────────┐
           │                                                          │
           │  sync = GlobalSync(current_history)                      │
           │                                                          │
           │  ┌─────────────────────────────────────────────────┐    │
           │  │ For each sampled neuron pair (i, j):            │    │
           │  │                                                  │    │
           │  │   h_i = history[..., i]  # (B, S, t)            │    │
           │  │   h_j = history[..., j]  # (B, S, t)            │    │
           │  │                                                  │    │
           │  │   decay_weights = exp(-decay × [t-1, t-2, ...]) │    │
           │  │   sync_ij = sum(h_i × h_j × decay_weights)      │    │
           │  └─────────────────────────────────────────────────┘    │
           │                                                          │
           │  Output: sync (B, S, sync_pairs)                         │
           └──────────────────────────────────────────────────────────┘

        3. COMPUTE GLOBAL NLM
           ┌──────────────────────────────────────────────────────────┐
           │                                                          │
           │  nlm_out = GlobalNLM(current_history)                    │
           │                                                          │
           │  For each neuron d in [0, D):                            │
           │  ┌─────────────────────────────────────────────────┐    │
           │  │   history_d = history[..., d]  # (B, S, t)      │    │
           │  │                                                  │    │
           │  │   # Private MLP for this neuron                  │    │
           │  │   h = Linear_d(history_d)  # temporal → hidden   │    │
           │  │   h = GELU(h)                                    │    │
           │  │   out_d = Linear_d(h)      # hidden → 1          │    │
           │  └─────────────────────────────────────────────────┘    │
           │                                                          │
           │  Output: nlm_out (B, S, D)                               │
           └──────────────────────────────────────────────────────────┘

        4. LAYER FORWARD
           ┌──────────────────────────────────────────────────────────┐
           │                                                          │
           │  state, post_act = Layer.forward_with_global_context(    │
           │      state, sync, nlm_out, static_K, static_V            │
           │  )                                                       │
           │                                                          │
           │  (see CTMLayer diagram below)                            │
           └──────────────────────────────────────────────────────────┘

        5. UPDATE GLOBAL HISTORY
           ┌──────────────────────────────────────────────────────────┐
           │                                                          │
           │  global_history[:, :, global_step] = post_act            │
           │  global_step += 1                                        │
           │                                                          │
           │  all_states.append(state)  # for tick selection          │
           └──────────────────────────────────────────────────────────┘

return final_state, all_states
```

---

## CTMLayer: Single Thought Step

Each CTMLayer processes one step of the thought process, receiving precomputed
global sync and NLM outputs.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CTMLayer                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUTS:                                                                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────┐  ┌────────┐   │
│  │   state    │  │global_sync │  │global_nlm  │  │static_K│  │static_V│   │
│  │  (B,S,D)   │  │(B,S,P)     │  │  (B,S,D)   │  │(B,S,H,d)│ │(B,S,H,d)│  │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └────┬───┘  └────┬───┘   │
│        │               │               │              │           │        │
│        │               │               │              └─────┬─────┘        │
│        │               │               │                    │              │
│  ══════╪═══════════════╪═══════════════╪════════════════════╪══════════   │
│        │               │               │                    │              │
│        │               │               ▼                    │              │
│        │               │        ┌────────────┐              │              │
│        │               │        │  norm_nlm  │              │              │
│        │               │        └─────┬──────┘              │              │
│        │               │              │                     │              │
│        │               │              ▼                     │              │
│        │     ┌─────────┴─────────────(+)◄────── Residual    │              │
│        │     │                        │                     │              │
│        ▼     ▼                        ▼                     │              │
│  ┌───────────────┐             ┌────────────┐              │              │
│  │   norm_attn   │             │   state    │              │              │
│  └───────┬───────┘             └─────┬──────┘              │              │
│          │                           │                      │              │
│          │    ┌──────────────────────┘                      │              │
│          │    │                                             │              │
│          ▼    ▼                                             ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │                     Cross-Attention                              │      │
│  │  ┌─────────┐                                                     │      │
│  │  │ Q_proj  │ ◄── state (query the data based on current thought)│      │
│  │  └────┬────┘                                                     │      │
│  │       │         ┌───────────────────────────────────────┐       │      │
│  │       │         │     FlashAttention-2 (causal)         │       │      │
│  │       │         │                                        │       │      │
│  │       └────────►│  Q ────┐                              │       │      │
│  │                 │        ├──► Attention ──► O_proj ─────┼──►obs │      │
│  │  static_K ─────►│  K ────┤                              │       │      │
│  │  static_V ─────►│  V ────┘                              │       │      │
│  │                 │                                        │       │      │
│  │                 └───────────────────────────────────────┘       │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                              │                             │
│                                              ▼                             │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │                       Synapse Model                              │      │
│  │                                                                  │      │
│  │   ┌───────────┐   ┌───────────┐   ┌───────────┐                │      │
│  │   │   state   │   │    obs    │   │global_sync│                │      │
│  │   │  (B,S,D)  │   │  (B,S,D)  │   │  (B,S,P)  │                │      │
│  │   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘                │      │
│  │         │               │               │                       │      │
│  │         └───────────────┼───────────────┘                       │      │
│  │                         │                                        │      │
│  │                         ▼                                        │      │
│  │                  ┌─────────────┐                                │      │
│  │                  │   concat    │  (B, S, D + D + P)             │      │
│  │                  └──────┬──────┘                                │      │
│  │                         │                                        │      │
│  │                         ▼                                        │      │
│  │         ┌─────────────────────────────────┐                     │      │
│  │         │         U-Net MLP               │                     │      │
│  │         │  ┌─────┐      ┌─────┐           │                     │      │
│  │         │  │enc1 │ ──►  │enc2 │  (encoder)│                     │      │
│  │         │  └──┬──┘      └──┬──┘           │                     │      │
│  │         │     │  skip      │              │                     │      │
│  │         │     │   ↓        ▼              │                     │      │
│  │         │     │       ┌─────┐             │                     │      │
│  │         │     └──────►│dec2 │  (decoder)  │                     │      │
│  │         │             └──┬──┘             │                     │      │
│  │         │                │                │                     │      │
│  │         │                ▼                │                     │      │
│  │         │           ┌─────┐               │                     │      │
│  │         │           │dec1 │               │                     │      │
│  │         │           └──┬──┘               │                     │      │
│  │         └──────────────┼──────────────────┘                     │      │
│  │                        │                                        │      │
│  │                        ▼                                        │      │
│  │                 synapse_out (B, S, D)                           │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                              │                             │
│                                              ▼                             │
│                                       ┌─────(+)◄────── Residual            │
│                                       │                                    │
│                                       ▼                                    │
│                                ┌────────────┐                              │
│                                │  norm_ffn  │                              │
│                                └─────┬──────┘                              │
│                                      │                                     │
│                                      ▼                                     │
│                    ┌─────────────────────────────────┐                    │
│                    │           OptimizedMLP          │                    │
│                    │  ┌──────────────────────────┐   │                    │
│                    │  │ gate_up_proj (fused)     │   │                    │
│                    │  │ (D) → (2 × D_ff)         │   │                    │
│                    │  └───────────┬──────────────┘   │                    │
│                    │              │                   │                    │
│                    │         ┌────┴────┐             │                    │
│                    │         ▼         ▼             │                    │
│                    │      [gate]    [up]             │                    │
│                    │         │         │             │                    │
│                    │         ▼         │             │                    │
│                    │      SiLU(gate) × up            │                    │
│                    │              │                   │                    │
│                    │              ▼                   │                    │
│                    │  ┌──────────────────────────┐   │                    │
│                    │  │ down_proj                │   │                    │
│                    │  │ (D_ff) → (D)             │   │                    │
│                    │  └───────────┬──────────────┘   │                    │
│                    └──────────────┼──────────────────┘                    │
│                                   │                                        │
│                                   ▼                                        │
│                            ┌─────(+)◄────── Residual                       │
│                            │                                               │
│                            ▼                                               │
│                     ┌────────────┐                                         │
│                     │ norm_post  │                                         │
│                     └─────┬──────┘                                         │
│                           │                                                │
│             ┌─────────────┴─────────────┐                                 │
│             │                           │                                  │
│             ▼                           ▼                                  │
│      ┌────────────┐              ┌────────────┐                           │
│      │   state    │              │  post_act  │  → to global_history      │
│      │  (B,S,D)   │              │  (B,S,D)   │                           │
│      └────────────┘              └────────────┘                           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Neuron-Level Models (NLMs)

Each of the D neurons has its own private MLP that processes its temporal history.
This is efficiently implemented using batched einsum operations.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NeuronLevelModels                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: history (B, S, T, D) - temporal history for all neurons             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Conceptually: D independent MLPs                                    │   │
│  │                                                                       │   │
│  │  Neuron 0:  history[..., 0] ──► MLP_0 ──► out[..., 0]               │   │
│  │  Neuron 1:  history[..., 1] ──► MLP_1 ──► out[..., 1]               │   │
│  │  Neuron 2:  history[..., 2] ──► MLP_2 ──► out[..., 2]               │   │
│  │     ...           ...            ...          ...                     │   │
│  │  Neuron D-1: history[..., D-1] ──► MLP_{D-1} ──► out[..., D-1]      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  IMPLEMENTATION: Batched einsum for efficiency                              │
│                                                                              │
│  Parameters:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  w_in:  (D, max_ticks, nlm_hidden)  - input projection per neuron   │   │
│  │  b_in:  (D, nlm_hidden)                                              │   │
│  │  w_out: (D, nlm_hidden, 1)          - output projection per neuron  │   │
│  │  b_out: (D, 1)                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Forward pass:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  1. Pad history to max_ticks (zero-pad on left)                      │   │
│  │     history_padded: (B, S, max_ticks, D)                             │   │
│  │                                                                       │   │
│  │  2. Transpose for batched matmul                                     │   │
│  │     h: (B, S, D, max_ticks)                                          │   │
│  │                                                                       │   │
│  │  3. Input projection (batched across D)                              │   │
│  │     h = einsum('bsdt,dth->bsdh', h, w_in) + b_in                     │   │
│  │     h: (B, S, D, nlm_hidden)                                         │   │
│  │                                                                       │   │
│  │  4. Activation                                                        │   │
│  │     h = GELU(h)                                                       │   │
│  │                                                                       │   │
│  │  5. Output projection                                                 │   │
│  │     out = einsum('bsdh,dho->bsdo', h, w_out) + b_out                 │   │
│  │     out: (B, S, D, 1)                                                 │   │
│  │                                                                       │   │
│  │  6. Squeeze                                                           │   │
│  │     out: (B, S, D)                                                    │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  OUTPUT: (B, S, D) - updated activation for each neuron                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Synchronization Module

Computes neural synchronization from the correlation structure of the post-activation
history. Uses sampled neuron pairs for efficiency (O(P) instead of O(D²)).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SynchronizationModule                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: history (B, S, T, D) - post-activation history                      │
│                                                                              │
│  REGISTERED BUFFERS (fixed at init):                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  pair_idx1: (P,) - first neuron index for each pair                  │   │
│  │  pair_idx2: (P,) - second neuron index for each pair                 │   │
│  │                                                                       │   │
│  │  Example with D=4, P=3:                                               │   │
│  │    pair_idx1 = [0, 2, 1]                                              │   │
│  │    pair_idx2 = [3, 2, 0]                                              │   │
│  │    → pairs: (0,3), (2,2), (1,0)                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  LEARNABLE PARAMETERS:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  decay_raw: (P,) - learnable decay rate per pair                     │   │
│  │  decay = softplus(decay_raw)  # ensures decay >= 0                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Forward pass:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  1. Extract histories for sampled pairs                               │   │
│  │     h1 = history[..., pair_idx1]  # (B, S, T, P)                     │   │
│  │     h2 = history[..., pair_idx2]  # (B, S, T, P)                     │   │
│  │                                                                       │   │
│  │  2. Compute decay weights (recent activations weighted more)          │   │
│  │     time_indices = [0, 1, 2, ..., T-1]                                │   │
│  │     decay_weights[t, p] = exp(-decay[p] × (T - 1 - t))               │   │
│  │                                                                       │   │
│  │     Example with T=4, decay=0.5:                                      │   │
│  │       t=0: exp(-0.5 × 3) = 0.22  (oldest, lowest weight)             │   │
│  │       t=1: exp(-0.5 × 2) = 0.37                                       │   │
│  │       t=2: exp(-0.5 × 1) = 0.61                                       │   │
│  │       t=3: exp(-0.5 × 0) = 1.00  (newest, highest weight)            │   │
│  │                                                                       │   │
│  │  3. Normalize weights                                                 │   │
│  │     decay_weights /= sqrt(sum(decay_weights))                        │   │
│  │                                                                       │   │
│  │  4. Compute weighted correlation                                      │   │
│  │     sync = sum_t(h1[t] × h2[t] × decay_weights[t])                   │   │
│  │     sync: (B, S, P)                                                   │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  OUTPUT: (B, S, P) - synchronization values for P neuron pairs             │
│                                                                              │
│  VISUALIZATION:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Neurons:   0    1    2    3    4    ...   D-1                       │   │
│  │             │    │    │    │    │          │                          │   │
│  │  History:   ┼────┼────┼────┼────┼──────────┼                          │   │
│  │   tick 0    ●    ●    ●    ●    ●          ●                          │   │
│  │   tick 1    ●    ●    ●    ●    ●          ●                          │   │
│  │   tick 2    ●    ●    ●    ●    ●          ●                          │   │
│  │   tick 3    ●    ●    ●    ●    ●          ●                          │   │
│  │                                                                       │   │
│  │  Sampled pairs:                                                       │   │
│  │    pair 0: neuron 0 ←──────────────────► neuron 3                    │   │
│  │            └─ correlation over ticks ──┘                              │   │
│  │                                                                       │   │
│  │    pair 1: neuron 2 ←──────────────────► neuron 2  (self-correlation)│   │
│  │                                                                       │   │
│  │    pair 2: neuron 1 ←──────────────────► neuron 4                    │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Input Encoder

A shallow transformer encoder that processes input tokens once and produces static
key-value pairs for cross-attention.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Input Encoder                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: x (B, S, D) - embedded tokens                                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                  TransformerBlock 0                            │  │   │
│  │  │  ┌─────────┐    ┌─────────────────┐    ┌─────────┐            │  │   │
│  │  │  │ RMSNorm │ → │ Self-Attention   │ → │  (+)    │            │  │   │
│  │  │  └─────────┘    │ (FlashAttn+RoPE)│    └────┬────┘            │  │   │
│  │  │                 └─────────────────┘         │                  │  │   │
│  │  │                                             ▼                  │  │   │
│  │  │  ┌─────────┐    ┌─────────────────┐    ┌─────────┐            │  │   │
│  │  │  │ RMSNorm │ → │ MLP (SwiGLU)    │ → │  (+)    │            │  │   │
│  │  │  └─────────┘    └─────────────────┘    └────┬────┘            │  │   │
│  │  └────────────────────────────────────────────┼──────────────────┘  │   │
│  │                                                │                     │   │
│  │                                                ▼                     │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                  TransformerBlock 1                            │  │   │
│  │  │  (same structure as Block 0)                                   │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                │                     │   │
│  └────────────────────────────────────────────────┼─────────────────────┘   │
│                                                   │                          │
│                                                   ▼                          │
│                                            encoded (B, S, D)                 │
│                                                   │                          │
│                     ┌─────────────────────────────┼───────────────────┐     │
│                     │                             │                   │     │
│                     ▼                             ▼                   ▼     │
│              ┌────────────┐                ┌────────────┐      ┌──────────┐│
│              │   K_proj   │                │   V_proj   │      │ encoded  ││
│              │ Linear(D→H×d)               │ Linear(D→H×d)     │ (B,S,D)  ││
│              └─────┬──────┘                └─────┬──────┘      └────┬─────┘│
│                    │                             │                   │      │
│                    ▼                             ▼                   │      │
│              ┌────────────┐                ┌────────────┐           │      │
│              │ Reshape to │                │ Reshape to │           │      │
│              │ (B,S,H,d)  │                │ (B,S,H,d)  │           │      │
│              └─────┬──────┘                └─────┬──────┘           │      │
│                    │                             │                   │      │
│                    ▼                             │                   │      │
│              ┌────────────┐                      │                   │      │
│              │ Apply RoPE │                      │                   │      │
│              └─────┬──────┘                      │                   │      │
│                    │                             │                   │      │
│                    ▼                             ▼                   ▼      │
│              ┌────────────┐                ┌────────────┐    ┌────────────┐│
│              │  static_K  │                │  static_V  │    │initial_state│
│              │ (B,S,H,d)  │                │ (B,S,H,d)  │    │  (B,S,D)   ││
│              └────────────┘                └────────────┘    └────────────┘│
│                                                                              │
│  These are computed ONCE and reused for all CTM ticks                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## CTM Loss with Tick Selection

The loss function selects the best tick for each position, enabling adaptive compute.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CTMLoss                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT:                                                                     │
│    all_logits: List of T tensors, each (B, S, V)                           │
│    labels: (B, S)                                                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  1. COMPUTE PER-TICK LOSSES                                          │   │
│  │                                                                       │   │
│  │     For each tick t in [0, T):                                        │   │
│  │       shift_logits = logits[t][:, :-1]   # predict next token        │   │
│  │       shift_labels = labels[:, 1:]                                    │   │
│  │       loss_t = CrossEntropy(shift_logits, shift_labels)              │   │
│  │                                                                       │   │
│  │     tick_losses: (T, B, S-1)                                          │   │
│  │                                                                       │   │
│  │     ┌─────────────────────────────────────────────────────────┐      │   │
│  │     │  Example tick_losses for 8 ticks, batch=2, seq=10:      │      │   │
│  │     │                                                          │      │   │
│  │     │  Tick 0: [6.92, 6.91, 6.90, ...]  (higher early)        │      │   │
│  │     │  Tick 1: [6.91, 6.90, 6.89, ...]                        │      │   │
│  │     │  Tick 2: [6.90, 6.89, 6.88, ...]                        │      │   │
│  │     │  ...                                                     │      │   │
│  │     │  Tick 7: [6.85, 6.84, 6.83, ...]  (lower later)         │      │   │
│  │     └─────────────────────────────────────────────────────────┘      │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  2. TICK SELECTION STRATEGIES                                        │   │
│  │                                                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  "min_loss" (default)                                          │  │   │
│  │  │                                                                 │  │   │
│  │  │  For each position, select tick with minimum loss:             │  │   │
│  │  │                                                                 │  │   │
│  │  │    selected_ticks = argmin(tick_losses, dim=0)                 │  │   │
│  │  │    loss = tick_losses.gather(selected_ticks).mean()            │  │   │
│  │  │                                                                 │  │   │
│  │  │  This trains the model on its BEST prediction at each pos.    │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  "max_certainty"                                               │  │   │
│  │  │                                                                 │  │   │
│  │  │  For each position, select tick with highest confidence:       │  │   │
│  │  │                                                                 │  │   │
│  │  │    probs = softmax(logits)                                     │  │   │
│  │  │    certainty = probs.max(dim=-1)                               │  │   │
│  │  │    selected_ticks = argmax(certainty, dim=0)                   │  │   │
│  │  │    loss = tick_losses.gather(selected_ticks).mean()            │  │   │
│  │  │                                                                 │  │   │
│  │  │  Trains on the model's MOST CONFIDENT prediction.             │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  "weighted"                                                    │  │   │
│  │  │                                                                 │  │   │
│  │  │  Soft attention over ticks based on inverse loss:             │  │   │
│  │  │                                                                 │  │   │
│  │  │    weights = softmax(-tick_losses / tau, dim=0)                │  │   │
│  │  │    loss = (weights * tick_losses).sum(dim=0).mean()            │  │   │
│  │  │                                                                 │  │   │
│  │  │  Differentiable version of min_loss.                          │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  "all"                                                         │  │   │
│  │  │                                                                 │  │   │
│  │  │  Average loss across all ticks (most stable):                  │  │   │
│  │  │                                                                 │  │   │
│  │  │    loss = tick_losses.mean()                                   │  │   │
│  │  │                                                                 │  │   │
│  │  │  Good for early training / warmup.                             │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  "last"                                                        │  │   │
│  │  │                                                                 │  │   │
│  │  │  Always use final tick:                                        │  │   │
│  │  │                                                                 │  │   │
│  │  │    loss = tick_losses[-1].mean()                               │  │   │
│  │  │                                                                 │  │   │
│  │  │  Standard non-adaptive training.                               │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  OUTPUT:                                                                    │
│    loss: scalar                                                             │
│    metrics: {                                                               │
│      per_tick_loss: (T,),        # average loss at each tick               │
│      avg_selected_tick: scalar,  # mean tick index (measure of "thinking") │
│      tick_distribution: (T,),    # how often each tick is selected         │
│    }                                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow Example

Here's a trace through the entire model for a single forward pass:

```
═══════════════════════════════════════════════════════════════════════════════
INPUT: tokens = [[Hello, world, !], [How, are, you]]  (B=2, S=3)
═══════════════════════════════════════════════════════════════════════════════

1. TOKEN EMBEDDING
   ─────────────────────────────────────────────────────────────────────────
   tokens (2, 3) → Embedding → x (2, 3, 128)

2. INPUT ENCODER
   ─────────────────────────────────────────────────────────────────────────
   x (2, 3, 128) → TransformerBlock × 2 → encoded (2, 3, 128)

   encoded → K_proj → RoPE → static_K (2, 3, 4, 32)   [4 heads, 32 dim]
   encoded → V_proj →      → static_V (2, 3, 4, 32)

3. CTM CORE (n_layer=2, num_ticks=4, total_steps=8)
   ─────────────────────────────────────────────────────────────────────────

   initial_state = encoded (2, 3, 128)
   global_history = zeros(2, 3, 8, 128)

   ┌─────────────────────────────────────────────────────────────────────┐
   │ LAYER 0                                                             │
   ├─────────────────────────────────────────────────────────────────────┤
   │                                                                      │
   │ Tick 0 (global_step=0):                                             │
   │   history = []  (empty)                                              │
   │   sync = zeros(2, 3, 512)                                            │
   │   nlm_out = zeros(2, 3, 128)                                         │
   │   state, post_act = Layer0(state, sync, nlm_out, K, V)              │
   │   global_history[:,:,0] = post_act                                   │
   │   all_states.append(state)                                           │
   │                                                                      │
   │ Tick 1 (global_step=1):                                             │
   │   history = global_history[:,:,0:1]  # 1 prior step                 │
   │   sync = GlobalSync(history)         # correlations from 1 step     │
   │   nlm_out = GlobalNLM(history)                                       │
   │   state, post_act = Layer0(state, sync, nlm_out, K, V)              │
   │   global_history[:,:,1] = post_act                                   │
   │   all_states.append(state)                                           │
   │                                                                      │
   │ Tick 2 (global_step=2):                                             │
   │   history = global_history[:,:,0:2]  # 2 prior steps                │
   │   ...                                                                │
   │                                                                      │
   │ Tick 3 (global_step=3):                                             │
   │   history = global_history[:,:,0:3]  # 3 prior steps                │
   │   ...                                                                │
   │                                                                      │
   └─────────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────────┐
   │ LAYER 1                                                             │
   ├─────────────────────────────────────────────────────────────────────┤
   │                                                                      │
   │ Tick 0 (global_step=4):                                             │
   │   history = global_history[:,:,0:4]  # ALL 4 prior steps            │
   │   ════════════════════════════════   (includes Layer 0!)            │
   │   sync = GlobalSync(history)         # correlations span Layer 0   │
   │   nlm_out = GlobalNLM(history)       # NLM sees full history        │
   │   state, post_act = Layer1(state, sync, nlm_out, K, V)              │
   │   global_history[:,:,4] = post_act                                   │
   │   all_states.append(state)                                           │
   │                                                                      │
   │ Tick 1 (global_step=5):                                             │
   │   history = global_history[:,:,0:5]  # 5 prior steps                │
   │   ...                                                                │
   │                                                                      │
   │ Tick 2 (global_step=6):                                             │
   │   history = global_history[:,:,0:6]  # 6 prior steps                │
   │   ...                                                                │
   │                                                                      │
   │ Tick 3 (global_step=7):                                             │
   │   history = global_history[:,:,0:7]  # 7 prior steps                │
   │   ...                                                                │
   │                                                                      │
   └─────────────────────────────────────────────────────────────────────┘

   OUTPUTS:
     final_state: (2, 3, 128)
     all_states: [s0, s1, s2, s3, s4, s5, s6, s7]  # 8 states

4. OUTPUT PROJECTION
   ─────────────────────────────────────────────────────────────────────────

   For each state in all_states:
     logits_t = lm_head(final_norm(state_t))  # (2, 3, vocab_size)

   all_logits: [logits_0, logits_1, ..., logits_7]  # 8 sets of logits

5. LOSS COMPUTATION (with tick selection)
   ─────────────────────────────────────────────────────────────────────────

   For each tick t:
     shift_logits = logits_t[:, :-1]    # (2, 2, V) - predict pos 1,2
     shift_labels = labels[:, 1:]       # (2, 2)    - true tokens 1,2
     loss_t = CrossEntropy(shift_logits, shift_labels)  # (2, 2)

   tick_losses: (8, 2, 2)  # 8 ticks, batch 2, seq_len-1

   With "min_loss" selection:
     For each (batch, position):
       selected_tick = argmin over ticks
     loss = average of selected losses

═══════════════════════════════════════════════════════════════════════════════
```

---

## Memory Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Memory Considerations                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GLOBAL HISTORY BUFFER                                                      │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  Shape: (B, S, n_layer × num_ticks, D)                                      │
│  Dtype: bfloat16 (half memory vs float32)                                   │
│                                                                              │
│  Example:                                                                   │
│    B=8, S=512, n_layer=6, num_ticks=8, D=768                               │
│    total_steps = 6 × 8 = 48                                                 │
│    Memory = 8 × 512 × 48 × 768 × 2 bytes = 302 MB                          │
│                                                                              │
│  STATIC KV CACHE                                                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  Shape: K (B, S, n_kv_head, head_dim), V (B, S, n_kv_head, head_dim)       │
│  Dtype: bfloat16                                                            │
│                                                                              │
│  Example:                                                                   │
│    B=8, S=512, n_kv_head=4, head_dim=64                                    │
│    Memory (K+V) = 2 × 8 × 512 × 4 × 64 × 2 bytes = 4 MB                    │
│                                                                              │
│  ALL STATES (for tick selection)                                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  List of n_layer × num_ticks tensors, each (B, S, D)                       │
│  (These are cloned copies for the forward pass)                             │
│                                                                              │
│  Example:                                                                   │
│    48 × (8 × 512 × 768 × 4 bytes) = 604 MB (float32)                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Reference

```python
CTMConfig(
    # Core dimensions
    vocab_size=50257,        # Vocabulary size
    d_model=768,             # Hidden dimension (D)
    n_head=12,               # Query attention heads
    n_kv_head=4,             # KV attention heads (GQA)
    n_layer=6,               # Number of CTM layers
    max_seq_len=2048,        # Maximum sequence length
    d_ff=None,               # FFN dimension (auto: 2 × 4D/3, rounded to 256)

    # CTM-specific
    num_ticks=8,             # Internal iterations per layer
    nlm_hidden=64,           # NLM hidden dimension
    nlm_depth=2,             # NLM MLP depth
    sync_pairs=512,          # Sampled neuron pairs for sync

    # Training
    dropout=0.0,             # Dropout rate
    rope_theta=500000.0,     # RoPE base frequency
    use_gradient_checkpointing=False,
)
```

---

## Key Equations

### Synchronization (S^t)

For neuron pair (i, j) with learnable decay r_{ij}:

```
R_{ij}^t = [exp(-r_{ij}·(t-1)), exp(-r_{ij}·(t-2)), ..., exp(0)]

S_{ij}^t = (Z_i^{0:t})^T · diag(R_{ij}^t) · Z_j^{0:t} / sqrt(sum(R_{ij}^t))
```

Where Z_i^{0:t} is the history of post-activations for neuron i.

### NLM Update

For each neuron d:

```
h_d = GELU(W_d^{in} · history_d + b_d^{in})
out_d = W_d^{out} · h_d + b_d^{out}
```

Where history_d is the temporal history of neuron d's activations.

### CTMLayer Update

```
state = state + norm(nlm_out)                           # integrate NLM
obs = CrossAttn(norm(state), static_K, static_V)        # query data
synapse_out = SynapseModel(state, obs, sync)            # integrate all
state = state + synapse_out
state = state + FFN(norm(state))                        # transform
post_act = norm(state)                                  # for history
```
