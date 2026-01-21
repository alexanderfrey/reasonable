# PEM Global Sync Architecture Diagram

## Current Implementation (train_pem_global.py)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         PEM LOOP WITH GLOBAL SYNC                                       │
│                                                                                         │
│   Input: Token Embeddings from Backbone (B, S, d_model=1536)                            │
│                              │                                                          │
│                              ▼                                                          │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │                    LOOP ITERATION (repeat num_steps times)                       │  │
│   │                                                                                  │  │
│   │   State: observation (B,S,d_model), cumulative_sync, world_state                 │  │
│   │                              │                                                   │  │
│   │   ┌──────────────────────────┼──────────────────────────┐                        │  │
│   │   │          STATE COMBINER (Gated Fusion)              │                        │  │
│   │   │                                                     │                        │  │
│   │   │   features ──┬──► [features, observation] ──► gate_net ──► gate [0,1]        │  │
│   │   │              │                                        │                      │  │
│   │   │   observation─┘   [features, observation] ──► transform                      │  │
│   │   │                                                       │                      │  │
│   │   │              loop_features = features + gate * (transform - features)        │  │
│   │   │                              │                                               │  │
│   │   │              (gate ≈ 0.30, initialized from -0.85)                           │  │
│   │   └──────────────────────────────┼───────────────────────────────────────────────┘  │
│   │                                  │                                                  │
│   │                                  ▼                                                  │
│   │   ┌──────────────────────────────────────────────────────────────────────────────┐  │
│   │   │                         CTM MODULES                                          │  │
│   │   │                                                                              │  │
│   │   │    ┌─────────────────────┐              ┌─────────────────────┐              │  │
│   │   │    │   PREDICTION CTM    │              │    SURPRISE CTM     │              │  │
│   │   │    │                     │              │                     │              │  │
│   │   │    │  z_0 = init(loop_f) │              │  z_0 = init(loop_f) │              │  │
│   │   │    │      + world_bias ◄─┼──────────────┼──── world_state     │              │  │
│   │   │    │                     │              │                     │              │  │
│   │   │    │  for t in T ticks:  │              │  for t in T ticks:  │              │  │
│   │   │    │    a_t = Synapse    │              │    a_t = Synapse    │              │  │
│   │   │    │    z_t = NLM(A_t)   │              │    z_t = NLM(A_t)   │              │  │
│   │   │    │    z_t += tick_bias ◄──────────────┼──── world_state     │              │  │
│   │   │    │    S_t = Sync(Z_t)  │              │    S_t = Sync(Z_t)  │              │  │
│   │   │    │    o_t = CrossAttn  │              │    o_t = CrossAttn  │              │  │
│   │   │    │                     │              │                     │              │  │
│   │   │    │  Output:            │              │  Output:            │              │  │
│   │   │    │   - predictions     │              │   - surprise mag    │              │  │
│   │   │    │   - Z_history       │              │   - Z_history       │              │  │
│   │   │    │   - certainty       │              │   - certainty       │              │  │
│   │   │    └──────────┬──────────┘              └──────────┬──────────┘              │  │
│   │   │               │                                    │                         │  │
│   │   └───────────────┼────────────────────────────────────┼─────────────────────────┘  │
│   │                   │                                    │                            │
│   │                   └──────────────┬─────────────────────┘                            │
│   │                                  │                                                  │
│   │                                  ▼                                                  │
│   │   ┌──────────────────────────────────────────────────────────────────────────────┐  │
│   │   │                      GLOBAL SYNC MODULE                                      │  │
│   │   │                                                                              │  │
│   │   │   Z_pred_history ──► Sync(Z·Z^T) ──► sync_pred (B,S,sync_pairs)              │  │
│   │   │   Z_surp_history ──► Sync(Z·Z^T) ──► sync_surp (B,S,sync_pairs)              │  │
│   │   │                                          │                                   │  │
│   │   │                     ┌─────────────────────┘                                  │  │
│   │   │                     ▼                                                        │  │
│   │   │            Cross-Module Attention                                            │  │
│   │   │            (which modules sync together?)                                    │  │
│   │   │                     │                                                        │  │
│   │   │                     ▼                                                        │  │
│   │   │              global_sync (B,S,sync_pairs)                                    │  │
│   │   │                     │                                                        │  │
│   │   └─────────────────────┼────────────────────────────────────────────────────────┘  │
│   │                         │                                                           │
│   │                         ▼                                                           │
│   │   ┌──────────────────────────────────────────────────────────────────────────────┐  │
│   │   │                 OSCILLATORY WORLD MODEL                                      │  │
│   │   │                                                                              │  │
│   │   │   WRITE PATH (surprise-gated):                                               │  │
│   │   │   ┌────────────────────────────────────────────────────────────────────┐     │  │
│   │   │   │  features ──► compress ──► amp_modulator ──► amplitude_mod         │     │  │
│   │   │   │                        └──► phase_modulator ──► phase_shift        │     │  │
│   │   │   │                                                                    │     │  │
│   │   │   │  surprise ──► sigmoid(bias + surp*scale) ──► write_gate            │     │  │
│   │   │   │                                                 │                  │     │  │
│   │   │   │  gated_amp = amp_mod * write_gate               │                  │     │  │
│   │   │   │  modulated_amps = base_amps * (1 + gated_amp)   │                  │     │  │
│   │   │   │                                                 │                  │     │  │
│   │   │   │  effective_phase = phases + freq*dt + phase_shift*gate             │     │  │
│   │   │   │                                                                    │     │  │
│   │   │   │  memory_states = modulated_amps * sin(effective_phase)             │     │  │
│   │   │   │                           │                                        │     │  │
│   │   │   │                           ▼                                        │     │  │
│   │   │   │                 ┌─────────────────────┐                             │     │  │
│   │   │   │                 │  64 Oscillators     │                             │     │  │
│   │   │   │                 │  periods: 8 - 4096  │                             │     │  │
│   │   │   │                 │  (phrase to doc)    │                             │     │  │
│   │   │   │                 └──────────┬──────────┘                             │     │  │
│   │   │   └────────────────────────────┼───────────────────────────────────────┘     │  │
│   │   │                                │                                             │  │
│   │   │   READ PATH (sync-queried):    │                                             │  │
│   │   │   ┌────────────────────────────┼───────────────────────────────────────┐     │  │
│   │   │   │                            ▼                                       │     │  │
│   │   │   │  global_sync ──► Query projection ──► Q (B,S,d_world)              │     │  │
│   │   │   │  memory_states ──► Key projection ──► K (1,64,d_world)             │     │  │
│   │   │   │               └──► Value projection ─► V (1,64,d_world)            │     │  │
│   │   │   │                                                                    │     │  │
│   │   │   │              MultiheadAttention(Q, K, V)                           │     │  │
│   │   │   │                          │                                         │     │  │
│   │   │   │                          ▼                                         │     │  │
│   │   │   │              world_state (B, S, d_world)                           │     │  │
│   │   │   │              (position-specific context!)                          │     │  │
│   │   │   └────────────────────────────┬───────────────────────────────────────┘     │  │
│   │   │                                │                                             │  │
│   │   └────────────────────────────────┼─────────────────────────────────────────────┘  │
│   │                                    │                                                │
│   │                                    ▼                                                │
│   │   ┌──────────────────────────────────────────────────────────────────────────────┐  │
│   │   │                    PERCEPTION ATTENTION                                      │  │
│   │   │                                                                              │  │
│   │   │   global_sync ──► Query ──► Attend to features ──► new_observation           │  │
│   │   │                                                                              │  │
│   │   │   observation = α * old_observation + (1-α) * new_observation                │  │
│   │   │                 (α = observation_residual = 0.2)                             │  │
│   │   └──────────────────────────────────────────────────────────────────────────────┘  │
│   │                                    │                                                │
│   │                                    ▼                                                │
│   │                          [Next Loop Iteration]                                      │
│   │                                                                                     │
│   └─────────────────────────────────────────────────────────────────────────────────────┘
│                                        │                                                │
│                                        ▼                                                │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │                              LOSS COMPUTATION                                    │  │
│   │                                                                                  │  │
│   │   Prediction Loss (CTM Loss):                                                    │  │
│   │     - immediate: predict f[t+1] (exact next token, horizon=1)                    │  │
│   │     - shortterm: predict mean(f[t+1:t+65])                                       │  │
│   │     - longterm:  predict mean(f[t+1:t+257])                                      │  │
│   │     - Weighted: 1.0 * imm + 0.5 * short + 0.3 * long                             │  │
│   │                                                                                  │  │
│   │   Surprise Loss: MSE(predicted_surprise, raw_surprise)                           │  │
│   │                                                                                  │  │
│   │   Loop Improvement Loss: penalize if later steps are worse                       │  │
│   │                                                                                  │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components Summary

| Component | Purpose | Key Config |
|-----------|---------|------------|
| **State Combiner** | Fuse features + observation | gate_init=-0.85 (≈30%) |
| **Prediction CTM** | Predict future embeddings | T=4 ticks, d_neurons=64 |
| **Surprise CTM** | Estimate prediction error | T=4 ticks, d_neurons=32 |
| **Global Sync** | Cross-module synchronization | sync_pairs=256 |
| **Oscillatory World** | Content-based temporal memory | 64 oscillators, periods 8-4096 |
| **Multi-tick Injection** | World state at every CTM tick | Enabled |
| **Perception Attention** | Update observation from sync | residual=0.2 |

---

## Data Flow Summary

1. **Input**: Token embeddings (B, S, 1536) from frozen backbone
2. **State Combiner**: Blend features with previous observation (gate ≈ 0.30)
3. **CTMs**: Process with world state bias at z_0 AND each tick
4. **Global Sync**: Compute cross-module synchronization patterns
5. **Oscillatory World**: Write (surprise-gated) + Read (sync-queried)
6. **Perception**: Update observation for next iteration
7. **Loss**: Multi-horizon prediction + surprise calibration

---

## CTM Internal Architecture (per module)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CTM CORE (T ticks)                          │
│                                                                     │
│   Input: loop_features (B, S, d_model)                              │
│   World: world_state (B, S, d_world)                                │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  INITIALIZATION (t=0)                                       │   │
│   │                                                             │   │
│   │  z_0 = x_to_z0(loop_features)      # (B, S, d_neurons)      │   │
│   │      + world_to_z0(world_state)    # world bias             │   │
│   │                                                             │   │
│   │  o_0 = init_observation(loop_features)  # initial attended  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  TICK LOOP (t = 1..T)                                       │   │
│   │                                                             │   │
│   │  for t in range(T):                                         │   │
│   │      # 1. Synapse: combine z_t and observation              │   │
│   │      a_t = Synapse(z_t, o_t)        # pre-activations       │   │
│   │                                                             │   │
│   │      # 2. History: maintain sliding window                  │   │
│   │      A_history.append(a_t)          # (B, S, d_neurons, M)  │   │
│   │                                                             │   │
│   │      # 3. NLM: per-neuron MLPs process history              │   │
│   │      z_t = NLM(A_history)           # post-activations      │   │
│   │                                                             │   │
│   │      # 4. Multi-tick world injection (if enabled)           │   │
│   │      z_t += world_to_tick(world_state)  # continuous bias   │   │
│   │                                                             │   │
│   │      # 5. Sync: compute synchronization matrix              │   │
│   │      Z_history.append(z_t)                                  │   │
│   │      S_full, S_out, S_internal = Sync(Z_history)            │   │
│   │                                                             │   │
│   │      # 6. Cross-attention: sync-derived query to features   │   │
│   │      o_t = CrossAttn(Q=S_internal, KV=loop_features)        │   │
│   │      o_t = α * o_t_old + (1-α) * o_t  # residual blend      │   │
│   │                                                             │   │
│   │      # 7. Output at this tick                               │   │
│   │      y_t = sync_to_output(S_out)                            │   │
│   │                                                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│   Output:                                                           │
│     - z_T: final post-activations (for global sync)                 │
│     - Z_history: all tick activations (for sync computation)        │
│     - all_outputs: predictions at each tick (for CTM loss)          │
│     - certainty: output stability across ticks                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Oscillatory World Model Details

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OSCILLATORY WORLD MODEL                          │
│                                                                     │
│   LEARNABLE PARAMETERS:                                             │
│     - frequencies: (64,) oscillator frequencies                     │
│     - base_amplitudes: (64,) base amplitude per oscillator          │
│     - amp_modulator: MLP (256 → 128 → 64) content → amplitude       │
│     - phase_modulator: MLP (256 → 128 → 64) content → phase         │
│     - output_proj: Linear (64 → 256)                                │
│                                                                     │
│   RUNTIME BUFFERS (persistent state):                               │
│     - phases: (64,) current phase of each oscillator                │
│     - current_amplitudes: (64,) current modulated amplitudes        │
│                                                                     │
│   TIMESCALE HIERARCHY:                                              │
│     Oscillator 0:   period = 8     (phrase-level, ~8 tokens)        │
│     Oscillator 31:  period = 512   (paragraph-level)                │
│     Oscillator 63:  period = 4096  (document-level)                 │
│                                                                     │
│   MEMORY CAPACITY:                                                  │
│     - 64 oscillators × 2 values (amp, phase) = 128 floats           │
│     - Total params: ~99K (oscillators) + ~360K (readout) = ~459K    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Ablation Results (2026-01-21)

With `--immediate_horizon 1` (exact next-token prediction):

| Metric | With World Model | Without World Model |
|--------|------------------|---------------------|
| **ΔLoss** | **+0.133** | +0.001 |
| imm improvement | 0.161→0.102 (**-37%**) | 0.113→0.112 (**-1%**) |
| short improvement | 0.053→0.015 (**-72%**) | 0.012→0.012 (**0%**) |
| long improvement | 0.048→0.012 (**-75%**) | 0.009→0.009 (**0%**) |

**Conclusion**: The oscillatory world model is essential for loop iterations to provide value.

---

## Training Command

```bash
python -m pem.train_pem_global \
    --dataset local \
    --data_dir /path/to/texts \
    --multi_tick_world_injection \
    --immediate_horizon 1
```

Key flags:
- `--multi_tick_world_injection`: Inject world state at every CTM tick
- `--immediate_horizon 1`: Predict exact next token (harder task)
- `--num_oscillators 64`: Number of oscillators (memory slots)
- `--disable_oscillatory_world`: Ablation without world model
