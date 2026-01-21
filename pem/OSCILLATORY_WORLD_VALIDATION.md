# Oscillatory World Model Validation - 2026-01-21

## Summary

Validation of the oscillatory world model implementation (commit 75d40fd) revealed **3 critical bugs** and **1 broken test suite**. All issues have been **FIXED AND VERIFIED**.

Subsequently, the architecture was **redesigned** to use content-based memory with surprise gating, which improved the ablation benefit from **-0.6% to +2.6%**.

---

## Architecture Evolution

### Original Architecture (Sync-based) - NOT WORKING
```
Sync (processing state) → Mean compress → Oscillators → Broadcast world_state
```
**Problem:** Sync patterns capture HOW neurons fire together, not WHAT content is being processed. The world model couldn't store meaningful context.

### New Architecture (Content-based) - WORKING
```
Features (content) ──────► Oscillators ◄────── Surprise (importance gate)
                               │
                               ▼
Sync (query) ─────────► Cross-attention ─────► Position-specific context
```

**Key insight:** The world model should store **what happened** (content), gated by **what mattered** (surprise), retrieved by **what's relevant now** (sync).

---

## Bugs Found and Fixed

### [FIXED] BUG #1: frequencies.detach() blocks all learning

**Location:** `pem/oscillatory_world.py`

**Original Problem:** The `.detach()` call and `torch.no_grad()` blocks prevented gradients from flowing to the `frequencies` parameter.

**Fix Applied:** Refactored `forward()` to:
1. Keep `self.phases` buffer DETACHED (persistent state across batches)
2. Compute DIFFERENTIABLE contributions locally: `effective_phase = self.phases.detach() + freq_contribution + phase_shift`
3. Update buffer safely with `torch.no_grad()` after computing differentiable output

---

### [FIXED] BUG #2: update_gate_value field no longer exists

**Location:** `pem/train_pem_global.py:965-966`

**Original Problem:** Training crashed with `AttributeError: 'GlobalSyncOutput' object has no attribute 'update_gate_value'`

**Fix Applied:** Use `oscillator_metrics._asdict()` for NamedTuple conversion.

---

### [FIXED] BUG #3: PEM loop uses non-differentiable world_state

**Location:** `pem/pem_loop_global.py:572`

**Original Problem:** The PEM loop called `get_world_state()` which uses the non-differentiable `read()` method.

**Fix Applied:**
1. Added `world_state` field to `PEMLoopGlobalState` NamedTuple
2. Store differentiable `global_sync_output.world_state` in new_state
3. Use `state.world_state` (from previous step) in prediction

---

### [FIXED] BUG #4: Test suite out of date

**Location:** `pem/test_global_sync.py:35`

**Original Problem:** Test expected tuple unpacking but `CTMCore.forward()` returns `CTMCoreOutput` NamedTuple.

**Fix Applied:** Use NamedTuple attribute access.

---

## Architectural Improvements Made

### 1. Attention-weighted sync compression
**Location:** `pem/global_sync.py`

Replaced mean compression with learned attention-weighted pooling:
```python
attn_scores = self.sync_compression_attn(sync)  # (B, S, 1)
attn_weights = F.softmax(attn_scores.view(-1), dim=0)
sync_compressed = (attn_weights * sync).sum(dim=(0, 1))
```

### 2. Position-specific world context via cross-attention
**Location:** `pem/global_sync.py`

Added cross-attention where positions query oscillator memory:
```python
# Query: sync patterns (B, S, sync_pairs) -> (B, S, d_world_output)
query = self.osc_query_proj(sync)

# Key/Value: oscillator memory states
memory = osc_output.memory_states
key = self.osc_key_proj(memory)
value = self.osc_value_proj(memory)

# Cross-attention output: position-specific world context
attn_out, _ = self.osc_cross_attn(query, key, value)
world_state = self.osc_output_norm(attn_out + query)
```

### 3. Content-based memory writing with surprise gating
**Location:** `pem/oscillatory_world.py`

Oscillators now receive:
- **Features** (content) for amplitude modulation - determines WHAT to store
- **Surprise** (importance) for write gating - determines HOW STRONGLY to write

```python
def forward(self, features, surprise=None, dt=1.0):
    # Surprise gating: higher surprise = more important to remember
    if surprise is not None:
        write_gate = torch.sigmoid(self.surprise_gate_bias + surprise * self.surprise_gate_scale)

    # Features determine what to store
    amp_mod = self.amp_modulator(features)
    gated_amp_mod = amp_mod * write_gate  # Scale by importance

    modulated_amps = self.base_amplitudes * (1 + gated_amp_mod * self.config.max_amp_modulation)
```

### 4. CTMCore handles position-specific world_state
**Location:** `pem/ctm_base.py`

Updated to handle both broadcast and position-specific world_state:
```python
if world_state.dim() == 1:
    # Broadcast mode: (d_world_state,) -> (B, S, d_neurons)
    z_world = self.world_to_z0(world_state)
    z_t = z_t + z_world.unsqueeze(0).unsqueeze(0)
else:
    # Position-specific mode: (B, S, d_world_state) -> (B, S, d_neurons)
    z_world = self.world_to_z0(world_state)
    z_t = z_t + z_world
```

---

## Evaluation Results Comparison

### Sync-based (original, after bug fixes)
```
✓ CONTEXT BENEFIT: 44.6% improvement late vs early
✗ ABLATION: World model adds -0.6% extra context benefit  ← NEGATIVE!
✓ FREQ LEARNING: 48.4% change
✓ UTILIZATION: 100.0% active
```

### Content-based with surprise gating (final)
```
✓ CONTEXT BENEFIT: 48.8% improvement late vs early
✗ ABLATION: World model adds +2.6% extra context benefit  ← POSITIVE!
✓ FREQ LEARNING: 86.3% change
✓ UTILIZATION: 100.0% active
```

### Improvement Summary
| Metric | Sync-based | Content-based | Change |
|--------|------------|---------------|--------|
| Ablation benefit | -0.6% | **+2.6%** | +3.2% |
| Frequency learning | 48.4% | **86.3%** | +37.9% |
| Context benefit | 44.6% | 48.8% | +4.2% |

The world model now provides **positive** extra context benefit instead of negative.

---

## Files Modified

1. `pem/oscillatory_world.py` - Content-based memory with surprise gating
2. `pem/global_sync.py` - Cross-attention, feature compression, position-specific output
3. `pem/ctm_base.py` - Handle position-specific world_state
4. `pem/pem_loop_global.py` - Pass features and surprise to global_sync
5. `pem/train_pem_global.py` - Fixed stale update_gate_value reference
6. `pem/test_global_sync.py` - Updated to use CTMCoreOutput NamedTuple

---

## Current Status

All tests pass:
```
============================================================
ALL TESTS PASSED
============================================================
```

The architecture is functional and showing positive results, but not yet at the 5% ablation threshold.

---

## Future Work

1. **Hyperparameter tuning**
   - Number of oscillators (currently 64)
   - Frequency range (currently 8-4096 steps)
   - Surprise gate bias/scale
   - Cross-attention heads

2. **Longer training**
   - Current evaluation: 5 epochs, 20 documents
   - May need more data for world model to learn useful patterns

3. **Alternative surprise signals**
   - Currently using surprise magnitude
   - Could try prediction error, attention entropy, etc.

4. **Multi-tick world state injection**
   - Currently only affects z_0
   - Could inject at each CTM tick for stronger influence

5. **Auxiliary prediction loss**
   - Directly train oscillators to predict future content
   - Would give clearer learning signal

---

## How to Resume

### Run tests
```bash
python -m pem.test_global_sync
```

### Run evaluation
```bash
python -m pem.evaluate_world_model --num_docs 20 --num_epochs 5
```

### Quick evaluation
```bash
python -m pem.evaluate_world_model --quick
```

### Training with world model
```bash
python -m pem.train_pem_global \
    --d_model 256 \
    --use_oscillatory_world \
    --num_oscillators 64 \
    --batch_size 2 \
    --max_steps 1000
```
