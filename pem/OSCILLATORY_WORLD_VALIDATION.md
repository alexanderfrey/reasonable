# Oscillatory World Model Validation - 2026-01-21

## Summary

Validation of the oscillatory world model implementation (commit 75d40fd) revealed **2 critical bugs** and **1 broken test suite**. All issues have been **FIXED AND VERIFIED**.

---

## Bugs Found and Fixed

### [FIXED] BUG #1: frequencies.detach() blocks all learning

**Location:** `pem/oscillatory_world.py`

**Original Problem:** The `.detach()` call and `torch.no_grad()` blocks prevented gradients from flowing to the `frequencies` parameter.

**Fix Applied:** Refactored `forward()` to:
1. Keep `self.phases` buffer DETACHED (persistent state across batches)
2. Compute DIFFERENTIABLE contributions locally: `effective_phase = self.phases.detach() + freq_contribution + phase_shift`
3. Update buffer safely with `torch.no_grad()` after computing differentiable output

**Verification:**
```
=== GRADIENT FLOW TEST (after fix) ===
frequencies.grad norm: 15.185818
✓ FIXED: frequencies now receive gradients!

=== ALL PARAMETER GRADIENTS ===
✓ frequencies: norm=15.185818
✓ base_amplitudes: norm=1.850612
✓ amp_modulator.0.weight: norm=0.386060
✓ phase_modulator.0.weight: norm=1.046561
✓ output_proj.weight: norm=11.378499
(all 12 parameters receive gradients)

=== MULTIPLE FORWARD CALLS ===
After 3 forward calls - frequencies.grad norm: 32.325695
✓ No in-place modification error!
```

---

### [FIXED] BUG #2: update_gate_value field no longer exists

**Location:** `pem/train_pem_global.py:965-966`

**Original Problem:** Training crashed with `AttributeError: 'GlobalSyncOutput' object has no attribute 'update_gate_value'`

**Fix Applied:**
```python
# Now uses oscillator_metrics (NamedTuple) instead:
if final_output.global_sync.oscillator_metrics is not None:
    osc_metrics = final_output.global_sync.oscillator_metrics
    osc_dict = osc_metrics._asdict() if hasattr(osc_metrics, '_asdict') else osc_metrics
    for key, value in osc_dict.items():
        if isinstance(value, (int, float)):
            metrics[f'world_state/osc_{key}'] = value
```

**Verification:** Training completes without errors.

---

### [FIXED] BUG #3: Test suite out of date

**Location:** `pem/test_global_sync.py:35`

**Original Problem:** Test expected tuple unpacking but `CTMCore.forward()` returns `CTMCoreOutput` NamedTuple.

**Fix Applied:**
```python
result = core(x)
post_act = result.post_activations
sync_matrix = result.sync_matrix
output = result.output
all_outputs = result.all_outputs
all_activations = result.all_activations
```

**Verification:**
```
=== Testing CTM Base ===
CTM Base: PASSED

=== Testing PredictionCTM ===
PredictionCTM: PASSED

=== Testing SurpriseCTM ===
SurpriseCTM: PASSED

=== Testing GlobalSyncModule ===
GlobalSyncModule: PASSED

=== Testing PEMLoopGlobal ===
Loss: 1.8347
(all tests pass)
```

---

## Training Verification

30-step training run completed successfully:

```
Step    10 | Loss: 1.8185 | Oscillator | amp=0.998 active=1.00 mod(a/φ)=0.036/0.030 ✓
Step    20 | Loss: 1.5723 | Oscillator | amp=1.000 active=1.00 mod(a/φ)=0.035/0.029 ✓
Step    30 | Loss: 1.2950 | Oscillator | amp=1.000 active=1.00 mod(a/φ)=0.033/0.026 ✓

All systems nominal ✓
```

Loss decreased from 1.82 to 1.30 over 30 steps with oscillator metrics being logged correctly.

---

## Remaining Design Considerations (Not Bugs)

### Mean compression loses context
**Location:** `pem/global_sync.py:596`
```python
sync_compressed = sync.mean(dim=(0, 1))  # (sync_pairs,)
```
Averaging over batch and sequence may lose position-specific information. Consider attention-weighted compression if needed.

### World state only affects z_0
**Location:** `pem/ctm_base.py:549-553`

World state is added only to initial post-activations. With T=4 ticks, influence may diminish. Consider injecting at multiple ticks if stronger world state influence is needed.

---

## Final Validation Status

| Component | Status |
|-----------|--------|
| Oscillatory phase evolution | ✓ Works |
| Amplitude modulation network | ✓ Works, receives gradients |
| Phase modulation network | ✓ Works, receives gradients |
| Output projection | ✓ Works, receives gradients |
| **Frequency learning** | ✓ **FIXED** - now receives gradients |
| **Training loop integration** | ✓ **FIXED** - no crashes |
| **Test suite** | ✓ **FIXED** - all tests pass |
| GlobalSyncOutput fields | ✓ Correct oscillator fields exist |

---

## Files Modified

1. `pem/oscillatory_world.py` - Refactored forward() for proper gradient flow
2. `pem/train_pem_global.py` - Fixed stale update_gate_value reference
3. `pem/test_global_sync.py` - Updated to use CTMCoreOutput NamedTuple

---

## Conclusion

All critical bugs have been fixed and verified. The oscillatory world model now:
- Learns frequencies through gradient descent
- Integrates correctly with the training loop
- Passes all unit tests
- Shows healthy training dynamics with decreasing loss

The implementation is ready for further experimentation and training.
