# Bug Fixes and Architectural Improvements TODO

This document tracks known issues discovered during code review. Each item includes context, the problem, affected files, and proposed solutions.

---

## Critical Priority

### 1. Non-Causal Memory Retrieval

**Status:** ✅ COMPLETED (2026-01-02) - Implemented "Query Before Forward" approach

**Problem:**
Memory retrieval queries using `hidden_states[:, -1, :]` (the last token's hidden state, which has attended to the full sequence via self-attention), then injects the retrieved memory into ALL positions before recomputing logits. This means earlier tokens' predictions are influenced by information from later tokens via the memory pathway.

This breaks causal language modeling during training - position 0's logit is influenced by what's at position N.

**Affected Files:**
- `experiential.py:1157-1192` (MemoryAugmentedGPT.forward)

**Current Code Flow:**
```python
# 1. GPT forward pass (causal attention)
logits, hidden_states = self.gpt(input_ids, ...)

# 2. Query memory with LAST position (has seen everything)
query_state = hidden_states[:, -1, :]  # <-- Problem: contains future info
query = self.query_proj(query_state)
episodic_retrieved, _ = self.memory.retrieve_soft(query, ...)

# 3. Integrate memory into ALL positions
hidden_states = self._integrate_memory(hidden_states, episodic_retrieved, ...)

# 4. Recompute ALL logits (now position 0 sees future via memory)
logits = self.gpt.lm_head(self.gpt.final_norm(hidden_states))
```

**Proposed Solutions (choose one):**

**Option A: Query Before Forward (Recommended)**
Query memory using the PREVIOUS step's state (or a separate context embedding), not the current sequence's hidden states:
```python
def forward(self, input_ids, prev_memory_query=None, ...):
    # Query memory BEFORE seeing current sequence
    if prev_memory_query is not None and self.memory.size > 0:
        retrieved = self.memory.retrieve_soft(prev_memory_query, ...)

    # GPT forward with memory as prefix/conditioning
    logits, hidden_states = self.gpt(input_ids, memory_context=retrieved, ...)

    # Prepare query for NEXT step
    next_memory_query = hidden_states[:, -1, :].detach()

    return logits, hidden_states, next_memory_query
```

**Option B: Only Affect New Positions**
During training, only apply memory to positions that haven't been seen yet (useful for continuation):
```python
def _integrate_memory(self, hidden_states, retrieved, start_pos=0):
    # Only modify positions >= start_pos
    if start_pos > 0:
        hidden_states[:, start_pos:, :] = hidden_states[:, start_pos:, :] + retrieved
```

**Option C: Cross-Attention Integration**
Add memory as cross-attention context inside transformer layers (requires modifying GPT architecture):
```python
# In TransformerBlock
x = x + self.self_attn(x)
x = x + self.cross_attn(x, memory_bank)  # Memory accessed causally per-layer
x = x + self.ffn(x)
```

**Testing:**
After fix, verify with:
```python
# Prediction at position i should be identical regardless of tokens at positions > i
logits1 = model(tokens[:, :10])[:, 5, :]  # Predict token 6
logits2 = model(tokens[:, :20])[:, 5, :]  # Same prediction for token 6
assert torch.allclose(logits1, logits2)  # Should pass if causal
```

---

## High Priority

### 2. Inconsistent Persistent State Handling

**Status:** ✅ COMPLETED (2026-01-02) - Added reset_state() to train_memory_augmented.py

**Problem:**
Two training scripts handle persistent state oppositely, both incorrectly:

1. `pretrain.py:2027-2028` resets state EVERY batch → state never influences predictions (defeats the purpose)
2. `train_memory_augmented.py` NEVER resets → state leaks across unrelated shuffled documents

**Affected Files:**
- `pretrain.py:2025-2028`
- `train_memory_augmented.py:141-231` (train_epoch function)
- `train_resume_interruption.py` (new file, needs review)

**Current Code:**
```python
# pretrain.py - resets every batch (state is useless)
if experiential_module is not None and hasattr(experiential_module, 'reset_state'):
    experiential_module.reset_state(batch_size=batch["input_ids"].size(0))

# train_memory_augmented.py - never resets (state leaks)
for step, batch in enumerate(pbar):
    # No reset anywhere!
    logits, hidden, mem_out = memory_gpt(input_ids, ...)
```

**Proposed Solution:**

The fix depends on the data regime:

**For Shuffled Batches (independent sequences):**
Reset state at the START of each batch, but allow it to persist within the sequence:
```python
for batch in dataloader:
    model.experiential.reset_state(batch_size=batch_size)  # Fresh start
    # State can now build up within the sequence
    logits, hidden, mem_out = model(batch['input_ids'], ...)
```

**For Sequential/Document-Aware Batches (like train_resume_interruption.py):**
Reset state at document boundaries, persist within document:
```python
for doc_chunks in dataloader:
    model.reset_memory()  # New document = fresh memory
    model.experiential.reset_state()  # Fresh state

    for chunk in doc_chunks.chunks:
        # State persists across chunks of SAME document
        logits, hidden, mem_out = model(chunk, ...)
```

**Implementation Checklist:**
- [ ] Add `reset_state()` call at batch start in `train_memory_augmented.py`
- [ ] Audit `pretrain.py` - if using experiential, decide: sequential data (keep state) vs shuffled (reset)
- [ ] Add clear comments explaining the state management policy
- [ ] Consider adding a `state_management` parameter: `'reset_per_batch'`, `'reset_per_document'`, `'persistent'`

---

### 3. Closed Loop Components Not Trained

**Status:** ✅ COMPLETED (2026-01-02) - Updated memory_augmented_loss to use combined_experiential_loss

**Problem:**
The v0.3.2 "self-prediction closed loop" features are implemented but receive zero gradients:

1. `target` is detached (`experiential.py:690`), so `self_modulator` gets no gradient from `experiential_loss`
2. Training scripts use `experiential_loss`, not `combined_experiential_loss` which includes `meta_surprise_loss` and `self_mod_loss`
3. The `surprise_predictor` and `self_modulator` are dead code during training

**Affected Files:**
- `experiential.py:690` (target detachment)
- `experiential.py:836-892` (combined_experiential_loss - exists but unused)
- `pretrain.py:1785` (uses experiential_loss only)
- `train_memory_augmented.py:185` (uses memory_augmented_loss → experiential_loss)

**Current Code:**
```python
# experiential.py:690 - target is detached
'target': modulated_output.detach()  # No gradients to self_modulator!

# Training scripts only use basic loss
loss = experiential_loss(prediction, target)  # Doesn't train meta-surprise
```

**Proposed Solution:**

**Step 1: Update training scripts to use combined loss**
```python
# In train_memory_augmented.py and pretrain.py
from experiential import combined_experiential_loss

# Replace:
# exp_loss = experiential_loss(output['prediction'], output['target'])

# With:
exp_loss, loss_dict = combined_experiential_loss(
    output,
    exp_weight=1.0,
    meta_weight=0.1,      # Train surprise predictor
    self_mod_weight=0.1,  # Train self-modulator
)
```

**Step 2: Decide on target gradient flow**
The target detachment is intentional to prevent the predictor from "cheating" by making the target easy to predict. But we need gradients for self_modulator.

Option A: Separate losses (current `combined_experiential_loss` approach):
- Keep target detached for contrastive loss
- Add explicit `self_mod_loss` that trains the modulator

Option B: Partial gradient flow:
- Allow gradients through modulated_output for self_mod components only
- Use `detach()` selectively

**Step 3: Update memory_augmented_loss**
```python
# In experiential.py, update memory_augmented_loss to include meta/self-mod losses
def memory_augmented_loss(..., meta_weight=0.1, self_mod_weight=0.1):
    # ... existing LM loss ...

    if memory_output.get('prediction') is not None:
        exp_loss, exp_dict = combined_experiential_loss(
            memory_output,
            exp_weight=exp_weight,
            meta_weight=meta_weight,
            self_mod_weight=self_mod_weight,
        )
        # ... rest of function ...
```

**Testing:**
```python
# Verify gradients flow to meta-surprise components
model.zero_grad()
output = model.experiential(hidden_states)
loss, _ = combined_experiential_loss(output)
loss.backward()

assert model.experiential.surprise_predictor[0].weight.grad is not None
assert model.experiential.self_modulator[0].weight.grad is not None
```

---

## Medium Priority

### 4. Persistent State Always Detached (No Gradient Through Time)

**Status:** ✅ COMPLETED (2026-01-02) - Implemented truncated BPTT with tbptt_steps parameter

**Problem:**
The state update mechanism detaches gradients twice, making it impossible to train the state gate to learn what information to retain:

```python
# experiential.py:677-678
new_state = self._update_state(modulated_output.detach(), prev_state)
self._persistent_state = new_state.detach()  # Double detach!
```

The GRU-style gate (`self.state_gate`) cannot learn because:
1. Input `modulated_output` is detached
2. Output `new_state` is detached

**Affected Files:**
- `experiential.py:677-678` (state update)
- `experiential.py:542-550` (_update_state method)

**Why It's Detached:**
Intentional to prevent OOM from huge computation graphs across many forward passes. Without detachment, the graph would grow indefinitely.

**Proposed Solutions:**

**Option A: Truncated BPTT (Recommended)**
Allow gradients for K steps, then detach:
```python
def forward(self, hidden_states, ..., tbptt_steps=5):
    # ... compute new_state ...

    # Track how many steps since last detach
    self._steps_since_detach += 1

    if self._steps_since_detach >= tbptt_steps:
        self._persistent_state = new_state.detach()
        self._steps_since_detach = 0
    else:
        self._persistent_state = new_state  # Keep gradients!
```

**Option B: Auxiliary State Prediction Loss**
Train the state gate with a separate loss that doesn't require full backprop:
```python
# Predict what the next state should be (from a target)
state_pred_loss = F.mse_loss(new_state, target_state.detach())
```

**Option C: Accept Limitation**
Document that state gate uses heuristic initialization and doesn't learn. Focus training signal elsewhere.

**Implementation Notes:**
- Option A requires careful memory management
- Need to call `model.experiential.detach_state()` at appropriate intervals
- Consider making `tbptt_steps` a config parameter

---

### 5. Untrained Affect Heads (Valence/Arousal)

**Status:** ✅ COMPLETED (2026-01-02) - Removed affect from salience; now salience = surprise only

**Problem:**
Crystallization (memory storage) depends on salience, which is computed as:
```python
salience = surprise * arousal * valence.abs()
```

But `valence_head` and `arousal_head` have no supervision signal - they output whatever random initialization + incidental gradients produce. Memory storage decisions are essentially random with respect to emotional content.

**Affected Files:**
- `experiential.py:446-463` (affect head definitions)
- `experiential.py:629-649` (affect computation and salience)

**Proposed Solutions:**

**Option A: Remove Affect from Salience (Quick Fix)**
If we can't train affect, don't use it:
```python
# Simple salience based only on surprise
salience = surprise  # Remove affect dependency
```

**Option B: Heuristic Affect Labels**
Use proxy signals as weak supervision:
```python
# Heuristics for valence:
# - Positive words in vocabulary → positive valence
# - Loss decrease → positive valence
# - High perplexity → negative valence (confusion)

# Heuristics for arousal:
# - High surprise → high arousal
# - Rare tokens → high arousal
# - Fast-changing hidden states → high arousal
```

**Option C: Contrastive Affect Learning**
Train affect heads to distinguish between different types of content:
```python
# Assuming we have some content labels (e.g., dialogue vs action vs description)
# Train arousal to be high for action, low for exposition
affect_loss = contrastive_loss(arousal, content_type_labels)
```

**Option D: Self-Supervised Affect**
Use temporal consistency as supervision:
```python
# Affect should be smooth within a passage, change at boundaries
temporal_smoothness_loss = F.mse_loss(valence[:-1], valence[1:])
```

**Recommendation:**
Start with Option A (remove from salience) for correctness, then explore Option B/D if affect-aware memory is desired.

---

### 6. Wrong Dictionary Key in train_memory_augmented.py

**Status:** ✅ COMPLETED (2026-01-02) - Changed memory_size to episodic_size

**Problem:**
Code references `mem_out['memory_size']` but the actual key is `episodic_size`:

```python
# train_memory_augmented.py:215, 224
history['memory_size'].append(mem_out['memory_size'])  # KeyError!
```

**Affected Files:**
- `train_memory_augmented.py:163, 215, 224`

**Fix:**
```python
# Change all occurrences of mem_out['memory_size'] to:
mem_out['episodic_size']

# Or add for backwards compatibility in MemoryAugmentedGPT.forward():
memory_output['memory_size'] = memory_output['episodic_size']  # Alias
```

**This is a simple find-replace fix.**

---

## Architectural Questions to Resolve

These aren't bugs but design decisions that affect the fix approach:

### Q1: Causal Memory Integration Strategy

**Decision needed:** How should memory be integrated causally?

| Option | Complexity | Correctness | Performance |
|--------|------------|-------------|-------------|
| Query before forward | Low | High | Good |
| Per-token streaming | High | Perfect | Slow |
| Cross-attention layers | Medium | High | Medium |

**Recommendation:** Start with "query before forward" - simplest correct solution.

### Q2: What Should Persistent State Represent?

**Decision needed:** Is persistent state meant to be:
- A) A running summary of the document (requires sequential batching)
- B) A learned prior/bias for the model (can work with shuffled batches)
- C) A working memory for multi-step reasoning (requires careful reset policy)

This affects how we fix the state handling inconsistency.

### Q3: Train Self-Modulation or Keep Diagnostic?

**Decision needed:** Should meta-surprise and self-modulation be:
- A) Trained end-to-end (use combined_experiential_loss)
- B) Diagnostic signals only (current behavior, but then remove from architecture to reduce complexity)

**Recommendation:** If we keep the components, train them. Otherwise, remove dead code.

---

## Implementation Order

Suggested order based on dependencies and impact:

1. **Fix #6** (wrong key) - 5 minutes, prevents crashes
2. **Fix #2** (state handling) - 1 hour, foundational for other fixes
3. **Fix #1** (causal memory) - 2-4 hours, critical correctness issue
4. **Fix #3** (train closed loop) - 1-2 hours, makes v0.3.2 features actually work
5. **Fix #5** (affect heads) - 1 hour, either remove or add supervision
6. **Fix #4** (state gradients) - 2 hours, optional enhancement

---

## Validation Plan

After fixes, run these validations:

```bash
# 1. Unit tests pass
python -m pytest tests/

# 2. Causality check
python -c "
from experiential import MemoryAugmentedGPT
# ... test that predictions are causal ...
"

# 3. Gradient flow check
python -c "
# Verify gradients reach all trainable components
# - surprise_predictor
# - self_modulator
# - state_gate (if enabling TBPTT)
# - affect heads (if adding supervision)
"

# 4. Memory benefit improves with training
python train_resume_interruption.py --max_steps 1000
# Should see positive memory_benefit by end
```

---

## Recently Completed

### 7. Surprise Signal Overhaul (CE-Based)

**Status:** ✅ COMPLETED (2026-01-03)

**Problem:**
The original MLP-based surprise signal (predicting `h_end` from `h_mid`) was fundamentally flawed:
- The MLP only sees `h_mid`, but `h_end` depends on tokens `mid+1` to `end` which the MLP never sees
- This made the prediction task essentially impossible
- Surprise was stuck at ~1.0 ± 0.05 regardless of content

**Solution Implemented:**
Replaced MLP-based surprise with **relative surprisal × novelty** signal:

```python
# New surprise computation
s_t = CE(logits_t, token_t+1)                    # Per-token cross-entropy
excess_t = (s_t - ema_mu) / (ema_sigma + eps)    # Relative to EMA baseline
novelty_t = 1 - max_cosine(h_t, memory_keys)     # Unlike stored memories
surprise_t = relu(excess_t) * novelty_t          # Both conditions
chunk_surprise = mean(topk(surprise_t, k))       # Robust aggregation
```

**Key Benefits:**
1. Uses GPT's own prediction error (principled signal)
2. Normalized by running baseline (avoids storing rare proper nouns)
3. Novelty gating prevents redundant memory storage
4. Top-k aggregation is robust to outliers

**Files Modified:**
- `experiential.py`:
  - Added EMA buffers (`ema_mu`, `ema_sigma`, `ema_initialized`) to `ExperientialStream`
  - Added `compute_excess_surprisal()` method
  - Added `compute_novelty()` method
  - Added `compute_surprise_signal()` method
  - Added `get_keys()` to `EpisodicMemory`
  - Updated `ExperientialStream.forward()` to use new signal when `per_token_ce` provided
  - Updated `MemoryAugmentedGPT.forward()` to compute and pass `per_token_ce`

**Results:**
- **Before:** Surprise stuck at ~1.0 ± 0.05
- **After:** Surprise range [8.7, 21.7] with meaningful variation
- Narrative validation shows 100% correlation with structural markers

---

*Last updated: 2026-01-03*
*Review triggered by: Architecture audit*
