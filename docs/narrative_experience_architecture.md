# Narrative Experience Architecture

Design notes for modifying the transformer to "experience" narrative moments.

**Goal**: Build a model that maintains internal state reflecting what's happening in text — tracking scenes, characters, events, and narrative tension as it reads.

---

## 1. Current Architecture Baseline

The existing GPT implementation (`model.py`) is a standard autoregressive transformer:
- FlashAttention-2 with GQA
- Static KV cache for inference
- Fused RoPE, SwiGLU MLP
- No persistent state across sequences
- No explicit world modeling

Training (`pretrain.py`) uses:
- Chunked token streams with EOS as document separator
- Standard cross-entropy next-token prediction
- No document boundary awareness
- Independent sequences per batch

---

## 2. What "Experiencing Moments" Means Computationally

When humans read narrative:

| Human Experience | Computational Analog |
|------------------|---------------------|
| Mental model of scene | Persistent state vector tracking who/what/where |
| Surprise/tension | Prediction error signal |
| Remembering events | Episodic memory storage |
| Understanding updates | State transition mechanism |
| Anticipation | Future state prediction |

A standard transformer has none of this explicitly — any "understanding" is implicit and transient within a single forward pass.

---

## 3. Proposed Architecture: Latent Slots + Episodic Memory

### 3.1 Core Components

```
┌─────────────────────────────────────────────────────────┐
│  Latent Slots z_t ∈ R^{S × d}                           │
│  - Persistent across chunks                             │
│  - "Working memory" / current narrative state           │
│  - Updated via cross-attention + GRU                    │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│  Transformer Backbone                                   │
│  - Existing layers, conditioned on slots + memory       │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│  Episodic Memory M                                      │
│  - Key-value store of past events                       │
│  - Writer decides what to store (surprise-gated)        │
│  - Reader retrieves relevant context                    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Slot Update Mechanism

```python
# After processing chunk with transformer
h_t = transformer_output  # [B, seq_len, d_model]

# Cross-attention from slots to hidden states
z_tilde = cross_attention(
    query=z_t,           # [B, S, d]
    key=h_t,             # [B, seq_len, d]
    value=h_t
)

# GRU-style gated update
z_{t+1} = GRU(z_t, z_tilde)
```

### 3.3 Memory Interface

```python
# Memory item structure
@dataclass
class MemoryItem:
    key: Tensor      # [d] - for retrieval matching
    value: Tensor    # [d] - content to retrieve
    timestamp: int   # when stored
    salience: float  # importance score

# Write decision (surprise-gated)
surprise = 1 - cosine_sim(predicted_future, actual_future)
write_prob = sigmoid(a * surprise + b)

# Read (soft attention over memory)
query = project(current_hidden_state)
weights = softmax(query @ memory_keys.T / tau)
retrieved = weights @ memory_values
```

---

## 4. Training Losses

### 4.1 Loss A: Future Latent Prediction (Primary)

Forces slots to capture predictive state about narrative.

```python
# After reading chunk t, predict embedding of chunk t+1
predicted_future = g_theta(z_{t+1})  # projection head
actual_future = pool(encode(x_{t+1})).detach()  # stop-gradient

# InfoNCE contrastive loss
loss_pred = -log(
    exp(sim(predicted, actual) / tau) /
    sum(exp(sim(predicted, negatives) / tau))
)
```

**Why it matters**: Slots must capture "state of the story" to predict what comes next.

### 4.2 Loss B: Retrospective Reconstruction

Prevents slots from only being forward-looking.

```python
# Predict embedding of current chunk from updated slots
predicted_past = r_theta(z_{t+1})
actual_past = pool(encode(x_t)).detach()

loss_retro = mse(predicted_past, actual_past)
# Or contrastive version
```

### 4.3 Loss C: Temporal Consistency

Controls how much slots change per chunk.

```python
# Base drift penalty
drift = ||z_{t+1} - z_t||^2

# Gated by surprise (allow change when something happens)
loss_drift = (1 - alpha_t) * drift
# where alpha_t ∈ [0,1] is surprise/salience
```

### 4.4 Loss D: Slot Diversity

Prevents slot collapse.

```python
# Orthogonality regularization
Z = stack(slots)  # [S, d]
loss_ortho = ||Z @ Z.T - I||_F^2
```

### 4.5 Memory Budget

Prevents storing everything.

```python
loss_budget = lambda * E[write_probability]
# Or hard constraint: top-k events per N tokens
```

---

## 5. Training Tasks (Self-Supervised)

### 5.1 Resume After Interruption

Most important for forcing memory use.

```
1. Read chunks 1..t, build slots + memory
2. Discard transformer hidden states (but keep slots, memory)
3. Continue from chunk t+1 using only slots + memory
4. Compute loss on chunks t+1..T
```

If the model can't rely on full token history, it learns to store the right episodes.

### 5.2 Masked Entity Fill

Create synthetic retrieval queries from future text.

```
1. Find entity mention in chunk t+k: "John opened the letter"
2. Mask it: "_____ opened the letter"
3. Query memory with masked context
4. Model must retrieve earlier mention of John
```

### 5.3 Cue-Episode Contrastive Matching

Train memory keys to be retrievable.

```python
# Positive: cue from later text → episode from correct earlier time
# Negatives: episodes from other docs or wrong positions

loss_cue = InfoNCE(query=cue_embedding, positive=correct_episode, negatives=wrong_episodes)
```

---

## 6. Integration Challenges with Current Codebase

### 6.1 Data Pipeline Changes Required

Current `PretokenizedDataset` returns independent sequences. Need:

```python
class NarrativeChunkDataset(Dataset):
    """Yields consecutive chunks from same document."""

    def __init__(self, token_file, chunk_size, chunks_per_sample):
        # Track document boundaries (EOS positions)
        # Return sequences of consecutive chunks
        pass

    def __getitem__(self, idx):
        return {
            "chunks": [chunk_1, chunk_2, ..., chunk_n],  # consecutive
            "doc_id": document_identifier,
        }
```

### 6.2 FlashAttention Compatibility

FlashAttention doesn't support heterogeneous attention masks. Options:

1. **Prepend memory as tokens** — but causal mask is wrong (memory should be visible to all positions)
2. **Separate cross-attention layer** — adds latency, breaks flash-attn optimization
3. **Use flash-attn for self-attention, standard attention for memory** — mixed approach

Recommendation: Start with option 3, accept the speed hit during development.

### 6.3 Forward Pass Restructuring

Current:
```python
def forward(self, input_ids, input_pos=None):
    x = self.token_embedding(input_ids)
    for layer in self.layers:
        x = layer(x, cos, sin, kv_cache, input_pos)
    return self.lm_head(self.final_norm(x))
```

Required:
```python
def forward(self, input_ids, slots=None, memory=None, input_pos=None):
    x = self.token_embedding(input_ids)

    # Retrieve from memory
    if memory is not None:
        retrieved = self.memory_reader(x, memory)
        x = x + retrieved  # or concat, or cross-attend

    for layer in self.layers:
        x = layer(x, cos, sin, kv_cache, input_pos)

    # Update slots
    if slots is not None:
        slots = self.slot_updater(slots, x)

    logits = self.lm_head(self.final_norm(x))
    return logits, slots, memory_write_candidates
```

### 6.4 Training Loop Changes

Current loop processes independent batches. Need:

```python
for doc_chunks in dataloader:
    slots = initial_slots(batch_size)
    memory = empty_memory(batch_size)

    for chunk_idx, chunk in enumerate(doc_chunks):
        logits, slots, write_candidates = model(chunk, slots, memory)

        # Compute losses
        loss_lm = cross_entropy(logits, labels)
        loss_pred = future_prediction_loss(slots, next_chunk)
        # ... other losses

        # Update memory
        memory = memory_writer(memory, write_candidates, surprise)

        # Slots persist to next chunk (detach for TBPTT)
        if chunk_idx % tbptt_steps == 0:
            slots = slots.detach()
```

---

## 7. Alternative: Explicit World State Modeling

If slots don't self-organize into interpretable narrative elements, consider structured state:

```python
class NarrativeState(nn.Module):
    def __init__(self, d_model):
        self.location = nn.Parameter(torch.zeros(d_model))
        self.characters_present = nn.ParameterList([...])
        self.mood = nn.Parameter(torch.zeros(d_model))
        self.active_conflict = nn.Parameter(torch.zeros(d_model))
        self.time_embedding = nn.Parameter(torch.zeros(d_model))

        self.state_updater = StateUpdateHead(d_model)

    def update(self, hidden_states):
        # Predict: did location change? character enter/exit? mood shift?
        deltas = self.state_updater(hidden_states)
        # Apply updates
```

**Training signal**: Extract from text via:
- Scene break detection (location changes)
- Dialogue attribution (character presence)
- Sentiment analysis (mood)
- Named entity recognition (character tracking)

More supervision required, but more interpretable results.

---

## 8. Implementation Phases

### Phase 0: Data Pipeline (Prerequisite)
- [ ] Track document boundaries in tokenized data
- [ ] Create dataset yielding consecutive chunks from same document
- [ ] Modify dataloader for variable-length document batches

### Phase 1: Slots Only (No Memory)
- [ ] Add S learnable slot vectors to model
- [ ] Implement cross-attention from slots to hidden states
- [ ] Implement GRU update for slots
- [ ] Add Loss A (future prediction)
- [ ] Add Loss C (drift control)
- [ ] Add Loss D (slot diversity)
- [ ] Train and evaluate: do slots capture narrative state?

### Phase 2: Memory Reader
- [ ] Implement memory storage structure (key-value + metadata)
- [ ] Implement soft-attention reader
- [ ] Integrate retrieved context into transformer (cross-attention layer)
- [ ] Train with "resume after interruption" task
- [ ] Initially: store everything, bounded buffer

### Phase 3: Memory Writer
- [ ] Implement surprise signal computation
- [ ] Implement learned write gate
- [ ] Add memory budget loss
- [ ] Train writer to store useful events

### Phase 4: Refinement
- [ ] Tune hyperparameters (S slots, memory size, loss weights)
- [ ] Implement memory consolidation (optional)
- [ ] Evaluate on narrative understanding benchmarks
- [ ] Analyze what slots and memory actually learn

---

## 9. Evaluation Metrics

### Intrinsic
- Perplexity (should improve with state tracking)
- Future chunk prediction accuracy
- Memory retrieval precision/recall

### Narrative-Specific
- Character tracking accuracy (who's in scene?)
- Location consistency (does model know where we are?)
- Event ordering (can model answer "what happened before X?")
- Long-range coherence in generation

### Interpretability
- Slot activation analysis (do slots correlate with narrative elements?)
- Memory content inspection (what gets stored?)
- Attention pattern analysis (what does model retrieve when?)

---

## 10. Open Questions

1. **How many slots?** Too few = can't capture complex state. Too many = redundancy/collapse.

2. **Slot specialization**: Should different slots be encouraged to track different things? How?

3. **Memory capacity**: Fixed size with eviction? Unbounded growth? Hierarchical (recent detailed, old summarized)?

4. **Gradient flow**: Full backprop through all chunks? TBPTT? Detach slots/memory periodically?

5. **Multi-document batching**: How to batch documents of different lengths? Pad? Pack?

6. **Inference-time behavior**: How do slots/memory work during generation? When to write during autoregressive decoding?

---

## 11. References

- Memorizing Transformers (Wu et al., 2022) — k-NN memory retrieval
- Perceiver (Jaegle et al., 2021) — latent bottleneck architecture
- Compressive Transformers (Rae et al., 2019) — memory compression
- RETRO (Borgeaud et al., 2022) — retrieval-augmented LM
- Recurrent Memory Transformer (Bulatov et al., 2022) — memory tokens
- Mamba (Gu & Dao, 2023) — state-space alternative to attention

---

*Document created: 2024-12-30*
*Status: Design phase — not yet implemented*
