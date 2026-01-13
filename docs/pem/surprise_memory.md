# Surprise Memory (Episodic Memory)

Remember not just THAT we were surprised, but WHY.

## Role

- **Episodic storage**: Store surprising events with full context
- **Retrieval**: When similar context appears, recall past surprises
- **Learning from mistakes**: "I was wrong here before because..."

## Core Insight

The value isn't just remembering surprises—it's **using them** to make better predictions.

When similar context appears:
1. Retrieve: "Last time I saw something like this..."
2. Recall: "...I predicted X but got Y"
3. Adjust: "...so this time I should consider Y-like outcomes"

## Implementation

```python
class SurpriseMemory(nn.Module):
    """
    Remember not just THAT we were surprised, but WHY.

    Stores: (context, prediction, actual, surprise_encoding)
    Retrieves: when similar context appears
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.memory_size = config.memory_size  # Max memories to store
        self.n_heads = config.memory_heads     # For multi-head retrieval

        # Memory banks (as parameters for gradient flow, or buffers for non-diff)
        self.register_buffer('memory_keys', torch.zeros(self.memory_size, config.d_model))
        self.register_buffer('memory_values', torch.zeros(self.memory_size, config.d_model * 3))
        self.register_buffer('memory_ages', torch.zeros(self.memory_size))
        self.register_buffer('write_pointer', torch.tensor(0))

        # Key encoder: context → retrieval key
        self.key_encoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Value encoder: (prediction, actual, surprise) → stored value
        self.value_encoder = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model * 3),
            nn.GELU(),
            nn.Linear(config.d_model * 3, config.d_model * 3),
        )

        # Retrieval attention
        self.retrieval_attn = nn.MultiheadAttention(
            config.d_model, self.n_heads, batch_first=True
        )

        # Output projection: retrieved memories → usable signal
        self.output_proj = nn.Linear(config.d_model * 3, config.d_model)

    def write(
        self,
        context: Tensor,          # (B, S, D) context when surprise occurred
        prediction: Tensor,       # (B, 1, D) what was predicted
        actual: Tensor,           # (B, 1, D) what actually happened
        surprise_dir: Tensor,     # (B, 1, D) encoded surprise direction
    ):
        """
        Store a surprising event in memory.

        Writes one memory per batch element.
        """
        B = context.shape[0]

        # Context summary as key (what situation was this?)
        context_summary = context.mean(dim=1)  # (B, D)
        keys = self.key_encoder(context_summary)  # (B, D)

        # Pack value: what predicted, what happened, why surprising
        values = torch.cat([
            prediction.squeeze(1),   # (B, D)
            actual.squeeze(1),       # (B, D)
            surprise_dir.squeeze(1), # (B, D)
        ], dim=-1)  # (B, D*3)
        values = self.value_encoder(values)

        # Write to memory (circular buffer)
        for i in range(B):
            idx = (self.write_pointer + i) % self.memory_size
            self.memory_keys[idx] = keys[i].detach()
            self.memory_values[idx] = values[i].detach()
            self.memory_ages[idx] = 0  # Fresh memory

        self.write_pointer = (self.write_pointer + B) % self.memory_size
        self.memory_ages += 1  # Age all memories

    def read(
        self,
        query_context: Tensor,  # (B, S, D) current context
        top_k: int = 5,
    ) -> Tensor:
        """
        Retrieve relevant past surprises.

        Args:
            query_context: Current context to match against
            top_k: Number of memories to retrieve

        Returns:
            retrieved: (B, D) aggregated memory signal
        """
        B, S, D = query_context.shape

        # Query from context summary
        context_summary = query_context.mean(dim=1)  # (B, D)
        query = self.key_encoder(context_summary)    # (B, D)

        # Compute similarity to all memory keys
        # memory_keys: (M, D), query: (B, D)
        similarity = torch.matmul(query, self.memory_keys.T)  # (B, M)

        # Mask out empty memories (age 0 means never written)
        mask = self.memory_ages == 0
        similarity = similarity.masked_fill(mask.unsqueeze(0), float('-inf'))

        # Soft top-k via softmax
        weights = F.softmax(similarity / 0.1, dim=-1)  # (B, M)

        # Retrieve weighted values
        retrieved_values = torch.matmul(weights, self.memory_values)  # (B, D*3)

        # Project to usable signal
        retrieved = self.output_proj(retrieved_values)  # (B, D)

        return retrieved.unsqueeze(1)  # (B, 1, D)
```

## Memory Structure

Each memory entry contains:

| Field | Shape | Description |
|-------|-------|-------------|
| Key | (D,) | Context encoding (for retrieval) |
| Prediction | (D,) | What was predicted |
| Actual | (D,) | What actually happened |
| Surprise | (D,) | Encoded surprise direction |
| Age | (1,) | How old this memory is |

## Retrieval Mechanism

### Similarity-Based
```python
# Current context → query
# Find memories with similar context
# Return what happened in those situations
```

### Multi-Head Retrieval
Different heads can retrieve for different purposes:
- Head 1: Similar topic/domain
- Head 2: Similar syntactic structure
- Head 3: Similar sentiment/tone

### Temporal Weighting
- Recent memories might be more relevant
- Or: old memories that keep being relevant are important
- Configurable decay/boosting

## Differentiable vs Non-Differentiable

### Option A: Fully Differentiable (NTM-style)
- Soft attention over all memories
- Gradients flow through read/write
- Pro: End-to-end trainable
- Con: Expensive, all memories always involved

### Option B: Hard Retrieval + Soft Use
- Hard top-k retrieval (non-differentiable)
- Soft aggregation of retrieved memories
- Pro: Sparse, efficient
- Con: Retrieval not optimized by gradient

### Option C: Hybrid
- Write is non-differentiable (detached)
- Read is differentiable
- Pro: Stable memories, trainable usage
- Con: Can't learn what to remember

## Using Retrieved Memories

```python
# In the experience loop:
memory_retrieval = memory.read(context)  # (B, 1, D)

# Inject into sync core
sync = sync_core(
    features,
    surprise_signal=surprise_dir * surprise_mag,
    memory_retrieval=memory_retrieval,  # "I've seen something like this before..."
)

# Memory can also modulate prediction
predictions = prediction_module(sync, context, memory_hint=memory_retrieval)
```

## Open Questions

1. **Capacity**: Fixed size with overwrite? Growing? Compression?
2. **What to store**: Every surprise? Only above threshold? Sampled?
3. **Forgetting**: Should old memories decay? Be overwritten? Consolidated?
4. **Cross-document**: Should memories persist across documents/sessions?
5. **Memory of non-surprises**: Store confirmed predictions too?
