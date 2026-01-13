# Experience Loop (Main Processing)

The complete forward pass: perceive, predict, compare, experience, remember.

## The Loop

```python
class PredictiveExperienceMachine(nn.Module):
    """
    Complete PEM architecture.

    The system doesn't just process tokens—it EXPERIENCES them.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Perception layer (transformer)
        self.feature_extractor = FeatureExtractor(config)

        # Experiential core
        self.prediction_module = PredictionModule(config)
        self.surprise_module = SurpriseModule(config)
        self.surprise_memory = SurpriseMemory(config)
        self.sync_core = SyncCore(config)

        # Output (if needed for downstream tasks)
        self.output_head = nn.Linear(config.sync_pairs, config.vocab_size)

        # Initial sync state
        self.initial_sync = nn.Parameter(torch.randn(1, 1, config.sync_pairs) * 0.02)

    def forward(
        self,
        tokens: Tensor,
        return_experiences: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Dict]]]:
        """
        Process tokens through the experience loop.

        Args:
            tokens: (B, S) input token IDs
            return_experiences: If True, return per-position experience dicts

        Returns:
            output: (B, S, V) logits or task output
            experiences: List of experience dicts (if return_experiences)
        """
        B, S = tokens.shape
        device = tokens.device

        # 1. PERCEIVE: Extract features from all tokens at once
        features = self.feature_extractor(tokens)  # (B, S, D)

        # Initialize
        sync = self.initial_sync.expand(B, S, -1)  # (B, S, sync_pairs)
        predicted_f = None
        all_experiences = []

        # Process position by position (causal)
        for t in range(S):
            # Current context (causal: only see up to t)
            context = features[:, :t+1, :]  # (B, t+1, D)
            current_f = features[:, t:t+1, :]  # (B, 1, D)

            # 2. COMPARE: How does reality match expectation?
            if predicted_f is not None:
                surprise_mag, surprise_dir = self.surprise_module(
                    predicted_f, current_f, context
                )
                surprise_signal = surprise_dir * surprise_mag
            else:
                # First position: no prediction yet
                surprise_mag = torch.zeros(B, 1, device=device)
                surprise_signal = torch.zeros(B, 1, self.config.d_model, device=device)

            # 3. REMEMBER: Store if surprising, retrieve if relevant
            if t > 0 and surprise_mag.mean() > self.config.surprise_threshold:
                self.surprise_memory.write(
                    context[:, :-1, :],  # Context before current token
                    predicted_f,
                    current_f,
                    surprise_dir
                )

            memory_retrieval = self.surprise_memory.read(context)

            # 4. THINK: Sync processes features + surprise + memory
            # Note: sync_core processes the current position in context of all previous
            sync_input = features[:, :t+1, :]
            sync_out = self.sync_core(
                sync_input,
                surprise_signal=surprise_signal,
                memory_retrieval=memory_retrieval,
            )
            # Take sync for current position
            current_sync = sync_out[:, -1:, :]  # (B, 1, sync_pairs)

            # Update full sync state
            if t == 0:
                sync = current_sync
            else:
                sync = torch.cat([sync[:, :t, :], current_sync], dim=1)

            # 5. PREDICT: What do I expect next?
            if t < S - 1:
                predictions = self.prediction_module(current_sync, context)
                predicted_f = predictions['immediate']  # (B, 1, D)

            # 6. EXPERIENCE: Package the subjective state
            if return_experiences:
                experience = {
                    'position': t,
                    'sync': current_sync.detach(),
                    'surprise_magnitude': surprise_mag.detach(),
                    'surprise_direction': surprise_dir.detach() if t > 0 else None,
                    'prediction': predicted_f.detach() if predicted_f is not None else None,
                    'memory_retrieved': memory_retrieval.detach(),
                }
                all_experiences.append(experience)

        # Output from final sync state
        output = self.output_head(sync)  # (B, S, V)

        if return_experiences:
            return output, all_experiences
        return output
```

## Step-by-Step Walkthrough

### Position t=0: First Token

```python
# Perceive
features = extract("The")  # Rich representation of "The"

# No prediction yet (nothing to compare)
surprise = 0

# No memory retrieval (nothing stored yet)
memory = empty

# Sync processes just this token
sync = sync_core(features["The"], surprise=0, memory=empty)

# Predict next
predicted_f = predict(sync)  # Expects: noun, article continuation, etc.
```

### Position t=1: Second Token

```python
# Perceive
features = extract("The cat")  # Now have context

# Compare prediction to reality
actual_f = features["cat"]
surprise_mag, surprise_dir = compare(predicted_f, actual_f)
# If "cat" was expected: low surprise
# If "quantum" appeared: high surprise

# Memory operations
if surprise > threshold:
    memory.write(context="The", predicted=predicted_f, actual=actual_f)
memory_retrieval = memory.read(context="The cat")

# Sync integrates everything
sync = sync_core(
    features,
    surprise=surprise_dir * surprise_mag,  # Modulates dynamics
    memory=memory_retrieval  # Past relevant surprises
)

# Predict next
predicted_f = predict(sync)  # Given "The cat", expects: verb, adjective, etc.
```

### Position t=N: Surprise Occurs!

```python
# Context: "The cat sat on the"
# Prediction: expects "mat", "floor", "chair", etc.
# Actual: "quantum"

surprise_mag = 0.95  # Very high!
surprise_dir = encode_what_was_unexpected()

# Store this surprise
memory.write(
    context="The cat sat on the",
    predicted="mat-like features",
    actual="quantum features",
    surprise=surprise_dir
)

# Sync runs with high modulation
# → Large state changes
# → Pattern disruption
# → Re-synchronization around new understanding

sync = sync_core(features, surprise=HIGH, memory=...)

# Next prediction now accounts for this unusual text
predicted_f = predict(sync)  # Adjusted expectations
```

## Batched vs Sequential

The loop above is **sequential per position** (for clarity). In practice:

### Parallel Feature Extraction
```python
# All positions at once
features = self.feature_extractor(tokens)  # (B, S, D)
```

### Sequential Experience (Causal)
```python
# Must be sequential because:
# - Prediction depends on previous sync
# - Surprise depends on previous prediction
# - Memory depends on previous surprises
for t in range(S):
    ...
```

### Optimization: Chunked Processing
```python
# Process in chunks for efficiency
chunk_size = 64
for chunk_start in range(0, S, chunk_size):
    chunk_end = min(chunk_start + chunk_size, S)
    # Process chunk with accumulated state
    ...
```

## Experience Output

When `return_experiences=True`, get rich introspection:

```python
output, experiences = model(tokens, return_experiences=True)

for exp in experiences:
    print(f"Position {exp['position']}:")
    print(f"  Surprise: {exp['surprise_magnitude'].item():.3f}")
    print(f"  Memory retrieved: {exp['memory_retrieved'] is not None}")
    print(f"  Sync norm: {exp['sync'].norm().item():.3f}")
```

This enables:
- Visualization of "attention" via surprise
- Analysis of what the model "remembers"
- Understanding of sync pattern evolution

## Output Head Options

### Option A: Classification/Generation
```python
# Standard LM head
output = self.output_head(sync)  # (B, S, V) logits
```

### Option B: No Token Output
```python
# Pure experience machine - no generation
# Output is the experience itself
return sync, all_experiences
```

### Option C: Task-Specific
```python
# QA, sentiment, etc.
output = self.task_head(sync[:, -1, :])  # Use final sync
```

## Open Questions

1. **Causality**: How to handle bidirectional contexts efficiently?
2. **Parallelization**: Can we parallelize the experience loop?
3. **Gradient flow**: How do gradients flow through the sequential loop?
4. **Warm-up**: How many positions before sync is "meaningful"?
