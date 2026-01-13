# Prediction Module (Expectation Generation)

The system constantly asks: "Given my current understanding (sync), what do I expect to see next?"

## Role

- **Expectation generation**: Sync state → predicted features
- **Multi-scale prediction**: Immediate, short-term, long-term
- **The basis for surprise**: Predictions enable comparison to reality

## Implementation

```python
class PredictionModule(nn.Module):
    """
    Sync state → predicted features.

    The system constantly asks: "Given my current understanding
    (sync), what do I expect to see next?"
    """

    def __init__(self, config):
        super().__init__()
        self.sync_pairs = config.sync_pairs
        self.d_model = config.d_model

        # Sync → query for what to predict
        self.sync_to_query = nn.Linear(config.sync_pairs, config.d_model)

        # Cross-attend to context to form prediction
        self.prediction_attn = nn.MultiheadAttention(
            config.d_model, config.n_head, batch_first=True
        )

        # Output projections for multi-scale prediction
        self.immediate_head = nn.Linear(config.d_model, config.d_model)  # Next token
        self.shortterm_head = nn.Linear(config.d_model, config.d_model)  # Next phrase
        self.longterm_head = nn.Linear(config.d_model, config.d_model)   # Topic trajectory

    def forward(
        self,
        sync: Tensor,        # (B, S, sync_pairs) current cognitive state
        context: Tensor,     # (B, S, D) feature context
    ) -> Dict[str, Tensor]:
        """
        Generate predictions at multiple time scales.

        Returns:
            predictions: Dict with 'immediate', 'shortterm', 'longterm' features
        """
        B, S, _ = sync.shape

        # Sync determines WHAT to predict (query formation)
        query = self.sync_to_query(sync)  # (B, S, D)

        # Attend to context to form prediction basis
        pred_basis, _ = self.prediction_attn(
            query, context, context
        )  # (B, S, D)

        # Multi-scale predictions
        predictions = {
            'immediate': self.immediate_head(pred_basis[:, -1:, :]),  # Next position
            'shortterm': self.shortterm_head(pred_basis[:, -1:, :]),  # Next few
            'longterm': self.longterm_head(pred_basis.mean(dim=1, keepdim=True)),  # Global
        }

        return predictions
```

## Multi-Scale Prediction

Different scales of prediction enable different kinds of surprise:

### Immediate (Next Token Features)
- What word/subword comes next?
- Surprise here = unexpected word choice
- Example: "The cat sat on the quantum" → HIGH surprise at "quantum"

### Short-term (Phrase Structure)
- What grammatical/semantic structure follows?
- Surprise here = unexpected syntax or phrase type
- Example: Expecting noun phrase, getting a question

### Long-term (Topic/Trajectory)
- Where is this document going?
- Surprise here = topic shift, tone change
- Example: Technical paper suddenly becomes poetic

## Design Considerations

### Why Predict Features, Not Tokens?

1. **Richer signal**: Features capture semantics, tokens are arbitrary symbols
2. **Softer predictions**: Can express uncertainty in feature space
3. **Multi-scale natural**: Features compose across scales

### Sync Drives Prediction

The key insight: **sync encodes what the system is "thinking about"**

- If sync represents "cooking", predictions should be food-related
- If sync represents "quantum physics", predictions should be science-related
- Sync IS the internal model of what's happening

### Prediction vs Generation

This is NOT autoregressive generation. We're predicting:
- What features SHOULD appear (expectation)
- Not what features we WANT to generate

The system is a reader, not a writer (at this level).

## Interface

```python
# Generate predictions from current sync state
predictions = prediction_module(sync, context)

# Extract immediate prediction for comparison
predicted_f = predictions['immediate']  # (B, 1, D)

# Later: compare to actual
actual_f = features[:, t+1:t+2, :]  # (B, 1, D)
surprise = surprise_module(predicted_f, actual_f)
```

## Open Questions

1. How far ahead to predict? (1 token? 10? variable?)
2. Should predictions be probabilistic (distribution) or point estimates?
3. How to handle prediction at document boundaries?
4. Should sync directly output prediction, or modulate a separate predictor?
