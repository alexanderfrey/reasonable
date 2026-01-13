# Feature Extractor (Perception Layer)

The "eyes" of the system. Transforms raw tokens into rich contextual features.

## Role

- **Perception, not cognition**: Extracts what's there, doesn't interpret meaning
- **Could be pretrained**: Leverage existing transformer knowledge
- **Frozen or slow-learning**: Stable feature space for the experiential core

## Implementation

```python
class FeatureExtractor(nn.Module):
    """
    Standard transformer - the 'eyes' of the system.

    Produces rich contextual features, NOT predictions.
    This is perception, not cognition.
    """

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        self.norm = RMSNorm(config.d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: (B, S) token IDs

        Returns:
            features: (B, S, D) contextual features
        """
        x = self.embedding(tokens)

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)
```

## Design Considerations

### Why Transformers?

- Proven feature extractors for language
- Self-attention captures contextual relationships
- Can leverage pretrained weights (GPT, LLaMA, etc.)

### Frozen vs Trainable

**Option A: Frozen**
- Stable feature space
- Experiential core learns on fixed representations
- Risk: Feature extractor not optimized for this task

**Option B: Slow-learning**
- Lower learning rate than experiential core
- Gradual adaptation
- Risk: Feature drift destabilizes sync

**Option C: Jointly trained**
- End-to-end optimization
- Risk: May collapse to trivial solutions

### Feature Granularity

The feature extractor could output at multiple granularities:
- **Token-level**: Features per position
- **Phrase-level**: Pooled features over spans
- **Document-level**: Global context vector

The sync core needs to know what scale of features to predict.

## Interface with Experiential Core

```python
# Features flow UP to the experiential core
features = feature_extractor(tokens)  # (B, S, D)

# Current position features for surprise computation
current_f = features[:, -1, :]  # (B, D)

# Full context for prediction
context = features  # (B, S, D)
```

## Open Questions

1. Should features be compressed before going to sync core?
2. Multiple feature extractors for different scales?
3. How much of transformer to use? (early layers = syntax, late = semantics)
