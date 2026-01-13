# Surprise Module (Comparison)

Compare prediction to reality. Compute meaningful surprise.

## Role

- **Comparison**: Predicted features vs actual features
- **Meaningful surprise**: Not all errors are equally surprising
- **Context-dependent**: Same error can be surprising or not depending on context

## Core Insight

Surprise isn't just L2 distance. It's **meaningful** difference.

- "cat" vs "dog" after "The ___ barked" → HIGH surprise (dogs bark, cats don't)
- "cat" vs "dog" after "The pet was" → LOW surprise (both valid)
- "cat" vs "quantum" after anything → VERY HIGH surprise (semantic distance)

## Implementation

```python
class SurpriseModule(nn.Module):
    """
    Compare prediction to reality.

    Surprise isn't just L2 distance - it's MEANINGFUL difference.
    Some mismatches matter more than others.
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model

        # Learned metric: what constitutes "surprising"?
        self.error_encoder = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model),  # pred, actual, diff
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Context-dependent surprise gating
        self.context_gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),  # error_enc, context
            nn.GELU(),
            nn.Linear(config.d_model, 1),
            nn.Sigmoid(),
        )

        # Surprise direction encoder (WHAT was unexpected)
        self.direction_encoder = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

    def forward(
        self,
        predicted: Tensor,   # (B, 1, D) predicted features
        actual: Tensor,      # (B, 1, D) actual features
        context: Tensor,     # (B, S, D) full context
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute surprise magnitude and direction.

        Args:
            predicted: What we expected
            actual: What we got
            context: Full context for gating

        Returns:
            surprise_magnitude: (B, 1) scalar surprise per sample
            surprise_direction: (B, 1, D) what was unexpected (for memory)
        """
        # Raw prediction error
        raw_error = predicted - actual  # (B, 1, D)

        # Encode the error with context of what was predicted and what arrived
        error_input = torch.cat([predicted, actual, raw_error], dim=-1)
        error_encoded = self.error_encoder(error_input)  # (B, 1, D)

        # Context-dependent gating
        # Use mean context as summary (could be more sophisticated)
        context_summary = context.mean(dim=1, keepdim=True)  # (B, 1, D)
        gate_input = torch.cat([error_encoded, context_summary], dim=-1)
        surprise_magnitude = self.context_gate(gate_input)  # (B, 1, 1)
        surprise_magnitude = surprise_magnitude.squeeze(-1)  # (B, 1)

        # Direction: WHAT was unexpected (not just that something was)
        direction_input = torch.cat([error_encoded, raw_error], dim=-1)
        surprise_direction = self.direction_encoder(direction_input)  # (B, 1, D)

        return surprise_magnitude, surprise_direction
```

## Surprise Magnitude vs Direction

### Magnitude (Scalar)
- **How surprising** was this?
- Used to gate memory writes (only store if surprising enough)
- Used to modulate sync dynamics (think harder when surprised)

### Direction (Vector)
- **What** was unexpected?
- Stored in memory (for retrieval later)
- Injected into sync (to update internal model)

## Context-Dependent Surprise

The same prediction error means different things in different contexts:

```python
# Example: predicted "happy", got "sad"

# Context A: "She won the lottery and felt..."
# Prediction: "happy" → Actual: "sad"
# Surprise: HIGH (contradicts context)

# Context B: "The weather affected her mood..."
# Prediction: "happy" → Actual: "sad"
# Surprise: MEDIUM (both plausible)

# Context C: "After the funeral, she felt..."
# Prediction: "happy" → Actual: "sad"
# Surprise: LOW (sad is expected, prediction was wrong)
```

The context gate learns these distinctions.

## Training the Surprise Module

### Option A: Human Surprise Annotations
- Annotate passages where humans find text surprising
- Train surprise magnitude to match human ratings
- Expensive but gold standard

### Option B: Self-Supervised
- Surprise should correlate with prediction difficulty
- If many things could follow, low surprise for any of them
- If one thing strongly expected, high surprise for deviations

### Option C: Contrastive
- High surprise for semantically distant substitutions
- Low surprise for synonyms/paraphrases
- Use embedding distances as proxy

## Interface

```python
# Compute surprise when new token arrives
surprise_mag, surprise_dir = surprise_module(
    predicted_f,  # What we expected
    actual_f,     # What we got
    context       # Full context
)

# Use magnitude for gating
if surprise_mag > threshold:
    memory.write(context, predicted_f, actual_f, surprise_dir)

# Use direction for sync update
sync = sync_core(features, surprise_signal=surprise_dir * surprise_mag)
```

## Open Questions

1. Should surprise be computed at multiple scales (matching prediction scales)?
2. How to handle cumulative surprise (many small surprises = one big)?
3. Should there be "negative surprise" (confirming expectations)?
4. How to calibrate surprise across different domains/texts?
