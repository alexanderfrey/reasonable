# Experiment v0.1: Experiential Stream — Minimal Validation

**Goal**: Validate the core premise with minimal code changes.

**Core question**: Can we train a model to predict "what comes next" in latent space, and does this capture anything meaningful?

---

## 1. Minimal Design

### The Simplest Test

Within a single sequence, split into two halves:
```
Sequence: [token_0, token_1, ..., token_n]
           |<--- first half --->|<--- second half --->|

First half  → encode → state vector
Second half → encode → target embedding (stop gradient)
State vector → predict → predicted embedding

Loss: predicted should match target, not other sequences' targets
```

**No data pipeline changes needed.** Works within existing training loop.

### Why This Works

If the model can predict the latent representation of the second half from the first half, it means:
1. The state encoder captures something about "where the narrative is going"
2. The predictor learns temporal dependencies in latent space
3. Surprise (prediction error) will naturally emerge

---

## 2. Architecture (Minimal)

```python
class ExperientialStreamV01(nn.Module):
    """
    Minimal experiential stream for validation.

    - Encodes first half of sequence into state
    - Predicts embedding of second half
    - Computes surprise as prediction error
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()

        # State encoder: pool first half into single vector
        self.state_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.state_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Predictor: state → predicted future embedding
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

        # Target encoder: pool second half (shared architecture, could be separate)
        self.target_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.target_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, hidden_states: Tensor) -> dict:
        """
        Args:
            hidden_states: [batch, seq_len, d_model] from transformer

        Returns:
            dict with state, prediction, target, surprise
        """
        batch_size, seq_len, d_model = hidden_states.shape
        mid = seq_len // 2

        first_half = hidden_states[:, :mid, :]   # [B, mid, d]
        second_half = hidden_states[:, mid:, :]  # [B, seq_len-mid, d]

        # Encode first half into state
        query = self.state_query.expand(batch_size, -1, -1)
        state, _ = self.state_attn(query, first_half, first_half)
        state = state.squeeze(1)  # [B, d]

        # Predict future embedding
        prediction = self.predictor(state)  # [B, d]

        # Encode second half into target (stop gradient for contrastive)
        target_query = self.target_query.expand(batch_size, -1, -1)
        target, _ = self.target_attn(target_query, second_half, second_half)
        target = target.squeeze(1)  # [B, d]

        # Compute surprise (prediction error)
        with torch.no_grad():
            surprise = 1 - F.cosine_similarity(prediction, target, dim=-1)  # [B]

        return {
            'state': state,
            'prediction': prediction,
            'target': target.detach(),  # stop gradient
            'surprise': surprise
        }
```

---

## 3. Loss Function

### InfoNCE Contrastive Loss

```python
def experiential_loss(prediction: Tensor, target: Tensor, temperature: float = 0.1) -> Tensor:
    """
    Contrastive loss: prediction should match its target, not others in batch.

    Args:
        prediction: [batch, d_model] - predicted future embedding
        target: [batch, d_model] - actual future embedding (detached)
        temperature: softmax temperature

    Returns:
        scalar loss
    """
    # Normalize
    pred_norm = F.normalize(prediction, dim=-1)
    target_norm = F.normalize(target, dim=-1)

    # Similarity matrix [batch, batch]
    sim_matrix = torch.mm(pred_norm, target_norm.t()) / temperature

    # Labels: diagonal (each prediction matches its own target)
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

    # Cross entropy loss
    loss = F.cross_entropy(sim_matrix, labels)

    return loss
```

### Why Contrastive?

- MSE would just match magnitude, not semantics
- Contrastive forces the model to distinguish THIS sequence's future from OTHER sequences' futures
- This requires capturing sequence-specific information in the state

---

## 4. Integration with Training

### Option A: Auxiliary Loss (Recommended for v0.1)

Add to existing training loop with minimal changes:

```python
# In train_step, after getting hidden states:
hidden_states = model.forward_hidden(input_ids)  # need to expose this

# Experiential processing
exp_output = experiential_module(hidden_states)
exp_loss = experiential_loss(exp_output['prediction'], exp_output['target'])

# Combined loss
total_loss = lm_loss + exp_loss_weight * exp_loss
```

### Option B: Separate Training Script

Create `train_experiential.py` that:
1. Loads pretrained model (frozen)
2. Only trains the experiential module
3. Validates on held-out sequences

**Recommendation**: Start with Option B for faster iteration.

---

## 5. Success Metrics

### Primary: Prediction Accuracy

```python
def prediction_accuracy(prediction: Tensor, target: Tensor, all_targets: Tensor) -> float:
    """
    What fraction of predictions match the correct target?

    Accuracy = 1/N means random (batch size N)
    Accuracy > 1/N means learning something
    """
    pred_norm = F.normalize(prediction, dim=-1)
    target_norm = F.normalize(all_targets, dim=-1)

    # Each prediction's most similar target
    sims = torch.mm(pred_norm, target_norm.t())
    predicted_idx = sims.argmax(dim=-1)
    correct_idx = torch.arange(len(prediction), device=prediction.device)

    accuracy = (predicted_idx == correct_idx).float().mean()
    return accuracy.item()
```

**Success threshold**:
- Random baseline: 1/batch_size (e.g., 1/32 = 3.1%)
- Target: >50% accuracy after training

### Secondary: Surprise Correlation

Does high surprise correlate with "interesting" events?

```python
def analyze_surprise(model, experiential, texts: List[str], tokenizer):
    """
    Look at where surprise is highest in sample texts.
    """
    for text in texts:
        tokens = tokenizer.encode(text)
        # Process in sliding windows
        # Record surprise at each position
        # Visualize: are peaks at scene changes, reveals, etc.?
```

### Tertiary: State Probing

What does the state encode?

```python
def probe_state(states: Tensor, labels: Tensor):
    """
    Train linear probe: can state predict narrative features?
    - Genre
    - Sentiment
    - Character presence
    - etc.
    """
    probe = nn.Linear(d_model, n_classes)
    # Train on (state, label) pairs
    # Report accuracy
```

---

## 6. Experiment Plan

### Phase 1: Sanity Check (1-2 hours)

1. Implement `ExperientialStreamV01`
2. Create standalone test script
3. Verify shapes, gradients flow
4. Train for 100 steps on small data
5. Check: does loss decrease?

### Phase 2: Validation (4-8 hours)

1. Train experiential module on frozen GPT
2. Use existing pretokenized data
3. Train for 1000-5000 steps
4. Measure prediction accuracy
5. Target: accuracy > 50% (vs ~3% random)

### Phase 3: Interpretation (if Phase 2 succeeds)

1. Analyze surprise on sample texts
2. Probe states for interpretable features
3. Visualize what's learned

---

## 7. Files to Create

```
reasonable/
├── experiential.py          # The module
├── train_experiential.py    # Standalone training script
└── test_experiential.py     # Quick sanity checks
```

---

## 8. Minimal Code Needed

### experiential.py (~80 lines)
- ExperientialStreamV01 class
- experiential_loss function
- prediction_accuracy metric

### train_experiential.py (~150 lines)
- Load pretrained model
- Freeze backbone
- Train experiential module only
- Log metrics

### test_experiential.py (~50 lines)
- Test shapes
- Test gradient flow
- Test on dummy data

---

## 9. What We'll Learn

**If it works (accuracy > 50%)**:
- The state encoder captures predictive information
- Latent prediction is learnable
- Foundation for full experiential stream is valid

**If it fails (accuracy ≈ random)**:
- Need richer state representation
- Need different prediction target
- Need cross-chunk context (can't predict within-sequence)

Either outcome is valuable information.

---

## 10. Next Steps After v0.1

If successful:
1. Add persistent state across chunks (requires data pipeline)
2. Add affect prediction (valence/arousal)
3. Add salience gating (for episodic crystallization)
4. Integrate with main training loop

If unsuccessful:
1. Analyze failure mode
2. Try different state encoder (learned queries, different pooling)
3. Try different prediction target (token-level, shorter horizon)
4. Consider if within-sequence prediction is too easy/hard
