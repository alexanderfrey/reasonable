# Training Objectives

How to train a system that experiences rather than just predicts.

## The Challenge

Standard NTP (next-token prediction) won't work because:
1. We're predicting **features**, not tokens
2. We want calibrated **surprise**, not just accuracy
3. We need the **memory** to be useful
4. The **sync** patterns need to be meaningful

## Multi-Objective Training

```python
class PEMLoss(nn.Module):
    def __init__(self, config):
        self.lambda_pred = config.lambda_pred        # Feature prediction
        self.lambda_surprise = config.lambda_surprise  # Surprise calibration
        self.lambda_memory = config.lambda_memory    # Memory usefulness
        self.lambda_sync = config.lambda_sync        # Sync consistency
        self.lambda_task = config.lambda_task        # Downstream task

    def forward(self, model_output, targets, experiences):
        loss = 0

        # 1. Feature Prediction Loss
        loss += self.lambda_pred * self.prediction_loss(experiences)

        # 2. Surprise Calibration Loss
        loss += self.lambda_surprise * self.surprise_loss(experiences)

        # 3. Memory Usefulness Loss
        loss += self.lambda_memory * self.memory_loss(experiences)

        # 4. Sync Consistency Loss
        loss += self.lambda_sync * self.sync_loss(experiences)

        # 5. Downstream Task Loss
        loss += self.lambda_task * self.task_loss(model_output, targets)

        return loss
```

## 1. Feature Prediction Loss

The model should predict features accurately—but not collapse.

### Contrastive Prediction
```python
def prediction_loss(self, experiences):
    """
    Contrastive loss: predicted features should be closer to actual
    than to random negatives.
    """
    total_loss = 0

    for t, exp in enumerate(experiences[:-1]):
        predicted = exp['prediction']      # (B, 1, D)
        actual = experiences[t+1]['features']  # (B, 1, D)

        # Sample negatives (other positions, other batches)
        negatives = sample_negatives(actual, num_neg=16)  # (B, 16, D)

        # Positive similarity
        pos_sim = F.cosine_similarity(predicted, actual, dim=-1)  # (B, 1)

        # Negative similarities
        neg_sim = F.cosine_similarity(
            predicted.unsqueeze(2),  # (B, 1, 1, D)
            negatives.unsqueeze(1),  # (B, 1, 16, D)
            dim=-1
        )  # (B, 1, 16)

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (B, 1, 17)
        labels = torch.zeros(B, 1, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits.squeeze(1), labels.squeeze(1))

        total_loss += loss

    return total_loss / len(experiences)
```

### Why Contrastive?
- Prevents collapse (predicting same thing always)
- Doesn't require exact matching (soft targets)
- Naturally handles multi-modal predictions

## 2. Surprise Calibration Loss

The model should be surprised when it **should** be surprised.

### Option A: Human Annotations
```python
def surprise_loss_supervised(self, experiences, human_surprise_labels):
    """
    Train surprise to match human judgments.
    Requires annotated data: which passages are surprising?
    """
    predicted_surprise = torch.stack([
        exp['surprise_magnitude'] for exp in experiences
    ], dim=1)  # (B, S)

    return F.binary_cross_entropy(predicted_surprise, human_surprise_labels)
```

### Option B: Self-Supervised via Prediction Error
```python
def surprise_loss_self_supervised(self, experiences):
    """
    Surprise should correlate with actual prediction error.
    High prediction error → should have high surprise.
    """
    for t, exp in enumerate(experiences[:-1]):
        predicted = exp['prediction']
        actual = experiences[t+1]['features']

        # Actual error (ground truth for surprise)
        actual_error = (predicted - actual).norm(dim=-1)  # (B, 1)

        # Predicted surprise
        predicted_surprise = exp['surprise_magnitude']  # (B, 1)

        # They should correlate
        # Use smooth L1 or MSE
        loss += F.smooth_l1_loss(predicted_surprise, actual_error.detach())

    return loss
```

### Option C: Contrastive Surprise
```python
def surprise_loss_contrastive(self, experiences):
    """
    Semantic distance should predict surprise.
    Synonyms → low surprise
    Unrelated words → high surprise
    """
    # Use embedding distance as proxy for "should be surprising"
    # Train surprise module to match
    ...
```

## 3. Memory Usefulness Loss

Retrieved memories should **help** predictions.

```python
def memory_loss(self, experiences):
    """
    Predictions should be better WITH memory than WITHOUT.
    """
    # Run prediction twice: with and without memory
    loss = 0

    for exp in experiences:
        # Prediction error with memory
        pred_with_memory = exp['prediction_with_memory']
        error_with = (pred_with_memory - exp['actual']).norm()

        # Prediction error without memory
        pred_without_memory = exp['prediction_without_memory']
        error_without = (pred_without_memory - exp['actual']).norm()

        # Memory should help (error_with < error_without)
        # Hinge loss: penalize if memory makes prediction worse
        margin = 0.1
        loss += F.relu(error_with - error_without + margin)

    return loss
```

### Memory Retrieval Quality
```python
def memory_retrieval_loss(self, experiences):
    """
    Retrieved memories should be relevant to current context.
    """
    for exp in experiences:
        query_context = exp['context']
        retrieved_memory = exp['memory_retrieved']

        # Relevance score (could be learned or heuristic)
        relevance = compute_relevance(query_context, retrieved_memory)

        # Should retrieve relevant memories
        loss += -relevance.mean()  # Maximize relevance

    return loss
```

## 4. Sync Consistency Loss

Sync patterns should be **meaningful** and **stable** (when not surprised).

### Consistency When Not Surprised
```python
def sync_consistency_loss(self, experiences):
    """
    Similar inputs should produce similar sync (when surprise is low).
    Sync should change when surprise is high.
    """
    loss = 0

    for t in range(1, len(experiences)):
        prev_sync = experiences[t-1]['sync']
        curr_sync = experiences[t]['sync']
        surprise = experiences[t]['surprise_magnitude']

        # Sync similarity
        sync_sim = F.cosine_similarity(prev_sync, curr_sync, dim=-1)

        # When surprise is LOW: sync should be STABLE (high similarity)
        # When surprise is HIGH: sync should CHANGE (low similarity)
        target_sim = 1.0 - surprise  # Low surprise → want high sim

        loss += F.mse_loss(sync_sim, target_sim)

    return loss
```

### Sync Diversity
```python
def sync_diversity_loss(self, experiences):
    """
    Different inputs should produce different sync patterns.
    Prevent collapse to trivial solution.
    """
    # Collect syncs across batch
    syncs = torch.stack([exp['sync'] for exp in experiences], dim=0)  # (T, B, sync_pairs)

    # Cross-batch similarity (should be low for different inputs)
    # Within-batch, different positions should have different syncs
    ...
```

## 5. Downstream Task Loss

Still need to do something useful.

```python
def task_loss(self, output, targets):
    """
    Standard cross-entropy for language modeling or other tasks.
    """
    return F.cross_entropy(
        output.view(-1, output.size(-1)),
        targets.view(-1)
    )
```

## Training Schedule

### Phase 1: Feature Extractor Warm-up
```python
# Freeze experiential core, train feature extractor
# Or use pretrained weights
for param in model.sync_core.parameters():
    param.requires_grad = False
```

### Phase 2: Joint Training
```python
# Unfreeze all, balance losses
optimizer = AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    loss = (
        1.0 * prediction_loss +
        0.5 * surprise_loss +
        0.3 * memory_loss +
        0.2 * sync_loss +
        1.0 * task_loss
    )
    loss.backward()
    optimizer.step()
```

### Phase 3: Fine-tuning
```python
# Task-specific fine-tuning
# Maybe freeze feature extractor, train experiential core
```

## Curriculum Learning

Start simple, increase complexity:

1. **Short sequences**: Easier to learn prediction
2. **Low surprise data**: Establish baseline patterns
3. **Introduce surprises**: Gradually add surprising content
4. **Full complexity**: All data types

```python
def get_curriculum_data(epoch):
    if epoch < 10:
        return filter_low_surprise(dataset)
    elif epoch < 20:
        return filter_medium_surprise(dataset)
    else:
        return dataset  # Full complexity
```

## Open Questions

1. **Loss balancing**: How to weight the multiple objectives?
2. **Surprise labels**: Can we get enough annotated surprise data?
3. **Memory training**: Differentiable vs RL for memory operations?
4. **Negative sampling**: How to sample good negatives for contrastive loss?
5. **Evaluation**: How to measure "quality of experience"?
