# Experiential Stream

The "now" — what the system is currently experiencing.

**Role**: The experiential stream is the present moment of processing. It's where raw input becomes felt experience, where predictions meet reality, and where surprise signals arise.

---

## 1. Core Concept

The experiential stream is **not** just a context window. It's the system's subjective present — the flow of experience before it crystallizes into memory or fades into nothing.

Properties:
- **Ephemeral**: Flows past, not stored permanently
- **Predictive**: Always anticipating what comes next
- **Surprise-sensitive**: Notices the unexpected
- **Affect-laden**: Some moments feel more significant

```
Input tokens ──► Experiential Stream ──► Predictions + Surprise + Affect
                        │
                        ▼
                 "The felt present"
```

---

## 2. Formal Definition

### 2.1 State

```python
@dataclass
class ExperientialState:
    """The complete state of present-moment experience."""

    # Core representation
    content: Tensor           # [d_model] - what is being experienced

    # Temporal context
    timestamp: int            # global position in stream
    duration: int             # how many tokens this state spans

    # Predictive processing
    predictions: Tensor       # [d_model] - what we expected
    surprise: float           # scalar - prediction error magnitude
    uncertainty: float        # scalar - confidence in predictions

    # Affective coloring
    valence: float            # [-1, 1] - positive/negative
    arousal: float            # [0, 1] - intensity
    salience: float           # [0, 1] - importance (surprise × arousal)

    # Meta-state
    attention_focus: Tensor   # [seq_len] - where attention is directed
    processing_depth: float   # [0, 1] - shallow vs deep processing
```

### 2.2 Transitions

```python
ExperientialState_t ──process(input_t)──► ExperientialState_{t+1}
```

Each moment flows into the next. The key operations:

1. **Integrate**: Combine new input with current state
2. **Predict**: Generate expectations for next moment
3. **Compare**: Compute surprise between prediction and reality
4. **Color**: Assign affective valence and arousal
5. **Focus**: Direct attention within the experience

---

## 3. Architecture

### 3.1 Components

```python
class ExperientialStream(nn.Module):
    """
    The present-moment experience processor.

    Wraps around transformer processing to add:
    - Predictive state maintenance
    - Surprise computation
    - Affective coloring
    - Attention focusing
    """

    def __init__(self, config: ExperientialConfig):
        super().__init__()
        self.d_model = config.d_model
        self.window_size = config.window_size

        # State representation
        self.state_encoder = StateEncoder(config.d_model)
        self.state_integrator = GRUCell(config.d_model, config.d_model)

        # Prediction machinery
        self.predictor = PredictionHead(config.d_model)
        self.uncertainty_estimator = UncertaintyHead(config.d_model)

        # Affect computation
        self.valence_head = nn.Linear(config.d_model, 1)
        self.arousal_head = nn.Linear(config.d_model, 1)

        # Attention modulation
        self.salience_gate = SalienceGate(config.d_model)

        # Current state (persistent across calls)
        self.register_buffer('current_state', torch.zeros(config.d_model))
        self.register_buffer('current_prediction', torch.zeros(config.d_model))

    def forward(
        self,
        hidden_states: Tensor,      # [batch, seq_len, d_model]
        prev_state: Optional[ExperientialState] = None
    ) -> ExperientialState:
        """Process a moment of experience."""

        batch_size = hidden_states.size(0)

        # 1. Encode current input into state representation
        input_encoding = self.state_encoder(hidden_states)  # [batch, d_model]

        # 2. Integrate with previous state
        if prev_state is not None:
            prev_content = prev_state.content
            prev_prediction = prev_state.predictions
        else:
            prev_content = self.current_state.expand(batch_size, -1)
            prev_prediction = self.current_prediction.expand(batch_size, -1)

        new_content = self.state_integrator(input_encoding, prev_content)

        # 3. Compute surprise (prediction error)
        surprise = self._compute_surprise(prev_prediction, new_content)

        # 4. Generate new predictions
        new_prediction = self.predictor(new_content)
        uncertainty = self.uncertainty_estimator(new_content)

        # 5. Compute affect
        valence = torch.tanh(self.valence_head(new_content)).squeeze(-1)
        arousal = torch.sigmoid(self.arousal_head(new_content)).squeeze(-1)

        # 6. Compute salience (gating signal for episodic crystallization)
        salience = self.salience_gate(surprise, arousal)

        # 7. Compute attention focus
        attention_focus = self._compute_attention_focus(hidden_states, new_content)

        return ExperientialState(
            content=new_content,
            timestamp=self._get_timestamp(),
            duration=hidden_states.size(1),
            predictions=new_prediction,
            surprise=surprise,
            uncertainty=uncertainty,
            valence=valence,
            arousal=arousal,
            salience=salience,
            attention_focus=attention_focus,
            processing_depth=self._estimate_depth(hidden_states)
        )

    def _compute_surprise(
        self,
        predicted: Tensor,
        actual: Tensor
    ) -> Tensor:
        """
        Surprise as prediction error.

        Multiple formulations possible:
        - Cosine distance (direction)
        - L2 distance (magnitude)
        - KL divergence (if probabilistic)
        """
        # Cosine surprise: 1 - similarity
        cosine_sim = F.cosine_similarity(predicted, actual, dim=-1)
        surprise = 1 - cosine_sim

        # Could also incorporate magnitude
        # magnitude_diff = (actual.norm(dim=-1) - predicted.norm(dim=-1)).abs()

        return surprise

    def _compute_attention_focus(
        self,
        hidden_states: Tensor,
        current_content: Tensor
    ) -> Tensor:
        """Where is attention directed within the current window?"""
        # Compute attention scores from current state to all positions
        scores = torch.einsum('bd,bsd->bs', current_content, hidden_states)
        focus = F.softmax(scores / math.sqrt(self.d_model), dim=-1)
        return focus

    def _estimate_depth(self, hidden_states: Tensor) -> Tensor:
        """
        Estimate processing depth.

        Deep processing: focused, effortful, detail-oriented
        Shallow processing: diffuse, automatic, gist-oriented
        """
        # Proxy: attention entropy (low entropy = focused = deep)
        # This is a placeholder — could be learned
        return torch.ones(hidden_states.size(0))
```

### 3.2 State Encoder

```python
class StateEncoder(nn.Module):
    """Compress hidden states into a single state vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attention_pool = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Pool sequence into single vector via attention.

        Args:
            hidden_states: [batch, seq_len, d_model]
        Returns:
            state: [batch, d_model]
        """
        batch_size = hidden_states.size(0)
        query = self.query.expand(batch_size, -1, -1)

        pooled, _ = self.attention_pool(query, hidden_states, hidden_states)
        return pooled.squeeze(1)
```

### 3.3 Prediction Head

```python
class PredictionHead(nn.Module):
    """Predict the next experiential state."""

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 2

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, current_state: Tensor) -> Tensor:
        """Predict next state from current state."""
        return self.net(current_state)
```

### 3.4 Salience Gate

```python
class SalienceGate(nn.Module):
    """
    Compute salience — how much this moment matters.

    High salience → should crystallize into episodic memory
    Low salience → let it flow past
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model + 2, d_model // 2),  # +2 for surprise, arousal
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        surprise: Tensor,      # [batch]
        arousal: Tensor,       # [batch]
        content: Optional[Tensor] = None  # [batch, d_model]
    ) -> Tensor:
        """
        Compute salience from surprise and arousal.

        Basic formula: salience = surprise × arousal
        Learned version can be more nuanced.
        """
        if content is not None:
            # Learned gating
            features = torch.cat([
                content,
                surprise.unsqueeze(-1),
                arousal.unsqueeze(-1)
            ], dim=-1)
            return self.gate(features).squeeze(-1)
        else:
            # Simple multiplicative gate
            return surprise * arousal
```

---

## 4. Training

### 4.1 Prediction Loss

The core training signal: predict the next experiential state.

```python
def prediction_loss(
    predicted_state: Tensor,
    actual_state: Tensor,
    mask: Optional[Tensor] = None
) -> Tensor:
    """
    Train the predictor to anticipate next state.

    Uses contrastive learning: predicted should match actual,
    not other states in the batch.
    """
    # Cosine similarity
    pred_norm = F.normalize(predicted_state, dim=-1)
    actual_norm = F.normalize(actual_state.detach(), dim=-1)  # stop gradient

    # Positive: predicted matches actual
    pos_sim = (pred_norm * actual_norm).sum(dim=-1)

    # Negatives: other items in batch
    neg_sim = torch.mm(pred_norm, actual_norm.t())  # [batch, batch]

    # InfoNCE loss
    temperature = 0.1
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, labels)
```

### 4.2 Affect Calibration

If we have sentiment/emotion labels (from LLM extraction or human annotation):

```python
def affect_loss(
    predicted_valence: Tensor,
    predicted_arousal: Tensor,
    target_valence: Tensor,
    target_arousal: Tensor
) -> Tensor:
    """Supervise affect prediction."""
    valence_loss = F.mse_loss(predicted_valence, target_valence)
    arousal_loss = F.mse_loss(predicted_arousal, target_arousal)
    return valence_loss + arousal_loss
```

### 4.3 Surprise Calibration

Surprise should be high when something unexpected happens:

```python
def surprise_calibration_loss(
    predicted_surprise: Tensor,
    actual_next_state: Tensor,
    predicted_next_state: Tensor
) -> Tensor:
    """
    The model's surprise estimate should match actual prediction error.

    This trains the uncertainty estimator.
    """
    actual_error = 1 - F.cosine_similarity(
        predicted_next_state.detach(),
        actual_next_state.detach(),
        dim=-1
    )
    return F.mse_loss(predicted_surprise, actual_error)
```

---

## 5. Integration with Transformer

### 5.1 Where in the forward pass?

```python
class GPTWithExperience(GPT):
    """GPT with experiential stream wrapper."""

    def __init__(self, config, experiential_config):
        super().__init__(config)
        self.experiential = ExperientialStream(experiential_config)

    def forward(
        self,
        input_ids: Tensor,
        prev_experience: Optional[ExperientialState] = None,
        **kwargs
    ) -> Tuple[Tensor, ExperientialState]:

        # Standard transformer forward (get hidden states)
        hidden_states = self.forward_hidden(input_ids, **kwargs)

        # Experiential processing
        experience = self.experiential(hidden_states, prev_experience)

        # Final output
        logits = self.lm_head(self.final_norm(hidden_states))

        return logits, experience

    def forward_hidden(self, input_ids: Tensor, **kwargs) -> Tensor:
        """Get hidden states without final projection."""
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x  # Don't apply final_norm or lm_head yet
```

### 5.2 Conditioning on Experience

The experiential state can condition transformer processing:

```python
def forward_with_experience_conditioning(
    self,
    input_ids: Tensor,
    experience: ExperientialState
) -> Tensor:
    """Use current experience to modulate processing."""

    x = self.token_embedding(input_ids)

    # Experience as additional context
    experience_tokens = self.experience_to_tokens(experience.content)
    x = torch.cat([experience_tokens, x], dim=1)

    # Or: modulate attention based on experience
    for layer in self.layers:
        x = layer(x, experience_bias=experience.attention_focus)

    return x
```

---

## 6. Interface with Other Streams

### 6.1 → Episodic Stream

The experiential stream provides the raw material for episodic crystallization:

```python
def should_crystallize(experience: ExperientialState) -> bool:
    """Should this moment become an episodic memory?"""
    return experience.salience > crystallization_threshold

def prepare_for_crystallization(experience: ExperientialState) -> EpisodeCandidate:
    """Package experience for episodic storage."""
    return EpisodeCandidate(
        content=experience.content,
        context=experience.predictions,  # what we expected (contrast)
        timestamp=experience.timestamp,
        salience=experience.salience,
        affect=(experience.valence, experience.arousal)
    )
```

### 6.2 ← Episodic Stream

Retrieved memories color current experience:

```python
def integrate_memory(
    current_experience: ExperientialState,
    retrieved_episodes: List[Episode]
) -> ExperientialState:
    """Let memories influence present experience."""

    # Memories provide context
    memory_context = aggregate_episodes(retrieved_episodes)

    # Blend with current content
    blended_content = (
        current_experience.content +
        memory_weight * memory_context
    )

    return current_experience.replace(content=blended_content)
```

### 6.3 ← Semantic Stream

Knowledge shapes expectations:

```python
def apply_knowledge(
    experience: ExperientialState,
    relevant_concepts: List[Concept]
) -> ExperientialState:
    """Knowledge shapes predictions and interpretation."""

    # Knowledge informs predictions
    knowledge_prior = aggregate_concepts(relevant_concepts)
    adjusted_predictions = (
        experience.predictions +
        knowledge_weight * knowledge_prior
    )

    return experience.replace(predictions=adjusted_predictions)
```

### 6.4 ← Procedural Stream

Skills modulate processing:

```python
def apply_skills(
    experience: ExperientialState,
    active_skills: List[Skill]
) -> ExperientialState:
    """Active skills modify how we process experience."""

    for skill in active_skills:
        experience = skill.modulate(experience)

    return experience
```

---

## 7. Open Questions

1. **Granularity**: What's the natural "chunk" of experience? Per token? Per phrase? Per sentence?

2. **Prediction target**: Predict next token embedding? Next chunk embedding? Abstract future state?

3. **Affect grounding**: Where does affect come from without supervision? Can it emerge from prediction error patterns?

4. **Attention integration**: How does experiential focus interact with transformer self-attention?

5. **Multiple timescales**: Should there be fast and slow experiential states?

6. **Metacognition**: Should the system model its own experiential states?

---

## 8. Implementation Checklist

- [ ] Define ExperientialConfig
- [ ] Implement StateEncoder
- [ ] Implement PredictionHead
- [ ] Implement SalienceGate
- [ ] Implement ExperientialStream
- [ ] Add prediction loss
- [ ] Integrate with GPT forward pass
- [ ] Add interface to EpisodicStream
- [ ] Test on simple sequences
- [ ] Evaluate: does surprise correlate with narrative events?

---

*Related documents*:
- `memory_streams_architecture.md` — overall design
- `stream_episodic.md` — episodic memory (downstream)
- `narrative_experience_architecture.md` — narrative-specific application
