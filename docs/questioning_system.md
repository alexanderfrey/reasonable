# Questioning System: Prediction Failure → Information Seeking

## Core Thesis

**Questions are the cognitive response to prediction failure.**

When the model's predictions about the world don't match reality, it experiences surprise. This prediction error can be resolved in several ways:
1. **Update the model** — Learn from the error (passive)
2. **Think harder** — Use internal computation to resolve uncertainty (active, internal)
3. **Ask a question** — Seek information to resolve the error (active, external)
4. **Form a hypothesis** — Speculate and continue with uncertainty
5. **Ignore** — Accept the error as noise (low importance)

Questioning is not merely a user-interface feature — it's a **fundamental cognitive mechanism** for actively resolving prediction failures rather than passively accepting them.

## Extended Thesis: Certainty as the Unified Control Signal

*Inspired by Continuous Thought Machines (CTM) — see `ctm_ideas.md`*

**Certainty bridges internal processing and external action.** The same signal that tells the model "keep thinking" also tells it "ask for help":

```
High certainty   → Stop processing, output answer
Medium certainty → Keep thinking (more internal ticks)
Low certainty    → After max thinking, ASK FOR HELP
```

This unifies several previously separate mechanisms:
- Understanding loops (iterate until confident)
- Question triggering (ask when uncertain)
- Action decisions (act when thresholds crossed)
- Memory crystallization (remember when certain about importance)

### The Certainty-Driven Processing Loop

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CERTAINTY-DRIVEN PROCESSING                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Input ──▶ ┌────────────────────────────────────────────────────────────┐      │
│             │              INTERNAL THINKING (ticks t = 1...T)            │      │
│             │                                                             │      │
│             │   for t in range(max_ticks):                               │      │
│             │       hidden = think_step(hidden, soma, memory)            │      │
│             │       certainty = certainty_head(hidden)                   │      │
│             │                                                             │      │
│             │       if certainty > threshold:                            │      │
│             │           break  ─────────────────────────────────────────▶│ OUTPUT
│             │                                                             │      │
│             │   # If we exit loop without breaking:                      │      │
│             │   # Still uncertain after max thinking                     │      │
│             └─────────────────────────────────────────────────────────────┘      │
│                              │                                                    │
│                              │ certainty still low                                │
│                              ▼                                                    │
│             ┌─────────────────────────────────────────────────────────────┐      │
│             │              QUESTION EMERGENCE                              │      │
│             │                                                              │      │
│             │   # Question is the natural output when thinking fails      │      │
│             │   question = generate_question(                             │      │
│             │       hidden_state=hidden,                                  │      │
│             │       uncertainty_source=identify_uncertainty(hidden),      │      │
│             │       failed_prediction=what_was_expected(hidden),          │      │
│             │   )                                                          │      │
│             │                                                              │      │
│             │   return QuestionAction(question)                           │      │
│             └─────────────────────────────────────────────────────────────┘      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: Questions are not a separate system triggered by thresholds. They are the **natural output** when internal processing fails to achieve certainty.

## The Prediction-Question Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREDICTION-QUESTION LOOP                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐                                                          │
│   │   PREDICT    │  "Based on my model, I expect X"                         │
│   │   p(x|model) │                                                          │
│   └──────┬───────┘                                                          │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────┐                                                          │
│   │   OBSERVE    │  "I actually perceive Y"                                 │
│   │   reality    │                                                          │
│   └──────┬───────┘                                                          │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────┐     ┌─────────────────────────────────────────────┐     │
│   │   COMPARE    │────▶│  PREDICTION ERROR = |X - Y|                  │     │
│   │   X vs Y     │     │  (surprise, confusion, uncertainty)          │     │
│   └──────────────┘     └─────────────────┬───────────────────────────┘     │
│                                          │                                   │
│                        ┌─────────────────┼─────────────────┐                │
│                        ▼                 ▼                 ▼                │
│                 ┌────────────┐    ┌────────────┐    ┌────────────┐         │
│                 │   SMALL    │    │   MEDIUM   │    │   LARGE    │         │
│                 │   error    │    │   error    │    │   error    │         │
│                 └─────┬──────┘    └─────┬──────┘    └─────┬──────┘         │
│                       │                 │                 │                 │
│                       ▼                 ▼                 ▼                 │
│                 ┌────────────┐    ┌────────────┐    ┌────────────┐         │
│                 │   UPDATE   │    │  QUESTION  │    │   ALERT    │         │
│                 │   model    │    │  "Why Y    │    │   + seek   │         │
│                 │  silently  │    │   not X?"  │    │   help     │         │
│                 └────────────┘    └─────┬──────┘    └─────┬──────┘         │
│                                         │                 │                 │
│                                         ▼                 ▼                 │
│                                  ┌─────────────────────────────┐           │
│                                  │      FORMULATE QUESTION     │           │
│                                  │                             │           │
│                                  │  "What information would    │           │
│                                  │   resolve this error?"      │           │
│                                  └──────────────┬──────────────┘           │
│                                                 │                           │
│                              ┌──────────────────┼──────────────────┐       │
│                              ▼                  ▼                  ▼       │
│                       ┌────────────┐     ┌────────────┐     ┌──────────┐  │
│                       │  ASK USER  │     │  ASK SELF  │     │  DEFER   │  │
│                       │  external  │     │  internal  │     │  store   │  │
│                       │  query     │     │  reasoning │     │  for     │  │
│                       └─────┬──────┘     └─────┬──────┘     │  later   │  │
│                             │                  │            └──────────┘  │
│                             ▼                  ▼                           │
│                       ┌─────────────────────────────┐                     │
│                       │      RECEIVE ANSWER         │                     │
│                       │      (external or derived)  │                     │
│                       └──────────────┬──────────────┘                     │
│                                      │                                     │
│                                      ▼                                     │
│                       ┌─────────────────────────────┐                     │
│                       │      UPDATE MODEL           │                     │
│                       │      reduce prediction      │                     │
│                       │      error for future       │                     │
│                       └─────────────────────────────┘                     │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Types of Prediction Failure

Different failures call for different questions:

### 1. **Content Prediction Failure**
"I predicted the next word/token would be X, but it was Y"

```python
@dataclass
class ContentPredictionFailure:
    predicted: torch.Tensor      # What model expected
    observed: torch.Tensor       # What actually appeared
    error: float                 # Magnitude of surprise
    context: torch.Tensor        # What led to this prediction

    def to_question(self) -> Question:
        """Convert to question about content."""
        # "Why did Y appear instead of X?"
        # "What does Y mean in this context?"
        return Question(
            type='content_clarification',
            focus=self.observed,
            context=self.context,
            uncertainty=self.error,
        )
```

### 2. **Semantic Prediction Failure**
"I predicted the meaning/intent would be A, but it seems to be B"

```python
@dataclass
class SemanticPredictionFailure:
    predicted_meaning: torch.Tensor   # Expected semantic
    observed_meaning: torch.Tensor    # Apparent semantic
    ambiguity: float                  # How unclear is the meaning?

    def to_question(self) -> Question:
        """Convert to question about meaning."""
        # "Did you mean A or B?"
        # "Could you clarify what you meant by X?"
        return Question(
            type='semantic_clarification',
            alternatives=[self.predicted_meaning, self.observed_meaning],
            ambiguity=self.ambiguity,
        )
```

### 3. **Visual Prediction Failure**
"I predicted the image would contain X, but I see Y (or something unrecognizable)"

```python
@dataclass
class VisualPredictionFailure:
    predicted_visual: torch.Tensor    # Expected visual features
    observed_visual: torch.Tensor     # Actual visual features
    region: Tuple[int, int]           # Where in the image
    recognizable: bool                # Can I identify what I see?

    def to_question(self) -> Question:
        """Convert to question about visual content."""
        if not self.recognizable:
            # "What is this object?"
            return Question(type='visual_identification', region=self.region)
        else:
            # "Why is X here instead of Y?"
            return Question(type='visual_explanation',
                          expected=self.predicted_visual,
                          observed=self.observed_visual)
```

### 4. **Self-Prediction Failure (Meta)**
"I predicted I would react with state S, but I actually feel state S'"

```python
@dataclass
class SelfPredictionFailure:
    predicted_soma: torch.Tensor      # Expected internal state
    actual_soma: torch.Tensor         # Actual internal state
    self_surprise: float              # How unexpected was my reaction?

    def to_question(self) -> Question:
        """Convert to question about self."""
        # "Why did I react this way?"
        # "What about this triggered this response?"
        return Question(
            type='self_reflection',
            unexpected_state=self.actual_soma,
            trigger_context=...,
        )
```

### 5. **Coherence Failure**
"These things don't fit together — something is inconsistent"

```python
@dataclass
class CoherenceFailure:
    element_a: torch.Tensor           # First element
    element_b: torch.Tensor           # Second element
    expected_relation: str            # How they should relate
    actual_relation: str              # How they actually relate

    def to_question(self) -> Question:
        """Convert to question about consistency."""
        # "How can A and B both be true?"
        # "What am I missing that would make this coherent?"
        return Question(
            type='coherence_resolution',
            conflicting_elements=[self.element_a, self.element_b],
        )
```

---

## Certainty-Driven Architecture

*This section integrates ideas from Continuous Thought Machines (CTM) with our questioning system.*

### Core Components

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CERTAINTY-DRIVEN QUESTIONING ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         CERTAINTY MODULE                                    │ │
│  │                                                                             │ │
│  │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │ │
│  │  │  CERTAINTY  │     │ UNCERTAINTY │     │  THINKING   │                  │ │
│  │  │    HEAD     │────▶│   SOURCE    │────▶│  BUDGET     │                  │ │
│  │  │  c(h) → [0,1]│     │  IDENTIFIER │     │  ALLOCATOR  │                  │ │
│  │  └─────────────┘     └─────────────┘     └─────────────┘                  │ │
│  │         │                   │                   │                          │ │
│  └─────────┼───────────────────┼───────────────────┼──────────────────────────┘ │
│            │                   │                   │                            │
│            ▼                   ▼                   ▼                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      THINKING LOOP                                       │   │
│  │                                                                          │   │
│  │   ┌──────────┐    ┌──────────┐    ┌──────────┐         ┌──────────┐    │   │
│  │   │  Tick 1  │───▶│  Tick 2  │───▶│  Tick 3  │───▶ ... │  Tick T  │    │   │
│  │   │ c=0.3    │    │ c=0.5    │    │ c=0.7    │         │ c=0.85   │    │   │
│  │   └──────────┘    └──────────┘    └──────────┘         └────┬─────┘    │   │
│  │                                                              │          │   │
│  │        certainty increasing with thinking ──────────────────▶│          │   │
│  │                                                              │          │   │
│  └──────────────────────────────────────────────────────────────┼──────────┘   │
│                                                                 │               │
│                    ┌────────────────────────────────────────────┼────┐         │
│                    │                                            │    │         │
│                    ▼                                            ▼    ▼         │
│            ┌──────────────┐                            ┌──────────────────┐    │
│            │   CERTAIN    │                            │    UNCERTAIN     │    │
│            │   c > θ      │                            │    c < θ after   │    │
│            │              │                            │    max ticks     │    │
│            └──────┬───────┘                            └────────┬─────────┘    │
│                   │                                             │              │
│                   ▼                                             ▼              │
│            ┌──────────────┐                            ┌──────────────────┐    │
│            │   OUTPUT     │                            │   GENERATE       │    │
│            │   ANSWER     │                            │   QUESTION       │    │
│            └──────────────┘                            └──────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### CertaintyHead

The certainty head estimates how confident the model is in its current understanding:

```python
class CertaintyHead(nn.Module):
    """
    Estimates certainty of current hidden state.

    Certainty is NOT the same as prediction confidence (softmax temperature).
    It's a meta-cognitive assessment: "Do I understand this well enough?"

    Trained to be calibrated: high certainty should correlate with correctness.
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model

        # Main certainty estimation
        self.certainty_net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Optional: certainty from multiple signals
        self.use_multi_signal = config.certainty_multi_signal
        if self.use_multi_signal:
            # Combine: hidden state, prediction entropy, soma state
            self.multi_signal_net = nn.Sequential(
                nn.Linear(config.d_model + 1 + config.d_soma, config.d_model // 2),
                nn.GELU(),
                nn.Linear(config.d_model // 2, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        hidden: torch.Tensor,              # [B, seq_len, d_model]
        prediction_entropy: Optional[torch.Tensor] = None,  # [B, seq_len]
        soma: Optional[torch.Tensor] = None,  # [B, d_soma]
    ) -> torch.Tensor:
        """
        Compute certainty score.

        Returns:
            certainty: [B, seq_len] or [B] if pooled
        """
        if self.use_multi_signal and prediction_entropy is not None and soma is not None:
            # Expand soma to match sequence
            soma_expanded = soma.unsqueeze(1).expand(-1, hidden.shape[1], -1)
            entropy_expanded = prediction_entropy.unsqueeze(-1)

            combined = torch.cat([hidden, entropy_expanded, soma_expanded], dim=-1)
            certainty = self.multi_signal_net(combined).squeeze(-1)
        else:
            certainty = self.certainty_net(hidden).squeeze(-1)

        return certainty

    def calibration_loss(
        self,
        certainty: torch.Tensor,  # [B]
        was_correct: torch.Tensor,  # [B] binary
    ) -> torch.Tensor:
        """
        Calibration loss: certainty should match actual correctness.

        High certainty + wrong = bad
        Low certainty + right = suboptimal but acceptable
        High certainty + right = good
        Low certainty + wrong = good (honest uncertainty)
        """
        return F.binary_cross_entropy(certainty, was_correct.float())
```

### UncertaintySourceIdentifier

When certainty is low, identify WHERE the uncertainty comes from:

```python
class UncertaintySourceIdentifier(nn.Module):
    """
    Identifies the source of uncertainty in the hidden state.

    This helps generate targeted questions:
    - If uncertainty is about a word → ask about that word
    - If uncertainty is about an image region → point to that region
    - If uncertainty is about coherence → ask about the relationship
    """

    def __init__(self, config):
        super().__init__()

        # Attention over hidden states to find uncertainty source
        self.uncertainty_attention = nn.MultiheadAttention(
            config.d_model, num_heads=4, batch_first=True
        )

        # Learnable "uncertainty query" - what are we uncertain about?
        self.uncertainty_query = nn.Parameter(torch.randn(1, 1, config.d_model))

        # Classify uncertainty type
        self.uncertainty_type_head = nn.Linear(config.d_model, 5)
        # Types: [content, semantic, visual, self, coherence]

        # Extract the uncertain element
        self.element_extractor = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        hidden: torch.Tensor,      # [B, seq_len, d_model]
        certainty: torch.Tensor,   # [B, seq_len]
    ) -> Dict[str, torch.Tensor]:
        """
        Identify source of uncertainty.
        """
        B, L, D = hidden.shape

        # Expand query
        query = self.uncertainty_query.expand(B, -1, -1)

        # Weight hidden states by inverse certainty (attend to uncertain parts)
        uncertainty_weights = 1 - certainty  # [B, seq_len]
        weighted_hidden = hidden * uncertainty_weights.unsqueeze(-1)

        # Attend to find the most uncertain element
        attended, attn_weights = self.uncertainty_attention(
            query, weighted_hidden, weighted_hidden
        )  # [B, 1, d_model]

        # Classify uncertainty type
        uncertainty_type_logits = self.uncertainty_type_head(attended.squeeze(1))
        uncertainty_type = uncertainty_type_logits.argmax(dim=-1)

        # Extract the uncertain element representation
        uncertain_element = self.element_extractor(attended.squeeze(1))

        # Find which position is most uncertain
        most_uncertain_pos = uncertainty_weights.argmax(dim=-1)

        return {
            'uncertainty_type': uncertainty_type,
            'uncertainty_type_logits': uncertainty_type_logits,
            'uncertain_element': uncertain_element,
            'attention_weights': attn_weights.squeeze(1),  # [B, seq_len]
            'most_uncertain_position': most_uncertain_pos,
        }
```

### ThinkingLoop

The core processing loop that thinks until certain or generates a question:

```python
class ThinkingLoop(nn.Module):
    """
    CTM-inspired thinking loop with certainty-driven halting.

    Process:
    1. Initialize from input
    2. Iteratively refine hidden state
    3. Check certainty at each tick
    4. If certain: output answer
    5. If still uncertain after max ticks: generate question
    """

    def __init__(self, config):
        super().__init__()
        self.max_ticks = config.max_thinking_ticks
        self.certainty_threshold = config.certainty_threshold
        self.min_ticks = config.min_thinking_ticks  # Always think at least this much

        # Single thinking step
        self.think_step = ThinkStep(config)

        # Certainty estimation
        self.certainty_head = CertaintyHead(config)

        # Uncertainty source identification
        self.uncertainty_identifier = UncertaintySourceIdentifier(config)

        # Question generation (when uncertain)
        self.question_generator = QuestionFromUncertainty(config)

        # Track thinking history (for synchronization / analysis)
        self.track_history = config.track_thinking_history

    def forward(
        self,
        hidden: torch.Tensor,       # [B, seq_len, d_model]
        soma: torch.Tensor,         # [B, d_soma]
        memory: Optional[torch.Tensor] = None,  # Retrieved memories
        return_trajectory: bool = False,
    ) -> Dict[str, Any]:
        """
        Think until certain or max ticks reached.
        """
        B = hidden.shape[0]

        # Initialize tracking
        certainty_trajectory = []
        hidden_trajectory = [] if return_trajectory else None

        # Thinking loop
        final_tick = self.max_ticks
        halted_early = torch.zeros(B, dtype=torch.bool, device=hidden.device)

        for tick in range(self.max_ticks):
            # One thinking step
            hidden, step_info = self.think_step(hidden, soma, memory)

            # Compute certainty
            certainty = self.certainty_head(hidden, soma=soma)  # [B, seq_len]
            mean_certainty = certainty.mean(dim=-1)  # [B]

            certainty_trajectory.append(mean_certainty)
            if return_trajectory:
                hidden_trajectory.append(hidden.clone())

            # Check halting condition (after min_ticks)
            if tick >= self.min_ticks:
                newly_halted = (mean_certainty > self.certainty_threshold) & ~halted_early
                halted_early = halted_early | newly_halted

                # If all samples have halted, we can stop
                if halted_early.all():
                    final_tick = tick + 1
                    break

        # Stack trajectories
        certainty_trajectory = torch.stack(certainty_trajectory, dim=1)  # [B, ticks]

        # Final certainty
        final_certainty = certainty_trajectory[:, -1]

        # Determine output type for each sample
        is_certain = final_certainty > self.certainty_threshold
        needs_question = ~is_certain

        # Generate questions for uncertain samples
        questions = None
        if needs_question.any():
            uncertainty_info = self.uncertainty_identifier(hidden, certainty)
            questions = self.question_generator(
                hidden_state=hidden,
                uncertainty_info=uncertainty_info,
                soma=soma,
                certainty=final_certainty,
            )

        return {
            'hidden': hidden,
            'certainty': final_certainty,
            'certainty_trajectory': certainty_trajectory,
            'hidden_trajectory': hidden_trajectory,
            'ticks_used': final_tick,
            'is_certain': is_certain,
            'needs_question': needs_question,
            'questions': questions,
            'halted_early': halted_early,
        }


class ThinkStep(nn.Module):
    """
    A single thinking step that refines the hidden state.

    Incorporates:
    - Self-attention (refine based on context)
    - Memory attention (incorporate retrieved memories)
    - Soma modulation (internal state affects thinking)
    """

    def __init__(self, config):
        super().__init__()

        # Self-refinement
        self.self_attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, batch_first=True
        )

        # Memory integration (if available)
        self.memory_attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, batch_first=True
        )

        # Soma modulation
        self.soma_gate = nn.Sequential(
            nn.Linear(config.d_soma, config.d_model),
            nn.Sigmoid(),
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        soma: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """One thinking step."""

        # Self-attention
        residual = hidden
        hidden = self.norm1(hidden)
        hidden_attn, self_attn_weights = self.self_attn(hidden, hidden, hidden)
        hidden = residual + hidden_attn

        # Memory attention (if available)
        if memory is not None:
            residual = hidden
            hidden = self.norm2(hidden)
            mem_attn, mem_attn_weights = self.memory_attn(hidden, memory, memory)
            hidden = residual + mem_attn
        else:
            mem_attn_weights = None

        # Soma modulation
        soma_gate = self.soma_gate(soma).unsqueeze(1)  # [B, 1, d_model]
        hidden = hidden * soma_gate

        # FFN
        residual = hidden
        hidden = self.norm3(hidden)
        hidden = residual + self.ffn(hidden)

        return hidden, {
            'self_attn_weights': self_attn_weights,
            'memory_attn_weights': mem_attn_weights,
        }
```

### QuestionFromUncertainty

Generates questions when thinking fails to reach certainty:

```python
class QuestionFromUncertainty(nn.Module):
    """
    Generate questions from the uncertainty that remains after thinking.

    The question should:
    1. Target the specific source of uncertainty
    2. Request exactly the information needed to become certain
    3. Be phrased appropriately for the uncertainty type
    """

    def __init__(self, config):
        super().__init__()

        # Question content generator
        self.question_content = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Question type refinement (based on uncertainty type)
        self.type_to_template = nn.Linear(5, config.n_question_templates)

        # Urgency estimation (how badly do we need this answered?)
        self.urgency_head = nn.Sequential(
            nn.Linear(config.d_model + 1, 1),  # content + certainty
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_state: torch.Tensor,         # [B, seq_len, d_model]
        uncertainty_info: Dict,
        soma: torch.Tensor,
        certainty: torch.Tensor,            # [B] final certainty
    ) -> List[Question]:
        """
        Generate questions for uncertain samples.
        """
        B = hidden_state.shape[0]

        # Pool hidden state
        hidden_pooled = hidden_state.mean(dim=1)  # [B, d_model]

        # Combine with uncertain element
        combined = torch.cat([
            hidden_pooled,
            uncertainty_info['uncertain_element']
        ], dim=-1)

        # Generate question content embedding
        question_content = self.question_content(combined)

        # Determine question type from uncertainty type
        uncertainty_type_onehot = F.one_hot(
            uncertainty_info['uncertainty_type'], num_classes=5
        ).float()
        template_logits = self.type_to_template(uncertainty_type_onehot)

        # Compute urgency (lower certainty = higher urgency)
        urgency_input = torch.cat([question_content, (1 - certainty).unsqueeze(-1)], dim=-1)
        urgency = self.urgency_head(urgency_input).squeeze(-1)

        # Create Question objects
        questions = []
        for b in range(B):
            # Map uncertainty type to question source
            uncertainty_type_names = ['content', 'semantic', 'visual', 'self', 'coherence']
            source_type = uncertainty_type_names[uncertainty_info['uncertainty_type'][b].item()]

            question = Question(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                source_type=source_type,
                source_failure=None,  # Will be filled if we have specific failure
                source_location={
                    'position': uncertainty_info['most_uncertain_position'][b].item(),
                    'attention': uncertainty_info['attention_weights'][b].cpu().numpy(),
                },
                type=f'{source_type}_from_thinking',
                focus=question_content[b],
                context=hidden_pooled[b],
                uncertainty_magnitude=1 - certainty[b].item(),
                uncertainty_type=source_type,
                information_need='resolution',
                resolution_strategies=['ask_user'],  # Thinking already failed
                preferred_strategy='ask_user',
                cost_of_asking=0.3,  # Base cost
                cost_of_not_knowing=urgency[b].item(),
                status='pending',
                answer=None,
                resolution_confidence=0.0,
                # New fields for certainty-driven system
                thinking_ticks_used=None,  # Will be filled by caller
                final_certainty=certainty[b].item(),
            )
            questions.append(question)

        return questions
```

### Certainty-Driven Loss Function

Training the certainty-driven system:

```python
class CertaintyDrivenLoss(nn.Module):
    """
    Loss function for certainty-driven processing.

    Inspired by CTM: optimize at both minimum loss tick AND maximum certainty tick.

    Components:
    1. Task loss at best tick (minimize prediction error)
    2. Task loss at most certain tick (certainty should correlate with correctness)
    3. Calibration loss (certainty should be well-calibrated)
    4. Efficiency loss (prefer fewer ticks when possible)
    """

    def __init__(self, config):
        super().__init__()
        self.calibration_weight = config.calibration_loss_weight
        self.efficiency_weight = config.efficiency_loss_weight
        self.certainty_bonus_weight = config.certainty_bonus_weight

    def forward(
        self,
        logits_trajectory: List[torch.Tensor],  # [ticks] of [B, seq_len, vocab]
        targets: torch.Tensor,                   # [B, seq_len]
        certainty_trajectory: torch.Tensor,      # [B, ticks]
        was_correct: torch.Tensor,               # [B] binary
    ) -> Dict[str, torch.Tensor]:
        """
        Compute certainty-driven loss.
        """
        B = targets.shape[0]
        T = len(logits_trajectory)

        # Compute task loss at each tick
        task_losses = []
        for t in range(T):
            loss_t = F.cross_entropy(
                logits_trajectory[t].view(-1, logits_trajectory[t].size(-1)),
                targets.view(-1),
                reduction='none'
            ).view(B, -1).mean(dim=-1)  # [B]
            task_losses.append(loss_t)
        task_losses = torch.stack(task_losses, dim=1)  # [B, T]

        # Find tick with minimum loss (per sample)
        t_min_loss = task_losses.argmin(dim=1)  # [B]
        loss_at_min = task_losses.gather(1, t_min_loss.unsqueeze(1)).squeeze(1)

        # Find tick with maximum certainty (per sample)
        t_max_certainty = certainty_trajectory.argmax(dim=1)  # [B]
        loss_at_max_cert = task_losses.gather(1, t_max_certainty.unsqueeze(1)).squeeze(1)
        certainty_at_max = certainty_trajectory.gather(1, t_max_certainty.unsqueeze(1)).squeeze(1)

        # Combined task loss (CTM-style)
        task_loss = (loss_at_min + loss_at_max_cert) / 2

        # Calibration loss: certainty should match correctness
        calibration_loss = F.binary_cross_entropy(certainty_at_max, was_correct.float())

        # Efficiency loss: prefer halting early
        # Penalize using more ticks than necessary
        ticks_used = (certainty_trajectory < self.certainty_threshold).sum(dim=1).float()
        efficiency_loss = ticks_used.mean() / T  # Normalized

        # Certainty bonus: reward high certainty when correct
        certainty_bonus = -(certainty_at_max * was_correct.float()).mean()

        # Total loss
        total_loss = (
            task_loss.mean() +
            self.calibration_weight * calibration_loss +
            self.efficiency_weight * efficiency_loss +
            self.certainty_bonus_weight * certainty_bonus
        )

        return {
            'total_loss': total_loss,
            'task_loss': task_loss.mean(),
            'calibration_loss': calibration_loss,
            'efficiency_loss': efficiency_loss,
            'certainty_bonus': certainty_bonus,
            't_min_loss': t_min_loss.float().mean(),
            't_max_certainty': t_max_certainty.float().mean(),
            'mean_certainty': certainty_at_max.mean(),
        }
```

### Integration with Main Architecture

How certainty-driven questioning integrates with MemoryAugmentedGPT:

```python
class MemoryAugmentedGPT:
    def __init__(self, ..., use_certainty_driven=True):
        # ... existing init ...

        if use_certainty_driven:
            self.thinking_loop = ThinkingLoop(config)
            self.certainty_threshold = config.certainty_threshold

    def forward(self, input_ids, ...):
        # 1. Initial encoding
        hidden = self.gpt(input_ids, soma_q_bias=soma_q_bias)

        # 2. Experiential processing (surprise, memory retrieval, etc.)
        exp_output = self.experiential(hidden)
        hidden = exp_output['hidden']

        # 3. Certainty-driven thinking loop
        if self.thinking_loop is not None:
            thinking_output = self.thinking_loop(
                hidden=hidden,
                soma=self.self_state.get_soma(),
                memory=exp_output.get('retrieved_memories'),
            )

            hidden = thinking_output['hidden']

            # If questions were generated, add to output
            if thinking_output['questions'] is not None:
                memory_output['questions'] = thinking_output['questions']
                memory_output['needs_user_input'] = True

            # Track certainty
            memory_output['certainty'] = thinking_output['certainty']
            memory_output['ticks_used'] = thinking_output['ticks_used']

        # 4. Output
        logits = self.lm_head(hidden)

        return logits, hidden, memory_output
```

---

## The Question as First-Class Object

A question is not just a string — it's a structured object representing an information need:

```python
@dataclass
class Question:
    """A question arising from prediction failure."""

    # Identity
    id: str                           # Unique identifier
    timestamp: float                  # When generated

    # Source
    source_type: str                  # 'content', 'semantic', 'visual', 'self', 'coherence'
    source_failure: PredictionFailure # The failure that generated this
    source_location: Dict             # Where in processing (layer, position, etc.)

    # Content
    type: str                         # Question type within source
    focus: torch.Tensor               # What the question is about (embedding)
    context: torch.Tensor             # Surrounding context

    # Uncertainty characterization
    uncertainty_magnitude: float      # How large is the prediction error?
    uncertainty_type: str             # 'ambiguous', 'novel', 'contradictory', 'missing'
    information_need: str             # What kind of answer would help?

    # Resolution
    resolution_strategies: List[str]  # ['ask_user', 'reason', 'hypothesize', 'memory']
    preferred_strategy: str           # Best strategy for this question
    cost_of_asking: float             # Social/practical cost of asking user
    cost_of_not_knowing: float        # Cost of proceeding without answer

    # State
    status: str                       # 'pending', 'asked', 'answered', 'resolved', 'abandoned'
    answer: Optional[Any]             # The answer if received
    resolution_confidence: float      # How confident in resolution?

    def should_ask_user(self) -> bool:
        """Decide if this question warrants user interruption."""
        return (
            self.cost_of_not_knowing > self.cost_of_asking and
            self.preferred_strategy == 'ask_user' and
            self.uncertainty_magnitude > ASK_THRESHOLD
        )

    def can_self_resolve(self) -> bool:
        """Can this question potentially be answered through reasoning?"""
        return 'reason' in self.resolution_strategies

    def to_natural_language(self) -> str:
        """Convert to human-readable question."""
        # Template-based generation conditioned on question type and focus
        ...
```

## Question Generation from Prediction Error

```python
class QuestionGenerator(nn.Module):
    """
    Transforms prediction failures into questions.

    This is the core mapping: error → information need → question
    """

    def __init__(self, config):
        self.d_model = config.d_model

        # Analyze the prediction failure
        self.failure_analyzer = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),  # predicted + observed
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Classify what type of question is needed
        self.question_type_classifier = nn.Linear(config.d_model, config.n_question_types)

        # Estimate costs
        self.cost_estimator = nn.Sequential(
            nn.Linear(config.d_model + config.d_soma, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 2),  # [cost_of_asking, cost_of_not_knowing]
        )

        # Generate question embedding (what information is needed)
        self.information_need_head = nn.Linear(config.d_model, config.d_model)

        # Strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(config.d_model + 2, config.d_model // 2),  # failure + costs
            nn.GELU(),
            nn.Linear(config.d_model // 2, 4),  # [ask_user, reason, hypothesize, memory]
        )

    def forward(
        self,
        predicted: torch.Tensor,       # [B, d_model] what was expected
        observed: torch.Tensor,        # [B, d_model] what was observed
        context: torch.Tensor,         # [B, d_model] surrounding context
        soma: torch.Tensor,            # [B, d_soma] current internal state
        error_magnitude: torch.Tensor, # [B] prediction error
    ) -> Dict[str, Any]:
        """
        Generate questions from prediction failures.
        """
        B = predicted.shape[0]

        # Analyze the failure
        failure_repr = self.failure_analyzer(
            torch.cat([predicted, observed], dim=-1)
        )  # [B, d_model]

        # What type of question?
        question_type_logits = self.question_type_classifier(failure_repr)
        question_type = question_type_logits.argmax(dim=-1)

        # What information is needed?
        information_need = self.information_need_head(failure_repr)

        # Estimate costs
        costs = self.cost_estimator(
            torch.cat([failure_repr, soma], dim=-1)
        )  # [B, 2]
        cost_of_asking = costs[:, 0]
        cost_of_not_knowing = costs[:, 1]

        # Select resolution strategy
        strategy_input = torch.cat([failure_repr, costs], dim=-1)
        strategy_logits = self.strategy_selector(strategy_input)
        preferred_strategy = strategy_logits.argmax(dim=-1)

        # Should we ask?
        should_ask = (
            (cost_of_not_knowing > cost_of_asking) &
            (preferred_strategy == 0) &  # ask_user strategy
            (error_magnitude > self.ask_threshold)
        )

        return {
            'failure_repr': failure_repr,
            'question_type': question_type,
            'question_type_logits': question_type_logits,
            'information_need': information_need,
            'cost_of_asking': cost_of_asking,
            'cost_of_not_knowing': cost_of_not_knowing,
            'strategy_logits': strategy_logits,
            'preferred_strategy': preferred_strategy,
            'should_ask': should_ask,
        }
```

## Integration: Question Buffer

Questions don't need to be asked immediately. They accumulate and can be:
- Batched ("I have several questions about this...")
- Prioritized (most important first)
- Resolved internally before asking
- Abandoned if context changes

```python
class QuestionBuffer:
    """
    Accumulates questions, manages priorities, handles resolution.
    """

    def __init__(self, config):
        self.max_pending = config.max_pending_questions
        self.pending: List[Question] = []
        self.asked: List[Question] = []
        self.resolved: List[Question] = []

        # Priority scoring
        self.priority_scorer = nn.Sequential(
            nn.Linear(config.d_model + 2, 1),  # question + costs → priority
        )

    def add(self, question: Question):
        """Add a new question to the buffer."""
        if len(self.pending) >= self.max_pending:
            # Remove lowest priority question
            self._evict_lowest_priority()
        self.pending.append(question)
        self._recompute_priorities()

    def get_top_questions(self, k: int = 3) -> List[Question]:
        """Get the k highest priority questions to ask."""
        sorted_pending = sorted(
            self.pending,
            key=lambda q: q.priority,
            reverse=True
        )
        return [q for q in sorted_pending[:k] if q.should_ask_user()]

    def try_self_resolve(self, reasoner: 'Reasoner') -> List[Question]:
        """Attempt to resolve questions through reasoning."""
        resolved = []
        still_pending = []

        for question in self.pending:
            if question.can_self_resolve():
                result = reasoner.attempt_resolution(question)
                if result['resolved']:
                    question.status = 'resolved'
                    question.answer = result['answer']
                    question.resolution_confidence = result['confidence']
                    resolved.append(question)
                else:
                    still_pending.append(question)
            else:
                still_pending.append(question)

        self.pending = still_pending
        self.resolved.extend(resolved)
        return resolved

    def receive_answer(self, question_id: str, answer: Any):
        """Process an answer to a question."""
        for question in self.asked:
            if question.id == question_id:
                question.answer = answer
                question.status = 'answered'
                self._integrate_answer(question)
                break

    def _integrate_answer(self, question: Question):
        """Integrate the answer back into the system."""
        # This is where the answer updates:
        # 1. The model's understanding
        # 2. Memory (store Q&A pair)
        # 3. Soma (satisfaction from resolution)
        pass

    def context_changed(self, new_context: torch.Tensor):
        """Update questions based on new context."""
        # Some questions may become irrelevant
        # Others may become more urgent
        still_relevant = []
        for question in self.pending:
            relevance = self._compute_relevance(question, new_context)
            if relevance > RELEVANCE_THRESHOLD:
                question.priority *= relevance  # Adjust priority
                still_relevant.append(question)
            else:
                question.status = 'abandoned'
        self.pending = still_relevant
```

## Integration with Experiential System

The experiential system already computes surprise. We hook question generation into this:

```python
# In experiential.py, after computing surprise

class ExperientialModule(nn.Module):
    def forward(self, hidden_states, ...):
        # ... existing processing ...

        # Compute prediction and surprise
        predicted = self.predictor(hidden_states[:, :-1, :])
        observed = hidden_states[:, 1:, :]
        surprise = self.compute_surprise(predicted, observed)

        # NEW: Generate questions from high-surprise positions
        if self.questioning_enabled:
            high_surprise_mask = surprise > self.surprise_threshold

            if high_surprise_mask.any():
                questions = self.question_generator(
                    predicted=predicted[high_surprise_mask],
                    observed=observed[high_surprise_mask],
                    context=self._get_context(hidden_states, high_surprise_mask),
                    soma=self.self_state.get_soma() if self.self_state else None,
                    error_magnitude=surprise[high_surprise_mask],
                )

                # Add to question buffer
                for q in self._create_question_objects(questions):
                    self.question_buffer.add(q)

        # ... rest of processing ...
```

## Integration with Soma

Questions affect and are affected by soma:

```python
class QuestionSomaIntegration:
    """
    Bidirectional relationship between questions and soma.
    """

    # Questions → Soma effects
    QUESTION_SOMA_EFFECTS = {
        'pending_uncertainty': {
            'valence': -0.1,      # Slight negative (unresolved = uncomfortable)
            'arousal': +0.2,      # Elevated alertness
            'engagement': +0.3,   # Increased attention
        },
        'asked_waiting': {
            'valence': 0.0,       # Neutral (action taken)
            'arousal': +0.1,      # Slight anticipation
            'engagement': +0.2,
        },
        'resolved_satisfied': {
            'valence': +0.3,      # Positive (closure)
            'arousal': -0.1,      # Relaxation
            'certainty': +0.4,    # Increased confidence
        },
        'abandoned_frustrated': {
            'valence': -0.2,      # Negative
            'arousal': -0.1,      # Slight deflation
        },
    }

    def update_soma_from_questions(self, soma: torch.Tensor, question_buffer: QuestionBuffer) -> torch.Tensor:
        """Update soma based on question states."""

        # Count questions in each state
        n_pending = len(question_buffer.pending)
        n_waiting = len([q for q in question_buffer.asked if q.status == 'asked'])
        n_resolved = len([q for q in question_buffer.resolved if q.status == 'resolved'])

        # Accumulate effects
        delta = torch.zeros_like(soma)

        if n_pending > 0:
            delta += n_pending * self._effect_to_tensor('pending_uncertainty')
        if n_waiting > 0:
            delta += n_waiting * self._effect_to_tensor('asked_waiting')
        if n_resolved > 0:
            delta += self._effect_to_tensor('resolved_satisfied')

        return soma + delta * self.question_soma_scale

    # Soma → Question effects
    def soma_affects_questioning(self, soma: torch.Tensor, question: Question) -> Question:
        """Soma state modulates question properties."""

        # High anxiety → more likely to ask
        if self._get_anxiety(soma) > ANXIETY_THRESHOLD:
            question.cost_of_not_knowing *= 1.5

        # High confidence → less likely to ask
        if self._get_confidence(soma) > CONFIDENCE_THRESHOLD:
            question.cost_of_asking *= 1.3

        # High curiosity → more questions, lower threshold
        if self._get_curiosity(soma) > CURIOSITY_THRESHOLD:
            question.uncertainty_magnitude *= 1.2  # Amplify

        return question
```

## Integration with Action Module

Questions are a type of action. Update `action_module.md` concepts:

```python
# Enhanced action space with question-specific actions

class QuestionAction(Action):
    """
    Action specifically for asking questions.
    More nuanced than generic 'contact_user'.
    """

    name: str = 'ask_question'
    tier: int = 1  # User-facing

    # Question-specific properties
    question: Question
    urgency: float              # How urgent is this question?
    interruptibility: str       # 'immediate', 'next_pause', 'batch'
    fallback_strategy: str      # What to do if user doesn't respond

    def execute(self, context: Dict) -> Dict:
        """Execute the question-asking action."""

        # Format question for user
        formatted = self.question.to_natural_language()

        # Add context if helpful
        if self.question.uncertainty_type == 'ambiguous':
            formatted += f"\n(I see multiple possible interpretations...)"
        elif self.question.uncertainty_type == 'novel':
            formatted += f"\n(This is new to me...)"

        return {
            'interrupt': self.interruptibility == 'immediate',
            'message': formatted,
            'expects_response': True,
            'question_id': self.question.id,
            'fallback': self.fallback_strategy,
        }


class ActionSpace:
    """Enhanced action space with rich question support."""

    def __init__(self, config):
        # ... existing actions ...

        # Question-specific actions
        self.question_actions = {
            'ask_clarification': QuestionAction(
                question_type='semantic_clarification',
                urgency_default=0.7,
                interruptibility='next_pause',
            ),
            'ask_identification': QuestionAction(
                question_type='visual_identification',
                urgency_default=0.8,
                interruptibility='immediate',
            ),
            'ask_confirmation': QuestionAction(
                question_type='hypothesis_confirmation',
                urgency_default=0.5,
                interruptibility='batch',
            ),
            'ask_elaboration': QuestionAction(
                question_type='detail_request',
                urgency_default=0.4,
                interruptibility='batch',
            ),
        }
```

## Integration with Vision Module

Visual prediction failures generate visual questions:

```python
# In vision_module.py

class VisualQuestionIntegration:
    """
    Visual prediction failures → questions about what is seen.
    """

    def visual_failure_to_question(
        self,
        visual_surprise: Dict,
        visual_tokens: torch.Tensor,
        comprehension: Dict,
    ) -> List[Question]:
        """Convert visual understanding failures to questions."""

        questions = []

        for region_idx in range(visual_tokens.shape[1]):
            if comprehension['needs_resolution'][region_idx]:

                failure = VisualPredictionFailure(
                    predicted_visual=visual_surprise['predicted_visual'][:, region_idx],
                    observed_visual=visual_tokens[:, region_idx],
                    region=self._idx_to_region(region_idx),
                    recognizable=comprehension['confidence'][region_idx] > 0.3,
                )

                question = failure.to_question()

                # Add visual-specific metadata
                question.visual_region = self._idx_to_region(region_idx)
                question.visual_context = self._get_surrounding_regions(region_idx)
                question.can_point_to = True  # Can highlight in image

                questions.append(question)

        return questions
```

## Integration with Synthesis Module

Questions can be asked internally during synthesis:

```python
# In synthesis_module.py

class InternalQuestioning:
    """
    During synthesis (reflect, dream, etc.), the model can ask itself questions.
    """

    def synthesize_with_questioning(self, mode: str, soma: torch.Tensor) -> Dict:
        """
        Synthesis that includes internal questioning.
        """

        if mode == 'reflect':
            # Generate reflection
            reflection = self._reflect(soma)

            # Check for self-prediction failures
            self_surprise = self._compute_self_surprise(reflection)

            if self_surprise > threshold:
                # Ask internal question
                internal_question = Question(
                    type='self_reflection',
                    source_type='self',
                    focus=reflection['unexpected_element'],
                    resolution_strategies=['reason'],  # No user asking during synthesis
                )

                # Attempt self-resolution
                resolution = self._reason_about_self(internal_question)

                if resolution['resolved']:
                    reflection['insight'] = resolution['answer']
                else:
                    reflection['unresolved_question'] = internal_question

        return reflection
```

## The Question Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       QUESTION LIFECYCLE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. BIRTH                                                               │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │ Prediction failure detected → Question object created         │     │
│   │ - Source identified (content/visual/self/coherence)           │     │
│   │ - Information need characterized                              │     │
│   │ - Costs estimated                                             │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│   2. TRIAGE                                                             │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │ Resolution strategy selected:                                 │     │
│   │ - Can self-resolve? → Try reasoning first                     │     │
│   │ - Must ask user? → Add to question buffer                     │     │
│   │ - Low priority? → Defer or abandon                            │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│              ┌───────────────┼───────────────┐                         │
│              ▼               ▼               ▼                         │
│   3a. SELF-RESOLUTION   3b. USER QUERY   3c. DEFERRAL                  │
│   ┌─────────────────┐  ┌─────────────┐  ┌─────────────┐               │
│   │ Reasoning loop  │  │ Format      │  │ Store in    │               │
│   │ - Context       │  │ question    │  │ buffer      │               │
│   │ - Memory        │  │ for user    │  │ May become  │               │
│   │ - Inference     │  │             │  │ irrelevant  │               │
│   └────────┬────────┘  └──────┬──────┘  └─────────────┘               │
│            │                  │                                         │
│            ▼                  ▼                                         │
│   ┌─────────────────┐  ┌─────────────────────────────────────────┐    │
│   │ Resolved?       │  │ User responds                           │    │
│   │ - Yes → Done    │  │ - Answer received                       │    │
│   │ - No → Ask user │  │ - Integrated into understanding         │    │
│   └─────────────────┘  │ - Stored in memory (Q&A pair)           │    │
│                        └─────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│   4. RESOLUTION                                                         │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │ Answer integrated:                                            │     │
│   │ - Model updated (reduce future prediction error)              │     │
│   │ - Memory stored (remember this Q&A)                           │     │
│   │ - Soma updated (satisfaction, reduced uncertainty)            │     │
│   │ - Question archived                                           │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Training Questions

### Supervised: When to Ask

Train on examples where humans asked questions:

```python
# Dataset of (context, surprise, human_asked_question) triples
question_decision_loss = F.binary_cross_entropy(
    model.should_ask_prediction,
    human_asked_binary
)

question_content_loss = F.cross_entropy(
    model.question_type_prediction,
    human_question_type
)
```

### Reinforcement: Quality of Questions

Learn from outcomes:

```python
# Reward: did asking this question improve subsequent performance?
reward = (
    performance_after_answer - performance_before_asking
    - cost_of_interruption
)

# Policy gradient on question-asking decision
question_policy_loss = -reward * log_prob_of_asking
```

### Self-Supervised: Question-Answer Consistency

Questions and answers should be consistent:

```python
# Given an answer, can we reconstruct what question was asked?
answer_to_question_loss = reconstruction_loss(
    derived_question_from_answer,
    original_question
)
```

## Open Questions

1. **Question granularity**: One detailed question or several simple ones?

2. **Question timing**: Interrupt immediately or wait for natural pause?

3. **Question phrasing**: How much context to include? How technical?

4. **Failed resolution**: What if reasoning fails AND user doesn't respond?

5. **Question memory**: How long to remember Q&A pairs? Forever?

6. **Meta-questions**: Can the model ask questions about its own questions? ("Am I asking too much?")

7. **Question suppression**: When should the model NOT ask, even with high uncertainty? (Social cost, redundancy, user preference)

8. **Collaborative questioning**: Can multiple questions be combined? "I'm confused about A, B, and C..."

## Implementation Phases

### Phase 1: Certainty Infrastructure (NEW - CTM-inspired)
- [ ] Implement CertaintyHead with calibration loss
- [ ] Implement UncertaintySourceIdentifier
- [ ] Basic certainty tracking in forward pass
- [ ] Certainty visualization / logging

### Phase 2: Thinking Loop
- [ ] Implement ThinkStep (single refinement step)
- [ ] Implement ThinkingLoop with halting condition
- [ ] Memory attention in thinking steps
- [ ] Soma modulation of thinking
- [ ] Adaptive tick count based on certainty

### Phase 3: Question Generation (Updated)
- [ ] Implement QuestionFromUncertainty (emerges from thinking failure)
- [ ] Implement Question dataclass with certainty fields
- [ ] Hook into thinking loop output
- [ ] Basic question buffer

### Phase 4: Certainty-Driven Loss
- [ ] Implement CertaintyDrivenLoss (CTM-style dual tick selection)
- [ ] Calibration loss component
- [ ] Efficiency loss (prefer fewer ticks)
- [ ] Integration with existing training loop

### Phase 5: Resolution Strategies
- [ ] Self-resolution through reasoning (more thinking)
- [ ] User-asking through action module (when thinking fails)
- [ ] Hypothesis formation for deferral
- [ ] Strategy selection based on certainty trajectory

### Phase 6: Soma Integration
- [ ] Certainty → soma (high certainty = positive valence)
- [ ] Question state → soma effects
- [ ] Soma → certainty threshold modulation
- [ ] Satisfaction signal on resolution

### Phase 7: Multi-Modal
- [ ] Visual prediction failures → visual questions
- [ ] Cross-modal questions (text about images, etc.)
- [ ] Question grounding in visual regions
- [ ] Certainty across modalities

### Phase 8: Training & Evaluation
- [ ] Supervised question timing
- [ ] RL question quality
- [ ] Self-supervised Q&A consistency
- [ ] Certainty calibration evaluation
- [ ] Adaptive computation efficiency metrics
