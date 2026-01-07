# Questioning System: Prediction Failure → Information Seeking

## Core Thesis

**Questions are the cognitive response to prediction failure.**

When the model's predictions about the world don't match reality, it experiences surprise. This prediction error can be resolved in several ways:
1. **Update the model** — Learn from the error (passive)
2. **Ask a question** — Seek information to resolve the error (active)
3. **Form a hypothesis** — Speculate and continue with uncertainty
4. **Ignore** — Accept the error as noise (low importance)

Questioning is not merely a user-interface feature — it's a **fundamental cognitive mechanism** for actively resolving prediction failures rather than passively accepting them.

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

### Phase 1: Question Generation
- [ ] Implement Question dataclass
- [ ] Implement QuestionGenerator from prediction error
- [ ] Hook into experiential surprise computation
- [ ] Basic question buffer

### Phase 2: Resolution Strategies
- [ ] Self-resolution through reasoning
- [ ] User-asking through action module
- [ ] Hypothesis formation for deferral
- [ ] Strategy selection network

### Phase 3: Soma Integration
- [ ] Question state → soma effects
- [ ] Soma → question threshold modulation
- [ ] Satisfaction signal on resolution

### Phase 4: Multi-Modal
- [ ] Visual prediction failures → visual questions
- [ ] Cross-modal questions (text about images, etc.)
- [ ] Question grounding in visual regions

### Phase 5: Training
- [ ] Supervised question timing
- [ ] RL question quality
- [ ] Self-supervised Q&A consistency
