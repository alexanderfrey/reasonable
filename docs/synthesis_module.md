# Synthesis Module: Proposal

## Overview

A synthesis module that enables **generative creation from internal state** — the model producing content that didn't exist in input or memory, driven by soma.

This fills the gap between:
- **Reactive processing**: Input → process → update soma → crystallize (passive)
- **Retrieval**: Fetch existing memories → influence processing (recall)
- **Synthesis**: Generate something new conditioned on internal state (creative)

## Architecture Status

### Core Components
- [ ] **SynthesisModule** — Generate content from soma
- [ ] **InnerMonologue** — Pre-forward "thinking" that conditions processing
- [ ] **GenerativeConsolidation** — Synthesize abstractions from episodes
- [ ] **CounterfactualSynthesis** — Generate alternative scenarios
- [ ] **DreamSynthesis** — Offline creative generation

### Integration
- [ ] Hook synthesis into forward pass (inner monologue)
- [ ] Hook synthesis into consolidation (generative abstraction)
- [ ] Hook synthesis into action module (intention generation)
- [ ] Synthesis evaluation (is the generated content helpful?)

---

## The Synthesis Gap

Currently we have:
- **Input** → process → update soma → crystallize memory (reactive)
- **Retrieval** → pull existing memories → influence processing (recall)

What's missing: **generation from within** — the model creating something that didn't exist in input or memory, driven by internal state.

## Where Could Synthesis Happen?

### 1. Inner Monologue (Pre-Forward Synthesis)

Before processing the next chunk, synthesize "thoughts" that condition processing:

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│   SOMA ──▶ SYNTHESIZE("What am I thinking?") ──▶ inner_text │
│                          │                                   │
│                          ▼                                   │
│              Embed inner_text → prepend to hidden states     │
│                          │                                   │
│                          ▼                                   │
│              Process input with inner context                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

The model generates a few tokens of "inner speech" based on soma, then uses that as additional context. Like thinking before speaking.

**When**: High self-surprise, uncertainty, or at regular intervals
**What**: Brief reflections, questions to self, hypotheses

### 2. Memory Synthesis (Consolidation-Time)

During episodic → semantic consolidation, don't just average — *synthesize* new knowledge:

```
episodes = [e1, e2, e3, ...]  # Similar experiences
                │
                ▼
    SYNTHESIZE("What pattern connects these?")
                │
                ▼
    new_concept = generated abstraction (not weighted average)
```

This is generative consolidation — the model writes a "summary" that becomes semantic memory.

### 3. Counterfactual Synthesis (Reflection-Time)

Synthesize alternative outcomes:

```
actual_experience + soma → SYNTHESIZE("What if I had...")
                                    │
                                    ▼
                          counterfactual_experience
                                    │
                                    ▼
                          Compare: learn from difference
```

"What if I had been more skeptical here?" Generate that alternative, learn from it.

### 4. Goal/Intention Synthesis (Action-Time)

Before taking an action, synthesize the intention:

```
soma + context → SYNTHESIZE("I want to...")
                        │
                        ▼
                   intention_text
                        │
                        ▼
              Ground action in explicit intention
```

This makes actions interpretable and deliberate.

### 5. Dream Synthesis (Offline)

During "downtime," synthesize novel combinations:

```
sample random memories + noise → SYNTHESIZE(free generation)
                                        │
                                        ▼
                                novel_content
                                        │
                                        ▼
                        Evaluate: is this coherent? valuable?
                                        │
                                        ▼
                        Store if good → creative insight
```

This is imagination — generating content that wasn't experienced.

## A Unified Synthesis Module

```python
class SynthesisModule(nn.Module):
    """
    Generates content from internal state.

    Unlike prediction (continue input) or retrieval (fetch memory),
    synthesis creates something new conditioned on soma.

    Modes:
    - 'reflect': generate inner monologue
    - 'abstract': generate summary from multiple inputs
    - 'counterfactual': generate alternative scenarios
    - 'intend': generate intentions/goals
    - 'dream': free generation with noise
    """

    def __init__(self, config):
        self.d_soma = config.d_soma
        self.d_model = config.d_model

        # Soma → generation seed
        # Different projections for different synthesis modes
        self.mode_projections = nn.ModuleDict({
            'reflect': nn.Linear(d_soma, d_model),
            'abstract': nn.Linear(d_soma + d_model, d_model),  # soma + memory aggregate
            'counterfactual': nn.Linear(d_soma + d_model, d_model),  # soma + experience
            'intend': nn.Linear(d_soma, d_model),
            'dream': nn.Linear(d_soma, d_model),
        })

        # Control how much synthesis vs. input-driven
        self.synthesis_gate = nn.Sequential(
            nn.Linear(d_soma, 1),
            nn.Sigmoid()
        )

        # The actual generator (could share weights with main LM)
        self.generator = None  # Pointer to GPT's generation capability

    def synthesize(
        self,
        mode: str,
        soma: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        max_tokens: int = 32,
        temperature: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        """
        Synthesize content based on internal state.

        Returns:
            - tokens: generated token IDs
            - hidden: hidden states of generated content
            - gate: how much synthesis influenced this
        """
        # Get mode-specific seed
        if context is not None:
            seed_input = torch.cat([soma, context], dim=-1)
        else:
            seed_input = soma

        seed = self.mode_projections[mode](seed_input)

        # How much to synthesize?
        gate = self.synthesis_gate(soma)

        # Generate
        if mode == 'dream':
            # Add noise for creativity
            seed = seed + temperature * torch.randn_like(seed)

        tokens, hidden = self.generator.generate_from_seed(
            seed,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return {
            'tokens': tokens,
            'hidden': hidden,
            'gate': gate,
            'mode': mode,
        }
```

## Integration Point: The Inner Loop

```
┌────────────────────────────────────────────────────────────────────┐
│                         PROCESSING LOOP                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐     ┌───────────────┐     ┌──────────────────────┐  │
│  │  INPUT   │────▶│   SYNTHESIZE  │────▶│     FORWARD PASS     │  │
│  │  (text)  │     │ (inner voice) │     │  (input + synthesis) │  │
│  └──────────┘     └───────────────┘     └──────────┬───────────┘  │
│                          ▲                         │               │
│                          │                         ▼               │
│                   ┌──────┴───────┐        ┌───────────────┐       │
│                   │     SOMA     │◀───────│  EXPERIENTIAL │       │
│                   │              │        │    STREAM     │       │
│                   └──────────────┘        └───────────────┘       │
│                          │                         │               │
│                          ▼                         ▼               │
│                   ┌──────────────┐        ┌───────────────┐       │
│                   │  SYNTHESIZE  │        │  CRYSTALLIZE  │       │
│                   │ (reflection) │        │   (memory)    │       │
│                   └──────────────┘        └───────────────┘       │
│                          │                                         │
│                          ▼                                         │
│                   ┌──────────────┐                                 │
│                   │    ACTION    │                                 │
│                   │   (maybe)    │                                 │
│                   └──────────────┘                                 │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Synthesis Modes: Detailed Design

### Mode 1: Reflect (Inner Monologue)

**Purpose**: Generate brief thoughts that condition subsequent processing.

**Trigger**:
- High self-surprise ("I didn't expect that")
- High uncertainty ("I'm not sure about this")
- Regular intervals (periodic reflection)

**Output**: Short token sequence (8-32 tokens) representing current thought

**Usage**:
```python
def forward_with_synthesis(self, input_ids, ...):
    soma = self.self_state.get_soma()

    if self.should_synthesize(soma):
        inner = self.synthesize(mode='reflect', soma=soma, max_tokens=16)
        # Prepend inner thought to processing
        # Model "sees" its own thought before input
```

**Examples**:
- "This seems contradictory to what I read earlier..."
- "I should pay attention to the numbers here..."
- "This reminds me of a pattern I've seen..."

### Mode 2: Abstract (Generative Consolidation)

**Purpose**: Synthesize semantic knowledge from episodic memories.

**Trigger**: Multiple similar episodes accumulated

**Output**: A generated summary/abstraction that captures the pattern

**Usage**:
```python
def consolidate_with_synthesis(self, episodes: List[Episode]):
    # Aggregate episode embeddings
    episode_aggregate = self.aggregate(episodes)

    # Synthesize abstraction
    abstraction = self.synthesize(
        mode='abstract',
        soma=self.soma,
        context=episode_aggregate,
        max_tokens=64
    )

    # Store as semantic memory
    self.semantic.store(abstraction)
```

**Example**:
- Episodes: [user asked about X, user asked about Y, user asked about Z]
- Synthesis: "Users often ask about [pattern]. The key insight is [abstraction]."

### Mode 3: Counterfactual

**Purpose**: Generate alternative scenarios for learning.

**Trigger**: High regret signal, learning opportunity detected

**Output**: Alternative version of an experience

**Usage**:
```python
def learn_from_counterfactual(self, experience):
    # Generate "what if" alternative
    counterfactual = self.synthesize(
        mode='counterfactual',
        soma=self.soma,
        context=experience.embedding,
        max_tokens=48
    )

    # Compare outcomes
    actual_value = self.evaluate(experience)
    counter_value = self.evaluate(counterfactual)

    # Learn from difference
    self.update_policy(actual_value - counter_value)
```

### Mode 4: Intend (Intention Synthesis)

**Purpose**: Generate explicit intentions before taking actions.

**Trigger**: Action module decides to act

**Output**: Statement of intention grounding the action

**Usage**:
```python
def act_with_intention(self, action):
    # Synthesize intention
    intention = self.synthesize(
        mode='intend',
        soma=self.soma,
        max_tokens=24
    )

    # Execute action grounded in intention
    result = self.execute(action, intention=intention)

    # Log for interpretability
    self.log_action(action, intention, result)
```

**Example**:
- Action: contact_user
- Intention: "I want to clarify the ambiguity in the requirements because I'm uncertain about the expected behavior."

### Mode 5: Dream (Creative Synthesis)

**Purpose**: Generate novel content during idle time.

**Trigger**: No active processing, scheduled "dreaming" period

**Output**: Novel combinations, creative insights

**Usage**:
```python
def dream_cycle(self):
    # Sample random memories
    memories = self.memory.sample_random(k=5)
    memory_aggregate = self.aggregate(memories)

    # Add noise for creativity
    noise = torch.randn_like(self.soma) * self.dream_temperature

    # Synthesize freely
    dream = self.synthesize(
        mode='dream',
        soma=self.soma + noise,
        context=memory_aggregate,
        max_tokens=128,
        temperature=1.0  # Higher for creativity
    )

    # Evaluate: is this coherent and valuable?
    if self.evaluate_dream(dream) > threshold:
        self.store_insight(dream)
```

## Interesting Tensions

### 1. Synthesis vs. Hallucination

Synthesis is generative — but so is hallucination. The difference:
- Synthesis is *grounded* in soma (traceable reason for generating)
- Synthesis is *marked* as synthetic (not confused with input)
- Synthesis is *evaluated* (does it help or hurt?)

```python
class SynthesisOutput:
    content: torch.Tensor
    grounding: Dict  # What triggered this synthesis
    confidence: float  # How confident in this synthesis
    is_synthetic: bool = True  # Always marked
```

### 2. Efficiency vs. Richness

Synthesis takes compute. When is it worth it?

| Strategy | When | Cost |
|----------|------|------|
| Always | Every forward pass | High |
| Threshold | When soma signals need | Medium |
| Scheduled | Every N steps | Predictable |
| On-demand | When requested | Low |

Recommendation: Threshold-based with short syntheses (8-16 tokens).

### 3. Private vs. Public

Synthesized content could be:

| Type | Description | Use |
|------|-------------|-----|
| **Private** | Only feeds back into processing | Inner monologue |
| **Public** | Becomes output to user | Expressing thoughts |
| **Stored** | Becomes memory | Insights, abstractions |

### 4. Control

Who controls synthesis?

| Controller | Mechanism |
|------------|-----------|
| Automatic | Soma thresholds trigger synthesis |
| Requested | User asks "what are you thinking?" |
| Scheduled | Regular intervals (like breathing) |
| Suppressed | User/system disables synthesis |

## Implementation Sketch

```python
class SynthesisModule(nn.Module):
    def __init__(self, config, gpt_model):
        super().__init__()
        self.config = config
        self.gpt = gpt_model  # Share generation capability

        # Mode-specific seed projections
        self.reflect_proj = nn.Linear(config.d_soma, config.d_model)
        self.abstract_proj = nn.Linear(config.d_soma + config.d_model, config.d_model)
        self.counterfactual_proj = nn.Linear(config.d_soma + config.d_model, config.d_model)
        self.intend_proj = nn.Linear(config.d_soma, config.d_model)
        self.dream_proj = nn.Linear(config.d_soma, config.d_model)

        # Should we synthesize?
        self.synthesis_gate = nn.Sequential(
            nn.Linear(config.d_soma, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid()
        )

        # Synthesis quality evaluator
        self.quality_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )

    def should_synthesize(self, soma: torch.Tensor) -> torch.Tensor:
        """Decide whether to synthesize based on soma."""
        return self.synthesis_gate(soma)

    def get_seed(self, mode: str, soma: torch.Tensor,
                 context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get generation seed for mode."""
        if mode == 'reflect':
            return self.reflect_proj(soma)
        elif mode == 'abstract':
            return self.abstract_proj(torch.cat([soma, context], dim=-1))
        elif mode == 'counterfactual':
            return self.counterfactual_proj(torch.cat([soma, context], dim=-1))
        elif mode == 'intend':
            return self.intend_proj(soma)
        elif mode == 'dream':
            noise = torch.randn_like(soma) * self.config.dream_noise
            return self.dream_proj(soma + noise)

    def synthesize(
        self,
        mode: str,
        soma: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        max_tokens: int = 32,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate synthetic content."""

        # Get seed
        seed = self.get_seed(mode, soma, context)

        # Generate tokens using GPT
        # seed becomes the initial hidden state
        tokens, hidden_states = self._generate_from_seed(
            seed, max_tokens, temperature
        )

        # Evaluate quality
        quality = self.quality_head(hidden_states[:, -1, :])

        return {
            'tokens': tokens,
            'hidden': hidden_states,
            'quality': quality,
            'mode': mode,
            'grounding': {
                'soma': soma.detach(),
                'context': context.detach() if context is not None else None,
            }
        }

    def _generate_from_seed(
        self,
        seed: torch.Tensor,  # [B, d_model]
        max_tokens: int,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate tokens autoregressively from seed hidden state."""

        batch_size = seed.size(0)
        device = seed.device

        # Start with seed as hidden state
        hidden = seed.unsqueeze(1)  # [B, 1, d_model]

        tokens = []
        all_hidden = [hidden]

        for _ in range(max_tokens):
            # Project to logits
            logits = self.gpt.lm_head(self.gpt.final_norm(hidden[:, -1:, :]))

            # Sample
            if temperature > 0:
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            tokens.append(next_token)

            # Get next hidden (simplified - real version would use full transformer)
            next_embed = self.gpt.token_embedding(next_token)
            hidden = torch.cat([hidden, next_embed], dim=1)
            all_hidden.append(next_embed)

        return torch.cat(tokens, dim=1), torch.cat(all_hidden, dim=1)
```

## Integration with Forward Pass

```python
class MemoryAugmentedGPT:
    def forward(self, input_ids, ...):

        # Get current soma
        soma = self.self_state.get_soma() if self.self_state else None

        # Should we synthesize inner monologue?
        synthesis_output = None
        if self.synthesis is not None and soma is not None:
            synth_prob = self.synthesis.should_synthesize(soma)

            if synth_prob > self.config.synthesis_threshold:
                synthesis_output = self.synthesis.synthesize(
                    mode='reflect',
                    soma=soma,
                    max_tokens=self.config.inner_monologue_tokens,
                    temperature=0.7
                )

                # Prepend synthetic hidden states to input processing
                # This means the model "thinks" before seeing input
                synthetic_hidden = synthesis_output['hidden']
                # ... integrate with forward pass ...

        # Continue with normal forward pass
        logits, hidden_states = self.gpt(input_ids, ...)

        # ... rest of processing ...

        # Add synthesis info to output
        memory_output['synthesis'] = synthesis_output
```

## Open Questions

1. **Generation Quality**: How do we ensure synthesized content is coherent and useful?

2. **Training Signal**: How do we train the synthesis module?
   - Supervised: Examples of good inner monologue?
   - RL: Reward based on downstream task performance?
   - Self-supervised: Consistency with soma state?

3. **Computational Cost**: Synthesis adds generation steps. How to budget?

4. **Interference**: Does synthetic content interfere with processing real input?

5. **Recursion**: Can synthesis trigger more synthesis? (Thinking about thinking...)

## Implementation Phases

### Phase 1: Infrastructure
- [ ] Basic SynthesisModule with reflect mode
- [ ] Integration hook in forward pass
- [ ] Synthesis gate training

### Phase 2: Inner Monologue
- [ ] Full reflect mode implementation
- [ ] Prepend synthetic hidden to processing
- [ ] Quality evaluation and filtering

### Phase 3: Generative Consolidation
- [ ] Abstract mode for memory consolidation
- [ ] Replace averaging with synthesis in semantic memory
- [ ] Evaluate abstraction quality

### Phase 4: Advanced Modes
- [ ] Counterfactual synthesis for learning
- [ ] Intention synthesis for actions
- [ ] Dream synthesis for creativity

## Relationship to Other Modules

| Module | Synthesis Role |
|--------|---------------|
| **Soma** | Provides grounding state for synthesis |
| **Memory** | Receives synthesized abstractions |
| **Action** | Uses synthesized intentions |
| **ExperientialStream** | Triggers synthesis based on surprise |

The synthesis module is the **generative counterpart** to the experiential stream — where experiential processes input into state, synthesis generates output from state.
