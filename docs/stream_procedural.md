# Procedural Stream

How to do — skills, habits, patterns of action.

**Role**: The procedural stream holds implicit knowledge about *how* to do things. Unlike semantic memory (knowing *that*), procedural memory is knowing *how* — skills that are executed rather than recalled.

---

## 1. Core Concept

Procedural knowledge is:
- **Implicit**: You can ride a bike without explaining how
- **Automatic**: Executed without conscious effort (once learned)
- **Skill-based**: Improves with practice
- **Context-triggered**: Activates in appropriate situations

For a language model, procedural knowledge might include:
- How to maintain narrative coherence
- How to shift between formal and casual registers
- How to build tension in a story
- How to structure an argument
- The "craft" of writing

```
Semantic knowledge: "Dialogue should reveal character"
Procedural knowledge: *actually writing dialogue that reveals character*
```

---

## 2. Formal Definition

### 2.1 Skill Structure

```python
@dataclass
class Skill:
    """A procedural capability — knowing how to do something."""

    # Identity
    id: int
    name: Optional[str]
    description: Optional[str]

    # Trigger (when to activate)
    trigger_pattern: Tensor        # [d_model] - context that activates this skill
    trigger_threshold: float       # similarity threshold for activation

    # Execution (how to do it)
    execution_module: nn.Module    # the learned procedure

    # Competence
    proficiency: float            # [0, 1] - skill level
    practice_count: int           # how many times practiced
    last_practice: int            # when last used

    # Learning
    acquisition_source: str        # "consolidation", "imitation", "practice"
    learning_rate: float           # how fast this skill improves

    # Relations
    prerequisites: List[int]       # skills needed before this one
    components: List[int]          # sub-skills this is composed of


@dataclass
class SkillExecution:
    """Record of a skill being executed."""
    skill_id: int
    context: Tensor               # input context
    output_modulation: Tensor     # how skill modified processing
    success_signal: Optional[float]  # feedback if available
    timestamp: int
```

### 2.2 Skill Library

```python
@dataclass
class SkillLibrary:
    """The procedural memory structure."""

    skills: Dict[int, Skill]       # id → skill

    # Index for fast trigger matching
    trigger_index: Tensor          # [n_skills, d_model]

    # Hierarchy
    skill_tree: Dict[int, List[int]]  # parent → children

    # Active skills
    currently_active: List[int]    # skills engaged right now

    # Statistics
    total_skills: int
    total_executions: int
```

---

## 3. Architecture

### 3.1 Main Module

```python
class ProceduralStream(nn.Module):
    """
    Procedural memory: skills and how to execute them.

    Operations:
    - Match: find skills relevant to current context
    - Execute: apply skill to modify processing
    - Practice: improve skill through feedback
    - Acquire: learn new skills
    - Compose: combine skills into higher-order procedures
    """

    def __init__(self, config: ProceduralConfig):
        super().__init__()
        self.d_model = config.d_model
        self.max_active_skills = config.max_active_skills

        # Trigger matching
        self.trigger_projection = nn.Linear(config.d_model, config.d_model)

        # Skill composition
        self.skill_composer = SkillComposer(config.d_model)

        # Skill library
        self.library = SkillLibrary(
            skills={},
            trigger_index=torch.zeros(0, config.d_model),
            skill_tree={},
            currently_active=[],
            total_skills=0,
            total_executions=0
        )

        # Base skills (always available, learned during training)
        self._init_base_skills(config)

    def _init_base_skills(self, config: ProceduralConfig):
        """Initialize learnable base skills."""
        n_base_skills = config.n_base_skills

        for i in range(n_base_skills):
            skill = Skill(
                id=i,
                name=f"base_skill_{i}",
                description=None,
                trigger_pattern=nn.Parameter(
                    torch.randn(config.d_model) * 0.02
                ),
                trigger_threshold=0.5,
                execution_module=SkillModule(config.d_model),
                proficiency=0.1,
                practice_count=0,
                last_practice=0,
                acquisition_source="base",
                learning_rate=0.01,
                prerequisites=[],
                components=[]
            )
            self.library.skills[i] = skill
            self.library.total_skills += 1

        self._rebuild_trigger_index()

    def match(
        self,
        context: Tensor,
        top_k: Optional[int] = None
    ) -> List[Skill]:
        """
        Find skills relevant to current context.

        Args:
            context: [batch, d_model] or [d_model]
            top_k: max skills to return (default: max_active_skills)

        Returns:
            List of matched skills, sorted by relevance
        """
        if self.library.total_skills == 0:
            return []

        if context.dim() == 1:
            context = context.unsqueeze(0)

        top_k = top_k or self.max_active_skills

        # Project context
        query = self.trigger_projection(context)  # [batch, d_model]

        # Compute similarities to all triggers
        similarities = F.cosine_similarity(
            query.unsqueeze(1),  # [batch, 1, d_model]
            self.library.trigger_index.unsqueeze(0),  # [1, n_skills, d_model]
            dim=-1
        )  # [batch, n_skills]

        # Filter by threshold and get top-k
        matched = []
        for skill_id, skill in self.library.skills.items():
            sim = similarities[0, skill_id].item()
            if sim > skill.trigger_threshold:
                matched.append((skill, sim))

        # Sort by similarity
        matched.sort(key=lambda x: x[1], reverse=True)
        matched = matched[:top_k]

        return [skill for skill, _ in matched]

    def execute(
        self,
        skills: List[Skill],
        hidden_states: Tensor,
        context: Optional[Tensor] = None
    ) -> Tensor:
        """
        Execute skills to modulate processing.

        Args:
            skills: skills to execute
            hidden_states: [batch, seq_len, d_model] - current states
            context: optional additional context

        Returns:
            modulated: [batch, seq_len, d_model] - modified states
        """
        if not skills:
            return hidden_states

        # Execute each skill and combine modulations
        modulations = []
        for skill in skills:
            mod = skill.execution_module(hidden_states, context)
            # Weight by proficiency
            mod = mod * skill.proficiency
            modulations.append(mod)

            # Record execution
            skill.practice_count += 1
            skill.last_practice = self._current_step()

        # Combine modulations (could be sum, attention, learned)
        combined = self.skill_composer(modulations, hidden_states)

        # Apply as residual
        modulated = hidden_states + combined

        # Update library statistics
        self.library.total_executions += len(skills)
        self.library.currently_active = [s.id for s in skills]

        return modulated

    def practice(
        self,
        skill: Skill,
        execution: SkillExecution,
        feedback: float
    ):
        """
        Improve skill based on feedback.

        Positive feedback → increase proficiency
        Negative feedback → decrease proficiency or adjust
        """
        # Update proficiency with exponential moving average
        delta = feedback - 0.5  # center around neutral
        skill.proficiency = torch.clamp(
            torch.tensor(skill.proficiency + skill.learning_rate * delta),
            0.0, 1.0
        ).item()

        # If consistently negative, might need to adjust trigger
        if feedback < 0.3 and skill.practice_count > 10:
            self._adjust_trigger(skill, execution.context)

    def acquire(
        self,
        trigger_pattern: Tensor,
        execution_examples: List[Tuple[Tensor, Tensor]],  # (input, output) pairs
        name: Optional[str] = None,
        source: str = "practice"
    ) -> Skill:
        """
        Learn a new skill from examples.

        Args:
            trigger_pattern: context that should activate this skill
            execution_examples: (input, output) pairs showing the skill
            name: optional name
            source: how the skill was acquired

        Returns:
            Newly created skill
        """
        skill_id = self.library.total_skills

        # Create execution module
        module = SkillModule(self.d_model)

        # Train module on examples
        self._train_skill_module(module, execution_examples)

        skill = Skill(
            id=skill_id,
            name=name,
            description=None,
            trigger_pattern=trigger_pattern.detach().clone(),
            trigger_threshold=0.5,
            execution_module=module,
            proficiency=0.3,  # starts modest
            practice_count=0,
            last_practice=self._current_step(),
            acquisition_source=source,
            learning_rate=0.01,
            prerequisites=[],
            components=[]
        )

        self.library.skills[skill_id] = skill
        self.library.total_skills += 1
        self._rebuild_trigger_index()

        return skill

    def compose(
        self,
        component_skills: List[int],
        name: Optional[str] = None
    ) -> Skill:
        """
        Create a higher-order skill from component skills.

        The new skill chains or combines the components.
        """
        skill_id = self.library.total_skills

        components = [self.library.skills[i] for i in component_skills]

        # Composite trigger: combination of component triggers
        trigger = torch.stack([s.trigger_pattern for s in components]).mean(dim=0)

        # Composite execution: chain or blend components
        module = CompositeSkillModule(
            self.d_model,
            [s.execution_module for s in components]
        )

        skill = Skill(
            id=skill_id,
            name=name,
            description=f"Composite of {[s.name for s in components]}",
            trigger_pattern=trigger,
            trigger_threshold=0.6,  # slightly higher threshold
            execution_module=module,
            proficiency=min(s.proficiency for s in components),  # limited by weakest
            practice_count=0,
            last_practice=self._current_step(),
            acquisition_source="composition",
            learning_rate=0.005,  # slower learning for complex skills
            prerequisites=[],
            components=component_skills
        )

        self.library.skills[skill_id] = skill
        self.library.total_skills += 1

        # Update hierarchy
        for cid in component_skills:
            if cid not in self.library.skill_tree:
                self.library.skill_tree[cid] = []
            self.library.skill_tree[cid].append(skill_id)

        self._rebuild_trigger_index()

        return skill

    def automate(
        self,
        deliberate_pattern: Tensor,
        examples: List[Tuple[Tensor, Tensor]]
    ) -> Skill:
        """
        Convert deliberate processing pattern into automatic skill.

        What was once effortful becomes effortless.
        """
        return self.acquire(
            trigger_pattern=deliberate_pattern,
            execution_examples=examples,
            name=None,
            source="automation"
        )

    def _train_skill_module(
        self,
        module: nn.Module,
        examples: List[Tuple[Tensor, Tensor]],
        epochs: int = 10
    ):
        """Train a skill module on examples."""
        optimizer = torch.optim.Adam(module.parameters(), lr=0.01)

        for _ in range(epochs):
            total_loss = 0
            for input_states, target_output in examples:
                output = module(input_states)
                loss = F.mse_loss(output, target_output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def _adjust_trigger(self, skill: Skill, bad_context: Tensor):
        """Adjust trigger to avoid bad matches."""
        # Move trigger away from bad context
        skill.trigger_pattern = skill.trigger_pattern - 0.1 * bad_context
        skill.trigger_pattern = F.normalize(skill.trigger_pattern, dim=-1)
        self._rebuild_trigger_index()

    def _rebuild_trigger_index(self):
        """Rebuild trigger index for fast matching."""
        if self.library.total_skills == 0:
            self.library.trigger_index = torch.zeros(0, self.d_model)
        else:
            triggers = []
            for i in range(self.library.total_skills):
                if i in self.library.skills:
                    triggers.append(self.library.skills[i].trigger_pattern)
            self.library.trigger_index = torch.stack(triggers)

    def _current_step(self) -> int:
        """Get current step (placeholder)."""
        return 0
```

### 3.2 Skill Module

```python
class SkillModule(nn.Module):
    """
    A learnable procedure that modulates processing.

    Takes hidden states, returns modulation to apply.
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or d_model

        # Lightweight MLP for modulation
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

        # Gating (how much to apply)
        self.gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states: Tensor,
        context: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute modulation for hidden states.

        Args:
            hidden_states: [batch, seq_len, d_model]
            context: optional [batch, d_model] context

        Returns:
            modulation: [batch, seq_len, d_model]
        """
        # Compute raw modulation
        modulation = self.net(hidden_states)

        # Apply gating
        gate = self.gate(hidden_states)  # [batch, seq_len, 1]
        modulation = modulation * gate

        return modulation
```

### 3.3 Composite Skill Module

```python
class CompositeSkillModule(nn.Module):
    """
    A skill composed of multiple sub-skills.
    """

    def __init__(
        self,
        d_model: int,
        component_modules: List[nn.Module]
    ):
        super().__init__()
        self.components = nn.ModuleList(component_modules)

        # Learn how to combine components
        self.combiner = nn.Sequential(
            nn.Linear(d_model * len(component_modules), d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(
        self,
        hidden_states: Tensor,
        context: Optional[Tensor] = None
    ) -> Tensor:
        """Execute all components and combine."""
        # Get modulation from each component
        modulations = [
            comp(hidden_states, context)
            for comp in self.components
        ]

        # Combine
        combined = torch.cat(modulations, dim=-1)
        return self.combiner(combined)
```

### 3.4 Skill Composer

```python
class SkillComposer(nn.Module):
    """
    Compose multiple skill modulations into one.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )

    def forward(
        self,
        modulations: List[Tensor],
        hidden_states: Tensor
    ) -> Tensor:
        """
        Combine multiple skill modulations.

        Args:
            modulations: list of [batch, seq_len, d_model]
            hidden_states: [batch, seq_len, d_model] - original states

        Returns:
            combined: [batch, seq_len, d_model]
        """
        if len(modulations) == 0:
            return torch.zeros_like(hidden_states)

        if len(modulations) == 1:
            return modulations[0]

        # Stack modulations as "skill tokens"
        stacked = torch.stack(modulations, dim=2)  # [batch, seq_len, n_skills, d_model]
        batch, seq_len, n_skills, d = stacked.shape

        # Flatten for attention
        stacked = stacked.view(batch * seq_len, n_skills, d)
        hidden_flat = hidden_states.view(batch * seq_len, 1, d)

        # Attend over skills
        combined, _ = self.attention(hidden_flat, stacked, stacked)
        combined = combined.view(batch, seq_len, d)

        return combined
```

---

## 4. Training

### 4.1 Skill Execution Loss

Train skills to produce helpful modulations:

```python
def skill_execution_loss(
    modulated_states: Tensor,
    original_states: Tensor,
    target_output: Tensor,
    model: nn.Module
) -> Tensor:
    """
    Skills should improve task performance.

    Compare: performance with skill vs without skill.
    """
    # Output with skill
    output_with = model.lm_head(model.final_norm(modulated_states))
    loss_with = F.cross_entropy(
        output_with.view(-1, output_with.size(-1)),
        target_output.view(-1)
    )

    # Output without skill
    output_without = model.lm_head(model.final_norm(original_states))
    loss_without = F.cross_entropy(
        output_without.view(-1, output_without.size(-1)),
        target_output.view(-1)
    )

    # Skill should help
    advantage = loss_without - loss_with
    return -advantage  # minimize negative advantage = maximize advantage
```

### 4.2 Trigger Alignment Loss

Train triggers to match appropriate contexts:

```python
def trigger_alignment_loss(
    skill: Skill,
    positive_contexts: List[Tensor],  # should trigger
    negative_contexts: List[Tensor],  # shouldn't trigger
    temperature: float = 0.1
) -> Tensor:
    """
    Contrastive loss for trigger patterns.
    """
    trigger = skill.trigger_pattern

    pos_sims = torch.stack([
        F.cosine_similarity(trigger.unsqueeze(0), ctx.unsqueeze(0))
        for ctx in positive_contexts
    ])

    neg_sims = torch.stack([
        F.cosine_similarity(trigger.unsqueeze(0), ctx.unsqueeze(0))
        for ctx in negative_contexts
    ])

    # Positives should be similar, negatives dissimilar
    pos_loss = -pos_sims.mean()
    neg_loss = F.relu(neg_sims - skill.trigger_threshold).mean()

    return pos_loss + neg_loss
```

### 4.3 Proficiency from Practice

```python
def update_proficiency(
    skill: Skill,
    success_signals: List[float]
) -> None:
    """
    Update skill proficiency based on practice outcomes.

    Uses exponential moving average with learning rate.
    """
    for signal in success_signals:
        delta = (signal - 0.5) * skill.learning_rate
        skill.proficiency = max(0.0, min(1.0, skill.proficiency + delta))
        skill.practice_count += 1
```

---

## 5. Skill Acquisition

### 5.1 From Consolidation

```python
def skill_from_consolidation(
    procedural: ProceduralStream,
    semantic: SemanticStream,
    concept: Concept,
    action_examples: List[Tuple[Tensor, Tensor]]
) -> Skill:
    """
    Convert semantic knowledge into procedural skill.

    "Knowing that X" → "Knowing how to do X"
    """
    # Use concept embedding as trigger
    trigger = concept.embedding

    # Learn execution from examples
    skill = procedural.acquire(
        trigger_pattern=trigger,
        execution_examples=action_examples,
        name=f"skill_from_{concept.name or concept.id}",
        source="consolidation"
    )

    return skill
```

### 5.2 From Imitation

```python
def skill_from_imitation(
    procedural: ProceduralStream,
    expert_behavior: List[Tuple[Tensor, Tensor, Tensor]]  # (context, input, output)
) -> Skill:
    """
    Learn skill by imitating expert behavior.
    """
    # Extract trigger pattern from contexts
    contexts = torch.stack([ctx for ctx, _, _ in expert_behavior])
    trigger = contexts.mean(dim=0)

    # Extract (input, output) pairs
    examples = [(inp, out) for _, inp, out in expert_behavior]

    return procedural.acquire(
        trigger_pattern=trigger,
        execution_examples=examples,
        name=None,
        source="imitation"
    )
```

### 5.3 From Exploration

```python
def skill_from_exploration(
    procedural: ProceduralStream,
    successful_episodes: List[SkillExecution]
) -> Optional[Skill]:
    """
    Discover new skill from successful explorations.

    If random actions repeatedly succeed in similar contexts,
    consolidate into a skill.
    """
    if len(successful_episodes) < 3:
        return None

    # Check if contexts are similar
    contexts = torch.stack([ep.context for ep in successful_episodes])
    similarities = F.cosine_similarity(
        contexts.unsqueeze(1),
        contexts.unsqueeze(0),
        dim=-1
    )

    if similarities.mean() < 0.7:
        return None  # too diverse

    # Extract skill
    trigger = contexts.mean(dim=0)
    examples = [(ep.context, ep.output_modulation) for ep in successful_episodes]

    return procedural.acquire(
        trigger_pattern=trigger,
        execution_examples=examples,
        name=None,
        source="exploration"
    )
```

---

## 6. Interface with Other Streams

### 6.1 ← Experiential Stream (Skill Activation)

```python
def receive_from_experiential(
    procedural: ProceduralStream,
    experience: ExperientialState
) -> List[Skill]:
    """
    Match skills to current experience.
    """
    return procedural.match(experience.content)
```

### 6.2 → Experiential Stream (Skill Application)

```python
def modulate_experience(
    procedural: ProceduralStream,
    experience: ExperientialState,
    hidden_states: Tensor
) -> Tensor:
    """
    Apply active skills to modulate processing.
    """
    active_skills = procedural.match(experience.content)
    return procedural.execute(active_skills, hidden_states)
```

### 6.3 ← Semantic Stream (Skill Grounding)

```python
def ground_in_knowledge(
    procedural: ProceduralStream,
    semantic: SemanticStream,
    skill: Skill
) -> List[Concept]:
    """
    Find knowledge relevant to a skill.
    """
    return semantic.query(skill.trigger_pattern, top_k=3)
```

---

## 7. For Language Models Specifically

### 7.1 Example Skills

```python
LANGUAGE_SKILLS = {
    "narrative_coherence": {
        "description": "Maintain consistent narrative thread",
        "trigger": "story context",
        "effect": "biases toward coherent continuation"
    },
    "register_shift": {
        "description": "Shift between formal and casual",
        "trigger": "register mismatch detected",
        "effect": "adjusts style appropriately"
    },
    "tension_building": {
        "description": "Build narrative tension",
        "trigger": "conflict introduction",
        "effect": "pacing, foreshadowing"
    },
    "dialogue_craft": {
        "description": "Write natural dialogue",
        "trigger": "dialogue context",
        "effect": "character voice, subtext"
    },
    "exposition_weaving": {
        "description": "Weave in background information",
        "trigger": "need for context",
        "effect": "natural information delivery"
    }
}
```

### 7.2 Skill Discovery from Writing

```python
def discover_writing_skills(
    procedural: ProceduralStream,
    expert_texts: List[str],
    model: nn.Module
) -> List[Skill]:
    """
    Discover skills from expert writing.

    Analyze what expert writers do differently.
    """
    discovered = []

    # Process expert texts, looking for:
    # - State transitions that improve quality
    # - Patterns that appear repeatedly
    # - Contexts where specific modulations help

    return discovered
```

---

## 8. Open Questions

1. **Skill granularity**: Fine-grained (word-level) vs coarse (document-level)?

2. **Skill interference**: What if multiple skills conflict?

3. **Skill transfer**: Can skills transfer across domains?

4. **Conscious vs automatic**: How to model the transition from deliberate to automatic?

5. **Skill unlearning**: Can bad habits be unlearned?

6. **Skill visualization**: How to interpret what a skill does?

---

## 9. Implementation Checklist

- [ ] Define ProceduralConfig
- [ ] Implement Skill dataclass
- [ ] Implement SkillModule
- [ ] Implement CompositeSkillModule
- [ ] Implement SkillComposer
- [ ] Implement ProceduralStream.match
- [ ] Implement ProceduralStream.execute
- [ ] Implement ProceduralStream.acquire
- [ ] Implement ProceduralStream.practice
- [ ] Add skill execution loss
- [ ] Add trigger alignment loss
- [ ] Test skill learning on simple tasks
- [ ] Integrate with transformer forward pass

---

*Related documents*:
- `memory_streams_architecture.md` — overall design
- `stream_experiential.md` — experiential stream (activation source)
- `stream_semantic.md` — semantic stream (knowledge grounding)
