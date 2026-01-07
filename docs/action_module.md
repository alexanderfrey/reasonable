# Action Module: Proposal

## Overview

An action module that gives the model **agency** — the ability to take discrete actions based on its internal state (soma) and context. Actions include contacting the user, adjusting internal parameters, and performing memory operations.

This extends the experiential architecture from passive (react to text) to active (choose to intervene).

## Architecture Status

### Core Components
- [ ] **ActionSpace** — Define available actions and their semantics
- [ ] **ActionPolicy** — Select actions based on soma + context
- [ ] **ActionExecutor** — Execute selected actions
- [ ] **UserContactInterface** — Generate messages to user
- [ ] **ActionValue** — Estimate value/appropriateness of actions

### Integration
- [ ] Hook into MemoryAugmentedGPT forward pass
- [ ] Action triggers based on soma thresholds
- [ ] Interrupt mechanism for user contact
- [ ] Action history tracking

---

## Motivation

The current system is **reactive**: text comes in, soma updates, memories crystallize. The model has no way to:

1. **Ask for help** when confused or uncertain
2. **Express disagreement** when something conflicts with its "values" (ideal self)
3. **Request clarification** when input is ambiguous
4. **Pause and reflect** before continuing

Humans do all of these. An action module enables the model to be an **agent** rather than just a processor.

## Action Space

### Tier 1: User-Facing Actions

| Action | Trigger Signal | Effect |
|--------|---------------|--------|
| `contact_user` | High uncertainty, high self-surprise | Interrupt flow, generate message to user |
| `request_clarification` | Ambiguity detected, low confidence | Ask specific question about input |
| `express_state` | Soma threshold crossed | Share internal state ("I'm finding this confusing") |
| `disagree` | Conflict with ideal self | Voice disagreement with content |

### Tier 2: Internal Actions

| Action | Trigger Signal | Effect |
|--------|---------------|--------|
| `adjust_attention` | Soma-driven | Modify attention bias for next forward |
| `increase_retrieval` | High novelty, low certainty | Pull more memories for context |
| `pause_and_reflect` | High meta-surprise | Re-process before continuing |
| `consolidate_now` | Pattern detected | Trigger episodic → semantic consolidation |

### Tier 3: Memory Actions

| Action | Trigger Signal | Effect |
|--------|---------------|--------|
| `force_crystallize` | Very high salience | Store memory regardless of threshold |
| `forget` | Repeated harm from memory | Remove unhelpful memory |
| `mark_important` | Ideal self alignment | Boost memory salience |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ACTION MODULE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐  │
│  │    SOMA     │────▶│ ACTION POLICY │────▶│ ACTION EXECUTOR │  │
│  │  + context  │     │  π(a|s,c)     │     │                 │  │
│  └─────────────┘     └──────────────┘     └────────┬────────┘  │
│                              │                      │           │
│                              ▼                      ▼           │
│                      ┌──────────────┐      ┌───────────────┐   │
│                      │ ACTION VALUE │      │ USER CONTACT  │   │
│                      │   V(s,a)     │      │  INTERFACE    │   │
│                      └──────────────┘      └───────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. ActionSpace

```python
@dataclass
class Action:
    name: str
    tier: int  # 1=user-facing, 2=internal, 3=memory
    requires_generation: bool  # Does this action produce text?
    reversible: bool

class ActionSpace:
    actions: List[Action]

    def available_actions(self, context: Dict) -> List[Action]:
        """Return actions available in current context."""
        # E.g., can't contact user during generation
        pass
```

### 2. ActionPolicy

Selects which action (if any) to take based on soma and context.

```python
class ActionPolicy(nn.Module):
    """
    Policy network: π(action | soma, context, hidden_state)

    Outputs:
    - action_logits: [B, n_actions] probability over action space
    - action_value: [B, n_actions] estimated value of each action
    - should_act: [B] whether to take any action (vs. continue normally)
    """

    def __init__(self, config):
        self.soma_encoder = nn.Linear(config.d_soma, config.d_policy)
        self.context_encoder = nn.Linear(config.d_model, config.d_policy)

        # Action selection head
        self.action_head = nn.Sequential(
            nn.Linear(config.d_policy * 2, config.d_policy),
            nn.GELU(),
            nn.Linear(config.d_policy, config.n_actions)
        )

        # Should we act at all? (vs. continue passive processing)
        self.act_gate = nn.Sequential(
            nn.Linear(config.d_policy * 2, 1),
            nn.Sigmoid()
        )

        # Value estimation for each action
        self.value_head = nn.Sequential(
            nn.Linear(config.d_policy * 2 + config.n_actions, config.d_policy),
            nn.GELU(),
            nn.Linear(config.d_policy, config.n_actions)
        )

    def forward(self, soma, context):
        soma_enc = self.soma_encoder(soma)
        ctx_enc = self.context_encoder(context)
        combined = torch.cat([soma_enc, ctx_enc], dim=-1)

        # Should we act?
        act_prob = self.act_gate(combined)

        # Which action?
        action_logits = self.action_head(combined)

        # What's the value of each action?
        action_values = self.value_head(
            torch.cat([combined, F.softmax(action_logits, dim=-1)], dim=-1)
        )

        return {
            'act_prob': act_prob,
            'action_logits': action_logits,
            'action_values': action_values,
        }
```

### 3. UserContactInterface

Generates messages to the user when `contact_user` action is selected.

```python
class UserContactInterface(nn.Module):
    """
    Generates user-facing messages conditioned on soma and reason for contact.

    Unlike normal generation (predict next token), this is:
    - Initiated by the model, not prompted
    - Grounded in internal state (why am I contacting?)
    - Structured (question, statement, expression)
    """

    def __init__(self, config):
        # Encode the reason for contact
        self.reason_encoder = nn.Linear(config.d_soma + config.d_model, config.d_model)

        # Message type classifier
        self.message_type = nn.Linear(config.d_model, 4)  # question, statement, expression, other

        # Seed for generation
        self.gen_seed = nn.Linear(config.d_model, config.d_model)

    def forward(self, soma, context, action_reason):
        """
        Prepare for user contact.

        Returns:
            - message_type: what kind of message to generate
            - gen_seed: seed hidden state for autoregressive generation
            - grounding: what triggered this contact (for interpretability)
        """
        reason_enc = self.reason_encoder(
            torch.cat([soma, context], dim=-1)
        )

        msg_type = self.message_type(reason_enc)
        seed = self.gen_seed(reason_enc)

        return {
            'message_type': msg_type,
            'generation_seed': seed,
            'grounding': {
                'soma_state': soma,
                'trigger': action_reason,
            }
        }
```

### 4. ActionExecutor

Executes the selected action.

```python
class ActionExecutor:
    """
    Execute actions and handle their effects.
    """

    def execute(self, action: Action, context: Dict) -> Dict:
        if action.name == 'contact_user':
            return self._contact_user(context)
        elif action.name == 'adjust_attention':
            return self._adjust_attention(context)
        elif action.name == 'force_crystallize':
            return self._force_crystallize(context)
        # ... etc

    def _contact_user(self, context):
        """
        Interrupt normal processing and generate message to user.

        Returns:
            - interrupt: True (signals caller to pause normal flow)
            - message: generated message to show user
            - expects_response: whether to wait for user reply
        """
        message = self.user_interface(
            context['soma'],
            context['hidden_state'],
            context['action_reason']
        )

        return {
            'interrupt': True,
            'message': message,
            'expects_response': message['message_type'] == 'question',
        }
```

## Trigger Conditions

When should the model consider taking an action?

### Soma-Based Triggers

```python
def should_consider_action(soma_output: Dict) -> bool:
    """Check if any action trigger is active."""

    # High self-surprise: "I didn't expect to react this way"
    if soma_output['self_surprise'] > SELF_SURPRISE_THRESHOLD:
        return True

    # Large discrepancy from ideal self
    if soma_output['ideal_discrepancy'] > IDEAL_DISCREPANCY_THRESHOLD:
        return True

    # Extreme soma values (very negative valence, very high arousal)
    soma = soma_output['soma']
    if soma.abs().max() > SOMA_EXTREME_THRESHOLD:
        return True

    # High uncertainty (low confidence)
    if soma_output['self_confidence'] < CONFIDENCE_THRESHOLD:
        return True

    return False
```

### Context-Based Triggers

```python
def context_triggers_action(exp_output: Dict) -> bool:
    """Check context-based triggers."""

    # Very high surprise in text
    if exp_output['surprise'].max() > SURPRISE_THRESHOLD:
        return True

    # Ambiguity detected (entropy of predictions)
    if exp_output.get('prediction_entropy', 0) > ENTROPY_THRESHOLD:
        return True

    return False
```

## Integration with Forward Pass

```python
class MemoryAugmentedGPT:
    def forward(self, input_ids, ...):
        # ... existing forward pass ...

        # After experiential processing and soma update:
        if self.action_module is not None:
            # Check if we should consider acting
            if should_consider_action(self_state_output):
                # Run action policy
                action_decision = self.action_module.policy(
                    soma=self_state_output['soma'],
                    context=hidden_states[:, -1, :],
                )

                # Should we act?
                if action_decision['act_prob'] > ACT_THRESHOLD:
                    # Select action
                    action_idx = action_decision['action_logits'].argmax(dim=-1)
                    action = self.action_module.action_space[action_idx]

                    # Execute action
                    action_result = self.action_module.execute(action, {
                        'soma': self_state_output['soma'],
                        'hidden_state': hidden_states[:, -1, :],
                        'action_reason': self._get_action_reason(self_state_output),
                    })

                    # Handle interrupts (e.g., contact_user)
                    if action_result.get('interrupt'):
                        memory_output['action_interrupt'] = action_result
                        return logits, hidden_states, memory_output

        # ... continue normal processing ...
```

## Training

### Supervised Learning (Phase 1)

Train on examples of when humans would:
- Ask for clarification
- Express confusion
- Request help

```python
# Training signal: predict when human annotators said "I would ask here"
action_loss = F.cross_entropy(
    action_decision['action_logits'],
    human_action_labels
)
```

### Reinforcement Learning (Phase 2)

Learn from outcomes:
- Did contacting user lead to better understanding?
- Did the adjustment help?

```python
# Reward signal: did the action improve subsequent processing?
reward = compute_action_reward(
    action_taken,
    pre_action_state,
    post_action_state,
    user_response  # if applicable
)

# Policy gradient
action_advantage = reward - action_decision['action_values'][action_idx]
policy_loss = -action_advantage * action_decision['action_logits'][action_idx]
```

### Self-Supervised (Phase 3)

Learn from consistency:
- Actions should align with stated reasons
- Repeated actions should converge to good patterns

## User Contact: Detailed Design

### Message Types

| Type | Purpose | Example |
|------|---------|---------|
| `clarification_request` | Resolve ambiguity | "Could you clarify what you mean by X?" |
| `uncertainty_expression` | Signal low confidence | "I'm not certain about this, but..." |
| `disagreement` | Conflict with values/ideal | "This seems inconsistent with..." |
| `state_expression` | Share internal experience | "I'm finding this text confusing because..." |
| `pause_request` | Need more time | "Let me think about this more carefully." |

### Generation Approach

```python
def generate_user_message(self, soma, context, message_type):
    """
    Generate a message to the user.

    The message is grounded in:
    1. Current soma state (why am I contacting?)
    2. Context (what specifically triggered this?)
    3. Message type (what kind of communication?)
    """

    # Template-based for reliability, with learned interpolation
    templates = {
        'clarification_request': "Could you clarify {SLOT}?",
        'uncertainty_expression': "I'm uncertain about {SLOT}. {ELABORATION}",
        'disagreement': "This seems {ASSESSMENT} because {REASON}.",
        'state_expression': "I'm experiencing {STATE} regarding {TOPIC}.",
    }

    # Fill slots using soma-conditioned generation
    template = templates[message_type]
    slots = self.extract_slots(template)

    for slot in slots:
        slot_content = self.generate_slot(
            slot_name=slot,
            soma=soma,
            context=context,
            max_tokens=20
        )
        template = template.replace(f"{{{slot}}}", slot_content)

    return template
```

### Safety Considerations

1. **Rate limiting**: Don't spam the user with messages
2. **Confidence threshold**: Only contact when genuinely needed
3. **Graceful degradation**: If user doesn't respond, continue processing
4. **Transparency**: Always explain why contact was initiated

```python
class ContactRateLimiter:
    def __init__(self, min_interval: int = 100):  # tokens between contacts
        self.last_contact_pos = -float('inf')
        self.min_interval = min_interval

    def can_contact(self, current_pos: int) -> bool:
        if current_pos - self.last_contact_pos < self.min_interval:
            return False
        return True

    def record_contact(self, pos: int):
        self.last_contact_pos = pos
```

## Open Questions

1. **Autonomy vs. Control**: How much agency should the model have? Should user be able to disable all actions?

2. **Training Signal**: Where do we get ground truth for "when to act"? Human annotation? Simulation? Self-play?

3. **Interruption UX**: How does the user experience an interruption? Modal? Inline? Asynchronous?

4. **Multi-turn**: If the model contacts user and gets a response, how does that feed back into soma/state?

5. **Action Composition**: Can actions be composed? (e.g., "pause_and_reflect" then "contact_user")

6. **Reversibility**: How to handle if an action was wrong? Can the model "apologize" and course-correct?

## Implementation Phases

### Phase 1: Infrastructure
- [ ] Define ActionSpace with Tier 1 actions only
- [ ] Implement ActionPolicy with simple threshold triggers
- [ ] Add action hooks to forward pass
- [ ] Create UserContactInterface with template-based generation

### Phase 2: User Contact
- [ ] Implement full contact_user action
- [ ] Add rate limiting and safety checks
- [ ] Create UI/UX for interruptions
- [ ] Collect human feedback on contact quality

### Phase 3: Internal Actions
- [ ] Implement Tier 2 actions (adjust_attention, etc.)
- [ ] Learn action policies from outcomes
- [ ] Add action to memory (what did I do and why)

### Phase 4: Full Agency
- [ ] Implement Tier 3 memory actions
- [ ] RL training for action selection
- [ ] Multi-action compositions
- [ ] Long-horizon action planning
