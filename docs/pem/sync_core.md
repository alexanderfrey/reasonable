# Sync Core (Meta-Cognition Layer)

CTM-style synchronization as the cognitive substrate. Surprise modulates dynamics.

## Role

- **Cognitive state**: Sync IS the representation (not activations)
- **Surprise integration**: High surprise → disrupted sync → re-learning
- **Temporal dynamics**: NLMs process history independently per neuron
- **Memory integration**: Retrieved memories influence sync evolution

## Core Insight

From CTM: **Synchronization patterns between neurons = representation**

- Which neurons fire together encodes meaning
- Surprise should disrupt these patterns (something unexpected happened)
- Re-synchronization after disruption = updating understanding

## Implementation

```python
class SyncCore(nn.Module):
    """
    CTM-style synchronization as the cognitive substrate.

    Key insight: Surprise MODULATES sync dynamics.
    - Low surprise → stable sync patterns (understanding confirmed)
    - High surprise → sync disruption → re-synchronization (learning)
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.sync_pairs = config.sync_pairs
        self.num_ticks = config.num_ticks

        # Per-neuron temporal processors (NLMs)
        self.nlm = NeuronLevelModel(
            d_model=config.d_model,
            nlm_hidden=config.nlm_hidden,
            nlm_depth=config.nlm_depth,
            max_ticks=config.num_ticks,
        )

        # Synchronization measurement
        self.sync = SynchronizationModule(
            d_model=config.d_model,
            sync_pairs=config.sync_pairs,
        )

        # Surprise → dynamics modulation
        self.surprise_to_modulation = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Sigmoid(),  # Output in [0, 1] for multiplicative modulation
        )

        # Memory → state injection
        self.memory_gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.Sigmoid(),
        )
        self.memory_proj = nn.Linear(config.d_model, config.d_model)

        # Normalization
        self.z_norm = RMSNorm(config.d_model)

        # Tick embeddings
        self.tick_embed = nn.Embedding(config.num_ticks, config.d_model)

    def forward(
        self,
        features: Tensor,           # (B, S, D) from feature extractor
        surprise_signal: Tensor,    # (B, 1, D) surprise direction * magnitude
        memory_retrieval: Tensor,   # (B, 1, D) retrieved memory signal
        num_ticks: Optional[int] = None,
    ) -> Tensor:
        """
        Run sync core for multiple ticks.

        Args:
            features: Input features from transformer
            surprise_signal: Surprise to integrate (can be zero)
            memory_retrieval: Retrieved memories (can be zero)
            num_ticks: Override tick count (for adaptive compute)

        Returns:
            sync: (B, S, sync_pairs) final synchronization state
        """
        num_ticks = num_ticks or self.num_ticks
        B, S, D = features.shape
        device = features.device

        # Initialize state from features
        z = features  # (B, S, D)

        # Compute surprise modulation
        # High surprise → modulation near 1 (more change)
        # Low surprise → modulation near 0 (stable)
        # Broadcast surprise to all positions
        surprise_expanded = surprise_signal.expand(B, S, D)
        dynamics_modulation = self.surprise_to_modulation(surprise_expanded)

        # Compute memory gate
        # How much should retrieved memories influence state?
        memory_expanded = memory_retrieval.expand(B, S, D)
        gate_input = torch.cat([z, memory_expanded], dim=-1)
        memory_gate = self.memory_gate(gate_input)
        memory_contribution = memory_gate * self.memory_proj(memory_expanded)

        # Initialize history
        z_history = [z]

        for tick in range(num_ticks):
            tick_idx = torch.tensor(tick, device=device)
            tick_emb = self.tick_embed(tick_idx)  # (D,)

            # Stack history for NLM
            history_tensor = torch.stack(z_history, dim=2)  # (B, S, T, D)

            # NLM processes history → new state
            z_new = self.nlm(history_tensor)

            # Modulate change by surprise
            # High surprise → accept more of NLM output (bigger update)
            # Low surprise → stay closer to previous state
            z_change = z_new - z
            z = z + dynamics_modulation * z_change

            # Inject memory contribution (gated)
            z = z + memory_contribution * (1.0 / num_ticks)  # Spread across ticks

            # Add tick embedding
            z = z + tick_emb

            # Normalize
            z = self.z_norm(z)

            # Update history
            z_history.append(z)

        # Compute final sync from full history
        full_history = torch.stack(z_history, dim=2)  # (B, S, T+1, D)
        sync = self.sync(full_history)  # (B, S, sync_pairs)

        return sync
```

## Surprise Modulation

The key mechanism: **surprise controls how much the state changes**

```python
# Low surprise: "This matches my expectations"
dynamics_modulation ≈ 0.1
z_new = z + 0.1 * (nlm_output - z)  # Small update, stable patterns

# High surprise: "This is unexpected!"
dynamics_modulation ≈ 0.9
z_new = z + 0.9 * (nlm_output - z)  # Large update, disrupted patterns
```

This creates an "attention" mechanism based on surprise:
- Expected content → skim (low processing)
- Unexpected content → focus (deep processing)

## Memory Integration

Retrieved memories influence sync evolution:

```python
# Memory says: "Last time in similar context, X happened"
# This should bias current sync toward patterns that accommodate X

memory_contribution = memory_gate * memory_proj(memory_retrieval)
z = z + memory_contribution
```

The gate learns when memories are relevant vs distracting.

## Sync as Representation

After all ticks, sync captures:
- **Which neurons correlated** during processing
- **How patterns evolved** over ticks
- **The cognitive state** of having processed this input with this surprise

Sync, not z, is passed to downstream modules (prediction, output).

## Adaptive Compute

Surprise can control tick count:

```python
def forward_adaptive(self, features, surprise_signal, memory_retrieval):
    # More ticks for more surprising content
    surprise_magnitude = surprise_signal.abs().mean()

    if surprise_magnitude > 0.8:
        num_ticks = self.max_ticks      # Think hard
    elif surprise_magnitude > 0.4:
        num_ticks = self.default_ticks  # Normal processing
    else:
        num_ticks = self.min_ticks      # Quick pass

    return self.forward(features, surprise_signal, memory_retrieval, num_ticks)
```

## NLM Design for Surprise

NLMs should develop **diverse temporal dynamics**:
- Some neurons: fast responders (react to surprise quickly)
- Some neurons: slow integrators (maintain stable context)
- Some neurons: oscillators (carry temporal structure)

This diversity is achieved through initialization:
```python
# Different neurons have different "time constants"
neuron_scales = torch.exp(torch.linspace(-0.7, 0.7, d_model))
```

## Open Questions

1. **Surprise injection point**: Before NLM? After? As gating?
2. **Memory timing**: Inject at start? Spread across ticks? At end?
3. **Tick count**: Fixed? Adaptive? Learned halting?
4. **Multiple sync scales**: Local sync vs global sync?
5. **Sync → prediction**: Direct projection? Through attention?
