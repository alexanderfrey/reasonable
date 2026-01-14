# Global Sync Architecture

## Overview

A new architectural twist where every module works internally with the CTM architecture
(NLM -> Sync -> Synapse) and every module outputs its NLM post-activations into a global
Sync module that produces a global synchronization status used to attend to new incoming features.

## Core Idea

Instead of having a single CTM for prediction with surprise modulating attention, we have
**multiple specialized CTM-based modules** that each process information with their own
internal dynamics. Their collective neural states feed into a **Global Sync Module** that
determines what to attend to next.

This aligns with **Global Workspace Theory (GWT)** - multiple specialized processors
contributing to a unified conscious state.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│   Features ─────────────────────────────────────────────────────────────────┐    │
│       │                                                                     │    │
│       ▼                                                                     │    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │    │
│   │                    CTM-Based Specialized Modules                    │   │    │
│   │                                                                     │   │    │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │    │
│   │  │ Prediction  │  │  Surprise   │  │   Valence   │  │  Curiosity  │ │   │    │
│   │  │    CTM      │  │    CTM      │  │    CTM      │  │    CTM      │ │   │    │
│   │  │             │  │             │  │             │  │             │ │   │    │
│   │  │ NLM→Sync→   │  │ NLM→Sync→   │  │ NLM→Sync→   │  │ NLM→Sync→   │ │   │    │
│   │  │ Synapse     │  │ Synapse     │  │ Synapse     │  │ Synapse     │ │   │    │
│   │  │      │      │  │      │      │  │      │      │  │      │      │ │   │    │
│   │  │      ▼      │  │      ▼      │  │      ▼      │  │      ▼      │ │   │    │
│   │  │ predictions │  │  surprise   │  │   valence   │  │  curiosity  │ │   │    │
│   │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │   │    │
│   │         │                │                │                │        │   │    │
│   │         │    NLM Post-Activations (neural population states)        │   │    │
│   │         ▼                ▼                ▼                ▼        │   │    │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │    │
│   │  │ h_pred(t)   │  │ h_surp(t)   │  │ h_val(t)    │  │ h_cur(t)    │ │   │    │
│   │  │ (B,S,D_n)   │  │ (B,S,D_n)   │  │ (B,S,D_n)   │  │ (B,S,D_n)   │ │   │    │
│   │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │   │    │
│   │         │                │                │                │        │   │    │
│   │  ┌─────────────┐  ┌─────────────┐                                   │   │    │
│   │  │ Activation  │  │ Imagination │    (+ more modules...)            │   │    │
│   │  │    CTM      │  │    CTM      │                                   │   │    │
│   │  │      │      │  │      │      │                                   │   │    │
│   │  │      ▼      │  │      ▼      │                                   │   │    │
│   │  │  arousal    │  │  imagined   │                                   │   │    │
│   │  └──────┬──────┘  └──────┬──────┘                                   │   │    │
│   │         │                │                                          │   │    │
│   │         ▼                ▼                                          │   │    │
│   │  ┌─────────────┐  ┌─────────────┐                                   │   │    │
│   │  │ h_act(t)    │  │ h_imag(t)   │                                   │   │    │
│   │  └──────┬──────┘  └──────┬──────┘                                   │   │    │
│   └─────────┼────────────────┼──────────────────────────────────────────┘   │    │
│             │                │                                              │    │
│             └───────┬────────┴────────┬─────────┬─────────┬────────────┘    │    │
│                     │                 │         │         │                 │    │
│                     ▼                 ▼         ▼         ▼                 │    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │    │
│   │                         GLOBAL SYNC MODULE                          │   │    │
│   │                                                                     │   │    │
│   │    ┌────────────────────────────────────────────────────────────┐   │   │    │
│   │    │  Concatenate all NLM post-activations:                     │   │   │    │
│   │    │  H_global = [h_pred; h_surp; h_val; h_cur; h_act; h_imag]  │   │   │    │
│   │    │            (B, S, D_n * num_modules)                       │   │   │    │
│   │    └────────────────────────────────────────────────────────────┘   │   │    │
│   │                              │                                      │   │    │
│   │                              ▼                                      │   │    │
│   │    ┌────────────────────────────────────────────────────────────┐   │   │    │
│   │    │  Cross-Module Synchronization:                             │   │   │    │
│   │    │  sync_matrix = H_global @ H_global.T                       │   │   │    │
│   │    │                                                            │   │   │    │
│   │    │  Which modules are "in sync"?                              │   │   │    │
│   │    │  - Prediction aligned with Surprise? → expected error      │   │   │    │
│   │    │  - Curiosity aligned with Valence? → motivated exploration │   │   │    │
│   │    │  - Imagination aligned with Prediction? → grounded fantasy │   │   │    │
│   │    └────────────────────────────────────────────────────────────┘   │   │    │
│   │                              │                                      │   │    │
│   │                              ▼                                      │   │    │
│   │    ┌────────────────────────────────────────────────────────────┐   │   │    │
│   │    │  Global Sync State:                                        │   │   │    │
│   │    │  sync_global = f(sync_matrix, personality, memory)         │   │   │    │
│   │    │              (B, S, sync_pairs)                            │   │   │    │
│   │    └────────────────────────────────────────────────────────────┘   │   │    │
│   │                              │                                      │   │    │
│   └──────────────────────────────┼──────────────────────────────────────┘   │    │
│                                  │                                          │    │
│                                  ▼                                          │    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │    │
│   │                      GLOBAL ATTENTION                               │   │    │
│   │                                                                     │   │    │
│   │    sync_global ──► OscillationQueryBuilder ──► query                │   │    │
│   │                                                    │                │   │    │
│   │                                                    ▼                │   │    │
│   │    features (KV cache) ◄────────────── CrossAttention               │   │    │
│   │                                                    │                │   │    │
│   │                                                    ▼                │   │    │
│   │                                              observation ───────────┼───┘    │
│   │                                                                     │        │
│   └─────────────────────────────────────────────────────────────────────┘        │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Cross-Module Synchronization Matrix

The global sync matrix captures **how different cognitive functions align**:

```
                 Pred   Surp   Val    Cur    Act    Imag
            ┌──────────────────────────────────────────────┐
Prediction  │  1.0    0.8    0.3    0.5    0.6    0.4     │  → "Am I predicting what surprises me?"
Surprise    │  0.8    1.0    0.7    0.6    0.9    0.3     │  → "Is surprise emotional?"
Valence     │  0.3    0.7    1.0    0.4    0.5    0.6     │  → "Is this good/bad aligned with imagination?"
Curiosity   │  0.5    0.6    0.4    1.0    0.7    0.8     │  → "Am I curious about what I imagine?"
Activation  │  0.6    0.9    0.5    0.7    1.0    0.5     │  → "Am I aroused by surprise?"
Imagination │  0.4    0.3    0.6    0.8    0.5    1.0     │  → "Is imagination driven by curiosity?"
            └──────────────────────────────────────────────┘
```

## What This Architecture Enables

1. **Coherent attention**: Only attend when modules agree (high global sync)
2. **Conflict detection**: Low sync between Valence and Curiosity → "I'm curious but scared"
3. **Grounded imagination**: High Imagination-Prediction sync → realistic mental simulation
4. **Emotional salience**: High Surprise-Valence sync → emotionally significant events
5. **Motivated exploration**: High Curiosity-Activation sync → engaged exploration

## CTM Module Base Interface

Each CTM-based module exposes both its result AND its neural state:

```python
class CTMModuleOutput(NamedTuple):
    """Output from any CTM-based module."""
    result: torch.Tensor           # Module-specific output (predictions, surprise, etc.)
    post_activations: torch.Tensor # (B, S, D_neurons) NLM state for global sync
    local_sync: torch.Tensor       # (B, S, D_n, D_n) internal sync matrix
    certainty: torch.Tensor        # (B, S) confidence in result


class CTMModule(nn.Module):
    """Base class for all CTM-based modules."""

    def __init__(self, config: CTMModuleConfig):
        super().__init__()
        self.nlm = NeuralLogicModule(config)
        self.sync = LocalSyncModule(config)
        self.synapse = SynapseModule(config)
        self.readout = nn.Linear(config.d_neurons, config.d_output)

    def forward(self, x: torch.Tensor, ...) -> CTMModuleOutput:
        h = self.nlm.init_state(x)
        all_outputs = []

        for tick in range(self.T):
            h = self.nlm(h, x)              # NLM neurons process
            local_sync = self.sync(h)        # Compute local sync
            h = self.synapse(h, local_sync)  # Update via synapse
            all_outputs.append(h)

        # Select output using CTM's t1/t2 mechanism
        result = self.readout(h)
        certainty = self.compute_certainty(all_outputs)

        return CTMModuleOutput(
            result=result,
            post_activations=h,      # For global sync
            local_sync=local_sync,   # For analysis
            certainty=certainty,
        )
```

## Global Sync Module

```python
class GlobalSyncModule(nn.Module):
    """
    Combines post-activations from all CTM modules into global sync state.

    This is the "global workspace" where all specialized processors meet.
    """

    def __init__(self, config: GlobalSyncConfig):
        super().__init__()
        self.num_modules = config.num_modules
        self.d_neurons = config.d_neurons

        # Project each module's activations to common space
        self.module_projections = nn.ModuleList([
            nn.Linear(config.d_neurons, config.d_sync_space)
            for _ in range(config.num_modules)
        ])

        # Cross-module attention
        self.cross_module_attn = nn.MultiheadAttention(
            embed_dim=config.d_sync_space,
            num_heads=config.n_heads,
        )

        # Sync integrator (includes personality, memory)
        self.integrator = SyncIntegrator(config)

    def forward(
        self,
        module_activations: List[torch.Tensor],  # [(B,S,D_n), ...] from each module
        personality: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> GlobalSyncOutput:
        # 1. Project each module to common space
        projected = [
            proj(act) for proj, act in zip(self.module_projections, module_activations)
        ]

        # 2. Stack: (num_modules, B, S, D_sync)
        stacked = torch.stack(projected, dim=0)

        # 3. Cross-module sync matrix
        # Which modules are aligned at each position?
        sync_matrix = torch.einsum('mbsd,nbsd->mnsb', stacked, stacked)
        sync_matrix = sync_matrix / math.sqrt(self.d_sync_space)

        # 4. Cross-module attention (modules attend to each other)
        # Reshape for attention: (B*S, num_modules, D_sync)
        B, S, D = projected[0].shape
        attn_input = stacked.permute(1, 2, 0, 3).reshape(B*S, self.num_modules, -1)

        attended, attn_weights = self.cross_module_attn(
            attn_input, attn_input, attn_input
        )

        # 5. Integrate with personality and memory
        global_sync = self.integrator(
            attended.reshape(B, S, -1),
            personality,
            memory_state,
            sync_matrix,
        )

        return GlobalSyncOutput(
            sync=global_sync,              # (B, S, sync_pairs)
            cross_module_sync=sync_matrix, # (num_modules, num_modules, S, B)
            attention_weights=attn_weights,
        )
```

## Specialized CTM Modules

### PredictionCTM
- **Input**: Features from backbone
- **Output**: Predictions at multiple temporal scales
- **Post-activations**: "What I expect to see"

### SurpriseCTM
- **Input**: Predictions + actual features
- **Output**: Surprise magnitude and direction
- **Post-activations**: "What violated my expectations"

### ValenceCTM
- **Input**: Features + surprise
- **Output**: Good/bad valence signal
- **Post-activations**: "How I feel about this"

### CuriosityCTM
- **Input**: Features + uncertainty estimates
- **Output**: Exploration bonus, information gain
- **Post-activations**: "What I want to know more about"

### ActivationCTM
- **Input**: Features + surprise + valence
- **Output**: Arousal level, attention temperature
- **Post-activations**: "How engaged I am"

### ImaginationCTM
- **Input**: Features + curiosity + valence
- **Output**: Imagined future states
- **Post-activations**: "What I'm simulating"

## Comparison: Current vs Proposed

| Aspect | Current Architecture | Proposed Architecture |
|--------|---------------------|----------------------|
| Sync source | CTMPrediction only | All modules |
| Module interaction | Sequential | Parallel + Global Sync |
| Attention basis | Surprise only | Cross-module coherence |
| "Consciousness" | Single stream | Global workspace |
| Module architecture | Mixed (CTM + simple) | All CTM-based |
| Interpretability | Limited | Rich (sync matrix) |

## Implementation Plan

1. **Phase 1: CTMModule Base Class**
   - Extract common CTM logic into reusable base
   - Define CTMModuleOutput interface
   - Test with existing CTMPrediction

2. **Phase 2: Convert Existing Modules**
   - SurpriseModule → SurpriseCTM
   - ValenceModule → ValenceCTM
   - CuriosityModule → CuriosityCTM
   - ActivationModule → ActivationCTM
   - ImaginationModule → ImaginationCTM

3. **Phase 3: Global Sync Module**
   - Implement GlobalSyncModule
   - Cross-module attention
   - Integration with personality/memory

4. **Phase 4: New PEM Loop**
   - Wire all CTM modules in parallel
   - Feed post-activations to GlobalSync
   - Global attention from sync state

5. **Phase 5: Training**
   - Multi-task loss (each module + global coherence)
   - Curriculum: start with fewer modules, add progressively

## Open Questions

1. **Should modules run in parallel or sequence?**
   - Parallel: faster, but no inter-module dependencies within tick
   - Sequential: slower, but surprise can inform valence within same tick

2. **How many ticks per module?**
   - Same T for all? Or different T based on complexity?
   - Prediction may need more ticks than Valence

3. **Should global sync run every tick or only at end?**
   - Every tick: richer dynamics, more compute
   - End only: simpler, faster

4. **How to handle module-specific inputs?**
   - Some modules need outputs from others (Surprise needs Predictions)
   - Stagger computation? Or use previous tick's outputs?
