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

## Emergent Self via Persistent Sync Patterns

**Status: IMPLEMENTED** ✅

### Vision

Build toward an emergent "self" by making sync patterns **persistent** across time. The world model isn't a separate module - it **emerges from** accumulated synchronization patterns.

Core principle: **Higher cognition emerges from sync, not alongside it.**

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Data format** | Continuous stream of pages | No batch complexity, natural reading flow |
| **Reset policy** | Never reset | Accumulate general world knowledge across all books |
| **Influence mechanism** | Initialize z_0 | World state biases what CTM attends to from the start |

### Architecture

```
         Continuous Stream: Book1_p1, Book1_p2, ..., Book2_p1, Book2_p2, ...
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────┐
│              PERSISTENT SYNC STATE (S_world)                │
│                                                             │
│   Shape: (d_sync_state,) - single vector, never reset       │
│                                                             │
│   • Lives across ALL forward passes                         │
│   • Updated incrementally via GRU-style gated mechanism     │
│   • IS the world model (emergent, not engineered)          │
│   • Accumulates knowledge across entire library             │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ S_world → z_0 (initial post-activations)
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
     ┌──────────┐       ┌──────────┐       ┌──────────┐
     │  Page 1  │       │  Page 2  │       │  Page N  │
     │          │       │          │       │          │
     │ z_0=f(Sw)│       │ z_0=f(Sw)│       │ z_0=f(Sw)│
     │ ↓        │       │ ↓        │       │ ↓        │
     │ CTM ticks│       │ CTM ticks│       │ CTM ticks│
     │ ↓        │       │ ↓        │       │ ↓        │
     │ S_new_1  │       │ S_new_2  │       │ S_new_N  │
     └────┬─────┘       └────┬─────┘       └────┬─────┘
          │                  │                  │
          └─────► update ◄───┴─────► update ◄───┘
                    │                   │
                    ▼                   ▼
              S_world_1 ──────► S_world_2 ──────► ...
```

### Implementation Components

#### 1. PersistentSyncState (`global_sync.py`)
```python
class PersistentSyncState(nn.Module):
    """Maintains sync state across time - the emergent world model."""

    def __init__(self, d_sync: int = 256):
        self.register_buffer('S_world', torch.zeros(d_sync))
        self.register_buffer('update_count', torch.tensor(0))

    def get_state(self) -> torch.Tensor:
        return self.S_world

    @torch.no_grad()
    def update(self, S_new: torch.Tensor):
        self.S_world.copy_(S_new)
        self.update_count.add_(1)
```

#### 2. SyncUpdateGate (`global_sync.py`)
GRU-style gating for selective incorporation of new sync patterns:
```python
class SyncUpdateGate(nn.Module):
    """Decides how much to incorporate new sync vs keep old."""

    def forward(self, S_world, S_new) -> torch.Tensor:
        # GRU mechanics: reset gate, update gate, candidate
        r = sigmoid(self.reset_gate([S_world, S_new]))
        z = sigmoid(self.update_gate([S_world, S_new]))  # ~0.12 initially
        candidate = tanh(self.candidate([r * S_world, S_new]))
        return (1 - z) * S_world + z * candidate
```

#### 3. World State → z_0 Projection (`ctm_base.py`)
```python
# In CTMCore.__init__
self.world_to_z0 = nn.Sequential(
    nn.Linear(d_world_state, d_neurons),
    nn.Tanh(),  # Same range as post-activations
)

# In CTMCore.forward
if world_state is not None:
    z_world = self.world_to_z0(world_state)
    z_t = z_t + z_world.unsqueeze(0).unsqueeze(0)  # Bias initial state
```

#### 4. Training Loop Integration (`train_pem_global.py`)
```python
for page in continuous_stream:
    # Forward pass - world state influences z_0
    output, state = model(page)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Commit updated world state AFTER backward (detached)
    model.commit_world_state(output.world_state.detach())
```

### Configuration

```python
# GlobalSyncConfig
use_persistent_state: bool = True   # Enable/disable
d_sync_state: int = 256             # World state dimension

# CTMBaseConfig
use_world_state: bool = True        # Enable world state initialization
d_world_state: int = 256            # Must match d_sync_state
```

### Metrics & Logging

| Metric | What It Shows |
|--------|---------------|
| `world_state/norm` | How much knowledge accumulates |
| `world_state/update_gate` | How much new info incorporated (0=ignore, 1=replace) |
| `world_state/update_count` | Number of pages processed |
| `world_state/mean`, `world_state/std` | Distribution statistics |

### What Emerges

| Level | What It Is | How It Emerges |
|-------|------------|----------------|
| **World Model** | S_world captures narrative understanding | Accumulated sync patterns |
| **Continuity** | Same "reader" across pages | Persistent state carries forward |
| **Attention Bias** | Focus on what matters | S_world shapes CTM initial state |
| **Future: Self-Model** | Patterns about own processing | Sync patterns that predict other sync patterns |
| **Future: Metacognition** | Awareness of own states | Surprise about S_world predictions |

### Files Modified

| File | Changes |
|------|---------|
| `pem/global_sync.py` | `PersistentSyncState`, `SyncUpdateGate`, integrated with `GlobalSyncModule` |
| `pem/ctm_base.py` | `world_to_z0` projection, `world_state` param in `CTMCore.forward()` |
| `pem/prediction_ctm.py` | Pass `world_state` through to core |
| `pem/pem_loop_global.py` | Wire world state: read before CTM, update after, new methods |
| `pem/train_pem_global.py` | `commit_world_state()` after backward, wandb logging |
| `pem/data/continuous_loader.py` | New continuous streaming data loaders |

### Future Directions

Once persistent sync is working, natural next steps:

1. **Self-prediction**: Predict what S_world will become → metacognition
2. **Variable thinking**: S_world influences T (how long to think)
3. **Memory integration**: S_world helps decide what to remember
4. **Multiple timescales**: Fast S_world (current page) + slow S_world (lifetime)

---

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
