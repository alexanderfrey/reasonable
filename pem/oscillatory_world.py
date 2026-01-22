"""
Oscillatory World State - Learnable oscillator-based world model for CTM.

Core insight: Instead of storing a state vector with gradients that don't flow through
historical states, represent the world model as oscillating neuron activations at
different frequencies. This provides:
1. Natural timescale separation (slow vs fast oscillators)
2. Differentiable dynamics (phase evolution is deterministic)
3. Learnable modulation (how input affects oscillators gets gradients)
4. Inference-time adaptation (oscillators keep running without backprop)

The key mechanism:
- Frequencies and base amplitudes are LEARNED PARAMETERS (get gradients)
- Phase and amplitude modulation networks are LEARNED (get gradients)
- Runtime phases/amplitudes are BUFFERS (deterministic evolution, no gradients needed)

Timescale hierarchy (with default parameters):
    Oscillator 0:  f = 1/8      -> ~8 steps memory (phrase-level)
    Oscillator 1:  f = 1/16     -> ~16 steps memory
    ...
    Oscillator 63: f = 1/4096   -> ~4096 steps memory (document-level)

Usage:
    osc_world = OscillatoryWorldState(num_oscillators=64, d_sync_input=256)

    # Each forward step:
    osc_world.advance()  # Deterministic phase evolution
    osc_world.modulate(sync_signal)  # Learned modulation (gets gradients!)
    world_state = osc_world.read()  # Sample current oscillator state
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OscillatorOutput(NamedTuple):
    """Output from oscillatory world state forward pass."""
    output: torch.Tensor           # (d_output,) projected world state
    memory_states: torch.Tensor    # (num_oscillators,) raw oscillator values for cross-attention
    phase_dist: torch.Tensor       # (num_oscillators,) smooth phase distance [0, 2] - how far from write phase
    estimated_age: torch.Tensor    # (num_oscillators,) estimated age in steps (within period)
    write_strengths: torch.Tensor  # (num_oscillators,) current write intensity
    write_self_states: torch.Tensor  # (num_oscillators, d_self_state) self-state at write time


class OscillatorMetrics(NamedTuple):
    """Metrics from oscillatory world state for monitoring."""
    # Phase statistics
    phase_mean: float
    phase_std: float
    phase_entropy: float  # How spread out are the phases?

    # Amplitude statistics
    amplitude_mean: float
    amplitude_std: float
    amplitude_max: float

    # Frequency utilization
    active_oscillator_frac: float  # Fraction with amplitude > threshold
    frequency_weighted_amplitude: float  # Are slow or fast oscillators dominant?

    # Modulation strength
    amp_mod_mean: float
    phase_mod_mean: float

    # Output statistics
    output_norm: float
    output_mean: float
    output_std: float


@dataclass
class OscillatoryWorldConfig:
    """Configuration for OscillatoryWorldState.

    New architecture: Content-based memory with surprise gating.
    - Features (content) write to memory via amplitude modulation
    - Surprise gates write strength (unexpected = important to remember)
    - Sync queries memory via cross-attention (handled in GlobalSyncModule)

    Extended with self-state tracking for autobiographical memory:
    - World oscillators (0 to N-num_self) store environment content
    - Self oscillators (N-num_self to N) store cognitive/prediction state
    - Write_self_states buffer stores self-state at write time for each oscillator
    """

    # Oscillator parameters
    num_oscillators: int = 64  # Number of oscillators (memory slots)
    min_period: int = 8        # Fastest oscillator period (steps)
    max_period: int = 4096     # Slowest oscillator period (steps)

    # Input dimensions
    d_feature_input: int = 256  # Dimension of content features for writing
    d_sync_input: int = 256     # Dimension of sync input (legacy, for compatibility)
    d_output: int = 256         # Output dimension

    # Modulation network hidden dim
    modulation_hidden_mult: int = 2  # Hidden = num_oscillators * mult

    # Constraints
    max_amp_modulation: float = 1.0   # Maximum amplitude modulation (Tanh output)
    max_phase_shift: float = 0.5      # Maximum phase shift per step (fraction of pi)

    # Surprise gating
    surprise_gate_bias: float = 0.5   # Base write strength (0.5 = moderate baseline)
    surprise_gate_scale: float = 1.0  # How much surprise amplifies writing

    # Initialization
    init_amplitude_scale: float = 1.0  # Initial amplitude scale

    # Phase-tagged writes (temporal awareness)
    phase_write_threshold: float = 0.1   # Min |amp_mod| to count as significant write
    write_strength_decay: float = 0.99   # Decay factor for old write strengths

    # Self-state tracking (autobiographical memory)
    num_self_oscillators: int = 16       # Oscillators dedicated to self-state (indices N-16 to N)
    d_self_state: int = 64               # Dimension of self-state input
    self_write_threshold: float = 0.1    # Min self-state change to trigger write


class OscillatoryWorldState(nn.Module):
    """
    Oscillator-based world state that evolves deterministically with learnable modulation.

    The key insight: We learn the MODULATION FUNCTION, not the state itself.
    The state evolution is deterministic (phase advance), but how input affects
    the oscillators (amplitude/phase modulation) is learned and gets gradients.

    Architecture:
        ┌──────────────────────────────────────────────────────────────────────┐
        │                    OscillatoryWorldState                             │
        │                                                                      │
        │  LEARNED PARAMETERS (get gradients):                                 │
        │    - frequencies: (N,) oscillator frequencies                        │
        │    - base_amplitudes: (N,) base amplitude per oscillator             │
        │    - amp_modulator: MLP that modulates amplitudes                    │
        │    - phase_modulator: MLP that shifts phases                         │
        │    - output_proj: Linear projection to output dimension              │
        │                                                                      │
        │  RUNTIME BUFFERS (evolve deterministically, no gradients):           │
        │    - phases: (N,) current phase of each oscillator                   │
        │    - current_amplitudes: (N,) current modulated amplitudes           │
        │                                                                      │
        │  FORWARD PASS:                                                       │
        │    1. advance() - phases += 2π * frequencies * dt                    │
        │    2. modulate(sync) - amplitudes/phases affected by input           │
        │    3. read() - returns amplitudes * sin(phases) projected to d_out   │
        └──────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: OscillatoryWorldConfig):
        super().__init__()
        self.config = config
        N = config.num_oscillators

        # === LEARNABLE PARAMETERS ===

        # Initialize frequencies with log-uniform spacing (slow to fast)
        # Frequency = 1/period, so period range [min_period, max_period]
        # gives frequency range [1/max_period, 1/min_period]
        init_periods = torch.logspace(
            math.log10(config.min_period),
            math.log10(config.max_period),
            N,
        )
        init_frequencies = 1.0 / init_periods
        self.frequencies = nn.Parameter(init_frequencies)

        # Learnable base amplitudes (start uniform)
        self.base_amplitudes = nn.Parameter(
            torch.ones(N) * config.init_amplitude_scale
        )

        # === CONTENT WRITE PATH ===
        # Amplitude modulation network: FEATURES -> what to store in each oscillator
        # Input: content features -> Output: amplitude scaling per oscillator
        hidden_dim = N * config.modulation_hidden_mult
        self.amp_modulator = nn.Sequential(
            nn.Linear(config.d_feature_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, N),
            nn.Tanh(),  # Bound to [-1, 1] for stable modulation
        )

        # Phase modulation network: FEATURES -> timing/phase encoding
        # Input: content features -> Output: phase shift per oscillator
        self.phase_modulator = nn.Sequential(
            nn.Linear(config.d_feature_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, N),
            nn.Tanh(),  # Bound phase shifts
        )

        # === SURPRISE GATING ===
        # Surprise determines HOW STRONGLY to write (importance gate)
        # Higher surprise = more important to remember
        # surprise_gate = sigmoid(surprise_gate_bias + surprise * surprise_gate_scale)
        self.surprise_gate_bias = config.surprise_gate_bias
        self.surprise_gate_scale = config.surprise_gate_scale

        # Output projection: oscillator state -> d_output
        self.output_proj = nn.Linear(N, config.d_output)

        # === RUNTIME BUFFERS (not parameters) ===

        # Current phases (0 to 2π)
        # Initialize with random phases for diversity
        self.register_buffer('phases', torch.rand(N) * 2 * math.pi)

        # Current (modulated) amplitudes
        self.register_buffer('current_amplitudes', torch.ones(N) * config.init_amplitude_scale)

        # Track last modulation values for monitoring
        self.register_buffer('_last_amp_mod', torch.zeros(N))
        self.register_buffer('_last_phase_mod', torch.zeros(N))
        self.register_buffer('_last_self_state_change', torch.tensor(0.0))
        self.register_buffer('_last_write_gate_self', torch.tensor(0.0))

        # Update counter
        self.register_buffer('_update_count', torch.tensor(0, dtype=torch.long))

        # === PHASE-TAGGED WRITES (temporal awareness) ===
        # Track when each oscillator was written to (phase at write time)
        self.register_buffer('write_phases', torch.zeros(N))
        # Track write intensity for each oscillator (decays over time)
        self.register_buffer('write_strengths', torch.zeros(N))
        # Controls whether persistent buffers are updated on forward()
        # (Allows callers to explicitly disable updates if needed)
        self._state_updates_enabled = True

        # === SELF-STATE AT WRITE TIME (autobiographical memory) ===
        # Store compressed self-state when each oscillator was written
        # This creates memory of "what I was experiencing when I learned this"
        self.register_buffer('write_self_states', torch.zeros(N, config.d_self_state))
        # Track last self-state for detecting cognitive shifts
        self.register_buffer('_last_self_state', torch.zeros(config.d_self_state))

        # === SELF-STATE MODULATION NETWORKS ===
        # For self oscillators (indices N - num_self_oscillators : N)
        # Input: self-state vector (prediction summary + surprise + confidence)
        # Output: amplitude modulation for self oscillators
        N_self = config.num_self_oscillators
        self.self_amp_modulator = nn.Sequential(
            nn.Linear(config.d_self_state, N_self * 2),
            nn.GELU(),
            nn.Linear(N_self * 2, N_self),
            nn.Tanh(),
        )

        self.self_phase_modulator = nn.Sequential(
            nn.Linear(config.d_self_state, N_self * 2),
            nn.GELU(),
            nn.Linear(N_self * 2, N_self),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable learning."""
        # Amplitude modulator: start near identity (small modulation)
        for i, m in enumerate(self.amp_modulator.modules()):
            if isinstance(m, nn.Linear):
                if i == len(list(self.amp_modulator.modules())) - 2:
                    # Last linear before Tanh: small init for small initial modulation
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Phase modulator: start with very small phase shifts
        for i, m in enumerate(self.phase_modulator.modules()):
            if isinstance(m, nn.Linear):
                if i == len(list(self.phase_modulator.modules())) - 2:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Self-state amplitude modulator: small init for gradual self-state learning
        for i, m in enumerate(self.self_amp_modulator.modules()):
            if isinstance(m, nn.Linear):
                if i == len(list(self.self_amp_modulator.modules())) - 2:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Self-state phase modulator: small init
        for i, m in enumerate(self.self_phase_modulator.modules()):
            if isinstance(m, nn.Linear):
                if i == len(list(self.self_phase_modulator.modules())) - 2:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Output projection
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        features: torch.Tensor,
        surprise: Optional[torch.Tensor] = None,
        self_state: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ) -> OscillatorOutput:
        """
        Content-based memory write with surprise gating and self-state tracking.

        Extended architecture with world + self oscillators:
        - World oscillators (0 to N_world): store environment content, gated by surprise
        - Self oscillators (N_world to N): store cognitive state, gated by self-state change
        - write_self_states: store self-state at write time for autobiographical memory

        Gradient paths:
        - frequencies → freq_contribution → effective_phase → output
        - base_amplitudes → modulated_amps → output
        - features → amp_modulator → amp_mod_world → modulated_amps → output
        - features → phase_modulator → phase_shift_world → effective_phase → output
        - surprise → write_gate_world → modulated_amps → output
        - self_state → self_amp_modulator → amp_mod_self → modulated_amps → output
        - self_state → self_phase_modulator → phase_shift_self → effective_phase → output

        Args:
            features: (d_feature_input,) compressed content features
            surprise: Optional scalar or (1,) surprise magnitude for gating
            self_state: Optional (d_self_state,) cognitive state for self oscillators
            dt: Time step for phase advance

        Returns:
            OscillatorOutput with projected output, memory states, and temporal metadata
        """
        N = self.config.num_oscillators
        N_self = self.config.num_self_oscillators
        N_world = N - N_self

        # === STATE UPDATE CONTROL ===
        # Caller can disable updates to avoid double-advance during checkpointing.
        skip_buffer_update = not self._state_updates_enabled

        # === WORLD OSCILLATOR GATING (indices 0:N_world) ===
        # Higher surprise = more important to remember = stronger write
        if surprise is not None:
            if surprise.numel() > 1:
                surprise = surprise.mean()
            write_gate_world = torch.sigmoid(
                self.surprise_gate_bias + surprise * self.surprise_gate_scale
            )
        else:
            write_gate_world = torch.tensor(
                self.surprise_gate_bias,
                device=features.device,
                dtype=features.dtype
            ).sigmoid()

        # === SELF OSCILLATOR GATING (indices N_world:N) ===
        # Gate by self-state change magnitude (cognitive shift = worth remembering)
        if self_state is not None:
            self_state_change = (self_state - self._last_self_state).norm()
            write_gate_self = torch.sigmoid(
                self_state_change - self.config.self_write_threshold
            )
            self._last_self_state_change.copy_(self_state_change.detach())
            self._last_write_gate_self.copy_(write_gate_self.detach())
        else:
            write_gate_self = torch.tensor(0.0, device=features.device, dtype=features.dtype)
            self._last_self_state_change.zero_()
            self._last_write_gate_self.zero_()

        # === DIFFERENTIABLE COMPUTATIONS ===

        # Frequency contribution to phase (DIFFERENTIABLE w.r.t. frequencies)
        freq_contribution = 2 * math.pi * self.frequencies * dt

        # === WORLD OSCILLATOR MODULATION (from features) ===
        amp_mod_world_full = self.amp_modulator(features)  # (N,), in [-1, 1]
        amp_mod_world = amp_mod_world_full[:N_world]
        phase_mod_world_full = self.phase_modulator(features)  # (N,), in [-1, 1]
        phase_mod_world = phase_mod_world_full[:N_world]

        gated_amp_mod_world = amp_mod_world * write_gate_world

        # === SELF OSCILLATOR MODULATION (from self_state) ===
        if self_state is not None:
            amp_mod_self = self.self_amp_modulator(self_state)  # (N_self,), in [-1, 1]
            phase_mod_self = self.self_phase_modulator(self_state)  # (N_self,), in [-1, 1]
            gated_amp_mod_self = amp_mod_self * write_gate_self
        else:
            gated_amp_mod_self = torch.zeros(N_self, device=features.device, dtype=features.dtype)
            phase_mod_self = torch.zeros(N_self, device=features.device, dtype=features.dtype)

        # === COMBINE WORLD + SELF ===
        gated_amp_mod = torch.cat([gated_amp_mod_world, gated_amp_mod_self])
        phase_mod = torch.cat([phase_mod_world, phase_mod_self])

        # Compute combined write gate for phase shift scaling
        write_gate_combined = torch.cat([
            write_gate_world.expand(N_world),
            write_gate_self.expand(N_self)
        ])

        modulated_amps = self.base_amplitudes * (1 + gated_amp_mod * self.config.max_amp_modulation)
        modulated_amps = modulated_amps.clamp(min=0.0)

        phase_shift = phase_mod * self.config.max_phase_shift * math.pi * write_gate_combined

        # === EFFECTIVE PHASE COMPUTATION ===
        effective_phase = self.phases.detach() + freq_contribution + phase_shift

        # === COMPUTE PHASE DISTANCE AND ESTIMATED AGE ===
        # Smooth phase distance: 1 - cos(Δphase) gives [0, 2] range
        # 0 = just written (same phase), 2 = half period ago (opposite phase)
        phase_diff = effective_phase - self.write_phases
        phase_dist = 1 - torch.cos(phase_diff)  # (N,), in [0, 2]

        # Estimated age in steps (within one period)
        # age = phase_diff / (2π * frequency)
        estimated_age = phase_diff.abs() / (2 * math.pi * self.frequencies + 1e-8)

        # === UPDATE BUFFERS FOR NEXT CALL (detached) ===
        if not skip_buffer_update:
            with torch.no_grad():
                # Store modulation values for monitoring
                self._last_amp_mod.copy_(gated_amp_mod.detach())
                self._last_phase_mod.copy_(phase_mod.detach())
                self.current_amplitudes.copy_(modulated_amps.detach())

                # Update phase buffer
                new_phases = (self.phases + freq_contribution.detach() + phase_shift.detach()) % (2 * math.pi)
                self.phases.copy_(new_phases)

                # === PHASE-TAGGED WRITES ===
                # Decay old write strengths
                self.write_strengths.mul_(self.config.write_strength_decay)

                # Update write_phases and write_strengths for oscillators that were written to
                write_mask = (gated_amp_mod.abs() > self.config.phase_write_threshold)
                write_intensity = gated_amp_mod.abs().detach()

                # Only update write_phases for oscillators with significant writes
                self.write_phases = torch.where(
                    write_mask,
                    new_phases,
                    self.write_phases
                )
                self.write_strengths = torch.where(
                    write_mask,
                    write_intensity,
                    self.write_strengths
                )

                # === AUTOBIOGRAPHICAL MEMORY ===
                # Store current self_state for oscillators that were written to
                if self_state is not None:
                    for i in range(N):
                        if write_mask[i]:
                            self.write_self_states[i] = self_state.detach()
                    # Update last self-state for next comparison
                    self._last_self_state.copy_(self_state.detach())

                self._update_count.add_(1)

        # === COMPUTE OUTPUT ===
        osc_values = modulated_amps * torch.sin(effective_phase)  # (N,)
        output = self.output_proj(osc_values)  # (d_output,)

        return OscillatorOutput(
            output=output,
            memory_states=osc_values,
            phase_dist=phase_dist,
            estimated_age=estimated_age,
            write_strengths=self.write_strengths.clone(),
            write_self_states=self.write_self_states.clone(),
        )

    # Legacy methods for compatibility - now just call forward()
    def advance(self, dt: float = 1.0):
        """Legacy method - phase advance is now integrated into forward()."""
        pass  # No-op, advance happens in forward()

    def modulate(self, sync_input: torch.Tensor):
        """Legacy method - modulation is now integrated into forward()."""
        pass  # No-op, modulation happens in forward()

    def read(self) -> torch.Tensor:
        """Legacy method - reading is now integrated into forward()."""
        # Return current state projection (no differentiable modulation)
        osc_values = self.current_amplitudes * torch.sin(self.phases)
        return self.output_proj(osc_values)

    def reset_phases(self, random: bool = True):
        """
        Reset oscillator phases and all temporal tracking buffers.

        Args:
            random: If True, randomize phases. If False, set to zero.
        """
        with torch.no_grad():
            if random:
                self.phases.copy_(torch.rand_like(self.phases) * 2 * math.pi)
            else:
                self.phases.zero_()
            self.current_amplitudes.copy_(self.base_amplitudes)
            self._update_count.zero_()
            # Reset phase-tagged write buffers
            self.write_phases.zero_()
            self.write_strengths.zero_()
            # Reset self-state tracking
            self.write_self_states.zero_()
            self._last_self_state.zero_()

    def set_state_updates_enabled(self, enabled: bool) -> None:
        """Enable/disable persistent buffer updates during forward()."""
        self._state_updates_enabled = bool(enabled)

    def detach_state(self):
        """
        Detach persistent state from computation graph.

        With the new forward() implementation, the buffer is always kept detached,
        so this method is now a no-op. Kept for API compatibility.
        """
        pass  # Buffer is always detached in new implementation

    def get_oscillator_state(self) -> Dict[str, torch.Tensor]:
        """Get current oscillator state for monitoring."""
        return {
            'frequencies': self.frequencies.detach(),
            'base_amplitudes': self.base_amplitudes.detach(),
            'current_amplitudes': self.current_amplitudes.detach(),
            'phases': self.phases.detach(),
            'last_amp_mod': self._last_amp_mod.detach(),
            'last_phase_mod': self._last_phase_mod.detach(),
        }

    def get_metrics(self) -> OscillatorMetrics:
        """Compute monitoring metrics."""
        with torch.no_grad():
            N = self.config.num_oscillators

            # Phase statistics
            phases = self.phases
            phase_mean = phases.mean().item()
            phase_std = phases.std().item()

            # Phase entropy: how spread out are phases?
            # Bin phases into 16 bins and compute entropy
            num_bins = 16
            bin_edges = torch.linspace(0, 2 * math.pi, num_bins + 1, device=phases.device)
            hist = torch.histc(phases, bins=num_bins, min=0, max=2 * math.pi)
            hist = hist / hist.sum()  # Normalize to probability
            phase_entropy = -(hist * torch.log(hist + 1e-10)).sum().item()

            # Amplitude statistics
            amps = self.current_amplitudes
            amplitude_mean = amps.mean().item()
            amplitude_std = amps.std().item()
            amplitude_max = amps.max().item()

            # Frequency utilization
            active_threshold = 0.1 * amps.max().item()
            active_frac = (amps > active_threshold).float().mean().item()

            # Frequency-weighted amplitude: are slow or fast oscillators dominant?
            # Higher value = slow oscillators more active
            freqs = self.frequencies
            freq_weighted_amp = (amps * (1.0 / freqs)).sum().item() / (amps.sum().item() + 1e-8)

            # Modulation strength
            amp_mod_mean = self._last_amp_mod.abs().mean().item()
            phase_mod_mean = self._last_phase_mod.abs().mean().item()

            # Output statistics
            output = self.read()
            output_norm = output.norm().item()
            output_mean = output.mean().item()
            output_std = output.std().item()

            return OscillatorMetrics(
                phase_mean=phase_mean,
                phase_std=phase_std,
                phase_entropy=phase_entropy,
                amplitude_mean=amplitude_mean,
                amplitude_std=amplitude_std,
                amplitude_max=amplitude_max,
                active_oscillator_frac=active_frac,
                frequency_weighted_amplitude=freq_weighted_amp,
                amp_mod_mean=amp_mod_mean,
                phase_mod_mean=phase_mod_mean,
                output_norm=output_norm,
                output_mean=output_mean,
                output_std=output_std,
            )

    def get_phase_diff_features(self) -> Dict[str, torch.Tensor]:
        """
        Get phase-difference based features for temporal awareness.

        Returns statistics about memory freshness and self-state diversity.
        These can be logged and used to understand temporal dynamics.

        Returns:
            Dict with temporal and self-state metrics.
        """
        with torch.no_grad():
            N = self.config.num_oscillators
            N_self = self.config.num_self_oscillators
            N_world = N - N_self

            # Compute current phase distance from write phases
            phase_diff = self.phases - self.write_phases
            phase_dist = 1 - torch.cos(phase_diff)  # [0, 2]

            # Estimated age in steps
            estimated_age = phase_diff.abs() / (2 * math.pi * self.frequencies + 1e-8)

            stats = {}

            # Overall stats
            stats['phase_dist_mean'] = phase_dist.mean()
            stats['phase_dist_std'] = phase_dist.std()
            stats['estimated_age_mean'] = estimated_age.mean()
            stats['estimated_age_std'] = estimated_age.std()

            # Write activity
            stats['write_fraction'] = (self.write_strengths > self.config.phase_write_threshold).float().mean()
            stats['write_strength_mean'] = self.write_strengths.mean()

            # World oscillator stats (indices 0:N_world)
            stats['world_phase_dist_mean'] = phase_dist[:N_world].mean()
            stats['world_estimated_age_mean'] = estimated_age[:N_world].mean()
            stats['world_write_fraction'] = (
                self.write_strengths[:N_world] > self.config.phase_write_threshold
            ).float().mean()

            # Self oscillator stats (indices N_world:N)
            stats['self_phase_dist_mean'] = phase_dist[N_world:].mean()
            stats['self_estimated_age_mean'] = estimated_age[N_world:].mean()
            stats['self_write_fraction'] = (
                self.write_strengths[N_world:] > self.config.phase_write_threshold
            ).float().mean()

            # Self-state diversity: are we storing varied cognitive states?
            # Higher variance = more diverse self-states stored
            self_state_var = self.write_self_states.var(dim=0).mean()
            stats['self_state_diversity'] = self_state_var

            # Self-state recency correlation with write_strengths
            # Measures if recent writes have stronger stored self-states
            recent_mask = phase_dist < 0.5  # Recently written
            if recent_mask.any():
                recent_self_norm = self.write_self_states[recent_mask].norm(dim=-1).mean()
                old_self_norm = self.write_self_states[~recent_mask].norm(dim=-1).mean() if (~recent_mask).any() else recent_self_norm
                stats['self_state_recency_ratio'] = recent_self_norm / (old_self_norm + 1e-8)
            else:
            stats['self_state_recency_ratio'] = torch.tensor(1.0, device=self.phases.device)

            # Self-state gating diagnostics
            stats['self_state_change'] = self._last_self_state_change.detach()
            stats['self_write_gate'] = self._last_write_gate_self.detach()

            return stats

    def compute_self_world_coherence(self) -> Dict[str, torch.Tensor]:
        """
        Measure alignment between self and world oscillators.

        High coherence: self and world written together (integrated experience)
        Low coherence: self and world out of sync (dissociated processing)

        This measures whether cognitive self-state and world content are
        being processed in a coordinated way.

        Returns:
            Dict with coherence metrics.
        """
        with torch.no_grad():
            N = self.config.num_oscillators
            N_self = self.config.num_self_oscillators
            N_world = N - N_self

            world_phases = self.phases[:N_world]
            self_phases = self.phases[N_world:]

            # Mean phase vectors (Kuramoto order parameter style)
            world_cos = torch.cos(world_phases).mean()
            world_sin = torch.sin(world_phases).mean()
            self_cos = torch.cos(self_phases).mean()
            self_sin = torch.sin(self_phases).mean()

            # World coherence: how synchronized are world oscillators with each other?
            world_coherence = torch.sqrt(world_cos**2 + world_sin**2)

            # Self coherence: how synchronized are self oscillators with each other?
            self_coherence = torch.sqrt(self_cos**2 + self_sin**2)

            # Self-world alignment: cos(world_mean_phase - self_mean_phase)
            # High = self and world in phase, low = out of phase
            self_world_alignment = world_cos * self_cos + world_sin * self_sin

            # Write timing coherence: are self and world being written at similar times?
            world_write_active = (self.write_strengths[:N_world] > self.config.phase_write_threshold).float()
            self_write_active = (self.write_strengths[N_world:] > self.config.phase_write_threshold).float()
            # Measure overlap in write activity
            write_timing_coherence = (world_write_active.mean() * self_write_active.mean()).sqrt()

            return {
                'self_world_alignment': self_world_alignment,
                'world_coherence': world_coherence,
                'self_coherence': self_coherence,
                'write_timing_coherence': write_timing_coherence,
            }


def create_oscillatory_world(
    num_oscillators: int = 64,
    min_period: int = 8,
    max_period: int = 4096,
    d_sync_input: int = 256,
    d_output: int = 256,
    **kwargs,
) -> OscillatoryWorldState:
    """Factory function to create OscillatoryWorldState."""
    config = OscillatoryWorldConfig(
        num_oscillators=num_oscillators,
        min_period=min_period,
        max_period=max_period,
        d_sync_input=d_sync_input,
        d_output=d_output,
        **kwargs,
    )
    return OscillatoryWorldState(config)
