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

        # Update counter
        self.register_buffer('_update_count', torch.tensor(0, dtype=torch.long))

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

        # Output projection
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        features: torch.Tensor,
        surprise: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ) -> OscillatorOutput:
        """
        Content-based memory write with surprise gating.

        New architecture:
        - Features (content) determine WHAT to store in each oscillator
        - Surprise determines HOW STRONGLY to write (importance gate)
        - Output provides memory states for cross-attention querying

        The key insight is that we keep self.phases as a DETACHED buffer (persistent
        state), but compute DIFFERENTIABLE contributions on top of it.

        Gradient paths:
        - frequencies → freq_contribution → effective_phase → output
        - base_amplitudes → modulated_amps → output
        - features → amp_modulator → amp_mod → modulated_amps → output
        - features → phase_modulator → phase_shift → effective_phase → output
        - surprise → write_gate → modulated_amps → output

        Args:
            features: (d_feature_input,) compressed content features
            surprise: Optional scalar or (1,) surprise magnitude for gating
            dt: Time step for phase advance

        Returns:
            OscillatorOutput with projected output and raw memory states
        """
        # === SURPRISE GATING ===
        # Higher surprise = more important to remember = stronger write
        if surprise is not None:
            # Ensure surprise is a scalar
            if surprise.numel() > 1:
                surprise = surprise.mean()
            # Compute write gate: sigmoid(bias + surprise * scale)
            write_gate = torch.sigmoid(
                self.surprise_gate_bias + surprise * self.surprise_gate_scale
            )
        else:
            # Default: moderate write strength
            write_gate = torch.tensor(
                self.surprise_gate_bias,
                device=features.device,
                dtype=features.dtype
            ).sigmoid()

        # === DIFFERENTIABLE COMPUTATIONS ===

        # Frequency contribution to phase (DIFFERENTIABLE w.r.t. frequencies)
        freq_contribution = 2 * math.pi * self.frequencies * dt

        # Amplitude modulation from FEATURES (DIFFERENTIABLE)
        # Features determine WHAT content to store in each oscillator
        amp_mod = self.amp_modulator(features)  # (N,), in [-1, 1]

        # Apply surprise gate: scale the amplitude modulation by importance
        # When surprise is high, we write more strongly
        gated_amp_mod = amp_mod * write_gate

        modulated_amps = self.base_amplitudes * (1 + gated_amp_mod * self.config.max_amp_modulation)
        modulated_amps = modulated_amps.clamp(min=0.0)

        # Phase modulation from FEATURES (DIFFERENTIABLE)
        # Features also influence timing/phase encoding
        phase_mod = self.phase_modulator(features)  # (N,), in [-1, 1]
        phase_shift = phase_mod * self.config.max_phase_shift * math.pi * write_gate

        # === EFFECTIVE PHASE COMPUTATION ===
        # base_phase: detached accumulated phase from buffer (no gradients)
        # freq_contribution: differentiable (gradients to frequencies)
        # phase_shift: differentiable (gradients to phase_modulator)
        effective_phase = self.phases.detach() + freq_contribution + phase_shift

        # === UPDATE BUFFER FOR NEXT CALL (detached) ===
        with torch.no_grad():
            # Store modulation values for monitoring
            self._last_amp_mod.copy_(amp_mod.detach())
            self._last_phase_mod.copy_(phase_mod.detach())
            self.current_amplitudes.copy_(modulated_amps.detach())

            # Update phase buffer: accumulated_phase + freq + phase_shift
            # This is the state that will be used in the NEXT forward call
            new_phases = (self.phases + freq_contribution.detach() + phase_shift.detach()) % (2 * math.pi)
            self.phases.copy_(new_phases)
            self._update_count.add_(1)

        # === COMPUTE OUTPUT ===
        # osc_values: differentiable w.r.t. modulated_amps and effective_phase
        # These are the raw oscillator states that serve as MEMORY
        osc_values = modulated_amps * torch.sin(effective_phase)  # (num_oscillators,)

        # Project to output dimension (for legacy compatibility)
        output = self.output_proj(osc_values)  # (d_output,)

        # Return both projected output and raw oscillator values
        # Raw values are used as memory keys/values for cross-attention
        return OscillatorOutput(output=output, memory_states=osc_values)

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
        Reset oscillator phases.

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
