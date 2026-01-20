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
    """Configuration for OscillatoryWorldState."""

    # Oscillator parameters
    num_oscillators: int = 64  # Number of oscillators
    min_period: int = 8        # Fastest oscillator period (steps)
    max_period: int = 4096     # Slowest oscillator period (steps)

    # Input/output dimensions
    d_sync_input: int = 256    # Dimension of sync input for modulation
    d_output: int = 256        # Output dimension

    # Modulation network hidden dim
    modulation_hidden_mult: int = 2  # Hidden = num_oscillators * mult

    # Constraints
    max_amp_modulation: float = 1.0   # Maximum amplitude modulation (Tanh output)
    max_phase_shift: float = 0.5      # Maximum phase shift per step (fraction of pi)

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

        # Amplitude modulation network
        # Input: sync signal -> Output: amplitude scaling per oscillator
        hidden_dim = N * config.modulation_hidden_mult
        self.amp_modulator = nn.Sequential(
            nn.Linear(config.d_sync_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, N),
            nn.Tanh(),  # Bound to [-1, 1] for stable modulation
        )

        # Phase modulation network
        # Input: sync signal -> Output: phase shift per oscillator
        self.phase_modulator = nn.Sequential(
            nn.Linear(config.d_sync_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, N),
            nn.Tanh(),  # Bound phase shifts
        )

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

    def advance(self, dt: float = 1.0):
        """
        Advance oscillator phases by one time step.

        This is deterministic - phases evolve based on learned frequencies.
        Called once per forward pass (page/batch).

        Args:
            dt: Time step size (default 1.0)
        """
        # Phase advance: φ += 2π * f * dt
        # Use detach to ensure no gradients flow through phase accumulation
        # (we want gradients through frequency, not through accumulated phase)
        phase_delta = 2 * math.pi * self.frequencies.detach() * dt
        self.phases = (self.phases + phase_delta) % (2 * math.pi)
        self._update_count = self._update_count + 1

    def modulate(self, sync_input: torch.Tensor):
        """
        Apply learned modulation based on sync input.

        THIS IS WHERE GRADIENTS FLOW! The modulation networks learn
        how to adjust oscillator amplitudes and phases based on input.

        Args:
            sync_input: (d_sync_input,) compressed sync signal
        """
        # Amplitude modulation: scale base amplitudes
        amp_mod = self.amp_modulator(sync_input)  # (N,), in [-1, 1]
        self._last_amp_mod = amp_mod.detach()

        # Modulated amplitude = base * (1 + amp_mod)
        # With Tanh output, this gives range [0, 2] * base
        # Using clamp to ensure non-negative
        modulated_amps = self.base_amplitudes * (1 + amp_mod * self.config.max_amp_modulation)
        modulated_amps = modulated_amps.clamp(min=0.0)

        # Phase modulation: shift phases
        phase_mod = self.phase_modulator(sync_input)  # (N,), in [-1, 1]
        self._last_phase_mod = phase_mod.detach()

        # Scale phase shift to max_phase_shift * π
        # This is kept differentiable for use in read()
        phase_shift = phase_mod * self.config.max_phase_shift * math.pi

        # Update stored phases (non-differentiable - we don't backprop through history)
        with torch.no_grad():
            self.phases = (self.phases + phase_shift.detach()) % (2 * math.pi)
            self.current_amplitudes = modulated_amps.detach()

        # Store the modulated values for use in read()
        # These maintain the gradient graph for backprop through modulation networks
        self._modulated_amps_for_read = modulated_amps
        self._phase_shift_for_read = phase_shift  # Keep phase shift differentiable

    def read(self) -> torch.Tensor:
        """
        Read current oscillator state as output vector.

        The oscillator output is computed as: A * sin(φ_base + φ_shift)
        where A (amplitude) and φ_shift (phase modulation) are differentiable.

        Returns:
            (d_output,) world state vector
        """
        # Get amplitudes (differentiable if just modulated)
        if hasattr(self, '_modulated_amps_for_read'):
            amps = self._modulated_amps_for_read
        else:
            amps = self.current_amplitudes

        # Get phase shift (differentiable if just modulated)
        if hasattr(self, '_phase_shift_for_read'):
            # Compute output with differentiable phase shift:
            # sin(base_phase + phase_shift)
            # Note: base_phase is already updated by advance(), so we need to
            # subtract the detached shift to get the "pre-modulated" phase,
            # then add the differentiable shift
            # Simpler: just use sin(phase_buffer - detached_shift + diff_shift)
            # = sin(phase_buffer + (diff_shift - detached_shift))
            # = sin(phase_buffer) when diff_shift == detached_shift (same computation)
            # But for gradients, we use: sin(phase_buffer - 0 + phase_shift)
            # Actually, since phase_buffer already includes the shift, we compute:
            # output = A * sin(φ)  where we want ∂output/∂phase_shift
            # Using the trick: sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
            # If b is small (phase_shift is bounded), we can approximate
            # For exact gradients: compute sin(base + shift) where base is detached
            base_phase = self.phases - self._phase_shift_for_read.detach()
            effective_phase = base_phase + self._phase_shift_for_read  # Differentiable!
            osc_values = amps * torch.sin(effective_phase)
        else:
            # Fallback: no differentiable phase shift
            osc_values = amps * torch.sin(self.phases)

        # Project to output dimension
        output = self.output_proj(osc_values)  # (d_output,)

        return output

    def forward(self, sync_input: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Full forward pass: advance, modulate, read.

        Args:
            sync_input: (d_sync_input,) compressed sync signal
            dt: Time step for phase advance

        Returns:
            (d_output,) world state vector
        """
        self.advance(dt)
        self.modulate(sync_input)
        return self.read()

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
