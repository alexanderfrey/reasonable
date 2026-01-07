"""
SelfState — Latent Internal State with Bidirectional Textual Interface

Implements an "inner state" (soma) that:
1. Accumulates from sub-signals (surprise, arousal, valence, novelty, certainty)
2. Feeds back to modulate prediction/attention
3. Has a self-model that predicts own reactions
4. Has bidirectional text interface:
   - describe_self: soma → text (lossy projection for inspection)
   - interpret_self: text → soma update (narrative self-modification)
5. Maintains actual vs ideal self with discrepancy signal
6. Tracks temperament (slow-moving personality baseline)

The soma is the PRIMARY representation. Text is secondary but bidirectional —
reading your own description can modify your state.

Architecture:
                    ┌─────────────────────────────────┐
                    │         SELF-MODEL              │
                    │  "How I expect to react"        │
                    │  predicted_soma(t+1|input)      │
                    └───────────┬─────────────────────┘
                                │ prediction
                                ▼
┌─────────────┐    ┌─────────────────────────────────┐
│   Input     │───▶│           SOMA                  │
│   signals   │    │  Integrated internal state      │
└─────────────┘    │  [surprise, arousal, valence,   │
                   │   novelty, certainty, ...]      │
                   └───────────┬─────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        describe_self    update self-model  interpret_self
        (soma → text)    (who I am)         (text → soma)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class SelfStateConfig:
    """Configuration for SelfState module."""
    d_model: int  # Model hidden dimension
    d_soma: int = 64  # Soma (internal state) dimension

    # Sub-signal dimensions (inputs to soma integration)
    n_signals: int = 6  # surprise, arousal, valence, novelty, certainty, engagement

    # Decay rates for temporal integration
    soma_decay: float = 0.9  # How much soma persists across steps (mood inertia)
    temperament_decay: float = 0.999  # Very slow baseline drift (personality)

    # Self-model configuration
    self_model_hidden_mult: int = 2

    # Text interface configuration
    vocab_size: int = 50257  # GPT-2 default, will be set from model
    description_max_tokens: int = 32  # Max tokens for self-description
    narrative_gain: float = 0.1  # How much self-interpretation affects soma

    # Ideal self configuration
    use_ideal_self: bool = True
    ideal_self_dim: int = 64  # Same as soma by default
    self_discrepancy_weight: float = 0.5  # Weight for actual-ideal discrepancy

    # Feedback modulation
    soma_to_attention: bool = False  # Whether soma modulates attention Q vectors
    n_head: int = 8  # Number of attention heads (for Q bias shape)
    head_dim: int = 64  # Dimension per head (for Q bias shape)
    soma_attention_scale: float = 0.1  # Scale factor for Q bias (small = subtle)
    soma_to_logits: bool = False  # Whether soma biases output logits
    soma_gate_hidden: bool = True  # Gate hidden states by soma

    # Grounding configuration
    ground_signals: List[str] = field(default_factory=lambda: [
        'surprise', 'arousal', 'valence', 'novelty', 'certainty', 'engagement'
    ])


class SomaIntegrator(nn.Module):
    """
    Integrates moment-to-moment signals into persistent soma state.

    Soma has inertia — it doesn't flip instantly. A system in a "curious"
    state stays curious across multiple inputs. This is mood vs. emotion.

    soma(t) = decay * soma(t-1) + (1-decay) * integrate(signals)
    """

    def __init__(self, config: SelfStateConfig):
        super().__init__()
        self.config = config
        self.d_soma = config.d_soma
        self.n_signals = config.n_signals
        self.decay = config.soma_decay

        # Signal integration: map raw signals to soma space
        # Signals: [surprise, arousal, valence, novelty, certainty, engagement]
        self.signal_proj = nn.Sequential(
            nn.Linear(config.n_signals, config.d_soma * 2),
            nn.GELU(),
            nn.Linear(config.d_soma * 2, config.d_soma),
            nn.Tanh()  # Bound integrated signals
        )

        # Context-dependent integration: how much each signal matters
        # depends on current soma state
        self.signal_gate = nn.Sequential(
            nn.Linear(config.d_soma + config.n_signals, config.n_signals),
            nn.Sigmoid()
        )

        # Soma state (not a parameter, a buffer that persists)
        self.register_buffer('_soma', None)
        self.register_buffer('_temperament', None)
        self._batch_size = None

    def _init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize soma to zeros (neutral state)."""
        return torch.zeros(batch_size, self.d_soma, device=device)

    def _get_or_init_soma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get current soma or initialize if needed."""
        if self._soma is None or self._batch_size != batch_size:
            self._soma = self._init_state(batch_size, device)
            self._batch_size = batch_size
        return self._soma.to(device)

    def _get_or_init_temperament(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get temperament baseline or initialize."""
        if self._temperament is None or self._temperament.size(0) != batch_size:
            self._temperament = self._init_state(batch_size, device)
        return self._temperament.to(device)

    def forward(
        self,
        signals: torch.Tensor,  # [B, n_signals]
        update_state: bool = True,
        certainty: Optional[torch.Tensor] = None,  # [B] for certainty-modulated updates
    ) -> Dict[str, torch.Tensor]:
        """
        Integrate signals into soma state.

        Args:
            signals: [B, n_signals] raw sub-signals
                     (surprise, arousal, valence, novelty, certainty, engagement)
            update_state: whether to update persistent state
            certainty: [B] optional certainty for modulating update strength.
                       When uncertain, be more conservative in state updates.

        Returns:
            dict with:
                - soma: [B, d_soma] current integrated state
                - soma_delta: [B, d_soma] change from previous
                - temperament: [B, d_soma] slow-moving baseline
                - soma_relative: [B, d_soma] soma minus temperament
        """
        batch_size = signals.size(0)
        device = signals.device

        prev_soma = self._get_or_init_soma(batch_size, device)
        temperament = self._get_or_init_temperament(batch_size, device)

        # Compute signal gating: which signals matter given current state
        gate_input = torch.cat([prev_soma.detach(), signals], dim=-1)
        signal_weights = self.signal_gate(gate_input)  # [B, n_signals]

        # Weight signals by their importance
        weighted_signals = signals * signal_weights  # [B, n_signals]

        # Project to soma space
        signal_contribution = self.signal_proj(weighted_signals)  # [B, d_soma]

        # Certainty-modulated integration: when uncertain, be more conservative
        # This prevents uncertain signals from causing large state changes
        if certainty is not None:
            # certainty_gate ∈ [0.3, 1.0] - never fully stop updates
            certainty_gate = 0.3 + 0.7 * certainty.unsqueeze(-1)  # [B, 1]
            signal_contribution = signal_contribution * certainty_gate

        # Integrate with decay (EMA-style)
        new_soma = self.decay * prev_soma + (1 - self.decay) * signal_contribution

        # Compute delta for self-model training
        soma_delta = new_soma - prev_soma

        if update_state:
            self._soma = new_soma.detach()
            # Update temperament with very slow decay
            self._temperament = (
                self.config.temperament_decay * temperament +
                (1 - self.config.temperament_decay) * new_soma.detach()
            )

        # Relative soma: deviation from baseline temperament
        soma_relative = new_soma - temperament

        return {
            'soma': new_soma,
            'soma_delta': soma_delta,
            'temperament': temperament,
            'soma_relative': soma_relative,
            'signal_weights': signal_weights,
        }

    def reset_state(self):
        """Reset soma to neutral (e.g., between documents)."""
        self._soma = None
        # Note: temperament persists across resets

    def reset_all(self):
        """Reset both soma and temperament."""
        self._soma = None
        self._temperament = None
        self._batch_size = None


class SelfModel(nn.Module):
    """
    Predicts how soma will change given current state and input.

    This is the "expectation of how I will react." The self-model is a
    learned representation of "given who I am now and what I'm about to
    experience, here's how I'll feel."

    Self-surprise (mismatch between predicted and actual delta) is meaningful:
    - High self-surprise → "I didn't expect to react this way" → update self-model
    - Low self-surprise → "I know myself well" → stable identity
    """

    def __init__(self, config: SelfStateConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.d_soma * config.self_model_hidden_mult

        # Predict soma change from current soma + input embedding
        # Input: [soma, input_embedding] → predicted_delta_soma
        self.predictor = nn.Sequential(
            nn.Linear(config.d_soma + config.d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.d_soma),
            nn.Tanh()  # Bound predictions
        )

        # Confidence head: how certain is the self-model about its prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_soma + config.d_model, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize to predict small changes with medium confidence."""
        # Small initial weights for predictor output
        nn.init.xavier_uniform_(self.predictor[-2].weight, gain=0.1)
        nn.init.zeros_(self.predictor[-2].bias)
        # Confidence starts at ~0.5
        nn.init.zeros_(self.confidence_head[-2].weight)
        nn.init.zeros_(self.confidence_head[-2].bias)

    def forward(
        self,
        soma: torch.Tensor,  # [B, d_soma]
        input_embedding: torch.Tensor,  # [B, d_model]
    ) -> Dict[str, torch.Tensor]:
        """
        Predict how soma will change.

        Returns:
            dict with:
                - predicted_delta: [B, d_soma] expected change
                - confidence: [B, 1] how confident in prediction
        """
        combined = torch.cat([soma, input_embedding], dim=-1)

        predicted_delta = self.predictor(combined)
        confidence = self.confidence_head(combined)

        return {
            'predicted_delta': predicted_delta,
            'confidence': confidence,
        }

    def compute_self_surprise(
        self,
        predicted_delta: torch.Tensor,
        actual_delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute self-surprise: how wrong was the self-model.

        Returns:
            self_surprise: [B] scalar measure of self-surprise per sample
        """
        # L2 distance between predicted and actual change
        diff = predicted_delta - actual_delta
        self_surprise = torch.norm(diff, dim=-1)  # [B]
        return self_surprise


class SelfDescriptionHead(nn.Module):
    """
    Projects soma state to text (lossy projection for inspection).

    The text description is NOT the soma — it's a readable summary.
    Like how we can describe an emotion without the description being the emotion.

    Can generate autoregressive descriptions seeded by soma state.
    """

    def __init__(self, config: SelfStateConfig):
        super().__init__()
        self.config = config

        # Soma → initial hidden state for description generation
        self.soma_to_hidden = nn.Sequential(
            nn.Linear(config.d_soma, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Simple MLP for next-token prediction from hidden state
        # (In practice, this would use the GPT's own decoder)
        self.description_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.vocab_size),
        )

        # Optional: template-based description (structured output)
        # Predicts values for template: "I feel [INTENSITY] [VALENCE],
        # with [AROUSAL] arousal and [CERTAINTY] certainty."
        self.template_values = nn.Sequential(
            nn.Linear(config.d_soma, config.d_soma),
            nn.GELU(),
            nn.Linear(config.d_soma, 5),  # intensity, valence, arousal, certainty, engagement
        )

    def forward(
        self,
        soma: torch.Tensor,  # [B, d_soma]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate description from soma.

        Returns:
            dict with:
                - hidden_seed: [B, d_model] seed for autoregressive generation
                - template_values: [B, 5] structured values for template
                - logits: [B, vocab_size] next-token logits (for first token)
        """
        hidden_seed = self.soma_to_hidden(soma)
        logits = self.description_head(hidden_seed)
        template_values = self.template_values(soma)

        return {
            'hidden_seed': hidden_seed,
            'template_values': template_values,
            'logits': logits,
        }

    def generate_description_tokens(
        self,
        soma: torch.Tensor,
        max_tokens: int = 32,
        temperature: float = 0.7,
    ) -> torch.Tensor:
        """
        Generate description tokens autoregressively.

        Note: This is a simplified version. In practice, you'd use the
        main GPT model conditioned on soma for generation.

        Returns:
            tokens: [B, max_tokens] generated token ids
        """
        batch_size = soma.size(0)
        device = soma.device

        hidden = self.soma_to_hidden(soma)  # [B, d_model]
        tokens = []

        for _ in range(max_tokens):
            logits = self.description_head(hidden)  # [B, vocab_size]

            # Sample next token
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)  # [B, 1]
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            tokens.append(next_token)

            # Update hidden (simplified — real version would use transformer)
            # This is a placeholder; actual implementation uses the GPT
            hidden = hidden + 0.1 * torch.randn_like(hidden)

        return torch.cat(tokens, dim=-1)  # [B, max_tokens]


class SelfInterpretationHead(nn.Module):
    """
    Reads text and updates soma state (narrative self-modification).

    Reading your own description affects your state. This creates
    narrative self-modification: articulating "I'm anxious" can either
    reduce anxiety (naming tames) or amplify it (rumination).

    The sign and magnitude of the effect is learned.
    """

    def __init__(self, config: SelfStateConfig):
        super().__init__()
        self.config = config
        self.narrative_gain = config.narrative_gain

        # Text embedding → soma delta
        # Input: text embedding (from GPT) of self-description
        # Output: delta to add to soma
        self.interpreter = nn.Sequential(
            nn.Linear(config.d_model, config.d_soma * 2),
            nn.GELU(),
            nn.LayerNorm(config.d_soma * 2),
            nn.Linear(config.d_soma * 2, config.d_soma),
            nn.Tanh()  # Bound the effect
        )

        # Valence of interpretation: does reading this help or hurt?
        # Positive = calming/integrating, Negative = amplifying/ruminating
        self.interpretation_valence = nn.Sequential(
            nn.Linear(config.d_model + config.d_soma, config.d_soma),
            nn.GELU(),
            nn.Linear(config.d_soma, 1),
            nn.Tanh()  # [-1, 1]: negative = rumination, positive = naming tames
        )

    def forward(
        self,
        text_embedding: torch.Tensor,  # [B, d_model]
        current_soma: torch.Tensor,  # [B, d_soma]
    ) -> Dict[str, torch.Tensor]:
        """
        Interpret text and compute soma update.

        Returns:
            dict with:
                - soma_delta: [B, d_soma] update to apply to soma
                - interpretation_valence: [B, 1] whether reading helps or hurts
                - updated_soma: [B, d_soma] soma after narrative modification
        """
        # Compute raw interpretation
        raw_delta = self.interpreter(text_embedding)

        # Compute valence: does reading this help or amplify?
        combined = torch.cat([text_embedding, current_soma], dim=-1)
        valence = self.interpretation_valence(combined)  # [B, 1]

        # Apply narrative gain with valence-dependent sign
        # Positive valence → delta moves toward neutral (calming)
        # Negative valence → delta amplifies current state (rumination)
        effective_delta = self.narrative_gain * raw_delta * valence

        # Compute updated soma
        updated_soma = current_soma + effective_delta

        return {
            'soma_delta': effective_delta,
            'interpretation_valence': valence,
            'updated_soma': updated_soma,
        }


class IdealSelf(nn.Module):
    """
    Represents the "desired self" — an aspirational state.

    The tension between actual and ideal self becomes a learning signal,
    following self-discrepancy theory in psychology.

    The ideal self can be:
    - Learned from positive experiences (what states led to good outcomes)
    - Specified externally (alignment/values)
    - Derived from the temperament (who I have been)
    """

    def __init__(self, config: SelfStateConfig):
        super().__init__()
        self.config = config

        # Ideal self as a learnable parameter
        # This represents "who I want to be"
        self.ideal_state = nn.Parameter(torch.zeros(config.ideal_self_dim))

        # Project soma to ideal-self space for comparison
        if config.d_soma != config.ideal_self_dim:
            self.soma_to_ideal_space = nn.Linear(config.d_soma, config.ideal_self_dim)
        else:
            self.soma_to_ideal_space = nn.Identity()

        # Context-dependent ideal: ideal self may vary by situation
        self.context_modulator = nn.Sequential(
            nn.Linear(config.d_model, config.ideal_self_dim),
            nn.Tanh()
        )

        # Discrepancy → regulation signal
        self.regulation_head = nn.Sequential(
            nn.Linear(config.ideal_self_dim, config.d_soma),
            nn.Tanh()
        )

    def forward(
        self,
        soma: torch.Tensor,  # [B, d_soma]
        context: Optional[torch.Tensor] = None,  # [B, d_model]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute actual vs ideal self discrepancy.

        Returns:
            dict with:
                - ideal_self: [B, ideal_dim] contextualized ideal state
                - actual_in_ideal_space: [B, ideal_dim] soma projected
                - discrepancy: [B, ideal_dim] ideal - actual
                - discrepancy_magnitude: [B] scalar discrepancy
                - regulation_signal: [B, d_soma] signal to move toward ideal
        """
        batch_size = soma.size(0)
        device = soma.device

        # Base ideal state (broadcast to batch)
        ideal = self.ideal_state.unsqueeze(0).expand(batch_size, -1)

        # Contextualize if context provided
        if context is not None:
            context_mod = self.context_modulator(context)
            ideal = ideal + 0.3 * context_mod  # Slight context-dependent shift

        # Project soma to ideal space
        actual = self.soma_to_ideal_space(soma)

        # Compute discrepancy
        discrepancy = ideal - actual
        discrepancy_magnitude = torch.norm(discrepancy, dim=-1)  # [B]

        # Compute regulation signal: how to move toward ideal
        regulation_signal = self.regulation_head(discrepancy)

        return {
            'ideal_self': ideal,
            'actual_in_ideal_space': actual,
            'discrepancy': discrepancy,
            'discrepancy_magnitude': discrepancy_magnitude,
            'regulation_signal': regulation_signal,
        }


class SomaFeedback(nn.Module):
    """
    Feeds soma state back into the model's processing.

    The soma modulates:
    - Hidden states (gating) — post-attention
    - Attention Q vectors (per-head bias) — shifts what each head "looks for"
    - Output logits (prediction bias) — optional

    This creates the feedback loop where internal state shapes perception.
    """

    def __init__(self, config: SelfStateConfig):
        super().__init__()
        self.config = config

        # Soma → hidden state gate
        if config.soma_gate_hidden:
            self.hidden_gate = nn.Sequential(
                nn.Linear(config.d_soma, config.d_model),
                nn.Sigmoid()
            )
            # Additive modulation (residual)
            self.hidden_add = nn.Sequential(
                nn.Linear(config.d_soma, config.d_model),
                nn.Tanh()
            )

        # Soma → attention Q bias (shifts what each head "looks for")
        # Output shape: [B, n_head, head_dim]
        if config.soma_to_attention:
            q_bias_dim = config.n_head * config.head_dim
            self.q_bias_proj = nn.Sequential(
                nn.Linear(config.d_soma, config.d_soma * 2),
                nn.GELU(),
                nn.Linear(config.d_soma * 2, q_bias_dim),
                nn.Tanh()  # Bound the bias
            )
            self.n_head = config.n_head
            self.head_dim = config.head_dim
            self.attention_scale = config.soma_attention_scale

            # Initialize to produce near-zero bias initially
            nn.init.zeros_(self.q_bias_proj[-2].weight)
            nn.init.zeros_(self.q_bias_proj[-2].bias)

        # Soma → logit bias (optional)
        if config.soma_to_logits:
            self.logit_bias = nn.Sequential(
                nn.Linear(config.d_soma, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.vocab_size),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, seq_len, d_model]
        soma: torch.Tensor,  # [B, d_soma]
        certainty: Optional[torch.Tensor] = None,  # [B] for certainty-scaled feedback
    ) -> Dict[str, torch.Tensor]:
        """
        Modulate hidden states by soma.

        Args:
            hidden_states: [B, seq_len, d_model] model hidden states
            soma: [B, d_soma] internal state
            certainty: [B] optional certainty for scaling feedback strength.
                       When uncertain, reduce soma's influence on processing.

        Returns:
            dict with:
                - modulated_hidden: [B, seq_len, d_model]
                - q_bias: [B, n_head, head_dim] if soma_to_attention enabled
                - logit_bias: [B, vocab_size] if enabled
        """
        result = {}

        # Hidden state modulation
        if self.config.soma_gate_hidden:
            gate = self.hidden_gate(soma).unsqueeze(1)  # [B, 1, d_model]
            add = self.hidden_add(soma).unsqueeze(1)  # [B, 1, d_model]

            # Certainty-scaled feedback: when uncertain, let hidden states flow more freely
            # This prevents uncertain internal state from corrupting processing
            if certainty is not None:
                # feedback_strength ∈ [0.2, 1.0]
                feedback_strength = 0.2 + 0.8 * certainty.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

                # Blend gate toward 1 (no gating) when uncertain
                gate = 1 - feedback_strength * (1 - gate)
                # Reduce additive bias when uncertain
                add = add * feedback_strength

            modulated = hidden_states * gate + 0.1 * add
            result['modulated_hidden'] = modulated
        else:
            result['modulated_hidden'] = hidden_states

        # Attention Q bias
        if self.config.soma_to_attention:
            # Project soma to Q bias shape
            q_bias_flat = self.q_bias_proj(soma)  # [B, n_head * head_dim]
            q_bias = q_bias_flat.view(-1, self.n_head, self.head_dim)  # [B, n_head, head_dim]
            # Apply scale factor (small = subtle modulation)
            q_bias = self.attention_scale * q_bias
            result['q_bias'] = q_bias

        # Logit bias
        if self.config.soma_to_logits:
            result['logit_bias'] = self.logit_bias(soma)

        return result


class SelfState(nn.Module):
    """
    Complete self-state system with latent soma and bidirectional text interface.

    Primary: Latent soma that integrates sub-signals and has inertia
    Secondary: Text description and interpretation (bidirectional)

    Components:
    - SomaIntegrator: accumulates signals into persistent state
    - SelfModel: predicts own reactions (self-awareness)
    - SelfDescriptionHead: soma → text (lossy projection)
    - SelfInterpretationHead: text → soma update (narrative modification)
    - IdealSelf: aspirational state with discrepancy signal
    - SomaFeedback: modulates model processing
    """

    def __init__(self, config: SelfStateConfig):
        super().__init__()
        self.config = config

        # Core components
        self.integrator = SomaIntegrator(config)
        self.self_model = SelfModel(config)

        # Text interface (bidirectional)
        self.describer = SelfDescriptionHead(config)
        self.interpreter = SelfInterpretationHead(config)

        # Ideal self
        if config.use_ideal_self:
            self.ideal_self = IdealSelf(config)
        else:
            self.ideal_self = None

        # Feedback modulation
        self.feedback = SomaFeedback(config)

        # Signal extraction: convert experiential outputs to signal vector
        self.signal_names = config.ground_signals

    def extract_signals(
        self,
        exp_output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Extract signal vector from experiential stream output.

        Expected keys in exp_output:
        - surprise: [B] prediction error signal
        - arousal: [B] activation level
        - valence: [B] emotional valence
        - novelty (optional): [B] novelty vs memories
        - (certainty computed from surprise)
        - (engagement from attention metrics)

        Returns:
            signals: [B, n_signals]
        """
        batch_size = exp_output['surprise'].size(0)
        device = exp_output['surprise'].device

        signals = []

        # Surprise (required)
        surprise = exp_output['surprise']
        if surprise.dim() == 0:
            surprise = surprise.unsqueeze(0).expand(batch_size)
        signals.append(surprise.unsqueeze(-1))

        # Arousal (required)
        arousal = exp_output.get('arousal', torch.zeros(batch_size, device=device))
        if arousal.dim() == 0:
            arousal = arousal.unsqueeze(0).expand(batch_size)
        signals.append(arousal.unsqueeze(-1))

        # Valence (required)
        valence = exp_output.get('valence', torch.zeros(batch_size, device=device))
        if valence.dim() == 0:
            valence = valence.unsqueeze(0).expand(batch_size)
        signals.append(valence.unsqueeze(-1))

        # Novelty (optional, from surprise_t if available)
        novelty = exp_output.get('novelty_t', None)
        if novelty is not None:
            # Aggregate over sequence
            novelty = novelty.mean(dim=-1) if novelty.dim() > 1 else novelty
        else:
            novelty = torch.zeros(batch_size, device=device)
        signals.append(novelty.unsqueeze(-1))

        # Certainty: use unified certainty from CertaintyHead if available,
        # otherwise fall back to simple inverse of surprise
        if 'certainty' in exp_output and exp_output['certainty'] is not None:
            certainty = exp_output['certainty']
            if certainty.dim() == 0:
                certainty = certainty.unsqueeze(0).expand(batch_size)
        else:
            # Fallback: simple inverse of surprise
            certainty = 1.0 - torch.clamp(surprise, 0, 1)
        signals.append(certainty.unsqueeze(-1))

        # Engagement (could be attention entropy, for now proxy from arousal)
        engagement = arousal  # Placeholder
        signals.append(engagement.unsqueeze(-1))

        return torch.cat(signals, dim=-1)  # [B, n_signals]

    def forward(
        self,
        exp_output: Dict[str, torch.Tensor],
        hidden_states: torch.Tensor,  # [B, seq_len, d_model]
        input_embedding: Optional[torch.Tensor] = None,  # [B, d_model] for self-model
        self_description_embedding: Optional[torch.Tensor] = None,  # For narrative loop
        update_state: bool = True,
        compute_description: bool = False,
    ) -> Dict[str, Any]:
        """
        Full forward pass through self-state system.

        Args:
            exp_output: dict from ExperientialStream with surprise, arousal, valence
            hidden_states: [B, seq_len, d_model] current hidden states
            input_embedding: [B, d_model] input for self-model prediction
            self_description_embedding: [B, d_model] embedding of self-description
                                        (for narrative self-modification)
            update_state: whether to update persistent state
            compute_description: whether to compute text description

        Returns:
            dict with:
                - soma: current integrated state
                - soma_delta: change from previous
                - soma_relative: deviation from temperament
                - temperament: slow-moving baseline
                - predicted_delta: self-model prediction
                - self_surprise: |predicted - actual| delta
                - modulated_hidden: hidden states after soma feedback
                - description_logits: logits for self-description (if compute_description)
                - ideal_discrepancy: actual-ideal gap (if use_ideal_self)
                - regulation_signal: signal toward ideal (if use_ideal_self)
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device

        # 1. Extract signals from experiential output
        signals = self.extract_signals(exp_output)

        # Extract certainty for modulation (unified certainty from CertaintyHead)
        certainty = exp_output.get('certainty', None)

        # 2. Integrate signals into soma (with certainty modulation)
        integrator_out = self.integrator(signals, update_state=update_state, certainty=certainty)
        soma = integrator_out['soma']
        soma_delta = integrator_out['soma_delta']

        # 3. Self-model prediction (if input embedding provided)
        predicted_delta = None
        self_surprise = None
        self_confidence = None
        if input_embedding is not None:
            # Get previous soma for prediction (before this step's update)
            prev_soma = soma - soma_delta  # Reconstruct previous

            self_model_out = self.self_model(prev_soma.detach(), input_embedding)
            predicted_delta = self_model_out['predicted_delta']
            self_confidence = self_model_out['confidence']

            # Compute self-surprise
            self_surprise = self.self_model.compute_self_surprise(
                predicted_delta, soma_delta.detach()
            )

        # 4. Narrative self-modification (if description embedding provided)
        narrative_delta = None
        interpretation_valence = None
        if self_description_embedding is not None:
            interpret_out = self.interpreter(self_description_embedding, soma)
            narrative_delta = interpret_out['soma_delta']
            interpretation_valence = interpret_out['interpretation_valence']

            # Apply narrative modification to soma
            if update_state:
                soma = interpret_out['updated_soma']
                self.integrator._soma = soma.detach()

        # 5. Ideal self discrepancy (if enabled)
        ideal_out = None
        if self.ideal_self is not None:
            # Use endpoint hidden state as context
            context = hidden_states[:, -1, :]
            ideal_out = self.ideal_self(soma, context)

        # 6. Feedback modulation of hidden states (with certainty scaling)
        feedback_out = self.feedback(hidden_states, soma, certainty=certainty)
        modulated_hidden = feedback_out['modulated_hidden']

        # 7. Generate description (if requested)
        description_out = None
        if compute_description:
            description_out = self.describer(soma)

        # Compile output
        result = {
            # Core soma state
            'soma': soma,
            'soma_delta': soma_delta,
            'soma_relative': integrator_out['soma_relative'],
            'temperament': integrator_out['temperament'],
            'signal_weights': integrator_out['signal_weights'],

            # Self-model
            'predicted_delta': predicted_delta,
            'self_surprise': self_surprise,
            'self_confidence': self_confidence,

            # Narrative modification
            'narrative_delta': narrative_delta,
            'interpretation_valence': interpretation_valence,

            # Feedback
            'modulated_hidden': modulated_hidden,
            'attention_bias': feedback_out.get('attention_bias'),
            'logit_bias': feedback_out.get('logit_bias'),
        }

        # Ideal self
        if ideal_out is not None:
            result.update({
                'ideal_self': ideal_out['ideal_self'],
                'actual_in_ideal_space': ideal_out['actual_in_ideal_space'],
                'ideal_discrepancy': ideal_out['discrepancy'],
                'ideal_discrepancy_magnitude': ideal_out['discrepancy_magnitude'],
                'regulation_signal': ideal_out['regulation_signal'],
            })

        # Description
        if description_out is not None:
            result.update({
                'description_logits': description_out['logits'],
                'description_hidden_seed': description_out['hidden_seed'],
                'description_template_values': description_out['template_values'],
            })

        return result

    def get_soma(self) -> Optional[torch.Tensor]:
        """Get current soma state."""
        return self.integrator._soma

    def get_temperament(self) -> Optional[torch.Tensor]:
        """Get current temperament baseline."""
        return self.integrator._temperament

    def reset_soma(self):
        """Reset soma to neutral (between documents)."""
        self.integrator.reset_state()

    def reset_all(self):
        """Reset all state (soma + temperament)."""
        self.integrator.reset_all()

    def snapshot(self) -> Dict[str, Any]:
        """Capture state for checkpointing."""
        state = {
            'soma': self.integrator._soma.detach().cpu() if self.integrator._soma is not None else None,
            'temperament': self.integrator._temperament.detach().cpu() if self.integrator._temperament is not None else None,
            'batch_size': self.integrator._batch_size,
        }
        if self.ideal_self is not None:
            state['ideal_state'] = self.ideal_self.ideal_state.detach().cpu()
        return state

    def restore(self, snapshot: Dict[str, Any], device: Optional[torch.device] = None):
        """Restore state from checkpoint."""
        if snapshot.get('soma') is not None:
            soma = snapshot['soma']
            if device is not None:
                soma = soma.to(device)
            self.integrator._soma = soma
        if snapshot.get('temperament') is not None:
            temp = snapshot['temperament']
            if device is not None:
                temp = temp.to(device)
            self.integrator._temperament = temp
        self.integrator._batch_size = snapshot.get('batch_size')
        if self.ideal_self is not None and 'ideal_state' in snapshot:
            ideal = snapshot['ideal_state']
            if device is not None:
                ideal = ideal.to(device)
            self.ideal_self.ideal_state.data.copy_(ideal)


def compute_self_state_losses(
    self_state_output: Dict[str, torch.Tensor],
    config: SelfStateConfig,
) -> Dict[str, torch.Tensor]:
    """
    Compute training losses for self-state system.

    Losses:
    1. Self-model loss: minimize self-surprise (predict own reactions well)
    2. Ideal discrepancy loss: minimize gap to ideal self (move toward desired state)
    3. Interpretation consistency: description-interpretation cycle should be stable

    Returns:
        dict with individual losses and combined loss
    """
    losses = {}
    total_loss = 0.0

    # 1. Self-model loss: predict own soma changes well
    if self_state_output.get('self_surprise') is not None:
        # Weight by inverse confidence: harder when confident but wrong
        self_surprise = self_state_output['self_surprise']
        confidence = self_state_output.get('self_confidence', torch.ones_like(self_surprise.unsqueeze(-1)))

        # Higher loss when confident but wrong
        weighted_surprise = self_surprise * (0.5 + confidence.squeeze(-1))
        self_model_loss = weighted_surprise.mean()

        losses['self_model'] = self_model_loss
        total_loss = total_loss + self_model_loss

    # 2. Ideal discrepancy loss: encourage movement toward ideal
    if self_state_output.get('ideal_discrepancy_magnitude') is not None:
        discrepancy = self_state_output['ideal_discrepancy_magnitude']
        ideal_loss = config.self_discrepancy_weight * discrepancy.mean()

        losses['ideal_discrepancy'] = ideal_loss
        total_loss = total_loss + ideal_loss

    # 3. Regulation signal should be acted upon
    # (This encourages the system to actually move toward ideal)
    if self_state_output.get('regulation_signal') is not None:
        reg = self_state_output['regulation_signal']
        delta = self_state_output.get('soma_delta', torch.zeros_like(reg))

        # Delta should align with regulation signal
        alignment = F.cosine_similarity(reg, delta, dim=-1)
        regulation_loss = (1 - alignment).mean() * 0.1

        losses['regulation_alignment'] = regulation_loss
        total_loss = total_loss + regulation_loss

    losses['total'] = total_loss
    return losses
