"""
Imagination Module - Mental simulation and generative representation.

Imagination generates internal representations BEYOND the input:
- Scene imagery: Constructing mental visuals from descriptions
- Mind modeling: Theory of mind - inferring others' mental states
- Counterfactuals: "What if" simulations
- Prospection: Imagining future scenarios

Unlike other PEM modules that evaluate what IS, imagination generates what ISN'T.
The imagined features are in the same space as perception features, so they can
be processed by existing modules (prediction, surprise, valence, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, NamedTuple, List
import math


@dataclass
class ImaginationConfig:
    """Configuration for the Imagination module."""

    d_model: int = 1536
    hidden_dim: Optional[int] = None  # Defaults to d_model // 2

    # Scene generation
    use_scene_generation: bool = True
    scene_layers: int = 2

    # Mind modeling (theory of mind)
    use_mind_modeling: bool = True
    mind_model_layers: int = 2
    max_entities: int = 8  # Max number of entities to model

    # Counterfactual generation
    use_counterfactuals: bool = True
    num_counterfactuals: int = 3  # Number of alternative scenarios to generate

    # General
    dropout: float = 0.0
    use_memory: bool = True  # Draw from memory for richer imagination
    use_personality: bool = True  # Personality colors imagination

    def __post_init__(self):
        if self.hidden_dim is None:
            self.hidden_dim = self.d_model // 2


class ImaginationOutput(NamedTuple):
    """Output from the Imagination module."""

    imagined_features: torch.Tensor  # (B, S, D) - generated mental imagery
    vividness: torch.Tensor  # (B, S, 1) - how vivid/clear the imagination
    mind_states: Optional[torch.Tensor] = None  # (B, S, D) - inferred mental states
    counterfactuals: Optional[torch.Tensor] = None  # (B, num_cf, S, D) - alternative scenarios
    imagination_mask: Optional[torch.Tensor] = None  # (B, S, 1) - where imagination is active


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class ImaginationTriggerDetector(nn.Module):
    """
    Detects when imagination should be triggered.

    Imagination is triggered by:
    - Explicit requests: "Imagine...", "Picture this..."
    - Scene descriptions: "The room was dark and cold"
    - Character emotions: "She felt betrayed"
    - Abstract concepts: "Justice requires sacrifice"
    - Counterfactual markers: "What if...", "If only..."

    The detector learns to identify these triggers from context.
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        self.trigger_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.trigger_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Detect imagination triggers.

        Args:
            features: (B, S, D) input features

        Returns:
            trigger_probs: (B, S, 1) probability of imagination trigger at each position
        """
        return self.trigger_net(features)


class SceneGenerator(nn.Module):
    """
    Generates mental imagery from descriptions.

    Takes text/context features and generates "imagined" visual representations.
    These are in the same feature space as actual perception, allowing them
    to be processed by downstream modules.

    "The room was dark and cold" → internal visual representation
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Encoder: context → latent imagination space
        encoder_layers = []
        in_dim = d_model
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else hidden_dim
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.GELU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Generator: latent → imagined features (same space as perception)
        generator_layers = []
        in_dim = hidden_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else d_model
            generator_layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                generator_layers.append(nn.GELU())
                if dropout > 0:
                    generator_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.generator = nn.Sequential(*generator_layers)

        # Vividness estimator: how clear/detailed is the generated imagery
        self.vividness_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.norm = RMSNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for module in [self.encoder, self.generator, self.vividness_net]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,  # (B, S, D)
        memory: Optional[torch.Tensor] = None,  # (B, S, D) memory context
        personality: Optional[torch.Tensor] = None,  # (D,) personality embedding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mental imagery from features.

        Args:
            features: Input features (descriptions, context)
            memory: Optional memory to enrich imagination
            personality: Optional personality to color imagination

        Returns:
            imagined: (B, S, D) generated imagery features
            vividness: (B, S, 1) vividness/clarity of imagination
        """
        B, S, D = features.shape

        # Combine with memory if available (richer imagination)
        if memory is not None:
            combined = features + 0.3 * memory
        else:
            combined = features

        # Encode to latent imagination space
        latent = self.encoder(combined)

        # Apply personality bias if available
        if personality is not None:
            # Project personality to latent space and use as bias
            personality_expanded = personality.unsqueeze(0).unsqueeze(0)
            # Simple modulation: scale latent by personality-derived factor
            latent = latent * (1.0 + 0.1 * torch.tanh(personality_expanded.mean(dim=-1, keepdim=True)))

        # Generate imagined features
        imagined = self.generator(latent)
        imagined = self.norm(imagined)

        # Estimate vividness
        vividness = self.vividness_net(imagined)

        return imagined, vividness


class MindModeler(nn.Module):
    """
    Theory of Mind - infers mental states of described entities.

    When reading about characters, we automatically infer:
    - What they're thinking
    - What they're feeling
    - What they believe
    - What they want

    This module models those inferred mental states.

    "She felt betrayed" → inferred mental state representation
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        n_layers: int = 2,
        max_entities: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_entities = max_entities

        # Entity detection: identify entities in the text
        self.entity_detector = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_entities),
            nn.Softmax(dim=-1),
        )

        # Mind state inference network
        # For each position, infer the mental state of the relevant entity
        mind_layers = []
        in_dim = d_model
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else d_model
            mind_layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                mind_layers.append(nn.GELU())
                if dropout > 0:
                    mind_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mind_inference = nn.Sequential(*mind_layers)

        # Mental state components: beliefs, desires, emotions, intentions
        self.belief_proj = nn.Linear(d_model, d_model // 4)
        self.desire_proj = nn.Linear(d_model, d_model // 4)
        self.emotion_proj = nn.Linear(d_model, d_model // 4)
        self.intention_proj = nn.Linear(d_model, d_model // 4)

        # Combine mental state components
        self.combine = nn.Linear(d_model, d_model)

        self.norm = RMSNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for module in [self.entity_detector, self.mind_inference]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        for proj in [self.belief_proj, self.desire_proj, self.emotion_proj, self.intention_proj, self.combine]:
            nn.init.normal_(proj.weight, std=0.02)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
        self,
        features: torch.Tensor,  # (B, S, D)
        context: Optional[torch.Tensor] = None,  # (B, S, D) additional context
    ) -> torch.Tensor:
        """
        Infer mental states of entities in the text.

        Args:
            features: Input features
            context: Optional additional context

        Returns:
            mind_states: (B, S, D) inferred mental states
        """
        B, S, D = features.shape

        # Detect which entity is relevant at each position
        entity_weights = self.entity_detector(features)  # (B, S, max_entities)

        # Infer raw mental state
        if context is not None:
            combined = features + 0.3 * context
        else:
            combined = features

        raw_mind = self.mind_inference(combined)  # (B, S, D)

        # Decompose into mental state components
        beliefs = self.belief_proj(raw_mind)  # (B, S, D//4)
        desires = self.desire_proj(raw_mind)  # (B, S, D//4)
        emotions = self.emotion_proj(raw_mind)  # (B, S, D//4)
        intentions = self.intention_proj(raw_mind)  # (B, S, D//4)

        # Combine components
        combined_mind = torch.cat([beliefs, desires, emotions, intentions], dim=-1)  # (B, S, D)
        mind_states = self.combine(combined_mind)
        mind_states = self.norm(mind_states)

        return mind_states


class CounterfactualGenerator(nn.Module):
    """
    Generates counterfactual scenarios - "what if" alternatives.

    When reading, we often consider alternatives:
    - "What if he had stayed?"
    - "If only she had known..."
    - "Things could have been different if..."

    This module generates alternative scenario representations.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_counterfactuals: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_counterfactuals = num_counterfactuals

        # Divergence point detector: where could things have gone differently?
        self.divergence_detector = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Counterfactual generators (one per alternative)
        self.cf_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, d_model),
            )
            for _ in range(num_counterfactuals)
        ])

        # Plausibility estimator: how plausible is each counterfactual?
        self.plausibility = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.norm = RMSNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for module in [self.divergence_detector, self.plausibility] + list(self.cf_generators):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,  # (B, S, D)
        context: Optional[torch.Tensor] = None,  # (B, S, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate counterfactual scenarios.

        Args:
            features: Input features (actual scenario)
            context: Optional context

        Returns:
            counterfactuals: (B, num_cf, S, D) alternative scenarios
            plausibilities: (B, num_cf, S, 1) plausibility of each
        """
        B, S, D = features.shape

        # Detect divergence points
        divergence = self.divergence_detector(features)  # (B, S, 1)

        # Generate counterfactuals
        counterfactuals = []
        plausibilities = []

        for i, generator in enumerate(self.cf_generators):
            # Generate alternative
            cf = generator(features)  # (B, S, D)
            cf = self.norm(cf)

            # Weight by divergence (alternatives matter more at divergence points)
            cf = cf * (0.5 + 0.5 * divergence)

            # Estimate plausibility
            combined = torch.cat([features, cf], dim=-1)  # (B, S, 2D)
            plaus = self.plausibility(combined)  # (B, S, 1)

            counterfactuals.append(cf)
            plausibilities.append(plaus)

        # Stack: (B, num_cf, S, D)
        counterfactuals = torch.stack(counterfactuals, dim=1)
        plausibilities = torch.stack(plausibilities, dim=1)

        return counterfactuals, plausibilities


class ImaginationIntegrator(nn.Module):
    """
    Integrates imagination components into unified imagined features.

    Combines:
    - Scene imagery
    - Mind models (theory of mind)
    - Counterfactual considerations

    Into a single imagined representation that can be processed
    by downstream modules (prediction, surprise, valence, etc.).
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        # Integration network
        # Input: scene + mind + aggregated counterfactuals + original features
        self.integrator = nn.Sequential(
            nn.Linear(d_model * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

        # Gating: how much imagination vs perception
        self.imagination_gate = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.norm = RMSNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for module in [self.integrator, self.imagination_gate]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,  # (B, S, D) original features
        scene: torch.Tensor,  # (B, S, D) scene imagery
        mind_states: Optional[torch.Tensor] = None,  # (B, S, D) theory of mind
        counterfactuals: Optional[torch.Tensor] = None,  # (B, num_cf, S, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate imagination components.

        Args:
            features: Original input features
            scene: Generated scene imagery
            mind_states: Inferred mental states
            counterfactuals: Alternative scenarios

        Returns:
            imagined: (B, S, D) integrated imagined features
            imagination_strength: (B, S, 1) how much imagination is active
        """
        B, S, D = features.shape

        # Default mind states to zeros if not provided
        if mind_states is None:
            mind_states = torch.zeros_like(features)

        # Aggregate counterfactuals (mean across alternatives)
        if counterfactuals is not None:
            cf_aggregated = counterfactuals.mean(dim=1)  # (B, S, D)
        else:
            cf_aggregated = torch.zeros_like(features)

        # Combine all components
        combined = torch.cat([features, scene, mind_states, cf_aggregated], dim=-1)

        # Integrate
        imagined = self.integrator(combined)
        imagined = self.norm(imagined)

        # Compute imagination strength (gating)
        gate_input = torch.cat([features, imagined], dim=-1)
        imagination_strength = self.imagination_gate(gate_input)

        # Blend: strong imagination trigger → more imagined, weak → more original
        # But always keep imagined features available for downstream
        final_imagined = imagination_strength * imagined + (1 - imagination_strength) * scene

        return final_imagined, imagination_strength


class ImaginationModule(nn.Module):
    """
    Complete Imagination/Mental Simulation module.

    Generates internal representations BEYOND the input:
    - Scene imagery from descriptions
    - Theory of mind for characters
    - Counterfactual alternatives
    - Prospective simulations

    The imagined features are in the same space as perception features,
    allowing them to be processed by existing PEM modules.
    """

    def __init__(self, config: ImaginationConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        # Trigger detection: when should imagination be active?
        self.trigger_detector = ImaginationTriggerDetector(
            d_model=config.d_model,
            hidden_dim=hidden_dim,
            dropout=config.dropout,
        )

        # Scene generation
        if config.use_scene_generation:
            self.scene_generator = SceneGenerator(
                d_model=config.d_model,
                hidden_dim=hidden_dim,
                n_layers=config.scene_layers,
                dropout=config.dropout,
            )
        else:
            self.scene_generator = None

        # Mind modeling (theory of mind)
        if config.use_mind_modeling:
            self.mind_modeler = MindModeler(
                d_model=config.d_model,
                hidden_dim=hidden_dim,
                n_layers=config.mind_model_layers,
                max_entities=config.max_entities,
                dropout=config.dropout,
            )
        else:
            self.mind_modeler = None

        # Counterfactual generation
        if config.use_counterfactuals:
            self.counterfactual_generator = CounterfactualGenerator(
                d_model=config.d_model,
                hidden_dim=hidden_dim,
                num_counterfactuals=config.num_counterfactuals,
                dropout=config.dropout,
            )
        else:
            self.counterfactual_generator = None

        # Integration
        self.integrator = ImaginationIntegrator(
            d_model=config.d_model,
            hidden_dim=hidden_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        features: torch.Tensor,  # (B, S, D) input features
        memory: Optional[torch.Tensor] = None,  # (B, S, D) memory context
        personality: Optional[torch.Tensor] = None,  # (D,) personality embedding
        context: Optional[torch.Tensor] = None,  # (B, S, D) additional context
    ) -> ImaginationOutput:
        """
        Generate imagination/mental simulation.

        Args:
            features: Input features to imagine from
            memory: Memory context for richer imagination
            personality: Personality to color imagination
            context: Additional context

        Returns:
            ImaginationOutput with imagined features and metadata
        """
        B, S, D = features.shape

        # 1. Detect imagination triggers
        trigger_probs = self.trigger_detector(features)  # (B, S, 1)

        # 2. Generate scene imagery
        if self.scene_generator is not None:
            scene, vividness = self.scene_generator(
                features,
                memory=memory if self.config.use_memory else None,
                personality=personality if self.config.use_personality else None,
            )
        else:
            scene = features
            vividness = torch.ones(B, S, 1, device=features.device)

        # 3. Model minds (theory of mind)
        if self.mind_modeler is not None:
            mind_states = self.mind_modeler(features, context=context)
        else:
            mind_states = None

        # 4. Generate counterfactuals
        if self.counterfactual_generator is not None:
            counterfactuals, plausibilities = self.counterfactual_generator(features, context=context)
        else:
            counterfactuals = None

        # 5. Integrate all imagination components
        imagined, imagination_strength = self.integrator(
            features=features,
            scene=scene,
            mind_states=mind_states,
            counterfactuals=counterfactuals,
        )

        # 6. Apply trigger gating (imagination only where triggered)
        # Combine trigger probability with imagination strength
        final_mask = trigger_probs * imagination_strength

        # Scale imagined features by mask (but keep them available)
        # High mask = strong imagination, low mask = weak imagination
        imagined_gated = imagined * final_mask + features * (1 - final_mask)

        return ImaginationOutput(
            imagined_features=imagined_gated,
            vividness=vividness * trigger_probs,  # Vividness weighted by trigger
            mind_states=mind_states,
            counterfactuals=counterfactuals,
            imagination_mask=final_mask,
        )

    def imagine_scenario(
        self,
        features: torch.Tensor,
        scenario_prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Explicitly imagine a specific scenario.

        Useful for prospection (imagining future scenarios)
        or explicit imagination requests.

        Args:
            features: Current features
            scenario_prompt: Optional prompt for what to imagine

        Returns:
            imagined: (B, S, D) imagined scenario features
        """
        if scenario_prompt is not None:
            combined = features + 0.5 * scenario_prompt
        else:
            combined = features

        output = self.forward(combined)
        return output.imagined_features

    def infer_mind(
        self,
        features: torch.Tensor,
        entity_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Explicitly infer mental states (theory of mind).

        Args:
            features: Features describing entity/character
            entity_context: Additional context about the entity

        Returns:
            mind_states: (B, S, D) inferred mental states
        """
        if self.mind_modeler is None:
            return torch.zeros_like(features)

        return self.mind_modeler(features, context=entity_context)


class ImaginationLoss(nn.Module):
    """
    Loss function for training the Imagination module.

    Encourages:
    1. Coherent imagination (imagined features should be internally consistent)
    2. Grounded imagination (imagination should relate to input)
    3. Diverse counterfactuals (alternatives should be different from each other)
    4. Plausible mind models (inferred mental states should be consistent)
    """

    def __init__(
        self,
        coherence_weight: float = 0.3,
        grounding_weight: float = 0.3,
        diversity_weight: float = 0.2,
        plausibility_weight: float = 0.2,
    ):
        super().__init__()
        self.coherence_weight = coherence_weight
        self.grounding_weight = grounding_weight
        self.diversity_weight = diversity_weight
        self.plausibility_weight = plausibility_weight

    def forward(
        self,
        imagination_output: ImaginationOutput,
        original_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute imagination loss.

        Args:
            imagination_output: Output from ImaginationModule
            original_features: Original input features

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        imagined = imagination_output.imagined_features
        losses = {}

        # 1. Coherence: imagined features should be smooth across positions
        if imagined.shape[1] > 1:
            diff = imagined[:, 1:] - imagined[:, :-1]
            coherence_loss = (diff ** 2).mean()
        else:
            coherence_loss = torch.tensor(0.0, device=imagined.device)
        losses["coherence_loss"] = coherence_loss

        # 2. Grounding: imagination should relate to input (not random)
        # Cosine similarity between imagined and original
        cos_sim = F.cosine_similarity(
            imagined.view(-1, imagined.shape[-1]),
            original_features.view(-1, original_features.shape[-1]),
            dim=-1
        ).mean()
        grounding_loss = 1 - cos_sim  # Penalize low similarity
        losses["grounding_loss"] = grounding_loss

        # 3. Diversity: counterfactuals should be different from each other
        if imagination_output.counterfactuals is not None:
            cfs = imagination_output.counterfactuals  # (B, num_cf, S, D)
            num_cf = cfs.shape[1]
            if num_cf > 1:
                diversity_loss = 0.0
                count = 0
                for i in range(num_cf):
                    for j in range(i + 1, num_cf):
                        sim = F.cosine_similarity(
                            cfs[:, i].reshape(-1, cfs.shape[-1]),
                            cfs[:, j].reshape(-1, cfs.shape[-1]),
                            dim=-1
                        ).mean()
                        diversity_loss += sim  # Penalize high similarity
                        count += 1
                diversity_loss = diversity_loss / count if count > 0 else torch.tensor(0.0, device=imagined.device)
            else:
                diversity_loss = torch.tensor(0.0, device=imagined.device)
        else:
            diversity_loss = torch.tensor(0.0, device=imagined.device)
        losses["diversity_loss"] = diversity_loss

        # 4. Plausibility: vividness should be moderate (not always 0 or 1)
        vividness = imagination_output.vividness
        # Encourage vividness to use full range
        plausibility_loss = (vividness.mean() - 0.5) ** 2 + F.relu(0.1 - vividness.std())
        losses["plausibility_loss"] = plausibility_loss

        # Total loss
        total_loss = (
            self.coherence_weight * coherence_loss
            + self.grounding_weight * grounding_loss
            + self.diversity_weight * diversity_loss
            + self.plausibility_weight * plausibility_loss
        )

        return total_loss, losses


def create_imagination_module(
    d_model: int = 1536,
    hidden_dim: Optional[int] = None,
    use_scene_generation: bool = True,
    use_mind_modeling: bool = True,
    use_counterfactuals: bool = True,
    **kwargs,
) -> ImaginationModule:
    """
    Factory function to create an ImaginationModule.

    Args:
        d_model: Model dimension
        hidden_dim: Hidden layer dimension
        use_scene_generation: Enable scene imagery generation
        use_mind_modeling: Enable theory of mind
        use_counterfactuals: Enable counterfactual generation
        **kwargs: Additional config parameters

    Returns:
        Configured ImaginationModule
    """
    config = ImaginationConfig(
        d_model=d_model,
        hidden_dim=hidden_dim,
        use_scene_generation=use_scene_generation,
        use_mind_modeling=use_mind_modeling,
        use_counterfactuals=use_counterfactuals,
        **kwargs,
    )
    return ImaginationModule(config)
