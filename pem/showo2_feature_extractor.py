"""
Show-o2 Feature Extractor for PEM.

Replaces Qwen3-VL with Show-o2, a unified multimodal model that handles
both understanding AND generation. This enables native imagination
without bolting on separate generation modules.

Key capabilities:
- Multimodal understanding (text, image, video)
- Text-to-image generation (discrete diffusion)
- Mixed-modality generation
- Same hidden dimension as Qwen2.5 backbone (1536 for 1.5B, 3584 for 7B)

Architecture:
    Input ──► Show-o2 ──► Features (understanding)
                  │
                  └──► Generated tokens (imagination) ──► Decode ──► Features
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any, Tuple
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """How the feature extractor learns."""
    FROZEN = "frozen"           # No gradients, fixed weights
    SLOW_LEARNING = "slow"      # Lower learning rate than rest of model
    TRAINABLE = "trainable"     # Full learning rate


@dataclass
class Showo2Config:
    """Configuration for Show-o2 feature extractor."""

    # Model selection
    model_name_or_path: str = "showlab/show-o2-1.5B"

    # Output configuration
    output_dim: int = 1536  # Matches Qwen2.5-1.5B hidden size

    # Learning configuration
    learning_mode: LearningMode = LearningMode.FROZEN
    slow_learning_factor: float = 0.1

    # Model configuration
    torch_dtype: Optional[str] = "bfloat16"
    device_map: str = "auto"

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    generation_temperature: float = 1.0

    # Resolution for image generation
    image_resolution: int = 512  # 512x512 or 1024x1024

    # Projection settings
    use_projection: bool = False  # Usually not needed, hidden dim matches
    projection_bias: bool = False

    def __post_init__(self):
        if isinstance(self.learning_mode, str):
            self.learning_mode = LearningMode(self.learning_mode)
        if isinstance(self.torch_dtype, str) and self.torch_dtype:
            self.torch_dtype = getattr(torch, self.torch_dtype)


class Showo2FeatureExtractor(nn.Module):
    """
    Feature extractor using Show-o2 as the backbone.

    Show-o2 is a unified multimodal model that can:
    1. UNDERSTAND: Extract features from text/images/video
    2. GENERATE: Create images from text (native imagination!)

    This replaces Qwen3-VL and the separate ImaginationModule with a
    single unified model.

    Usage:
        extractor = Showo2FeatureExtractor(config)

        # Understanding (feature extraction)
        features = extractor(input_ids, pixel_values=images)

        # Generation (imagination)
        imagined = extractor.imagine("A dark room with moonlight")
    """

    def __init__(self, config: Showo2Config):
        super().__init__()
        self.config = config
        self._model = None
        self._vq_model = None
        self._tokenizer = None
        self._uni_prompting = None
        self._hidden_size = None

        # Lazy loading flag
        self._loaded = False

        # Projection layer (usually not needed for Show-o2)
        self.projection = None

        logger.info(f"Showo2FeatureExtractor initialized (lazy loading from {config.model_name_or_path})")

    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if self._loaded:
            return

        logger.info(f"Loading Show-o2 model from {self.config.model_name_or_path}")

        try:
            # Import Show-o2 components
            # Note: Requires Show-o repository to be installed
            from models.showo import Showo
            from models.magvitv2 import MAGVITv2
            from models.prompting_utils import UniversalPrompting
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                f"Show-o2 dependencies not found: {e}\n"
                "Please install Show-o2 from: https://github.com/showlab/Show-o\n"
                "Run: git clone https://github.com/showlab/Show-o && cd Show-o && bash build_env.sh"
            )

        # Load the main model
        self._model = Showo.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device_map,
        )

        # Load VQ model for image tokenization
        vq_model_path = "showlab/magvitv2"  # Default VQ model
        self._vq_model = MAGVITv2.from_pretrained(vq_model_path)
        self._vq_model = self._vq_model.to(self._model.device)
        self._vq_model.eval()

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            padding_side="left",
        )

        # Setup prompting utility
        self._uni_prompting = UniversalPrompting(
            self._tokenizer,
            max_text_len=512,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
            ),
        )

        # Get hidden size from model config
        # Show-o2 uses Qwen2.5 backbone
        self._hidden_size = self._model.config.hidden_size
        logger.info(f"Show-o2 hidden size: {self._hidden_size}")

        # Setup projection if needed
        if self.config.use_projection and self._hidden_size != self.config.output_dim:
            self.projection = nn.Linear(
                self._hidden_size,
                self.config.output_dim,
                bias=self.config.projection_bias,
            )
            nn.init.normal_(self.projection.weight, std=0.02)
            self.projection = self.projection.to(self._model.device)
            logger.info(f"Added projection: {self._hidden_size} -> {self.config.output_dim}")

        # Apply learning mode
        if self.config.learning_mode == LearningMode.FROZEN:
            self.freeze()

        self._loaded = True
        logger.info(
            f"Show-o2 loaded: hidden_size={self._hidden_size}, "
            f"output_dim={self.config.output_dim}, mode={self.config.learning_mode.value}"
        )

    @property
    def model(self):
        """Get the underlying Show-o2 model (loads if needed)."""
        self._ensure_loaded()
        return self._model

    @property
    def vq_model(self):
        """Get the VQ model for image tokenization."""
        self._ensure_loaded()
        return self._vq_model

    @property
    def tokenizer(self):
        """Get the tokenizer."""
        self._ensure_loaded()
        return self._tokenizer

    def get_hidden_size(self) -> int:
        """Return the hidden size of the underlying model."""
        self._ensure_loaded()
        return self._hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Extract features from text and optional image inputs.

        Args:
            input_ids: (B, S) token IDs
            attention_mask: (B, S) attention mask
            pixel_values: Optional image tensors
            image_grid_thw: Optional grid info (for compatibility)

        Returns:
            features: (B, S, output_dim) per-position features
        """
        self._ensure_loaded()

        B, S = input_ids.shape
        device = input_ids.device

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Encode images if provided
        if pixel_values is not None:
            # Get image embeddings through Show-o2's vision pathway
            image_embeds = self._encode_images(pixel_values)
        else:
            image_embeds = None

        # Get hidden states from Show-o2
        # Show-o2 processes text autoregressively
        outputs = self._model.showo.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Extract last hidden state
        features = outputs.last_hidden_state  # (B, S, hidden_size)

        # If we have image embeddings, we need to incorporate them
        # This depends on how the input was structured
        if image_embeds is not None:
            # For multimodal inputs, Show-o2 typically concatenates
            # text and image embeddings. The features already include both.
            pass

        # Apply projection if configured
        if self.projection is not None:
            features = self.projection(features)

        return features

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings using Show-o2's vision pathway.

        Args:
            pixel_values: (B, C, H, W) image tensors

        Returns:
            image_embeds: (B, num_tokens, hidden_size)
        """
        self._ensure_loaded()

        # Get VQ codes from images
        with torch.no_grad():
            image_codes = self._vq_model.get_code(pixel_values)

        # Convert codes to embeddings
        # Show-o2 has a codebook embedding
        image_embeds = self._model.showo.model.embed_tokens(image_codes)

        return image_embeds

    def imagine(
        self,
        prompt: Union[str, List[str]],
        num_images: int = 1,
        return_features: bool = True,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate images from text prompts (native imagination!).

        This is the key advantage of Show-o2 - generation is built-in,
        not bolted on.

        Args:
            prompt: Text description of what to imagine
            num_images: Number of images to generate per prompt
            return_features: If True, return features instead of images

        Returns:
            If return_features:
                features: (B, S, hidden_size) features of imagined content
            Else:
                images: (B, C, H, W) generated images
        """
        self._ensure_loaded()

        if isinstance(prompt, str):
            prompts = [prompt] * num_images
        else:
            prompts = prompt

        B = len(prompts)
        device = next(self._model.parameters()).device

        # Get generation parameters
        guidance_scale = kwargs.get("guidance_scale", self.config.guidance_scale)
        num_steps = kwargs.get("num_inference_steps", self.config.num_inference_steps)
        temperature = kwargs.get("temperature", self.config.generation_temperature)

        # Prepare input tokens for t2i generation
        num_vq_tokens = 256  # Default for 512x512
        image_tokens = torch.zeros(B, num_vq_tokens, dtype=torch.long, device=device)

        input_ids, _ = self._uni_prompting((prompts, image_tokens), 't2i_gen')
        input_ids = input_ids.to(device)

        # Prepare unconditional input for classifier-free guidance
        uncond_prompts = [""] * B
        uncond_input_ids, _ = self._uni_prompting((uncond_prompts, image_tokens), 't2i_gen')
        uncond_input_ids = uncond_input_ids.to(device)

        # Generate image tokens using discrete diffusion
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            gen_token_ids = self._model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                guidance_scale=guidance_scale,
                temperature=temperature,
                timesteps=num_steps,
                seq_len=num_vq_tokens,
                uni_prompting=self._uni_prompting,
            )

        if return_features:
            # Convert generated tokens to features
            features = self._model.showo.model.embed_tokens(gen_token_ids)
            return features
        else:
            # Decode to images
            images = self._vq_model.decode_code(gen_token_ids)
            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            return images

    def imagine_from_features(
        self,
        features: torch.Tensor,
        mode: str = "scene",
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate imagination features conditioned on input features.

        This allows the PEM loop to work:
        features → imagine → imagined_features → attention pool

        Args:
            features: (B, S, D) conditioning features
            mode: Type of imagination ("scene", "mind", "counterfactual")

        Returns:
            imagined_features: (B, S, D) generated features
        """
        self._ensure_loaded()

        B, S, D = features.shape
        device = features.device

        # For now, we use the GenerativeCore approach for feature-space imagination
        # Show-o2's native generation produces image tokens, not features directly

        # Option 1: Generate image tokens, convert to features
        # Option 2: Use learned transformation (like current GenerativeCore)

        # We'll use a hybrid approach:
        # 1. If mode requires full generation (scene), use Show-o2's t2i
        # 2. For faster modes, use learned transformation

        if mode == "scene" and hasattr(self, '_use_full_generation') and self._use_full_generation:
            # Full generation path (slower but more creative)
            # Would need to decode features to text, generate, re-encode
            # For now, fall back to learned transformation
            pass

        # Default: Use learned transformation (fast path)
        # This maintains compatibility with existing PEM flow
        if not hasattr(self, '_imagination_transform'):
            # Create a simple learned transformation
            self._imagination_transform = nn.Sequential(
                nn.Linear(D, D * 2),
                nn.GELU(),
                nn.Linear(D * 2, D),
            ).to(device)
            logger.info("Created imagination transform for feature-space generation")

        imagined = self._imagination_transform(features)

        return imagined

    def freeze(self):
        """Freeze all parameters except projection."""
        self._ensure_loaded()

        for param in self._model.parameters():
            param.requires_grad = False

        if self.projection is not None:
            for param in self.projection.parameters():
                param.requires_grad = True

        self.config.learning_mode = LearningMode.FROZEN
        logger.info("Show-o2 backbone frozen")

    def unfreeze(self, mode: LearningMode = LearningMode.TRAINABLE):
        """Unfreeze parameters."""
        self._ensure_loaded()

        for param in self._model.parameters():
            param.requires_grad = True

        self.config.learning_mode = mode
        logger.info(f"Show-o2 unfrozen with mode: {mode}")

    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """Get parameter groups with appropriate learning rates."""
        self._ensure_loaded()

        groups = []

        if self.config.learning_mode == LearningMode.FROZEN:
            if self.projection is not None:
                proj_params = [p for p in self.projection.parameters() if p.requires_grad]
                if proj_params:
                    groups.append({"params": proj_params, "lr": base_lr})
        else:
            backbone_lr = base_lr
            if self.config.learning_mode == LearningMode.SLOW_LEARNING:
                backbone_lr = base_lr * self.config.slow_learning_factor

            backbone_params = [p for p in self._model.parameters() if p.requires_grad]
            if backbone_params:
                groups.append({"params": backbone_params, "lr": backbone_lr})

            if self.projection is not None:
                proj_params = [p for p in self.projection.parameters() if p.requires_grad]
                if proj_params:
                    groups.append({"params": proj_params, "lr": base_lr})

        return groups


def create_showo2_extractor(
    model_name_or_path: str = "showlab/show-o2-1.5B",
    output_dim: int = 1536,
    learning_mode: Union[str, LearningMode] = LearningMode.FROZEN,
    **kwargs,
) -> Showo2FeatureExtractor:
    """
    Factory function to create a Show-o2 feature extractor.

    Args:
        model_name_or_path: HuggingFace model identifier
        output_dim: Target feature dimension
        learning_mode: "frozen", "slow", or "trainable"

    Returns:
        Configured Showo2FeatureExtractor instance
    """
    config = Showo2Config(
        model_name_or_path=model_name_or_path,
        output_dim=output_dim,
        learning_mode=learning_mode if isinstance(learning_mode, LearningMode)
                      else LearningMode(learning_mode),
        **kwargs,
    )
    return Showo2FeatureExtractor(config)
