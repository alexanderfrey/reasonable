"""
Feature Extractor (Perception Layer) for PEM.

The "eyes" of the system - transforms raw tokens/images into rich contextual features.
Designed to be swappable between different backends (Show-o2, Qwen3-VL, other HF models, custom).

Show-o2 is the preferred backend as it provides unified understanding AND generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any, Tuple
from enum import Enum
import logging

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
class FeatureExtractorConfig:
    """Configuration for feature extractors."""

    # Model selection
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"

    # Output configuration
    output_dim: int = 1536  # Target feature dimension for PEM

    # Learning configuration
    learning_mode: LearningMode = LearningMode.FROZEN
    slow_learning_factor: float = 0.1  # LR multiplier when slow_learning

    # Model configuration
    torch_dtype: Optional[str] = "bfloat16"  # "float16", "bfloat16", "float32"
    device_map: str = "auto"
    attn_implementation: Optional[str] = "flash_attention_2"  # or "sdpa", "eager"

    # Which layer(s) to extract features from
    # None = last layer, int = specific layer, "all" = all layers
    extract_layer: Optional[Union[int, str]] = None

    # Projection settings
    use_projection: bool = True  # Project to output_dim
    projection_bias: bool = False

    # Trust remote code (needed for some HF models)
    trust_remote_code: bool = True

    def __post_init__(self):
        if isinstance(self.learning_mode, str):
            self.learning_mode = LearningMode(self.learning_mode)
        if isinstance(self.torch_dtype, str):
            self.torch_dtype = getattr(torch, self.torch_dtype)


class FeatureExtractor(ABC, nn.Module):
    """
    Abstract base class for feature extractors.

    All feature extractors must:
    - Accept tokens and optionally images/videos
    - Return per-position features: (B, S, D)
    - Support freezing/slow-learning modes
    """

    def __init__(self, config: FeatureExtractorConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Extract features from inputs.

        Args:
            input_ids: (B, S) token IDs
            attention_mask: (B, S) attention mask
            pixel_values: Optional image/video tensors
            image_grid_thw: Optional image grid info for Qwen-VL

        Returns:
            features: (B, S, output_dim) contextual features
        """
        pass

    @abstractmethod
    def get_hidden_size(self) -> int:
        """Return the hidden size of the underlying model."""
        pass

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.config.learning_mode = LearningMode.FROZEN
        logger.info("Feature extractor frozen")

    def unfreeze(self, mode: LearningMode = LearningMode.TRAINABLE):
        """Unfreeze parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self.config.learning_mode = mode
        logger.info(f"Feature extractor unfrozen with mode: {mode}")

    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """
        Get parameter groups with appropriate learning rates.

        Useful for optimizers when using slow_learning mode.
        """
        if self.config.learning_mode == LearningMode.FROZEN:
            return []  # No trainable params

        lr = base_lr
        if self.config.learning_mode == LearningMode.SLOW_LEARNING:
            lr = base_lr * self.config.slow_learning_factor

        return [{"params": self.parameters(), "lr": lr}]


class Qwen3VLFeatureExtractor(FeatureExtractor):
    """
    Feature extractor using Qwen3-VL as the backbone.

    Wraps Qwen3VLModel to extract per-token features from the last hidden state.
    Supports text, images, and video inputs.
    """

    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config)

        # Lazy imports to avoid dependency issues
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers >= 4.57.0 required for Qwen3-VL. "
                "Install with: pip install transformers>=4.57.0"
            )

        logger.info(f"Loading Qwen3-VL model from {config.model_name_or_path}")

        # Load the full model first (includes pretrained weights for all layers)
        # Then extract the inner model (without LM head)
        full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_name_or_path,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            attn_implementation=config.attn_implementation,
            trust_remote_code=config.trust_remote_code,
        )

        # Extract the base model (Qwen3VLModel) from the full model
        self.model = full_model.model

        # Store hidden size
        self._hidden_size = full_model.config.text_config.hidden_size

        # Clean up the LM head we don't need
        del full_model.lm_head
        del full_model

        # Load processor for handling multimodal inputs
        self.processor = AutoProcessor.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )

        # Optional projection layer to match PEM dimensions
        if config.use_projection and self._hidden_size != config.output_dim:
            self.projection = nn.Linear(
                self._hidden_size,
                config.output_dim,
                bias=config.projection_bias,
            )
            # Initialize projection close to identity for small changes
            nn.init.normal_(self.projection.weight, std=0.02)
            if config.projection_bias:
                nn.init.zeros_(self.projection.bias)

            # Move projection to the same device as the model
            model_device = next(self.model.parameters()).device
            self.projection = self.projection.to(model_device)
            logger.info(f"Added projection: {self._hidden_size} -> {config.output_dim}")
        else:
            self.projection = None
            if config.use_projection:
                logger.info(f"No projection needed: hidden_size matches output_dim ({self._hidden_size})")

        # Apply learning mode
        if config.learning_mode == LearningMode.FROZEN:
            self.freeze()

        logger.info(
            f"Qwen3VLFeatureExtractor initialized: "
            f"hidden_size={self._hidden_size}, output_dim={config.output_dim}, "
            f"mode={config.learning_mode.value}"
        )

    def get_hidden_size(self) -> int:
        return self._hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Extract features from text and optional image/video inputs.

        Args:
            input_ids: (B, S) token IDs
            attention_mask: (B, S) attention mask (optional, created if None)
            pixel_values: (num_images, C, H, W) image tensors (optional)
            pixel_values_videos: (num_videos, T, C, H, W) video tensors (optional)
            image_grid_thw: (num_images, 3) grid info for images
            video_grid_thw: (num_videos, 3) grid info for videos

        Returns:
            features: (B, S, output_dim) per-position features
        """
        B, S = input_ids.shape

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Forward through Qwen3-VL
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            output_hidden_states=self.config.extract_layer is not None,
            **kwargs,
        )

        # Extract features from appropriate layer
        if self.config.extract_layer is None:
            # Use last hidden state
            features = outputs.last_hidden_state  # (B, S, hidden_size)
        elif isinstance(self.config.extract_layer, int):
            # Use specific layer
            features = outputs.hidden_states[self.config.extract_layer]
        elif self.config.extract_layer == "all":
            # Average all layers
            features = torch.stack(outputs.hidden_states, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unknown extract_layer: {self.config.extract_layer}")

        # Apply projection if configured
        if self.projection is not None:
            # Ensure projection is on the same device and dtype as features
            if self.projection.weight.device != features.device or self.projection.weight.dtype != features.dtype:
                self.projection = self.projection.to(device=features.device, dtype=features.dtype)
            features = self.projection(features)

        return features

    def process_inputs(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[List[Any]] = None,
        videos: Optional[List[Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process raw text/image/video inputs into model-ready tensors.

        Convenience method that uses the processor to handle multimodal inputs.

        Args:
            texts: List of text strings
            images: List of PIL images or image paths/URLs
            videos: List of video paths

        Returns:
            Dict with input_ids, attention_mask, and optional pixel_values
        """
        # Build messages format for the processor
        messages = []
        for i, text in enumerate(texts or [""]):
            content = []
            if images and i < len(images):
                content.append({"type": "image", "image": images[i]})
            if videos and i < len(videos):
                content.append({"type": "video", "video": videos[i]})
            content.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": content})

        # Process through the processor
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        # Remove token_type_ids if present (not used by Qwen)
        inputs.pop("token_type_ids", None)

        return inputs

    def freeze(self):
        """Freeze the backbone, keep projection trainable if it exists."""
        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Keep projection trainable (if exists) for fine-tuning to PEM
        # This allows adapting the feature space while keeping perception fixed
        if self.projection is not None:
            for param in self.projection.parameters():
                param.requires_grad = True

        self.config.learning_mode = LearningMode.FROZEN
        logger.info("Feature extractor backbone frozen (projection remains trainable)")

    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """Get parameter groups with separate LR for backbone and projection."""
        groups = []

        if self.config.learning_mode == LearningMode.FROZEN:
            # Only projection params (if trainable)
            if self.projection is not None:
                proj_params = [p for p in self.projection.parameters() if p.requires_grad]
                if proj_params:
                    groups.append({"params": proj_params, "lr": base_lr})
        else:
            # All trainable params
            backbone_lr = base_lr
            if self.config.learning_mode == LearningMode.SLOW_LEARNING:
                backbone_lr = base_lr * self.config.slow_learning_factor

            backbone_params = [p for p in self.model.parameters() if p.requires_grad]
            if backbone_params:
                groups.append({"params": backbone_params, "lr": backbone_lr})

            if self.projection is not None:
                proj_params = [p for p in self.projection.parameters() if p.requires_grad]
                if proj_params:
                    groups.append({"params": proj_params, "lr": base_lr})

        return groups


def create_feature_extractor(
    model_name_or_path: str = "showlab/show-o2-1.5B",
    output_dim: int = 1536,
    learning_mode: Union[str, LearningMode] = LearningMode.FROZEN,
    **kwargs,
) -> FeatureExtractor:
    """
    Factory function to create a feature extractor.

    Automatically selects the appropriate implementation based on model name.
    Default is Show-o2, which provides unified understanding AND generation.

    Args:
        model_name_or_path: HuggingFace model identifier or path
            - "showlab/show-o2-1.5B" (default, recommended)
            - "showlab/show-o2-7B" (larger model)
            - "Qwen/Qwen3-VL-*" (legacy, understanding only)
        output_dim: Target feature dimension for PEM
        learning_mode: "frozen", "slow", or "trainable"
        **kwargs: Additional config options

    Returns:
        Configured FeatureExtractor instance
    """
    # Select implementation based on model name
    model_lower = model_name_or_path.lower()

    if "show-o" in model_lower or "showo" in model_lower:
        # Show-o2 - preferred unified model
        from .showo2_feature_extractor import (
            Showo2Config,
            Showo2FeatureExtractor,
            LearningMode as Showo2LearningMode,
        )
        config = Showo2Config(
            model_name_or_path=model_name_or_path,
            output_dim=output_dim,
            learning_mode=learning_mode if isinstance(learning_mode, Showo2LearningMode)
                          else Showo2LearningMode(learning_mode.value if isinstance(learning_mode, LearningMode) else learning_mode),
            **{k: v for k, v in kwargs.items() if k in Showo2Config.__dataclass_fields__},
        )
        return Showo2FeatureExtractor(config)

    elif "qwen" in model_lower and ("vl" in model_lower or "vision" in model_lower):
        # Qwen3-VL - legacy support
        config = FeatureExtractorConfig(
            model_name_or_path=model_name_or_path,
            output_dim=output_dim,
            learning_mode=learning_mode if isinstance(learning_mode, LearningMode)
                          else LearningMode(learning_mode),
            **kwargs,
        )
        return Qwen3VLFeatureExtractor(config)

    else:
        # Default to Show-o2
        logger.warning(
            f"Unknown model type '{model_name_or_path}', "
            f"defaulting to Show-o2 loader..."
        )
        from .showo2_feature_extractor import (
            Showo2Config,
            Showo2FeatureExtractor,
            LearningMode as Showo2LearningMode,
        )
        config = Showo2Config(
            model_name_or_path=model_name_or_path,
            output_dim=output_dim,
            learning_mode=learning_mode if isinstance(learning_mode, Showo2LearningMode)
                          else Showo2LearningMode(learning_mode.value if isinstance(learning_mode, LearningMode) else learning_mode),
        )
        return Showo2FeatureExtractor(config)
