"""
Janus Pro 1B Feature Extractor for PEM.

Janus Pro 1B is a unified multimodal model that handles both understanding
AND generation natively.

Key capabilities:
- Multimodal understanding (text, image)
- Text-to-image generation (autoregressive with CFG)
- Same hidden dimension compatibility (1024 for 1B model)

Architecture:
    Input ──► Janus Pro ──► Features (understanding)
                  │
                  └──► Generated tokens (imagination) ──► Decode ──► Features

Reference: https://github.com/deepseek-ai/Janus
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """How the feature extractor learns."""
    FROZEN = "frozen"           # No gradients, fixed weights
    SLOW_LEARNING = "slow"      # Lower learning rate than rest of model
    TRAINABLE = "trainable"     # Full learning rate


@dataclass
class JanusProConfig:
    """Configuration for Janus Pro feature extractor."""

    # Model selection
    model_name_or_path: str = "deepseek-ai/Janus-Pro-1B"

    # Output configuration
    output_dim: int = 1536  # Target feature dimension for PEM (will project from 2048)

    # Learning configuration
    learning_mode: LearningMode = LearningMode.FROZEN
    slow_learning_factor: float = 0.1

    # Model configuration
    torch_dtype: Optional[str] = "bfloat16"
    device_map: str = "auto"

    # Generation settings
    generation_temperature: float = 1.0
    cfg_weight: float = 5.0  # Classifier-free guidance weight
    parallel_size: int = 4  # Number of parallel generations (batch)

    # Resolution for image generation
    image_resolution: int = 384  # Janus uses 384x384
    patch_size: int = 16
    image_token_num_per_image: int = 576  # (384/16)^2 = 576 tokens

    # Projection settings
    use_projection: bool = True  # Project from hidden_size to output_dim
    projection_bias: bool = False

    def __post_init__(self):
        if isinstance(self.learning_mode, str):
            self.learning_mode = LearningMode(self.learning_mode)
        if isinstance(self.torch_dtype, str) and self.torch_dtype:
            self.torch_dtype = getattr(torch, self.torch_dtype)


class JanusProFeatureExtractor(nn.Module):
    """
    Feature extractor using Janus Pro 1B as the backbone.

    Janus Pro is a unified multimodal model that can:
    1. UNDERSTAND: Extract features from text/images
    2. GENERATE: Create images from text (native imagination!)

    A compact and efficient unified multimodal model.

    Usage:
        extractor = JanusProFeatureExtractor(config)

        # Understanding (feature extraction)
        features = extractor(input_ids, pixel_values=images)

        # Generation (imagination)
        imagined = extractor.imagine("A dark room with moonlight")
    """

    def __init__(self, config: JanusProConfig):
        super().__init__()
        self.config = config
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._hidden_size = None

        # Lazy loading flag
        self._loaded = False

        # Projection layer (created after loading)
        self.projection = None

        # Imagination transform for feature-space generation
        self._imagination_transform = None

        logger.info(f"JanusProFeatureExtractor initialized (lazy loading from {config.model_name_or_path})")

    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if self._loaded:
            return

        logger.info(f"Loading Janus Pro model from {self.config.model_name_or_path}")

        try:
            from transformers import AutoModelForCausalLM
            from janus.models import MultiModalityCausalLM, VLChatProcessor
        except ImportError as e:
            raise ImportError(
                f"Janus Pro dependencies not found: {e}\n"
                "Please install Janus from: https://github.com/deepseek-ai/Janus\n"
                "Run: pip install -e .[janus] or clone the repo and install dependencies"
            )

        # Load processor and tokenizer
        self._processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            self.config.model_name_or_path
        )
        self._tokenizer = self._processor.tokenizer

        # Load the main model
        self._model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
        )

        # Move to appropriate dtype and device
        if self.config.torch_dtype:
            self._model = self._model.to(self.config.torch_dtype)

        # Handle device placement
        if self.config.device_map == "auto" and torch.cuda.is_available():
            self._model = self._model.cuda()
        elif self.config.device_map != "auto":
            self._model = self._model.to(self.config.device_map)

        self._model.eval()

        # Get hidden size from model config
        # Janus uses a nested config: model.config.language_config.hidden_size
        if hasattr(self._model.config, 'language_config'):
            self._hidden_size = self._model.config.language_config.hidden_size
        elif hasattr(self._model, 'language_model') and hasattr(self._model.language_model, 'config'):
            self._hidden_size = self._model.language_model.config.hidden_size
        else:
            # Fallback for Janus Pro 1B
            self._hidden_size = 2048
            logger.warning("Could not find hidden_size in config, using default 2048")
        logger.info(f"Janus Pro hidden size: {self._hidden_size}")

        # Setup projection if needed (Janus 1B: 2048 -> 1536 for PEM)
        if self.config.use_projection and self._hidden_size != self.config.output_dim:
            self.projection = nn.Linear(
                self._hidden_size,
                self.config.output_dim,
                bias=self.config.projection_bias,
            )
            nn.init.normal_(self.projection.weight, std=0.02)
            device = next(self._model.parameters()).device
            dtype = next(self._model.parameters()).dtype
            self.projection = self.projection.to(device=device, dtype=dtype)
            logger.info(f"Added projection: {self._hidden_size} -> {self.config.output_dim}")

        # Mark as loaded BEFORE calling freeze() to avoid recursion
        self._loaded = True

        # Apply learning mode
        if self.config.learning_mode == LearningMode.FROZEN:
            self.freeze()
        logger.info(
            f"Janus Pro loaded: hidden_size={self._hidden_size}, "
            f"output_dim={self.config.output_dim}, mode={self.config.learning_mode.value}"
        )

    @property
    def model(self):
        """Get the underlying Janus Pro model (loads if needed)."""
        self._ensure_loaded()
        return self._model

    @property
    def processor(self):
        """Get the VLChatProcessor."""
        self._ensure_loaded()
        return self._processor

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
            pixel_values: Optional image tensors (B, C, H, W)
            image_grid_thw: Optional grid info (for compatibility, not used by Janus)

        Returns:
            features: (B, S, output_dim) per-position features
        """
        self._ensure_loaded()

        B, S = input_ids.shape
        device = input_ids.device

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Get embeddings - handle multimodal if images provided
        if pixel_values is not None:
            # Use Janus's multimodal understanding pathway
            # This uses SigLIP vision encoder + adaptor to merge image features
            inputs_embeds, attention_mask = self._encode_images_for_understanding(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            # Text-only: just get text embeddings
            inputs_embeds = self._model.language_model.get_input_embeddings()(input_ids)

        # Forward through language model to get hidden states
        outputs = self._model.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Extract last hidden state
        features = outputs.last_hidden_state  # (B, S, hidden_size)

        # Apply projection if configured
        if self.projection is not None:
            # Ensure projection is on correct device/dtype
            if self.projection.weight.device != features.device:
                self.projection = self.projection.to(device=features.device, dtype=features.dtype)
            features = self.projection(features)

        return features

    def _encode_images_for_understanding(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images and merge with text embeddings for multimodal understanding.

        Janus uses SigLIP vision encoder + adaptor for understanding pathway.

        Args:
            pixel_values: (B, C, H, W) or (B, N, C, H, W) image tensors
            input_ids: (B, S) token IDs with image placeholders
            attention_mask: (B, S) attention mask

        Returns:
            inputs_embeds: (B, S, hidden_size) merged text+image embeddings
            attention_mask: (B, S) updated attention mask
        """
        self._ensure_loaded()

        # Janus has a prepare_inputs_embeds method that handles multimodal fusion
        # It uses the vision encoder for understanding and replaces placeholder tokens
        if hasattr(self._model, 'prepare_inputs_embeds'):
            # Use Janus's native multimodal fusion
            inputs_embeds, attention_mask, _ = self._model.prepare_inputs_embeds(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )
            return inputs_embeds, attention_mask

        # Fallback: just use text embeddings (images won't be processed)
        logger.warning("Multimodal fusion not available, using text-only features")
        inputs_embeds = self._model.language_model.get_input_embeddings()(input_ids)
        return inputs_embeds, attention_mask

    @torch.inference_mode()
    def imagine(
        self,
        prompt: Union[str, List[str]],
        num_images: int = 1,
        return_features: bool = True,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
        """
        Generate images from text prompts (native imagination!).

        This is the key advantage of Janus - generation is built-in.
        Uses classifier-free guidance for high-quality generation.

        Args:
            prompt: Text description of what to imagine
            num_images: Number of images to generate per prompt
            return_features: If True, return features instead of images

        Returns:
            If return_features:
                features: (B, S, hidden_size) features of imagined content
            Else:
                images: (B, H, W, 3) generated images as numpy arrays
        """
        self._ensure_loaded()

        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype

        # Get generation parameters
        cfg_weight = kwargs.get("cfg_weight", self.config.cfg_weight)
        temperature = kwargs.get("temperature", self.config.generation_temperature)
        parallel_size = kwargs.get("parallel_size", min(num_images, self.config.parallel_size))
        img_size = self.config.image_resolution
        patch_size = self.config.patch_size
        image_token_num = self.config.image_token_num_per_image

        all_features = []
        all_images = []

        for prompt_text in prompts:
            # Prepare conversation format
            conversation = [
                {"role": "<|User|>", "content": prompt_text},
                {"role": "<|Assistant|>", "content": ""},
            ]

            # Apply chat template
            sft_format = self._processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self._processor.sft_format,
                system_prompt="",
            )
            full_prompt = sft_format + self._processor.image_start_tag

            # Encode prompt
            input_ids = self._tokenizer.encode(full_prompt)
            input_ids = torch.LongTensor(input_ids).to(device)

            # Setup for CFG: interleave conditional and unconditional
            tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.long, device=device)
            for i in range(parallel_size * 2):
                tokens[i, :] = input_ids
                if i % 2 != 0:
                    # Unconditional: mask out content
                    tokens[i, 1:-1] = self._processor.pad_id

            inputs_embeds = self._model.language_model.get_input_embeddings()(tokens)
            generated_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.long, device=device)

            # Autoregressive generation with CFG
            past_key_values = None
            for i in range(image_token_num):
                outputs = self._model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
                hidden_states = outputs.last_hidden_state

                # Get logits from generation head
                logits = self._model.gen_head(hidden_states[:, -1, :])

                # Apply CFG
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

                # Sample
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)

                # Prepare next input
                next_token_expanded = torch.cat([
                    next_token.unsqueeze(dim=1),
                    next_token.unsqueeze(dim=1)
                ], dim=1).view(-1)
                img_embeds = self._model.prepare_gen_img_embeds(next_token_expanded)
                inputs_embeds = img_embeds.unsqueeze(dim=1)

            if return_features:
                # Convert generated tokens to features
                features = self._model.language_model.get_input_embeddings()(generated_tokens)
                # Reshape to match expected format: (parallel_size, num_tokens, hidden_size)
                all_features.append(features)
            else:
                # Decode to images
                shape = [parallel_size, 8, img_size // patch_size, img_size // patch_size]
                dec = self._model.gen_vision_model.decode_code(
                    generated_tokens.to(dtype=torch.int),
                    shape=shape
                )
                dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
                dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

                # Create visual images
                visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
                visual_img[:, :, :] = dec
                all_images.append(visual_img)

        if return_features:
            # Stack all features: (total_images, num_tokens, hidden_size)
            features = torch.cat(all_features, dim=0)
            # Apply projection if needed
            if self.projection is not None:
                features = self.projection(features)
            return features
        else:
            # Stack all images
            return np.concatenate(all_images, axis=0)

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

        # For feature-space imagination, we use a learned transformation
        # This maintains compatibility with the PEM flow without requiring
        # full generation (which would be slow)

        if mode == "scene" and hasattr(self, '_use_full_generation') and self._use_full_generation:
            # Full generation path - would decode features to text, generate image, re-encode
            # This is expensive but produces richer imagination
            # For now, fall back to learned transformation
            pass

        # Default: Use learned transformation (fast path)
        if self._imagination_transform is None:
            # Create the transformation layers
            self._imagination_transform = nn.Sequential(
                nn.Linear(D, D * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(D * 2, D),
            ).to(device=device, dtype=features.dtype)

            # Initialize for small perturbations initially
            for m in self._imagination_transform.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            logger.info("Created imagination transform for feature-space generation")

        # Ensure transform is on correct device
        if next(self._imagination_transform.parameters()).device != device:
            self._imagination_transform = self._imagination_transform.to(device=device, dtype=features.dtype)

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
        logger.info("Janus Pro backbone frozen")

    def unfreeze(self, mode: LearningMode = LearningMode.TRAINABLE):
        """Unfreeze parameters."""
        self._ensure_loaded()

        for param in self._model.parameters():
            param.requires_grad = True

        self.config.learning_mode = mode
        logger.info(f"Janus Pro unfrozen with mode: {mode}")

    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """Get parameter groups with appropriate learning rates."""
        self._ensure_loaded()

        groups = []

        if self.config.learning_mode == LearningMode.FROZEN:
            # Only projection and imagination transform params
            if self.projection is not None:
                proj_params = [p for p in self.projection.parameters() if p.requires_grad]
                if proj_params:
                    groups.append({"params": proj_params, "lr": base_lr})

            if self._imagination_transform is not None:
                transform_params = [p for p in self._imagination_transform.parameters() if p.requires_grad]
                if transform_params:
                    groups.append({"params": transform_params, "lr": base_lr})
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

    def process_inputs(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[List[Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process raw text/image inputs into model-ready tensors.

        Convenience method that uses the processor to handle multimodal inputs.

        Args:
            texts: List of text strings
            images: List of PIL images

        Returns:
            Dict with input_ids, attention_mask, and optional pixel_values
        """
        self._ensure_loaded()

        if texts is None:
            texts = [""]

        # For Janus, we use the VLChatProcessor
        # Build conversation format
        all_inputs = {}
        input_ids_list = []

        for i, text in enumerate(texts):
            if images and i < len(images):
                # Multimodal input
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{text}",
                        "images": [images[i]],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                inputs = self._processor(
                    conversations=conversation,
                    return_tensors="pt",
                )
                # pixel_values would be in inputs
                if "pixel_values" not in all_inputs and "pixel_values" in inputs:
                    all_inputs["pixel_values"] = inputs["pixel_values"]
            else:
                # Text-only input
                conversation = [
                    {"role": "<|User|>", "content": text},
                    {"role": "<|Assistant|>", "content": ""},
                ]
                inputs = self._processor(
                    conversations=conversation,
                    return_tensors="pt",
                )

            input_ids_list.append(inputs["input_ids"])

        # Pad and batch input_ids
        max_len = max(ids.shape[1] for ids in input_ids_list)
        batched_ids = torch.zeros(len(input_ids_list), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(input_ids_list), max_len, dtype=torch.long)

        for i, ids in enumerate(input_ids_list):
            batched_ids[i, :ids.shape[1]] = ids[0]
            attention_mask[i, :ids.shape[1]] = 1

        all_inputs["input_ids"] = batched_ids
        all_inputs["attention_mask"] = attention_mask

        return all_inputs


def create_janus_pro_extractor(
    model_name_or_path: str = "deepseek-ai/Janus-Pro-1B",
    output_dim: int = 1536,
    learning_mode: Union[str, LearningMode] = LearningMode.FROZEN,
    **kwargs,
) -> JanusProFeatureExtractor:
    """
    Factory function to create a Janus Pro feature extractor.

    Args:
        model_name_or_path: HuggingFace model identifier
        output_dim: Target feature dimension
        learning_mode: "frozen", "slow", or "trainable"

    Returns:
        Configured JanusProFeatureExtractor instance
    """
    config = JanusProConfig(
        model_name_or_path=model_name_or_path,
        output_dim=output_dim,
        learning_mode=learning_mode if isinstance(learning_mode, LearningMode)
                      else LearningMode(learning_mode),
        **{k: v for k, v in kwargs.items() if k in JanusProConfig.__dataclass_fields__},
    )
    return JanusProFeatureExtractor(config)
