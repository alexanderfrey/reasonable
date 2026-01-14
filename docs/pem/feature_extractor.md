# Feature Extractor (Perception Layer)

The "eyes" of the system. Transforms raw tokens/images into rich contextual features.

## Role

- **Perception, not cognition**: Extracts what's there, doesn't interpret meaning
- **Unified understanding + generation**: Janus Pro provides both feature extraction AND image generation
- **Could be pretrained**: Leverage existing multimodal knowledge
- **Frozen or slow-learning**: Stable feature space for the experiential core

## Primary Implementation: Janus Pro 1B

Janus Pro is the **preferred backbone** for PEM because it provides both understanding AND generation in a unified, compact model.

### Why Janus Pro?

| Feature | Qwen3-VL (legacy) | Janus Pro 1B |
|---------|------------------|--------------|
| Understanding | ✓ | ✓ |
| Generation | ✗ (need separate model) | ✓ (CFG-based) |
| Image generation | ✗ | ✓ |
| Text-to-image | ✗ | ✓ |
| Model size | 2B | 1B/7B |
| Hidden dimension | 1536 | 2048 (projected to 1536) |

### Architecture

Janus Pro uses:
- **DeepSeek** architecture as the LLM backbone
- **Vision encoder** for image understanding
- **Autoregressive generation** with classifier-free guidance (CFG)
- **Discrete image tokens** for both understanding and generation

```
Input ──► Janus Pro ──► Features (understanding)
              │
              └──► Generated tokens (imagination) ──► Decode ──► Features
```

## Implementation

```python
from pem import create_feature_extractor

# Default: Janus Pro 1B
extractor = create_feature_extractor()

# Or specify explicitly
extractor = create_feature_extractor(
    model_name_or_path="deepseek-ai/Janus-Pro-1B",
    output_dim=1536,  # PEM feature dimension
    learning_mode="frozen",
)

# Feature extraction (understanding)
features = extractor(input_ids, pixel_values=images)
# features: (B, S, 1536)

# Image generation (imagination)
imagined_features = extractor.imagine("A forest at sunset", return_features=True)

# Or get actual images
images = extractor.imagine("A forest at sunset", return_features=False)
```

### Configuration

```python
@dataclass
class JanusProConfig:
    model_name_or_path: str = "deepseek-ai/Janus-Pro-1B"
    output_dim: int = 1536          # Target dimension for PEM
    learning_mode: str = "frozen"   # "frozen", "slow", or "trainable"
    torch_dtype: str = "bfloat16"

    # Generation settings
    generation_temperature: float = 1.0
    cfg_weight: float = 5.0         # Classifier-free guidance strength
    parallel_size: int = 4          # Batch size for generation

    # Image settings
    image_resolution: int = 384     # 384x384 images
    image_token_num_per_image: int = 576  # (384/16)^2 tokens per image
```

## Supported Backends

The factory function automatically selects the right implementation:

```python
# Janus Pro (default, recommended)
extractor = create_feature_extractor("deepseek-ai/Janus-Pro-1B")

# Qwen3-VL (legacy, understanding only)
extractor = create_feature_extractor("Qwen/Qwen3-VL-2B-Instruct")
```

## Learning Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **frozen** | No gradients, fixed weights | Stable perception, recommended |
| **slow** | Lower LR than experiential core | Gradual adaptation |
| **trainable** | Full learning rate | End-to-end fine-tuning |

## Interface with Experiential Core

```python
# Features flow UP to the experiential core
features = extractor(tokens, pixel_values=images)  # (B, S, 1536)

# Current position features for surprise computation
current_f = features[:, -1, :]  # (B, 1536)

# Full context for prediction
context = features  # (B, S, 1536)

# Imagination for mental simulation
imagined = extractor.imagine_from_features(features, mode="scene")
```

## Integration with Imagination

When Janus Pro is used, the ImaginationModule can leverage **native generation**:

```python
from pem import create_feature_extractor, create_imagination_module

# Feature extractor with native generation
extractor = create_feature_extractor()

# Imagination module
imagination = create_imagination_module(d_model=1536)

# Connect them - imagination now uses Janus Pro's generation!
imagination.set_feature_extractor(extractor)

# Native generation for scene imagery
output = imagination(features)
# Uses Janus Pro's imagine_from_features() under the hood
```

### Generation Modes

The `imagine_from_features()` method supports different modes:

| Mode | Description | Example |
|------|-------------|---------|
| `"scene"` | Mental imagery | "The forest was dark" → visual scene |
| `"mind"` | Theory of mind | "She smiled nervously" → her thoughts |
| `"counterfactual"` | Alternatives | "He took the key" → what if he didn't? |

## Key Files

| File | Description |
|------|-------------|
| `pem/feature_extractor.py` | Base class, Qwen3-VL impl, factory function |
| `pem/janus_pro_feature_extractor.py` | **Janus Pro 1B implementation** |

## Generation Example

```python
# Text-to-image generation
extractor = create_feature_extractor()

# Generate images
images = extractor.imagine(
    prompt="A cat sitting on a windowsill, watching raindrops",
    num_images=4,
    return_features=False,  # Get actual images
    cfg_weight=5.0,         # CFG strength
    temperature=1.0,
)
# images: numpy array (4, 384, 384, 3)

# Or get features for the PEM loop
features = extractor.imagine(
    prompt="A cat sitting on a windowsill",
    return_features=True,
)
# features: (4, 576, 1536) - 576 tokens per image
```

## Design Considerations

### Projection Layer

Janus Pro 1B has a hidden size of 2048, but PEM uses 1536 for compatibility. A projection layer automatically handles this:

```python
# Hidden size: 2048 → output_dim: 1536
projection = nn.Linear(2048, 1536, bias=False)
```

The projection is trainable even when the backbone is frozen, allowing the feature space to adapt to PEM.

### Lazy Loading

Janus Pro is loaded lazily on first use to avoid blocking on large model downloads:

```python
extractor = create_feature_extractor()  # Fast, no loading yet

# Model loads here on first forward pass
features = extractor(input_ids)
```

### CFG-Based Generation

Janus Pro uses classifier-free guidance for image generation:

1. Generate with conditional input (prompt)
2. Generate with unconditional input (empty prompt)
3. Combine: `logits = uncond + cfg_weight * (cond - uncond)`
4. Sample from combined distribution

This produces high-quality, controllable generation.
