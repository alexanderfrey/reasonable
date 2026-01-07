# Vision Module: Proposal

## Overview

A vision module that enables the model to **perceive**, **understand**, and **reason** about images through the experiential architecture. The core principle: **the model should strive to understand everything it sees**, and when it encounters something it doesn't understand, it should either ask questions or actively try to figure it out.

This extends the system from text-only to multimodal perception while preserving the core experiential principles: surprise-driven attention, soma-modulated processing, and memory crystallization.

## Core Principle: Understanding-Driven Vision

Unlike passive image encoding, this module implements **active visual understanding**:

1. **Comprehension checking** — For each region/object, assess: "Do I understand what this is?"
2. **Uncertainty detection** — Identify what is confusing, ambiguous, or novel
3. **Active resolution** — Either ask the user for clarification OR attempt to figure it out through reasoning
4. **Completeness drive** — The system is not satisfied until it has a coherent understanding of the whole scene

```
┌─────────────────────────────────────────────────────────────────┐
│                 UNDERSTANDING-DRIVEN VISION LOOP                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐     ┌──────────────┐     ┌───────────────────┐   │
│   │  IMAGE  │────▶│   ENCODE     │────▶│  INITIAL PARSE    │   │
│   └─────────┘     └──────────────┘     │  "What do I see?" │   │
│                                         └─────────┬─────────┘   │
│                                                   │              │
│                                                   ▼              │
│                                        ┌───────────────────┐    │
│                                        │ COMPREHENSION     │    │
│                           ┌───────────▶│ CHECK             │    │
│                           │            │ "Do I understand  │    │
│                           │            │  each part?"      │    │
│                           │            └─────────┬─────────┘    │
│                           │                      │              │
│                           │            ┌─────────┴─────────┐    │
│                           │            ▼                   ▼    │
│                           │   ┌─────────────┐    ┌────────────┐ │
│                           │   │ UNDERSTOOD  │    │ UNCERTAIN  │ │
│                           │   │ confidence  │    │ confusion  │ │
│                           │   │ > threshold │    │ detected   │ │
│                           │   └──────┬──────┘    └─────┬──────┘ │
│                           │          │                 │        │
│                           │          ▼                 ▼        │
│                           │   ┌─────────────┐  ┌─────────────┐  │
│                           │   │  INTEGRATE  │  │  RESOLVE    │  │
│                           │   │  into       │  │             │  │
│                           │   │  memory     │  ├─────────────┤  │
│                           │   └─────────────┘  │ ASK USER    │  │
│                           │                    │ "What is    │  │
│                           │                    │  this?"     │  │
│                           │                    ├─────────────┤  │
│                           │                    │ REASON      │  │
│                           │                    │ Use context │  │
│                           │                    │ + memory    │  │
│                           │                    ├─────────────┤  │
│                           │                    │ HYPOTHESIZE │  │
│                           │                    │ "Maybe it's │  │
│                           │                    │  a..."      │  │
│                           │                    └──────┬──────┘  │
│                           │                          │          │
│                           └──────────────────────────┘          │
│                                    (iterate until understood)   │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture Status

### Core Components
- [ ] **VisualEncoder** — Encode images to latent representations
- [ ] **VisualTokenizer** — Convert visual features to token-like embeddings
- [ ] **VisualSurprise** — Compute visual prediction error / novelty
- [ ] **VisualSomaIntegration** — Visual signals → soma update
- [ ] **VisualMemory** — Store and retrieve visual episodes
- [ ] **VisualGrounding** — Connect visual regions to language

### Understanding-Driven Components
- [ ] **ComprehensionChecker** — Assess understanding of each region
- [ ] **UncertaintyResolver** — Decide how to resolve confusion
- [ ] **VisualReasoner** — Multi-step reasoning to figure things out
- [ ] **VisualUnderstandingLoop** — Iterate until comprehension achieved

### Questioning Integration (see `questioning_system.md`)
- [ ] **VisualPredictionFailure** — Visual prediction error as question source
- [ ] **Visual Question Generation** — Convert visual uncertainty to questions
- [ ] **Grounded Questions** — Questions that point to specific image regions

### Integration
- [ ] Hook visual embeddings into transformer attention
- [ ] Visual surprise → memory crystallization
- [ ] Soma modulation of visual attention
- [ ] Cross-modal retrieval (text → image, image → text)
- [ ] Action module integration for asking user questions
- [ ] Understanding state affects soma (curiosity, confusion, satisfaction)

---

## Motivation

The current system processes text and develops internal states. But:

1. **Grounding** — Language without perception is ungrounded. "Red" means nothing without seeing red.
2. **Richer experience** — Visual input provides dense, immediate signals that text cannot capture.
3. **Memory** — Episodic memories are often visual. "I remember seeing..." not just "I remember reading..."
4. **Reasoning** — Many problems are easier to reason about visually (spatial, physical, relational).

Humans don't process vision separately from language — they're integrated. A word can evoke an image; an image can evoke words. The vision module should achieve this integration.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VISION MODULE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌─────────────────┐     ┌──────────────────────────┐ │
│  │    IMAGE     │────▶│  VISUAL ENCODER │────▶│    VISUAL TOKENS         │ │
│  │   [H,W,C]    │     │  (ViT / CNN)    │     │    [N_patches, d_model]  │ │
│  └──────────────┘     └─────────────────┘     └────────────┬─────────────┘ │
│                                                             │               │
│                       ┌─────────────────────────────────────┼───────────┐   │
│                       │                                     ▼           │   │
│                       │              ┌──────────────────────────────┐   │   │
│                       │              │     CROSS-MODAL ATTENTION    │   │   │
│                       │              │   text attends to visual     │   │   │
│                       │              │   visual attends to text     │   │   │
│                       │              └──────────────┬───────────────┘   │   │
│                       │                             │                   │   │
│  ┌────────────────────┼─────────────────────────────┼───────────────┐   │   │
│  │     EXPERIENTIAL   │                             ▼               │   │   │
│  │                    │              ┌──────────────────────────┐   │   │   │
│  │  ┌─────────────┐   │              │    VISUAL SURPRISE       │   │   │   │
│  │  │   VISUAL    │◀──┘              │  "I didn't expect to see │   │   │   │
│  │  │   MEMORY    │                  │   this"                  │   │   │   │
│  │  │             │◀─────────────────│  prediction_error(v)     │   │   │   │
│  │  └─────────────┘   crystallize    └────────────┬─────────────┘   │   │   │
│  │        │           if salient                  │                 │   │   │
│  │        │                                       ▼                 │   │   │
│  │        │                          ┌──────────────────────────┐   │   │   │
│  │        │                          │    SOMA INTEGRATION      │   │   │   │
│  │        └─────────────────────────▶│  visual_valence          │   │   │   │
│  │              retrieve             │  visual_arousal          │   │   │   │
│  │                                   │  visual_novelty          │   │   │   │
│  │                                   └──────────────────────────┘   │   │   │
│  └──────────────────────────────────────────────────────────────────┘   │   │
│                                                                          │   │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. VisualEncoder

Encodes raw images into latent representations.

```python
class VisualEncoder(nn.Module):
    """
    Encode images to patch embeddings compatible with transformer.

    Options:
    - ViT-style: split into patches, linear projection
    - CNN backbone: use pretrained features (CLIP, DINOv2)
    - Hybrid: CNN features + transformer refinement
    """

    def __init__(self, config):
        self.patch_size = config.patch_size  # e.g., 14 or 16
        self.d_model = config.d_model

        # Option 1: Simple patch embedding (trainable from scratch)
        self.patch_embed = nn.Conv2d(
            3, config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        # Option 2: Pretrained backbone (frozen or fine-tuned)
        # self.backbone = load_pretrained_vision_model(config.backbone)
        # self.proj = nn.Linear(backbone_dim, config.d_model)

        # Positional encoding for patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_patches, config.d_model)
        )

        # Special tokens
        self.visual_start = nn.Parameter(torch.randn(1, 1, config.d_model))
        self.visual_end = nn.Parameter(torch.randn(1, 1, config.d_model))

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, C, H, W] normalized images

        Returns:
            visual_tokens: [B, N_patches + 2, d_model] (with start/end tokens)
            patch_positions: [B, N_patches, 2] (row, col) for each patch
        """
        B = images.shape[0]

        # Extract patches
        patches = self.patch_embed(images)  # [B, d_model, H', W']
        patches = patches.flatten(2).transpose(1, 2)  # [B, N_patches, d_model]

        # Add positional encoding
        patches = patches + self.pos_embed[:, :patches.shape[1], :]

        # Add start/end tokens
        visual_tokens = torch.cat([
            self.visual_start.expand(B, -1, -1),
            patches,
            self.visual_end.expand(B, -1, -1),
        ], dim=1)

        return {
            'visual_tokens': visual_tokens,
            'n_patches': patches.shape[1],
            'patch_grid': (images.shape[2] // self.patch_size,
                          images.shape[3] // self.patch_size),
        }
```

### 2. VisualSurprise

Computes prediction error for visual input — "did I expect to see this?"

```python
class VisualSurprise(nn.Module):
    """
    Visual surprise based on prediction error.

    Two modes:
    1. Context-based: Given text/soma, predict what image should look like
    2. Self-supervised: Predict masked patches from visible patches
    """

    def __init__(self, config):
        self.d_model = config.d_model

        # Context-conditioned visual predictor
        # "Given what I'm reading/feeling, what do I expect to see?"
        self.context_to_visual = nn.Sequential(
            nn.Linear(config.d_model + config.d_soma, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Patch-level surprise (per-patch prediction error)
        self.patch_predictor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(config.d_model, nhead=8, batch_first=True),
            num_layers=2
        )

        # Aggregate to scalar surprise
        self.surprise_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,  # [B, N, d_model]
        context: torch.Tensor,         # [B, d_model] from text
        soma: torch.Tensor,            # [B, d_soma]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute visual surprise.

        Returns:
            surprise: [B] overall visual surprise
            patch_surprise: [B, N] per-patch surprise map
            predicted_visual: [B, N, d_model] what was expected
        """
        B, N, D = visual_tokens.shape

        # What did I expect to see given context + soma?
        context_cond = self.context_to_visual(
            torch.cat([context, soma], dim=-1)
        )  # [B, d_model]

        # Expand to match visual tokens
        expected_visual = context_cond.unsqueeze(1).expand(-1, N, -1)

        # Prediction error per patch
        patch_error = (visual_tokens - expected_visual).pow(2).sum(dim=-1)  # [B, N]
        patch_surprise = patch_error / (D ** 0.5)  # Normalize

        # Also compute self-supervised surprise (mask and predict)
        # This captures "is this image internally consistent?"
        refined = self.patch_predictor(visual_tokens)
        internal_error = (refined - visual_tokens).pow(2).sum(dim=-1)

        # Combine context-based and internal surprise
        combined_surprise = 0.7 * patch_surprise + 0.3 * internal_error

        # Aggregate to overall surprise
        # Weight by attention (some patches matter more)
        pooled = visual_tokens.mean(dim=1)  # [B, d_model]
        overall_surprise = self.surprise_head(pooled).squeeze(-1)  # [B]

        return {
            'surprise': overall_surprise,
            'patch_surprise': combined_surprise,
            'predicted_visual': expected_visual,
            'salient_patches': (combined_surprise > combined_surprise.mean(dim=1, keepdim=True)),
        }
```

### 3. VisualSomaIntegration

Converts visual signals into soma updates.

```python
class VisualSomaIntegration(nn.Module):
    """
    Extract affective signals from visual input to update soma.

    Visual signals:
    - Valence: positive/negative (beauty, threat, disgust)
    - Arousal: activating/calming (motion, contrast, complexity)
    - Novelty: familiar/unfamiliar (seen before? typical?)
    """

    def __init__(self, config):
        self.d_model = config.d_model
        self.n_signals = 6  # Match text signals

        # Extract affective features from visual tokens
        self.affective_extractor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, self.n_signals),
            nn.Tanh(),  # Bounded signals
        )

        # Global image features for overall assessment
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
        )

        # Attention over patches for signal extraction
        self.signal_attention = nn.MultiheadAttention(
            config.d_model, num_heads=4, batch_first=True
        )

        # Learnable signal queries
        self.signal_queries = nn.Parameter(
            torch.randn(1, self.n_signals, config.d_model)
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,  # [B, N, d_model]
        visual_surprise: torch.Tensor,  # [B, N] per-patch surprise
    ) -> Dict[str, torch.Tensor]:
        """
        Extract soma signals from visual input.

        Returns:
            signals: [B, n_signals] - [surprise, arousal, valence, novelty, certainty, engagement]
        """
        B = visual_tokens.shape[0]

        # Attend to visual tokens with signal queries
        queries = self.signal_queries.expand(B, -1, -1)
        attended, attn_weights = self.signal_attention(
            queries, visual_tokens, visual_tokens
        )  # [B, n_signals, d_model]

        # Extract signals
        signals = self.affective_extractor(attended)  # [B, n_signals, n_signals]
        signals = signals.diagonal(dim1=-2, dim2=-1)  # [B, n_signals]

        # Override surprise with computed visual surprise
        signals[:, 0] = visual_surprise.mean(dim=1)

        return {
            'signals': signals,
            'signal_attention': attn_weights,  # Where did each signal come from?
        }
```

### 4. VisualMemory

Stores and retrieves visual episodes.

```python
class VisualMemory(nn.Module):
    """
    Episodic memory for visual experiences.

    Stores:
    - Compressed visual representation
    - Associated text/context
    - Soma state at encoding
    - Salience score

    Retrieves:
    - By visual similarity (see something similar)
    - By text query (recall what X looked like)
    - By soma state (what did I see when I felt like this?)
    """

    def __init__(self, config):
        self.d_model = config.d_model
        self.d_memory = config.d_visual_memory
        self.max_memories = config.max_visual_memories

        # Compress visual tokens to memory
        self.visual_compressor = nn.Sequential(
            nn.Linear(config.d_model, config.d_memory),
            nn.GELU(),
            nn.Linear(config.d_memory, config.d_memory),
        )

        # Context encoder (text + soma)
        self.context_encoder = nn.Linear(
            config.d_model + config.d_soma, config.d_memory
        )

        # Memory banks (registered as buffers for persistence)
        self.register_buffer('visual_memories',
            torch.zeros(self.max_memories, config.d_memory))
        self.register_buffer('context_memories',
            torch.zeros(self.max_memories, config.d_memory))
        self.register_buffer('salience_scores',
            torch.zeros(self.max_memories))
        self.register_buffer('memory_count', torch.tensor(0))

        # Retrieval
        self.query_proj = nn.Linear(config.d_model, config.d_memory)

    def crystallize(
        self,
        visual_tokens: torch.Tensor,  # [B, N, d_model]
        context: torch.Tensor,         # [B, d_model]
        soma: torch.Tensor,            # [B, d_soma]
        salience: torch.Tensor,        # [B] crystallization score
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Store visual experience if salient enough.
        """
        # Pool visual tokens
        visual_pooled = visual_tokens.mean(dim=1)  # [B, d_model]

        # Compress
        visual_mem = self.visual_compressor(visual_pooled)  # [B, d_memory]
        context_mem = self.context_encoder(
            torch.cat([context, soma], dim=-1)
        )  # [B, d_memory]

        crystallized = []
        for b in range(visual_tokens.shape[0]):
            if salience[b] > threshold:
                idx = self.memory_count % self.max_memories
                self.visual_memories[idx] = visual_mem[b]
                self.context_memories[idx] = context_mem[b]
                self.salience_scores[idx] = salience[b]
                self.memory_count += 1
                crystallized.append(b)

        return {
            'crystallized': crystallized,
            'memory_count': self.memory_count.item(),
        }

    def retrieve(
        self,
        query: torch.Tensor,  # [B, d_model] - text or visual query
        mode: str = 'visual',  # 'visual', 'context', or 'both'
        top_k: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieve relevant visual memories.
        """
        query_proj = self.query_proj(query)  # [B, d_memory]

        if mode == 'visual':
            scores = torch.matmul(query_proj, self.visual_memories.T)
        elif mode == 'context':
            scores = torch.matmul(query_proj, self.context_memories.T)
        else:  # both
            v_scores = torch.matmul(query_proj, self.visual_memories.T)
            c_scores = torch.matmul(query_proj, self.context_memories.T)
            scores = 0.5 * v_scores + 0.5 * c_scores

        # Weight by salience
        scores = scores * self.salience_scores.unsqueeze(0)

        # Top-k
        top_scores, top_indices = scores.topk(top_k, dim=-1)

        return {
            'indices': top_indices,
            'scores': top_scores,
            'visual_memories': self.visual_memories[top_indices],
            'context_memories': self.context_memories[top_indices],
        }
```

### 5. CrossModalAttention

Enables text and vision to attend to each other.

```python
class CrossModalAttention(nn.Module):
    """
    Bidirectional attention between text and visual modalities.

    - Text → Visual: "What in the image relates to this word?"
    - Visual → Text: "What words describe this patch?"
    """

    def __init__(self, config):
        self.d_model = config.d_model
        self.n_heads = config.n_cross_heads

        # Text attending to visual
        self.text_to_visual = nn.MultiheadAttention(
            config.d_model, config.n_cross_heads, batch_first=True
        )

        # Visual attending to text
        self.visual_to_text = nn.MultiheadAttention(
            config.d_model, config.n_cross_heads, batch_first=True
        )

        # Gating: how much to mix in cross-modal information
        self.text_gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid()
        )
        self.visual_gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid()
        )

        # Layer norms
        self.text_norm = nn.LayerNorm(config.d_model)
        self.visual_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        text_hidden: torch.Tensor,    # [B, T, d_model]
        visual_tokens: torch.Tensor,  # [B, V, d_model]
        text_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Cross-modal attention.

        Returns:
            text_enhanced: [B, T, d_model] text with visual context
            visual_enhanced: [B, V, d_model] visual with text context
            text_to_visual_attn: [B, T, V] attention weights
            visual_to_text_attn: [B, V, T] attention weights
        """
        # Text attends to visual
        text_cross, t2v_attn = self.text_to_visual(
            text_hidden, visual_tokens, visual_tokens,
            key_padding_mask=visual_mask
        )

        # Visual attends to text
        visual_cross, v2t_attn = self.visual_to_text(
            visual_tokens, text_hidden, text_hidden,
            key_padding_mask=text_mask
        )

        # Gated residual
        text_gate = self.text_gate(torch.cat([text_hidden, text_cross], dim=-1))
        text_enhanced = self.text_norm(text_hidden + text_gate * text_cross)

        visual_gate = self.visual_gate(torch.cat([visual_tokens, visual_cross], dim=-1))
        visual_enhanced = self.visual_norm(visual_tokens + visual_gate * visual_cross)

        return {
            'text_enhanced': text_enhanced,
            'visual_enhanced': visual_enhanced,
            'text_to_visual_attn': t2v_attn,
            'visual_to_text_attn': v2t_attn,
        }
```

### 6. VisualGrounding

Connects visual regions to language concepts.

```python
class VisualGrounding(nn.Module):
    """
    Ground language in visual regions.

    "The red ball" → highlight the red ball patches
    "Top left corner" → attend to top-left patches
    """

    def __init__(self, config):
        self.d_model = config.d_model

        # Text → visual region predictor
        self.region_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Spatial position encoding
        self.spatial_embed = nn.Parameter(
            torch.randn(1, config.max_patches, config.d_model)
        )

    def ground(
        self,
        text_query: torch.Tensor,     # [B, d_model] - e.g., "the red ball"
        visual_tokens: torch.Tensor,  # [B, V, d_model]
        patch_grid: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Find visual regions corresponding to text query.

        Returns:
            attention_map: [B, H, W] soft attention over image
            grounded_visual: [B, d_model] visual features for query
        """
        B, V, D = visual_tokens.shape
        H, W = patch_grid

        # Project query
        query = self.region_predictor(text_query)  # [B, d_model]

        # Compute attention over patches
        scores = torch.matmul(
            query.unsqueeze(1),  # [B, 1, d_model]
            visual_tokens.transpose(1, 2)  # [B, d_model, V]
        ).squeeze(1)  # [B, V]

        attention = F.softmax(scores, dim=-1)

        # Reshape to spatial
        # Skip start/end tokens if present
        if V == H * W + 2:
            attention_spatial = attention[:, 1:-1].view(B, H, W)
        else:
            attention_spatial = attention.view(B, H, W)

        # Weighted visual features
        grounded_visual = (attention.unsqueeze(-1) * visual_tokens).sum(dim=1)

        return {
            'attention_map': attention_spatial,
            'grounded_visual': grounded_visual,
            'attention_flat': attention,
        }
```

## Integration with MemoryAugmentedGPT

```python
class MemoryAugmentedGPT:
    def __init__(self, ..., use_vision=False, vision_config=None):
        if use_vision:
            self.vision = VisionModule(vision_config)

    def forward(
        self,
        input_ids,
        images: Optional[torch.Tensor] = None,  # [B, C, H, W] or [B, N_images, C, H, W]
        ...
    ):
        # 1. Encode text as usual
        text_embeds = self.embed_tokens(input_ids)

        # 2. Encode images if present
        if images is not None and self.vision is not None:
            visual_out = self.vision.encode(images)
            visual_tokens = visual_out['visual_tokens']  # [B, V, d_model]

            # 3. Compute visual surprise
            if self.self_state is not None:
                text_context = text_embeds.mean(dim=1)  # Simple pooling
                soma = self.self_state.get_soma()
                visual_surprise = self.vision.compute_surprise(
                    visual_tokens, text_context, soma
                )

                # 4. Update soma with visual signals
                visual_signals = self.vision.extract_signals(
                    visual_tokens, visual_surprise['patch_surprise']
                )
                # These get integrated in the normal self_state update

            # 5. Interleave or prepend visual tokens
            # Option A: Prepend (image, then text)
            combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

            # Option B: Interleave at <image> tokens
            # combined_embeds = self._interleave_visual(text_embeds, visual_tokens, input_ids)
        else:
            combined_embeds = text_embeds
            visual_tokens = None

        # 6. Run transformer with cross-modal attention
        hidden_states = self.gpt(
            inputs_embeds=combined_embeds,
            visual_tokens=visual_tokens,  # For cross-attention layers
            soma_q_bias=soma_q_bias,
        )

        # 7. Crystallize visual memories if salient
        if visual_tokens is not None and self.vision.memory is not None:
            salience = self._compute_visual_salience(visual_surprise, soma)
            self.vision.memory.crystallize(
                visual_tokens, text_context, soma, salience
            )

        return logits, hidden_states, memory_output
```

## Active Understanding System

The heart of the vision module: an iterative reasoning loop that doesn't stop until the image is understood.

### UnderstandingState

Tracks what is and isn't understood about the current image:

```python
@dataclass
class UnderstandingState:
    """Tracks comprehension of visual scene."""

    # Per-region understanding
    region_confidence: torch.Tensor    # [N_regions] how well each region is understood
    region_labels: List[str]           # Current best guess for each region
    region_uncertainty: torch.Tensor   # [N_regions] uncertainty type (ambiguous, novel, occluded, etc.)

    # Global scene understanding
    scene_coherence: float             # Does the scene make sense as a whole?
    unresolved_questions: List[str]    # Questions the model has about the image
    hypotheses: List[Dict]             # Current hypotheses about uncertain regions

    # Resolution attempts
    reasoning_steps: List[str]         # What reasoning has been tried
    questions_asked: List[str]         # Questions asked to user
    user_answers: List[str]            # Answers received

    def is_complete(self, threshold: float = 0.8) -> bool:
        """Is understanding sufficient?"""
        return (
            self.region_confidence.mean() > threshold and
            self.scene_coherence > threshold and
            len(self.unresolved_questions) == 0
        )
```

### ComprehensionChecker

Assesses understanding of each visual element:

```python
class ComprehensionChecker(nn.Module):
    """
    For each region/object, assess: Do I understand what this is?

    Understanding means:
    1. Can label it (what is it?)
    2. Can explain it (why is it here?)
    3. Fits with context (makes sense in scene)
    4. Connects to memory (seen similar before)
    """

    def __init__(self, config):
        self.d_model = config.d_model

        # Region-level understanding assessor
        self.understanding_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 4),  # [labelable, explainable, contextual, familiar]
        )

        # Uncertainty classifier: WHY don't I understand?
        self.uncertainty_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 5),  # [ambiguous, novel, occluded, blurry, contradictory]
        )

        # Scene coherence: does everything fit together?
        self.coherence_head = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,   # [B, N, d_model]
        scene_context: torch.Tensor,   # [B, d_model] global context
        memory_matches: torch.Tensor,  # [B, N] similarity to memory
    ) -> Dict[str, torch.Tensor]:
        """
        Assess comprehension of visual input.
        """
        B, N, D = visual_tokens.shape

        # Per-region understanding scores
        understanding = self.understanding_head(visual_tokens)  # [B, N, 4]
        understanding = torch.sigmoid(understanding)

        # Overall confidence = product of understanding factors
        confidence = understanding.prod(dim=-1)  # [B, N]

        # For low-confidence regions, classify the uncertainty type
        uncertainty_logits = self.uncertainty_classifier(visual_tokens)  # [B, N, 5]
        uncertainty_type = uncertainty_logits.argmax(dim=-1)  # [B, N]

        # Scene coherence
        coherence = self.coherence_head(scene_context)  # [B, 1]

        return {
            'confidence': confidence,
            'understanding_factors': understanding,
            'uncertainty_type': uncertainty_type,
            'uncertainty_scores': F.softmax(uncertainty_logits, dim=-1),
            'scene_coherence': coherence.squeeze(-1),
            'needs_resolution': confidence < 0.5,  # Boolean mask
        }
```

### UncertaintyResolver

When something isn't understood, decide how to resolve it:

```python
class UncertaintyResolver(nn.Module):
    """
    Decide how to resolve uncertainty:
    1. ASK USER - "What is this object?"
    2. REASON - Use context and memory to figure it out
    3. HYPOTHESIZE - Form tentative belief, continue with uncertainty
    4. IGNORE - Not important enough to resolve (background detail)
    """

    def __init__(self, config):
        self.d_model = config.d_model

        # Resolution strategy selector
        self.strategy_head = nn.Sequential(
            nn.Linear(config.d_model * 2 + config.d_soma, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 4),  # [ask, reason, hypothesize, ignore]
        )

        # Question generator (for ASK strategy)
        self.question_generator = QuestionGenerator(config)

        # Reasoning module (for REASON strategy)
        self.reasoner = VisualReasoner(config)

        # Hypothesis generator (for HYPOTHESIZE strategy)
        self.hypothesis_head = nn.Linear(config.d_model, config.d_model)

    def resolve(
        self,
        uncertain_regions: torch.Tensor,    # [B, K, d_model] regions needing resolution
        uncertainty_type: torch.Tensor,     # [B, K] why uncertain
        scene_context: torch.Tensor,        # [B, d_model]
        soma: torch.Tensor,                 # [B, d_soma] current internal state
        visual_memory: 'VisualMemory',
    ) -> Dict[str, Any]:
        """
        Attempt to resolve uncertainties.
        """
        B, K, D = uncertain_regions.shape

        # Decide strategy for each uncertain region
        strategy_input = torch.cat([
            uncertain_regions,
            scene_context.unsqueeze(1).expand(-1, K, -1),
            soma.unsqueeze(1).expand(-1, K, -1),
        ], dim=-1)

        strategy_logits = self.strategy_head(strategy_input)  # [B, K, 4]
        strategy = strategy_logits.argmax(dim=-1)  # [B, K]

        resolutions = []

        for b in range(B):
            for k in range(K):
                strat = strategy[b, k].item()
                region = uncertain_regions[b, k]
                unc_type = uncertainty_type[b, k].item()

                if strat == 0:  # ASK USER
                    question = self.question_generator(
                        region, unc_type, scene_context[b]
                    )
                    resolutions.append({
                        'strategy': 'ask',
                        'question': question,
                        'region_idx': k,
                        'awaiting_response': True,
                    })

                elif strat == 1:  # REASON
                    # Try to figure it out using context + memory
                    reasoning_result = self.reasoner(
                        region,
                        scene_context[b],
                        visual_memory
                    )
                    resolutions.append({
                        'strategy': 'reason',
                        'conclusion': reasoning_result['conclusion'],
                        'confidence': reasoning_result['confidence'],
                        'reasoning_chain': reasoning_result['steps'],
                        'region_idx': k,
                    })

                elif strat == 2:  # HYPOTHESIZE
                    # Form tentative belief
                    hypothesis = self.hypothesis_head(region)
                    resolutions.append({
                        'strategy': 'hypothesize',
                        'hypothesis': hypothesis,
                        'uncertainty_acknowledged': True,
                        'region_idx': k,
                    })

                else:  # IGNORE
                    resolutions.append({
                        'strategy': 'ignore',
                        'reason': 'low_importance',
                        'region_idx': k,
                    })

        return {
            'resolutions': resolutions,
            'questions_to_ask': [r for r in resolutions if r['strategy'] == 'ask'],
            'reasoning_performed': [r for r in resolutions if r['strategy'] == 'reason'],
        }
```

### Visual Prediction Failure → Question

Visual questions arise from visual prediction failures. This connects to the unified questioning system (see `questioning_system.md`):

```python
class VisualPredictionFailure(PredictionFailure):
    """
    Visual prediction failure: expected to see X, actually see Y.

    This is the visual instantiation of the general prediction failure concept.
    """
    predicted_visual: torch.Tensor    # What was expected
    observed_visual: torch.Tensor     # What was seen
    region: Tuple[int, int]           # Where in the image (row, col)
    patch_indices: List[int]          # Which patches
    recognizable: bool                # Can the observed content be identified?
    confidence: float                 # How confident in what we see?

    def to_question(self) -> 'Question':
        """
        Convert visual prediction failure to question.

        The question type depends on the nature of the failure:
        - Unrecognizable → "What is this?"
        - Unexpected → "Why is X here instead of Y?"
        - Ambiguous → "Is this A or B?"
        """
        from questioning_system import Question, QuestionType

        if not self.recognizable:
            return Question(
                source_type='visual',
                source_failure=self,
                type='visual_identification',
                focus=self.observed_visual,
                uncertainty_type='novel',
                information_need='identity',
                # Visual-specific: can point to region
                visual_region=self.region,
                can_highlight=True,
            )
        elif self.confidence < 0.5:
            return Question(
                source_type='visual',
                source_failure=self,
                type='visual_confirmation',
                focus=self.observed_visual,
                uncertainty_type='ambiguous',
                information_need='confirmation',
                visual_region=self.region,
                can_highlight=True,
            )
        else:
            return Question(
                source_type='visual',
                source_failure=self,
                type='visual_explanation',
                focus=self.observed_visual,
                uncertainty_type='unexpected',
                information_need='explanation',
                visual_region=self.region,
                expected=self.predicted_visual,
                can_highlight=True,
            )


class VisualQuestionGenerator(nn.Module):
    """
    Generate questions from visual prediction failures.

    Extends the base QuestionGenerator with visual-specific capabilities:
    - Region grounding (point to where in image)
    - Visual context (what else is in the scene)
    - Spatial relationships (relative to other objects)
    """

    def __init__(self, config):
        self.d_model = config.d_model

        # Question type selector (visual-specific types)
        self.question_type = nn.Linear(config.d_model, 7)
        # Types: [what_is, what_doing, why_here, what_relation, what_means, is_this, describe]

        # Visual grounding for question
        self.region_descriptor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.d_model),
        )

        # Spatial location encoding
        self.location_encoder = nn.Embedding(9, config.d_model // 4)  # 3x3 grid

        # Templates with visual grounding
        self.templates = {
            'what_is': "What is {VISUAL_DESCRIPTION} in the {LOCATION} of the image?",
            'what_doing': "What is {SUBJECT} doing in this image?",
            'why_here': "Why is there {OBJECT} {SPATIAL_RELATION}?",
            'what_relation': "What is the relationship between {A} and {B}?",
            'what_means': "What does {SYMBOL} signify here?",
            'is_this': "Is {REGION_DESCRIPTION} a {HYPOTHESIS}?",
            'describe': "Can you describe what you see {LOCATION}?",
        }

    def forward(
        self,
        visual_failure: VisualPredictionFailure,
        scene_context: torch.Tensor,
        all_visual_tokens: torch.Tensor,
    ) -> 'Question':
        """Generate a grounded visual question."""

        # Get region embedding
        region_embed = self.region_descriptor(visual_failure.observed_visual)

        # Encode spatial location
        location_idx = self._region_to_location_idx(visual_failure.region)
        location_embed = self.location_encoder(torch.tensor(location_idx))

        # Combine for question type selection
        combined = torch.cat([region_embed, location_embed, scene_context], dim=-1)
        q_type_logits = self.question_type(combined)
        q_type_idx = q_type_logits.argmax().item()
        q_type = list(self.templates.keys())[q_type_idx]

        # Create Question object (from questioning_system.md)
        question = visual_failure.to_question()
        question.type = q_type
        question.template = self.templates[q_type]
        question.grounding = {
            'region': visual_failure.region,
            'location_description': self._idx_to_location_name(location_idx),
            'visual_context': scene_context,
        }

        return question

    def _region_to_location_idx(self, region: Tuple[int, int]) -> int:
        """Convert region coordinates to 3x3 grid index."""
        # Simplified: map to 9 positions
        row, col = region
        # Normalize and bin
        return min(2, row // 3) * 3 + min(2, col // 3)

    def _idx_to_location_name(self, idx: int) -> str:
        """Convert grid index to natural language."""
        names = [
            'top-left', 'top-center', 'top-right',
            'middle-left', 'center', 'middle-right',
            'bottom-left', 'bottom-center', 'bottom-right',
        ]
        return names[idx]
```

### VisualReasoner

Attempts to figure out uncertain elements through reasoning:

```python
class VisualReasoner(nn.Module):
    """
    Multi-step reasoning to understand uncertain visual elements.

    Reasoning strategies:
    1. CONTEXTUAL - What else is in the scene that gives clues?
    2. MEMORY - Have I seen similar things before?
    3. COMPOSITIONAL - What are the parts? What do parts suggest?
    4. ELIMINATION - What can I rule out?
    5. ANALOGICAL - What is this similar to?
    """

    def __init__(self, config):
        self.d_model = config.d_model
        self.max_reasoning_steps = config.max_reasoning_steps

        # Reasoning step selector
        self.step_selector = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 5),  # 5 reasoning strategies
        )

        # Contextual reasoning: look at surrounding regions
        self.context_attention = nn.MultiheadAttention(
            config.d_model, num_heads=4, batch_first=True
        )

        # Memory lookup
        self.memory_query = nn.Linear(config.d_model, config.d_memory)

        # Conclusion confidence
        self.conclusion_confidence = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        uncertain_region: torch.Tensor,   # [d_model]
        scene_context: torch.Tensor,      # [d_model]
        visual_memory: 'VisualMemory',
        all_regions: Optional[torch.Tensor] = None,  # [N, d_model]
    ) -> Dict[str, Any]:
        """
        Multi-step reasoning about uncertain region.
        """
        current_understanding = uncertain_region
        reasoning_steps = []

        for step in range(self.max_reasoning_steps):
            # Decide what reasoning to try
            step_input = torch.cat([current_understanding, scene_context], dim=-1)
            strategy = self.step_selector(step_input).argmax().item()

            if strategy == 0:  # CONTEXTUAL
                # Attend to other regions for clues
                if all_regions is not None:
                    attended, _ = self.context_attention(
                        current_understanding.unsqueeze(0).unsqueeze(0),
                        all_regions.unsqueeze(0),
                        all_regions.unsqueeze(0),
                    )
                    current_understanding = current_understanding + attended.squeeze()
                    reasoning_steps.append({
                        'strategy': 'contextual',
                        'insight': 'looked at surrounding context'
                    })

            elif strategy == 1:  # MEMORY
                # Query visual memory
                query = self.memory_query(current_understanding)
                retrieved = visual_memory.retrieve(query.unsqueeze(0), top_k=3)
                if retrieved['scores'].max() > 0.5:
                    # Found relevant memory
                    current_understanding = current_understanding + retrieved['visual_memories'][0, 0]
                    reasoning_steps.append({
                        'strategy': 'memory',
                        'insight': 'found similar visual memory',
                        'similarity': retrieved['scores'][0, 0].item(),
                    })

            elif strategy == 2:  # COMPOSITIONAL
                # Analyze parts (would involve attention to sub-patches)
                reasoning_steps.append({
                    'strategy': 'compositional',
                    'insight': 'analyzed component parts'
                })

            elif strategy == 3:  # ELIMINATION
                # Rule out possibilities
                reasoning_steps.append({
                    'strategy': 'elimination',
                    'insight': 'ruled out unlikely interpretations'
                })

            elif strategy == 4:  # ANALOGICAL
                # Find analogies
                reasoning_steps.append({
                    'strategy': 'analogical',
                    'insight': 'found analogous concept'
                })

            # Check if we've reached sufficient confidence
            confidence = self.conclusion_confidence(current_understanding)
            if confidence > 0.8:
                break

        return {
            'conclusion': current_understanding,
            'confidence': confidence.item(),
            'steps': reasoning_steps,
            'n_steps': len(reasoning_steps),
        }
```

### VisualUnderstandingLoop

The main loop that drives understanding:

```python
class VisualUnderstandingLoop(nn.Module):
    """
    Iterative understanding process.

    Keep processing until:
    1. Everything is understood (confidence > threshold)
    2. Max iterations reached
    3. User answers pending questions
    """

    def __init__(self, config):
        self.comprehension_checker = ComprehensionChecker(config)
        self.uncertainty_resolver = UncertaintyResolver(config)
        self.max_iterations = config.max_understanding_iterations
        self.confidence_threshold = config.understanding_threshold

    def understand(
        self,
        visual_tokens: torch.Tensor,
        scene_context: torch.Tensor,
        soma: torch.Tensor,
        visual_memory: 'VisualMemory',
        user_interface: Optional['UserContactInterface'] = None,
    ) -> Dict[str, Any]:
        """
        Run understanding loop until image is comprehended.
        """
        state = UnderstandingState(
            region_confidence=torch.zeros(visual_tokens.shape[1]),
            region_labels=[],
            region_uncertainty=torch.zeros(visual_tokens.shape[1]),
            scene_coherence=0.0,
            unresolved_questions=[],
            hypotheses=[],
            reasoning_steps=[],
            questions_asked=[],
            user_answers=[],
        )

        for iteration in range(self.max_iterations):
            # Check comprehension
            comprehension = self.comprehension_checker(
                visual_tokens, scene_context,
                visual_memory.compute_similarity(visual_tokens)
            )

            state.region_confidence = comprehension['confidence']
            state.scene_coherence = comprehension['scene_coherence']

            # Are we done?
            if state.is_complete(self.confidence_threshold):
                return {
                    'understood': True,
                    'state': state,
                    'iterations': iteration + 1,
                }

            # Find uncertain regions
            uncertain_mask = comprehension['needs_resolution']
            if not uncertain_mask.any():
                break

            uncertain_regions = visual_tokens[uncertain_mask]

            # Try to resolve uncertainties
            resolution = self.uncertainty_resolver.resolve(
                uncertain_regions.unsqueeze(0),
                comprehension['uncertainty_type'][uncertain_mask].unsqueeze(0),
                scene_context.unsqueeze(0),
                soma.unsqueeze(0),
                visual_memory,
            )

            # Process resolutions
            for res in resolution['resolutions']:
                if res['strategy'] == 'ask':
                    # Generate question for user
                    if user_interface is not None:
                        state.questions_asked.append(res['question'])
                        state.unresolved_questions.append(res['question'])
                        # Would trigger user contact via action module

                elif res['strategy'] == 'reason':
                    state.reasoning_steps.append(res['reasoning_chain'])
                    # Update confidence for this region
                    if res['confidence'] > 0.7:
                        state.region_confidence[res['region_idx']] = res['confidence']

                elif res['strategy'] == 'hypothesize':
                    state.hypotheses.append({
                        'region': res['region_idx'],
                        'hypothesis': res['hypothesis'],
                    })
                    # Partial confidence
                    state.region_confidence[res['region_idx']] = 0.6

        return {
            'understood': state.is_complete(self.confidence_threshold),
            'state': state,
            'iterations': self.max_iterations,
            'remaining_questions': state.unresolved_questions,
        }
```

### Integration with Action Module

When the vision module needs to ask questions, it interfaces with the action module:

```python
def visual_understanding_with_action(
    self,
    image: torch.Tensor,
    action_module: 'ActionModule',
) -> Dict[str, Any]:
    """
    Understand image, asking questions if needed.
    """
    visual_tokens = self.vision.encode(image)

    # Run understanding loop
    understanding = self.vision.understanding_loop.understand(
        visual_tokens['visual_tokens'],
        context,
        soma,
        self.vision.memory,
    )

    # If there are questions to ask, trigger action module
    if understanding['remaining_questions']:
        for question in understanding['remaining_questions']:
            # Use action module to contact user
            action_result = action_module.execute(
                Action(name='contact_user', tier=1),
                {
                    'soma': soma,
                    'hidden_state': context,
                    'action_reason': 'visual_uncertainty',
                    'message_content': question,
                }
            )

            # When user responds, feed back into understanding
            if action_result.get('user_response'):
                understanding = self.vision.integrate_user_response(
                    understanding,
                    question,
                    action_result['user_response'],
                )

    return understanding
```

## Visual Reasoning Modes

### Mode 1: Describe
"What is in this image?"

```python
def describe_image(self, image):
    """Generate text description of image."""
    visual_tokens = self.vision.encode(image)

    # Start with visual tokens, generate text autoregressively
    prompt_ids = self.tokenizer.encode("<describe>")

    # The model "reasons" by attending to visual tokens
    # while generating descriptive text
    return self.generate(
        prompt_ids,
        visual_context=visual_tokens,
        max_tokens=100
    )
```

### Mode 2: Question Answering
"What color is the car?"

```python
def visual_qa(self, image, question):
    """Answer question about image."""
    visual_tokens = self.vision.encode(image)
    question_ids = self.tokenizer.encode(question)

    # Ground question in visual regions
    grounding = self.vision.ground(
        question_embedding,
        visual_tokens
    )

    # Generate answer attending to grounded regions
    return self.generate(
        question_ids,
        visual_context=visual_tokens,
        attention_bias=grounding['attention_map'],  # Bias toward relevant regions
    )
```

### Mode 3: Comparison
"How are these two images different?"

```python
def compare_images(self, image1, image2):
    """Compare two images."""
    vis1 = self.vision.encode(image1)
    vis2 = self.vision.encode(image2)

    # Compute difference in visual space
    diff = vis1['visual_tokens'] - vis2['visual_tokens']

    # Also retrieve memories of similar images
    mem1 = self.vision.memory.retrieve(vis1['visual_tokens'].mean(dim=1))
    mem2 = self.vision.memory.retrieve(vis2['visual_tokens'].mean(dim=1))

    # Generate comparison
    return self.generate(
        "<compare>",
        visual_context=torch.cat([vis1['visual_tokens'], vis2['visual_tokens']], dim=1),
    )
```

### Mode 4: Spatial Reasoning
"What is to the left of the dog?"

```python
def spatial_reasoning(self, image, query):
    """Answer spatial questions about image."""
    visual_tokens = self.vision.encode(image)

    # Parse spatial relation from query
    relation, target = self.parse_spatial_query(query)

    # Ground target ("dog") in image
    target_grounding = self.vision.ground(target, visual_tokens)

    # Apply spatial shift based on relation ("left of")
    shifted_attention = self.apply_spatial_relation(
        target_grounding['attention_map'],
        relation
    )

    # Describe what's in the shifted region
    return self.generate(
        query,
        visual_context=visual_tokens,
        attention_bias=shifted_attention,
    )
```

### Mode 5: Visual Memory Recall
"Have I seen something like this before?"

```python
def visual_recall(self, image):
    """Check visual memory for similar experiences."""
    visual_tokens = self.vision.encode(image)

    # Query visual memory
    retrieved = self.vision.memory.retrieve(
        visual_tokens.mean(dim=1),
        mode='visual',
        top_k=3
    )

    if retrieved['scores'].max() > threshold:
        # Found similar memory
        return {
            'familiar': True,
            'similar_context': retrieved['context_memories'],
            'similarity': retrieved['scores'],
        }
    else:
        return {'familiar': False, 'novelty': 'high'}
```

## Training

### Phase 1: Visual Encoding
Train visual encoder on image reconstruction / contrastive learning:

```python
# Contrastive: match image to caption
visual_loss = contrastive_loss(
    visual_tokens.mean(dim=1),  # Image embedding
    text_embedding,              # Caption embedding
)
```

### Phase 2: Cross-Modal Alignment
Train text-visual attention:

```python
# Image captioning
caption_loss = cross_entropy(
    predicted_caption_tokens,
    ground_truth_caption
)

# Visual grounding
grounding_loss = bce_loss(
    predicted_attention_map,
    ground_truth_region
)
```

### Phase 3: Visual Surprise Integration
Train visual surprise to correlate with memorability:

```python
# Surprise should predict what's remembered
memory_prediction_loss = bce_loss(
    visual_surprise['surprise'],
    was_image_remembered  # From human annotation or later recall
)
```

### Phase 4: Visual Memory
Train crystallization and retrieval:

```python
# Retrieval should find semantically similar images
retrieval_loss = contrastive_loss(
    retrieved_visual,
    query_visual,
    negatives
)
```

## Soma Integration

Visual input affects soma through dedicated signals:

| Signal | Visual Source | Effect on Soma |
|--------|--------------|----------------|
| Surprise | Prediction error from expected visual | High → alert, update self-model |
| Arousal | Motion, contrast, visual complexity | High → energized processing |
| Valence | Learned aesthetic/threat detection | Positive/negative affect |
| Novelty | Similarity to visual memories | High → exploration mode |
| Certainty | Consistency of visual features | Low → uncertainty state |
| Engagement | Attention to visual regions | High → focused processing |

```python
def compute_visual_soma_signals(self, visual_tokens, visual_surprise):
    """Convert visual input to soma signals."""
    signals = torch.zeros(B, 6)

    # Surprise: from visual prediction error
    signals[:, 0] = visual_surprise['surprise']

    # Arousal: from visual complexity/motion
    signals[:, 1] = self.arousal_detector(visual_tokens)

    # Valence: from learned aesthetic model
    signals[:, 2] = self.valence_detector(visual_tokens)

    # Novelty: inverse similarity to visual memories
    signals[:, 3] = 1.0 - self.memory_similarity(visual_tokens)

    # Certainty: consistency of visual features
    signals[:, 4] = self.certainty_estimator(visual_tokens)

    # Engagement: attention entropy (focused vs. diffuse)
    signals[:, 5] = self.engagement_from_attention(attn_weights)

    return signals
```

## Open Questions

### Encoding & Architecture

1. **Encoder choice**: Train from scratch or use pretrained (CLIP, DINOv2)? Pretrained is faster but may have different "visual vocabulary."

2. **Token interleaving**: Prepend visual tokens? Interleave at special tokens? Separate streams with cross-attention?

3. **Resolution vs. tokens**: More patches = more detail but more compute. Dynamic resolution based on image complexity?

4. **Visual memory format**: Store compressed tokens? Raw features? Reconstructable?

### Understanding-Driven

5. **When to ask vs. reason**: How to decide between asking the user and trying to figure it out? Cost of asking (interruption) vs. cost of being wrong?

6. **Confidence calibration**: How do we know if the model's "I understand this" is accurate? Need calibration training?

7. **Incomplete understanding**: When is partial understanding acceptable? Not everything in every image needs to be understood.

8. **Question quality**: How to ensure generated questions are useful and not annoying? Rate limiting? Relevance scoring?

9. **Reasoning depth**: How many reasoning steps before giving up? Diminishing returns vs. persistence?

10. **User response integration**: How does a user's answer update the model's understanding? Direct injection vs. re-processing?

### Experiential Integration

11. **Visual imagination**: Can the model "imagine" visual content during synthesis? Generate visual tokens from soma?

12. **Multi-image**: How to handle sequences of images? Video? Different treatment for temporal vs. spatial?

13. **Understanding → Soma**: How does "figuring something out" feel? Satisfaction signal when uncertainty resolves?

14. **Confusion as signal**: Should persistent confusion affect soma negatively? Drive help-seeking behavior?

15. **Visual curiosity**: Should the model develop preferences for certain types of visual content based on experience?

## Implementation Phases

### Phase 1: Basic Vision
- [ ] Implement VisualEncoder (ViT-style)
- [ ] Add visual tokens to transformer input
- [ ] Basic image description generation

### Phase 2: Visual Surprise
- [ ] Implement VisualSurprise
- [ ] Connect to soma signals
- [ ] Visual novelty detection

### Phase 3: Understanding Loop
- [ ] Implement ComprehensionChecker
- [ ] Implement UncertaintyResolver (ASK/REASON/HYPOTHESIZE/IGNORE)
- [ ] Implement QuestionGenerator
- [ ] Implement VisualReasoner with multi-step reasoning
- [ ] Wire understanding loop to forward pass

### Phase 4: Cross-Modal
- [ ] Implement CrossModalAttention
- [ ] Text-grounded visual attention
- [ ] Visual question answering

### Phase 5: Visual Memory
- [ ] Implement VisualMemory
- [ ] Crystallize salient visual moments
- [ ] Cross-modal retrieval

### Phase 6: Action Integration
- [ ] Connect QuestionGenerator to ActionModule
- [ ] Handle user responses and feed back into understanding
- [ ] Understanding state → soma (curiosity when uncertain, satisfaction when resolved)

### Phase 7: Full Integration
- [ ] Soma-modulated visual processing
- [ ] All visual reasoning modes
- [ ] Multi-image support
- [ ] End-to-end training with understanding objectives
