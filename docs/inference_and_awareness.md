# Inference & News Awareness System

This document describes the inference capabilities of the Memory-Augmented GPT system, including the continuous news awareness daemon.

## Overview

The memory-augmented architecture supports two primary inference modes:

1. **Interactive Inference** (`inference_memory.py`) - Process text, accumulate memories, generate continuations
2. **News Awareness Daemon** (`news_awareness.py`) - Continuous monitoring of global news with automatic knowledge crystallization

Both systems leverage the episodic memory to build persistent knowledge that can be queried and investigated.

---

## Inference Memory (`inference_memory.py`)

### Purpose

Run the trained Memory-Augmented GPT in inference mode with:
- Persistent episodic memory that accumulates across documents
- Memory-augmented text generation
- Semantic querying of accumulated knowledge
- Save/load memory state for continuity

### Usage

```bash
# Interactive mode - explore and query memories
python inference_memory.py --interactive

# Process a document and accumulate memories
python inference_memory.py --input_file document.txt --save_memory memories.pt

# Process multiple documents
python inference_memory.py --input_dir ./documents/ --extensions ".txt,.md"

# Generate text using accumulated memories
python inference_memory.py --generate "The policy implications" --max_new_tokens 100

# Load previous memory state and continue
python inference_memory.py --load_memory memories.pt --interactive

# Process without crystallizing (read-only memory mode)
python inference_memory.py --input_file new_doc.txt --no_crystallize
```

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/process <text>` | Process text and crystallize memories |
| `/generate <prompt>` | Generate continuation using memory |
| `/query <text>` | Find similar memories |
| `/memory` | Show memory summary |
| `/stats` | Show processing statistics |
| `/save <path>` | Save memory state |
| `/load <path>` | Load memory state |
| `/clear` | Clear all memories |
| `/file <path>` | Process a file |
| `/quit` | Exit |

### Key Classes

#### `MemoryInference`

Main wrapper for inference operations:

```python
from inference_memory import MemoryInference

# Initialize
inference = MemoryInference(
    checkpoint_path="memory_augmented_output/memory_gpt_epoch_1.pt",
    device="cuda",
    memory_capacity=1000,
    crystallization_threshold=0.2,
)

# Process text
result = inference.process_text(
    "Your text here...",
    crystallize=True,   # Store surprising moments
    use_memory=True,    # Retrieve from existing memories
)

# Generate with memory
output = inference.generate(
    "The implications are",
    max_new_tokens=100,
    use_memory=True,
)

# Query memory
results = inference.query_memory("climate change", top_k=5)

# Save/load state
inference.save_memory("my_memories.pt")
inference.load_memory("my_memories.pt")
```

### Memory Accumulation Flow

```
Document 1 → Process → Crystallize surprising moments → Memory Bank grows
                ↓
Document 2 → Process → Retrieve relevant context + Crystallize → Memory Bank grows
                ↓
Document N → Process → Rich retrieval from accumulated knowledge
                ↓
            Query Interface ← User investigates accumulated knowledge
```

---

## News Awareness System (`news_awareness.py`)

### Purpose

A production system for continuous news monitoring that:
- Fetches articles from configurable RSS feeds
- Processes news through the memory-augmented model
- Crystallizes important/surprising news into episodic memory
- Maintains persistent state across restarts
- Provides query interface for investigating accumulated knowledge

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    News Awareness Daemon                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   RSS Feeds ──→ Fetcher ──→ Deduplication ──→ Processor     │
│   (BBC, Reuters,          (hash-based)      (Memory-Aug GPT)│
│    NPR, Nature...)                               │          │
│                                                  ↓          │
│                                          ┌──────────────┐   │
│                                          │  Episodic    │   │
│                                          │   Memory     │   │
│                                          │  (5000 cap)  │   │
│                                          └──────────────┘   │
│                                                  ↑          │
│   Query Interface ←──────────────────────────────┘          │
│   (CLI / API)                                               │
│                                                              │
│   Persistence: awareness_state/                             │
│   - episodic_memory.pt                                      │
│   - awareness_state.pkl                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Usage

```bash
# Run as continuous daemon (fetches every 5 minutes)
python news_awareness.py --daemon

# Interactive exploration of accumulated knowledge
python news_awareness.py --interactive

# Query specific topics
python news_awareness.py --query "climate change"

# Show system summary
python news_awareness.py --summary

# Single fetch cycle (for testing)
python news_awareness.py --fetch_once

# Custom intervals
python news_awareness.py --daemon --fetch_interval 600 --save_interval 1200
```

### Default News Sources

| Category | Sources |
|----------|---------|
| General | BBC World, Reuters, AP News, Al Jazeera |
| US | NPR News, PBS NewsHour |
| Tech | Ars Technica, Hacker News |
| Science | Nature News, Science Daily |
| Economics | Financial Times World |

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/query <text>` | Query accumulated knowledge |
| `/summary` | Show system summary |
| `/recent` | Show recent crystallizations |
| `/briefing [topic]` | Generate news briefing |
| `/sources` | Show source statistics |
| `/important` | Show high-surprise articles |
| `/fetch` | Manually fetch and process news |
| `/save` | Save current state |
| `/memory` | Show memory details |
| `/quit` | Exit |

### Key Classes

#### `NewsAwarenessSystem`

Main system orchestrator:

```python
from news_awareness import NewsAwarenessSystem

# Initialize
system = NewsAwarenessSystem(
    checkpoint_path="memory_augmented_output/memory_gpt_epoch_1.pt",
    state_dir="awareness_state",
    device="cuda",
    memory_capacity=5000,
    crystallization_threshold=0.3,
    fetch_interval=300,   # 5 minutes
    save_interval=600,    # 10 minutes
)

# Run daemon (blocking)
system.run_daemon()

# Or manual control
articles = system.fetch_feeds()
results = system.process_batch(articles)

# Query
results = system.query("artificial intelligence", top_k=10)

# Get summary
summary = system.get_summary()

# Generate briefing
briefing = system.generate_briefing(topic="technology")
```

#### `NewsArticle`

Data class for articles:

```python
@dataclass
class NewsArticle:
    id: str                    # Hash of URL
    title: str
    summary: str
    url: str
    source: str
    published: Optional[datetime]
    fetched: datetime
    processed: bool = False
    crystallized: bool = False
    surprise_score: float = 0.0
```

### State Persistence

All state is saved to the `state_dir` (default: `awareness_state/`):

| File | Contents |
|------|----------|
| `episodic_memory.pt` | Crystallized memories (PyTorch tensor) |
| `awareness_state.pkl` | Article history, processed IDs, statistics |

The system automatically:
- Saves state at configurable intervals
- Resumes from saved state on restart
- Deduplicates articles by URL hash

### Salience & Crystallization

Articles are crystallized based on the model's **surprise score**:

- **High salience (>1.0)**: Very surprising content - novel information the model hasn't seen
- **Medium salience (0.5-1.0)**: Moderately surprising - related to known patterns but with new elements
- **Low salience (<0.5)**: Expected content - similar to previously seen patterns

The `crystallization_threshold` parameter controls the minimum salience for memory storage.

### Monitoring & Logging

Logs are written to:
- Console (INFO level)
- `news_awareness.log` file

Example log output:
```
2024-01-05 13:53:00 - Crystallized [FT World] EU deference to Trump... (surprise=1.476)
2024-01-05 13:53:01 - Batch complete: 170 articles, 59 memories, avg_surprise=0.342
2024-01-05 13:53:02 - Saved state: 96 memories
```

### Production Deployment

For production deployment:

```bash
# Run with nohup
nohup python news_awareness.py --daemon > /dev/null 2>&1 &

# Or with systemd service
# Create /etc/systemd/system/news-awareness.service

# Or with Docker
docker run -d \
  -v ./awareness_state:/app/awareness_state \
  -v ./checkpoints:/app/checkpoints \
  news-awareness --daemon
```

### Custom Feed Configuration

```python
CUSTOM_FEEDS = [
    ("My Source 1", "https://example.com/rss"),
    ("My Source 2", "https://other.com/feed.xml"),
]

system = NewsAwarenessSystem(
    checkpoint_path="...",
    feeds=CUSTOM_FEEDS,
)
```

---

## Memory Query Semantics

When querying memory, the system:

1. Encodes the query text through the model
2. Extracts the memory query embedding
3. Computes cosine similarity against all stored episode contents
4. Returns top-k matches with similarity scores

**Note**: Query quality depends on training. Early in training, retrieval may not be semantically meaningful. With more training (positive retrieval benefit), queries become more accurate.

---

## Integration with Training

The inference systems use checkpoints from `train_memory_augmented.py`:

```python
# Training produces:
# - memory_augmented_output/memory_gpt_epoch_1.pt

# Checkpoint contains:
# - memory_gpt_state_dict: Model weights
# - episodic_memory: Crystallized memories from training
# - optimizer_state_dict: For resuming training
# - config: Model configuration
# - args: Training arguments
```

---

## Troubleshooting

### Low retrieval quality
- Train longer until `retrieval_benefit > 0` on evaluation
- Increase `contrastive_weight` during training
- Lower `crystallization_threshold` to store more memories

### Memory growing too large
- Increase `crystallization_threshold`
- Reduce `memory_capacity`
- Enable memory decay (built into EpisodicMemory)

### CUDA out of memory
- Reduce `memory_capacity`
- Use `--device cpu` for inference
- Process shorter chunks

### Feedparser errors
```bash
pip install feedparser
```

---

## Future Enhancements

- [ ] REST API for querying
- [ ] Web dashboard for monitoring
- [ ] Semantic consolidation (cluster similar memories)
- [ ] Multi-modal support (images from news)
- [ ] Alerting on high-surprise events
- [ ] Integration with notification systems
