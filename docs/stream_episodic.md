# Episodic Stream

What happened — discrete events worth remembering.

**Role**: The episodic stream stores specific events that crystallized from the experiential stream. It provides the system with autobiographical memory — the ability to remember *what happened* at *specific times*.

---

## 1. Core Concept

Not everything we experience becomes a memory. Most moments flow past unrecorded. Only salient events — high surprise, high affect, high relevance — crystallize into discrete episodes.

Properties:
- **Selective**: Only salient moments are stored
- **Contextual**: Each episode includes when/where/what context
- **Retrievable**: Can be queried by similarity
- **Decaying**: Unused memories fade over time
- **Consolidatable**: Patterns across episodes become semantic knowledge

```
Experiential Stream ──crystallize──► Episode ──store──► Episodic Buffer
                                                              │
                                                              ▼
                                                     retrieve by query
                                                              │
                                                              ▼
                                                    influence current processing
```

---

## 2. Formal Definition

### 2.1 Episode Structure

```python
@dataclass
class Episode:
    """A discrete memory of something that happened."""

    # Identity
    id: int                    # unique identifier
    timestamp: int             # when (global step / position)

    # Content
    content: Tensor            # [d_model] - what happened (embedding)
    context: Tensor            # [d_model] - surrounding state

    # For retrieval
    key: Tensor                # [d_key] - retrieval key (may differ from content)

    # Metadata
    salience: float            # how important at encoding time
    valence: float             # emotional valence [-1, 1]
    arousal: float             # emotional arousal [0, 1]

    # Lifecycle
    creation_time: int         # when created
    last_access_time: int      # when last retrieved
    access_count: int          # how often retrieved
    strength: float            # memory strength (decays without access)

    # Source tracking
    source_tokens: Optional[List[int]]  # original token IDs (if stored)
    source_document: Optional[str]      # document identifier


@dataclass
class EpisodeCandidate:
    """A potential episode before crystallization decision."""
    content: Tensor
    context: Tensor
    timestamp: int
    salience: float
    affect: Tuple[float, float]  # (valence, arousal)
```

### 2.2 Buffer Structure

```python
@dataclass
class EpisodicBuffer:
    """The storage structure for episodes."""

    episodes: List[Episode]      # stored memories
    capacity: int                # maximum episodes

    # Indices for fast retrieval
    key_index: Tensor            # [n_episodes, d_key] - stacked keys
    temporal_index: Dict[int, int]  # timestamp → episode id

    # Statistics
    total_stored: int            # lifetime count
    total_evicted: int           # lifetime evictions
    total_consolidated: int      # lifetime consolidations
```

---

## 3. Architecture

### 3.1 Main Module

```python
class EpisodicStream(nn.Module):
    """
    Episodic memory: stores and retrieves discrete events.

    Operations:
    - Crystallize: experience → episode
    - Store: add episode to buffer
    - Retrieve: query → relevant episodes
    - Decay: weaken unused memories
    - Evict: remove weakest memories
    - Consolidate: episodes → semantic knowledge
    """

    def __init__(self, config: EpisodicConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_key = config.d_key or config.d_model
        self.capacity = config.capacity
        self.crystallization_threshold = config.crystallization_threshold
        self.decay_rate = config.decay_rate

        # Key projection (content → retrieval key)
        self.key_projection = nn.Linear(config.d_model, self.d_key)

        # Value projection (for retrieval output)
        self.value_projection = nn.Linear(config.d_model, config.d_model)

        # Query projection (current state → query)
        self.query_projection = nn.Linear(config.d_model, self.d_key)

        # Crystallization gate (learned)
        self.crystallization_gate = CrystallizationGate(config.d_model)

        # Buffer (non-parametric, grows during processing)
        self.buffer = EpisodicBuffer(
            episodes=[],
            capacity=config.capacity,
            key_index=torch.zeros(0, self.d_key),
            temporal_index={},
            total_stored=0,
            total_evicted=0,
            total_consolidated=0
        )

    def process(
        self,
        experience: ExperientialState,
        current_step: int
    ) -> Tuple[Optional[Episode], List[Episode]]:
        """
        Main processing: possibly crystallize, retrieve relevant.

        Returns:
            crystallized: Episode if crystallized, None otherwise
            retrieved: List of retrieved relevant episodes
        """
        # 1. Should this experience become an episode?
        candidate = self._prepare_candidate(experience)
        crystallized = None

        if self._should_crystallize(candidate):
            crystallized = self._crystallize(candidate, current_step)
            self._store(crystallized)

        # 2. Retrieve relevant episodes for current context
        retrieved = self.retrieve(experience.content, top_k=5)

        # 3. Decay all memories slightly
        self._decay_all()

        return crystallized, retrieved

    def _prepare_candidate(
        self,
        experience: ExperientialState
    ) -> EpisodeCandidate:
        """Package experience as potential episode."""
        return EpisodeCandidate(
            content=experience.content,
            context=experience.predictions,
            timestamp=experience.timestamp,
            salience=experience.salience,
            affect=(experience.valence, experience.arousal)
        )

    def _should_crystallize(self, candidate: EpisodeCandidate) -> bool:
        """Decide if this moment should become a memory."""
        # Simple threshold
        if candidate.salience > self.crystallization_threshold:
            return True

        # Could also use learned gate
        # return self.crystallization_gate(candidate) > 0.5

        return False

    def _crystallize(
        self,
        candidate: EpisodeCandidate,
        current_step: int
    ) -> Episode:
        """Convert candidate to full episode."""
        episode_id = self.buffer.total_stored

        # Compute retrieval key
        key = self.key_projection(candidate.content)

        return Episode(
            id=episode_id,
            timestamp=candidate.timestamp,
            content=candidate.content.detach().clone(),
            context=candidate.context.detach().clone(),
            key=key.detach().clone(),
            salience=float(candidate.salience),
            valence=float(candidate.affect[0]),
            arousal=float(candidate.affect[1]),
            creation_time=current_step,
            last_access_time=current_step,
            access_count=0,
            strength=1.0,
            source_tokens=None,
            source_document=None
        )

    def _store(self, episode: Episode):
        """Add episode to buffer, evicting if necessary."""
        if len(self.buffer.episodes) >= self.capacity:
            self._evict_weakest()

        self.buffer.episodes.append(episode)
        self.buffer.temporal_index[episode.timestamp] = episode.id
        self.buffer.total_stored += 1

        # Update key index
        self._rebuild_key_index()

    def retrieve(
        self,
        query: Tensor,
        top_k: int = 5,
        current_step: Optional[int] = None
    ) -> List[Episode]:
        """
        Retrieve relevant episodes for a query.

        Args:
            query: [batch, d_model] or [d_model] - current state
            top_k: number of episodes to retrieve
            current_step: for updating access times

        Returns:
            List of most relevant episodes
        """
        if len(self.buffer.episodes) == 0:
            return []

        # Project query
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query_key = self.query_projection(query)  # [batch, d_key]

        # Compute similarities
        similarities = torch.mm(
            F.normalize(query_key, dim=-1),
            F.normalize(self.buffer.key_index, dim=-1).t()
        )  # [batch, n_episodes]

        # Get top-k
        top_k = min(top_k, len(self.buffer.episodes))
        _, top_indices = similarities[0].topk(top_k)

        # Retrieve and update access
        retrieved = []
        for idx in top_indices.tolist():
            episode = self.buffer.episodes[idx]
            episode.access_count += 1
            episode.strength = min(1.0, episode.strength + 0.1)  # strengthen on access
            if current_step is not None:
                episode.last_access_time = current_step
            retrieved.append(episode)

        return retrieved

    def retrieve_soft(
        self,
        query: Tensor,
        temperature: float = 0.1
    ) -> Tuple[Tensor, Tensor]:
        """
        Soft retrieval for differentiable training.

        Returns:
            values: [batch, d_model] - weighted sum of episode values
            weights: [batch, n_episodes] - attention weights
        """
        if len(self.buffer.episodes) == 0:
            return torch.zeros_like(query), torch.zeros(query.size(0), 0)

        # Project query
        query_key = self.query_projection(query)  # [batch, d_key]

        # Compute attention weights
        similarities = torch.mm(
            query_key,
            self.buffer.key_index.t()
        ) / temperature  # [batch, n_episodes]

        weights = F.softmax(similarities, dim=-1)

        # Compute weighted values
        values = torch.stack([
            self.value_projection(ep.content)
            for ep in self.buffer.episodes
        ])  # [n_episodes, d_model]

        retrieved_values = torch.mm(weights, values)  # [batch, d_model]

        return retrieved_values, weights

    def _decay_all(self):
        """Apply decay to all memories."""
        for episode in self.buffer.episodes:
            episode.strength *= (1 - self.decay_rate)

    def _evict_weakest(self):
        """Remove the weakest memory to make room."""
        if len(self.buffer.episodes) == 0:
            return

        # Find weakest (lowest strength × salience)
        scores = [
            ep.strength * ep.salience
            for ep in self.buffer.episodes
        ]
        weakest_idx = min(range(len(scores)), key=lambda i: scores[i])

        # Remove
        evicted = self.buffer.episodes.pop(weakest_idx)
        self.buffer.total_evicted += 1

        # Clean up indices
        if evicted.timestamp in self.buffer.temporal_index:
            del self.buffer.temporal_index[evicted.timestamp]
        self._rebuild_key_index()

    def _rebuild_key_index(self):
        """Rebuild the key index after changes."""
        if len(self.buffer.episodes) == 0:
            self.buffer.key_index = torch.zeros(0, self.d_key)
        else:
            self.buffer.key_index = torch.stack([
                ep.key for ep in self.buffer.episodes
            ])

    def get_consolidation_candidates(
        self,
        min_cluster_size: int = 3
    ) -> List[List[Episode]]:
        """
        Find clusters of similar episodes for consolidation.

        Episodes that are frequently accessed and similar
        are candidates for becoming semantic knowledge.
        """
        if len(self.buffer.episodes) < min_cluster_size:
            return []

        # Cluster by key similarity
        keys = self.buffer.key_index
        similarities = torch.mm(
            F.normalize(keys, dim=-1),
            F.normalize(keys, dim=-1).t()
        )

        # Simple clustering: group highly similar episodes
        clusters = []
        used = set()

        for i in range(len(self.buffer.episodes)):
            if i in used:
                continue

            cluster = [i]
            used.add(i)

            for j in range(i + 1, len(self.buffer.episodes)):
                if j in used:
                    continue
                if similarities[i, j] > 0.8:  # high similarity threshold
                    cluster.append(j)
                    used.add(j)

            if len(cluster) >= min_cluster_size:
                clusters.append([
                    self.buffer.episodes[idx] for idx in cluster
                ])

        return clusters
```

### 3.2 Crystallization Gate

```python
class CrystallizationGate(nn.Module):
    """
    Learned gate for crystallization decisions.

    Beyond simple salience threshold, learns what's worth remembering.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + 3, d_model // 2),  # +3 for salience, valence, arousal
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, candidate: EpisodeCandidate) -> Tensor:
        """Predict crystallization probability."""
        features = torch.cat([
            candidate.content,
            torch.tensor([
                candidate.salience,
                candidate.affect[0],
                candidate.affect[1]
            ], device=candidate.content.device)
        ], dim=-1)

        return self.net(features)
```

---

## 4. Training

### 4.1 Retrieval Loss

Train keys to be retrievable by future queries:

```python
def retrieval_contrastive_loss(
    query: Tensor,           # from future context
    positive_key: Tensor,    # key of episode that should be retrieved
    negative_keys: Tensor,   # keys of episodes that shouldn't
    temperature: float = 0.1
) -> Tensor:
    """
    Train keys so relevant episodes are retrieved.

    Positive: episode from earlier in same document that's relevant
    Negatives: episodes from other documents or irrelevant positions
    """
    query_norm = F.normalize(query, dim=-1)
    pos_norm = F.normalize(positive_key, dim=-1)
    neg_norm = F.normalize(negative_keys, dim=-1)

    pos_sim = (query_norm * pos_norm).sum(dim=-1, keepdim=True)
    neg_sim = torch.mm(query_norm.unsqueeze(0), neg_norm.t()).squeeze(0)

    logits = torch.cat([pos_sim, neg_sim], dim=-1) / temperature
    labels = torch.zeros(1, dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, labels)
```

### 4.2 Crystallization Loss

Train the crystallization gate via credit assignment:

```python
def crystallization_credit_loss(
    crystallized_episodes: List[Episode],
    was_retrieved: List[bool],
    helped_prediction: List[float]  # how much each retrieval reduced loss
) -> Tensor:
    """
    Train crystallization gate: store things that will be useful later.

    If an episode was retrieved and helped, crystallizing it was good.
    If an episode was never retrieved, crystallizing it was wasteful.
    """
    rewards = []
    for ep, retrieved, helped in zip(
        crystallized_episodes, was_retrieved, helped_prediction
    ):
        if retrieved and helped > 0:
            reward = helped  # positive: good to store
        elif not retrieved:
            reward = -0.1  # small negative: wasted storage
        else:
            reward = 0.0  # retrieved but didn't help
        rewards.append(reward)

    # REINFORCE-style gradient
    # (In practice, would need crystallization probabilities saved)
    return -torch.tensor(rewards).mean()
```

### 4.3 Resume-After-Interruption Loss

Force the model to use episodic memory:

```python
def resume_loss(
    model,
    document_chunks: List[Tensor],
    interrupt_at: int,
    resume_from: int
) -> Tensor:
    """
    Read chunks 1..interrupt_at, building memory.
    Discard hidden states.
    Resume from chunk resume_from using only memory.
    Measure prediction quality.
    """
    # Phase 1: Read and build memory
    experience = None
    for chunk in document_chunks[:interrupt_at]:
        logits, experience = model(chunk, prev_experience=experience)
        # Memory is updated internally

    # Discard transformer state (keep only memory)
    memory_state = model.episodic.buffer  # preserve
    model.reset_hidden_state()  # discard

    # Phase 2: Resume with only memory
    total_loss = 0
    experience = None  # reset experiential state too

    for chunk in document_chunks[resume_from:]:
        logits, experience = model(
            chunk,
            prev_experience=experience,
            memory_only=True  # can only use episodic retrieval
        )
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            chunk[:, 1:].contiguous().view(-1)
        )
        total_loss += loss

    return total_loss
```

---

## 5. Memory Dynamics

### 5.1 Decay Curve

```python
def memory_strength_over_time(
    initial_strength: float,
    decay_rate: float,
    time_since_creation: int,
    access_boosts: List[int]  # times when accessed
) -> float:
    """
    Memory strength decays but is boosted by retrieval.

    Follows power-law forgetting with rehearsal benefits.
    """
    strength = initial_strength
    current_time = 0

    for access_time in sorted(access_boosts):
        # Decay until access
        time_gap = access_time - current_time
        strength *= (1 - decay_rate) ** time_gap

        # Boost from access
        strength = min(1.0, strength + 0.2)
        current_time = access_time

    # Final decay to present
    final_gap = time_since_creation - current_time
    strength *= (1 - decay_rate) ** final_gap

    return strength
```

### 5.2 Eviction Policy

```python
class EvictionPolicy:
    """Strategies for choosing what to forget."""

    @staticmethod
    def weakest_first(episodes: List[Episode]) -> int:
        """Evict lowest strength × salience."""
        scores = [ep.strength * ep.salience for ep in episodes]
        return min(range(len(scores)), key=lambda i: scores[i])

    @staticmethod
    def oldest_first(episodes: List[Episode]) -> int:
        """Evict oldest (FIFO)."""
        return 0

    @staticmethod
    def least_recently_used(episodes: List[Episode]) -> int:
        """Evict least recently accessed (LRU)."""
        return min(
            range(len(episodes)),
            key=lambda i: episodes[i].last_access_time
        )

    @staticmethod
    def consolidate_then_evict(
        episodes: List[Episode],
        semantic_stream: 'SemanticStream'
    ) -> int:
        """Try to consolidate before evicting."""
        # Find weakest
        weakest = EvictionPolicy.weakest_first(episodes)

        # Check if it can be consolidated
        similar = [
            i for i, ep in enumerate(episodes)
            if i != weakest and
            F.cosine_similarity(
                episodes[weakest].key.unsqueeze(0),
                ep.key.unsqueeze(0)
            ) > 0.7
        ]

        if len(similar) >= 2:
            # Consolidate cluster into semantic memory
            cluster = [episodes[weakest]] + [episodes[i] for i in similar]
            semantic_stream.consolidate(cluster)

        return weakest
```

---

## 6. Interface with Other Streams

### 6.1 ← Experiential Stream (Input)

```python
def receive_from_experiential(
    episodic: EpisodicStream,
    experience: ExperientialState,
    step: int
) -> Optional[Episode]:
    """
    Receive potential episode from experiential stream.

    Decide whether to crystallize based on salience.
    """
    if experience.salience > episodic.crystallization_threshold:
        return episodic.crystallize_and_store(experience, step)
    return None
```

### 6.2 → Experiential Stream (Output)

```python
def provide_to_experiential(
    episodic: EpisodicStream,
    current_context: Tensor
) -> Tensor:
    """
    Provide retrieved memories to color current experience.
    """
    retrieved_values, weights = episodic.retrieve_soft(current_context)
    return retrieved_values  # blended memory content
```

### 6.3 → Semantic Stream (Consolidation)

```python
def consolidate_to_semantic(
    episodic: EpisodicStream,
    semantic: 'SemanticStream'
) -> List['Concept']:
    """
    Extract patterns from episodes into semantic knowledge.

    Called periodically (e.g., during "sleep" phase).
    """
    clusters = episodic.get_consolidation_candidates()
    new_concepts = []

    for cluster in clusters:
        concept = semantic.consolidate(cluster)
        if concept is not None:
            new_concepts.append(concept)

            # Optionally weaken source episodes
            for episode in cluster:
                episode.strength *= 0.5

    return new_concepts
```

---

## 7. Open Questions

1. **Key vs. Content**: Should retrieval keys be learned separately from content, or derived from it?

2. **Batch handling**: How to handle variable memory sizes across batch items?

3. **Gradient flow**: Should gradients flow through retrieved memories? When?

4. **Capacity**: Fixed size? Growing? Hierarchical?

5. **Consolidation trigger**: When to consolidate? Periodically? On eviction? On similarity threshold?

6. **Source preservation**: Should we store original tokens, or just embeddings?

---

## 8. Implementation Checklist

- [ ] Define EpisodicConfig (using constructor args instead)
- [x] Implement Episode dataclass (`experiential.py`)
- [x] Implement EpisodicBuffer → EpisodicMemory class
- [x] Implement EpisodicStream.crystallize → should_crystallize()
- [x] Implement EpisodicStream.store → store()
- [x] Implement EpisodicStream.retrieve (hard) → retrieve(), retrieve_by_time(), retrieve_by_salience()
- [ ] Implement EpisodicStream.retrieve_soft (differentiable)
- [ ] Implement decay mechanism (decay_rate defined, not yet applied)
- [x] Implement eviction policies (priority-based: salience × recency × retrieval_count)
- [ ] Add retrieval contrastive loss
- [ ] Add resume-after-interruption training
- [ ] Implement consolidation interface
- [x] Test memory persistence across chunks

### Current Implementation (`experiential.py`)

The `EpisodicMemory` class provides:
- Episode dataclass with timestamp, content, context, salience, affect, retrieval_count
- `store()` - store episodes with automatic capacity management
- `retrieve()` - similarity-based retrieval with top-k
- `retrieve_by_time()` - get most recent episodes
- `retrieve_by_salience()` - get most salient episodes
- `should_crystallize()` - threshold-based crystallization decision
- Priority-based eviction: `salience × recency × (1 + retrieval_count)`

---

*Related documents*:
- `memory_streams_architecture.md` — overall design
- `stream_experiential.md` — experiential stream (upstream)
- `stream_semantic.md` — semantic stream (consolidation target)
