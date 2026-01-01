# Semantic Stream

What I know — facts, concepts, patterns.

**Role**: The semantic stream holds abstracted knowledge distilled from episodes. It's timeless, general, and connected — a web of concepts rather than a sequence of events.

---

## 1. Core Concept

Semantic memory is **what you know** without remembering **when you learned it**.

- "Paris is the capital of France" — semantic
- "I learned that in 3rd grade" — episodic

The semantic stream:
- **Abstracts** from specific episodes to general patterns
- **Connects** related concepts into a knowledge graph
- **Persists** indefinitely (unlike decaying episodes)
- **Informs** predictions and understanding

```
Episodes ──consolidate──► Concepts ──connect──► Knowledge Graph
                              │
                              ▼
                      inform processing
                              │
                              ▼
                   shape predictions & understanding
```

---

## 2. Formal Definition

### 2.1 Concept Structure

```python
@dataclass
class Concept:
    """A piece of abstracted knowledge."""

    # Identity
    id: int                         # unique identifier
    name: Optional[str]             # human-readable label (if available)

    # Content
    embedding: Tensor               # [d_model] - distributed representation
    prototype: Optional[Tensor]     # [d_model] - central tendency (if clustered)

    # Provenance
    source_episodes: List[int]      # which episodes it was distilled from
    consolidation_time: int         # when created

    # Confidence
    confidence: float               # [0, 1] - how certain
    evidence_count: int             # how many supporting episodes

    # Relations
    connections: Dict[int, ConceptRelation]  # related concepts

    # Hierarchy
    parent: Optional[int]           # more general concept
    children: List[int]             # more specific concepts
    abstraction_level: int          # 0 = concrete, higher = more abstract


@dataclass
class ConceptRelation:
    """A relationship between two concepts."""
    target_id: int
    relation_type: str              # "is-a", "has-a", "causes", "related-to", etc.
    strength: float                 # [0, 1]
    evidence: List[int]             # episode IDs supporting this relation
```

### 2.2 Knowledge Graph Structure

```python
@dataclass
class KnowledgeGraph:
    """The semantic memory structure."""

    concepts: Dict[int, Concept]           # id → concept

    # Indices for fast retrieval
    embedding_index: Tensor                # [n_concepts, d_model]
    name_index: Dict[str, int]             # name → id

    # Graph structure
    adjacency: Dict[int, List[int]]        # id → connected ids

    # Statistics
    total_concepts: int
    total_relations: int
```

---

## 3. Architecture

### 3.1 Main Module

```python
class SemanticStream(nn.Module):
    """
    Semantic memory: abstracted, connected knowledge.

    Operations:
    - Consolidate: episodes → concept
    - Connect: link related concepts
    - Query: retrieve relevant knowledge
    - Generalize: abstract to higher levels
    """

    def __init__(self, config: SemanticConfig):
        super().__init__()
        self.d_model = config.d_model
        self.min_evidence = config.min_evidence  # min episodes for concept

        # Pattern extraction
        self.pattern_extractor = PatternExtractor(config.d_model)

        # Relation prediction
        self.relation_predictor = RelationPredictor(config.d_model)

        # Query interface
        self.query_projection = nn.Linear(config.d_model, config.d_model)

        # Knowledge graph (grows during processing)
        self.knowledge = KnowledgeGraph(
            concepts={},
            embedding_index=torch.zeros(0, config.d_model),
            name_index={},
            adjacency={},
            total_concepts=0,
            total_relations=0
        )

    def consolidate(
        self,
        episodes: List[Episode],
        require_min_evidence: bool = True
    ) -> Optional[Concept]:
        """
        Distill episodes into a concept.

        Example:
            Episodes: "John at cafe", "John orders coffee", "John tips well"
            Concept: "John is a regular cafe customer"

        The specific memories may fade, but the knowledge remains.
        """
        if require_min_evidence and len(episodes) < self.min_evidence:
            return None

        # Extract common pattern
        episode_embeddings = torch.stack([ep.content for ep in episodes])
        pattern = self.pattern_extractor(episode_embeddings)

        # Create concept
        concept_id = self.knowledge.total_concepts
        concept = Concept(
            id=concept_id,
            name=None,  # could be generated or extracted
            embedding=pattern.detach().clone(),
            prototype=episode_embeddings.mean(dim=0).detach().clone(),
            source_episodes=[ep.id for ep in episodes],
            consolidation_time=self._current_step(),
            confidence=min(1.0, len(episodes) / 10.0),
            evidence_count=len(episodes),
            connections={},
            parent=None,
            children=[],
            abstraction_level=0
        )

        # Add to knowledge graph
        self._add_concept(concept)

        # Find and create relations to existing concepts
        self._discover_relations(concept)

        return concept

    def query(
        self,
        query: Tensor,
        top_k: int = 5
    ) -> List[Concept]:
        """Retrieve relevant concepts for current context."""
        if self.knowledge.total_concepts == 0:
            return []

        query_emb = self.query_projection(query)

        # Compute similarities
        similarities = F.cosine_similarity(
            query_emb.unsqueeze(0),
            self.knowledge.embedding_index,
            dim=-1
        )

        top_k = min(top_k, self.knowledge.total_concepts)
        _, top_indices = similarities.topk(top_k)

        return [
            self.knowledge.concepts[idx.item()]
            for idx in top_indices
        ]

    def query_soft(
        self,
        query: Tensor,
        temperature: float = 0.1
    ) -> Tuple[Tensor, Tensor]:
        """
        Soft retrieval for differentiable training.

        Returns:
            knowledge: [batch, d_model] - weighted concept embeddings
            weights: [batch, n_concepts] - attention weights
        """
        if self.knowledge.total_concepts == 0:
            return torch.zeros_like(query), torch.zeros(query.size(0), 0)

        query_emb = self.query_projection(query)

        # Attention over concepts
        scores = torch.mm(
            query_emb,
            self.knowledge.embedding_index.t()
        ) / temperature

        weights = F.softmax(scores, dim=-1)
        knowledge = torch.mm(weights, self.knowledge.embedding_index)

        return knowledge, weights

    def connect(
        self,
        concept_a: int,
        concept_b: int,
        relation_type: str,
        strength: float,
        evidence: List[int] = None
    ):
        """Create or strengthen a relation between concepts."""
        if concept_a not in self.knowledge.concepts:
            return
        if concept_b not in self.knowledge.concepts:
            return

        relation = ConceptRelation(
            target_id=concept_b,
            relation_type=relation_type,
            strength=strength,
            evidence=evidence or []
        )

        # Add to concept
        self.knowledge.concepts[concept_a].connections[concept_b] = relation

        # Update adjacency
        if concept_a not in self.knowledge.adjacency:
            self.knowledge.adjacency[concept_a] = []
        if concept_b not in self.knowledge.adjacency[concept_a]:
            self.knowledge.adjacency[concept_a].append(concept_b)

        self.knowledge.total_relations += 1

    def generalize(
        self,
        concepts: List[int],
        abstraction_name: Optional[str] = None
    ) -> Optional[Concept]:
        """
        Create a more abstract concept from specific ones.

        Example:
            Concepts: "coffee", "tea", "hot chocolate"
            Abstraction: "hot beverages"
        """
        if len(concepts) < 2:
            return None

        # Get concept embeddings
        embeddings = torch.stack([
            self.knowledge.concepts[cid].embedding
            for cid in concepts
        ])

        # Extract higher-level pattern
        abstract_pattern = self.pattern_extractor(embeddings)

        # Determine abstraction level
        max_level = max(
            self.knowledge.concepts[cid].abstraction_level
            for cid in concepts
        )

        # Create abstract concept
        abstract_id = self.knowledge.total_concepts
        abstract = Concept(
            id=abstract_id,
            name=abstraction_name,
            embedding=abstract_pattern.detach().clone(),
            prototype=embeddings.mean(dim=0).detach().clone(),
            source_episodes=[],  # derived from concepts, not episodes
            consolidation_time=self._current_step(),
            confidence=min(c.confidence for c in
                          [self.knowledge.concepts[cid] for cid in concepts]),
            evidence_count=sum(c.evidence_count for c in
                              [self.knowledge.concepts[cid] for cid in concepts]),
            connections={},
            parent=None,
            children=concepts,
            abstraction_level=max_level + 1
        )

        self._add_concept(abstract)

        # Update children to point to parent
        for cid in concepts:
            self.knowledge.concepts[cid].parent = abstract_id
            self.connect(cid, abstract_id, "is-a", 1.0)

        return abstract

    def traverse(
        self,
        start_concept: int,
        relation_types: Optional[List[str]] = None,
        max_hops: int = 2
    ) -> List[Concept]:
        """
        Traverse knowledge graph from a starting concept.

        Returns concepts reachable within max_hops.
        """
        visited = {start_concept}
        frontier = [start_concept]
        result = [self.knowledge.concepts[start_concept]]

        for hop in range(max_hops):
            next_frontier = []
            for cid in frontier:
                concept = self.knowledge.concepts[cid]
                for target_id, relation in concept.connections.items():
                    if target_id in visited:
                        continue
                    if relation_types and relation.relation_type not in relation_types:
                        continue
                    visited.add(target_id)
                    next_frontier.append(target_id)
                    result.append(self.knowledge.concepts[target_id])
            frontier = next_frontier

        return result

    def _add_concept(self, concept: Concept):
        """Add concept to knowledge graph."""
        self.knowledge.concepts[concept.id] = concept
        self.knowledge.total_concepts += 1

        if concept.name:
            self.knowledge.name_index[concept.name] = concept.id

        # Update embedding index
        self._rebuild_embedding_index()

    def _discover_relations(self, new_concept: Concept):
        """Find relations between new concept and existing ones."""
        if self.knowledge.total_concepts <= 1:
            return

        # Compare to all existing concepts
        for cid, existing in self.knowledge.concepts.items():
            if cid == new_concept.id:
                continue

            # Predict relation
            relation = self.relation_predictor(
                new_concept.embedding,
                existing.embedding
            )

            if relation is not None and relation.strength > 0.5:
                self.connect(
                    new_concept.id,
                    cid,
                    relation.relation_type,
                    relation.strength
                )

    def _rebuild_embedding_index(self):
        """Rebuild embedding index for fast retrieval."""
        if self.knowledge.total_concepts == 0:
            self.knowledge.embedding_index = torch.zeros(0, self.d_model)
        else:
            self.knowledge.embedding_index = torch.stack([
                self.knowledge.concepts[i].embedding
                for i in range(self.knowledge.total_concepts)
            ])

    def _current_step(self) -> int:
        """Get current training step (placeholder)."""
        return 0
```

### 3.2 Pattern Extractor

```python
class PatternExtractor(nn.Module):
    """
    Extract common pattern from a set of embeddings.

    Goes beyond simple averaging to find latent structure.
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 2

        # Attention pooling
        self.attention = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Pattern refinement
        self.refiner = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Extract pattern from embeddings.

        Args:
            embeddings: [n_items, d_model] - items to find pattern in

        Returns:
            pattern: [d_model] - extracted common pattern
        """
        # Add batch dimension
        embeddings = embeddings.unsqueeze(0)  # [1, n_items, d_model]

        # Attention pooling
        pooled, _ = self.attention(self.query, embeddings, embeddings)
        pooled = pooled.squeeze(0).squeeze(0)  # [d_model]

        # Refine
        pattern = self.refiner(pooled)

        return pattern
```

### 3.3 Relation Predictor

```python
class RelationPredictor(nn.Module):
    """
    Predict relation between two concepts.
    """

    RELATION_TYPES = [
        "is-a",         # hyponymy
        "has-a",        # meronymy
        "causes",       # causation
        "related-to",   # general association
        "opposite-of",  # antonymy
        "none"          # no significant relation
    ]

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Relation classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # concat + diff
            nn.GELU(),
            nn.Linear(d_model, len(self.RELATION_TYPES))
        )

        # Strength predictor
        self.strength_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        concept_a: Tensor,
        concept_b: Tensor
    ) -> Optional[ConceptRelation]:
        """
        Predict relation from a to b.

        Args:
            concept_a: [d_model] - source concept
            concept_b: [d_model] - target concept

        Returns:
            ConceptRelation or None if no significant relation
        """
        # Concatenate, add element-wise difference for asymmetric relations
        combined = torch.cat([
            concept_a,
            concept_b,
            concept_a - concept_b  # captures directionality
        ], dim=-1)

        # Predict relation type
        type_logits = self.classifier(combined)
        type_probs = F.softmax(type_logits, dim=-1)
        type_idx = type_probs.argmax().item()
        relation_type = self.RELATION_TYPES[type_idx]

        if relation_type == "none":
            return None

        # Predict strength
        strength = self.strength_head(combined).item()

        if strength < 0.3:  # too weak
            return None

        return ConceptRelation(
            target_id=-1,  # to be filled by caller
            relation_type=relation_type,
            strength=strength,
            evidence=[]
        )
```

---

## 4. Training

### 4.1 Consolidation Quality Loss

Train the pattern extractor to find meaningful patterns:

```python
def consolidation_quality_loss(
    extracted_pattern: Tensor,
    source_episodes: List[Episode],
    held_out_episode: Episode
) -> Tensor:
    """
    The pattern should generalize to held-out examples.

    Train by: extract pattern from N-1 episodes,
    measure how well it matches the Nth.
    """
    # Pattern should be similar to held-out
    similarity = F.cosine_similarity(
        extracted_pattern.unsqueeze(0),
        held_out_episode.content.unsqueeze(0)
    )

    return 1 - similarity  # minimize distance
```

### 4.2 Relation Prediction Loss

Train the relation predictor with supervision:

```python
def relation_loss(
    predicted_type: Tensor,      # [n_types] logits
    predicted_strength: Tensor,  # [1] strength
    target_type: int,            # ground truth type index
    target_strength: float       # ground truth strength
) -> Tensor:
    """
    Supervised loss for relation prediction.

    Requires labeled concept pairs (from WordNet, ConceptNet, or LLM extraction).
    """
    type_loss = F.cross_entropy(
        predicted_type.unsqueeze(0),
        torch.tensor([target_type])
    )

    strength_loss = F.mse_loss(
        predicted_strength,
        torch.tensor([target_strength])
    )

    return type_loss + strength_loss
```

### 4.3 Knowledge Retrieval Loss

Train knowledge to be retrievable when relevant:

```python
def knowledge_retrieval_loss(
    query: Tensor,              # current context
    relevant_concept: Concept,  # should be retrieved
    irrelevant_concepts: List[Concept],  # shouldn't be retrieved
    temperature: float = 0.1
) -> Tensor:
    """
    Contrastive loss for knowledge retrieval.
    """
    pos_sim = F.cosine_similarity(
        query.unsqueeze(0),
        relevant_concept.embedding.unsqueeze(0)
    ) / temperature

    neg_sims = torch.stack([
        F.cosine_similarity(
            query.unsqueeze(0),
            c.embedding.unsqueeze(0)
        )
        for c in irrelevant_concepts
    ]) / temperature

    logits = torch.cat([pos_sim, neg_sims])
    labels = torch.zeros(1, dtype=torch.long)

    return F.cross_entropy(logits.unsqueeze(0), labels)
```

---

## 5. Consolidation Dynamics

### 5.1 When to Consolidate

```python
class ConsolidationTrigger:
    """Strategies for when to consolidate episodes into concepts."""

    @staticmethod
    def periodic(step: int, period: int = 1000) -> bool:
        """Consolidate every N steps."""
        return step % period == 0

    @staticmethod
    def on_sleep(is_sleep_phase: bool) -> bool:
        """Consolidate during 'sleep' (offline processing)."""
        return is_sleep_phase

    @staticmethod
    def on_similarity_threshold(
        episodic: 'EpisodicStream',
        threshold: float = 0.8
    ) -> bool:
        """Consolidate when similar episodes accumulate."""
        clusters = episodic.get_consolidation_candidates()
        return len(clusters) > 0

    @staticmethod
    def on_capacity_pressure(
        episodic: 'EpisodicStream',
        threshold: float = 0.9
    ) -> bool:
        """Consolidate when episodic buffer is nearly full."""
        return len(episodic.buffer.episodes) > episodic.capacity * threshold
```

### 5.2 Abstraction Hierarchy

```python
def build_abstraction_hierarchy(
    semantic: SemanticStream,
    min_cluster_size: int = 3,
    max_levels: int = 3
) -> None:
    """
    Build multi-level abstraction hierarchy.

    Level 0: Concrete concepts (from episodes)
    Level 1: First-order abstractions
    Level 2: Higher-order abstractions
    ...
    """
    for level in range(max_levels):
        # Get concepts at current level
        concepts_at_level = [
            c for c in semantic.knowledge.concepts.values()
            if c.abstraction_level == level and c.parent is None
        ]

        if len(concepts_at_level) < min_cluster_size:
            break

        # Cluster similar concepts
        clusters = cluster_concepts(concepts_at_level, min_cluster_size)

        # Create abstractions
        for cluster in clusters:
            semantic.generalize([c.id for c in cluster])
```

---

## 6. Interface with Other Streams

### 6.1 ← Episodic Stream (Consolidation Source)

```python
def receive_from_episodic(
    semantic: SemanticStream,
    episode_clusters: List[List[Episode]]
) -> List[Concept]:
    """
    Receive episode clusters for consolidation.

    Called by episodic stream when patterns emerge.
    """
    new_concepts = []
    for cluster in episode_clusters:
        concept = semantic.consolidate(cluster)
        if concept is not None:
            new_concepts.append(concept)
    return new_concepts
```

### 6.2 → Experiential Stream (Knowledge Application)

```python
def provide_to_experiential(
    semantic: SemanticStream,
    current_context: Tensor
) -> Tensor:
    """
    Provide relevant knowledge to inform current experience.

    Knowledge shapes predictions and understanding.
    """
    knowledge, _ = semantic.query_soft(current_context)
    return knowledge
```

### 6.3 → Procedural Stream (Skill Grounding)

```python
def ground_skill(
    semantic: SemanticStream,
    skill: 'Skill'
) -> List[Concept]:
    """
    Find knowledge relevant to a skill.

    Skills can be grounded in semantic knowledge.
    """
    return semantic.query(skill.trigger_pattern, top_k=3)
```

---

## 7. Open Questions

1. **Naming concepts**: How to generate human-readable names for concepts? LLM extraction? Learn from labeled data?

2. **Relation types**: Fixed set or learned? How many?

3. **Hierarchy depth**: How many abstraction levels? When to stop?

4. **Forgetting**: Should concepts ever be forgotten? How?

5. **Conflict resolution**: What if contradictory concepts emerge?

6. **Grounding**: How to ground concepts in perception/action (if applicable)?

---

## 8. Implementation Status

### Implemented ✅

- [x] Implement Concept dataclass
- [x] Implement ConceptRelation dataclass
- [x] Implement PatternExtractor (attention-based pooling)
- [x] Implement RelationPredictor (6 relation types)
- [x] Implement SemanticStream.consolidate
- [x] Implement SemanticStream.consolidate_from_memory (k-means clustering)
- [x] Implement SemanticStream.query
- [x] Implement SemanticStream.query_soft (differentiable)
- [x] Implement SemanticStream.connect
- [x] Implement SemanticStream.generalize
- [x] Tests for all operations

### Remaining

- [ ] Define SemanticConfig (currently inline params)
- [ ] Add consolidation triggers
- [ ] Add retrieval losses
- [ ] Build abstraction hierarchy automatically
- [ ] Test knowledge accumulation on real data
- [ ] Integration with MemoryAugmentedGPT

---

*Related documents*:
- `memory_streams_architecture.md` — overall design
- `stream_episodic.md` — episodic stream (consolidation source)
- `stream_procedural.md` — procedural stream (skill grounding)
