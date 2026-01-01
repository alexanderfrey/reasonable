"""
Run semantic consolidation on training data.

Processes training data through the memory-augmented model,
accumulates episodic memories, then consolidates into semantic concepts.

Usage:
    python run_semantic_consolidation.py --n_samples 500 --n_clusters 10
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

from model import GPT, GPTConfig
from experiential import (
    MemoryAugmentedGPT,
    EpisodicMemory,
    SemanticStream,
    Episode,
)
from pretrain import PretokenizedDataset


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load trained memory-augmented model."""
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Reconstruct config
    config_dict = ckpt.get('config', {})
    args = ckpt.get('args', {})

    config = GPTConfig(
        vocab_size=config_dict.get('vocab_size', 128256),
        d_model=config_dict.get('d_model', 1024),
        n_head=config_dict.get('n_head', 16),
        n_layer=config_dict.get('n_layer', 12),
        max_seq_len=config_dict.get('max_seq_len', 2048),
        n_kv_head=config_dict.get('n_kv_head'),
        d_ff=config_dict.get('d_ff'),
    )

    # Create model
    gpt = GPT(config)
    memory_gpt = MemoryAugmentedGPT(
        gpt,
        memory_capacity=args.get('memory_capacity', 1000),
        crystallization_threshold=args.get('crystallization_threshold', 0.2),
        memory_integration=args.get('integration', 'gated'),
    )

    # Load weights
    memory_gpt.load_state_dict(ckpt['memory_gpt_state_dict'], strict=False)
    memory_gpt = memory_gpt.to(device)
    memory_gpt.eval()

    return memory_gpt, config


def process_samples(
    memory_gpt: MemoryAugmentedGPT,
    dataset,
    tokenizer,
    device: torch.device,
    n_samples: int = 500,
    batch_size: int = 4
):
    """Process samples to build episodic memory."""
    print(f"\nProcessing {n_samples} samples to build episodic memory...")

    # Reset memory
    memory_gpt.reset_memory()

    crystallized = 0
    total_processed = 0
    sample_info = []  # Track what we crystallized

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for i in tqdm(range(0, len(indices), batch_size), desc="Building memory"):
            batch_indices = indices[i:i+batch_size]
            tokens_list = []

            for idx in batch_indices:
                sample = dataset[int(idx)]
                tokens_list.append(sample['input_ids'])

            if len(tokens_list) == 0:
                continue

            tokens = torch.stack(tokens_list).to(device)

            # Forward pass with crystallization
            logits, hidden, mem_out = memory_gpt(
                tokens,
                crystallize=True,
                use_memory=True
            )

            total_processed += len(tokens_list)

            # Track crystallization
            if mem_out.get('crystallized'):
                crystallized += 1
                # Decode a snippet for context
                for j, idx in enumerate(batch_indices):
                    if mem_out.get('salience') is not None:
                        salience = mem_out['salience'][j].item() if mem_out['salience'].dim() > 0 else mem_out['salience'].item()
                        valence = mem_out['valence'][j].item() if mem_out['valence'] is not None and mem_out['valence'].dim() > 0 else 0
                        arousal = mem_out['arousal'][j].item() if mem_out['arousal'] is not None and mem_out['arousal'].dim() > 0 else 0

                        if salience > memory_gpt.memory.crystallization_threshold:
                            text_snippet = tokenizer.decode(tokens[j][:50].tolist())
                            sample_info.append({
                                'idx': int(idx),
                                'salience': salience,
                                'valence': valence,
                                'arousal': arousal,
                                'text_start': text_snippet[:100]
                            })

    print(f"\nProcessed {total_processed} samples")
    print(f"Crystallized {memory_gpt.memory.size} episodes")
    print(f"Crystallization rate: {memory_gpt.memory.size / total_processed * 100:.1f}%")

    return sample_info


def analyze_memory(memory: EpisodicMemory):
    """Analyze the episodic memory before consolidation."""
    print("\n" + "=" * 60)
    print("EPISODIC MEMORY ANALYSIS")
    print("=" * 60)

    if memory.size == 0:
        print("Memory is empty!")
        return

    episodes = memory.episodes

    saliences = [ep.salience for ep in episodes]
    valences = [ep.valence for ep in episodes]
    arousals = [ep.arousal for ep in episodes]

    print(f"Total episodes: {len(episodes)}")
    print(f"\nSalience: min={min(saliences):.4f}, max={max(saliences):.4f}, mean={np.mean(saliences):.4f}")
    print(f"Valence:  min={min(valences):.4f}, max={max(valences):.4f}, mean={np.mean(valences):.4f}")
    print(f"Arousal:  min={min(arousals):.4f}, max={max(arousals):.4f}, mean={np.mean(arousals):.4f}")

    # Distribution of valence (negative vs positive)
    negative = sum(1 for v in valences if v < -0.1)
    neutral = sum(1 for v in valences if -0.1 <= v <= 0.1)
    positive = sum(1 for v in valences if v > 0.1)
    print(f"\nValence distribution: negative={negative}, neutral={neutral}, positive={positive}")


def run_consolidation(
    memory: EpisodicMemory,
    d_model: int,
    n_clusters: int = 10,
    min_cluster_size: int = 3
):
    """Run semantic consolidation on episodic memory."""
    print("\n" + "=" * 60)
    print("SEMANTIC CONSOLIDATION")
    print("=" * 60)

    # Create semantic stream
    semantic = SemanticStream(
        d_model=d_model,
        min_evidence=min_cluster_size,
        similarity_threshold=0.7,
        max_concepts=1000
    )

    # Move to same device as memory
    device = memory.episodes[0].content.device if memory.size > 0 else torch.device('cpu')
    semantic = semantic.to(device)

    # Run consolidation
    print(f"\nConsolidating {memory.size} episodes into ~{n_clusters} clusters...")
    start_time = time.time()

    concepts = semantic.consolidate_from_memory(
        memory,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size
    )

    elapsed = time.time() - start_time
    print(f"Consolidation took {elapsed:.2f}s")
    print(f"Created {len(concepts)} concepts")

    return semantic, concepts


def analyze_concepts(
    semantic: SemanticStream,
    concepts: list,
    memory: EpisodicMemory,
    tokenizer
):
    """Analyze the created concepts."""
    print("\n" + "=" * 60)
    print("CONCEPT ANALYSIS")
    print("=" * 60)

    if len(concepts) == 0:
        print("No concepts created!")
        return {}

    # Basic stats
    stats = semantic.get_stats()
    print(f"\nKnowledge graph stats:")
    print(f"  Total concepts: {stats['n_concepts']}")
    print(f"  Total relations: {stats['n_relations']}")
    print(f"  Avg evidence per concept: {stats['avg_evidence']:.1f}")
    print(f"  Avg confidence: {stats['avg_confidence']:.3f}")

    # Analyze each concept
    print("\n" + "-" * 60)
    print("CONCEPT DETAILS")
    print("-" * 60)

    concept_details = []

    for concept in concepts:
        print(f"\n[Concept {concept.id}]")
        print(f"  Evidence count: {concept.evidence_count}")
        print(f"  Confidence: {concept.confidence:.3f}")
        print(f"  Abstraction level: {concept.abstraction_level}")

        # Analyze source episodes
        if concept.source_episodes:
            # Get the actual episodes
            source_eps = [ep for ep in memory.episodes
                         if ep.timestamp in concept.source_episodes]

            if source_eps:
                avg_salience = np.mean([ep.salience for ep in source_eps])
                avg_valence = np.mean([ep.valence for ep in source_eps])
                avg_arousal = np.mean([ep.arousal for ep in source_eps])

                print(f"  Source episode stats:")
                print(f"    Avg salience: {avg_salience:.4f}")
                print(f"    Avg valence: {avg_valence:.4f}")
                print(f"    Avg arousal: {avg_arousal:.4f}")

                concept_details.append({
                    'id': concept.id,
                    'evidence_count': concept.evidence_count,
                    'confidence': concept.confidence,
                    'avg_salience': float(avg_salience),
                    'avg_valence': float(avg_valence),
                    'avg_arousal': float(avg_arousal),
                })

        # Show connections
        if concept.connections:
            print(f"  Connections: {len(concept.connections)}")
            for target_id, relation in list(concept.connections.items())[:3]:
                print(f"    → Concept {target_id}: {relation.relation_type} (strength={relation.strength:.3f})")

    return {
        'stats': stats,
        'concepts': concept_details
    }


def test_retrieval(
    semantic: SemanticStream,
    memory: EpisodicMemory,
    n_queries: int = 5
):
    """Test concept retrieval with random queries."""
    print("\n" + "=" * 60)
    print("RETRIEVAL TEST")
    print("=" * 60)

    if semantic.size == 0 or memory.size == 0:
        print("Nothing to retrieve!")
        return

    # Use random episodes as queries
    device = memory.episodes[0].content.device
    query_indices = np.random.choice(memory.size, min(n_queries, memory.size), replace=False)

    for idx in query_indices:
        query_ep = memory.episodes[idx]
        query = query_ep.content

        print(f"\nQuery (episode {query_ep.timestamp}, salience={query_ep.salience:.4f}):")

        # Hard retrieval
        results = semantic.query(query, top_k=3)
        print(f"  Top matches:")
        for concept, sim in results:
            print(f"    Concept {concept.id}: similarity={sim:.4f}, evidence={concept.evidence_count}")

        # Soft retrieval
        knowledge, weights = semantic.query_soft(query, temperature=0.1)
        top_weight_idx = weights.argmax().item()
        print(f"  Soft retrieval: top weight={weights.max():.4f} on concept {top_weight_idx}")


def test_generalization(semantic: SemanticStream):
    """Test concept generalization if we have enough concepts."""
    print("\n" + "=" * 60)
    print("GENERALIZATION TEST")
    print("=" * 60)

    if semantic.size < 3:
        print("Need at least 3 concepts for generalization test")
        return None

    # Try to generalize the first 3 concepts
    concept_ids = list(semantic.concepts.keys())[:3]

    print(f"Generalizing concepts {concept_ids}...")
    abstract = semantic.generalize(concept_ids, name="abstract_test")

    if abstract:
        print(f"Created abstract concept:")
        print(f"  ID: {abstract.id}")
        print(f"  Abstraction level: {abstract.abstraction_level}")
        print(f"  Children: {abstract.children}")
        print(f"  Evidence count: {abstract.evidence_count}")

        # Check parent-child relations
        for cid in concept_ids:
            child = semantic.concepts[cid]
            print(f"  Child {cid} parent: {child.parent}")

    return abstract


def main():
    parser = argparse.ArgumentParser(description="Run semantic consolidation")
    parser.add_argument("--checkpoint", default="memory_augmented_1024/memory_gpt_epoch_1.pt")
    parser.add_argument("--data", default="tiny_pretrain_output/training_meta-llama_Meta-Llama-3-8B_tokens.bin")
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--min_cluster_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="semantic_consolidation_results.json")

    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("SEMANTIC CONSOLIDATION EXPERIMENT")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {args.n_samples}")
    print(f"Target clusters: {args.n_clusters}")
    print(f"Min cluster size: {args.min_cluster_size}")
    print(f"Device: {device}")

    # Load model
    memory_gpt, config = load_checkpoint(args.checkpoint, device)
    d_model = config.d_model

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load dataset
    meta_path = args.data.replace('_tokens.bin', '_metadata.json')
    with open(meta_path) as f:
        meta = json.load(f)

    dataset = PretokenizedDataset(
        args.data,
        min(args.n_samples * 2, meta['num_examples']),
        max_seq_len=1024,
        stride=1024,
        data_type="Training"
    )

    # Process samples to build memory
    sample_info = process_samples(
        memory_gpt,
        dataset,
        tokenizer,
        device,
        n_samples=args.n_samples,
        batch_size=args.batch_size
    )

    # Analyze episodic memory
    analyze_memory(memory_gpt.memory)

    # Run semantic consolidation
    semantic, concepts = run_consolidation(
        memory_gpt.memory,
        d_model=d_model,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size
    )

    # Analyze concepts
    analysis = analyze_concepts(semantic, concepts, memory_gpt.memory, tokenizer)

    # Test retrieval
    test_retrieval(semantic, memory_gpt.memory)

    # Test generalization
    abstract = test_generalization(semantic)

    # Save results
    results = {
        'args': vars(args),
        'episodic_memory_size': memory_gpt.memory.size,
        'n_concepts_created': len(concepts),
        'analysis': analysis,
        'sample_crystallizations': sample_info[:20],  # First 20
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
