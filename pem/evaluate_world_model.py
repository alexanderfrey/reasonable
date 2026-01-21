"""
World Model Evaluation Script - Sequential Context Test

Tests whether the world model helps by maintaining context across sequential chunks
within the same document.

KEY INSIGHT: The world model should help on LATER chunks of a document because
it has accumulated context from earlier chunks. If we test on random chunks from
different documents, the world model can't help (wrong context).

TEST DESIGN:
1. For each document, process N consecutive chunks sequentially
2. Reset oscillator state between documents
3. Measure: Does loss decrease on later chunks vs early chunks?
4. Compare: With world model vs without world model

GO/NO-GO CRITERIA:
==================
1. CONTEXT BENEFIT: Loss on chunks 5+ should be >10% lower than chunks 1-2 (with world model)
2. ABLATION: World model should show >5% more context benefit than baseline
3. FREQUENCY LEARNING: Frequencies should change >10% from init
4. UTILIZATION: >50% of oscillators should be active

Usage:
    python -m pem.evaluate_world_model --data_dir /path/to/books/ --num_docs 20
"""

import argparse
import math
import os
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Iterator
import json

import torch
import torch.nn as nn
import numpy as np

from .pem_loop_global import PEMLoopGlobal, PEMLoopGlobalConfig
from .janus_pro_feature_extractor import JanusProFeatureExtractor, JanusProConfig, LearningMode


@dataclass
class EvaluationResults:
    """Results from world model evaluation."""
    # Context benefit (key metric)
    early_chunk_loss_with_wm: float      # Loss on chunks 1-2 with world model
    late_chunk_loss_with_wm: float       # Loss on chunks 5+ with world model
    context_benefit_with_wm: float       # % improvement late vs early

    early_chunk_loss_without_wm: float   # Loss on chunks 1-2 without world model
    late_chunk_loss_without_wm: float    # Loss on chunks 5+ without world model
    context_benefit_without_wm: float    # % improvement late vs early

    # Ablation
    ablation_context_benefit: float      # Extra context benefit from world model

    # Frequency learning
    freq_change_pct: float

    # Utilization
    active_oscillator_frac: float

    # Per-chunk loss curves
    chunk_losses_with_wm: List[float]
    chunk_losses_without_wm: List[float]

    # Verdict
    go_decision: bool
    reasons: List[str]


def find_text_files(data_dir: str, num_files: int = 0, seed: int = 42) -> List[Path]:
    """Find text files from directory."""
    all_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.txt') and not f.startswith('._'):
                all_files.append(Path(root) / f)

    if num_files > 0 and len(all_files) > num_files:
        random.seed(seed)
        return random.sample(all_files, num_files)
    return all_files


def load_document(file_path: Path, min_length: int = 5000) -> Optional[str]:
    """Load a document, return None if too short."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
        if len(text) < min_length:
            return None
        return text
    except:
        return None


def chunk_document(text: str, chunk_size: int = 1024, num_chunks: int = 10) -> List[str]:
    """Split document into consecutive chunks."""
    chunks = []
    stride = chunk_size  # Non-overlapping for cleaner evaluation

    for i in range(0, len(text) - chunk_size, stride):
        if len(chunks) >= num_chunks:
            break
        chunks.append(text[i:i + chunk_size])

    return chunks


class WorldModelEvaluator:
    """Evaluates world model with proper sequential context testing."""

    def __init__(
        self,
        device: torch.device,
        d_model: int = 1536,
        pred_d_neurons: int = 64,
        surp_d_neurons: int = 32,
        sync_pairs: int = 64,
        num_oscillators: int = 32,
    ):
        self.device = device
        self.d_model = d_model
        self.pred_d_neurons = pred_d_neurons
        self.surp_d_neurons = surp_d_neurons
        self.sync_pairs = sync_pairs
        self.num_oscillators = num_oscillators

        # Feature extractor
        print("Loading feature extractor...")
        janus_config = JanusProConfig(
            model_name_or_path="deepseek-ai/Janus-Pro-1B",
            output_dim=d_model,
            learning_mode=LearningMode.FROZEN,
        )
        self.feature_extractor = JanusProFeatureExtractor(janus_config)
        self.feature_extractor.to(device)
        self.feature_extractor.eval()
        print("Feature extractor loaded.")

    def create_model(self, use_world_model: bool) -> PEMLoopGlobal:
        """Create PEM model with or without world model."""
        config = PEMLoopGlobalConfig(
            d_model=self.d_model,
            pred_d_neurons=self.pred_d_neurons,
            surp_d_neurons=self.surp_d_neurons,
            pred_T=4,
            surp_T=3,
            pred_synapse_hidden=512,
            pred_nlm_hidden=32,
            pred_d_sync_out=self.sync_pairs,
            pred_d_sync_internal=self.sync_pairs,
            surp_synapse_hidden=256,
            surp_nlm_hidden=16,
            surp_d_sync_out=self.sync_pairs // 2,
            surp_d_sync_internal=self.sync_pairs // 2,
            sync_pairs=self.sync_pairs,
            use_oscillatory_world=use_world_model,
            num_oscillators=self.num_oscillators if use_world_model else 0,
        )
        model = PEMLoopGlobal(config)
        model.to(self.device)
        return model

    def reset_world_model(self, model: PEMLoopGlobal):
        """Reset oscillator phases to start fresh for a new document."""
        if hasattr(model, 'global_sync') and model.global_sync.oscillatory_world is not None:
            model.global_sync.oscillatory_world.reset_phases(random=True)

    def get_frequencies(self, model: PEMLoopGlobal) -> Optional[torch.Tensor]:
        """Get oscillator frequencies."""
        if hasattr(model, 'global_sync') and model.global_sync.oscillatory_world is not None:
            return model.global_sync.oscillatory_world.frequencies.detach().clone()
        return None

    def extract_features(self, text: str, max_length: int = 256) -> Optional[torch.Tensor]:
        """Extract features from text."""
        with torch.no_grad():
            tokenized = self.feature_extractor.tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            features = self.feature_extractor(input_ids, attention_mask=attention_mask)
            features = features.float()

            if features.shape[1] < 16:
                return None
            return features

    def process_document_sequential(
        self,
        model: PEMLoopGlobal,
        chunks: List[str],
        optimizer: torch.optim.Optimizer,
    ) -> List[float]:
        """
        Process document chunks SEQUENTIALLY, letting world model accumulate context.

        Returns loss for each chunk.
        """
        chunk_losses = []

        # Reset world model for fresh start on this document
        self.reset_world_model(model)

        for chunk_idx, chunk in enumerate(chunks):
            features = self.extract_features(chunk)
            if features is None:
                continue

            optimizer.zero_grad()

            # Compute targets
            targets = model.target_computer.compute_targets_efficient(features)

            # Forward pass - world model state persists between chunks!
            outputs, final_state = model(features, targets, num_steps=2)

            # Compute loss
            loss, loss_dict = model.compute_loss(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            chunk_losses.append(loss.item())

        return chunk_losses

    def evaluate_sequential_context(
        self,
        model: PEMLoopGlobal,
        documents: List[List[str]],  # List of documents, each is list of chunks
        num_epochs: int = 3,
        learning_rate: float = 1e-4,
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Evaluate model on sequential document chunks.

        Returns:
            avg_chunk_losses: Average loss at each chunk position across all docs
            all_doc_losses: Loss curves for each document
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()

        # Track losses by chunk position
        max_chunks = max(len(doc) for doc in documents)
        chunk_position_losses = [[] for _ in range(max_chunks)]
        all_doc_losses = []

        for epoch in range(num_epochs):
            random.shuffle(documents)

            for doc_idx, chunks in enumerate(documents):
                chunk_losses = self.process_document_sequential(model, chunks, optimizer)

                # Record losses by position
                for pos, loss in enumerate(chunk_losses):
                    if pos < max_chunks:
                        chunk_position_losses[pos].append(loss)

                all_doc_losses.append(chunk_losses)

                if (doc_idx + 1) % 5 == 0:
                    recent_avg = np.mean([l for losses in all_doc_losses[-5:] for l in losses])
                    print(f"  Epoch {epoch+1}, Doc {doc_idx+1}/{len(documents)} | Avg Loss: {recent_avg:.4f}")

        # Compute average loss at each chunk position
        avg_chunk_losses = [
            np.mean(losses) if losses else 0.0
            for losses in chunk_position_losses
        ]

        return avg_chunk_losses, all_doc_losses

    def evaluate(
        self,
        data_dir: str,
        num_docs: int = 20,
        chunks_per_doc: int = 10,
        num_epochs: int = 3,
    ) -> EvaluationResults:
        """Run full evaluation."""
        print("\n" + "="*60)
        print("WORLD MODEL EVALUATION - Sequential Context Test")
        print("="*60)

        # Load documents
        print(f"\nLoading documents from {data_dir}...")
        file_paths = find_text_files(data_dir, num_files=num_docs * 2)  # Extra in case some fail

        documents = []
        for fp in file_paths:
            if len(documents) >= num_docs:
                break
            text = load_document(fp)
            if text is None:
                continue
            chunks = chunk_document(text, chunk_size=1024, num_chunks=chunks_per_doc)
            if len(chunks) >= chunks_per_doc // 2:  # At least half the chunks
                documents.append(chunks)

        print(f"Loaded {len(documents)} documents with {chunks_per_doc} chunks each")

        if len(documents) < 5:
            raise ValueError("Not enough valid documents found")

        # Get initial frequencies
        model_with = self.create_model(use_world_model=True)
        freq_init = self.get_frequencies(model_with)

        # Evaluate WITH world model
        print("\n" + "-"*60)
        print("[1/2] Evaluating WITH world model...")
        print("-"*60)

        chunk_losses_with, all_losses_with = self.evaluate_sequential_context(
            model_with, documents, num_epochs=num_epochs
        )

        freq_final = self.get_frequencies(model_with)

        # Get oscillator metrics
        if model_with.global_sync.oscillatory_world is not None:
            osc_metrics = model_with.global_sync.oscillatory_world.get_metrics()
            active_frac = osc_metrics.active_oscillator_frac
        else:
            active_frac = 0.0

        # Evaluate WITHOUT world model
        print("\n" + "-"*60)
        print("[2/2] Evaluating WITHOUT world model...")
        print("-"*60)

        model_without = self.create_model(use_world_model=False)
        chunk_losses_without, all_losses_without = self.evaluate_sequential_context(
            model_without, documents, num_epochs=num_epochs
        )

        # Compute metrics
        # Early chunks = positions 0-1, Late chunks = positions 4+
        early_positions = [0, 1]
        late_positions = list(range(4, len(chunk_losses_with)))

        early_loss_with = np.mean([chunk_losses_with[i] for i in early_positions if i < len(chunk_losses_with)])
        late_loss_with = np.mean([chunk_losses_with[i] for i in late_positions if i < len(chunk_losses_with)])
        context_benefit_with = (early_loss_with - late_loss_with) / early_loss_with * 100 if early_loss_with > 0 else 0

        early_loss_without = np.mean([chunk_losses_without[i] for i in early_positions if i < len(chunk_losses_without)])
        late_loss_without = np.mean([chunk_losses_without[i] for i in late_positions if i < len(chunk_losses_without)])
        context_benefit_without = (early_loss_without - late_loss_without) / early_loss_without * 100 if early_loss_without > 0 else 0

        ablation_context_benefit = context_benefit_with - context_benefit_without

        # Frequency change
        if freq_init is not None and freq_final is not None:
            freq_change = (freq_final - freq_init).abs() / (freq_init.abs() + 1e-8)
            freq_change_pct = freq_change.mean().item() * 100
        else:
            freq_change_pct = 0.0

        # GO/NO-GO decision
        reasons = []
        go_criteria = []

        # Criterion 1: Context benefit with world model >10%
        if context_benefit_with > 10:
            go_criteria.append(True)
            reasons.append(f"✓ CONTEXT BENEFIT: {context_benefit_with:.1f}% improvement late vs early (>10% required)")
        else:
            go_criteria.append(False)
            reasons.append(f"✗ CONTEXT BENEFIT: {context_benefit_with:.1f}% improvement late vs early (<10% required)")

        # Criterion 2: World model provides extra context benefit >5%
        if ablation_context_benefit > 5:
            go_criteria.append(True)
            reasons.append(f"✓ ABLATION: World model adds {ablation_context_benefit:.1f}% extra context benefit (>5% required)")
        else:
            go_criteria.append(False)
            reasons.append(f"✗ ABLATION: World model adds {ablation_context_benefit:.1f}% extra context benefit (<5% required)")

        # Criterion 3: Frequency learning
        if freq_change_pct > 10:
            go_criteria.append(True)
            reasons.append(f"✓ FREQ LEARNING: {freq_change_pct:.1f}% change (>10% required)")
        else:
            go_criteria.append(False)
            reasons.append(f"✗ FREQ LEARNING: {freq_change_pct:.1f}% change (<10% required)")

        # Criterion 4: Utilization
        if active_frac > 0.5:
            go_criteria.append(True)
            reasons.append(f"✓ UTILIZATION: {active_frac*100:.1f}% active (>50% required)")
        else:
            go_criteria.append(False)
            reasons.append(f"✗ UTILIZATION: {active_frac*100:.1f}% active (<50% required)")

        go_decision = sum(go_criteria) >= 3

        results = EvaluationResults(
            early_chunk_loss_with_wm=early_loss_with,
            late_chunk_loss_with_wm=late_loss_with,
            context_benefit_with_wm=context_benefit_with,
            early_chunk_loss_without_wm=early_loss_without,
            late_chunk_loss_without_wm=late_loss_without,
            context_benefit_without_wm=context_benefit_without,
            ablation_context_benefit=ablation_context_benefit,
            freq_change_pct=freq_change_pct,
            active_oscillator_frac=active_frac,
            chunk_losses_with_wm=chunk_losses_with,
            chunk_losses_without_wm=chunk_losses_without,
            go_decision=go_decision,
            reasons=reasons,
        )

        return results

    def print_results(self, results: EvaluationResults):
        """Print formatted results."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)

        print("\n[CONTEXT BENEFIT - WITH WORLD MODEL]")
        print(f"  Early chunks (0-1) loss: {results.early_chunk_loss_with_wm:.4f}")
        print(f"  Late chunks (4+) loss:   {results.late_chunk_loss_with_wm:.4f}")
        print(f"  Context benefit:         {results.context_benefit_with_wm:.1f}%")

        print("\n[CONTEXT BENEFIT - WITHOUT WORLD MODEL]")
        print(f"  Early chunks (0-1) loss: {results.early_chunk_loss_without_wm:.4f}")
        print(f"  Late chunks (4+) loss:   {results.late_chunk_loss_without_wm:.4f}")
        print(f"  Context benefit:         {results.context_benefit_without_wm:.1f}%")

        print("\n[ABLATION]")
        print(f"  Extra context benefit from world model: {results.ablation_context_benefit:.1f}%")

        print("\n[LOSS BY CHUNK POSITION]")
        print("  Position | With WM | Without WM | Δ")
        print("  " + "-"*45)
        for i in range(min(len(results.chunk_losses_with_wm), len(results.chunk_losses_without_wm))):
            with_loss = results.chunk_losses_with_wm[i]
            without_loss = results.chunk_losses_without_wm[i]
            delta = without_loss - with_loss
            marker = "←" if delta > 0.01 else ""
            print(f"  {i:8d} | {with_loss:.4f}  | {without_loss:.4f}     | {delta:+.4f} {marker}")

        print("\n[OTHER METRICS]")
        print(f"  Frequency change: {results.freq_change_pct:.1f}%")
        print(f"  Active oscillators: {results.active_oscillator_frac*100:.1f}%")

        print("\n" + "="*60)
        print("GO/NO-GO ASSESSMENT")
        print("="*60)
        for reason in results.reasons:
            print(f"  {reason}")

        print("\n" + "="*60)
        if results.go_decision:
            print("  >>> DECISION: GO - World model shows value <<<")
        else:
            print("  >>> DECISION: NO-GO - World model not helping <<<")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate World Model - Sequential Context Test')
    parser.add_argument('--data_dir', type=str,
                        default='/media/alexander/Tank1/text_files/text_files/3c/',
                        help='Directory with text files')
    parser.add_argument('--num_docs', type=int, default=20,
                        help='Number of documents to evaluate')
    parser.add_argument('--chunks_per_doc', type=int, default=10,
                        help='Chunks per document')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Training epochs')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (5 docs, 2 epochs)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON')

    args = parser.parse_args()

    if args.quick:
        args.num_docs = 5
        args.num_epochs = 2
        print("Quick mode: 5 docs, 2 epochs")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    evaluator = WorldModelEvaluator(device=device)

    results = evaluator.evaluate(
        data_dir=args.data_dir,
        num_docs=args.num_docs,
        chunks_per_doc=args.chunks_per_doc,
        num_epochs=args.num_epochs,
    )

    evaluator.print_results(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'context_benefit_with_wm': results.context_benefit_with_wm,
                'context_benefit_without_wm': results.context_benefit_without_wm,
                'ablation_context_benefit': results.ablation_context_benefit,
                'freq_change_pct': results.freq_change_pct,
                'active_frac': results.active_oscillator_frac,
                'chunk_losses_with_wm': results.chunk_losses_with_wm,
                'chunk_losses_without_wm': results.chunk_losses_without_wm,
                'go_decision': results.go_decision,
                'reasons': results.reasons,
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
