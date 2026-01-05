"""
Resume-After-Interruption Training for Memory-Augmented GPT

This training paradigm forces the model to actually USE memory by:
1. Processing document chunks sequentially, building episodic memory
2. At a random point, "interrupting" - discarding hidden states but keeping memory
3. Resuming from the interrupted point - model must use memory to understand context
4. Computing loss on resumed chunks - this is the key training signal

The insight: without this, the model can always rely on the context window and
may never learn to leverage memory effectively.

Usage:
    # Quick test
    python train_resume_interruption.py --max_steps 500

    # Full training
    python train_resume_interruption.py --n_epochs 1 --batch_size 4

    # Adjust interruption probability
    python train_resume_interruption.py --interrupt_prob 0.5 --min_chunks_before_interrupt 3
"""

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm

from experiential import (
    MemoryAugmentedGPT,
    memory_augmented_loss,
    experiential_loss,
    combined_experiential_loss,
    prediction_accuracy,
)
from model import GPT, GPTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# NarrativeChunkDataset - yields consecutive chunks from same document
# =============================================================================

@dataclass
class DocumentChunks:
    """A document split into consecutive chunks."""
    doc_id: int
    chunks: List[torch.Tensor]  # List of [chunk_size] tensors
    n_chunks: int


class NarrativeChunkDataset(IterableDataset):
    """
    Dataset that yields consecutive chunks from the same document.

    Key features:
    - Finds document boundaries using EOS tokens
    - Splits documents into fixed-size chunks
    - Yields DocumentChunks objects with all chunks from a document
    - Enables resume-after-interruption training

    Usage:
        dataset = NarrativeChunkDataset(
            token_file="training_tokens.npy",
            chunk_size=512,
            min_chunks_per_doc=4,
            eos_token_id=128000
        )

        for doc in dataset:
            # doc.chunks is a list of consecutive chunks
            for chunk in doc.chunks:
                model(chunk)
    """

    def __init__(
        self,
        token_file: str,
        chunk_size: int = 512,
        min_chunks_per_doc: int = 3,
        max_chunks_per_doc: int = 20,
        eos_token_id: int = 128000,
        shuffle_docs: bool = True,
        seed: int = 42
    ):
        """
        Args:
            token_file: Path to pretokenized .npy or .bin file
            chunk_size: Size of each chunk (in tokens)
            min_chunks_per_doc: Minimum chunks required to use a document
            max_chunks_per_doc: Maximum chunks to take from a document
            eos_token_id: Token ID used as document separator
            shuffle_docs: Whether to shuffle document order
            seed: Random seed for shuffling
        """
        self.token_file = token_file
        self.chunk_size = chunk_size
        self.min_chunks_per_doc = min_chunks_per_doc
        self.max_chunks_per_doc = max_chunks_per_doc
        self.eos_token_id = eos_token_id
        self.shuffle_docs = shuffle_docs
        self.seed = seed

        # Load tokens
        self.tokens = self._load_tokens()
        logger.info(f"Loaded {len(self.tokens):,} tokens from {token_file}")

        # Find document boundaries
        self.doc_boundaries = self._find_doc_boundaries()
        logger.info(f"Found {len(self.doc_boundaries) - 1} documents")

        # Filter to documents with enough chunks
        self.valid_docs = self._get_valid_docs()
        logger.info(f"Valid documents (>= {min_chunks_per_doc} chunks): {len(self.valid_docs)}")

    def _load_tokens(self) -> np.ndarray:
        """Load tokens from file."""
        if self.token_file.endswith(".npy"):
            return np.load(self.token_file, mmap_mode="r")
        else:
            # Try to load metadata for dtype
            meta_path = self.token_file.replace("_tokens.bin", "_metadata.json")
            dtype = "uint32"
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                    dtype = meta.get("dtype", dtype)
            return np.memmap(self.token_file, dtype=np.dtype(dtype), mode="r")

    def _find_doc_boundaries(self) -> List[int]:
        """Find positions of EOS tokens (document boundaries)."""
        # This can be slow for large files, so we sample or use vectorized ops
        tokens_array = np.array(self.tokens)  # May need to chunk for huge files

        # Find EOS positions
        eos_positions = np.where(tokens_array == self.eos_token_id)[0]

        # Add start and end
        boundaries = [0] + (eos_positions + 1).tolist()
        if boundaries[-1] < len(tokens_array):
            boundaries.append(len(tokens_array))

        return boundaries

    def _get_valid_docs(self) -> List[Tuple[int, int, int]]:
        """Get documents with enough tokens for min_chunks_per_doc chunks."""
        valid = []
        min_tokens = self.min_chunks_per_doc * self.chunk_size

        for i in range(len(self.doc_boundaries) - 1):
            start = self.doc_boundaries[i]
            end = self.doc_boundaries[i + 1]
            doc_len = end - start

            if doc_len >= min_tokens:
                n_possible_chunks = doc_len // self.chunk_size
                valid.append((i, start, end, min(n_possible_chunks, self.max_chunks_per_doc)))

        return valid

    def _get_doc_chunks(self, doc_info: Tuple[int, int, int, int]) -> DocumentChunks:
        """Extract chunks from a document."""
        doc_id, start, end, n_chunks = doc_info

        chunks = []
        for i in range(n_chunks):
            chunk_start = start + i * self.chunk_size
            chunk_end = chunk_start + self.chunk_size

            if chunk_end <= end:
                chunk_tokens = np.array(self.tokens[chunk_start:chunk_end])
                chunks.append(torch.tensor(chunk_tokens, dtype=torch.long))

        return DocumentChunks(
            doc_id=doc_id,
            chunks=chunks,
            n_chunks=len(chunks)
        )

    def __iter__(self) -> Iterator[DocumentChunks]:
        """Iterate over documents, yielding DocumentChunks."""
        # Get worker info for distributed loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Split docs across workers
            per_worker = len(self.valid_docs) // worker_info.num_workers
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = start_idx + per_worker if worker_id < worker_info.num_workers - 1 else len(self.valid_docs)
            docs = self.valid_docs[start_idx:end_idx]
        else:
            docs = self.valid_docs

        # Shuffle if requested
        if self.shuffle_docs:
            docs = list(docs)
            rng = random.Random(self.seed)
            rng.shuffle(docs)

        for doc_info in docs:
            yield self._get_doc_chunks(doc_info)

    def __len__(self) -> int:
        """Return number of valid documents."""
        return len(self.valid_docs)


# =============================================================================
# Resume-After-Interruption Training Functions
# =============================================================================

def resume_after_interruption_loss(
    memory_gpt: MemoryAugmentedGPT,
    doc_chunks: DocumentChunks,
    device: torch.device,
    interrupt_point: Optional[int] = None,
    min_chunks_before: int = 2,
    min_chunks_after: int = 1,
    lm_weight: float = 1.0,
    exp_weight: float = 0.1,
    resume_weight: float = 2.0,  # Weight boost for resumed chunks
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute loss with resume-after-interruption paradigm.

    Process:
    1. Process chunks 0..interrupt_point (build memory, compute loss)
    2. Reset hidden state (but keep memory!)
    3. Process chunks interrupt_point..end (must use memory, weighted loss)

    Args:
        memory_gpt: The memory-augmented model
        doc_chunks: DocumentChunks from NarrativeChunkDataset
        device: Device to run on
        interrupt_point: Where to interrupt (None = random)
        min_chunks_before: Minimum chunks before interruption
        min_chunks_after: Minimum chunks after interruption
        lm_weight: Weight for language modeling loss
        exp_weight: Weight for experiential prediction loss
        resume_weight: Extra weight for post-interruption loss

    Returns:
        total_loss: Combined loss tensor
        metrics: Dictionary of loss components and metrics
    """
    chunks = doc_chunks.chunks
    n_chunks = len(chunks)

    # Determine interruption point
    if interrupt_point is None:
        # Random point ensuring enough chunks before and after
        max_interrupt = n_chunks - min_chunks_after
        min_interrupt = min_chunks_before
        if max_interrupt <= min_interrupt:
            interrupt_point = n_chunks // 2
        else:
            interrupt_point = random.randint(min_interrupt, max_interrupt)

    metrics = {
        'n_chunks': n_chunks,
        'interrupt_point': interrupt_point,
        'pre_interrupt_loss': 0.0,
        'post_interrupt_loss': 0.0,
        'lm_loss': 0.0,
        'exp_loss': 0.0,
        'memory_size_at_interrupt': 0,
        'memory_retrievals': 0,
    }

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # Track memory query for causal retrieval between chunks
    prev_memory_query = None

    # Phase 1: Process chunks before interruption (build memory)
    for i in range(interrupt_point):
        chunk = chunks[i].unsqueeze(0).to(device)  # [1, chunk_size]
        targets = chunk[:, 1:].contiguous()
        inputs = chunk[:, :-1].contiguous()

        # Causal memory retrieval: use previous chunk's query
        logits, hidden, mem_out = memory_gpt(
            inputs,
            crystallize=True,  # Build memory
            use_memory=True,
            prev_memory_query=prev_memory_query  # Causal: query from previous chunk
        )

        # Save query for next chunk
        prev_memory_query = mem_out.get('next_memory_query')

        loss, loss_dict = memory_augmented_loss(
            logits, targets, mem_out,
            lm_weight=lm_weight,
            exp_weight=exp_weight
        )

        total_loss = total_loss + loss
        metrics['pre_interrupt_loss'] += loss_dict['total_loss']
        metrics['lm_loss'] += loss_dict['lm_loss']
        metrics['exp_loss'] += loss_dict.get('exp_loss', 0.0)

    metrics['memory_size_at_interrupt'] = memory_gpt.memory.size

    # Phase 2: INTERRUPT - reset hidden state but keep memory AND query
    # The query from before interruption provides context for what was being processed
    memory_gpt.reset_hidden_state()
    # Note: prev_memory_query is preserved - it represents "what I was thinking about"

    # Phase 3: Process chunks after interruption (must use memory)
    for i in range(interrupt_point, n_chunks):
        chunk = chunks[i].unsqueeze(0).to(device)
        targets = chunk[:, 1:].contiguous()
        inputs = chunk[:, :-1].contiguous()

        # Causal memory retrieval: use previous chunk's query
        # After interruption, this lets the model use memory from before the break
        logits, hidden, mem_out = memory_gpt(
            inputs,
            crystallize=True,  # Continue building memory
            use_memory=True,   # Critical: must use memory now
            prev_memory_query=prev_memory_query  # Causal: query carries across interrupt
        )

        # Save query for next chunk
        prev_memory_query = mem_out.get('next_memory_query')

        loss, loss_dict = memory_augmented_loss(
            logits, targets, mem_out,
            lm_weight=lm_weight,
            exp_weight=exp_weight
        )

        # Apply resume weight - this is where learning to use memory happens
        weighted_loss = loss * resume_weight
        total_loss = total_loss + weighted_loss

        metrics['post_interrupt_loss'] += loss_dict['total_loss']
        metrics['lm_loss'] += loss_dict['lm_loss']
        metrics['exp_loss'] += loss_dict.get('exp_loss', 0.0)

        if mem_out.get('retrieved_episodic') is not None:
            metrics['memory_retrievals'] += 1

    # Normalize metrics
    if interrupt_point > 0:
        metrics['pre_interrupt_loss'] /= interrupt_point
    if n_chunks - interrupt_point > 0:
        metrics['post_interrupt_loss'] /= (n_chunks - interrupt_point)
    metrics['lm_loss'] /= n_chunks
    metrics['exp_loss'] /= n_chunks
    metrics['total_loss'] = total_loss.item()

    return total_loss, metrics


def train_with_interruption(
    memory_gpt: MemoryAugmentedGPT,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int = 1,
    max_steps: Optional[int] = None,
    log_interval: int = 10,
    interrupt_prob: float = 0.7,
    min_chunks_before: int = 2,
    resume_weight: float = 2.0,
    accumulation_steps: int = 1,
    freeze_gpt: bool = True,
) -> Dict[str, List[float]]:
    """
    Train with resume-after-interruption paradigm.

    Args:
        memory_gpt: Memory-augmented model
        dataloader: DataLoader yielding DocumentChunks
        optimizer: Optimizer for memory components
        device: Device to train on
        n_epochs: Number of epochs
        max_steps: Maximum steps (overrides epochs if set)
        log_interval: Steps between logging
        interrupt_prob: Probability of interrupting each document
        min_chunks_before: Minimum chunks before interruption
        resume_weight: Extra weight for post-interruption loss
        accumulation_steps: Gradient accumulation steps
        freeze_gpt: Whether to freeze GPT backbone

    Returns:
        Training history dictionary
    """
    memory_gpt.train()

    # Optionally freeze GPT backbone (train only memory components)
    if freeze_gpt:
        for param in memory_gpt.gpt.parameters():
            param.requires_grad = False
        logger.info("GPT backbone frozen - training memory components only")

    history = {
        'total_loss': [],
        'pre_interrupt_loss': [],
        'post_interrupt_loss': [],
        'memory_size': [],
        'interrupt_points': [],
    }

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_docs = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")

        for doc_chunks in pbar:
            # Reset memory for each new document
            memory_gpt.reset_memory()

            # Decide whether to interrupt this document
            do_interrupt = random.random() < interrupt_prob

            if do_interrupt and doc_chunks.n_chunks >= min_chunks_before + 1:
                # Resume-after-interruption training
                loss, metrics = resume_after_interruption_loss(
                    memory_gpt,
                    doc_chunks,
                    device,
                    min_chunks_before=min_chunks_before,
                    resume_weight=resume_weight,
                )
            else:
                # Standard sequential processing (no interruption)
                loss, metrics = resume_after_interruption_loss(
                    memory_gpt,
                    doc_chunks,
                    device,
                    interrupt_point=doc_chunks.n_chunks,  # No interruption
                    resume_weight=1.0,
                )

            # Scale for accumulation
            loss = loss / accumulation_steps
            loss.backward()

            if (global_step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(memory_gpt.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Track metrics
            history['total_loss'].append(metrics['total_loss'])
            history['pre_interrupt_loss'].append(metrics['pre_interrupt_loss'])
            history['post_interrupt_loss'].append(metrics['post_interrupt_loss'])
            history['memory_size'].append(metrics['memory_size_at_interrupt'])
            history['interrupt_points'].append(metrics['interrupt_point'])

            epoch_loss += metrics['total_loss']
            n_docs += 1
            global_step += 1

            # Logging
            if global_step % log_interval == 0:
                avg_loss = sum(history['total_loss'][-log_interval:]) / log_interval
                avg_post = sum(history['post_interrupt_loss'][-log_interval:]) / log_interval
                avg_mem = sum(history['memory_size'][-log_interval:]) / log_interval

                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'post_int': f'{avg_post:.4f}',
                    'mem': f'{avg_mem:.1f}',
                })

            if max_steps and global_step >= max_steps:
                break

        if max_steps and global_step >= max_steps:
            break

        logger.info(f"Epoch {epoch+1} complete. Avg loss: {epoch_loss/n_docs:.4f}")

    return history


def evaluate_memory_dependency(
    memory_gpt: MemoryAugmentedGPT,
    dataloader: DataLoader,
    device: torch.device,
    max_docs: int = 50,
) -> Dict[str, float]:
    """
    Evaluate how much the model depends on memory.

    Compares:
    1. Loss with memory (after interruption)
    2. Loss without memory (after interruption)

    A good memory-augmented model should have significantly lower loss
    when memory is available.
    """
    memory_gpt.eval()

    with_memory_loss = 0.0
    without_memory_loss = 0.0
    n_docs = 0

    with torch.no_grad():
        for doc_chunks in dataloader:
            if n_docs >= max_docs:
                break

            if doc_chunks.n_chunks < 4:
                continue

            interrupt_point = doc_chunks.n_chunks // 2

            # === With memory ===
            memory_gpt.reset_memory()
            prev_memory_query = None  # Track query for causal retrieval

            # Build memory (with causal query passing)
            for i in range(interrupt_point):
                chunk = doc_chunks.chunks[i].unsqueeze(0).to(device)
                _, _, mem_out = memory_gpt(
                    chunk[:, :-1],
                    crystallize=True,
                    use_memory=True,
                    prev_memory_query=prev_memory_query
                )
                prev_memory_query = mem_out.get('next_memory_query')

            # Interrupt: reset hidden state but keep memory AND query
            memory_gpt.reset_hidden_state()
            # prev_memory_query preserved - represents context from before interrupt

            for i in range(interrupt_point, doc_chunks.n_chunks):
                chunk = doc_chunks.chunks[i].unsqueeze(0).to(device)
                targets = chunk[:, 1:].contiguous()
                inputs = chunk[:, :-1].contiguous()

                logits, _, mem_out = memory_gpt(
                    inputs,
                    crystallize=False,
                    use_memory=True,
                    prev_memory_query=prev_memory_query
                )
                prev_memory_query = mem_out.get('next_memory_query')

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                with_memory_loss += loss.item()

            # === Without memory ===
            memory_gpt.reset_memory()
            # No query tracking needed - we won't use memory

            # Process chunks but don't use memory
            for i in range(interrupt_point):
                chunk = doc_chunks.chunks[i].unsqueeze(0).to(device)
                memory_gpt(chunk[:, :-1], crystallize=True, use_memory=False)

            # Interrupt and evaluate WITHOUT memory
            memory_gpt.reset_hidden_state()

            for i in range(interrupt_point, doc_chunks.n_chunks):
                chunk = doc_chunks.chunks[i].unsqueeze(0).to(device)
                targets = chunk[:, 1:].contiguous()
                inputs = chunk[:, :-1].contiguous()

                # No prev_memory_query - memory won't be retrieved anyway
                logits, _, _ = memory_gpt(inputs, crystallize=False, use_memory=False)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                without_memory_loss += loss.item()

            n_docs += 1

    n_chunks_evaluated = n_docs * (doc_chunks.n_chunks - interrupt_point)

    return {
        'with_memory_loss': with_memory_loss / max(1, n_chunks_evaluated),
        'without_memory_loss': without_memory_loss / max(1, n_chunks_evaluated),
        'memory_benefit': (without_memory_loss - with_memory_loss) / max(1, n_chunks_evaluated),
        'n_docs_evaluated': n_docs,
    }


# =============================================================================
# Model Loading and Setup
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device):
    """Load a pretrained GPT model."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    saved_config = checkpoint.get('config', {})
    args = checkpoint.get('args', {})
    if isinstance(args, dict):
        for key in ['vocab_size', 'd_model', 'n_head', 'n_layer', 'max_seq_len', 'n_kv_head', 'd_ff']:
            if key in args and key not in saved_config:
                saved_config[key] = args[key]

    # Infer d_ff from weights if needed
    d_ff = saved_config.get('d_ff')
    if d_ff is None:
        state_dict = checkpoint.get('model_state_dict', {})
        for key, tensor in state_dict.items():
            if 'ffn.gate_up_proj.weight' in key:
                d_ff = tensor.shape[0] // 2
                break

    config = GPTConfig(
        vocab_size=saved_config.get('vocab_size', 128256),
        d_model=saved_config.get('d_model', 768),
        n_head=saved_config.get('n_head', 12),
        n_layer=saved_config.get('n_layer', 12),
        max_seq_len=saved_config.get('max_seq_len', 1024),
        n_kv_head=saved_config.get('n_kv_head'),
        dropout=saved_config.get('dropout', 0.0),
        d_ff=d_ff,
    )

    model = GPT(config)
    state_dict = checkpoint['model_state_dict']

    def strip_prefix(name):
        for prefix in ['module.', '_orig_mod.']:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name

    state_dict = {strip_prefix(k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    return model, config


def collate_doc_chunks(batch: List[DocumentChunks]) -> DocumentChunks:
    """Collate function that just returns single document (batch_size=1 for docs)."""
    # For now, we process one document at a time
    return batch[0]


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Memory-Augmented GPT with Resume-After-Interruption"
    )

    # Paths
    parser.add_argument("--checkpoint", default="tiny_pretrain_output/model_best_eval.pt",
                        help="Path to pretrained GPT checkpoint")
    parser.add_argument("--data_file", default=None,
                        help="Path to tokenized data file (.npy or .bin)")
    parser.add_argument("--data_dir", default="tiny_pretrain_output",
                        help="Directory containing tokenized data")
    parser.add_argument("--output_dir", default="resume_interruption_output",
                        help="Output directory for checkpoints")

    # Data parameters
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Size of each chunk")
    parser.add_argument("--min_chunks_per_doc", type=int, default=4,
                        help="Minimum chunks required per document")
    parser.add_argument("--max_chunks_per_doc", type=int, default=20,
                        help="Maximum chunks to take from a document")
    parser.add_argument("--eos_token_id", type=int, default=128000,
                        help="EOS token ID for document boundaries")

    # Training parameters
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accumulation_steps", type=int, default=4)

    # Interruption parameters
    parser.add_argument("--interrupt_prob", type=float, default=0.7,
                        help="Probability of interrupting each document")
    parser.add_argument("--min_chunks_before", type=int, default=2,
                        help="Minimum chunks before interruption")
    parser.add_argument("--resume_weight", type=float, default=2.0,
                        help="Extra weight for post-interruption loss")

    # Memory parameters
    parser.add_argument("--memory_capacity", type=int, default=500,
                        help="Maximum episodes to store")
    parser.add_argument("--crystallization_threshold", type=float, default=0.2,
                        help="Salience threshold for memory storage")
    parser.add_argument("--integration", choices=['residual', 'gated', 'attention'],
                        default='gated', help="Memory integration mode")

    # Other
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_gpt", action="store_true", default=True,
                        help="Freeze GPT backbone")
    parser.add_argument("--no_freeze_gpt", action="store_false", dest="freeze_gpt")

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Find data file
    if args.data_file is None:
        # Look for training data in data_dir
        import glob
        patterns = [
            os.path.join(args.data_dir, "training_tokens.npy"),
            os.path.join(args.data_dir, "training*_tokens.npy"),
            os.path.join(args.data_dir, "*training*.npy"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                args.data_file = matches[0]
                break

        if args.data_file is None:
            raise FileNotFoundError(f"No training data found in {args.data_dir}")

    logger.info(f"Using data file: {args.data_file}")

    # Load model
    gpt_model, gpt_config = load_model(args.checkpoint, device)
    logger.info(f"Loaded GPT: d_model={gpt_config.d_model}, n_layer={gpt_config.n_layer}")

    # Create memory-augmented wrapper
    memory_gpt = MemoryAugmentedGPT(
        gpt_model,
        memory_capacity=args.memory_capacity,
        crystallization_threshold=args.crystallization_threshold,
        memory_integration=args.integration,
        use_experiential=True,
        use_semantic=True,
        pad_token_id=args.eos_token_id,  # EOS is used for padding in this dataset
    ).to(device)

    logger.info(f"Created MemoryAugmentedGPT with {args.integration} integration")

    # Create dataset and dataloader
    dataset = NarrativeChunkDataset(
        token_file=args.data_file,
        chunk_size=args.chunk_size,
        min_chunks_per_doc=args.min_chunks_per_doc,
        max_chunks_per_doc=args.max_chunks_per_doc,
        eos_token_id=args.eos_token_id,
        shuffle_docs=True,
        seed=args.seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # One document at a time
        collate_fn=collate_doc_chunks,
        num_workers=0,  # IterableDataset works best with 0 workers
    )

    # Optimizer (only for memory components if GPT is frozen)
    if args.freeze_gpt:
        trainable_params = [p for n, p in memory_gpt.named_parameters()
                          if 'gpt.' not in n and p.requires_grad]
    else:
        trainable_params = [p for p in memory_gpt.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    logger.info(f"Optimizer: AdamW with lr={args.lr}, {len(trainable_params)} trainable param groups")

    # Initial evaluation
    logger.info("Running initial memory dependency evaluation...")
    eval_dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_doc_chunks,
        num_workers=0,
    )
    initial_eval = evaluate_memory_dependency(memory_gpt, eval_dataloader, device, max_docs=20)
    logger.info(f"Initial eval: with_mem={initial_eval['with_memory_loss']:.4f}, "
                f"without_mem={initial_eval['without_memory_loss']:.4f}, "
                f"benefit={initial_eval['memory_benefit']:.4f}")

    # Train
    logger.info("Starting resume-after-interruption training...")
    history = train_with_interruption(
        memory_gpt,
        dataloader,
        optimizer,
        device,
        n_epochs=args.n_epochs,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        interrupt_prob=args.interrupt_prob,
        min_chunks_before=args.min_chunks_before,
        resume_weight=args.resume_weight,
        accumulation_steps=args.accumulation_steps,
        freeze_gpt=args.freeze_gpt,
    )

    # Final evaluation
    logger.info("Running final memory dependency evaluation...")
    final_eval = evaluate_memory_dependency(memory_gpt, eval_dataloader, device, max_docs=20)
    logger.info(f"Final eval: with_mem={final_eval['with_memory_loss']:.4f}, "
                f"without_mem={final_eval['without_memory_loss']:.4f}, "
                f"benefit={final_eval['memory_benefit']:.4f}")

    # Save results
    results = {
        'args': vars(args),
        'initial_eval': initial_eval,
        'final_eval': final_eval,
        'training_history': {k: v[-100:] for k, v in history.items()},  # Last 100 steps
    }

    with open(os.path.join(args.output_dir, "training_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Save model
    checkpoint_path = os.path.join(args.output_dir, "memory_gpt_resume_trained.pt")
    torch.save({
        'memory_gpt_state_dict': memory_gpt.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'results': results,
        'episodic_memory': memory_gpt.memory.snapshot(),
    }, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Summary
    improvement = final_eval['memory_benefit'] - initial_eval['memory_benefit']
    logger.info(f"\n{'='*60}")
    logger.info("Training Complete!")
    logger.info(f"Memory benefit improvement: {improvement:+.4f}")
    logger.info(f"  Initial: {initial_eval['memory_benefit']:.4f}")
    logger.info(f"  Final:   {final_eval['memory_benefit']:.4f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
