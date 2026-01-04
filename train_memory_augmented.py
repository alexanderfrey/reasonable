"""
Training script for Memory-Augmented GPT

Tests the core hypothesis: can episodic memory improve language modeling?

Usage:
    # Quick test (1000 steps)
    python train_memory_augmented.py --max_steps 1000

    # Full training
    python train_memory_augmented.py --n_epochs 1 --batch_size 8

    # Compare integration modes
    python train_memory_augmented.py --integration gated --max_steps 2000
"""

import argparse
import json
import logging
import math
import os
import time
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from experiential import (
    MemoryAugmentedGPT,
    memory_augmented_loss,
    experiential_loss,
    prediction_accuracy,
)
from model import GPT, GPTConfig

# Default paths
DEFAULT_CHECKPOINT = "tiny_pretrain_output/model_best_eval.pt"
DEFAULT_DATA_DIR = "tiny_pretrain_output"
DEFAULT_OUTPUT_DIR = "memory_augmented_output"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device):
    """Load a pretrained GPT model."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract config
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
                logger.info(f"Inferred d_ff={d_ff} from weights")
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

    logger.info(f"Model: d_model={config.d_model}, n_layer={config.n_layer}, vocab={config.vocab_size}")

    model = GPT(config)
    state_dict = checkpoint['model_state_dict']

    # Strip prefixes
    def strip_prefix(name):
        for prefix in ['module.', '_orig_mod.']:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name

    state_dict = {strip_prefix(k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    return model, config


def _load_token_memmap(token_file_path: str, dtype_name: Optional[str], data_type: str):
    if token_file_path.endswith(".npy"):
        return np.load(token_file_path, mmap_mode="r")
    if dtype_name is None:
        dtype_name = "uint32"
    return np.memmap(token_file_path, dtype=np.dtype(dtype_name), mode="r")


class DocumentSequentialDataset(Dataset):
    def __init__(
        self,
        token_file_path: str,
        doc_metadata_path: str,
        max_seq_len: int,
        stride: int,
        dtype_name: Optional[str],
        data_type: str = "Data"
    ):
        self.token_file_path = token_file_path
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.data_type = data_type

        if not os.path.exists(doc_metadata_path):
            raise FileNotFoundError(f"Doc metadata not found: {doc_metadata_path}")

        with open(doc_metadata_path, "r") as f:
            doc_meta = json.load(f)
        documents = doc_meta.get("documents", [])
        if not documents:
            raise ValueError(f"No documents found in {doc_metadata_path}")

        self.tokens = _load_token_memmap(token_file_path, dtype_name, data_type)

        self.start_positions = []
        self.doc_ids = []
        min_tokens_for_one_example = self.max_seq_len + 1

        for doc_id, doc in enumerate(documents):
            start = doc.get("start_token")
            end = doc.get("end_token")
            if start is None or end is None:
                continue
            if end - start < min_tokens_for_one_example:
                continue
            max_start = end - min_tokens_for_one_example
            for pos in range(start, max_start + 1, self.stride):
                self.start_positions.append(pos)
                self.doc_ids.append(doc_id)

        logger.info(
            f"{self.data_type} Sequential Dataset: {len(self.start_positions):,} examples "
            f"from {len(documents):,} documents using {self.token_file_path}"
        )

    def __len__(self):
        return len(self.start_positions)

    def __getitem__(self, idx):
        if idx >= len(self.start_positions):
            raise IndexError(
                f"Index {idx} out of bounds for {len(self.start_positions)} {self.data_type} examples"
            )

        start_idx = self.start_positions[idx]
        end_idx = start_idx + self.max_seq_len + 1
        token_chunk = self.tokens[start_idx:end_idx]

        input_ids = torch.tensor(token_chunk[:-1], dtype=torch.long)
        labels = torch.tensor(token_chunk[1:], dtype=torch.long)

        if input_ids.shape[0] != self.max_seq_len or labels.shape[0] != self.max_seq_len:
            raise ValueError(
                f"Data loading error: Unexpected sequence length at index {idx}. "
                f"Input: {input_ids.shape[0]}, Label: {labels.shape[0]}, Expected: {self.max_seq_len}."
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "doc_id": self.doc_ids[idx],
        }


class DocSequentialBatchSampler(Sampler[List[int]]):
    def __init__(self, doc_ids: List[int], batch_size: int, drop_last: bool = True):
        self.doc_ids = doc_ids
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._length = self._compute_length()

    def _compute_length(self) -> int:
        if not self.doc_ids:
            return 0
        total = 0
        current_doc = self.doc_ids[0]
        count = 0
        for doc_id in self.doc_ids:
            if doc_id != current_doc:
                total += count // self.batch_size if self.drop_last else math.ceil(count / self.batch_size)
                current_doc = doc_id
                count = 0
            count += 1
        total += count // self.batch_size if self.drop_last else math.ceil(count / self.batch_size)
        return total

    def __iter__(self):
        batch = []
        current_doc = None
        for idx, doc_id in enumerate(self.doc_ids):
            if current_doc is None:
                current_doc = doc_id
            if doc_id != current_doc:
                if batch and (not self.drop_last):
                    yield batch
                batch = []
                current_doc = doc_id
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        return self._length


def create_dataloader(
    data_dir: str,
    batch_size: int,
    seq_len: int,
    split: str = "training",
    sequential: bool = False,
    doc_metadata_path: Optional[str] = None
):
    """Create dataloader from pretokenized data."""
    from pretrain import PretokenizedDataset
    import glob

    # Find metadata
    pattern = os.path.join(data_dir, f"{split}*_metadata.json")
    meta_files = glob.glob(pattern)

    if not meta_files:
        raise FileNotFoundError(f"No metadata found matching {pattern}")

    with open(meta_files[0]) as f:
        meta = json.load(f)

    token_file = meta.get('token_file')
    if token_file and not os.path.isabs(token_file):
        token_file = os.path.join(data_dir, os.path.basename(token_file))

    num_examples = meta['num_examples']
    dtype_name = meta.get('dtype')
    stride = meta.get('stride', seq_len)

    logger.info(f"Loading {split} data: {num_examples:,} examples from {token_file}")

    if sequential and split == "training":
        if doc_metadata_path is None:
            doc_metadata_path = os.path.join(data_dir, "book_corpus_metadata.json")
        dataset = DocumentSequentialDataset(
            token_file,
            doc_metadata_path,
            max_seq_len=seq_len,
            stride=stride,
            dtype_name=dtype_name,
            data_type=split.capitalize()
        )
        batch_sampler = DocSequentialBatchSampler(dataset.doc_ids, batch_size, drop_last=True)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=2,
            pin_memory=True
        )
    else:
        dataset = PretokenizedDataset(
            token_file, num_examples, seq_len, seq_len, data_type=split.capitalize()
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "training"),
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )

    return dataloader


def train_epoch(
    memory_gpt: MemoryAugmentedGPT,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int = 50,
    max_steps: Optional[int] = None,
    lm_weight: float = 1.0,
    exp_weight: float = 0.1,
    affect_weight: float = 0.1,
    retrieval_weight: float = 0.1,
    retrieval_benefit_weight: float = 0.1,
    contrastive_weight: float = 0.1,
    accumulation_steps: int = 1,
    sequential: bool = False,
) -> Dict[str, List[float]]:
    """Train for one epoch."""
    memory_gpt.train()
    # Keep GPT backbone frozen initially
    for param in memory_gpt.gpt.parameters():
        param.requires_grad = False

    history = {
        'loss': [],
        'lm_loss': [],
        'exp_loss': [],
        'accuracy': [],
        'memory_size': [],
        'surprise': [],
        # Extended self-awareness metrics
        'affect_loss': [],
        'retrieval_loss': [],
        'meta_affect_surprise': [],
        'meta_retrieval_surprise': [],
        # Retrieval benefit metrics
        'retrieval_benefit': [],
        'retrieval_benefit_loss': [],
        # Contrastive loss
        'contrastive_loss': [],
    }

    total_loss = 0.0
    optimizer.zero_grad()

    prev_memory_query = None
    prev_doc_id = None

    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        # Target is shifted input (next token prediction)
        targets = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()

        if sequential:
            doc_ids = batch.get("doc_id")
            if doc_ids is None:
                raise ValueError("Sequential training requires doc_id in batch")
            if (doc_ids != doc_ids[0]).any():
                logger.warning("Batch spans multiple documents; resetting sequence state")
            batch_doc_id = int(doc_ids[0])
            if prev_doc_id is None or batch_doc_id != prev_doc_id:
                memory_gpt.reset_hidden_state()
                prev_memory_query = None
                prev_doc_id = batch_doc_id
        else:
            # Reset experiential state for each batch (shuffled data, not sequential)
            # This prevents state from leaking across unrelated documents
            if memory_gpt.experiential is not None:
                memory_gpt.experiential.reset_state(batch_size=input_ids.size(0))

        # Forward pass with causal memory retrieval
        # For shuffled data, prev_memory_query=None means no memory retrieval
        # (each batch is an independent sequence, no previous context to query from)
        # Memories are still crystallized and available for sequential evaluation
        logits, hidden, mem_out = memory_gpt(
            input_ids,
            crystallize=True,
            use_memory=True,
            prev_memory_query=prev_memory_query
        )
        if sequential and mem_out.get('next_memory_query') is not None:
            prev_memory_query = mem_out['next_memory_query'].detach()
        elif not sequential:
            prev_memory_query = None

        # Compute loss (includes extended self-awareness losses)
        loss, loss_dict = memory_augmented_loss(
            logits, targets, mem_out,
            lm_weight=lm_weight,
            exp_weight=exp_weight,
            affect_weight=affect_weight,
            retrieval_weight=retrieval_weight,
            retrieval_benefit_weight=retrieval_benefit_weight,
            contrastive_weight=contrastive_weight,
        )

        # Scale for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        total_loss += loss.item() * accumulation_steps

        # Optimizer step
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(memory_gpt.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Metrics
        if mem_out.get('prediction') is not None:
            acc = prediction_accuracy(mem_out['prediction'], mem_out['target'])
            surprise = mem_out['surprise'].mean().item()
        else:
            acc = 0.0
            surprise = 0.0

        history['loss'].append(loss_dict['total_loss'])
        history['lm_loss'].append(loss_dict['lm_loss'])
        history['exp_loss'].append(loss_dict.get('exp_loss', 0.0))
        history['accuracy'].append(acc)
        history['memory_size'].append(mem_out['episodic_size'])
        history['surprise'].append(surprise)
        # Extended self-awareness metrics
        history['affect_loss'].append(loss_dict.get('affect_loss', 0.0))
        history['retrieval_loss'].append(loss_dict.get('retrieval_loss', 0.0))
        history['meta_affect_surprise'].append(loss_dict.get('mean_meta_affect_surprise', 0.0))
        history['meta_retrieval_surprise'].append(loss_dict.get('mean_meta_retrieval_surprise', 0.0))
        # Retrieval benefit metrics
        history['retrieval_benefit'].append(loss_dict.get('retrieval_benefit', 0.0))
        history['retrieval_benefit_loss'].append(loss_dict.get('retrieval_benefit_loss', 0.0))
        # Contrastive loss
        history['contrastive_loss'].append(loss_dict.get('contrastive_loss', 0.0))

        # Log
        if step % log_interval == 0:
            avg_loss = total_loss / (step + 1)
            retrieval_benefit = loss_dict.get('retrieval_benefit', 0.0)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lm': f'{loss_dict["lm_loss"]:.3f}',
                'mem': mem_out['episodic_size'],
                'acc': f'{acc:.3f}',
                'ret_ben': f'{retrieval_benefit:.3f}',  # positive = memory helps
            })

        if max_steps and step >= max_steps:
            break

    return history


def evaluate(
    memory_gpt: MemoryAugmentedGPT,
    dataloader: DataLoader,
    device: torch.device,
    max_steps: int = 100
) -> Dict[str, float]:
    """Evaluate the model."""
    memory_gpt.eval()

    total_lm_loss = 0.0
    total_exp_loss = 0.0
    total_acc = 0.0
    n_steps = 0

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()

            logits, hidden, mem_out = memory_gpt(
                input_ids,
                crystallize=False,
                use_memory=True
            )

            loss, loss_dict = memory_augmented_loss(logits, targets, mem_out)

            total_lm_loss += loss_dict['lm_loss']
            total_exp_loss += loss_dict.get('exp_loss', 0.0)

            if mem_out.get('prediction') is not None:
                total_acc += prediction_accuracy(mem_out['prediction'], mem_out['target'])

            n_steps += 1

    return {
        'lm_loss': total_lm_loss / n_steps,
        'exp_loss': total_exp_loss / n_steps,
        'accuracy': total_acc / n_steps,
        'memory_size': memory_gpt.memory.size
    }


def main():
    parser = argparse.ArgumentParser(description="Train Memory-Augmented GPT")

    # Data/model paths
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)

    # Training params
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--sequential", action="store_true",
                        help="Train sequentially over corpus with doc-boundary resets")
    parser.add_argument("--doc_metadata", default=None,
                        help="Path to document metadata (default: data_dir/book_corpus_metadata.json)")

    # Memory params
    parser.add_argument("--integration", choices=['residual', 'gated', 'attention', 'cross_attention'], default='gated')
    parser.add_argument("--memory_capacity", type=int, default=1000)
    parser.add_argument("--crystallization_threshold", type=float, default=0.2)

    # Loss weights
    parser.add_argument("--lm_weight", type=float, default=1.0)
    parser.add_argument("--exp_weight", type=float, default=0.1)
    # Extended self-awareness loss weights
    parser.add_argument("--affect_weight", type=float, default=0.1,
                        help="Weight for meta-affect loss (predict own emotions)")
    parser.add_argument("--retrieval_weight", type=float, default=0.1,
                        help="Weight for meta-retrieval loss (predict what will be remembered)")
    # Retrieval benefit loss - trains memory to actually help prediction
    parser.add_argument("--retrieval_benefit_weight", type=float, default=0.1,
                        help="Weight for retrieval benefit loss (penalize when memory hurts)")
    # Contrastive learning for memory relevance
    parser.add_argument("--contrastive_weight", type=float, default=0.1,
                        help="Weight for contrastive memory loss (teach which memories are relevant)")
    # Memory pre-population (for cross-attention cold start)
    parser.add_argument("--prepopulate_steps", type=int, default=500,
                        help="Steps to run for memory pre-population (0 to disable)")
    parser.add_argument("--min_memories", type=int, default=100,
                        help="Minimum memories to create during pre-population")
    # Salience calibration
    parser.add_argument("--meta_surprise_salience_weight", type=float, default=1.0,
                        help="How much meta-surprise boosts salience (default 1.0)")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_interval", type=int, default=50)

    args = parser.parse_args()
    device = torch.device(args.device)

    if args.sequential and args.batch_size != 1:
        logger.warning("Sequential training requires batch_size=1; overriding.")
        args.batch_size = 1

    logger.info("=" * 60)
    logger.info("Memory-Augmented GPT Training")
    logger.info(f"  Integration: {args.integration}")
    logger.info(f"  Memory capacity: {args.memory_capacity}")
    logger.info(f"  Crystallization threshold: {args.crystallization_threshold}")
    logger.info(f"  Meta-surprise salience weight: {args.meta_surprise_salience_weight}")
    logger.info(f"  Batch size: {args.batch_size} x {args.accumulation_steps} accumulation")
    logger.info(f"  Sequential training: {args.sequential}")
    logger.info(f"  Loss weights: lm={args.lm_weight}, exp={args.exp_weight}, "
                f"affect={args.affect_weight}, retrieval={args.retrieval_weight}, "
                f"retrieval_benefit={args.retrieval_benefit_weight}, "
                f"contrastive={args.contrastive_weight}")
    logger.info(f"  Pre-population: {args.prepopulate_steps} steps, min {args.min_memories} memories")
    logger.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    gpt, config = load_model(args.checkpoint, device)

    # Wrap with memory
    memory_gpt = MemoryAugmentedGPT(
        gpt,
        memory_capacity=args.memory_capacity,
        crystallization_threshold=args.crystallization_threshold,
        memory_integration=args.integration,
        use_experiential=True,
        meta_surprise_salience_weight=args.meta_surprise_salience_weight,
    ).to(device)

    n_params = sum(p.numel() for p in memory_gpt.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # Data
    train_loader = create_dataloader(
        args.data_dir,
        args.batch_size,
        args.seq_len,
        "training",
        sequential=args.sequential,
        doc_metadata_path=args.doc_metadata
    )
    eval_loader = create_dataloader(args.data_dir, args.batch_size, args.seq_len, "evaluation")

    # Optimizer - only train memory components
    memory_params = [p for n, p in memory_gpt.named_parameters()
                     if 'gpt.' not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(memory_params, lr=args.lr, weight_decay=0.01)

    logger.info(f"Optimizing {len(memory_params)} parameter groups")

    # Pre-populate memory bank (avoids cold-start for cross-attention)
    if args.prepopulate_steps > 0:
        logger.info(f"\n--- Pre-populating memory bank ---")
        prepop_stats = memory_gpt.prepopulate_memory(
            train_loader,
            device,
            max_steps=args.prepopulate_steps,
            min_memories=args.min_memories
        )
        logger.info(f"Pre-population complete: {prepop_stats['memories_added']} memories created "
                    f"in {prepop_stats['steps']} steps (total: {prepop_stats['final_size']})")

    # Training loop
    all_history = []
    start_time = time.time()

    for epoch in range(args.n_epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{args.n_epochs} ---")

        # Train
        history = train_epoch(
            memory_gpt,
            train_loader,
            optimizer,
            device,
            log_interval=args.log_interval,
            max_steps=args.max_steps,
            lm_weight=args.lm_weight,
            exp_weight=args.exp_weight,
            affect_weight=args.affect_weight,
            retrieval_weight=args.retrieval_weight,
            retrieval_benefit_weight=args.retrieval_benefit_weight,
            contrastive_weight=args.contrastive_weight,
            accumulation_steps=args.accumulation_steps,
            sequential=args.sequential,
        )
        all_history.append(history)

        # Evaluate
        eval_metrics = evaluate(memory_gpt, eval_loader, device, max_steps=100)
        logger.info(f"Eval: lm_loss={eval_metrics['lm_loss']:.4f}, "
                    f"exp_loss={eval_metrics['exp_loss']:.4f}, "
                    f"acc={eval_metrics['accuracy']:.4f}, "
                    f"memory={eval_metrics['memory_size']}")

        # Save checkpoint
        save_path = os.path.join(args.output_dir, f"memory_gpt_epoch_{epoch+1}.pt")
        torch.save({
            'memory_gpt_state_dict': memory_gpt.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'config': vars(config),
            'args': vars(args),
            'eval_metrics': eval_metrics,
        }, save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    elapsed = time.time() - start_time
    logger.info(f"\nTraining complete in {elapsed/60:.1f} minutes")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Final Results:")
    final_history = all_history[-1]
    logger.info(f"  Final LM loss: {sum(final_history['lm_loss'][-100:])/100:.4f}")
    logger.info(f"  Final exp accuracy: {sum(final_history['accuracy'][-100:])/100:.4f}")
    logger.info(f"  Final memory size: {memory_gpt.memory.size}")
    logger.info(f"  Memory stats: {memory_gpt.get_memory_stats()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
