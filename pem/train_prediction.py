#!/usr/bin/env python3
"""
Training script for the PEM CTM Prediction Module.

This script trains the CTM-based prediction module to predict future token features
at multiple time horizons (immediate, short-term, long-term) using continuous
state evolution.

Usage:
    python -m pem.train_prediction --data_dir /path/to/text/files --num_books 1000

Requirements:
    - torch
    - transformers (for tokenizer)
    - Janus Pro model (for feature extraction)
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple, Iterator
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer

try:
    import wandb
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import numpy as np
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for CTM prediction module training."""
    # Data
    data_dir: str = "/media/alexander/Tank1/text_files/text_files/"
    num_books: int = 1000
    context_size: int = 256
    seed: int = 42
    val_fraction: float = 0.05

    # Model - use same tokenizer as feature extractor for consistency
    feature_extractor: str = "deepseek-ai/Janus-Pro-1B"
    feature_dim: int = 1536

    # CTM architecture (faithful to original paper)
    d_neurons: int = 512              # D - number of neurons
    M: int = 16                       # Pre-activation history length
    T: int = 8                        # Number of internal thinking ticks
    d_sync_out: int = 256             # Sync pairs for output
    d_sync_action: int = 256          # Sync pairs for attention
    synapse_hidden: int = 1024        # Hidden dim in synapse U-NET
    nlm_hidden: int = 64              # Hidden dim in per-neuron MLPs

    # Prediction horizons
    immediate_horizon: int = 8
    shortterm_horizon: int = 64
    longterm_horizon: int = 256

    # Training
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # Checkpointing
    checkpoint_dir: str = "checkpoints/prediction"
    save_every_n_steps: int = 500
    log_every_n_steps: int = 1
    eval_max_batches: int = 200
    eval_every_n_epochs: int = 0
    eval_every_n_steps: int = 100
    eval_log_every_n_batches: int = 10

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    num_workers: int = 4

    # Wandb logging
    use_wandb: bool = True
    wandb_project: str = "pem-ctm-prediction"
    wandb_run_name: Optional[str] = None
    wandb_log_activations_every: int = 100  # Log activation grid every N steps
    wandb_activation_grid_size: int = 64    # Number of neurons to show in grid (8x8)


def find_text_files(data_dir: str, num_files: int, seed: int = 42) -> List[Path]:
    """Find and sample text files from directory."""
    logger.info(f"Scanning for text files in {data_dir}...")

    all_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.txt') and not f.startswith('._'):
                all_files.append(Path(root) / f)

    logger.info(f"Found {len(all_files)} text files")

    # Sample random subset
    random.seed(seed)
    if len(all_files) > num_files:
        sampled = random.sample(all_files, num_files)
    else:
        sampled = all_files

    logger.info(f"Selected {len(sampled)} files for training")
    return sampled


def split_train_val(
    file_paths: List[Path],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    """Split files into train/val subsets."""
    if val_fraction <= 0.0 or len(file_paths) < 2:
        return file_paths, []

    files = file_paths.copy()
    rng = random.Random(seed)
    rng.shuffle(files)

    val_size = int(len(files) * val_fraction)
    val_size = max(1, min(val_size, len(files) - 1))

    val_files = files[:val_size]
    train_files = files[val_size:]
    return train_files, val_files


def load_and_clean_text(file_path: Path, min_length: int = 1000) -> Optional[str]:
    """Load text file and perform basic cleaning."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Basic cleaning
        text = text.strip()

        # Skip very short files
        if len(text) < min_length:
            return None

        return text
    except Exception as e:
        logger.debug(f"Failed to load {file_path}: {e}")
        return None


class BookDataset(IterableDataset):
    """
    Iterable dataset that streams text chunks from text files.

    Uses tokenizer for consistent chunking, yields text strings
    for feature extraction. Using the same tokenizer throughout
    ensures consistent token boundaries.
    """

    def __init__(
        self,
        file_paths: List[Path],
        tokenizer,
        context_size: int = 256,
        longterm_horizon: int = 256,
        min_text_length: int = 1000,
        shuffle: bool = True,
    ):
        self.file_paths = file_paths
        self.total_files = max(1, len(file_paths))
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.longterm_horizon = longterm_horizon
        self.min_text_length = min_text_length
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[str]:
        """Yield text chunks from files."""
        file_paths = self.file_paths.copy()

        if self.shuffle:
            random.shuffle(file_paths)

        for file_idx, file_path in enumerate(file_paths):
            text = load_and_clean_text(file_path, self.min_text_length)
            if text is None:
                continue

            # Tokenize entire text to get consistent chunking
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # We need extra tokens for prediction targets
            required_length = self.context_size + self.longterm_horizon

            # Create overlapping chunks
            for i in range(0, len(tokens) - required_length, self.context_size // 2):
                chunk_tokens = tokens[i:i + required_length]
                # Decode back to text for Janus to re-tokenize
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                progress = (file_idx + 1) / self.total_files
                yield chunk_text, progress


def collate_fn(batch: List[object]) -> dict:
    """Collate text chunks into a batch."""
    if batch and isinstance(batch[0], tuple):
        texts = [item[0] for item in batch]
        progress = max(item[1] for item in batch)
        return {"texts": texts, "progress": progress}
    return {"texts": batch, "progress": None}


def create_neuron_activation_grid(
    post_activation_history: torch.Tensor,
    n_neurons: int = 64,
    seq_idx: int = 0,
    batch_idx: int = 0,
) -> plt.Figure:
    """
    Create a grid visualization of neuron activations over internal ticks.

    Args:
        post_activation_history: (B, S, D, T) tensor where:
            - B = batch size
            - S = sequence length
            - D = number of neurons
            - T = internal ticks
        n_neurons: Number of neurons to display (should be a perfect square)
        seq_idx: Which sequence position to visualize
        batch_idx: Which batch item to visualize

    Returns:
        matplotlib Figure with grid of line plots
    """
    if not WANDB_AVAILABLE:
        return None

    # Get data for one sequence position from one batch item
    # Shape: (D, T)
    data = post_activation_history[batch_idx, seq_idx].detach().cpu().numpy()
    D, T = data.shape

    # Limit to n_neurons
    n_neurons = min(n_neurons, D)
    grid_size = int(np.sqrt(n_neurons))
    n_neurons = grid_size * grid_size  # Make it a perfect square

    # Sample neurons evenly across the range
    neuron_indices = np.linspace(0, D - 1, n_neurons, dtype=int)

    # Create figure
    fig, axes = plt.subplots(
        grid_size, grid_size,
        figsize=(12, 12),
        sharex=True,
        sharey=True,
    )
    fig.suptitle(f'Neuron Activations over {T} Internal Ticks\n(seq_pos={seq_idx})', fontsize=14)

    # Time axis
    t = np.arange(T)

    # Plot each neuron
    for idx, (ax, neuron_idx) in enumerate(zip(axes.flat, neuron_indices)):
        activation = data[neuron_idx]
        ax.plot(t, activation, linewidth=1.0, color='steelblue')
        ax.fill_between(t, 0, activation, alpha=0.3, color='steelblue')
        ax.set_title(f'N{neuron_idx}', fontsize=8, pad=2)
        ax.tick_params(axis='both', which='both', labelsize=6)
        ax.set_xlim(0, T - 1)

        # Add subtle grid
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Only show y-axis label on leftmost
        if idx % grid_size != 0:
            ax.set_yticklabels([])

    # Common labels
    fig.text(0.5, 0.02, 'Internal Tick (t)', ha='center', fontsize=10)
    fig.text(0.02, 0.5, 'Activation', va='center', rotation='vertical', fontsize=10)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])

    return fig


def create_sync_matrix_heatmap(
    sync_matrix: torch.Tensor,
    seq_idx: int = 0,
    batch_idx: int = 0,
) -> plt.Figure:
    """
    Create a heatmap visualization of the synchronization matrix.

    Args:
        sync_matrix: (B, S, D, D) tensor
        seq_idx: Which sequence position to visualize
        batch_idx: Which batch item to visualize

    Returns:
        matplotlib Figure with heatmap
    """
    if not WANDB_AVAILABLE:
        return None

    # Get data for one sequence position
    data = sync_matrix[batch_idx, seq_idx].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(data, cmap='RdBu_r', aspect='auto')
    ax.set_title(f'Synchronization Matrix S_t (seq_pos={seq_idx})', fontsize=12)
    ax.set_xlabel('Neuron j')
    ax.set_ylabel('Neuron i')
    plt.colorbar(im, ax=ax, label='Sync strength')

    plt.tight_layout()
    return fig


class PredictionTrainer:
    """Trainer for the prediction module."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.global_step = 0
        self.best_loss = float('inf')

        # Setup
        self._setup_tokenizer()
        self._setup_feature_extractor()
        self._setup_prediction_module()
        self._setup_optimizer()
        self._setup_loss_fn()

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None

    def _setup_tokenizer(self):
        """Load tokenizer from the same model as feature extractor for consistency."""
        logger.info(f"Loading tokenizer from: {self.config.feature_extractor}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.feature_extractor,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Tokenizer loaded, vocab_size={self.tokenizer.vocab_size}")

    def _setup_feature_extractor(self):
        """Load frozen feature extractor."""
        logger.info(f"Loading feature extractor: {self.config.feature_extractor}")
        from pem import create_feature_extractor

        self.feature_extractor = create_feature_extractor(
            model_name_or_path=self.config.feature_extractor,
            output_dim=self.config.feature_dim,
            learning_mode="frozen",
        )
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        logger.info("Feature extractor loaded and frozen")

    def _setup_prediction_module(self):
        """Initialize CTM prediction module (faithful to original paper)."""
        logger.info("Initializing CTM prediction module")
        from pem.ctm_prediction_module import CTMPrediction, CTMPredictionConfig
        from pem import PredictionTargets

        ctm_config = CTMPredictionConfig(
            d_model=self.config.feature_dim,
            d_neurons=self.config.d_neurons,
            d_sync_out=self.config.d_sync_out,
            d_sync_action=self.config.d_sync_action,
            M=self.config.M,
            T=self.config.T,
            synapse_hidden=self.config.synapse_hidden,
            nlm_hidden=self.config.nlm_hidden,
            # Horizons
            immediate_horizon=self.config.immediate_horizon,
            shortterm_horizon=self.config.shortterm_horizon,
            longterm_horizon=self.config.longterm_horizon,
        )

        self.prediction_module = CTMPrediction(ctm_config).to(self.device)
        self.target_computer = PredictionTargets(
            immediate_horizon=self.config.immediate_horizon,
            shortterm_horizon=self.config.shortterm_horizon,
            longterm_horizon=self.config.longterm_horizon,
        )

        # Count parameters
        num_params = sum(p.numel() for p in self.prediction_module.parameters())
        trainable = sum(p.numel() for p in self.prediction_module.parameters() if p.requires_grad)
        logger.info(f"CTM Prediction module: {num_params:,} params ({trainable:,} trainable)")
        logger.info(f"  Neurons: {self.config.d_neurons}, M: {self.config.M}, T: {self.config.T}")
        logger.info(f"  Sync pairs: out={self.config.d_sync_out}, action={self.config.d_sync_action}")

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.prediction_module.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def extract_features_from_text(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features and padding mask using frozen feature extractor."""
        with torch.no_grad():
            self.feature_extractor._ensure_loaded()
            janus_tokenizer = self.feature_extractor.tokenizer

            # Tokenize with Janus tokenizer
            encoded = janus_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.context_size + self.config.longterm_horizon,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(self.device).bool()

            # Get features
            features = self.feature_extractor(input_ids)
            features = features.float()  # Ensure float32

        return features, attention_mask

    def _unpack_batch(self, batch: object) -> Tuple[List[str], Optional[float]]:
        """Extract text and progress from a collated batch."""
        if isinstance(batch, dict):
            texts = batch.get("texts", [])
            progress = batch.get("progress")
            return texts, progress
        return batch, None

    def _prepare_batch(self, batch: List[str]) -> Tuple[Optional[Tuple[torch.Tensor, dict]], int]:
        """Prepare inputs and targets for a batch.

        CTM prediction doesn't need external sync - it generates its own
        internal state from features.
        """
        B = len(batch)
        features, padding_mask = self.extract_features_from_text(batch)
        _, S, _ = features.shape

        if S < self.config.context_size + self.config.longterm_horizon:
            return None, S

        context_features = features[:, :self.config.context_size]

        targets_full = self.target_computer.compute_targets_efficient(
            features,
            padding_mask=padding_mask,
        )

        targets = {
            'immediate': targets_full['immediate'][:, :self.config.context_size],
            'shortterm': targets_full['shortterm'][:, :self.config.context_size],
            'longterm': targets_full['longterm'][:, :self.config.context_size],
            'immediate_valid': targets_full['immediate_valid'][:, :self.config.context_size],
            'shortterm_valid': targets_full['shortterm_valid'][:, :self.config.context_size],
            'longterm_valid': targets_full['longterm_valid'][:, :self.config.context_size],
        }

        return (context_features, targets), S

    def _setup_loss_fn(self):
        """Setup the CTM loss function following the paper."""
        from pem.ctm_prediction_module import CTMLoss
        self.loss_fn = CTMLoss(
            immediate_weight=1.0,
            shortterm_weight=0.5,
            longterm_weight=0.3,
            use_cosine=True,
            use_mse=True,
            mse_weight=0.1,
        )

    def compute_loss(
        self,
        all_outputs: list,  # List of y_t at each tick
        targets: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute CTM loss across all internal ticks.

        Following the paper:
        - Compute loss at each tick
        - Find t1 = argmin(loss), t2 = argmax(certainty)
        - Final loss = (L_t1 + L_t2) / 2
        """
        return self.loss_fn(
            all_outputs,
            targets,
            self.prediction_module.readout_immediate,
            self.prediction_module.readout_shortterm,
            self.prediction_module.readout_longterm,
        )

    def train_step(self, batch: object, return_activations: bool = False) -> dict:
        """Single training step.

        Args:
            batch: Input batch
            return_activations: If True, include CTM activations in output for visualization
        """
        texts, progress = self._unpack_batch(batch)
        prepared, seq_len = self._prepare_batch(texts)
        if prepared is None:
            logger.warning(f"Sequence too short: {seq_len} tokens, skipping batch")
            return {
                "loss": 0.0,
                "skipped": 1,
                "batch_tokens": 0,
                "seq_len": seq_len,
                "progress": progress,
                "ctm_output": None,
            }

        context_features, targets = prepared
        autocast_ctx = torch.amp.autocast('cuda') if self.config.mixed_precision else nullcontext()
        with autocast_ctx:
            # CTM prediction with all tick outputs for proper CTM loss
            output = self.prediction_module(context_features, return_all_ticks=True)

            # CTM loss: compute at each tick, aggregate via min-loss and max-certainty
            loss, loss_dict = self.compute_loss(output.all_outputs, targets)

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Extract scalar values from loss_dict (some might be int tensors like t1, t2)
        loss_values = {}
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                loss_values[k] = v.item()
            else:
                loss_values[k] = v

        result = {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            **loss_values,
            "skipped": 0,
            "batch_tokens": len(texts) * self.config.context_size,
            "seq_len": seq_len,
            "progress": progress,
        }

        # Include CTM output for visualization if requested
        if return_activations:
            result["ctm_output"] = output

        return result

    def eval_step(self, batch: object) -> dict:
        """Single evaluation step (no gradients)."""
        texts, progress = self._unpack_batch(batch)
        prepared, seq_len = self._prepare_batch(texts)
        if prepared is None:
            return {
                "loss": 0.0,
                "skipped": 1,
                "batch_tokens": 0,
                "seq_len": seq_len,
                "progress": progress,
            }

        context_features, targets = prepared
        autocast_ctx = torch.amp.autocast('cuda') if self.config.mixed_precision else nullcontext()
        with torch.no_grad():
            with autocast_ctx:
                output = self.prediction_module(context_features, return_all_ticks=True)
                loss, loss_dict = self.compute_loss(output.all_outputs, targets)

        # Extract scalar values from loss_dict
        loss_values = {}
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                loss_values[k] = v.item()
            else:
                loss_values[k] = v

        return {
            "loss": loss.item(),
            **loss_values,
            "skipped": 0,
            "batch_tokens": len(texts) * self.config.context_size,
            "seq_len": seq_len,
            "progress": progress,
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int, val_loader: Optional[DataLoader] = None) -> dict:
        """Train for one epoch."""
        self.prediction_module.train()

        epoch_losses = []
        accumulated_loss = {}
        accumulation_count = 0

        start_time = time.time()
        last_log_time = start_time
        last_log_step = self.global_step
        examples_since_log = 0
        tokens_since_log = 0
        batches_since_log = 0
        skipped_since_log = 0
        seq_len_sum = 0
        seq_len_count = 0

        for batch_idx, batch in enumerate(dataloader):
            # Determine if we should capture activations for visualization
            should_log_activations = (
                self.config.use_wandb and
                WANDB_AVAILABLE and
                self.config.wandb_log_activations_every > 0 and
                (self.global_step + 1) % self.config.wandb_log_activations_every == 0
            )

            # Training step
            loss_dict = self.train_step(batch, return_activations=should_log_activations)
            skipped = loss_dict.pop("skipped", 0)
            batch_tokens = loss_dict.pop("batch_tokens", 0)
            seq_len = loss_dict.pop("seq_len", 0)
            progress = loss_dict.pop("progress", None)
            ctm_output = loss_dict.pop("ctm_output", None)

            # Accumulate losses for logging
            for k, v in loss_dict.items():
                accumulated_loss[k] = accumulated_loss.get(k, 0) + v
            accumulation_count += 1
            batches_since_log += 1
            skipped_since_log += skipped
            if batch_tokens > 0:
                examples_since_log += len(batch)
                tokens_since_log += batch_tokens
                seq_len_sum += seq_len
                seq_len_count += 1

            # Gradient step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.prediction_module.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.prediction_module.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # Average accumulated losses
                avg_loss = {k: v / accumulation_count for k, v in accumulated_loss.items()}
                epoch_losses.append(avg_loss)

                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    elapsed = time.time() - last_log_time
                    examples_per_sec = examples_since_log / max(elapsed, 1e-6)
                    tokens_per_sec = tokens_since_log / max(elapsed, 1e-6)
                    avg_seq_len = seq_len_sum / max(seq_len_count, 1)
                    lr = self.optimizer.param_groups[0]["lr"]
                    grad_norm_value = float(grad_norm) if torch.is_tensor(grad_norm) else float(grad_norm)
                    progress_text = f"Progress: {progress * 100:.1f}% | " if progress is not None else ""
                    # CTM-specific metrics
                    avg_t1 = avg_loss.get('t1', 0)  # Average tick selected by min-loss
                    avg_t2 = avg_loss.get('t2', 0)  # Average tick selected by max-certainty
                    certainty = avg_loss.get('certainty_mean', 0)
                    loss_final = avg_loss.get('loss_final', avg_loss['loss'])

                    logger.info(
                        f"Epoch {epoch} | Step {self.global_step} | "
                        f"Loss: {avg_loss['loss']:.4f} | "
                        f"Loss_final: {loss_final:.4f} | "
                        f"AvgTick: t1={avg_t1:.1f}, t2={avg_t2:.1f} | "
                        f"Certainty: {certainty:.3f} | "
                        f"LR: {lr:.2e} | "
                        f"GradNorm: {grad_norm_value:.2f} | "
                        f"Throughput: {tokens_per_sec:.0f} tok/s | "
                        f"{progress_text}"
                        f"Time: {elapsed:.1f}s"
                    )

                    # Wandb logging
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb_log = {
                            "train/loss": avg_loss['loss'],
                            "train/loss_t1": avg_loss.get('loss_t1', 0),
                            "train/loss_t2": avg_loss.get('loss_t2', 0),
                            "train/loss_final": loss_final,
                            "train/avg_t1": avg_t1,  # Average tick selected by min-loss
                            "train/avg_t2": avg_t2,  # Average tick selected by max-certainty
                            "train/certainty_mean": certainty,
                            "train/certainty_final": avg_loss.get('certainty_final', 0),
                            "train/lr": lr,
                            "train/grad_norm": grad_norm_value,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/examples_per_sec": examples_per_sec,
                            "train/epoch": epoch,
                        }

                        # Log per-tick losses
                        for t in range(self.config.T):
                            tick_loss = avg_loss.get(f'loss_tick_{t}', None)
                            if tick_loss is not None:
                                wandb_log[f"train/loss_tick_{t}"] = tick_loss

                        if progress is not None:
                            wandb_log["train/progress"] = progress

                        # Log activation visualizations if we captured them
                        if ctm_output is not None:
                            try:
                                # Create neuron activation grid
                                activation_fig = create_neuron_activation_grid(
                                    ctm_output.post_activation_history,
                                    n_neurons=self.config.wandb_activation_grid_size,
                                    seq_idx=0,  # First sequence position
                                    batch_idx=0,
                                )
                                if activation_fig is not None:
                                    wandb_log["activations/neuron_grid"] = wandb.Image(activation_fig)
                                    plt.close(activation_fig)

                                # Create sync matrix heatmap
                                sync_fig = create_sync_matrix_heatmap(
                                    ctm_output.sync_matrix,
                                    seq_idx=0,
                                    batch_idx=0,
                                )
                                if sync_fig is not None:
                                    wandb_log["activations/sync_matrix"] = wandb.Image(sync_fig)
                                    plt.close(sync_fig)

                                # Log activation statistics
                                Z = ctm_output.post_activation_history
                                wandb_log["activations/mean"] = Z.mean().item()
                                wandb_log["activations/std"] = Z.std().item()
                                wandb_log["activations/max"] = Z.max().item()
                                wandb_log["activations/min"] = Z.min().item()

                            except Exception as e:
                                logger.warning(f"Failed to log activations: {e}")

                        wandb.log(wandb_log, step=self.global_step)

                # Evaluation
                if val_loader is not None and self.config.eval_every_n_steps > 0:
                    if self.global_step % self.config.eval_every_n_steps == 0:
                        eval_loss = self.evaluate_steps(val_loader, f"step {self.global_step}")

                        # Log eval to wandb
                        if self.config.use_wandb and WANDB_AVAILABLE:
                            wandb.log({
                                "eval/loss": eval_loss['loss'],
                                "eval/immediate_loss": eval_loss.get('immediate_loss', 0),
                                "eval/shortterm_loss": eval_loss.get('shortterm_loss', 0),
                                "eval/longterm_loss": eval_loss.get('longterm_loss', 0),
                            }, step=self.global_step)

                        if eval_loss['loss'] < self.best_loss:
                            self.best_loss = eval_loss['loss']
                            self.save_checkpoint("best")
                            logger.info("New best model saved!")
                    last_log_time = time.time()
                    last_log_step = self.global_step
                    examples_since_log = 0
                    tokens_since_log = 0
                    batches_since_log = 0
                    skipped_since_log = 0
                    seq_len_sum = 0
                    seq_len_count = 0

                # Checkpointing
                if self.global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

                # Reset accumulation
                accumulated_loss = {}
                accumulation_count = 0

        # Compute epoch averages
        if epoch_losses:
            avg_epoch_loss = {
                k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
                for k in epoch_losses[0].keys()
            }
        else:
            avg_epoch_loss = {"loss": float('inf')}

        return avg_epoch_loss

    def evaluate_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """Evaluate for one epoch."""
        return self.evaluate_steps(dataloader, f"epoch {epoch}")

    def evaluate_steps(self, dataloader: DataLoader, step_label: str) -> dict:
        """Evaluate for a fixed number of steps."""
        was_training = self.prediction_module.training
        self.prediction_module.eval()
        accumulated_loss = {}
        accumulation_count = 0

        start_time = time.time()
        skipped = 0
        batches = 0
        eval_target = self.config.eval_max_batches if self.config.eval_max_batches else None

        for batch_idx, batch in enumerate(dataloader):
            if self.config.eval_max_batches and batch_idx >= self.config.eval_max_batches:
                break

            loss_dict = self.eval_step(batch)
            skipped += loss_dict.pop("skipped", 0)
            loss_dict.pop("batch_tokens", None)
            loss_dict.pop("seq_len", None)
            loss_dict.pop("progress", None)

            for k, v in loss_dict.items():
                accumulated_loss[k] = accumulated_loss.get(k, 0) + v
            accumulation_count += 1
            batches += 1

            if self.config.eval_log_every_n_batches > 0:
                if (batch_idx + 1) % self.config.eval_log_every_n_batches == 0:
                    if accumulation_count > 0:
                        avg_loss = {k: v / accumulation_count for k, v in accumulated_loss.items()}
                    else:
                        avg_loss = {"loss": float('inf')}
                    elapsed = time.time() - start_time
                    if eval_target:
                        progress = (batch_idx + 1) / eval_target * 100
                        progress_text = f"Progress: {progress:.1f}% | "
                        batch_text = f"Batch {batch_idx + 1}/{eval_target}"
                    else:
                        progress_text = ""
                        batch_text = f"Batch {batch_idx + 1}"
                    logger.info(
                        f"Eval {step_label} | {batch_text} | "
                        f"Loss: {avg_loss['loss']:.4f} | "
                        f"Immediate: {avg_loss.get('immediate_loss', 0):.4f} | "
                        f"ShortTerm: {avg_loss.get('shortterm_loss', 0):.4f} | "
                        f"LongTerm: {avg_loss.get('longterm_loss', 0):.4f} | "
                        f"{progress_text}"
                        f"Time: {elapsed:.1f}s"
                    )

        if accumulation_count > 0:
            avg_eval_loss = {
                k: v / accumulation_count for k, v in accumulated_loss.items()
            }
        else:
            avg_eval_loss = {"loss": float('inf')}

        elapsed = time.time() - start_time
        logger.info(
            f"Eval {step_label} | "
            f"Loss: {avg_eval_loss['loss']:.4f} | "
            f"Immediate: {avg_eval_loss.get('immediate_loss', 0):.4f} | "
            f"ShortTerm: {avg_eval_loss.get('shortterm_loss', 0):.4f} | "
            f"LongTerm: {avg_eval_loss.get('longterm_loss', 0):.4f} | "
            f"Batches: {batches} (skipped {skipped}) | "
            f"Time: {elapsed:.1f}s"
        )

        if was_training:
            self.prediction_module.train()

        return avg_eval_loss

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")

        torch.save({
            "global_step": self.global_step,
            "model_state_dict": self.prediction_module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "best_loss": self.best_loss,
        }, checkpoint_path)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.prediction_module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float('inf'))

        logger.info(f"Resumed from step {self.global_step}")

    def train(self, file_paths: List[Path]):
        """Main training loop."""
        train_files, val_files = split_train_val(
            file_paths,
            val_fraction=self.config.val_fraction,
            seed=self.config.seed,
        )

        # Initialize wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb_config = {
                "d_neurons": self.config.d_neurons,
                "M": self.config.M,
                "T": self.config.T,
                "d_sync_out": self.config.d_sync_out,
                "d_sync_action": self.config.d_sync_action,
                "synapse_hidden": self.config.synapse_hidden,
                "nlm_hidden": self.config.nlm_hidden,
                "feature_dim": self.config.feature_dim,
                "context_size": self.config.context_size,
                "immediate_horizon": self.config.immediate_horizon,
                "shortterm_horizon": self.config.shortterm_horizon,
                "longterm_horizon": self.config.longterm_horizon,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
                "num_train_files": len(train_files),
                "num_val_files": len(val_files),
            }
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=wandb_config,
            )
            # Watch model for gradient logging
            wandb.watch(self.prediction_module, log="gradients", log_freq=100)
            logger.info(f"Wandb initialized: {wandb.run.name}")
        elif self.config.use_wandb and not WANDB_AVAILABLE:
            logger.warning("Wandb requested but not available. Install with: pip install wandb matplotlib")

        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Train files: {len(train_files)} | Val files: {len(val_files)}")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Context size: {self.config.context_size}")
        logger.info(f"CTM: D={self.config.d_neurons}, M={self.config.M}, T={self.config.T}")
        logger.info(f"Horizons: immediate={self.config.immediate_horizon}, "
                   f"shortterm={self.config.shortterm_horizon}, "
                   f"longterm={self.config.longterm_horizon}")
        logger.info("=" * 60)

        # Create dataset
        dataset = BookDataset(
            file_paths=train_files,
            tokenizer=self.tokenizer,
            context_size=self.config.context_size,
            longterm_horizon=self.config.longterm_horizon,
            shuffle=True,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        if val_files:
            val_dataset = BookDataset(
                file_paths=val_files,
                tokenizer=self.tokenizer,
                context_size=self.config.context_size,
                longterm_horizon=self.config.longterm_horizon,
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_fn,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
        else:
            val_loader = None

        # Training loop
        for epoch in range(1, self.config.num_epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{self.config.num_epochs}")
            logger.info("=" * 60)

            epoch_start = time.time()
            avg_loss = self.train_epoch(dataloader, epoch, val_loader=val_loader)
            epoch_time = time.time() - epoch_start

            logger.info(f"\nEpoch {epoch} complete in {epoch_time:.1f}s")
            logger.info(f"Average loss: {avg_loss['loss']:.4f}")

            eval_loss = None
            if val_loader is not None and self.config.eval_every_n_epochs > 0 and (epoch % self.config.eval_every_n_epochs == 0):
                eval_loss = self.evaluate_epoch(val_loader, epoch)

            # Save best model (prefer eval loss if available)
            metric_loss = eval_loss['loss'] if eval_loss else avg_loss['loss']
            if metric_loss < self.best_loss:
                self.best_loss = metric_loss
                self.save_checkpoint("best")
                logger.info("New best model saved!")

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch}")

        # Final save
        self.save_checkpoint("final")
        logger.info("\nTraining complete!")

        # Finish wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train PEM CTM Prediction Module")

    # Data arguments
    parser.add_argument("--data_dir", type=str,
                       default="/media/alexander/Tank1/text_files/text_files/",
                       help="Directory containing text files")
    parser.add_argument("--num_books", type=int, default=1000,
                       help="Number of books to use for training")
    parser.add_argument("--context_size", type=int, default=256,
                       help="Context size in tokens")
    parser.add_argument("--val_fraction", type=float, default=0.05,
                       help="Fraction of files to reserve for evaluation")
    parser.add_argument("--eval_max_batches", type=int, default=200,
                       help="Max number of eval batches per epoch (0 = no limit)")
    parser.add_argument("--eval_every", type=int, default=1,
                       help="Run evaluation every N epochs")
    parser.add_argument("--eval_every_steps", type=int, default=100,
                       help="Run evaluation every N optimizer steps")
    parser.add_argument("--eval_log_every", type=int, default=10,
                       help="Log evaluation progress every N eval batches")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling/splitting")

    # Model arguments
    parser.add_argument("--feature_extractor", type=str,
                       default="deepseek-ai/Janus-Pro-1B",
                       help="Feature extractor model (tokenizer derived from same model)")

    # CTM architecture arguments (faithful to original paper)
    parser.add_argument("--d_neurons", type=int, default=512,
                       help="Number of neurons (D in paper)")
    parser.add_argument("--M", type=int, default=16,
                       help="Pre-activation history length")
    parser.add_argument("--T", type=int, default=20,
                       help="Number of internal thinking ticks")
    parser.add_argument("--d_sync_out", type=int, default=256,
                       help="Number of sync pairs for output")
    parser.add_argument("--d_sync_action", type=int, default=256,
                       help="Number of sync pairs for attention")
    parser.add_argument("--synapse_hidden", type=int, default=1024,
                       help="Hidden dim in synapse U-NET")
    parser.add_argument("--nlm_hidden", type=int, default=64,
                       help="Hidden dim in per-neuron MLPs")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--grad_accum", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--log_every", type=int, default=1,
                       help="Log every N optimizer steps")

    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str,
                       default="checkpoints/prediction",
                       help="Directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                       help="Checkpoint to resume from")

    # Hardware arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--no_mixed_precision", action="store_true",
                       help="Disable mixed precision training")

    # Wandb arguments
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="pem-ctm-prediction",
                       help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Wandb run name (auto-generated if not specified)")
    parser.add_argument("--wandb_log_activations_every", type=int, default=100,
                       help="Log activation visualizations every N steps (0 to disable)")
    parser.add_argument("--wandb_activation_grid_size", type=int, default=64,
                       help="Number of neurons to show in activation grid (must be perfect square)")

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        num_books=args.num_books,
        context_size=args.context_size,
        seed=args.seed,
        val_fraction=args.val_fraction,
        feature_extractor=args.feature_extractor,
        # CTM architecture
        d_neurons=args.d_neurons,
        M=args.M,
        T=args.T,
        d_sync_out=args.d_sync_out,
        d_sync_action=args.d_sync_action,
        synapse_hidden=args.synapse_hidden,
        nlm_hidden=args.nlm_hidden,
        # Training
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        gradient_accumulation_steps=args.grad_accum,
        log_every_n_steps=args.log_every,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        mixed_precision=not args.no_mixed_precision,
        eval_max_batches=args.eval_max_batches,
        eval_every_n_epochs=args.eval_every,
        eval_every_n_steps=args.eval_every_steps,
        eval_log_every_n_batches=args.eval_log_every,
        # Wandb
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_log_activations_every=args.wandb_log_activations_every,
        wandb_activation_grid_size=args.wandb_activation_grid_size,
    )

    # Find text files
    file_paths = find_text_files(config.data_dir, config.num_books, seed=config.seed)

    if not file_paths:
        logger.error("No text files found!")
        return

    # Create trainer
    trainer = PredictionTrainer(config)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(file_paths)


if __name__ == "__main__":
    main()
