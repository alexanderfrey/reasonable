#!/usr/bin/env python3
"""
Training script for Memory-Augmented PEM Prediction Module.

Extends train_prediction.py to train prediction + surprise together with a memory
system that:
1. Stores representations when surprise is HIGH
2. Retrieves from memory to IMPROVE predictions
3. Demonstrates LOWER surprise on repeated exposure to similar content

Usage:
    python -m pem.train_prediction_memory --data_dir /path/to/text/files --num_books 1000

Key additions over train_prediction.py:
- SemanticMemory with importance-based management
- Surprise computation and tracking
- Memory-benefit loss (penalize errors when relying on memory)
- Surprise calibration loss (learned magnitude should track raw)
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer

try:
    import wandb
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PerformanceContributionTracker:
    """
    Tracks gradient w.r.t. module outputs to measure performance contribution.

    Performance contribution = ||∂L/∂(module_output)||

    This tells us "how sensitive is the loss to this module's output",
    i.e., which modules matter most for the current loss.
    """

    def __init__(self):
        self.grad_norms = {}
        self.handles = []
        self._gradients = {}

    def register_hooks(self, model):
        """Register backward hooks on key CTM modules."""
        self.clear()

        # Access the CTM core through prediction module
        core = model.prediction.core

        # Hook for synapse output (a_t)
        def synapse_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self._gradients['synapse'] = grad_output[0].detach().norm().item()

        # Hook for NLM output (z_t)
        def nlm_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self._gradients['nlm'] = grad_output[0].detach().norm().item()

        # Hook for sync_to_output (y_t - the main output path)
        def sync_to_output_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self._gradients['sync_to_output'] = grad_output[0].detach().norm().item()

        # Hook for cross_attn output (o_t)
        def cross_attn_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self._gradients['cross_attn'] = grad_output[0].detach().norm().item()

        self.handles.append(core.synapse.register_full_backward_hook(synapse_hook))
        self.handles.append(core.nlm.register_full_backward_hook(nlm_hook))
        self.handles.append(core.sync_to_output.register_full_backward_hook(sync_to_output_hook))

        if core.cross_attn is not None:
            self.handles.append(core.cross_attn.register_full_backward_hook(cross_attn_hook))

    def get_contributions(self) -> dict:
        """Get performance contributions as fractions."""
        total = sum(self._gradients.values()) if self._gradients else 1.0
        if total > 0:
            fractions = {k: v / total for k, v in self._gradients.items()}
        else:
            fractions = {k: 0.0 for k in self._gradients}

        # Also return raw norms for debugging
        return {
            'fractions': fractions,
            'norms': self._gradients.copy(),
        }

    def clear(self):
        """Clear captured gradients."""
        self._gradients = {}

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._gradients = {}


@dataclass
class TrainingConfig:
    """Configuration for memory-augmented prediction training."""
    # Data
    data_dir: str = "/media/alexander/Tank1/text_files/text_files/"
    num_books: int = 1000
    context_size: int = 512
    seed: int = 42
    val_fraction: float = 0.05

    # Model
    feature_extractor: str = "deepseek-ai/Janus-Pro-1B"
    feature_dim: int = 1536

    # CTM architecture
    d_neurons: int = 512
    M: int = 16
    T: int = 8
    d_sync_out: int = 256
    d_sync_action: int = 256
    synapse_hidden: int = 1024
    nlm_hidden: int = 64
    internal_obs_residual: float = 0.1

    # Prediction horizons
    immediate_horizon: int = 64
    shortterm_horizon: int = 256
    longterm_horizon: int = 512

    # Token prediction
    vocab_size: int = 0
    token_prediction_weight: float = 0.5
    token_bottleneck: int = 256
    token_head_lr_scale: float = 0.1
    nlm_lr_scale: float = 1.0

    # Memory configuration (NEW)
    memory_slots: int = 2048
    memory_key_dim: int = 256
    memory_value_dim: int = 1536  # Now matches feature_dim for storing target features
    memory_context_window: int = 8  # Local context window for key encoding (total 2*8+1=17)
    memory_retrieval_temperature: float = 0.1
    memory_importance_decay: float = 0.8   # Faster decay (was 0.95) to evict stale memories
    memory_write_threshold: float = 0.1
    memory_attention_threshold: float = 0.5  # Higher threshold (was 0.3) - only use memory when confident
    memory_top_k: int = 4  # Number of top memories for CTM cross-attention

    # Surprise configuration (NEW)
    surprise_hidden_dim: int = 256
    surprise_cal_weight: float = 0.1     # Weight for surprise calibration loss
    memory_benefit_weight: float = 0.05  # Weight for memory benefit loss
    memory_alignment_weight: float = 0.1 # Weight for memory alignment loss (encourage retrieval)

    # Memory refresh (re-encode old keys with updated encoder)
    memory_refresh_every_n_steps: int = 100  # Refresh keys every N steps (0 to disable)

    # Memory verification (test if memory actually helps)
    memory_verify_every_n_steps: int = 200  # Run surprise decrease test every N steps (0 to disable)

    # Training
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # Checkpointing
    checkpoint_dir: str = "checkpoints/prediction_memory"
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

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "pem-prediction-memory"
    wandb_run_name: Optional[str] = None
    wandb_log_activations_every: int = 100
    wandb_activation_grid_size: int = 64


def find_text_files(data_dir: str, num_files: int, seed: int = 42) -> List[Path]:
    """Find and sample text files from directory."""
    logger.info(f"Scanning for text files in {data_dir}...")

    all_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.txt') and not f.startswith('._'):
                all_files.append(Path(root) / f)

    logger.info(f"Found {len(all_files)} text files")

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
        text = text.strip()
        if len(text) < min_length:
            return None
        return text
    except Exception as e:
        logger.debug(f"Failed to load {file_path}: {e}")
        return None


class BookDataset(IterableDataset):
    """Iterable dataset that streams text chunks from text files."""

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
        file_paths = self.file_paths.copy()
        if self.shuffle:
            random.shuffle(file_paths)

        for file_idx, file_path in enumerate(file_paths):
            text = load_and_clean_text(file_path, self.min_text_length)
            if text is None:
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            required_length = self.context_size + self.longterm_horizon

            for i in range(0, len(tokens) - required_length, self.context_size // 2):
                chunk_tokens = tokens[i:i + required_length]
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


class MemoryPredictionTrainer:
    """Trainer for memory-augmented prediction module."""

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

        # Running averages for memory benefit tracking
        self.benefit_pct_ema = 0.0
        self.ab_benefit_pct_ema = 0.0
        self.ema_alpha = 0.1  # Smoothing factor (higher = more weight on recent)

    def _setup_tokenizer(self):
        """Load tokenizer."""
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
        """Initialize MemoryAugmentedPrediction module."""
        logger.info("Initializing MemoryAugmentedPrediction module")
        from pem.memory_augmented_prediction import (
            MemoryAugmentedPrediction,
            MemoryAugmentedPredictionConfig,
        )
        from pem import PredictionTargets

        # memory_value_dim now matches feature_dim for storing target features
        memory_value_dim = self.config.feature_dim

        pred_config = MemoryAugmentedPredictionConfig(
            d_input=self.config.feature_dim,
            d_output=self.config.feature_dim,
            d_neurons=self.config.d_neurons,
            d_sync_out=self.config.d_sync_out,
            d_sync_internal=self.config.d_sync_action,
            M=self.config.M,
            T=self.config.T,
            synapse_hidden=self.config.synapse_hidden,
            nlm_hidden=self.config.nlm_hidden,
            internal_obs_residual=self.config.internal_obs_residual,
            immediate_horizon=self.config.immediate_horizon,
            shortterm_horizon=self.config.shortterm_horizon,
            longterm_horizon=self.config.longterm_horizon,
            vocab_size=self.config.vocab_size,
            token_prediction_weight=self.config.token_prediction_weight,
            token_bottleneck=self.config.token_bottleneck,
            # Memory config
            memory_slots=self.config.memory_slots,
            memory_key_dim=self.config.memory_key_dim,
            memory_value_dim=memory_value_dim,
            memory_context_window=self.config.memory_context_window,
            memory_retrieval_temperature=self.config.memory_retrieval_temperature,
            memory_importance_decay=self.config.memory_importance_decay,
            memory_write_threshold=self.config.memory_write_threshold,
            memory_attention_threshold=self.config.memory_attention_threshold,
            memory_top_k=self.config.memory_top_k,
            # Surprise config
            surprise_hidden_dim=self.config.surprise_hidden_dim,
        )

        self.prediction_module = MemoryAugmentedPrediction(pred_config).to(self.device)

        # Performance contribution tracker
        self.perf_tracker = PerformanceContributionTracker()
        self.perf_tracker.register_hooks(self.prediction_module)

        self.target_computer = PredictionTargets(
            immediate_horizon=self.config.immediate_horizon,
            shortterm_horizon=self.config.shortterm_horizon,
            longterm_horizon=self.config.longterm_horizon,
        )

        # Count parameters
        num_params = sum(p.numel() for p in self.prediction_module.parameters())
        trainable = sum(p.numel() for p in self.prediction_module.parameters() if p.requires_grad)
        logger.info(f"MemoryAugmentedPrediction: {num_params:,} params ({trainable:,} trainable)")
        logger.info(f"  Neurons: {self.config.d_neurons}, M: {self.config.M}, T: {self.config.T}")
        logger.info(f"  Memory: {self.config.memory_slots} slots, key_dim={self.config.memory_key_dim}")

    def _setup_optimizer(self):
        """Setup optimizer with differential learning rates."""
        token_head_params = []
        nlm_params = []
        other_params = []

        for name, param in self.prediction_module.named_parameters():
            if not param.requires_grad:
                continue
            if 'token_head' in name:
                token_head_params.append(param)
            elif 'nlm' in name:
                nlm_params.append(param)
            else:
                other_params.append(param)

        param_groups = []

        if other_params:
            param_groups.append({'params': other_params, 'lr': self.config.learning_rate})

        if nlm_params:
            nlm_lr = self.config.learning_rate * self.config.nlm_lr_scale
            param_groups.append({'params': nlm_params, 'lr': nlm_lr})
            logger.info(f"NLM LR: {nlm_lr:.2e} ({self.config.nlm_lr_scale}x base)")

        if token_head_params:
            token_head_lr = self.config.learning_rate * self.config.token_head_lr_scale
            param_groups.append({'params': token_head_params, 'lr': token_head_lr})
            logger.info(f"Token head LR: {token_head_lr:.2e} ({self.config.token_head_lr_scale}x base)")

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
        )

    def extract_features_from_text(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features, padding mask, and token IDs."""
        with torch.no_grad():
            self.feature_extractor._ensure_loaded()
            janus_tokenizer = self.feature_extractor.tokenizer

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

            features = self.feature_extractor(input_ids)
            features = features.float()

        return features, attention_mask, input_ids

    def _unpack_batch(self, batch: object) -> Tuple[List[str], Optional[float]]:
        """Extract text and progress from a collated batch."""
        if isinstance(batch, dict):
            texts = batch.get("texts", [])
            progress = batch.get("progress")
            return texts, progress
        return batch, None

    def _prepare_batch(self, batch: List[str]) -> Tuple[Optional[Tuple[torch.Tensor, dict]], int]:
        """Prepare inputs and targets for a batch."""
        features, padding_mask, input_ids = self.extract_features_from_text(batch)
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

        if self.config.vocab_size > 0:
            ctx = self.config.context_size
            targets['token_immediate'] = input_ids[:, self.config.immediate_horizon:ctx + self.config.immediate_horizon]
            targets['token_shortterm'] = input_ids[:, self.config.shortterm_horizon:ctx + self.config.shortterm_horizon]
            targets['token_longterm'] = input_ids[:, self.config.longterm_horizon:ctx + self.config.longterm_horizon]

        return (context_features, targets), S

    def _setup_loss_fn(self):
        """Setup the CTM loss function."""
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
        all_outputs: list,
        targets: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute CTM loss across all internal ticks."""
        return self.loss_fn(
            all_outputs,
            targets,
            self.prediction_module.prediction.readout_immediate,
            self.prediction_module.prediction.readout_shortterm,
            self.prediction_module.prediction.readout_longterm,
        )

    def compute_token_loss_per_tick(
        self,
        all_tick_outputs: list,
        targets: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute token prediction loss at key ticks."""
        T = len(all_tick_outputs)
        key_ticks = [0, T // 2, T - 1]
        key_ticks = sorted(set(t for t in key_ticks if 0 <= t < T))

        token_heads = {
            'immediate': self.prediction_module.prediction.token_head_immediate,
            'shortterm': self.prediction_module.prediction.token_head_shortterm,
            'longterm': self.prediction_module.prediction.token_head_longterm,
        }

        tick_losses = {}
        per_horizon_losses = {h: {} for h in ['immediate', 'shortterm', 'longterm']}

        for t in key_ticks:
            y_t = all_tick_outputs[t]
            tick_horizon_losses = []

            for horizon in ['immediate', 'shortterm', 'longterm']:
                logits = token_heads[horizon](y_t)
                target_ids = targets[f'token_{horizon}']

                logits_flat = logits.reshape(-1, logits.size(-1))
                target_flat = target_ids.reshape(-1)
                ce_loss = nn.functional.cross_entropy(logits_flat, target_flat)

                tick_horizon_losses.append(ce_loss)
                per_horizon_losses[horizon][t] = ce_loss

            tick_losses[t] = sum(tick_horizon_losses) / len(tick_horizon_losses)

        tick_loss_list = [tick_losses[t] for t in key_ticks]
        n_ticks = len(key_ticks)
        weights = [1.0 - 0.25 * i for i in range(n_ticks)]
        weight_sum = sum(weights)

        aggregated_loss = sum(w * loss for w, loss in zip(weights, tick_loss_list)) / weight_sum

        tick_loss_values = torch.stack(tick_loss_list)
        best_idx = torch.argmin(tick_loss_values.detach()).item()
        t1 = key_ticks[best_idx]

        loss_dict = {
            'token_loss': aggregated_loss.item(),
            'token_t1': t1,
            'token_loss_early': tick_losses[key_ticks[0]].item(),
            'token_loss_final': tick_losses[key_ticks[-1]].item(),
        }

        for t in key_ticks:
            loss_dict[f'token_loss_tick_{t}'] = tick_losses[t].item()

        for horizon in ['immediate', 'shortterm', 'longterm']:
            loss_dict[f'token_loss_{horizon}'] = per_horizon_losses[horizon][t1].item()

        return aggregated_loss, loss_dict

    def compute_surprise_losses(
        self,
        output,  # MemoryAugmentedPredictionOutput
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute surprise-related losses.

        1. Surprise calibration: learned magnitude should track raw cosine distance
        2. Memory benefit: if memory is used (high attention), predictions should be good
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        if output.surprise is None:
            return total_loss, loss_dict

        surprise = output.surprise

        # 1. Surprise calibration loss: |magnitude - raw_cosine_distance|
        # This encourages the learned magnitude to track the raw surprise signal
        surprise_cal_loss = F.mse_loss(surprise.magnitude, surprise.raw)
        loss_dict['surprise_cal_loss'] = surprise_cal_loss.item()
        total_loss = total_loss + self.config.surprise_cal_weight * surprise_cal_loss

        # 2. Memory benefit loss: when relying on memory, predictions should be good
        # memory_contrib = how much we're using memory (max attention)
        # pred_error = how wrong the prediction is (1 - cos_sim = surprise.raw)
        # If memory_contrib is high and pred_error is high, that's bad
        if output.memory_max_attention is not None:
            memory_contrib = output.memory_max_attention.mean()  # Average memory reliance
            pred_error = surprise.raw.mean()  # Average prediction error

            # Only penalize when actually using memory
            memory_benefit_loss = memory_contrib * pred_error
            loss_dict['memory_benefit_loss'] = memory_benefit_loss.item()
            total_loss = total_loss + self.config.memory_benefit_weight * memory_benefit_loss

            # 3. Memory alignment loss: encourage memory to be retrievable
            # High max_attention = good alignment between queries and stored keys
            # We want to REWARD high attention (= good retrieval), so use negative log
            # This encourages the key_encoder to produce keys that match future queries
            memory_alignment_loss = -torch.log(output.memory_max_attention + 1e-8).mean()
            loss_dict['memory_alignment_loss'] = memory_alignment_loss.item()
            total_loss = total_loss + self.config.memory_alignment_weight * memory_alignment_loss

        # Track surprise statistics
        loss_dict['surprise_magnitude_mean'] = surprise.magnitude.mean().item()
        loss_dict['surprise_raw_mean'] = surprise.raw.mean().item()

        return total_loss, loss_dict

    def compute_conditional_loss(
        self,
        output,  # MemoryAugmentedPredictionOutput
        targets: dict,
    ) -> dict:
        """
        Compute loss separately for high-attention vs low-attention positions.

        This tells us if memory is helping: loss_with_memory should be lower
        than loss_without_memory if memory is useful.
        """
        metrics = {}

        if output.memory_max_attention is None or output.surprise is None:
            return metrics

        # Use raw surprise as proxy for prediction error
        pred_error = output.surprise.raw  # (B, S)

        # Split by attention threshold
        high_attn_mask = output.memory_max_attention > self.config.memory_attention_threshold
        low_attn_mask = ~high_attn_mask

        # Count positions
        n_high = high_attn_mask.sum().item()
        n_low = low_attn_mask.sum().item()

        if n_high > 0:
            loss_high_attn = pred_error[high_attn_mask].mean().item()
            metrics['memory_loss_with_memory'] = loss_high_attn
            metrics['memory_n_high_attn'] = n_high

        if n_low > 0:
            loss_low_attn = pred_error[low_attn_mask].mean().item()
            metrics['memory_loss_without_memory'] = loss_low_attn
            metrics['memory_n_low_attn'] = n_low

        # Compute benefit: positive means memory helped
        if n_high > 0 and n_low > 0:
            benefit = loss_low_attn - loss_high_attn
            metrics['memory_benefit'] = benefit
            metrics['memory_benefit_pct'] = (benefit / max(loss_low_attn, 1e-8)) * 100

        return metrics

    @torch.no_grad()
    def compute_memory_ab_test(
        self,
        features: torch.Tensor,
        targets: dict,
    ) -> dict:
        """
        Direct A/B comparison: run with memory vs without memory.

        This works regardless of hit_rate by temporarily disabling memory
        integration entirely.

        Returns:
            Dict with loss_with, loss_without, and benefit metrics
        """
        was_training = self.prediction_module.training
        self.prediction_module.eval()

        # Run WITH memory (normal forward)
        out_with = self.prediction_module(features, targets=targets, write_to_memory=False)
        loss_with = out_with.surprise.raw.mean().item() if out_with.surprise else 0.0
        max_attn_with = out_with.memory_max_attention.mean().item() if out_with.memory_max_attention is not None else 0.0

        # Run WITHOUT memory (temporarily set threshold very high)
        old_threshold = self.prediction_module.config.memory_attention_threshold
        self.prediction_module.config.memory_attention_threshold = 999.0  # Never use memory

        out_without = self.prediction_module(features, targets=targets, write_to_memory=False)
        loss_without = out_without.surprise.raw.mean().item() if out_without.surprise else 0.0

        # Restore threshold
        self.prediction_module.config.memory_attention_threshold = old_threshold

        if was_training:
            self.prediction_module.train()

        # Compute benefit: positive means memory helped
        benefit = loss_without - loss_with
        benefit_pct = (benefit / max(loss_without, 1e-8)) * 100

        return {
            'ab_loss_with_memory': loss_with,
            'ab_loss_without_memory': loss_without,
            'ab_benefit': benefit,
            'ab_benefit_pct': benefit_pct,
            'ab_max_attn': max_attn_with,
            'ab_memory_helped': benefit > 0,
        }

    @torch.no_grad()
    def verify_surprise_decrease(
        self,
        features: torch.Tensor,
        targets: dict,
    ) -> dict:
        """
        Verify that surprise decreases on repeated content.

        This is a key verification test: the same content presented twice
        should have lower surprise the second time (memory helps).
        """
        was_training = self.prediction_module.training
        self.prediction_module.eval()

        # Save memory state
        memory_keys = self.prediction_module.memory.keys.clone()
        memory_values = self.prediction_module.memory.values.clone()
        memory_importance = self.prediction_module.memory.importance.clone()
        memory_occupied = self.prediction_module.memory.occupied.clone()
        memory_original_features = self.prediction_module.memory.original_features.clone()

        # Clear memory for clean test
        self.prediction_module.reset_memory()

        # First pass - memory empty, will write
        out1 = self.prediction_module(features, targets=targets, write_to_memory=True)
        surprise1 = out1.surprise.magnitude.mean().item() if out1.surprise else 0.0
        raw1 = out1.surprise.raw.mean().item() if out1.surprise else 0.0

        # Second pass - memory populated, should retrieve
        out2 = self.prediction_module(features, targets=targets, write_to_memory=False)
        surprise2 = out2.surprise.magnitude.mean().item() if out2.surprise else 0.0
        raw2 = out2.surprise.raw.mean().item() if out2.surprise else 0.0
        max_attn2 = out2.memory_max_attention.mean().item() if out2.memory_max_attention is not None else 0.0

        # Restore memory state
        self.prediction_module.memory.keys.copy_(memory_keys)
        self.prediction_module.memory.values.copy_(memory_values)
        self.prediction_module.memory.importance.copy_(memory_importance)
        self.prediction_module.memory.occupied.copy_(memory_occupied)
        self.prediction_module.memory.original_features.copy_(memory_original_features)

        if was_training:
            self.prediction_module.train()

        # Compute metrics
        surprise_decrease = surprise1 - surprise2
        raw_decrease = raw1 - raw2

        return {
            'surprise_first': surprise1,
            'surprise_second': surprise2,
            'surprise_decrease': surprise_decrease,
            'surprise_decrease_pct': (surprise_decrease / max(surprise1, 1e-8)) * 100,
            'raw_first': raw1,
            'raw_second': raw2,
            'raw_decrease': raw_decrease,
            'raw_decrease_pct': (raw_decrease / max(raw1, 1e-8)) * 100,
            'second_pass_max_attn': max_attn2,
            'memory_helped': surprise_decrease > 0,
        }

    def train_step(self, batch: object, return_activations: bool = False) -> dict:
        """Single training step."""
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
                "context_features": None,
                "targets": None,
            }

        context_features, targets = prepared
        autocast_ctx = torch.amp.autocast('cuda') if self.config.mixed_precision else nullcontext()

        with autocast_ctx:
            # Forward pass with targets for surprise computation
            output = self.prediction_module(
                context_features,
                targets=targets,
                write_to_memory=True,
            )

            # CTM prediction loss
            loss, loss_dict = self.compute_loss(output.all_tick_outputs, targets)

            # Token prediction loss
            if self.config.vocab_size > 0 and self.prediction_module.prediction.token_head_immediate is not None:
                token_loss, token_loss_dict = self.compute_token_loss_per_tick(
                    output.all_tick_outputs, targets
                )
                loss_dict.update(token_loss_dict)
                loss = loss + self.config.token_prediction_weight * token_loss

            # Surprise losses (calibration + memory benefit)
            surprise_loss, surprise_loss_dict = self.compute_surprise_losses(output)
            loss_dict.update(surprise_loss_dict)
            loss = loss + surprise_loss

        # Scale for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Get performance contributions (gradients captured by hooks during backward)
        perf_contrib = self.perf_tracker.get_contributions()
        self.perf_tracker.clear()

        # Decay memory importance (aging mechanism)
        self.prediction_module.decay_memory_importance()

        # Extract scalar values
        loss_values = {}
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                loss_values[k] = v.item()
            else:
                loss_values[k] = v

        # Add memory stats
        memory_stats = self.prediction_module.get_memory_stats()
        loss_values.update({f'memory_{k}': v for k, v in memory_stats.items()})

        # Add performance contribution stats
        loss_values.update({f'perf_frac_{k}': v for k, v in perf_contrib['fractions'].items()})
        loss_values.update({f'perf_norm_{k}': v for k, v in perf_contrib['norms'].items()})

        # Memory hit rate (positions with high attention)
        if output.memory_max_attention is not None:
            memory_hit_rate = (output.memory_max_attention > self.config.memory_attention_threshold).float().mean()
            loss_values['memory_hit_rate'] = memory_hit_rate.item()

        # Conditional loss: compare loss when using memory vs not
        conditional_metrics = self.compute_conditional_loss(output, targets)
        loss_values.update(conditional_metrics)

        # Update benefit EMA if we have a measurement
        if 'memory_benefit_pct' in conditional_metrics:
            self.benefit_pct_ema = (
                self.ema_alpha * conditional_metrics['memory_benefit_pct'] +
                (1 - self.ema_alpha) * self.benefit_pct_ema
            )
            loss_values['memory_benefit_pct_ema'] = self.benefit_pct_ema

        result = {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            **loss_values,
            "skipped": 0,
            "batch_tokens": len(texts) * self.config.context_size,
            "seq_len": seq_len,
            "progress": progress,
            # Store for verification (detached to avoid memory leaks)
            "context_features": context_features.detach(),
            "targets": {k: v.detach() if torch.is_tensor(v) else v for k, v in targets.items()},
        }

        if return_activations:
            result["ctm_output"] = output

        return result

    def eval_step(self, batch: object) -> dict:
        """Single evaluation step (no gradients, no memory writes)."""
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
                # Don't write to memory during eval
                output = self.prediction_module(
                    context_features,
                    targets=targets,
                    write_to_memory=False,
                )

                loss, loss_dict = self.compute_loss(output.all_tick_outputs, targets)

                if self.config.vocab_size > 0 and self.prediction_module.prediction.token_head_immediate is not None:
                    token_loss, token_loss_dict = self.compute_token_loss_per_tick(
                        output.all_tick_outputs, targets
                    )
                    loss_dict.update(token_loss_dict)
                    loss = loss + self.config.token_prediction_weight * token_loss

                surprise_loss, surprise_loss_dict = self.compute_surprise_losses(output)
                loss_dict.update(surprise_loss_dict)
                loss = loss + surprise_loss

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
        examples_since_log = 0
        tokens_since_log = 0
        batches_since_log = 0
        skipped_since_log = 0

        for batch_idx, batch in enumerate(dataloader):
            should_log_activations = (
                self.config.use_wandb and
                WANDB_AVAILABLE and
                self.config.wandb_log_activations_every > 0 and
                (self.global_step + 1) % self.config.wandb_log_activations_every == 0
            )

            loss_dict = self.train_step(batch, return_activations=should_log_activations)
            skipped = loss_dict.pop("skipped", 0)
            batch_tokens = loss_dict.pop("batch_tokens", 0)
            seq_len = loss_dict.pop("seq_len", 0)
            progress = loss_dict.pop("progress", None)
            ctm_output = loss_dict.pop("ctm_output", None)
            last_context_features = loss_dict.pop("context_features", None)
            last_targets = loss_dict.pop("targets", None)

            for k, v in loss_dict.items():
                accumulated_loss[k] = accumulated_loss.get(k, 0) + v
            accumulation_count += 1
            batches_since_log += 1

            skipped_since_log += skipped
            if batch_tokens > 0:
                examples_since_log += len(batch.get("texts", batch) if isinstance(batch, dict) else batch)
                tokens_since_log += batch_tokens

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

                # Periodic memory key refresh (re-encode old keys with updated encoder)
                if (self.config.memory_refresh_every_n_steps > 0 and
                    self.global_step % self.config.memory_refresh_every_n_steps == 0):
                    num_refreshed = self.prediction_module.refresh_memory_keys()
                    if num_refreshed > 0:
                        logger.debug(f"Refreshed {num_refreshed} memory keys at step {self.global_step}")

                # Periodic memory verification (test if memory actually helps)
                verify_metrics = None
                ab_metrics = None
                if (self.config.memory_verify_every_n_steps > 0 and
                    self.global_step % self.config.memory_verify_every_n_steps == 0 and
                    last_context_features is not None and last_targets is not None):
                    # Use last batch's features for verification
                    verify_metrics = self.verify_surprise_decrease(last_context_features, last_targets)
                    ab_metrics = self.compute_memory_ab_test(last_context_features, last_targets)

                    # Update A/B benefit EMA
                    self.ab_benefit_pct_ema = (
                        self.ema_alpha * ab_metrics['ab_benefit_pct'] +
                        (1 - self.ema_alpha) * self.ab_benefit_pct_ema
                    )

                    logger.info(
                        f"Memory verification @ step {self.global_step}: "
                        f"surprise {verify_metrics['surprise_first']:.4f} -> {verify_metrics['surprise_second']:.4f} "
                        f"({verify_metrics['surprise_decrease_pct']:+.1f}%) | "
                        f"A/B benefit: {ab_metrics['ab_benefit_pct']:+.1f}% (EMA: {self.ab_benefit_pct_ema:+.1f}%)"
                    )

                avg_loss = {k: v / accumulation_count for k, v in accumulated_loss.items()}
                epoch_losses.append(avg_loss)

                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    elapsed = time.time() - last_log_time
                    tokens_per_sec = tokens_since_log / max(elapsed, 1e-6)
                    lr = self.optimizer.param_groups[0]["lr"]
                    grad_norm_value = float(grad_norm) if torch.is_tensor(grad_norm) else float(grad_norm)

                    # Memory stats
                    mem_util = avg_loss.get('memory_utilization', 0) * 100
                    mem_hit = avg_loss.get('memory_hit_rate', 0) * 100
                    surp_mag = avg_loss.get('surprise_magnitude_mean', 0)
                    benefit_pct = avg_loss.get('memory_benefit_pct', 0)

                    # Performance contribution (which modules matter for loss)
                    syn_pct = avg_loss.get('perf_frac_synapse', 0) * 100
                    nlm_pct = avg_loss.get('perf_frac_nlm', 0) * 100
                    sync_pct = avg_loss.get('perf_frac_sync_to_output', 0) * 100
                    xattn_pct = avg_loss.get('perf_frac_cross_attn', 0) * 100

                    logger.info(
                        f"Epoch {epoch} | Step {self.global_step} | "
                        f"Loss: {avg_loss['loss']:.4f} | "
                        f"Surprise: {surp_mag:.4f} | "
                        f"Memory: {mem_util:.1f}% used, {mem_hit:.1f}% hit | "
                        f"Benefit: {benefit_pct:+.1f}% (EMA: {self.benefit_pct_ema:+.1f}%) | "
                        f"GradNorm: {grad_norm_value:.2f} | "
                        f"Throughput: {tokens_per_sec:.0f} tok/s"
                    )
                    logger.info(
                        f"  [PerfContrib] Syn: {syn_pct:.1f}% | NLM: {nlm_pct:.1f}% | Sync→Out: {sync_pct:.1f}% | XAttn: {xattn_pct:.1f}%"
                    )

                    # Wandb logging
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb_log = {
                            "train/loss": avg_loss['loss'],
                            "train/loss_final": avg_loss.get('loss_final', avg_loss['loss']),
                            "train/lr": lr,
                            "train/grad_norm": grad_norm_value,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/epoch": epoch,
                            # Surprise metrics
                            "surprise/magnitude_mean": avg_loss.get('surprise_magnitude_mean', 0),
                            "surprise/raw_mean": avg_loss.get('surprise_raw_mean', 0),
                            "surprise/calibration_loss": avg_loss.get('surprise_cal_loss', 0),
                            # Memory metrics
                            "memory/utilization": avg_loss.get('memory_utilization', 0),
                            "memory/avg_importance": avg_loss.get('memory_avg_importance', 0),
                            "memory/hit_rate": avg_loss.get('memory_hit_rate', 0),
                            "memory/benefit_loss": avg_loss.get('memory_benefit_loss', 0),
                            "memory/alignment_loss": avg_loss.get('memory_alignment_loss', 0),
                            # Memory benefit metrics (is memory helping?)
                            "memory/benefit": avg_loss.get('memory_benefit', 0),
                            "memory/benefit_pct": avg_loss.get('memory_benefit_pct', 0),
                            "memory/benefit_pct_ema": self.benefit_pct_ema,
                            "memory/loss_with_memory": avg_loss.get('memory_loss_with_memory', 0),
                            "memory/loss_without_memory": avg_loss.get('memory_loss_without_memory', 0),
                            # Performance contribution fractions (which modules matter for loss)
                            "perf_contrib/synapse": avg_loss.get('perf_frac_synapse', 0),
                            "perf_contrib/nlm": avg_loss.get('perf_frac_nlm', 0),
                            "perf_contrib/sync_to_output": avg_loss.get('perf_frac_sync_to_output', 0),
                            "perf_contrib/cross_attn": avg_loss.get('perf_frac_cross_attn', 0),
                        }

                        if progress is not None:
                            wandb_log["train/progress"] = progress

                        # Token prediction loss
                        if self.config.vocab_size > 0:
                            wandb_log["train/token_loss"] = avg_loss.get('token_loss', 0)

                        # Memory verification metrics (if just ran)
                        if verify_metrics is not None:
                            wandb_log.update({
                                "memory_verify/surprise_first": verify_metrics['surprise_first'],
                                "memory_verify/surprise_second": verify_metrics['surprise_second'],
                                "memory_verify/surprise_decrease": verify_metrics['surprise_decrease'],
                                "memory_verify/surprise_decrease_pct": verify_metrics['surprise_decrease_pct'],
                                "memory_verify/raw_decrease_pct": verify_metrics['raw_decrease_pct'],
                                "memory_verify/second_pass_attn": verify_metrics['second_pass_max_attn'],
                                "memory_verify/helped": float(verify_metrics['memory_helped']),
                            })

                        # A/B test metrics (if just ran)
                        if ab_metrics is not None:
                            wandb_log.update({
                                "memory_ab/loss_with_memory": ab_metrics['ab_loss_with_memory'],
                                "memory_ab/loss_without_memory": ab_metrics['ab_loss_without_memory'],
                                "memory_ab/benefit": ab_metrics['ab_benefit'],
                                "memory_ab/benefit_pct": ab_metrics['ab_benefit_pct'],
                                "memory_ab/benefit_pct_ema": self.ab_benefit_pct_ema,
                                "memory_ab/max_attn": ab_metrics['ab_max_attn'],
                                "memory_ab/helped": float(ab_metrics['ab_memory_helped']),
                            })

                        wandb.log(wandb_log, step=self.global_step)

                # Evaluation
                if val_loader is not None and self.config.eval_every_n_steps > 0:
                    if self.global_step % self.config.eval_every_n_steps == 0:
                        eval_loss = self.evaluate_steps(val_loader, f"step {self.global_step}")

                        if self.config.use_wandb and WANDB_AVAILABLE:
                            wandb.log({
                                "eval/loss": eval_loss['loss'],
                                "eval/surprise_magnitude_mean": eval_loss.get('surprise_magnitude_mean', 0),
                            }, step=self.global_step)

                        if eval_loss['loss'] < self.best_loss:
                            self.best_loss = eval_loss['loss']
                            self.save_checkpoint("best")
                            logger.info("New best model saved!")

                    last_log_time = time.time()
                    examples_since_log = 0
                    tokens_since_log = 0
                    batches_since_log = 0
                    skipped_since_log = 0

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

    def evaluate_steps(self, dataloader: DataLoader, step_label: str) -> dict:
        """Evaluate for a fixed number of steps."""
        was_training = self.prediction_module.training
        self.prediction_module.eval()
        accumulated_loss = {}
        accumulation_count = 0

        start_time = time.time()
        skipped = 0
        batches = 0

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

        if accumulation_count > 0:
            avg_eval_loss = {k: v / accumulation_count for k, v in accumulated_loss.items()}
        else:
            avg_eval_loss = {"loss": float('inf')}

        elapsed = time.time() - start_time
        logger.info(
            f"Eval {step_label} | "
            f"Loss: {avg_eval_loss['loss']:.4f} | "
            f"Surprise: {avg_eval_loss.get('surprise_magnitude_mean', 0):.4f} | "
            f"Batches: {batches} (skipped {skipped}) | "
            f"Time: {elapsed:.1f}s"
        )

        if was_training:
            self.prediction_module.train()

        return avg_eval_loss

    def save_checkpoint(self, name: str):
        """Save model checkpoint (includes memory state)."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")

        # Memory is automatically included via state_dict() (registered buffers)
        torch.save({
            "global_step": self.global_step,
            "model_state_dict": self.prediction_module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "best_loss": self.best_loss,
        }, checkpoint_path)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint (includes memory state)."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.prediction_module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float('inf'))

        # Memory state is automatically loaded via load_state_dict (registered buffers)
        memory_stats = self.prediction_module.get_memory_stats()
        logger.info(f"Resumed from step {self.global_step}, memory utilization: {memory_stats['utilization']*100:.1f}%")

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
                "memory_slots": self.config.memory_slots,
                "memory_key_dim": self.config.memory_key_dim,
                "surprise_cal_weight": self.config.surprise_cal_weight,
                "memory_benefit_weight": self.config.memory_benefit_weight,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
                "num_train_files": len(train_files),
                "num_val_files": len(val_files),
                "vocab_size": self.config.vocab_size,
            }
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=wandb_config,
            )
            logger.info(f"Wandb initialized: {wandb.run.name}")
        elif self.config.use_wandb and not WANDB_AVAILABLE:
            logger.warning("Wandb requested but not available.")

        logger.info("=" * 60)
        logger.info("Starting Memory-Augmented Prediction Training")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Train files: {len(train_files)} | Val files: {len(val_files)}")
        logger.info(f"Context: {self.config.context_size} tokens | Horizons: imm={self.config.immediate_horizon}, short={self.config.shortterm_horizon}, long={self.config.longterm_horizon}")
        logger.info(f"CTM: D={self.config.d_neurons}, M={self.config.M}, T={self.config.T}")
        logger.info(f"Memory: {self.config.memory_slots} slots, context_window={self.config.memory_context_window}, top_k={self.config.memory_top_k}")
        logger.info(f"Training: batch={self.config.batch_size}, grad_accum={self.config.gradient_accumulation_steps}, lr={self.config.learning_rate}")
        logger.info("=" * 60)

        # Create datasets
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

        val_loader = None
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

            # Memory stats at epoch end
            memory_stats = self.prediction_module.get_memory_stats()
            logger.info(f"Memory: {memory_stats['utilization']*100:.1f}% used, avg_importance={memory_stats['avg_importance']:.4f}")

            # Evaluation
            eval_loss = None
            if val_loader is not None and self.config.eval_every_n_epochs > 0 and (epoch % self.config.eval_every_n_epochs == 0):
                eval_loss = self.evaluate_steps(val_loader, f"epoch {epoch}")

            # Save best model
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

        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Memory-Augmented PEM Prediction Module")

    # Data arguments
    parser.add_argument("--data_dir", type=str,
                       default="/media/alexander/Tank1/text_files/text_files/",
                       help="Directory containing text files")
    parser.add_argument("--num_books", type=int, default=1000,
                       help="Number of books to use for training")
    parser.add_argument("--context_size", type=int, default=256,
                       help="Context size in tokens")
    parser.add_argument("--val_fraction", type=float, default=0.05,
                       help="Fraction of files for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Model arguments
    parser.add_argument("--feature_extractor", type=str,
                       default="deepseek-ai/Janus-Pro-1B",
                       help="Feature extractor model")

    # CTM architecture arguments
    parser.add_argument("--d_neurons", type=int, default=512,
                       help="Number of neurons")
    parser.add_argument("--M", type=int, default=16,
                       help="Pre-activation history length")
    parser.add_argument("--T", type=int, default=20,
                       help="Number of internal thinking ticks")
    parser.add_argument("--d_sync_out", type=int, default=256,
                       help="Sync pairs for output")
    parser.add_argument("--d_sync_action", type=int, default=256,
                       help="Sync pairs for attention")
    parser.add_argument("--synapse_hidden", type=int, default=1024,
                       help="Hidden dim in synapse")
    parser.add_argument("--nlm_hidden", type=int, default=64,
                       help="Hidden dim in NLM")

    # Token prediction arguments
    parser.add_argument("--vocab_size", type=int, default=0,
                       help="Vocab size for token prediction (0=disabled)")
    parser.add_argument("--token_prediction_weight", type=float, default=0.5,
                       help="Weight for token prediction loss")
    parser.add_argument("--token_head_lr_scale", type=float, default=0.1,
                       help="LR multiplier for token heads")
    parser.add_argument("--nlm_lr_scale", type=float, default=1.0,
                       help="LR multiplier for NLM")

    # Memory arguments (NEW)
    parser.add_argument("--memory_slots", type=int, default=2048,
                       help="Number of memory slots")
    parser.add_argument("--memory_key_dim", type=int, default=256,
                       help="Memory key dimension")
    parser.add_argument("--memory_context_window", type=int, default=8,
                       help="Local context window for key encoding (total 2*N+1 positions)")
    parser.add_argument("--memory_importance_decay", type=float, default=0.95,
                       help="Per-batch importance decay")
    parser.add_argument("--memory_retrieval_temperature", type=float, default=0.1,
                       help="Softmax temperature for retrieval (lower=sharper attention)")
    parser.add_argument("--memory_attention_threshold", type=float, default=0.3,
                       help="Min attention to use memory")
    parser.add_argument("--memory_top_k", type=int, default=4,
                       help="Number of top memories for CTM cross-attention")

    # Surprise arguments (NEW)
    parser.add_argument("--surprise_hidden_dim", type=int, default=256,
                       help="Hidden dim for surprise calibration")
    parser.add_argument("--surprise_cal_weight", type=float, default=0.1,
                       help="Weight for surprise calibration loss")
    parser.add_argument("--memory_benefit_weight", type=float, default=0.05,
                       help="Weight for memory benefit loss")
    parser.add_argument("--memory_alignment_weight", type=float, default=0.1,
                       help="Weight for memory alignment loss (encourage retrieval)")
    parser.add_argument("--memory_refresh_every", type=int, default=100,
                       help="Refresh memory keys every N steps (0 to disable)")
    parser.add_argument("--memory_verify_every", type=int, default=200,
                       help="Run memory verification every N steps (0 to disable)")

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
                       default="checkpoints/prediction_memory",
                       help="Directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                       help="Checkpoint to resume from")
    parser.add_argument("--eval_every_steps", type=int, default=100,
                       help="Evaluate every N steps")
    parser.add_argument("--eval_max_batches", type=int, default=200,
                       help="Max eval batches")

    # Hardware arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--no_mixed_precision", action="store_true",
                       help="Disable mixed precision")

    # Wandb arguments
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="pem-prediction-memory",
                       help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Wandb run name")

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
        # Token prediction
        vocab_size=args.vocab_size,
        token_prediction_weight=args.token_prediction_weight,
        token_head_lr_scale=args.token_head_lr_scale,
        nlm_lr_scale=args.nlm_lr_scale,
        # Memory (NEW)
        memory_slots=args.memory_slots,
        memory_key_dim=args.memory_key_dim,
        memory_context_window=args.memory_context_window,
        memory_importance_decay=args.memory_importance_decay,
        memory_retrieval_temperature=args.memory_retrieval_temperature,
        memory_attention_threshold=args.memory_attention_threshold,
        memory_top_k=args.memory_top_k,
        # Surprise (NEW)
        surprise_hidden_dim=args.surprise_hidden_dim,
        surprise_cal_weight=args.surprise_cal_weight,
        memory_benefit_weight=args.memory_benefit_weight,
        memory_alignment_weight=args.memory_alignment_weight,
        memory_refresh_every_n_steps=args.memory_refresh_every,
        memory_verify_every_n_steps=args.memory_verify_every,
        # Training
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        gradient_accumulation_steps=args.grad_accum,
        log_every_n_steps=args.log_every,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        mixed_precision=not args.no_mixed_precision,
        eval_every_n_steps=args.eval_every_steps,
        eval_max_batches=args.eval_max_batches,
        # Wandb
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    # Find text files
    file_paths = find_text_files(config.data_dir, config.num_books, seed=config.seed)

    if not file_paths:
        logger.error("No text files found!")
        return

    # Create trainer
    trainer = MemoryPredictionTrainer(config)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(file_paths)


if __name__ == "__main__":
    main()
