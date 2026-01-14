#!/usr/bin/env python3
"""
Training script for the PEM Curiosity Module.

Curiosity = the drive to reduce uncertainty and gain information.

Training signal: Curiosity should predict surprise.
- High curiosity + high surprise = well-calibrated
- Low curiosity + high surprise = should have been more curious (loss)

Prerequisites:
    - Trained prediction module checkpoint

Usage:
    python -m pem.train_curiosity \
        --prediction_checkpoint checkpoints/prediction/best.pt \
        --num_books 1000
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CuriosityTrainingConfig:
    """Configuration for curiosity module training."""
    # Data
    data_dir: str = "/media/alexander/Tank1/text_files/text_files/"
    num_books: int = 1000
    context_size: int = 256

    # Models
    tokenizer_name: str = "meta-llama/Meta-Llama-3-8B"
    feature_extractor: str = "deepseek-ai/Janus-Pro-1B"
    prediction_checkpoint: str = "checkpoints/prediction/best.pt"
    feature_dim: int = 1536
    sync_dim: int = 512

    # Prediction horizons (must match prediction module)
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

    # Loss weights
    calibration_weight: float = 1.0
    exploration_weight: float = 0.5
    novelty_weight: float = 0.3

    # Checkpointing
    checkpoint_dir: str = "checkpoints/curiosity"
    save_every_n_steps: int = 500
    log_every_n_steps: int = 50

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True


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
    """Streams text chunks from book files."""

    def __init__(
        self,
        file_paths: List[Path],
        tokenizer,
        context_size: int = 256,
        longterm_horizon: int = 256,
        shuffle: bool = True,
    ):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.longterm_horizon = longterm_horizon
        self.shuffle = shuffle

    def __iter__(self):
        file_paths = self.file_paths.copy()
        if self.shuffle:
            random.shuffle(file_paths)

        for file_path in file_paths:
            text = load_and_clean_text(file_path)
            if text is None:
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            required_length = self.context_size + self.longterm_horizon

            for i in range(0, len(tokens) - required_length, self.context_size // 2):
                chunk_tokens = tokens[i:i + required_length]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                yield chunk_text


class CuriosityTrainer:
    """Trainer for the curiosity module."""

    def __init__(self, config: CuriosityTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.global_step = 0
        self.best_loss = float('inf')

        # Setup components
        self._setup_tokenizer()
        self._setup_feature_extractor()
        self._setup_prediction_module()
        self._setup_surprise_module()
        self._setup_curiosity_module()
        self._setup_optimizer()

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None

    def _setup_tokenizer(self):
        """Load tokenizer."""
        logger.info(f"Loading tokenizer: {self.config.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

    def _setup_prediction_module(self):
        """Load trained prediction module (frozen)."""
        logger.info(f"Loading prediction module from: {self.config.prediction_checkpoint}")
        from pem import PredictionModule, PredictionConfig, PredictionTargets

        # Load checkpoint
        checkpoint = torch.load(self.config.prediction_checkpoint, map_location=self.device)

        # Recreate config
        pred_config = PredictionConfig(
            sync_pairs=self.config.sync_dim,
            d_model=self.config.feature_dim,
            n_head=8,
            immediate_horizon=self.config.immediate_horizon,
            shortterm_horizon=self.config.shortterm_horizon,
            longterm_horizon=self.config.longterm_horizon,
        )

        self.prediction_module = PredictionModule(pred_config).to(self.device)
        self.prediction_module.load_state_dict(checkpoint["model_state_dict"])
        self.prediction_module.eval()

        # Freeze prediction module
        for param in self.prediction_module.parameters():
            param.requires_grad = False

        self.target_computer = PredictionTargets(
            immediate_horizon=self.config.immediate_horizon,
            shortterm_horizon=self.config.shortterm_horizon,
            longterm_horizon=self.config.longterm_horizon,
        )

        logger.info("Prediction module loaded and frozen")

    def _setup_surprise_module(self):
        """Initialize surprise module (no training needed)."""
        from pem import SurpriseModule, SurpriseConfig

        surprise_config = SurpriseConfig(
            d_model=self.config.feature_dim,
            hidden_dim=self.config.feature_dim // 2,
        )
        self.surprise_module = SurpriseModule(surprise_config).to(self.device)
        self.surprise_module.eval()

        # Freeze surprise module
        for param in self.surprise_module.parameters():
            param.requires_grad = False

        logger.info("Surprise module initialized")

    def _setup_curiosity_module(self):
        """Initialize curiosity module (trainable)."""
        from pem import CuriosityModule, CuriosityConfig
        from pem.curiosity_module import CuriosityLoss

        curiosity_config = CuriosityConfig(
            d_model=self.config.feature_dim,
            hidden_dim=self.config.feature_dim // 2,
            use_temporal_novelty=True,
            novelty_memory_size=256,
        )
        self.curiosity_module = CuriosityModule(curiosity_config).to(self.device)

        self.curiosity_loss = CuriosityLoss(
            calibration_weight=self.config.calibration_weight,
            exploration_weight=self.config.exploration_weight,
            novelty_weight=self.config.novelty_weight,
        )

        num_params = sum(p.numel() for p in self.curiosity_module.parameters())
        trainable = sum(p.numel() for p in self.curiosity_module.parameters() if p.requires_grad)
        logger.info(f"Curiosity module: {num_params:,} params ({trainable:,} trainable)")

    def _setup_optimizer(self):
        """Setup optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.curiosity_module.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def extract_features_from_text(self, texts: List[str]) -> torch.Tensor:
        """Extract features from text."""
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
            features = self.feature_extractor(input_ids)
            features = features.float()

        return features

    def train_step(self, batch: List[str]) -> Dict[str, float]:
        """Single training step."""
        B = len(batch)

        # 1. Extract features (frozen)
        features = self.extract_features_from_text(batch)
        _, S, _ = features.shape

        if S < self.config.context_size + self.config.longterm_horizon:
            return {"loss": 0.0}

        context_features = features[:, :self.config.context_size]

        # 2. Get predictions (frozen prediction module)
        sync = torch.randn(
            B, self.config.context_size, self.config.sync_dim,
            device=self.device, dtype=torch.float32
        )

        with torch.no_grad():
            predictions = self.prediction_module(sync, context_features)

            # 3. Compute targets and surprise
            targets_full = self.target_computer.compute_targets(features)
            targets = {
                'immediate': targets_full['immediate'][:, :self.config.context_size],
                'shortterm': targets_full['shortterm'][:, :self.config.context_size],
                'longterm': targets_full['longterm'][:, :self.config.context_size],
                'immediate_valid': targets_full['immediate_valid'][:, :self.config.context_size],
                'shortterm_valid': targets_full['shortterm_valid'][:, :self.config.context_size],
                'longterm_valid': targets_full['longterm_valid'][:, :self.config.context_size],
            }

            # Compute surprise
            surprises = self.surprise_module(predictions, targets, context_features)

            # Get surprise magnitude (average across scales)
            surprise_magnitude = (
                surprises['immediate']['magnitude'] +
                surprises['shortterm']['magnitude'] +
                surprises['longterm']['magnitude']
            ) / 3.0

        # 4. Forward pass through curiosity module (trainable)
        if self.config.mixed_precision:
            with torch.amp.autocast('cuda'):
                curiosity_output = self.curiosity_module(
                    features=context_features,
                    predictions=predictions['immediate'],  # Use immediate predictions
                    context=context_features,
                    update_novelty=True,
                )
                loss, loss_dict = self.curiosity_loss(
                    curiosity_output=curiosity_output,
                    surprise_magnitude=surprise_magnitude,
                )
        else:
            curiosity_output = self.curiosity_module(
                features=context_features,
                predictions=predictions['immediate'],
                context=context_features,
                update_novelty=True,
            )
            loss, loss_dict = self.curiosity_loss(
                curiosity_output=curiosity_output,
                surprise_magnitude=surprise_magnitude,
            )

        # Scale for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            **{k: v.item() for k, v in loss_dict.items()}
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.curiosity_module.train()

        epoch_losses = []
        accumulated_loss = {}
        accumulation_count = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            loss_dict = self.train_step(batch)

            for k, v in loss_dict.items():
                accumulated_loss[k] = accumulated_loss.get(k, 0) + v
            accumulation_count += 1

            # Gradient step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.curiosity_module.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.curiosity_module.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                avg_loss = {k: v / accumulation_count for k, v in accumulated_loss.items()}
                epoch_losses.append(avg_loss)

                if self.global_step % self.config.log_every_n_steps == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Epoch {epoch} | Step {self.global_step} | "
                        f"Loss: {avg_loss['loss']:.4f} | "
                        f"Calibration: {avg_loss.get('calibration', 0):.4f} | "
                        f"Exploration: {avg_loss.get('exploration', 0):.4f} | "
                        f"Novelty: {avg_loss.get('novelty_correlation', 0):.4f} | "
                        f"Time: {elapsed:.1f}s"
                    )

                if self.global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

                accumulated_loss = {}
                accumulation_count = 0

        if epoch_losses:
            avg_epoch_loss = {
                k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
                for k in epoch_losses[0].keys()
            }
        else:
            avg_epoch_loss = {"loss": float('inf')}

        return avg_epoch_loss

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")

        torch.save({
            "global_step": self.global_step,
            "model_state_dict": self.curiosity_module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "best_loss": self.best_loss,
        }, checkpoint_path)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.curiosity_module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float('inf'))

    def train(self, file_paths: List[Path]):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("Starting Curiosity Module Training")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Files: {len(file_paths)}")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Prediction checkpoint: {self.config.prediction_checkpoint}")
        logger.info("=" * 60)

        dataset = BookDataset(
            file_paths=file_paths,
            tokenizer=self.tokenizer,
            context_size=self.config.context_size,
            longterm_horizon=self.config.longterm_horizon,
            shuffle=True,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=lambda x: x,
            num_workers=4,
            pin_memory=True,
        )

        for epoch in range(1, self.config.num_epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{self.config.num_epochs}")
            logger.info("=" * 60)

            # Reset novelty memory each epoch
            self.curiosity_module.reset_novelty()

            epoch_start = time.time()
            avg_loss = self.train_epoch(dataloader, epoch)
            epoch_time = time.time() - epoch_start

            logger.info(f"\nEpoch {epoch} complete in {epoch_time:.1f}s")
            logger.info(f"Average loss: {avg_loss['loss']:.4f}")

            if avg_loss['loss'] < self.best_loss:
                self.best_loss = avg_loss['loss']
                self.save_checkpoint("best")

            self.save_checkpoint(f"epoch_{epoch}")

        self.save_checkpoint("final")
        logger.info("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Train PEM Curiosity Module")

    parser.add_argument("--data_dir", type=str,
                       default="/media/alexander/Tank1/text_files/text_files/")
    parser.add_argument("--num_books", type=int, default=1000)
    parser.add_argument("--prediction_checkpoint", type=str,
                       default="checkpoints/prediction/best.pt",
                       help="Path to trained prediction module checkpoint")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str,
                       default="checkpoints/curiosity")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    config = CuriosityTrainingConfig(
        data_dir=args.data_dir,
        num_books=args.num_books,
        prediction_checkpoint=args.prediction_checkpoint,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    file_paths = find_text_files(config.data_dir, config.num_books)

    if not file_paths:
        logger.error("No text files found!")
        return

    trainer = CuriosityTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(file_paths)


if __name__ == "__main__":
    main()
