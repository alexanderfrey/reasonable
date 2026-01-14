#!/usr/bin/env python3
"""
Train the full PEM Experience Loop.

This trains the complete predictive processing loop:
    CTMPrediction → SurpriseModule → PerceptionAttention → (loop)

Usage:
    python -m pem.train_pem_loop --help
    python -m pem.train_pem_loop --device cuda --batch_size 4
    python -m pem.train_pem_loop --wandb_project pem-loop
"""

import argparse
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PEMLoopTrainingConfig:
    """Configuration for PEM loop training."""

    # Model configuration
    d_model: int = 1536
    d_neurons: int = 512
    d_sync_out: int = 256
    d_sync_action: int = 256
    M: int = 16
    T: int = 8
    synapse_hidden: int = 1024
    nlm_hidden: int = 64
    n_attention_heads: int = 8

    # Prediction horizons
    immediate_horizon: int = 8
    shortterm_horizon: int = 64
    longterm_horizon: int = 256

    # Loop configuration
    num_loop_steps: int = 3  # How many PEM loop iterations per sample
    surprise_scale: float = 1.0

    # Training configuration
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 10000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: bool = True

    # Data configuration
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    dataset_split: str = "train"
    context_size: int = 512
    min_seq_length: int = 128

    # Feature extractor
    feature_model: str = "deepseek-ai/Janus-Pro-1B"
    freeze_features: bool = True

    # Logging and checkpointing
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100
    save_every_n_steps: int = 500
    checkpoint_dir: str = "checkpoints/pem_loop"
    eval_max_batches: int = 50

    # Device
    device: str = "cuda"

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "pem-loop"
    wandb_run_name: Optional[str] = None
    wandb_log_every: int = 10


class PEMLoopTrainer:
    """Trainer for the full PEM experience loop."""

    def __init__(self, config: PEMLoopTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize wandb
        self.wandb_run = None
        if config.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=vars(config),
                )
                logger.info(f"Wandb initialized: {wandb.run.name}")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.wandb_run = None

        # Load feature extractor
        logger.info(f"Loading feature extractor: {config.feature_model}")
        from pem import create_feature_extractor
        self.feature_extractor = create_feature_extractor(
            model_name_or_path=config.feature_model,
            learning_mode="frozen" if config.freeze_features else "full",
            device_map=config.device,
        )
        self.feature_extractor.eval()

        # Get tokenizer from feature extractor
        self.tokenizer = self.feature_extractor.tokenizer

        # Create PEM loop
        logger.info("Creating PEM loop...")
        from pem.pem_loop import PEMLoop, PEMLoopConfig

        pem_config = PEMLoopConfig(
            d_model=config.d_model,
            d_perception=config.d_model,
            d_neurons=config.d_neurons,
            d_sync_out=config.d_sync_out,
            d_sync_action=config.d_sync_action,
            M=config.M,
            T=config.T,
            synapse_hidden=config.synapse_hidden,
            nlm_hidden=config.nlm_hidden,
            n_attention_heads=config.n_attention_heads,
            immediate_horizon=config.immediate_horizon,
            shortterm_horizon=config.shortterm_horizon,
            longterm_horizon=config.longterm_horizon,
            surprise_scale=config.surprise_scale,
        )
        self.pem_loop = PEMLoop(pem_config).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.pem_loop.parameters())
        trainable_params = sum(p.numel() for p in self.pem_loop.parameters() if p.requires_grad)
        logger.info(f"PEM Loop: {total_params:,} params ({trainable_params:,} trainable)")

        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.pem_loop.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.1,
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')

    def _unpack_batch(self, batch) -> Tuple[list, float]:
        """Unpack batch from dataloader."""
        if isinstance(batch, dict):
            texts = batch.get("text", batch.get("content", []))
            progress = batch.get("progress", 0.0)
        elif isinstance(batch, (list, tuple)):
            texts = batch[0] if len(batch) > 0 else []
            progress = batch[1] if len(batch) > 1 else 0.0
        else:
            texts = [batch]
            progress = 0.0

        if isinstance(texts, str):
            texts = [texts]

        return texts, progress

    def _prepare_batch(
        self,
        texts: list,
    ) -> Tuple[Optional[torch.Tensor], int]:
        """
        Prepare batch: tokenize and extract features.

        Returns:
            features: (B, S, D) or None if too short
            seq_len: actual sequence length
        """
        # Tokenize
        inputs = self.feature_extractor.process_inputs(texts=texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        seq_len = inputs["input_ids"].shape[1]
        if seq_len < self.config.min_seq_length:
            return None, seq_len

        # Truncate to context size
        if seq_len > self.config.context_size:
            inputs = {k: v[:, :self.config.context_size] for k, v in inputs.items()}
            seq_len = self.config.context_size

        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(**inputs)

        # Convert to float32
        features = features.float()

        return features, seq_len

    def train_step(self, batch) -> dict:
        """Single training step."""
        texts, progress = self._unpack_batch(batch)
        features, seq_len = self._prepare_batch(texts)

        if features is None:
            return {
                "loss": 0.0,
                "skipped": 1,
                "batch_tokens": 0,
                "seq_len": seq_len,
                "progress": progress,
            }

        autocast_ctx = torch.amp.autocast('cuda') if self.config.mixed_precision else nullcontext()

        with autocast_ctx:
            # Run PEM loop
            outputs, final_state = self.pem_loop(
                features=features,
                num_steps=self.config.num_loop_steps,
                return_all_ticks=True,
            )

            # Compute targets for loss
            targets = self.pem_loop.target_computer.compute_targets_efficient(features)

            # Compute loss
            loss, loss_dict = self.pem_loop.compute_loss(outputs, targets)

        # Scale for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Extract scalar values
        loss_values = {}
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                loss_values[k] = v.item()
            else:
                loss_values[k] = v

        # Get surprise stats from first and last step
        first_surprise = outputs[0].surprises.get('immediate', {}).get('magnitude', None)
        last_surprise = outputs[-1].surprises.get('immediate', {}).get('magnitude', None)

        if first_surprise is not None:
            loss_values['surprise_first'] = first_surprise.mean().item()
        if last_surprise is not None:
            loss_values['surprise_last'] = last_surprise.mean().item()

        # Cumulative surprise from final state
        loss_values['cumulative_surprise'] = final_state.cumulative_surprise.mean().item()

        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            **loss_values,
            "skipped": 0,
            "batch_tokens": len(texts) * seq_len,
            "seq_len": seq_len,
            "progress": progress,
        }

    def eval_step(self, batch) -> dict:
        """Single evaluation step."""
        texts, progress = self._unpack_batch(batch)
        features, seq_len = self._prepare_batch(texts)

        if features is None:
            return {
                "loss": 0.0,
                "skipped": 1,
                "batch_tokens": 0,
                "seq_len": seq_len,
                "progress": progress,
            }

        autocast_ctx = torch.amp.autocast('cuda') if self.config.mixed_precision else nullcontext()

        with torch.no_grad():
            with autocast_ctx:
                outputs, final_state = self.pem_loop(
                    features=features,
                    num_steps=self.config.num_loop_steps,
                    return_all_ticks=True,
                )

                targets = self.pem_loop.target_computer.compute_targets_efficient(features)
                loss, loss_dict = self.pem_loop.compute_loss(outputs, targets)

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
            "batch_tokens": len(texts) * seq_len,
            "seq_len": seq_len,
            "progress": progress,
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        val_loader: Optional[DataLoader] = None,
    ) -> dict:
        """Train for one epoch."""
        self.pem_loop.train()

        accumulated_loss = {}
        accumulation_count = 0
        start_time = time.time()
        total_tokens = 0
        skipped = 0

        for batch_idx, batch in enumerate(dataloader):
            if self.global_step >= self.config.max_steps:
                break

            loss_dict = self.train_step(batch)
            skipped += loss_dict.pop("skipped", 0)
            batch_tokens = loss_dict.pop("batch_tokens", 0)
            seq_len = loss_dict.pop("seq_len", 0)
            progress = loss_dict.pop("progress", 0.0)
            total_tokens += batch_tokens

            for k, v in loss_dict.items():
                accumulated_loss[k] = accumulated_loss.get(k, 0) + v
            accumulation_count += 1

            # Gradient step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.max_grad_norm > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.pem_loop.parameters(),
                        self.config.max_grad_norm,
                    )
                else:
                    grad_norm = 0.0

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    avg_loss = {k: v / accumulation_count for k, v in accumulated_loss.items()}
                    elapsed = time.time() - start_time
                    throughput = total_tokens / elapsed if elapsed > 0 else 0

                    # Key metrics
                    loss_val = avg_loss.get('loss', avg_loss.get('step0_loss', 0))
                    surprise_first = avg_loss.get('surprise_first', 0)
                    surprise_last = avg_loss.get('surprise_last', 0)
                    cumulative = avg_loss.get('cumulative_surprise', 0)

                    logger.info(
                        f"Epoch {epoch} | Step {self.global_step} | "
                        f"Loss: {loss_val:.4f} | "
                        f"Surprise: {surprise_first:.3f}→{surprise_last:.3f} | "
                        f"Cumulative: {cumulative:.3f} | "
                        f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                        f"GradNorm: {grad_norm:.2f} | "
                        f"Throughput: {throughput:.0f} tok/s"
                    )

                    # Wandb logging
                    if self.wandb_run and self.global_step % self.config.wandb_log_every == 0:
                        import wandb
                        wandb.log({
                            "train/loss": loss_val,
                            "train/surprise_first": surprise_first,
                            "train/surprise_last": surprise_last,
                            "train/cumulative_surprise": cumulative,
                            "train/grad_norm": grad_norm,
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/throughput": throughput,
                            "step": self.global_step,
                        })

                    # Reset accumulators
                    accumulated_loss = {}
                    accumulation_count = 0
                    start_time = time.time()
                    total_tokens = 0

                # Evaluation
                if val_loader is not None and self.global_step % self.config.eval_every_n_steps == 0:
                    eval_loss = self.evaluate(val_loader)
                    logger.info(f"Eval @ step {self.global_step} | Loss: {eval_loss:.4f}")

                    if self.wandb_run:
                        import wandb
                        wandb.log({
                            "eval/loss": eval_loss,
                            "step": self.global_step,
                        })

                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint("best")

                # Checkpointing
                if self.global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

        return accumulated_loss

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate on validation set."""
        self.pem_loop.eval()
        total_loss = 0.0
        count = 0

        for batch_idx, batch in enumerate(dataloader):
            if self.config.eval_max_batches and batch_idx >= self.config.eval_max_batches:
                break

            loss_dict = self.eval_step(batch)
            if loss_dict.get("skipped", 0) == 0:
                total_loss += loss_dict.get("loss", 0)
                count += 1

        self.pem_loop.train()
        return total_loss / max(count, 1)

    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")

        torch.save({
            "global_step": self.global_step,
            "model_state_dict": self.pem_loop.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "best_loss": self.best_loss,
        }, path)

        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.pem_loop.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float('inf'))

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop."""
        logger.info("Starting PEM loop training...")
        logger.info(f"Config: {self.config}")

        epoch = 0
        while self.global_step < self.config.max_steps:
            epoch += 1
            logger.info(f"Starting epoch {epoch}")
            self.train_epoch(train_loader, epoch, val_loader)

        logger.info("Training complete!")
        self.save_checkpoint("final")


def create_dataloader(config: PEMLoopTrainingConfig, split: str = "train") -> DataLoader:
    """Create dataloader from HuggingFace dataset."""
    logger.info(f"Loading dataset: {config.dataset_name}/{config.dataset_config}")

    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.dataset_split,
        streaming=True,
    )

    # For validation, skip the first portion and take a small subset
    if split == "val":
        dataset = dataset.skip(10000).take(1000)

    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        return {"text": texts}

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # Streaming doesn't support workers
    )


def main():
    parser = argparse.ArgumentParser(description="Train PEM Experience Loop")

    # Model args
    parser.add_argument("--d_model", type=int, default=1536)
    parser.add_argument("--d_neurons", type=int, default=512)
    parser.add_argument("--T", type=int, default=8, help="CTM internal ticks")
    parser.add_argument("--num_loop_steps", type=int, default=3, help="PEM loop iterations")

    # Training args
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--context_size", type=int, default=512)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    # Logging
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--eval_every_n_steps", type=int, default=100)
    parser.add_argument("--save_every_n_steps", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/pem_loop")

    # Wandb
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="pem-loop")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")

    args = parser.parse_args()

    # Create config
    config = PEMLoopTrainingConfig(
        d_model=args.d_model,
        d_neurons=args.d_neurons,
        T=args.T,
        num_loop_steps=args.num_loop_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        context_size=args.context_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device=args.device,
        log_every_n_steps=args.log_every_n_steps,
        eval_every_n_steps=args.eval_every_n_steps,
        save_every_n_steps=args.save_every_n_steps,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb and not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    # Create trainer
    trainer = PEMLoopTrainer(config)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Create dataloaders
    train_loader = create_dataloader(config, "train")
    val_loader = create_dataloader(config, "val")

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
