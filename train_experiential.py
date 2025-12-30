"""
Training script for Experiential Stream v0.1

Validates the core idea: can we predict "what comes next" in latent space?

Usage:
    # Quick sanity check (random data)
    python train_experiential.py --mode sanity

    # Train on pretrained model's representations
    python train_experiential.py --mode train --checkpoint path/to/model.pt

    # Use existing tokenized data
    python train_experiential.py --mode train --checkpoint path/to/model.pt --data_dir path/to/data
"""

import argparse
import logging
import math
import os
import sys
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from experiential import (
    ExperientialStreamV01,
    experiential_loss,
    prediction_accuracy,
    compute_metrics
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HiddenStateExtractor(nn.Module):
    """
    Wraps a GPT model to extract hidden states before lm_head.

    This avoids modifying the original model.py.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._hidden_states = None

        # Register hook on final_norm to capture hidden states
        self._hook = self.model.final_norm.register_forward_hook(self._capture_hidden)

    def _capture_hidden(self, module, input, output):
        """Hook to capture hidden states after final_norm."""
        self._hidden_states = output

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns both logits and hidden states.

        Returns:
            logits: [batch, seq_len, vocab_size]
            hidden_states: [batch, seq_len, d_model]
        """
        logits, _ = self.model(input_ids)
        hidden_states = self._hidden_states
        self._hidden_states = None  # clear for next call
        return logits, hidden_states

    def remove_hook(self):
        """Clean up hook when done."""
        self._hook.remove()


class RandomDataLoader:
    """
    Generate random token sequences for sanity checking.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        batch_size: int,
        n_batches: int,
        device: torch.device
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.device = device

    def __iter__(self):
        for _ in range(self.n_batches):
            input_ids = torch.randint(
                0, self.vocab_size,
                (self.batch_size, self.seq_len),
                device=self.device
            )
            yield {"input_ids": input_ids}

    def __len__(self):
        return self.n_batches


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load a pretrained GPT model."""
    from model import GPT, GPTConfig

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract config from checkpoint
    saved_config = checkpoint.get('config', {})
    args = checkpoint.get('args', {})
    if isinstance(args, dict):
        # Merge args into config (args may have more complete info)
        for key in ['vocab_size', 'd_model', 'n_head', 'n_layer', 'max_seq_len', 'n_kv_head']:
            if key in args and key not in saved_config:
                saved_config[key] = args[key]

    # Create config
    config = GPTConfig(
        vocab_size=saved_config.get('vocab_size', 50257),
        d_model=saved_config.get('d_model', 768),
        n_head=saved_config.get('n_head', 12),
        n_layer=saved_config.get('n_layer', 12),
        max_seq_len=saved_config.get('max_seq_len', 1024),
        n_kv_head=saved_config.get('n_kv_head'),
        dropout=saved_config.get('dropout', 0.0),
    )

    logger.info(f"Model config: d_model={config.d_model}, n_layer={config.n_layer}")

    # Create and load model
    model = GPT(config)
    state_dict = checkpoint['model_state_dict']

    # Strip prefixes if present
    def strip_prefix(name):
        for prefix in ['module.', '_orig_mod.']:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name

    state_dict = {strip_prefix(k): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, config


def create_small_model(device: torch.device) -> Tuple[nn.Module, 'GPTConfig']:
    """Create a small model for quick testing without a checkpoint."""
    from model import GPT, GPTConfig

    config = GPTConfig(
        vocab_size=1000,
        d_model=256,
        n_head=4,
        n_layer=4,
        max_seq_len=512,
        n_kv_head=2,
        dropout=0.0,
    )

    model = GPT(config).to(device)
    model.eval()

    logger.info(f"Created small test model: d_model={config.d_model}, n_layer={config.n_layer}")
    return model, config


def train_experiential(
    extractor: HiddenStateExtractor,
    experiential: ExperientialStreamV01,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int = 1,
    log_interval: int = 10,
    freeze_backbone: bool = True
) -> dict:
    """
    Train the experiential module on hidden states from the backbone.

    Args:
        extractor: Wrapper that extracts hidden states from GPT
        experiential: The experiential stream module to train
        dataloader: Data loader yielding {"input_ids": tensor}
        optimizer: Optimizer for experiential module
        device: Device to train on
        n_epochs: Number of epochs
        log_interval: Steps between logging
        freeze_backbone: If True, don't update backbone (recommended for v0.1)

    Returns:
        dict with training history
    """
    if freeze_backbone:
        extractor.model.eval()
        for param in extractor.model.parameters():
            param.requires_grad = False

    experiential.train()

    history = {
        'loss': [],
        'accuracy': [],
        'surprise_mean': [],
        'surprise_std': [],
    }

    total_steps = 0
    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=True)

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)

            # Get hidden states from frozen backbone
            with torch.no_grad() if freeze_backbone else torch.enable_grad():
                _, hidden_states = extractor(input_ids)

            # Forward through experiential module
            exp_output = experiential(hidden_states)

            # Compute loss
            loss = experiential_loss(exp_output['prediction'], exp_output['target'])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(experiential.parameters(), max_norm=1.0)
            optimizer.step()

            # Metrics
            acc = prediction_accuracy(exp_output['prediction'], exp_output['target'])
            surprise_mean = exp_output['surprise'].mean().item()
            surprise_std = exp_output['surprise'].std().item()

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_steps += 1
            total_steps += 1

            # Log
            if batch_idx % log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc:.4f}',
                    'surp': f'{surprise_mean:.3f}'
                })

            history['loss'].append(loss.item())
            history['accuracy'].append(acc)
            history['surprise_mean'].append(surprise_mean)
            history['surprise_std'].append(surprise_std)

        # Epoch summary
        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_acc = epoch_acc / max(epoch_steps, 1)
        logger.info(
            f"Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}, avg_acc={avg_acc:.4f}"
        )

    elapsed = time.time() - start_time
    logger.info(f"Training complete: {total_steps} steps in {elapsed:.1f}s")

    return history


def run_sanity_check(device: torch.device, n_steps: int = 100):
    """
    Quick sanity check with random data and small model.
    """
    logger.info("=" * 60)
    logger.info("Running sanity check...")
    logger.info("=" * 60)

    # Create small model
    model, config = create_small_model(device)
    extractor = HiddenStateExtractor(model)

    # Create experiential module
    experiential = ExperientialStreamV01(
        d_model=config.d_model,
        n_heads=4
    ).to(device)

    # Random data
    dataloader = RandomDataLoader(
        vocab_size=config.vocab_size,
        seq_len=128,
        batch_size=16,
        n_batches=n_steps,
        device=device
    )

    # Optimizer
    optimizer = torch.optim.Adam(experiential.parameters(), lr=1e-3)

    # Train
    history = train_experiential(
        extractor=extractor,
        experiential=experiential,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        n_epochs=1,
        log_interval=20,
        freeze_backbone=True
    )

    # Analyze results
    initial_acc = history['accuracy'][0]
    final_acc = history['accuracy'][-1]
    initial_loss = history['loss'][0]
    final_loss = history['loss'][-1]

    logger.info("")
    logger.info("=" * 60)
    logger.info("Sanity Check Results:")
    logger.info(f"  Initial: loss={initial_loss:.4f}, accuracy={initial_acc:.4f}")
    logger.info(f"  Final:   loss={final_loss:.4f}, accuracy={final_acc:.4f}")
    logger.info(f"  Random baseline accuracy: {1/16:.4f} (batch_size=16)")
    logger.info("")

    if final_loss < initial_loss:
        logger.info("  ✓ Loss decreased — learning is happening!")
    else:
        logger.warning("  ✗ Loss did not decrease — check implementation")

    if final_acc > initial_acc:
        logger.info("  ✓ Accuracy improved!")
    else:
        logger.warning("  ✗ Accuracy did not improve — might need more steps")

    if final_acc > 0.2:  # significantly above random (1/16 ≈ 0.06)
        logger.info("  ✓ Accuracy significantly above random!")
    else:
        logger.info("  ~ Accuracy close to random — expected for random data")

    logger.info("=" * 60)

    # Cleanup
    extractor.remove_hook()

    return history


def run_training(
    checkpoint_path: str,
    device: torch.device,
    data_dir: Optional[str] = None,
    n_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    max_steps: Optional[int] = None,
    output_dir: str = "experiential_output"
):
    """
    Train experiential module on pretrained model's representations.
    """
    logger.info("=" * 60)
    logger.info("Training Experiential Stream v0.1")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    if checkpoint_path and os.path.exists(checkpoint_path):
        model, config = load_model(checkpoint_path, device)
    else:
        logger.warning("No checkpoint provided, using small test model")
        model, config = create_small_model(device)

    extractor = HiddenStateExtractor(model)

    # Create experiential module
    experiential = ExperientialStreamV01(
        d_model=config.d_model,
        n_heads=4
    ).to(device)

    logger.info(f"Experiential module parameters: {sum(p.numel() for p in experiential.parameters()):,}")

    # Setup data
    if data_dir and os.path.exists(data_dir):
        # Try to load existing tokenized data
        from pretrain import PretokenizedDataset
        import glob
        import json

        # Find metadata file
        meta_files = glob.glob(os.path.join(data_dir, "*_metadata.json"))
        if meta_files:
            with open(meta_files[0]) as f:
                meta = json.load(f)

            token_file = meta.get('token_file') or meta_files[0].replace('_metadata.json', '_tokens.bin')
            num_examples = meta.get('num_examples', 1000)
            max_seq_len = meta.get('max_seq_len', 512)
            stride = meta.get('stride', max_seq_len)

            logger.info(f"Loading data from {token_file}")
            dataset = PretokenizedDataset(
                token_file, num_examples, max_seq_len, stride, data_type="Train"
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
            logger.info(f"Loaded {len(dataset)} examples")
        else:
            logger.warning("No metadata found, using random data")
            n_batches = max_steps or 500
            dataloader = RandomDataLoader(
                vocab_size=config.vocab_size,
                seq_len=min(512, config.max_seq_len),
                batch_size=batch_size,
                n_batches=n_batches,
                device=device
            )
    else:
        # Use random data
        logger.info("Using random data (no data_dir provided)")
        n_batches = max_steps or 500
        dataloader = RandomDataLoader(
            vocab_size=config.vocab_size,
            seq_len=min(512, config.max_seq_len),
            batch_size=batch_size,
            n_batches=n_batches,
            device=device
        )

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        experiential.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    # Train
    history = train_experiential(
        extractor=extractor,
        experiential=experiential,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        n_epochs=n_epochs,
        log_interval=10,
        freeze_backbone=True
    )

    # Save results
    final_acc = sum(history['accuracy'][-10:]) / 10 if len(history['accuracy']) >= 10 else history['accuracy'][-1]
    logger.info(f"Final accuracy (last 10): {final_acc:.4f}")

    # Save model
    save_path = os.path.join(output_dir, "experiential_v01.pt")
    torch.save({
        'model_state_dict': experiential.state_dict(),
        'd_model': config.d_model,
        'history': history,
    }, save_path)
    logger.info(f"Saved experiential module to {save_path}")

    # Cleanup
    extractor.remove_hook()

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Experiential Stream v0.1")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["sanity", "train"],
        default="sanity",
        help="Mode: 'sanity' for quick check, 'train' for full training"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing tokenized data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiential_output",
        help="Output directory for saving results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=3,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides epochs)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    args = parser.parse_args()
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    if args.mode == "sanity":
        run_sanity_check(device, n_steps=100)
    elif args.mode == "train":
        run_training(
            checkpoint_path=args.checkpoint,
            device=device,
            data_dir=args.data_dir,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_steps=args.max_steps,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
