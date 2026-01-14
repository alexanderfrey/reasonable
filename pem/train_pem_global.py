"""
Training script for PEM Loop with Global Sync Architecture.

Trains PredictionCTM + SurpriseCTM + GlobalSyncModule end-to-end.

Usage:
    python -m pem.train_pem_global --wandb_project pem-global --batch_size 4

Monitors:
    - Loss breakdown (prediction, surprise calibration, sync variance)
    - Cross-module synchronization patterns
    - Module contributions over time
    - Attention entropy
    - Per-tick outputs from CTM modules
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .pem_loop_global import PEMLoopGlobal, PEMLoopGlobalConfig, create_pem_loop_global
from .janus_pro_feature_extractor import JanusProFeatureExtractor, JanusProConfig


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    d_model: int = 1536
    pred_d_neurons: int = 256
    surp_d_neurons: int = 128
    pred_T: int = 4
    surp_T: int = 3
    sync_pairs: int = 256

    # Training
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Loop
    num_loop_steps: int = 2

    # Logging
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 1000

    # Data
    max_length: int = 512

    # WandB
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


def create_optimizer(model: nn.Module, config: TrainingConfig):
    """Create AdamW optimizer with weight decay."""
    # Separate parameters that should/shouldn't have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'embedding' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=config.learning_rate)

    return optimizer


def create_scheduler(optimizer, config: TrainingConfig):
    """Create learning rate scheduler with warmup."""
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        # Cosine decay
        progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_detailed_metrics(
    outputs: List,
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute detailed metrics for logging."""
    metrics = {}

    for step_idx, output in enumerate(outputs):
        prefix = f"step{step_idx}"

        # Prediction metrics
        for scale in ['immediate', 'shortterm', 'longterm']:
            pred = output.predictions[scale]
            target = targets[scale]
            valid = targets.get(f'{scale}_valid', None)

            if valid is not None and valid.any():
                pred_valid = pred[valid]
                target_valid = target[valid]
                cos_sim = F.cosine_similarity(pred_valid, target_valid, dim=-1).mean()
                metrics[f'{prefix}/{scale}_cos_sim'] = cos_sim.item()

        # Surprise metrics
        metrics[f'{prefix}/surprise_magnitude_mean'] = output.surprise.magnitude.mean().item()
        metrics[f'{prefix}/surprise_magnitude_std'] = output.surprise.magnitude.std().item()
        metrics[f'{prefix}/surprise_raw_mean'] = output.surprise.raw.mean().item()
        metrics[f'{prefix}/surprise_certainty'] = output.surprise.certainty.item()

        # Global sync metrics
        sync = output.global_sync
        metrics[f'{prefix}/global_sync_mean'] = sync.sync.mean().item()
        metrics[f'{prefix}/global_sync_std'] = sync.sync.std().item()

        # Cross-module sync (2x2 matrix for pred-surp)
        cross_sync = sync.cross_module_sync  # (2, 2, B, S)
        metrics[f'{prefix}/cross_sync_pred_pred'] = cross_sync[0, 0].mean().item()
        metrics[f'{prefix}/cross_sync_pred_surp'] = cross_sync[0, 1].mean().item()
        metrics[f'{prefix}/cross_sync_surp_pred'] = cross_sync[1, 0].mean().item()
        metrics[f'{prefix}/cross_sync_surp_surp'] = cross_sync[1, 1].mean().item()

        # Module contributions
        contrib = sync.module_contributions  # (B, S, 2)
        metrics[f'{prefix}/contrib_prediction'] = contrib[..., 0].mean().item()
        metrics[f'{prefix}/contrib_surprise'] = contrib[..., 1].mean().item()

        # Attention metrics
        attn = output.attention_weights  # (B, H, S, S)
        attn_entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1).mean()
        metrics[f'{prefix}/attention_entropy'] = attn_entropy.item()

        # Attention focus (how peaked is attention?)
        attn_max = attn.max(dim=-1).values.mean()
        metrics[f'{prefix}/attention_max'] = attn_max.item()

    # Cross-step metrics
    if len(outputs) > 1:
        # Surprise change across steps
        first_surp = outputs[0].surprise.magnitude.mean()
        last_surp = outputs[-1].surprise.magnitude.mean()
        metrics['surprise_change'] = (last_surp - first_surp).item()

        # Attention entropy change
        first_entropy = -(outputs[0].attention_weights * (outputs[0].attention_weights + 1e-8).log()).sum(dim=-1).mean()
        last_entropy = -(outputs[-1].attention_weights * (outputs[-1].attention_weights + 1e-8).log()).sum(dim=-1).mean()
        metrics['attention_entropy_change'] = (last_entropy - first_entropy).item()

    return metrics


def train_step(
    model: PEMLoopGlobal,
    features: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> Dict[str, float]:
    """Execute one training step."""
    model.train()
    optimizer.zero_grad()

    # Forward pass
    targets = model.target_computer.compute_targets_efficient(features)
    outputs, final_state = model(features, targets, num_steps=config.num_loop_steps)

    # Compute loss
    loss, loss_dict = model.compute_loss(outputs, targets)

    # Backward pass
    loss.backward()

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    # Optimizer step
    optimizer.step()

    # Compute detailed metrics
    with torch.no_grad():
        metrics = compute_detailed_metrics(outputs, targets)

    # Add loss and grad norm
    metrics['loss'] = loss.item()
    metrics['grad_norm'] = grad_norm.item()

    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            metrics[f'loss/{k}'] = v.item()

    # Cumulative sync stats
    metrics['cumulative_sync_mean'] = final_state.cumulative_sync.mean().item()
    metrics['cumulative_sync_std'] = final_state.cumulative_sync.std().item()

    return metrics


@torch.no_grad()
def eval_step(
    model: PEMLoopGlobal,
    features: torch.Tensor,
    config: TrainingConfig,
) -> Dict[str, float]:
    """Execute one evaluation step."""
    model.eval()

    targets = model.target_computer.compute_targets_efficient(features)
    outputs, final_state = model(features, targets, num_steps=config.num_loop_steps)

    loss, loss_dict = model.compute_loss(outputs, targets)
    metrics = compute_detailed_metrics(outputs, targets)

    metrics['eval/loss'] = loss.item()
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            metrics[f'eval/loss/{k}'] = v.item()

    return metrics


def create_dataloader(config: TrainingConfig, split: str = 'train'):
    """Create dataloader from HuggingFace dataset."""
    from datasets import load_dataset

    # Load FineWeb-Edu sample
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
    )

    if split == 'val':
        dataset = dataset.skip(10000).take(1000)

    def collate_fn(batch):
        texts = [item['text'][:config.max_length * 4] for item in batch]  # Rough char estimate
        return texts

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )

    return loader


def main():
    parser = argparse.ArgumentParser(description='Train PEM Loop with Global Sync')

    # Model args
    parser.add_argument('--d_model', type=int, default=1536)
    parser.add_argument('--pred_d_neurons', type=int, default=256)
    parser.add_argument('--surp_d_neurons', type=int, default=128)
    parser.add_argument('--pred_T', type=int, default=4)
    parser.add_argument('--surp_T', type=int, default=3)
    parser.add_argument('--sync_pairs', type=int, default=256)

    # Training args
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_loop_steps', type=int, default=2)

    # Logging args
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        d_model=args.d_model,
        pred_d_neurons=args.pred_d_neurons,
        surp_d_neurons=args.surp_d_neurons,
        pred_T=args.pred_T,
        surp_T=args.surp_T,
        sync_pairs=args.sync_pairs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        num_loop_steps=args.num_loop_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize wandb
    if config.wandb_project:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )

    # Create feature extractor (frozen)
    print("Loading Janus Pro feature extractor...")
    from .janus_pro_feature_extractor import LearningMode
    janus_config = JanusProConfig(
        model_name_or_path="deepseek-ai/Janus-Pro-1B",
        output_dim=config.d_model,
        learning_mode=LearningMode.FROZEN,
    )
    feature_extractor = JanusProFeatureExtractor(janus_config)
    feature_extractor.eval()
    print(f"Feature extractor loaded: {sum(p.numel() for p in feature_extractor.parameters()):,} params")

    # Create PEM loop with global sync
    print("Creating PEM Loop with Global Sync...")
    pem_config = PEMLoopGlobalConfig(
        d_model=config.d_model,
        pred_d_neurons=config.pred_d_neurons,
        surp_d_neurons=config.surp_d_neurons,
        pred_T=config.pred_T,
        surp_T=config.surp_T,
        sync_pairs=config.sync_pairs,
    )
    model = PEMLoopGlobal(pem_config).to(device)
    print(f"PEM Loop created: {sum(p.numel() for p in model.parameters()):,} params")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Create dataloader
    print("Creating dataloader...")
    train_loader = create_dataloader(config, split='train')
    train_iter = iter(train_loader)

    # Training loop
    print(f"\nStarting training for {config.max_steps} steps...")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Loop steps: {config.num_loop_steps}")
    print(f"  Pred neurons: {config.pred_d_neurons}, T={config.pred_T}")
    print(f"  Surp neurons: {config.surp_d_neurons}, T={config.surp_T}")
    print(f"  Sync pairs: {config.sync_pairs}")
    print()

    global_step = 0
    running_loss = 0.0
    start_time = time.time()

    while global_step < config.max_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Tokenize and extract features
        with torch.no_grad():
            # Tokenize
            tokenized = feature_extractor.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length,
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            # Extract features (cast to float32 for PEM model)
            features = feature_extractor(input_ids, attention_mask=attention_mask)
            features = features.float()  # Cast from bfloat16 to float32

        # Skip if too short
        if features.shape[1] < 32:
            continue

        # Training step
        metrics = train_step(model, features, optimizer, config)
        scheduler.step()

        running_loss += metrics['loss']
        global_step += 1

        # Logging
        if global_step % config.log_every == 0:
            avg_loss = running_loss / config.log_every
            elapsed = time.time() - start_time
            steps_per_sec = global_step / elapsed

            # Console output
            print(f"Step {global_step:5d} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Surp: {metrics['step0/surprise_magnitude_mean']:.3f}→{metrics.get('step1/surprise_magnitude_mean', metrics['step0/surprise_magnitude_mean']):.3f} | "
                  f"CrossSync: {metrics['step0/cross_sync_pred_surp']:.3f} | "
                  f"AttnEnt: {metrics['step0/attention_entropy']:.2f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"{steps_per_sec:.2f} steps/s")

            # WandB logging
            if config.wandb_project:
                log_dict = {
                    'train/loss': avg_loss,
                    'train/learning_rate': scheduler.get_last_lr()[0],
                    'train/steps_per_sec': steps_per_sec,
                }
                log_dict.update({f'train/{k}': v for k, v in metrics.items()})
                wandb.log(log_dict, step=global_step)

            running_loss = 0.0

        # Evaluation
        if global_step % config.eval_every == 0:
            # Use same batch for eval (simpler)
            eval_metrics = eval_step(model, features, config)

            print(f"  [Eval] Loss: {eval_metrics['eval/loss']:.4f}")

            if config.wandb_project:
                wandb.log(eval_metrics, step=global_step)

    print("\nTraining complete!")

    if config.wandb_project:
        wandb.finish()


if __name__ == '__main__':
    main()
