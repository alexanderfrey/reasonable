import tiktoken
import argparse
import os, math
from glob import glob
import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR

from model import GPTConfig, GPT, HeadConfig, MultiHeadGPT
from strategies import (
    SpanMaskingStrategy,
    InstructionFollowingStrategy,
    NextTokenStrategy,
    MixedStrategy,
)
from trainer import MultiHeadTrainer, SingleHeadTrainer
from data_utils import create_dataloaders


def load_checkpoint(
    path, model, optimizer=None, scheduler=None, scaler=None, device="cuda"
):
    """Modified to handle pretrain -> finetune transition"""
    try:
        checkpoint = torch.load(path, map_location="cpu")

        # Load only model weights, skip optimizer state when transitioning
        if dist.is_initialized():
            model_to_load = model.module if hasattr(model, "module") else model
        else:
            model_to_load = model

        # Load model state dict
        model_to_load.load_state_dict(checkpoint["model_state_dict"])

        # Don't load optimizer state when transitioning between modes
        if optimizer is not None and "optimizer_state_dict" in checkpoint:

            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if device != "cpu":
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
            except ValueError as e:
                print(f"Warning: Could not load optimizer state: {e}")
                print("Starting with fresh optimizer state")

        # Set model to training mode
        model.train()

        return {
            "epoch": checkpoint["epoch"],
            "global_step": checkpoint["global_step"],
            "train_metrics": checkpoint.get("train_metrics", {}),
            "val_metrics": checkpoint.get("val_metrics", {}),
            "config": checkpoint.get("config", {}),
        }

    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        raise


def count_parameters(model):
    """
    Count the number of trainable parameters in the single-head GPT model.
    Returns total params and a breakdown of parameters by component.
    """
    # Get the unwrapped model if using DDP
    model = model.module if hasattr(model, "module") else model

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count embedding parameters
    token_emb_params = model.token_embedding.weight.numel()
    pos_emb_params = model.position_embedding.weight.numel()
    total_emb_params = token_emb_params + pos_emb_params

    # Count parameters per transformer block
    layer_params = sum(
        p.numel() for p in model.blocks[0].parameters() if p.requires_grad
    )
    total_layer_params = layer_params * len(model.blocks)

    # Count final layer norm and head parameters
    ln_params = sum(p.numel() for p in model.ln_f.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.lm_head.parameters() if p.requires_grad)

    # Add attention parameters
    attention_params = sum(
        sum(p.numel() for p in block.attn.parameters() if p.requires_grad)
        for block in model.blocks
    )

    return {
        "total_params_M": total_params / 1_000_000,
        "embeddings": {
            "total": total_emb_params,
            "token": token_emb_params,
            "position": pos_emb_params,
        },
        "transformer_layers": total_layer_params,
        "layer_norms": ln_params,
        "head": head_params,
        "attention": attention_params,
        "total": total_params,
    }


def setup_distributed():
    """Initialize distributed training"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        return None, None, None

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    return rank, world_size, local_rank


def get_vocab_size(encoding_name="gpt2"):
    tokenizer = tiktoken.get_encoding(encoding_name)
    return tokenizer.n_vocab


def get_args():
    parser = argparse.ArgumentParser()

    # Model architecture arguments
    parser.add_argument(
        "--n_layer", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--n_head", type=int, default=6, help="Number of attention heads"
    )
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument(
        "--block_size", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )

    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay coefficient"
    )
    parser.add_argument(
        "--main_loss_weight",
        type=float,
        default=0.5,
        help="Weight for main pathway loss (0-1). Analysis pathway weight will be 1 - this value",
    )

    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        default="./training_data/*.jsonl",
        help="Path pattern to JSON files containing input-thought pairs",
    )
    parser.add_argument(
        "--max_thought_length",
        type=int,
        default=256,
        help="Maximum length for thought/analysis text",
    )

    # Generation arguments
    parser.add_argument(
        "--gen_every_n_steps",
        type=int,
        default=100,
        help="Generate samples every N training steps",
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        nargs="+",
        default=["Once upon a time", "In a galaxy far far away"],
        help="Prompts to use for generation during training",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter for text generation",
    )
    parser.add_argument(
        "--max_gen_length",
        type=int,
        default=100,
        help="Maximum length for generated continuations",
    )

    # Optimization arguments
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per worker",
    )
    parser.add_argument(
        "--opt_level", type=str, default="O2", help="AMP optimization level"
    )

    # Checkpoint arguments
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps (optional)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load (optional)",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Enable pre-training mode where both pathways predict next tokens",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.main_loss_weight < 0 or args.main_loss_weight > 1:
        raise ValueError("main_loss_weight must be between 0 and 1")

    return args


def configure_optimizer(model, args, train_loader):
    num_training_steps = len(train_loader) * args.epochs

    # Adjust parameter groups
    no_decay = ["bias", "LayerNorm.weight", "ln_f.weight"]
    optimizer_grouped_params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.005,  # Reduced weight decay
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    # Increased warmup steps
    warmup_steps = min(1000, int(num_training_steps * 0.1))

    # Increased learning rate
    base_lr = 3e-4  # Increased from 1e-4

    optimizer = bnb.optim.AdamW(
        optimizer_grouped_params,
        lr=base_lr,
        betas=(0.95, 0.999),  # Increased beta1
        eps=1e-8,
        block_wise=True,
        is_paged=True,
        optim_bits=8,
    )

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)

        progress = float(current_step - warmup_steps) / float(
            max(1, num_training_steps - warmup_steps)
        )
        return max(
            0.01, 0.5 * (1.0 + math.cos(math.pi * progress))
        )  # Reduced minimum LR

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Add gradient clipping in training loop
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    return optimizer, scheduler


if __name__ == "__main__":
    args = get_args()
    torch.backends.cudnn.benchmark = True

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if local_rank is not None else "cuda")

    # Get vocab size and create tokenizer
    vocab_size = get_vocab_size("gpt2")
    tokenizer = tiktoken.get_encoding("gpt2")

    config = GPTConfig(
        vocab_size=vocab_size,  # Remove heads configuration
        n_layer=args.n_layer,  # Add your desired number of layers
        n_head=args.n_head,  # Add your desired number of heads
        n_embd=args.n_embd,  # Add your desired embedding dimension
        block_size=args.block_size,  # Add your desired block size
        dropout=0.1,  # Add your desired dropout rate
        bias=True,  # Add if you want bias in linear layers
    )

    # 2. Replace MultiHeadGPT with GPT
    model = GPT(config)  # Use the single head GPT class instead of MultiHeadGPT
    model = model.to(device)

    # Initialize strategies based on training mode and arguments
    if args.pretrain:
        # For pretraining, we use NextTokenStrategy
        try:
            has_mask_token = "<mask>" in tokenizer._special_tokens
        except AttributeError:
            has_mask_token = False
            print(
                "Warning: Could not check special tokens, using default mask token ID"
            )

        main_strategy = NextTokenStrategy(tokenizer)
        second_strategy = NextTokenStrategy(tokenizer)
    else:
        # For finetuning, use InstructionFollowingStrategy for both pathways
        main_strategy = NextTokenStrategy(tokenizer=tokenizer)
        second_strategy = InstructionFollowingStrategy(
            tokenizer=tokenizer,
            instruction_token="[INST]",
            response_token="[/INST]",
            end_token="</s>",
            max_length=args.block_size,
        )

    # Create trainer with strategies
    strategy = NextTokenStrategy(
        tokenizer
    )  # Or InstructionFollowingStrategy based on your needs
    trainer = SingleHeadTrainer(strategy=strategy)

    # Initialize wandb only on rank 0
    if rank == 0:
        wandb.init(
            project="multihead-gpt-training",
            config={
                "vocab_size": vocab_size,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "block_size": config.block_size,
                "dropout": config.dropout,
                "learning_rate": args.lr,
                "batch_size": args.batch_size * (world_size or 1),
                "optimizer": "8-bit AdamW",
                "num_gpus": world_size or 1,
                "training_mode": "pretrain" if args.pretrain else "finetune",
                "main_strategy": main_strategy.__class__.__name__,
                "instruction_token": "[INST]" if not args.pretrain else None,
                "response_token": "[/INST]" if not args.pretrain else None,
            },
        )

    # Print parameter counts (only on rank 0)
    if rank == 0:
        param_counts = count_parameters(model)
        print(f"Total parameters: {param_counts['total_params_M']:.2f}M")
        print("\nBreakdown by component:")
        print(f"- Embeddings: {param_counts['embeddings']['total']:,}")
        print(f"- Token embeddings: {param_counts['embeddings']['token']:,}")
        print(f"- Position embeddings: {param_counts['embeddings']['position']:,}")
        print(f"- Transformer layers: {param_counts['transformer_layers']:,}")
        print(f"- Layer norm: {param_counts['layer_norms']:,}")
        print(f"- Head: {param_counts['head']:,}")
        print(f"- Attention total: {param_counts['attention']:,}")

    # Wrap model with DDP
    if world_size is not None:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Create dataloaders with appropriate strategies
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
        files_pattern=args.train_data,
        block_size=args.block_size,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        strategy=strategy,
        tokenizer=tokenizer,
    )

    optimizer, scheduler = configure_optimizer(model, args, train_loader)

    scaler = GradScaler()

    # Training loop
    best_val_loss = float("inf")
    start_epoch = 0

    if args.load_checkpoint is not None:
        checkpoint_info = load_checkpoint(
            args.load_checkpoint, model, optimizer, scheduler, scaler, device
        )
        start_epoch = checkpoint_info["epoch"] + 1
        trainer.set_global_step(checkpoint_info["global_step"])

        # Set sampler epoch for distributed training
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(start_epoch)

        # Ensure scheduler is at the right step
        for _ in range(checkpoint_info["global_step"]):
            scheduler.step()

        print(
            f"Resuming from epoch {start_epoch}, step {checkpoint_info['global_step']}"
        )

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics = trainer.train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            tokenizer=tokenizer,
            sample_prompts=args.sample_prompts,
            gen_every_n_steps=args.gen_every_n_steps,
            max_gen_length=args.max_gen_length,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_every_n_steps=args.save_every_n_steps,
            checkpoint_dir=args.checkpoint_dir,
            max_grad_norm=1.0,
            log_every_n_steps=10,
        )

        val_metrics = trainer.evaluate(
            model=model, val_loader=val_loader, device=device
        )

        # Log metrics and save checkpoints (only on rank 0)
        if rank == 0:
            mode_prefix = "pretrain" if args.pretrain else "finetune"
            wandb.log(
                {
                    "epoch": epoch,
                    f"{mode_prefix}/total_loss": train_metrics["total_loss"],
                    f"{mode_prefix}/perplexity": train_metrics["total_perplexity"],
                    "val/total_loss": val_metrics["total_loss"],
                    "val/perplexity": val_metrics["perplexity"],
                }
            )

            # Save best model
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                os.makedirs("checkpoints", exist_ok=True)

                model_state_dict = (
                    model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict()
                )

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "config": config.__dict__,
                    "best_val_loss": best_val_loss,
                    "strategy_config": (
                        {
                            "instruction_token": "[INST]",
                            "response_token": "[/INST]",
                            "end_token": "</s>",
                            "max_sequence_length": args.block_size,
                        }
                        if not args.pretrain
                        else None
                    ),
                }

                checkpoint_path = f"checkpoints/best_model_{mode_prefix}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"\nSaved best model checkpoint to {checkpoint_path}")
                print(f"Validation loss: {val_metrics['total_loss']:.4f}")

    if rank == 0:
        wandb.finish()

    # Cleanup distributed training
    if dist.is_initialized():
        dist.destroy_process_group()
