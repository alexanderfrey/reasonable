import tiktoken
import argparse
import os
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
from model import GPTConfig, GPT, HeadConfig, MultiHeadGPT
from strategies import (
    SpanMaskingStrategy,
    InstructionFollowingStrategy,
    NextTokenStrategy,
    MixedStrategy,
)
from trainer import MultiHeadTrainer
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
            "step": checkpoint["step"],
            "train_metrics": checkpoint.get("train_metrics", {}),
            "val_metrics": checkpoint.get("val_metrics", {}),
            "config": checkpoint.get("config", {}),
        }

    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        raise


def count_parameters(model):
    """
    Count the number of trainable parameters in the multi-head model.
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

    # Count layer norm parameters for each head
    ln_params = {
        name: sum(p.numel() for p in ln.parameters() if p.requires_grad)
        for name, ln in model.ln_fs.items()
    }
    total_ln_params = sum(ln_params.values())

    # Count head parameters for each head
    head_params = {
        name: sum(p.numel() for p in head.parameters() if p.requires_grad)
        for name, head in model.lm_heads.items()
    }
    total_head_params = sum(head_params.values())

    # Create detailed breakdown by pathway
    pathway_breakdown = {
        name: {"layer_norm": ln_params[name], "head": head_params[name]}
        for name in model.lm_heads.keys()
    }

    return {
        "total": total_params,
        "embeddings": {
            "total": total_emb_params,
            "token": token_emb_params,
            "position": pos_emb_params,
        },
        "transformer_layers": total_layer_params,
        "layer_norms": total_ln_params,
        "heads": total_head_params,
        "pathways": pathway_breakdown,
        # Add parameters per million for easy reference
        "total_params_M": total_params / 1_000_000,
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
        "--lateral_dim", type=int, default=192, help="Dimension of lateral connections"
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

    if args.lateral_dim > args.n_embd:
        raise ValueError("lateral_dim cannot be larger than n_embd")

    if args.max_thought_length > args.block_size:
        raise ValueError("max_thought_length cannot be larger than block_size")

    return args


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
        heads=[
            HeadConfig("primary", vocab_size=vocab_size, weight=1.0),
            HeadConfig("auxiliary", vocab_size=vocab_size, weight=0.5),
            # HeadConfig("classification", vocab_size=10, weight=0.3),
        ]
    )

    # Initialize strategies based on training mode and arguments
    if args.pretrain:
        # For pretraining, we can use different strategies for each pathway
        try:
            has_mask_token = "<mask>" in tokenizer._special_tokens
        except AttributeError:
            has_mask_token = False
            print(
                "Warning: Could not check special tokens, using default mask token ID"
            )

        main_strategy = NextTokenStrategy(tokenizer=tokenizer)
        second_strategy = NextTokenStrategy(tokenizer=tokenizer)
    else:
        # For finetuning, use instruction following for both pathways
        main_strategy = NextTokenStrategy(tokenizer=tokenizer)

    # Create trainer that will handle the strategies

    # Create trainer with strategies
    strategies = {
        "primary": main_strategy,
        "auxiliary": second_strategy,
    }
    trainer = MultiHeadTrainer(strategies=strategies)

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
            },
        )

    model = MultiHeadGPT(config)
    model = model.to(device)

    # Print parameter counts (only on rank 0)
    if rank == 0:
        param_counts = count_parameters(model)
        print(f"Total parameters: {param_counts['total_params_M']:.2f}M")
        print("\nBreakdown by component:")
        print(f"- Embeddings: {param_counts['embeddings']['total']:,}")
        print(f"  - Token embeddings: {param_counts['embeddings']['token']:,}")
        print(f"  - Position embeddings: {param_counts['embeddings']['position']:,}")
        print(f"- Transformer layers: {param_counts['transformer_layers']:,}")
        print(f"- Layer norms: {param_counts['layer_norms']:,}")
        print(f"- Head parameters: {param_counts['heads']:,}")

        print("\nPathway breakdown:")
        for name, pathway in param_counts["pathways"].items():
            print(f"\n{name} pathway:")
            print(f"- Layer norm: {pathway['layer_norm']:,}")
            print(f"- Head: {pathway['head']:,}")

    # Wrap model with DDP
    if world_size is not None:
        model = DDP(
            model, device_ids=[local_rank], find_unused_parameters=False
        )  # Add this flag

    # Create dataloaders with strategy-specific collate functions
    strategies = {
        "primary": NextTokenStrategy(tokenizer=tokenizer),
        "auxiliary": NextTokenStrategy(tokenizer=tokenizer),
    }

    # Create dataloaders
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
        files_pattern=args.train_data,
        block_size=args.block_size,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        strategies=strategies,
        tokenizer=tokenizer,
        is_instruction_data=False,
    )

    if args.load_checkpoint is not None and not args.pretrain:
        warmup_steps = 100  # Adjust as needed
        lr = args.lr * 0.1  # Start with lower learning rate

        optimizer = bnb.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.1,
            optim_bits=8,
            block_wise=True,
            is_paged=True,
        )

        num_training_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        # Create optimizer and scheduler
        optimizer = bnb.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.1,
            optim_bits=8,
            block_wise=True,
            is_paged=True,
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs)

    scaler = GradScaler()

    # Training loop
    best_val_loss = float("inf")
    start_epoch = 0

    if args.load_checkpoint is not None:
        checkpoint_info = load_checkpoint(
            args.load_checkpoint, model, optimizer, scheduler, scaler, device
        )
        start_epoch = checkpoint_info["epoch"] + 1
        trainer.set_global_step(checkpoint_info["step"])

    if args.load_checkpoint is not None:
        checkpoint_info = load_checkpoint(
            args.load_checkpoint, model, optimizer, scheduler, scaler, device
        )
        start_epoch = checkpoint_info["epoch"]
        trainer.set_global_step(checkpoint_info["step"])

        # Set sampler epoch for distributed training
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(start_epoch)

        # Ensure scheduler is at the right step
        for _ in range(checkpoint_info["step"]):
            scheduler.step()

        print(f"Resuming from epoch {start_epoch}, step {checkpoint_info['step']}")

    for epoch in range(start_epoch, args.epochs):

        train_metrics = trainer.train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
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
            mode_prefix = "pretrain" if args.pretrain else "train"
            wandb.log(
                {
                    "epoch": epoch,
                    f"{mode_prefix}/total_loss": train_metrics["total_loss"],
                    f"{mode_prefix}/primary_loss": train_metrics["primary_loss"],
                    f"{mode_prefix}/auxiliary_loss": train_metrics["auxiliary_loss"],
                    "val/total_loss": val_metrics["total_loss"],
                    "val/primary_loss": val_metrics["primary_loss"],
                    "val/auxiliary_loss": val_metrics["auxiliary_loss"],
                }
            )

            # Save best model
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                # Create checkpoint directory if it doesn't exist
                os.makedirs("checkpoints", exist_ok=True)

                # Get the underlying model if using DDP
                model_state_dict = (
                    model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict()
                )

                # Save checkpoint
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
