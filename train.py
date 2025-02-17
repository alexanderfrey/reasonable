import tiktoken
import argparse
import os, math
from glob import glob
from graphviz import Digraph
from typing import List, Dict, Optional, Union, Tuple
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

from model import GPTConfig, GPT, MoEGPT
from strategies import (
    TeacherForcingStrategy,
    InstructionFollowingStrategy,
    NextTokenStrategy,
)
from trainer import SingleHeadTrainer
from data_utils import create_dataloaders

# from clearml import Task

# task = Task.init(project_name="reasonable", task_name="misty_bird")

def interpolate_pos_embeddings(old_embeddings, new_length):
    """
    Interpolate position embeddings from one length to another.
    """
    old_length = old_embeddings.shape[0]
    hidden_dim = old_embeddings.shape[1]
    
    # Create indices for interpolation
    old_positions = torch.arange(0, old_length).float()
    new_positions = torch.arange(0, new_length).float() * (old_length - 1) / (new_length - 1)
    
    # Interpolate
    if old_length != new_length:
        new_embeddings = torch.zeros(new_length, hidden_dim)
        for dim in range(hidden_dim):
            old_embedding = old_embeddings[:, dim]
            new_embeddings[:, dim] = torch.interp(new_positions, old_positions, old_embedding)
        return new_embeddings
    return old_embeddings

def visualize_model_architecture(model, output_path="model_architecture"):
    """
    Create a visual representation of the model architecture including auxiliary heads
    Args:
        model: The GPT model instance
        output_path: Path to save the visualization
    """
    # Create a new directed graph
    dot = Digraph(comment="Model Architecture")
    dot.attr(rankdir="TB")  # Top to bottom layout

    # Global graph attributes
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue")

    # Add input node
    dot.node("input", "Input\nTokens", fillcolor="lightgreen")

    # Add embedding layers
    with dot.subgraph(name="cluster_0") as c:
        c.attr(label="Embeddings")
        c.node("token_emb", "Token\nEmbedding")
        c.node("pos_emb", "Position\nEmbedding")
        c.node("emb_sum", "+", shape="circle")

    dot.edge("input", "token_emb")
    dot.edge("token_emb", "emb_sum")
    dot.edge("pos_emb", "emb_sum")

    # Add transformer blocks
    prev_node = "emb_sum"
    num_blocks = len(model.blocks)

    for i in range(num_blocks):
        cluster_name = f"cluster_{i+1}"
        with dot.subgraph(name=cluster_name) as c:
            c.attr(label=f"Transformer Block {i+1}")

            # Layer norm 1
            ln1_name = f"ln1_{i}"
            c.node(ln1_name, "LayerNorm")

            # Attention
            attn_name = f"attn_{i}"
            c.node(attn_name, "Multi-Head\nAttention")

            # Add residual connection
            add1_name = f"add1_{i}"
            c.node(add1_name, "+", shape="circle")

            # Layer norm 2
            ln2_name = f"ln2_{i}"
            c.node(ln2_name, "LayerNorm")

            # Feed forward (check if MoE)
            if hasattr(model.blocks[i], "moe"):
                ff_name = f"moe_{i}"
                c.node(ff_name, "MoE Layer\n(Experts)", fillcolor="lightsalmon")
                c.node(f"router_{i}", "Router", fillcolor="lightpink")
                dot.edge(f"router_{i}", ff_name)
            else:
                ff_name = f"ff_{i}"
                c.node(ff_name, "Feed\nForward")

            # Add residual connection
            add2_name = f"add2_{i}"
            c.node(add2_name, "+", shape="circle")

            # Connect nodes within block
            dot.edge(prev_node, ln1_name)
            dot.edge(ln1_name, attn_name)
            dot.edge(attn_name, add1_name)
            dot.edge(prev_node, add1_name)
            dot.edge(add1_name, ln2_name)
            if hasattr(model.blocks[i], "moe"):
                dot.edge(ln2_name, f"router_{i}")
            else:
                dot.edge(ln2_name, ff_name)
            dot.edge(ff_name, add2_name)
            dot.edge(add1_name, add2_name)

            prev_node = add2_name

    # Final layer norm
    dot.node("ln_f", "Final\nLayerNorm")
    dot.edge(prev_node, "ln_f")

    # Main language modeling head
    dot.node("lm_head", "LM Head", fillcolor="lightgreen")
    dot.edge("ln_f", "lm_head")

    # Add auxiliary heads if present
    if hasattr(model, "aux_heads"):
        num_aux_heads = len(model.aux_heads)
        with dot.subgraph(name="cluster_aux") as c:
            c.attr(label="Auxiliary Heads")
            for i in range(num_aux_heads):
                aux_name = f"aux_head_{i}"
                c.node(aux_name, f"Aux Head {i+1}\n(t+{i+2})", fillcolor="lightyellow")
                dot.edge("ln_f", aux_name)

    # Save the visualization
    try:
        dot.render(output_path, format="png", cleanup=True)
        print(f"Architecture visualization saved to {output_path}.png")
    except Exception as e:
        print(f"Error saving visualization: {e}")

    return dot


def load_checkpoint(
    path, model, optimizer=None, scheduler=None, scaler=None, device="cuda"
):
    """Modified to handle pretrain -> finetune transition"""
    try:
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]
        
        # Remove position embedding from state dict since we're not using it anymore
        if "position_embedding.weight" in state_dict:
            del state_dict["position_embedding.weight"]
        
        if dist.is_initialized():
            model_to_load = model.module if hasattr(model, "module") else model
        else:
            model_to_load = model
            
        # Load the modified state dict with strict=False to ignore missing position embedding
        model_to_load.load_state_dict(state_dict, strict=False)
        
        # Load optimizer state
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

        # Load scheduler state
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except ValueError as e:
                print(f"Warning: Could not load scheduler state: {e}")

        # Load scaler state
        if scaler is not None and "scaler_state_dict" in checkpoint:
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except ValueError as e:
                print(f"Warning: Could not load scaler state: {e}")

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



def print_parameter_summary(param_dict):
    """Print a detailed summary of model parameters including auxiliary heads"""
    print("\n=== Model Parameter Summary ===")
    print(f"Total Parameters: {param_dict['total_params_M']:.2f}M")

    print("\nBreakdown:")
    print(f"  Embeddings: {param_dict['embeddings']['total']:,}")
    print(f"  Attention Mechanisms: {param_dict['attention']:,}")
    print(f"  Layer Normalizations: {param_dict['layer_norms']:,}")
    print(f"  Output Head: {param_dict['head']:,}")

    print("\nFeed Forward Components:")
    ff = param_dict["feed_forward"]
    print(f"  Standard FFN: {ff['standard_ffn']:,}")
    print(f"  MoE Total: {ff['moe_total']:,}")

    # Add auxiliary head information
    aux = param_dict["auxiliary_heads"]
    if aux["num_heads"] > 0:
        print("\nAuxiliary Head Statistics:")
        print(f"  Number of Auxiliary Heads: {aux['num_heads']}")
        print(f"  Total Auxiliary Parameters: {aux['total_params']:,}")
        print(f"  Parameters per Head: {aux['params_per_head']:,}")
        print(f"  Auxiliary Parameter %: {aux['param_percent']:.2f}%")

    print("\nMoE Statistics:")
    moe = param_dict["moe_stats"]
    print(f"  Number of MoE Blocks: {moe['num_moe_blocks']}")
    print(f"  Total Experts: {moe['total_experts']}")
    print(f"  Average Parameters per Expert: {moe['params_per_expert']:,}")
    print(f"  Router Parameters: {moe['router_params']:,}")
    print(f"  MoE Parameter %: {moe['moe_param_percent']:.2f}%")


# Usage example:
# param_dict = count_parameters(model)
# print_parameter_summary(param_dict)


def count_parameters(model):
    """
    Count the number of trainable parameters in the GPT model.
    Now includes auxiliary head counting.
    """
    # Get the unwrapped model if using DDP
    model = model.module if hasattr(model, "module") else model

    # Count embedding parameters (accounting for weight tying)
    token_emb_params = model.token_embedding.weight.numel()
    total_emb_params = token_emb_params  # Removed position embedding

    # Initialize counters for components
    attention_params = 0
    ff_params = 0
    moe_params = 0
    router_params = 0
    expert_params = 0
    layer_norm_params = 0
    aux_head_params = 0  # New counter for auxiliary heads

    # Count parameters by block type
    for block in model.blocks:
        # Attention parameters
        attention_params += sum(
            p.numel() for p in block.attn.parameters() if p.requires_grad
        )

        # Layer norms
        layer_norm_params += sum(
            p.numel() for p in block.ln1.parameters() if p.requires_grad
        )
        layer_norm_params += sum(
            p.numel() for p in block.ln2.parameters() if p.requires_grad
        )

        # Check if block is MoE or standard FFN
        if hasattr(block, "moe"):
            # Count MoE router parameters
            router_params += sum(
                p.numel() for p in block.moe.router.parameters() if p.requires_grad
            )

            # Count parameters for each expert
            expert_count = len(block.moe.experts)
            expert_params_this_layer = sum(
                sum(p.numel() for p in expert.parameters() if p.requires_grad)
                for expert in block.moe.experts
            )
            expert_params += expert_params_this_layer

            # Add to total MoE parameters
            moe_params += router_params + expert_params_this_layer
        else:
            # Standard FFN parameters
            ff_params += sum(
                p.numel() for p in block.ff.parameters() if p.requires_grad
            )

    # Count final layer norm parameters
    final_ln_params = sum(p.numel() for p in model.ln_f.parameters() if p.requires_grad)
    layer_norm_params += final_ln_params

    # For head parameters, don't count the weight matrix since it's tied to embeddings
    head_params = 0 if not model.lm_head.bias is None else model.lm_head.bias.numel()

    # Count auxiliary head parameters if they exist
    num_aux_heads = len(model.aux_heads) if hasattr(model, "aux_heads") else 0
    if num_aux_heads > 0:
        aux_head_params = sum(
            p.numel()
            for head in model.aux_heads
            for p in head.parameters()
            if p.requires_grad
        )

    # Calculate total parameters (accounting for weight tying)
    total_params = (
        total_emb_params  # Embeddings (now just token embeddings)
        + attention_params  # Attention layers
        + ff_params  # Standard FFN layers
        + moe_params  # MoE layers (including routers and experts)
        + layer_norm_params  # All layer norms
        + head_params  # Head (only bias if present)
        + aux_head_params  # Auxiliary heads
    )

    # Calculate MoE-specific statistics
    moe_blocks = sum(1 for block in model.blocks if hasattr(block, "moe"))
    total_experts = sum(
        len(block.moe.experts) for block in model.blocks if hasattr(block, "moe")
    )

    # Calculate average parameters per expert
    avg_expert_params = expert_params / total_experts if total_experts > 0 else 0

    return {
        "total_params_M": total_params / 1_000_000,
        "total_params": total_params,
        "embeddings": {
            "total": total_emb_params,
            "token": token_emb_params,
        },  # Removed position embedding from dict
        "attention": attention_params,
        "feed_forward": {
            "standard_ffn": ff_params,
            "moe_total": moe_params,
            "router_total": router_params,
            "expert_total": expert_params,
            "avg_per_expert": avg_expert_params,
        },
        "layer_norms": layer_norm_params,
        "head": head_params,
        "auxiliary_heads": {
            "num_heads": num_aux_heads,
            "total_params": aux_head_params,
            "params_per_head": (
                aux_head_params / num_aux_heads if num_aux_heads > 0 else 0
            ),
            "param_percent": (
                (aux_head_params / total_params) * 100 if num_aux_heads > 0 else 0
            ),
        },
        "moe_stats": {
            "num_moe_blocks": moe_blocks,
            "total_experts": total_experts,
            "params_per_expert": avg_expert_params,
            "router_params": router_params,
            "total_moe_params": moe_params,
            "moe_param_percent": (moe_params / total_params) * 100,
        },
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
        default=4,
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
    parser.add_argument(
        "--use_curriculum", action="store_true", help="Enable curriculum learning"
    )
    parser.add_argument(
        "--steps_per_file",
        type=int,
        default=1000,
        help="Number of steps to train on each file before adding next one",
    )
    parser.add_argument(
        "--train_data_patterns",
        nargs="+",
        type=str,
        help="List of file patterns for curriculum learning, in order of complexity",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=0,
        help="Number of steps to train on each file before adding next one",
    )
    parser.add_argument(
        "--n_aux_heads",
        type=int,
        default=0,
        help="Number of steps to train on each file before adding next one",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split for HuggingFace datasets",
    )
    parser.add_argument(
        "--dataset_revision",
        type=str,
        default="main",
        help="Dataset revision for HuggingFace datasets",
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
    base_lr = args.lr

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
            0.1, 0.5 * (1.0 + math.cos(math.pi * progress))
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
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        dropout=0.1,
        bias=True,
        n_aux_heads=args.n_aux_heads,
    )

    if args.num_experts == 0:
        model = GPT(config)
    else:
        model = MoEGPT(
            config,
            num_experts=args.num_experts,
            top_k=args.num_experts // 2,
            moe_layers=None,
        )
    model = model.to(device)

    # Initialize strategies based on training mode and arguments
    if args.pretrain:
        try:
            has_mask_token = "<mask>" in tokenizer._special_tokens
        except AttributeError:
            has_mask_token = False
            print(
                "Warning: Could not check special tokens, using default mask token ID"
            )

        strategy = TeacherForcingStrategy(tokenizer)
    else:
        strategy = InstructionFollowingStrategy(
            tokenizer=tokenizer,
            instruction_token="[INST]",
            response_token="[/INST]",
            end_token="</s>",
            pad_token_id=50256,
            max_length=args.block_size,
        )

    trainer = SingleHeadTrainer(strategy=strategy)

    # Initialize wandb only on rank 0
    if rank == 0:
        wandb_config = {
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
            "instruction_token": "[INST]" if not args.pretrain else None,
            "response_token": "[/INST]" if not args.pretrain else None,
        }

        # Add dataset info to wandb config
        if args.train_data.startswith(("hf://", "huggingface://")):
            wandb_config.update(
                {
                    "dataset_source": "huggingface",
                    "dataset_name": args.train_data.split("://")[1],
                    "dataset_split": args.dataset_split,
                    "dataset_revision": args.dataset_revision,
                }
            )

        wandb.init(
            project="multihead-gpt-training",
            config=wandb_config,
        )

    # Print parameter counts (only on rank 0)
    if rank == 0:
        param_dict = count_parameters(model)
        print_parameter_summary(param_dict)
        visualize_model_architecture(model)

    # Wrap model with DDP
    if world_size is not None:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Prepare dataset configuration
    if args.train_data.startswith(("hf://", "huggingface://")):
        dataset_config = {
            "path": args.train_data.split("://")[1],
            "split": args.dataset_split,
            "revision": args.dataset_revision,
        }
    else:
        dataset_config = args.train_data

    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
        files_pattern=dataset_config,
        block_size=args.block_size,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        strategy=strategy,
        tokenizer=tokenizer,
        n_aux_heads=args.n_aux_heads,
        prefetch_factor=args.prefetch_factor,
        num_workers=args.num_workers,
    )

    optimizer, scheduler = configure_optimizer(model, args, train_loader)
    scaler = GradScaler(enabled=True)

    # Training loop
    best_val_loss = float("inf")
    start_epoch = 0

    if args.load_checkpoint is not None:
        checkpoint_info = load_checkpoint(
            path=args.load_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

        start_epoch = checkpoint_info["epoch"] + 1
        trainer.set_global_step(checkpoint_info["global_step"])

        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(start_epoch)

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
            block_size=args.block_size,
            tokenizer=tokenizer,
            sample_prompts=args.sample_prompts,
            gen_every_n_steps=args.gen_every_n_steps,
            max_gen_length=args.max_gen_length,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_every_n_steps=args.save_every_n_steps,
            checkpoint_dir=args.checkpoint_dir,
            max_grad_norm=1.0,
            log_every_n_steps=10,
            use_moe=args.num_experts > 0,
            num_aux_heads=args.n_aux_heads,
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
                    f"{mode_prefix}/total_loss": train_metrics["final_total_loss"],
                    f"{mode_prefix}/perplexity": train_metrics["final_perplexity"],
                    "val/total_loss": val_metrics["total_loss"],
                    "val/perplexity": val_metrics["perplexity"],
                }
            )

            # Save best model checkpoint if validation loss improves
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                os.makedirs(args.checkpoint_dir, exist_ok=True)

                # Use the trainer's checkpoint saving method.
                trainer.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=trainer.global_step,
                    save_dir=args.checkpoint_dir,
                )

                checkpoint_path = f"{args.checkpoint_dir}/checkpoint_epoch{epoch}_step{trainer.global_step}.pt"
                print(f"\nSaved best model checkpoint to {checkpoint_path}")
                print(f"Validation loss: {val_metrics['total_loss']:.4f}")

    if rank == 0:
        wandb.finish()

    # Cleanup distributed training
    if dist.is_initialized():
        dist.destroy_process_group()
