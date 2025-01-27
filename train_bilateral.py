import tiktoken
import argparse
import os
from glob import glob
import bitsandbytes as bnb
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb  # for logging
from tqdm.auto import tqdm
import math
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from bilateral_model import BilateralGPTConfig, BilateralGPT
import json


@torch.no_grad()
def evaluate(model, val_loader, val_sampler, device, rank=0):
    """
    Evaluate the bilateral model on validation data.
    Returns metrics for both pathways.
    """
    model.eval()
    total_loss = 0
    total_main_loss = 0
    total_analysis_loss = 0

    # Set epoch for distributed sampler
    if val_sampler is not None:
        val_sampler.set_epoch(0)

    progress_bar = tqdm(val_loader, desc="Evaluating", disable=rank != 0)
    for batch in progress_bar:
        # Unpack the batch
        input_ids = batch["input_ids"].to(device)
        main_targets = batch["next_token_targets"].to(device)
        analysis_targets = batch["thought_targets"].to(device)

        with torch.amp.autocast("cuda"):
            # Get predictions from both pathways
            main_logits, analysis_logits = model(input_ids)

            # Calculate main pathway loss (next token prediction)
            main_loss = F.cross_entropy(
                main_logits.view(-1, main_logits.size(-1)), main_targets.view(-1)
            )

            # Calculate analysis pathway loss
            analysis_loss = F.cross_entropy(
                analysis_logits.view(-1, analysis_logits.size(-1)),
                analysis_targets.view(-1),
            )

            # Combine losses with equal weighting
            loss = (main_loss + analysis_loss) / 2

        # Accumulate losses
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_analysis_loss += analysis_loss.item()

        # Update progress bar
        if rank == 0:
            progress_bar.set_postfix(
                {
                    "total_loss": f"{loss.item():.4f}",
                    "main_loss": f"{main_loss.item():.4f}",
                    "analysis_loss": f"{analysis_loss.item():.4f}",
                }
            )

    # Gather losses from all processes
    if dist.is_initialized():
        # Create tensor of all losses
        losses = torch.tensor([total_loss, total_main_loss, total_analysis_loss]).to(
            device
        )
        dist.all_reduce(losses)
        total_loss, total_main_loss, total_analysis_loss = (
            losses / dist.get_world_size()
        )

    # Average losses over batches
    num_batches = len(val_loader)
    metrics = {
        "total_loss": total_loss / num_batches,
        "main_loss": total_main_loss / num_batches,
        "analysis_loss": total_analysis_loss / num_batches,
    }

    return metrics


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    train_metrics,
    val_metrics,
    config,
    path,
    rank=0,
):
    """
    Save checkpoint only on rank 0.

    Args:
        model: The bilateral model
        optimizer: The optimizer
        scheduler: The learning rate scheduler
        scaler: The gradient scaler for mixed precision training
        epoch: Current epoch number
        train_metrics: Dictionary containing training metrics (total_loss, main_loss, analysis_loss)
        val_metrics: Dictionary containing validation metrics (total_loss, main_loss, analysis_loss)
        config: BilateralGPTConfig instance
        path: Path to save the checkpoint
        rank: Process rank in distributed training
    """
    if rank == 0:
        # Get the underlying model if using DDP
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
        }

        # Save checkpoint
        torch.save(checkpoint, path)

        # Log checkpoint information
        print(f"\nSaved checkpoint to {path}")
        print(f"Checkpoint metrics:")
        print(
            f"- Train: total_loss={train_metrics['total_loss']:.4f}, "
            f"main_loss={train_metrics['main_loss']:.4f}, "
            f"analysis_loss={train_metrics['analysis_loss']:.4f}"
        )
        print(
            f"- Val: total_loss={val_metrics['total_loss']:.4f}, "
            f"main_loss={val_metrics['main_loss']:.4f}, "
            f"analysis_loss={val_metrics['analysis_loss']:.4f}"
        )


def load_checkpoint(
    path, model, optimizer=None, scheduler=None, scaler=None, device="cuda"
):
    """
    Load a checkpoint for the bilateral model.

    Args:
        path: Path to the checkpoint file
        model: The bilateral model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        scaler: Optional gradient scaler to load state
        device: Device to load the checkpoint to

    Returns:
        dict: Checkpoint information including epoch and metrics
    """
    checkpoint = torch.load(path, map_location=device)

    # Load model state
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # Optionally load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Optionally load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Optionally load scaler state
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Log checkpoint information
    print(f"\nLoaded checkpoint from {path}")
    print(f"Checkpoint metrics:")
    print(
        f"- Train: total_loss={checkpoint['train_metrics']['total_loss']:.4f}, "
        f"main_loss={checkpoint['train_metrics']['main_loss']:.4f}, "
        f"analysis_loss={checkpoint['train_metrics']['analysis_loss']:.4f}"
    )
    print(
        f"- Val: total_loss={checkpoint['val_metrics']['total_loss']:.4f}, "
        f"main_loss={checkpoint['val_metrics']['main_loss']:.4f}, "
        f"analysis_loss={checkpoint['val_metrics']['analysis_loss']:.4f}"
    )

    return {
        "epoch": checkpoint["epoch"],
        "train_metrics": checkpoint["train_metrics"],
        "val_metrics": checkpoint["val_metrics"],
        "config": checkpoint["config"],
    }


def count_parameters(model):
    """
    Count the number of trainable parameters in the bilateral model.
    Returns total params and a breakdown of parameters by component.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count embedding parameters (shared between pathways)
    token_emb_params = model.token_embedding.weight.numel()
    pos_emb_params = model.position_embedding.weight.numel()

    # Count parameters per layer
    layer_params = sum(
        p.numel() for p in model.blocks[0].parameters() if p.requires_grad
    )
    total_layer_params = layer_params * len(model.blocks)

    # Count final layer norms for both pathways
    final_ln_params_main = sum(
        p.numel() for p in model.ln_f_main.parameters() if p.requires_grad
    )
    final_ln_params_analysis = sum(
        p.numel() for p in model.ln_f_analysis.parameters() if p.requires_grad
    )

    # Count head parameters for both pathways
    main_head_params = model.lm_head.weight.numel()
    analysis_head_params = model.analysis_head.weight.numel()

    if model.config.bias:  # Add bias parameters if present
        main_head_params += model.lm_head.bias.numel()
        analysis_head_params += model.analysis_head.bias.numel()

    return {
        "total": total_params,
        "embeddings": token_emb_params + pos_emb_params,
        "transformer_layers": total_layer_params,
        "final_ln": final_ln_params_main
        + final_ln_params_analysis,  # Combined layer norm params
        "heads": main_head_params + analysis_head_params,  # Combined head params
        # Detailed breakdown of pathways
        "main_pathway": {"final_ln": final_ln_params_main, "head": main_head_params},
        "analysis_pathway": {
            "final_ln": final_ln_params_analysis,
            "head": analysis_head_params,
        },
    }


def generate_samples_ddp(model, tokenizer, sample_prompts, device, rank):
    """Generate samples in a DDP-safe way from both pathways of the bilateral model"""
    # Store original training state
    was_training = model.training

    # Switch to eval mode
    model.eval()

    # Sync all processes
    if dist.is_initialized():
        dist.barrier()

    samples = ""
    if rank == 0:  # Only generate on rank 0
        try:
            # Use model.module if it exists (DDP), otherwise use model directly
            generating_model = model.module if hasattr(model, "module") else model

            # Generate samples for each prompt
            sample_outputs = []
            for prompt in sample_prompts:
                # Tokenize the prompt
                tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

                generated_text = ""
                generated_analysis = ""

                with torch.no_grad():
                    # Generate tokens one at a time
                    for _ in range(100):  # Set a maximum length for generation
                        # Get predictions from both pathways
                        main_logits, analysis_logits = generating_model(tokens)

                        # Get next token prediction (from main pathway)
                        next_token_logits = main_logits[0, -1, :]
                        next_token = sample_top_p(
                            next_token_logits.unsqueeze(0), top_p=0.9
                        )
                        tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)

                        # Check for EOS token
                        if next_token.item() == tokenizer.encode("<|endoftext|>")[0]:
                            break

                    # Generate analysis from the analysis pathway using the full sequence
                    _, final_analysis_logits = generating_model(tokens)
                    analysis_tokens = generate_from_logits(
                        final_analysis_logits[0], top_p=0.9, max_tokens=50
                    )

                    # Decode the generated sequences
                    generated_text = tokenizer.decode(tokens[0].tolist())
                    generated_analysis = tokenizer.decode(analysis_tokens.tolist())

                # Format the sample output
                sample_output = (
                    f"Prompt: {prompt}\n"
                    f"Generated Continuation:\n{generated_text}\n"
                    f"Generated Analysis:\n{generated_analysis}\n"
                    f"{'='*50}\n"
                )
                sample_outputs.append(sample_output)

            # Combine all samples
            samples = "\n".join(sample_outputs)

        except Exception as e:
            print(f"Error generating samples: {str(e)}")

    # Sync all processes again
    if dist.is_initialized():
        dist.barrier()

    # Restore original training state
    if was_training:
        model.train()

    return samples


def sample_top_p(logits, top_p=0.9):
    """Sample from the distribution with top-p (nucleus) sampling"""
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter back the indices and zero out removed tokens
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    probs[indices_to_remove] = 0
    # Renormalize
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Sample from the filtered distribution
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def generate_from_logits(logits, top_p=0.9, max_tokens=50):
    """Generate a sequence of tokens from logits using top-p sampling"""
    generated_tokens = []
    current_token_idx = 0

    while current_token_idx < max_tokens:
        # Sample next token
        next_token_logits = logits[current_token_idx : current_token_idx + 1]
        next_token = sample_top_p(next_token_logits, top_p=top_p)

        # Add to generated sequence
        generated_tokens.append(next_token.item())
        current_token_idx += 1

        # Check for EOS token
        if next_token.item() == tokenizer.encode("<|endoftext|>")[0]:
            break

    return torch.tensor(generated_tokens)


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


class BilateralDataset(Dataset):
    """Dataset for bilateral model training with input text and thought annotations"""

    def __init__(self, files_pattern, block_size, encoding_name="gpt2"):
        super().__init__()
        self.block_size = block_size
        self.encoding_name = encoding_name

        # Load all JSON files matching the pattern
        self.examples = []
        for filepath in glob.glob(files_pattern):
            with open(filepath, "r", encoding="utf-8") as f:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    self.examples.extend(file_data)
                else:
                    self.examples.append(file_data)

        print(f"Loaded {len(self.examples)} examples from {files_pattern}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {"input": example["input"], "thought": example["thought"]}


def create_distributed_dataloaders(
    files_pattern,
    block_size,
    batch_size,
    rank,
    world_size,
    num_workers=4,
    prefetch_factor=2,
    encoding_name="gpt2",
    collate_fn=None,
):
    """Create distributed training and validation dataloaders for bilateral model"""

    # Create dataset
    dataset = BilateralDataset(files_pattern, block_size, encoding_name)

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, train_sampler, val_sampler


def get_data_info(files_pattern):
    """Get information about the dataset for logging purposes"""
    dataset = BilateralDataset(
        files_pattern, block_size=1024
    )  # block_size doesn't matter here

    # Analyze a sample of examples
    sample_size = min(1000, len(dataset))
    input_lengths = []
    thought_lengths = []

    for idx in range(sample_size):
        example = dataset[idx]
        input_lengths.append(len(example["input"].split()))
        thought_lengths.append(len(example["thought"].split()))

    info = {
        "total_examples": len(dataset),
        "avg_input_length": sum(input_lengths) / len(input_lengths),
        "avg_thought_length": sum(thought_lengths) / len(thought_lengths),
        "max_input_length": max(input_lengths),
        "max_thought_length": max(thought_lengths),
    }

    return info


def train_one_epoch(
    model,
    train_loader,
    train_sampler,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    tokenizer=None,
    gen_every_n_steps=None,
    sample_prompts=None,
    rank=0,
    gradient_accumulation_steps=1,
):
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_analysis_loss = 0
    global_step = epoch * len(train_loader)

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    if rank == 0:
        print(f"\nGeneration settings:")
        print(f"Tokenizer present: {tokenizer is not None}")
        print(f"gen_every_n_steps: {gen_every_n_steps}")
        print(f"sample_prompts: {sample_prompts}")

    optimizer.zero_grad()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=rank != 0)
    for batch_idx, batch in enumerate(progress_bar):
        # Unpack the batch
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        main_targets = batch["next_token_targets"].to(device, non_blocking=True)
        analysis_targets = batch["thought_targets"].to(device, non_blocking=True)

        # Determine if this is an accumulation step
        is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps != 0

        # Forward pass with autocast
        with torch.amp.autocast("cuda", dtype=torch.float16):
            # Get logits from both pathways
            main_logits, analysis_logits = model(input_ids)

            # Calculate losses for both pathways
            main_loss = F.cross_entropy(
                main_logits.view(-1, main_logits.size(-1)), main_targets.view(-1)
            )

            analysis_loss = F.cross_entropy(
                analysis_logits.view(-1, analysis_logits.size(-1)),
                analysis_targets.view(-1),
            )

            # Combine losses with equal weighting
            loss = (main_loss + analysis_loss) / 2
            loss = loss / gradient_accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        if not is_accumulation_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            scheduler.step()

        # Update metrics
        total_loss += loss.item() * gradient_accumulation_steps
        total_main_loss += main_loss.item() * gradient_accumulation_steps
        total_analysis_loss += analysis_loss.item() * gradient_accumulation_steps

        # Check for generation
        should_generate = (
            tokenizer is not None
            and gen_every_n_steps is not None
            and sample_prompts is not None
            and global_step > 0
            and global_step % gen_every_n_steps == 0
        )

        if should_generate:
            samples = generate_samples_ddp(
                model, tokenizer, sample_prompts, device, rank
            )

            if rank == 0:
                print(f"\nGenerated samples at step {global_step}:")
                print(samples)
                wandb.log(
                    {"generated_samples": wandb.Html(samples.replace("\n", "<br>"))}
                )

        if rank == 0 and not is_accumulation_step:
            current_lr = scheduler.get_last_lr()[0]

            # Update progress bar with both losses
            progress_bar.set_postfix(
                {
                    "total_loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "main_loss": f"{main_loss.item():.4f}",
                    "analysis_loss": f"{analysis_loss.item():.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": global_step,
                }
            )

            # Log to wandb
            wandb.log(
                {
                    "train_loss": loss.item() * gradient_accumulation_steps,
                    "main_loss": main_loss.item(),
                    "analysis_loss": analysis_loss.item(),
                    "learning_rate": current_lr,
                    "grad_scale": scaler.get_scale(),
                    "global_step": global_step,
                }
            )

        global_step += 1

    if dist.is_initialized():
        dist.all_reduce(torch.tensor([total_loss]).to(device))
        total_loss /= dist.get_world_size()

    return {
        "total_loss": total_loss / len(train_loader),
        "main_loss": total_main_loss / len(train_loader),
        "analysis_loss": total_analysis_loss / len(train_loader),
    }


def prepare_bilateral_batch(examples, tokenizer, max_length):
    """
    Prepare a batch of examples for the bilateral model.

    Args:
        examples: List of dictionaries containing 'input' and 'thought' keys
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length

    Returns:
        Dictionary containing input_ids, next_token_targets, and thought_targets
    """
    # Tokenize inputs for both pathways
    inputs = [ex["input"] for ex in examples]
    thoughts = [ex["thought"] for ex in examples]

    # Tokenize main pathway inputs (with shifting for next-token prediction)
    input_encodings = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Create shifted targets for next-token prediction
    main_targets = input_encodings["input_ids"].clone()
    main_targets = torch.roll(main_targets, shifts=-1, dims=1)
    main_targets[:, -1] = -100  # Mask last token

    # Tokenize analysis pathway targets (thoughts)
    thought_encodings = tokenizer(
        thoughts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": input_encodings["input_ids"],
        "next_token_targets": main_targets,
        "thought_targets": thought_encodings["input_ids"],
    }


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
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
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
        default="./data/*.json",
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
        default=10,
        help="Save checkpoint every N epochs",
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

    # Create bilateral config
    config = BilateralGPTConfig(
        vocab_size=vocab_size,
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=args.block_size,
        dropout=0.1,
        lateral_dim=192,  # Half of n_embd for lateral connections
    )

    # Initialize wandb only on rank 0
    if rank == 0:
        wandb.init(
            project="bilateral-gpt-training",
            config={
                "vocab_size": vocab_size,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "block_size": config.block_size,
                "dropout": config.dropout,
                "lateral_dim": config.lateral_dim,
                "learning_rate": args.lr,
                "batch_size": args.batch_size * (world_size or 1),  # Global batch size
                "optimizer": "8-bit AdamW",
                "num_gpus": world_size or 1,
            },
        )

    # Create bilateral model
    model = BilateralGPT(config)
    model = model.to(device)

    if rank == 0:  # Only print on main process
        param_counts = count_parameters(model)
        # Print parameter counts
        print(f"Model Parameter Counts:")
        print(f"Total Parameters: {param_counts['total']:,}")
        print(f"- Embeddings: {param_counts['embeddings']:,}")
        print("- Main Pathway:")
        print(f"  - Final Layer Norm: {param_counts['main_pathway']['final_ln']:,}")
        print(f"  - Head: {param_counts['main_pathway']['head']:,}")
        print("- Analysis Pathway:")
        print(f"  - Final Layer Norm: {param_counts['analysis_pathway']['final_ln']:,}")
        print(f"  - Head: {param_counts['analysis_pathway']['head']:,}")
        print(f"- Transformer Layers: {param_counts['transformer_layers']:,}")

    # Wrap model with DDP
    if world_size is not None:
        model = DDP(model, device_ids=[local_rank])

    # Create data collator for bilateral training
    def collate_fn(examples):
        return prepare_bilateral_batch(examples, tokenizer, args.block_size)

    # Create distributed dataloaders with bilateral data
    train_loader, val_loader, train_sampler, val_sampler = (
        create_distributed_dataloaders(
            args.train_data,
            block_size=args.block_size,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=collate_fn,
        )
    )

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

    checkpoint_dir = Path("checkpoints")
    if rank == 0:  # Only create directory on main process
        checkpoint_dir.mkdir(exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            train_sampler,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
            tokenizer=tokenizer,
            gen_every_n_steps=args.gen_every_n_steps,
            sample_prompts=args.sample_prompts,
            rank=rank,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, val_sampler, device, rank)

        # Log metrics (only on rank 0)
        if rank == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/total_loss": train_metrics["total_loss"],
                    "train/main_loss": train_metrics["main_loss"],
                    "train/analysis_loss": train_metrics["analysis_loss"],
                    "val/total_loss": val_metrics["total_loss"],
                    "val/main_loss": val_metrics["main_loss"],
                    "val/analysis_loss": val_metrics["analysis_loss"],
                }
            )

            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"- Main Loss: {train_metrics['main_loss']:.4f}")
            print(f"- Analysis Loss: {train_metrics['analysis_loss']:.4f}")
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"- Main Loss: {val_metrics['main_loss']:.4f}")
            print(f"- Analysis Loss: {val_metrics['analysis_loss']:.4f}")

        # Save checkpoints
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                train_metrics,
                val_metrics,
                config,
                checkpoint_dir / "best_model.pt",
                rank=rank,
            )

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                train_metrics,
                val_metrics,
                config,
                checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                rank=rank,
            )

    if rank == 0:
        wandb.finish()

    # Cleanup distributed training
    if dist.is_initialized():
        dist.destroy_process_group()
