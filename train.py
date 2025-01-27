import tiktoken
import argparse
import os
from glob import glob
import bitsandbytes as bnb
import torch
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig
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


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    Returns total params and a breakdown of parameters by component.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count embedding parameters
    token_emb_params = model.token_embedding.weight.numel()
    pos_emb_params = model.position_embedding.weight.numel()
    
    # Count parameters per layer
    layer_params = sum(p.numel() for p in model.blocks[0].parameters() if p.requires_grad)
    total_layer_params = layer_params * len(model.blocks)
    
    # Count final layer norm and head parameters
    final_ln_params = sum(p.numel() for p in model.ln_f.parameters() if p.requires_grad)
    head_params = model.lm_head.weight.numel()
    if model.config.bias:  # Add bias parameters if present
        head_params += model.lm_head.bias.numel()
    
    return {
        'total': total_params,
        'embeddings': token_emb_params + pos_emb_params,
        'transformer_layers': total_layer_params,
        'final_ln': final_ln_params,
        'head': head_params
    }

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return None, None, None

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
    return rank, world_size, local_rank

def create_distributed_dataloaders(files_pattern, block_size, batch_size, rank, world_size, num_workers=4, prefetch_factor=2, encoding_name="gpt2"):
    """Create distributed training and validation dataloaders with optimized settings"""
    dataset = TextDataset(files_pattern, block_size, encoding_name)
    
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
        drop_last=True  # Drop last incomplete batch for better GPU utilization
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,  # Adjust based on CPU cores available
        pin_memory=True,
        prefetch_factor=prefetch_factor,  # Prefetch 2 batches per worker
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )
    
    return train_loader, val_loader, train_sampler, val_sampler


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=40, device='cuda'):
    """Generate text using the model.
    
    Args:
        model: The GPT model
        tokenizer: The tiktoken tokenizer
        prompt: String prompt to start generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Number of top tokens to consider for sampling
        device: Device to run generation on
    
    Returns:
        Generated text as string
    """
    model.eval()
    # Encode the prompt
    prompt_tokens = tokenizer.encode(prompt)
    x = torch.tensor(prompt_tokens, dtype=torch.long)[None,...].to(device)
    
    # Generate tokens
    for _ in range(max_new_tokens):
        # Get model predictions
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = model(x)
        
        # Focus on last time step
        logits = logits[:, -1, :] / temperature
        
        # Optional: top-k sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Apply softmax to convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append sampled token to the sequence
        x = torch.cat((x, next_token), dim=1)
        
        # If we hit the EOT token, stop
        if next_token.item() == tokenizer.eot_token:
            break
    
    # Decode the generated tokens
    generated_tokens = x[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text

def generate_samples_ddp(model, tokenizer, sample_prompts, device, rank):
    """Generate samples in a DDP-safe way"""
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
            generating_model = model.module if hasattr(model, 'module') else model
            samples = generate_samples(generating_model, tokenizer, sample_prompts, device)
        except Exception as e:
            print(f"Error generating samples: {str(e)}")
    
    # Sync all processes again
    if dist.is_initialized():
        dist.barrier()
    
    # Restore original training state
    if was_training:
        model.train()
    
    return samples

def generate_samples(model, tokenizer, prompts, device='cuda'):
    """Generate samples from a list of prompts"""
    samples = []
    for prompt in prompts:
        sample = generate_text(model, tokenizer, prompt, device=device)
        samples.append(f"Prompt: {prompt}\nGenerated: {sample}\n")
    return "\n".join(samples)

# Add arguments for generation
def get_args():
    parser = argparse.ArgumentParser()
    # Original arguments
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--train_data", type=str, default="./text_files/*.txt")
    parser.add_argument("--gen_every_n_steps", type=int, default=100,
                       help="Generate samples every N training steps")
    parser.add_argument("--sample_prompts", type=str, nargs='+',
                       default=["Once upon a time", "In a galaxy far far away"],
                       help="Prompts to use for generation during training")
    
    # New optimization-related arguments
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of steps to accumulate gradients")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                       help="Number of batches to prefetch per worker")
    parser.add_argument("--opt_level", type=str, default="O2",
                       help="AMP optimization level")
    return parser.parse_args()


def train_one_epoch(model, train_loader, train_sampler, optimizer, scheduler, scaler, 
                   device, epoch, tokenizer=None, gen_every_n_steps=None, 
                   sample_prompts=None, rank=0, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0
    global_step = epoch * len(train_loader)
    
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    
    # Debug print at start of epoch
    if rank == 0:
        print(f"\nGeneration settings:")
        print(f"Tokenizer present: {tokenizer is not None}")
        print(f"gen_every_n_steps: {gen_every_n_steps}")
        print(f"sample_prompts: {sample_prompts}")
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}', disable=rank != 0)
    for batch_idx, (x, y) in enumerate(progress_bar):
        # Determine if this is an accumulation step
        is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps != 0
        
        # Move batch to device
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        # Forward pass with autocast
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
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
        
        # Check for generation - note we moved it outside the rank check
        should_generate = (
            tokenizer is not None and 
            gen_every_n_steps is not None and 
            sample_prompts is not None and 
            global_step > 0 and
            global_step % gen_every_n_steps == 0
        )
        
        if should_generate:
            samples = generate_samples_ddp(model, tokenizer, sample_prompts, device, rank)
            
            # Only log on rank 0
            if rank == 0:
                print(f"\nGenerated samples at step {global_step}:")
                print(samples)
                wandb.log({
                    'generated_samples': wandb.Html(samples.replace('\n', '<br>'))
                })
        
        if rank == 0 and not is_accumulation_step:
            current_lr = scheduler.get_last_lr()[0]
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{current_lr:.2e}',
                'step': global_step
            })
            
            # Log to wandb
            wandb.log({
                'train_loss': loss.item() * gradient_accumulation_steps,
                'learning_rate': current_lr,
                'grad_scale': scaler.get_scale(),
                'global_step': global_step
            })
        
        global_step += 1
    
    if dist.is_initialized():
        dist.all_reduce(torch.tensor([total_loss]).to(device))
        total_loss /= dist.get_world_size()
    
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(model, val_loader, val_sampler, device, rank=0):
    model.eval()
    total_loss = 0
    
    # Set epoch for distributed sampler
    if val_sampler is not None:
        val_sampler.set_epoch(0)
    
    for x, y in tqdm(val_loader, desc='Evaluating', disable=rank != 0):
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    
    # Gather loss from all processes
    if dist.is_initialized():
        dist.all_reduce(torch.tensor([total_loss]).to(device))
        total_loss /= dist.get_world_size()
    
    return total_loss / len(val_loader)

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, train_loss, val_loss, config, path, rank=0):
    """Save checkpoint only on rank 0"""
    if rank == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),  # Save the inner model
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config.__dict__,
        }, path)
        print(f"Saved checkpoint to {path}")

class TextDataset(Dataset):
    def __init__(self, files_pattern, block_size, encoding_name="gpt2"):
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.block_size = block_size
        
        # Get all files matching the pattern
        self.files = glob(files_pattern)
        if not self.files:
            raise ValueError(f"No files found matching pattern: {files_pattern}")
        
        print(f"Found {len(self.files)} files")
        
        # Read and tokenize all files
        self.tokens = []
        for file_path in self.files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                # Add tokens for this file
                self.tokens.extend(self.tokenizer.encode(text))
                # Add an end of text token between files
                self.tokens.append(self.tokenizer.eot_token)
            except Exception as e:
                print(f"Warning: Could not process file {file_path}: {str(e)}")
        
        print(f"Total tokens: {len(self.tokens)}")
        
    def __len__(self):
        return len(self.tokens) - self.block_size
        
    def __getitem__(self, i):
        # Get block_size tokens from position i
        chunk = self.tokens[i:i + self.block_size + 1]  # +1 for the target
        
        # Convert to torch tensors
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y

def create_dataloaders(files_pattern, block_size, batch_size, encoding_name="gpt2"):
    """Create training and validation dataloaders from multiple files"""
    dataset = TextDataset(files_pattern, block_size, encoding_name)
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader

# When creating your GPT model, use the tokenizer's vocab size
def get_vocab_size(encoding_name="gpt2"):
    tokenizer = tiktoken.get_encoding(encoding_name)
    return tokenizer.n_vocab

def setup_instruction_finetuning(pretrained_path, config, device, rank=0):
    """
    Setup model for instruction fine-tuning from pretrained weights
    """
    # Load the pretrained checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Create model with same config as pretrained
    pretrained_config = GPTConfig(**checkpoint['config'])
    model = GPT(pretrained_config)
    
    # Load pretrained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    if rank == 0:
        print("\nLoaded pretrained model with config:")
        for key, value in pretrained_config.__dict__.items():
            print(f"{key}: {value}")
    
    # Create optimizer with fine-tuning specific parameters
    optimizer = bnb.optim.AdamW(
        model.parameters(),
        lr=2e-5,  # Lower learning rate for fine-tuning
        weight_decay=0.1,
        optim_bits=8,
        block_wise=True,
        is_paged=True
    )
    
    # Create scheduler with shorter warmup and decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=1000,  # Shorter cycle length for fine-tuning
        T_mult=2,
        eta_min=1e-6
    )
    
    return model, optimizer, scheduler

# Example usage:
if __name__ == "__main__":
    args = get_args()
    torch.backends.cudnn.benchmark = True 

    # parser.add_argument("--pretrained_path", type=str, required=True,
    #                    help="Path to pretrained model checkpoint")
    # parser.add_argument("--instruction_data", type=str, required=True,
    #                    help="Path to instruction dataset JSON files")
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if local_rank is not None else 'cuda')
    
    # Get vocab size and create tokenizer
    vocab_size = get_vocab_size("gpt2")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create config
    config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=args.block_size,
        dropout=0.1
    )
    
    # Initialize wandb only on rank 0
    if rank == 0:
        wandb.init(
            project="gpt-training",
            config={
                "vocab_size": vocab_size,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "block_size": config.block_size,
                "dropout": config.dropout,
                "learning_rate": args.lr,
                "batch_size": args.batch_size * (world_size or 1),  # Global batch size
                "optimizer": "8-bit AdamW",
                "num_gpus": world_size or 1
            }
        )
    
    # Create model
    model = GPT(config)
    model = model.to(device)

    if rank == 0:  # Only print on main process
        param_counts = count_parameters(model)
        print("\nModel Parameter Counts:")
        print(f"Total Parameters: {param_counts['total']:,}")
        print(f"- Embeddings: {param_counts['embeddings']:,}")
        print(f"- Transformer Layers: {param_counts['transformer_layers']:,}")
        print(f"- Final Layer Norm: {param_counts['final_ln']:,}")
        print(f"- Output Head: {param_counts['head']:,}\n")
    
    # Wrap model with DDP
    if world_size is not None:
        model = DDP(model, device_ids=[local_rank])
    
    # Create distributed dataloaders
    train_loader, val_loader, train_sampler, val_sampler = create_distributed_dataloaders(
        args.train_data,
        block_size=args.block_size,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor
    )
    
    # Create optimizer and scheduler
    optimizer = bnb.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        optim_bits=8,      # Enable 8-bit optimization
        block_wise=True,   # Enable block-wise quantization
        is_paged=True      # Enable memory paging
    )
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader) * args.epochs
    )
    scaler = GradScaler()
    
    checkpoint_dir = Path("checkpoints")
    if rank == 0:  # Only create directory on main process
        checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, train_sampler, optimizer, scheduler, 
            scaler, device, epoch, tokenizer=tokenizer,
            gen_every_n_steps=args.gen_every_n_steps,
            sample_prompts=args.sample_prompts,
            rank=rank,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        
        # Evaluate
        val_loss = evaluate(model, val_loader, val_sampler, device, rank)
        
        # Log metrics (only on rank 0)
        if rank == 0:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
            })
            
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, train_loss, val_loss, config,
                checkpoint_dir / "best_model.pt",
                rank=rank
            )
        
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, train_loss, val_loss, config,
                checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                rank=rank
            )
    
    if rank == 0:
        wandb.finish()
    
    # Cleanup distributed training
    if dist.is_initialized():
        dist.destroy_process_group()

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']