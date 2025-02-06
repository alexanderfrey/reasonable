# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import tiktoken
import torch
import os
from strategies import RewardFunctions
from model import GPTConfig, GPT


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    Load checkpoint in the same format as it was saved
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model weights
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states if they exist
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        
        print(f"Successfully loaded checkpoint from epoch {epoch}, step {global_step}")
        return model, optimizer, scheduler, epoch, global_step
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, None, 0, 0


def create_gpt_config(checkpoint_path):
    """Extract config from checkpoint or return default config"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        if 'config' in checkpoint:
            return checkpoint['config']
    
    # Default config if not found in checkpoint
    return GPTConfig(
        vocab_size=50257,  # GPT-2 tokenizer vocab size
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=1024,
        bias=False,
        dropout=0.0,
        rope_theta=10000.0,
        n_latents=64,
        n_aux_heads=0,
    )

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# First create config and load the model
checkpoint_path = "path/to/your/checkpoint.pt"
config = create_gpt_config(checkpoint_path)

# Initialize base model with config
base_model = GPT(config)

# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model state
    if hasattr(base_model, "module"):
        base_model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        base_model.load_state_dict(checkpoint['model_state_dict'])
    
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    print(f"Successfully loaded checkpoint from epoch {start_epoch}, step {global_step}")
