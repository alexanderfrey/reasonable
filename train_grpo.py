# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import tiktoken
import torch
import os
from typing import List, Tuple, Optional

from strategies import RewardFunctions
from model import GPTConfig, GPT


def get_config_from_checkpoint(checkpoint_path: str) -> Tuple[GPTConfig, int]:
    """
    Infer config parameters from checkpoint state dict, including auxiliary heads configuration.

    Args:
        checkpoint_path: Path to the model checkpoint

    Returns:
        tuple: (GPTConfig, number of experts)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Count number of blocks
    n_layer = (
        max(
            [int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("blocks.")]
        )
        + 1
    )

    # Check if bias is used
    has_bias = any(["bias" in k for k in state_dict.keys()])

    # Get accurate dimensions from latent queries
    latent_queries = state_dict["blocks.0.attn.latent_queries"]
    n_latents = latent_queries.shape[1]  # 64
    n_head = latent_queries.shape[2]  # 6
    head_dim = latent_queries.shape[3]  # 128
    n_embd = n_head * head_dim  # 768

    # Get vocab size from token embedding
    vocab_size = state_dict["token_embedding.weight"].shape[0]

    # Count number of auxiliary heads
    aux_head_keys = [k for k in state_dict.keys() if k.startswith("aux_heads.")]
    n_aux_heads = len(
        set(
            [
                int(k.split(".")[1])
                for k in aux_head_keys
                if k.split(".")[-1] == "weight"
            ]
        )
    )

    # Determine number of experts
    expert_keys = [k for k in state_dict.keys() if "experts" in k]
    n_experts = (
        len(set([int(k.split("experts.")[1].split(".")[0]) for k in expert_keys]))
        if expert_keys
        else 0
    )

    # Create config with auxiliary heads
    config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_aux_heads=n_aux_heads,  # Add number of auxiliary heads
        bias=has_bias,
        block_size=1024,  # You might want to make this configurable
    )

    return config, n_experts


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    Load checkpoint in the same format as it was saved
    """
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load model weights
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer and scheduler states if they exist
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]

        print(f"Successfully loaded checkpoint from epoch {epoch}, step {global_step}")
        return model, optimizer, scheduler, epoch, global_step

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, None, 0, 0


def train_with_grpo(
    checkpoint_path: str,
    dataset_name: str = "imdb",
    output_dir: str = "gpt-grpo",
    learning_rate: float = 1e-5,
    batch_size: int = 8,
    num_epochs: int = 3,
    logging_steps: int = 10,
):

    system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: prompt. Assistant:"""
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")

    # Initialize model
    config, n_experts = get_config_from_checkpoint(checkpoint_path)
    base_model = GPT(config)

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        base_model.load_state_dict(checkpoint["model_state_dict"])

    # Training config
    training_args = GRPOConfig(
        output_dir=output_dir,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_steps=len(dataset) * num_epochs // batch_size,
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=base_model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint_latest.pt"
    )
    parser.add_argument("--dataset", default="imdb")
    parser.add_argument("--output_dir", default="gpt-grpo")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=10)

    args = parser.parse_args()

    train_with_grpo(
        checkpoint_path=args.checkpoint_path,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        logging_steps=args.logging_steps,
    )
