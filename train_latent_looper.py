import os, random, math
import glob
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import GPT2Tokenizer
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import wandb
from torch.nn.utils import clip_grad_norm_
from model import GPT
from latent_loop_model import GPTConfig, LatentLoopTransformer, LatentLoopLoss
from utils import (
    generate_text,
    count_parameters,
    load_checkpoint,
    prepare_training_data_including_thoughts,
    create_dataloader,
    fetch_training_data,
)


import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb  # Optional: for logging
from torch.nn.utils import clip_grad_norm_


def train_with_batches(
    model,
    train_loader,
    val_loader,
    vocab_size,
    device,
    tokenizer,
    optimizer,
    scheduler,
    *,
    epochs=1,
    max_steps=None,
    generate_every=100,
    generate_prompt="My name",
    max_gen_tokens=20,
    start_epoch=0,
    save_path="best_model.pth",
    use_wandb=False,
    gradient_accumulation_steps=1,
    early_stopping_patience=3,
    max_grad_norm=1.0,
):
    """
    Train the LatentLoopTransformer model with improved training loop and error handling.

    Args:
        model: The LatentLoopTransformer model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        vocab_size: Size of the vocabulary
        device: Device to train on
        tokenizer: Tokenizer for text generation
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        epochs: Number of training epochs
        max_steps: Maximum number of training steps (optional)
        generate_every: Generate sample text every N steps
        generate_prompt: Prompt to use for generation
        max_gen_tokens: Maximum tokens to generate
        start_epoch: Starting epoch number
        save_path: Path to save best model
        use_wandb: Whether to use Weights & Biases logging
        gradient_accumulation_steps: Number of steps to accumulate gradients
        early_stopping_patience: Number of epochs to wait before early stopping
        max_grad_norm: Maximum gradient norm for clipping
    """
    criterion = LatentLoopLoss(vocab_size=vocab_size).to(device)
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    # Setup automatic mixed precision training
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Initialize metrics dictionary for tracking across epochs
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "train_metrics": [],
        "val_metrics": [],
        "epochs": [],
    }

    try:
        for epoch in range(start_epoch, start_epoch + epochs):
            # Training Phase
            model.train()
            train_loss = 0.0
            train_metrics = {
                "generation_loss": 0.0,
                "latent_loss": 0.0,
                "consistency_loss": 0.0,
                "kl_loss": 0.0,
            }

            train_progress_bar = tqdm(
                train_loader, desc=f"Training Epoch {epoch + 1}", leave=False
            )

            optimizer.zero_grad()  # Zero gradients at start of epoch

            for batch_idx, batch in enumerate(train_progress_bar):
                if max_steps is not None and global_step >= max_steps:
                    print(f"Reached {max_steps} steps, stopping training.")
                    break

                try:
                    # Move batch to device
                    question = batch["question"].to(device)
                    context = batch["context"].to(device)
                    answer = batch["answer"].to(device)
                    intermediate_answers = batch.get("intermediate_answers")
                    if intermediate_answers is not None:
                        intermediate_answers = intermediate_answers.to(device)

                    # Forward pass with automatic mixed precision
                    with autocast(enabled=use_amp):
                        # Combine input
                        input_ids = torch.cat([question, context], dim=1)

                        # Forward pass through model
                        output = model(input_ids)

                        # Prepare targets
                        targets = {
                            "answer": answer,
                            "intermediate_states": intermediate_answers,
                        }

                        # Calculate loss
                        loss_dict = criterion(output, targets)
                        total_loss = loss_dict["total_loss"]
                        total_loss = (
                            total_loss / gradient_accumulation_steps
                        )  # Scale loss

                    # Backward pass with gradient accumulation
                    scaler.scale(total_loss).backward()

                    # Update metrics
                    train_loss += total_loss.item() * gradient_accumulation_steps
                    for key in train_metrics:
                        train_metrics[key] += loss_dict[key]

                    # Step optimizer after accumulating gradients
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Clip gradients
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), max_grad_norm)

                        # Optimizer step
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        global_step += 1

                        # Update progress bar
                        train_progress_bar.set_postfix(
                            total_loss=total_loss.item() * gradient_accumulation_steps,
                            gen_loss=loss_dict["generation_loss"],
                            latent_loss=loss_dict["latent_loss"],
                        )

                        # Generate sample text periodically
                        if generate_every and global_step % generate_every == 0:
                            generate_sample_text(
                                model,
                                tokenizer,
                                generate_prompt,
                                max_gen_tokens,
                                device,
                                global_step,
                            )

                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    continue

            # Calculate average training metrics
            num_batches = len(train_loader)
            train_loss /= num_batches
            for key in train_metrics:
                train_metrics[key] /= num_batches

            # Validation Phase
            val_loss, val_metrics = validate_model(
                model, val_loader, criterion, device, use_amp
            )

            # Update learning rate scheduler
            scheduler.step()

            # Log metrics
            if use_wandb:
                log_metrics(
                    train_loss, val_loss, train_metrics, val_metrics, global_step, epoch
                )

            # Store metrics in training history
            training_history["train_loss"].append(train_loss)
            training_history["val_loss"].append(val_loss)
            training_history["train_metrics"].append(train_metrics.copy())
            training_history["val_metrics"].append(val_metrics.copy())
            training_history["epochs"].append(epoch)

            # Print epoch summary
            print_epoch_summary(
                epoch, global_step, train_loss, val_loss, train_metrics, val_metrics
            )

            # Save best model and check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_loss,
                    global_step,
                    save_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {str(e)}")
        raise e
    finally:
        # Final cleanup and metric saving
        if use_wandb:
            wandb.finish()

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_val_loss": best_val_loss,
        "global_step": global_step,
        "training_history": training_history,  # Add complete training history
    }


def generate_sample_text(model, tokenizer, prompt, max_tokens, device, step):
    """Generate and print sample text from the model"""
    print(f"\n[Step {step}] Generating text for prompt: {prompt}")
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Assuming model has a generate method
        try:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                num_beams=4,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_p=0.9,
            )
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"[Step {step}] Generated text:\n{generated_text}\n")
        except Exception as e:
            print(f"Generation failed: {str(e)}")
    model.train()


def validate_model(model, val_loader, criterion, device, use_amp):
    """Run validation loop and return metrics"""
    model.eval()
    val_loss = 0.0
    val_metrics = {
        "generation_loss": 0.0,
        "latent_loss": 0.0,
        "consistency_loss": 0.0,
        "kl_loss": 0.0,
    }

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            question = batch["question"].to(device)
            context = batch["context"].to(device)
            answer = batch["answer"].to(device)
            intermediate_answers = batch.get("intermediate_answers")
            if intermediate_answers is not None:
                intermediate_answers = intermediate_answers.to(device)

            with autocast(enabled=use_amp):
                input_ids = torch.cat([question, context], dim=1)
                output = model(input_ids)
                targets = {
                    "answer": answer,
                    "intermediate_states": intermediate_answers,
                }
                loss_dict = criterion(output, targets)

            val_loss += loss_dict["total_loss"].item()
            for key in val_metrics:
                val_metrics[key] += loss_dict[key]

    # Calculate averages
    num_batches = len(val_loader)
    val_loss /= num_batches
    for key in val_metrics:
        val_metrics[key] /= num_batches

    return val_loss, val_metrics


def save_checkpoint(
    model, optimizer, scheduler, epoch, best_val_loss, global_step, save_path
):
    """Save model checkpoint"""
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "global_step": global_step,
        },
        save_path,
    )
    print(f"Model saved to {save_path}")


def log_metrics(train_loss, val_loss, train_metrics, val_metrics, global_step, epoch):
    """Log metrics to wandb"""
    wandb.log(
        {
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/generation_loss": train_metrics["generation_loss"],
            "train/latent_loss": train_metrics["latent_loss"],
            "train/consistency_loss": train_metrics["consistency_loss"],
            "train/kl_loss": train_metrics["kl_loss"],
            "val/generation_loss": val_metrics["generation_loss"],
            "val/latent_loss": val_metrics["latent_loss"],
            "val/consistency_loss": val_metrics["consistency_loss"],
            "val/kl_loss": val_metrics["kl_loss"],
            "epoch": epoch,
            "global_step": global_step,
        }
    )


def print_epoch_summary(
    epoch, global_step, train_loss, val_loss, train_metrics, val_metrics
):
    """Print summary of epoch metrics"""
    print(f"\nEpoch {epoch + 1} (Step {global_step}) Summary:")
    print(f"Training - Total Loss: {train_loss:.4f}")
    print("Training Metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Validation - Total Loss: {val_loss:.4f}")
    print("Validation Metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Train GPT-3 model with CLI parameters."
    )
    parser.add_argument(
        "--text_files_directory",
        type=str,
        default="./text_files",
        help="Path to the text files directory",
    )
    parser.add_argument(
        "--block_size", type=int, default=2048, help="Block size for training sequences"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of training steps"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="trained_model.pth",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--model_load_path",
        type=str,
        default=None,
        help="Path to a pre-trained model file to continue training",
    )
    parser.add_argument(
        "--prompt", type=str, default="My name is", help="Prompt for text generation"
    )
    args = parser.parse_args()

    from tokenizers import ByteLevelBPETokenizer

    # Load the tokenizer
    tokenizer = ByteLevelBPETokenizer(
        "./byte_level_bpe/vocab.json", "./byte_level_bpe/merges.txt"
    )

    # Ensure special tokens are added
    special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            print(f"Error: Token {token} was not added to the vocabulary.")
        else:
            print(f"Token {token} has ID: {token_id}")

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    # Load and preprocess text data
    raw_data_df = fetch_training_data(
        task_descriptions="reasoning_over_news",
        condition="model_version='gemini-2.0-flash-thinking-exp'",
    )

    processed_data = prepare_training_data_including_thoughts(
        raw_data_df, tokenizer, block_size=args.block_size
    )

    block_size = args.block_size

    train_data, val_data = train_test_split(
        processed_data, test_size=0.25, random_state=42
    )

    # Create DataLoaders for training and validation
    train_loader = create_dataloader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    val_loader = create_dataloader(val_data, batch_size=args.batch_size, shuffle=False)

    # Model configuration
    vocab_size = tokenizer.get_vocab_size()
    latent_model_args = dict(
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=block_size,
        bias=False,
        vocab_size=vocab_size,
        dropout=0.1,
    )
    language_model_args = dict(
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=block_size,
        bias=False,
        vocab_size=vocab_size,
        dropout=0.1,
    )

    latent_gpt = GPT(GPTConfig(**latent_model_args)).to(device)
    language_gpt = GPT(GPTConfig(**language_model_args)).to(device)

    # Combine into LatentLoopTransformer
    model = LatentLoopTransformer(latent_gpt, language_gpt, max_loops=10).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs
    )

    # Load model checkpoint if provided
    start_epoch = 0
    if args.model_load_path:
        try:
            start_epoch = load_checkpoint(
                args.model_load_path, model, optimizer, scheduler
            )
        except Exception as e:
            print(f"Failed to load checkpoint from {args.model_load_path}: {e}")
            exit(1)
    else:
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)

    print("Model created.")
    print(
        "Generated text:",
        generate_text(model, tokenizer, device, args.prompt, max_new_tokens=120),
    )
    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params}")

    # Train the model
    results = train_with_batches(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        device=device,
        epochs=args.epochs,
        max_steps=args.steps,
        optimizer=optimizer,
        scheduler=scheduler,
        sparsity_alpha=1e-5,
        start_epoch=start_epoch,
        save_path=args.model_save_path,
        generate_every=100,  # Generate text every 100 training steps
        generate_prompt="The house was built on",
        max_gen_tokens=100,
    )

    # history = results["training_history"]

    # Generate final text
    print(
        "Generated text:",
        generate_text(model, tokenizer, device, args.prompt, max_new_tokens=120),
    )
