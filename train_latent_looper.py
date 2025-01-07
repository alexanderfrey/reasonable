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
from model import GPT, GPTConfig
from latent_loop_model import LatentLoopTransformer, LatentLoopLoss
from utils import (
    generate_text,
    count_parameters,
    load_checkpoint,
    prepare_training_data_including_thoughts,
    create_data_loader,
    fetch_training_data,
)
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Add this before starting training
if torch.cuda.is_available():
    # Reset GPU state
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

    # Print GPU info
    print("\nGPU Information:")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    # Try a small tensor transfer test
    try:
        test_tensor = torch.ones(1, device="cuda")
        print("GPU test transfer successful")
    except Exception as e:
        print(f"GPU test transfer failed: {e}")


def test_gpu_operations():
    print("\nRunning GPU diagnostics...")

    try:
        # Test 1: Small tensor transfer
        print("\nTest 1: Basic tensor transfer")
        cpu_tensor = torch.arange(10, dtype=torch.long)
        gpu_tensor = cpu_tensor.to("cuda")
        print("Basic tensor transfer successful")

        # Test 2: Tensor with same shape as our data
        print("\nTest 2: Testing with data-like shapes")
        sample_question = torch.zeros((1, 1024), dtype=torch.long)
        sample_context = torch.zeros((1, 2048), dtype=torch.long)
        sample_answer = torch.zeros((1, 2048), dtype=torch.long)
        sample_intermediate = torch.zeros((1, 5, 512), dtype=torch.long)

        # Transfer each
        gpu_question = sample_question.to("cuda")
        print("Question transfer OK")
        gpu_context = sample_context.to("cuda")
        print("Context transfer OK")
        gpu_answer = sample_answer.to("cuda")
        print("Answer transfer OK")
        gpu_intermediate = sample_intermediate.to("cuda")
        print("Intermediate transfer OK")

        # Test 3: Concatenation
        print("\nTest 3: Testing concatenation")
        cat_result = torch.cat([gpu_question, gpu_context], dim=1)
        print(f"Concatenation successful, shape: {cat_result.shape}")

        # Test 4: Memory cleanup
        print("\nTest 4: Testing memory cleanup")
        del gpu_question, gpu_context, gpu_answer, gpu_intermediate, cat_result
        torch.cuda.empty_cache()
        print("Memory cleanup successful")

        print("\nAll GPU diagnostic tests passed!")
        return True

    except Exception as e:
        print(f"\nGPU diagnostic failed: {str(e)}")
        return False


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
                    # Move batch to device and print shapes
                    # print(f"\nBatch {batch_idx} shapes before device transfer:")
                    # print(f"Question shape: {batch['question'].shape}")
                    # print(f"Context shape: {batch['context'].shape}")
                    # print(f"Answer shape: {batch['answer'].shape}")
                    # print(
                    #     f"Intermediate answers shape: {batch['intermediate_answers'].shape}"
                    # )

                    question = batch["question"].to(device)
                    context = batch["context"].to(device)
                    answer = batch["answer"].to(device)
                    intermediate_answers = batch.get("intermediate_answers")
                    if intermediate_answers is not None:
                        intermediate_answers = intermediate_answers.to(device)

                    # Print shapes after device transfer
                    # print(f"\nShapes after device transfer:")
                    # print(f"Question shape: {question.shape}")
                    # print(f"Context shape: {context.shape}")
                    # print(
                    #     f"Input concatenated shape: {torch.cat([question, context], dim=1).shape}"
                    # )

                    # Forward pass with automatic mixed precision
                    with autocast(enabled=use_amp):
                        # Validate input dimensions before concatenation
                        if len(question.shape) != 2:
                            question = question.unsqueeze(0)
                        if len(context.shape) != 2:
                            context = context.unsqueeze(0)

                        # Combine input with shape validation
                        input_ids = torch.cat([question, context], dim=1)
                        # print(f"Final input_ids shape: {input_ids.shape}")

                        # Forward pass through model
                        output = model(input_ids)
                        # print(
                        #     f"Model output shape: {output.shape if isinstance(output, torch.Tensor) else 'dict'}"
                        # )

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
                        # if generate_every and global_step % generate_every == 0:
                        #     generate_sample_text(
                        #         model,
                        #         tokenizer,
                        #         generate_prompt,
                        #         max_gen_tokens,
                        #         device,
                        #         global_step,
                        #     )

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
    model.eval()
    val_loss = 0.0
    val_metrics = {
        "generation_loss": 0.0,
        "latent_loss": 0.0,
        "consistency_loss": 0.0,
        "kl_loss": 0.0,
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(val_loader, desc="Validation", leave=False)
        ):
            try:
                # print(f"\nValidating batch {batch_idx}")

                # 1. First verify all tensors before any device operations
                for key, tensor in batch.items():
                    # print(
                    #     f"{key} pre-transfer - shape: {tensor.shape}, "
                    #     f"dtype: {tensor.dtype}, device: {tensor.device}"
                    # )

                    # Ensure we're working with CPU tensors initially
                    if tensor.device.type != "cpu":
                        batch[key] = tensor.cpu()

                    # Ensure correct dtype
                    if tensor.dtype != torch.long:
                        batch[key] = tensor.to(torch.long)

                # 2. Create new tensors on device instead of transferring
                device_batch = {
                    key: torch.empty(tensor.shape, dtype=torch.long, device=device)
                    for key, tensor in batch.items()
                }

                # 3. Copy data to new device tensors
                for key, tensor in batch.items():
                    try:
                        device_batch[key].copy_(tensor)
                        # print(
                        #     f"{key} post-transfer - shape: {device_batch[key].shape}, "
                        #     f"device: {device_batch[key].device}"
                        # )
                    except Exception as e:
                        print(f"Error copying {key}: {str(e)}")
                        raise

                # 4. Extract individual tensors
                question = device_batch["question"]
                context = device_batch["context"]
                answer = device_batch["answer"]
                intermediate_answers = device_batch.get("intermediate_answers")

                # 5. Prepare input
                try:
                    input_ids = torch.cat([question, context], dim=1)
                    # print(f"Combined input shape: {input_ids.shape}")
                except Exception as e:
                    print("Error in concatenation:", str(e))
                    raise

                # 6. Model forward pass
                with autocast(enabled=use_amp):
                    try:
                        output = model(input_ids)
                        targets = {
                            "answer": answer,
                            "intermediate_states": intermediate_answers,
                        }
                        loss_dict = criterion(output, targets)
                    except Exception as e:
                        print("Error in forward pass:", str(e))
                        raise

                val_loss += loss_dict["total_loss"].item()
                for key in val_metrics:
                    val_metrics[key] += loss_dict[key]

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise

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


def validate_dataset(data_items):
    shapes = []
    for item in data_items:
        shape_dict = {
            "question": item["question"].shape,
            "context": item["context"].shape,
            "answer": item["answer"].shape,
            "intermediate_answers": item["intermediate_answers"].shape,
        }
        shapes.append(shape_dict)

        # Validate indices are within bounds
        for key, tensor in item.items():
            if torch.any(tensor >= tokenizer.vocab_size):
                print(
                    f"Found out of bounds token in {key}: {torch.max(tensor)} >= {tokenizer.vocab_size}"
                )

    # Check if all shapes are consistent
    first_shape = shapes[0]
    inconsistent = [i for i, s in enumerate(shapes) if s != first_shape]
    if inconsistent:
        print(f"Inconsistent shapes found at indices: {inconsistent}")
        print(f"Expected shapes: {first_shape}")
        print(f"Found shapes: {[shapes[i] for i in inconsistent]}")

    return len(inconsistent) == 0


def validate_token_indices(data_items, vocab_size):
    for idx, item in enumerate(data_items):
        for key, tensor in item.items():
            max_idx = torch.max(tensor).item()
            if max_idx >= vocab_size:
                print(
                    f"Found invalid token index in item {idx}, key {key}: {max_idx} >= {vocab_size}"
                )
                # Print a few examples of large indices
                large_indices = (tensor >= vocab_size).nonzero()
                if len(large_indices) > 0:
                    print(f"First few invalid indices: {large_indices[:5]}")
                    print(f"Corresponding values: {tensor[large_indices[:5]]}")
                return False
    return True


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

    from transformers import AutoTokenizer

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Set pad token to eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Ensure special tokens are added
    # special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
    # for token in special_tokens:
    #     token_id = tokenizer.token_to_id(token)
    #     if token_id is None:
    #         print(f"Error: Token {token} was not added to the vocabulary.")
    #     else:
    #         print(f"Token {token} has ID: {token_id}")

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    block_size = args.block_size

    # Load and preprocess text data
    raw_data_df = fetch_training_data(
        task_descriptions=["reasoning_over_news"],
        condition="model_version='gemini-2.0-flash-thinking-exp'",
        limit=100,
    )

    print(raw_data_df)

    processed_data = prepare_training_data_including_thoughts(
        raw_data_df, block_size, tokenizer
    )

    if not validate_dataset(processed_data):
        raise ValueError("Dataset validation failed!")

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(
        f"Max token ID in data: {max(max(batch['question'].max(), batch['context'].max(), batch['answer'].max()) for batch in processed_data)}"
    )

    train_data, val_data = train_test_split(
        processed_data, test_size=0.25, random_state=42
    )

    if not validate_token_indices(processed_data, tokenizer.vocab_size):
        raise ValueError("Found token indices larger than vocabulary size!")

    # Create DataLoaders for training and validation
    train_loader = create_data_loader(
        train_data, args.batch_size, tokenizer, shuffle=True
    )
    val_loader = create_data_loader(val_data, args.batch_size, tokenizer, shuffle=False)

    if not test_gpu_operations():
        raise RuntimeError("GPU diagnostics failed, cannot proceed with training")

    # If diagnostics pass, try a single batch test
    print("\nTesting with first batch...")
    try:
        # Get first batch
        first_batch = next(iter(train_loader))

        # Try operations on first batch
        print("Moving first batch to GPU...")
        question = first_batch["question"][:1].to("cuda")  # Just take first item
        context = first_batch["context"][:1].to("cuda")
        answer = first_batch["answer"][:1].to("cuda")
        intermediate = first_batch["intermediate_answers"][:1].to("cuda")

        print("First batch transfer successful")
        print(f"Question shape: {question.shape}")
        print(f"Context shape: {context.shape}")
        print(f"Answer shape: {answer.shape}")
        print(f"Intermediate shape: {intermediate.shape}")

        # Clean up
        del question, context, answer, intermediate
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"First batch test failed: {str(e)}")
        raise

    # Model configuration
    vocab_size = tokenizer.vocab_size
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
    model = LatentLoopTransformer(latent_gpt, language_gpt, max_loops=5).to(device)

    print("Model Configuration:")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Model embedding size: {model.latent_gpt.token_embedding.weight.size(0)}")
    print(f"Block size: {block_size}")
    print(
        f"Position embedding capacity: {model.latent_gpt.position_embedding.weight.shape}"
    )

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
    # print(
    #     "Generated text:",
    #     generate_text(model, tokenizer, device, args.prompt, max_new_tokens=120),
    # )
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
