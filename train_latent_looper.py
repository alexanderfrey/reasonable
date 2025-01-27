import os, random, math
import glob
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.amp import autocast as autocast_ctx, GradScaler
import bitsandbytes as bnb
import numpy as np
import wandb
from collections import defaultdict
from torch.nn.utils import clip_grad_norm_
from model import GPT, GPTConfig
from latent_loop_model import LatentReasoningTransformer, LatentLoopLoss
from utils import (
    generate_text,
    count_parameters,
    load_checkpoint,
    prepare_training_data_including_thoughts,
    create_data_loader,
    fetch_training_data,
    test_gpu_operations,
    validate_dataset,
    print_epoch_summary,
    ThoughtProcessor,
)


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


def train_with_batches(
    model,
    train_loader,
    val_loader,
    vocab_size,
    embedding_dim,
    latent_dim,
    device,
    tokenizer,
    optimizer,
    scheduler,
    *,
    epochs=1,
    probe_epochs=25,
    probe_lr=1e-4,
    max_steps=None,
    generate_every=100,
    generate_prompt="My name",
    max_gen_tokens=20,
    start_epoch=0,
    save_path="best_model.pth",
    probe_save_path="best_probe.pth",
    use_wandb=False,
    gradient_accumulation_steps=1,
    early_stopping_patience=3,
    max_grad_norm=1.0,
):
    print("\nPhase 1: Training Probe")
    probe_optimizer = torch.optim.AdamW(model.reasoning_probe.parameters(), lr=probe_lr)
    best_probe_val_loss = float("inf")

    # Train probe
    for epoch in range(probe_epochs):
        print(f"\nProbe Epoch {epoch + 1}/{probe_epochs}")
        model.train()
        probe_train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, dict):
                question = batch["question"].to(device)
                context = batch["context"].to(device)
                thoughts = batch["intermediate_answers"].to(device)
            else:
                question, context, _, thoughts = [b.to(device) for b in batch]

            # Process batch for probe
            probe_result = train_probe_batch(
                model=model,
                batch={
                    "question": question,
                    "context": context,
                    "intermediate_answers": thoughts,
                },
                probe_optimizer=probe_optimizer,
                device=device,
                tokenizer=tokenizer,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )

            if probe_result is not None:
                probe_train_loss += probe_result["probe_loss"]

        probe_train_loss /= len(train_loader)

        # Validate probe
        model.eval()
        probe_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    question = batch["question"].to(device)
                    context = batch["context"].to(device)
                    thoughts = batch["intermediate_answers"].to(device)
                else:
                    question, context, _, thoughts = [b.to(device) for b in batch]

                val_result = validate_probe_batch(
                    model=model,
                    batch={
                        "question": question,
                        "context": context,
                        "intermediate_answers": thoughts,
                    },
                    device=device,
                    tokenizer=tokenizer,
                )
                if val_result is not None:
                    probe_val_loss += val_result["probe_loss"]

        probe_val_loss /= len(val_loader)
        print(f"Probe Train Loss: {probe_train_loss:.4f}")
        print(f"Probe Val Loss: {probe_val_loss:.4f}")

        if probe_val_loss < best_probe_val_loss:
            best_probe_val_loss = probe_val_loss
            torch.save(
                {
                    "probe_state_dict": model.reasoning_probe.state_dict(),
                    "epoch": epoch,
                    "val_loss": probe_val_loss,
                },
                probe_save_path,
            )
            print(f"Saved new best probe model with val loss: {probe_val_loss:.4f}")

    # Load best probe and freeze
    print("\nLoading best probe model...")
    checkpoint = torch.load(probe_save_path)
    model.reasoning_probe.load_state_dict(checkpoint["probe_state_dict"])
    print("Freezing probe parameters...")
    for param in model.reasoning_probe.parameters():
        param.requires_grad = False

    print("\nPhase 2: Training Main Model")
    criterion = LatentLoopLoss(
        vocab_size=vocab_size, embedding_dim=embedding_dim, latent_dim=latent_dim
    ).to(device)

    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    training_history = {
        "train_loss": [],
        "val_loss": [],
        "train_metrics": [],
        "val_metrics": [],
        "per_loop_losses": [],
        "epochs": [],
    }

    try:
        for epoch in range(start_epoch, start_epoch + epochs):
            model.train()
            train_loss = 0.0
            train_metrics = {
                "generation_loss": 0.0,
                "latent_loss": 0.0,
                "per_loop_latent_losses": [0.0] * model.max_loops,
            }

            train_progress_bar = tqdm(
                train_loader, desc=f"Training Epoch {epoch + 1}", leave=False
            )

            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_progress_bar):
                if max_steps is not None and global_step >= max_steps:
                    break

                try:
                    if isinstance(batch, dict):
                        batch_loss, batch_metrics = process_batch(
                            batch=batch,
                            model=model,
                            device=device,
                            tokenizer=tokenizer,
                            is_training=True,
                            use_amp=use_amp,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            scaler=scaler,
                            criterion=criterion,
                        )
                    else:
                        question, context, answer, thoughts = [
                            b.to(device) for b in batch
                        ]
                        batch_dict = {
                            "question": question,
                            "context": context,
                            "answer": answer,
                            "intermediate_answers": thoughts,
                        }
                        batch_loss, batch_metrics = process_batch(
                            batch=batch_dict,
                            model=model,
                            device=device,
                            tokenizer=tokenizer,
                            is_training=True,
                            use_amp=use_amp,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            scaler=scaler,
                            criterion=criterion,
                        )

                    # Update metrics
                    train_loss += batch_loss
                    train_metrics["generation_loss"] += batch_metrics["generation_loss"]
                    train_metrics["latent_loss"] += batch_metrics["latent_loss"]
                    for i, loop_loss in enumerate(
                        batch_metrics["per_loop_latent_losses"]
                    ):
                        train_metrics["per_loop_latent_losses"][i] += loop_loss

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        global_step += 1

                        train_progress_bar.set_postfix(
                            total_loss=batch_loss,
                            gen_loss=batch_metrics["generation_loss"],
                            latent_loss=batch_metrics["latent_loss"],
                        )

                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    continue

            # Calculate averages and validate
            num_batches = len(train_loader)
            train_loss /= num_batches
            for key in train_metrics:
                if key == "per_loop_latent_losses":
                    train_metrics[key] = [
                        loss / num_batches for loss in train_metrics[key]
                    ]
                else:
                    train_metrics[key] /= num_batches

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_metrics = {
                "generation_loss": 0.0,
                "latent_loss": 0.0,
                "per_loop_latent_losses": [0.0] * model.max_loops,
            }

            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc="Validation", leave=False):
                    if isinstance(val_batch, dict):
                        batch_loss, batch_metrics = process_batch(
                            batch=val_batch,
                            model=model,
                            device=device,
                            tokenizer=tokenizer,
                            is_training=False,
                            use_amp=use_amp,
                            criterion=criterion,
                        )
                    else:
                        question, context, answer, thoughts = [
                            b.to(device) for b in val_batch
                        ]
                        val_batch_dict = {
                            "question": question,
                            "context": context,
                            "answer": answer,
                            "intermediate_answers": thoughts,
                        }
                        batch_loss, batch_metrics = process_batch(
                            batch=val_batch_dict,
                            model=model,
                            device=device,
                            tokenizer=tokenizer,
                            is_training=False,
                            use_amp=use_amp,
                            criterion=criterion,
                        )

                    val_loss += batch_loss
                    val_metrics["generation_loss"] += batch_metrics["generation_loss"]
                    val_metrics["latent_loss"] += batch_metrics["latent_loss"]
                    for i, loop_loss in enumerate(
                        batch_metrics["per_loop_latent_losses"]
                    ):
                        val_metrics["per_loop_latent_losses"][i] += loop_loss

            num_val_batches = len(val_loader)
            val_loss /= num_val_batches
            for key in val_metrics:
                if key == "per_loop_latent_losses":
                    val_metrics[key] = [
                        loss / num_val_batches for loss in val_metrics[key]
                    ]
                else:
                    val_metrics[key] /= num_val_batches

            scheduler.step()

            if use_wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_generation_loss": train_metrics["generation_loss"],
                        "train_latent_loss": train_metrics["latent_loss"],
                        **{
                            f"train_loop_{i}_loss": loss
                            for i, loss in enumerate(
                                train_metrics["per_loop_latent_losses"]
                            )
                        },
                        "epoch": epoch,
                        "global_step": global_step,
                    }
                )

            # Update training history
            training_history["train_loss"].append(train_loss)
            training_history["val_loss"].append(val_loss)
            training_history["train_metrics"].append(train_metrics.copy())
            training_history["val_metrics"].append(val_metrics.copy())
            training_history["per_loop_losses"].append(
                train_metrics["per_loop_latent_losses"]
            )
            training_history["epochs"].append(epoch)

            print_epoch_summary(
                epoch, global_step, train_loss, val_loss, train_metrics, val_metrics
            )

            # Save best model and check early stopping
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
                    training_history=training_history,
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
        if use_wandb:
            wandb.finish()

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_val_loss": best_val_loss,
        "global_step": global_step,
        "training_history": training_history,
        "per_loop_losses": training_history["per_loop_losses"],
    }


def validate_probe(model, val_loader, device, tokenizer):
    model.eval()
    total_loss = 0
    num_batches = 0

    # Enable gradient checkpointing for validation too
    torch.cuda.empty_cache()

    for batch in tqdm(val_loader, desc="Validating probe"):
        try:
            with torch.cuda.amp.autocast():
                question = batch["question"].to(device)
                context = batch["context"].to(device)
                teacher_steps = batch["intermediate_answers"].to(device)

                if len(question.shape) != 2:
                    question = question.unsqueeze(0)
                if len(context.shape) != 2:
                    context = context.unsqueeze(0)

                input_ids = torch.cat([question, context], dim=1)
                target_length = teacher_steps.size(-1)

                # Process in chunks
                with torch.no_grad():
                    outputs = model(input_ids, return_intermediate=True)
                    latent_states = outputs["latent_results"]
                    probe_logits = model.reasoning_probe(
                        latent_states, target_length=target_length
                    )

                # Ensure teacher_steps has correct shape
                if len(teacher_steps.shape) == 2:
                    teacher_steps = teacher_steps.unsqueeze(1).expand(
                        -1, probe_logits.size(1), -1
                    )

                # Calculate loss
                loss = F.cross_entropy(
                    probe_logits.view(-1, probe_logits.size(-1)),
                    teacher_steps.view(-1),
                    ignore_index=-100,
                )

                total_loss += loss.item()
                num_batches += 1

                # Log examples only for first batch
                if num_batches == 1:
                    print("\nExample probe outputs:")
                    for step_idx in range(min(3, probe_logits.size(1))):
                        step_tokens = probe_logits[0, step_idx].argmax(dim=-1)
                        step_text = tokenizer.decode(step_tokens)
                        print(f"Step {step_idx}: {step_text}")

                # Clear cache after each batch
                torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"Error processing batch: {str(e)}")
            torch.cuda.empty_cache()
            continue

    return total_loss / num_batches if num_batches > 0 else float("inf")


def train_probe_batch(
    model, batch, probe_optimizer, device, tokenizer, gradient_accumulation_steps=1
):
    """
    Improved training loop with better handling of special tokens and padding
    """
    try:
        with torch.cuda.amp.autocast():
            # Process inputs with special token handling
            question = batch["question"].to(device)
            context = batch["context"].to(device)
            teacher_steps = batch["intermediate_answers"].to(device)

            # Create attention masks
            attention_mask = (question != tokenizer.pad_token_id).float()

            # Forward pass with improved handling
            outputs = model(
                input_ids=question,
                attention_mask=attention_mask,
                context=context,
                return_intermediate=True,
            )

            # Rest of the training logic
            probe_logits = model.reasoning_probe(
                outputs["latent_results"], target_length=teacher_steps.size(-1)
            )

            # Improved loss calculation with token weights
            loss_fn = nn.CrossEntropyLoss(
                ignore_index=tokenizer.pad_token_id,
                label_smoothing=0.1,
                reduction="none",
            )

            # Calculate loss with proper masking
            loss = loss_fn(
                probe_logits.view(-1, probe_logits.size(-1)), teacher_steps.view(-1)
            )

            # Apply mask for padding
            mask = (teacher_steps != tokenizer.pad_token_id).float().view(-1)
            loss = (loss * mask).sum() / mask.sum()

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            loss.backward()

            return {
                "loss": loss.item() * gradient_accumulation_steps,
                "logits": probe_logits.detach(),
            }

    except RuntimeError as e:
        print(f"Error in batch: {str(e)}")
        torch.cuda.empty_cache()
        return None


def train_probe_epoch(
    model,
    train_loader,
    probe_optimizer,
    device,
    tokenizer,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
):
    model.eval()  # Keep main model frozen
    model.reasoning_probe.train()

    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training probe")):
        # Process batch
        batch_output = train_probe_batch(
            model=model,
            batch=batch,
            probe_optimizer=probe_optimizer,
            device=device,
            tokenizer=tokenizer,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        if batch_output is None:
            continue

        total_loss += batch_output["probe_loss"]
        num_batches += 1

        # Step optimizer after accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                model.reasoning_probe.parameters(), max_grad_norm
            )
            # Step optimizer
            probe_optimizer.step()
            probe_optimizer.zero_grad()

            # Log progress
            if (batch_idx + 1) % (gradient_accumulation_steps * 10) == 0:
                avg_loss = total_loss / num_batches
                print(f"\nStep {batch_idx + 1}, Average Loss: {avg_loss:.4f}")

    return total_loss / num_batches if num_batches > 0 else float("inf")


def validate_model(model, val_loader, criterion, device, use_amp):
    model.eval()
    val_loss = 0.0
    val_metrics = {
        "generation_loss": 0.0,
        "latent_loss": 0.0,
        "consistency_loss": 0.0,
        "kl_loss": 0.0,
        "per_loop_latent_losses": defaultdict(float),  # Track per-loop losses
    }
    num_intermediate_steps = defaultdict(int)  # Track counts per loop

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(val_loader, desc="Validation", leave=False)
        ):
            try:
                # Filter and process only tensor items
                device_batch = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        # Process tensor data
                        if value.device.type != "cpu":
                            value = value.cpu()
                        if value.dtype != torch.long:
                            value = value.to(torch.long)
                        # Create new tensor on device
                        device_tensor = torch.empty(
                            value.shape, dtype=torch.long, device=device
                        )
                        device_tensor.copy_(value)
                        device_batch[key] = device_tensor
                    else:
                        # Keep non-tensor data as is
                        device_batch[key] = value

                # Extract tensors for model input
                question = device_batch["question"]
                context = device_batch["context"]
                answer = device_batch["answer"]
                intermediate_answers = device_batch.get("intermediate_answers")

                # Prepare input
                try:
                    input_ids = torch.cat([question, context], dim=1)
                except Exception as e:
                    print("Error in concatenation:", str(e))
                    raise

                # Model forward pass
                with autocast_ctx(device_type="cuda", enabled=use_amp):
                    try:
                        output = model(input_ids)
                        targets = {
                            "answer": answer,
                            "intermediate_answers": intermediate_answers,
                        }
                        loss_dict = criterion(output, targets, tokenizer)
                    except Exception as e:
                        print("Error in forward pass:", str(e))
                        raise

                val_loss += loss_dict["total_loss"].item()

                # Handle both final generation and intermediate losses
                if "intermediate_state" in output:
                    # For intermediate state losses
                    loop_idx = output["loop_idx"]
                    val_metrics["per_loop_latent_losses"][loop_idx] += loss_dict[
                        "latent_loss"
                    ]
                    num_intermediate_steps[loop_idx] += 1
                    val_metrics["latent_loss"] += loss_dict[
                        "latent_loss"
                    ]  # Accumulate total latent loss
                else:
                    # For final generation losses
                    for key in val_metrics:
                        if key in loss_dict and key != "per_loop_latent_losses":
                            val_metrics[key] += loss_dict[key]

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise

        # Calculate averages
        num_batches = len(val_loader)
        val_loss /= num_batches

        # Average the regular metrics
        for key in val_metrics:
            if key != "per_loop_latent_losses":
                val_metrics[key] /= num_batches

        # Average the per-loop latent losses
        for loop_idx in val_metrics["per_loop_latent_losses"]:
            if num_intermediate_steps[loop_idx] > 0:
                val_metrics["per_loop_latent_losses"][
                    loop_idx
                ] /= num_intermediate_steps[loop_idx]

    return val_loss, val_metrics


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
    global_step,
    save_path,
    training_history=None,
):
    """
    Save model checkpoint with optional training history

    Args:
        model: The model to save
        optimizer: The optimizer state to save
        scheduler: The scheduler state to save
        epoch: Current epoch number
        best_val_loss: Best validation loss so far
        global_step: Global training step count
        save_path: Path to save the checkpoint
        training_history: Optional dictionary containing training metrics history
    """
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "global_step": global_step,
    }

    # Add training history if provided
    if training_history is not None:
        checkpoint["training_history"] = training_history

    torch.save(checkpoint, save_path)
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

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Set pad token to eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    special_tokens = [
        "<thought>",
        "</thought>",
        "<title>",
        "</title>",
        "<value>",
        "</value>",
    ]
    num_added_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens}
    )

    # Initialize thought processor
    thought_processor = ThoughtProcessor(tokenizer, max_length=args.block_size)

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

    if not validate_dataset(processed_data, tokenizer):
        raise ValueError("Dataset validation failed!")

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(
        f"Max token ID in data: {max(max(batch['question'].max(), batch['context'].max(), batch['answer'].max()) for batch in processed_data)}"
    )

    train_data, val_data = train_test_split(
        processed_data, test_size=0.25, random_state=42
    )

    # Create DataLoaders for training and validation
    train_loader = create_data_loader(
        train_data, args.batch_size, tokenizer, shuffle=True
    )
    val_loader = create_data_loader(val_data, args.batch_size, tokenizer, shuffle=False)

    if not test_gpu_operations():
        raise RuntimeError("GPU diagnostics failed, cannot proceed with training")

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

    if num_added_tokens > 0:
        latent_gpt.resize_token_embeddings(len(tokenizer))
        language_gpt.resize_token_embeddings(len(tokenizer))

    # Combine into LatentReasoningTransformer
    model = LatentReasoningTransformer(
        latent_gpt, language_gpt, max_loops=5, thought_processor=thought_processor
    ).to(device)

    print("Model Configuration:")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Model embedding size: {model.latent_gpt.token_embedding.weight.size(0)}")
    print(f"Block size: {block_size}")
    print(
        f"Position embedding capacity: {model.latent_gpt.position_embedding.weight.shape}"
    )

    optimizer = bnb.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        optim_bits=8,  # Enable 8-bit optimization
        block_wise=True,  # Enable block-wise quantization
        is_paged=True,  # Enable memory paging for even better memory efficiency
    )
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
    embedding_dim = 384  # This should match the sentence-transformer embedding dimension for "all-MiniLM-L6-v2"
    latent_dim = model.latent_gpt.config.n_embd
    print(f"Number of trainable parameters: {num_params}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Latent dimension: {latent_dim}")

    # Train the model
    results = train_with_batches(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
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
