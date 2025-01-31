import os
import torch
from tqdm.auto import tqdm
import wandb
import torch.distributed as dist


def print_batch_examples(batch, tokenizer):
    """Print a detailed example of what's being fed into the model"""
    print("\n" + "=" * 100)
    print("MODEL INPUT VISUALIZATION".center(100))
    print("=" * 100 + "\n")

    # Print batch information
    print("Batch Structure:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:<20} Shape: {list(value.shape)}")
    print("\n" + "-" * 100 + "\n")

    def format_text_block(text, max_len=80):
        """Format text into lines with proper wrapping"""
        lines = []
        current_line = ""
        for word in text.split():
            if len(current_line) + len(word) + 1 <= max_len:
                current_line += word + " "
            else:
                lines.append(current_line)
                current_line = word + " "
        lines.append(current_line)
        return "\n".join(lines)

    def safe_decode(tensor):
        try:
            tokens = tensor.tolist()
            valid_tokens = [t for t in tokens if 0 <= t < 50257]
            return tokenizer.decode(valid_tokens)
        except Exception as e:
            return f"[Error decoding: {str(e)}]"

    # Get first example from batch
    idx = 0

    print("\nInput:")
    input_text = safe_decode(batch["inputs"][idx])
    print(format_text_block(input_text))

    print("\nTarget:")
    target_text = safe_decode(batch["targets"][idx])
    print(format_text_block(target_text))

    print("\n" + "=" * 100)

    # Print token information for debugging
    print("\nDEBUG INFORMATION:")
    print("-" * 50)
    print("\nToken Counts:")
    print(f"Input tokens:  {len(batch['inputs'][idx])} tokens")
    print(f"Target tokens: {len(batch['targets'][idx])} tokens")

    print("\n" + "=" * 100 + "\n")


class MultiHeadTrainer:
    """Trainer for multi-head GPT model using specified strategies for each head"""

    def __init__(self, strategies):
        """
        Initialize trainer with strategies for each head.

        Args:
            strategies: Dict mapping head names to their respective strategies
        """
        self.strategies = strategies
        self.global_step = 0

    def save_checkpoint(self, model, optimizer, scheduler, epoch, step, save_dir):
        """Save model checkpoint and training state"""
        model_to_save = model.module if hasattr(model, "module") else model

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }

        save_path = f"{save_dir}/checkpoint_epoch{epoch}_step{step}.pt"
        torch.save(checkpoint, save_path)
        torch.save(checkpoint, f"{save_dir}/checkpoint_latest.pt")

    def train_epoch(
        self,
        model,
        train_loader,
        optimizer,
        scheduler,
        scaler,
        device,
        epoch,
        gradient_accumulation_steps=1,
        save_every_n_steps=None,
        checkpoint_dir=None,
        max_grad_norm=1.0,
        log_every_n_steps=10,
    ):
        """
        Train for one epoch using the specified strategies with W&B logging.

        Args:
            model: The multi-head model
            train_loader: DataLoader for training data
            optimizer: The optimizer
            scheduler: Learning rate scheduler
            scaler: Gradient scaler for mixed precision
            device: Device to train on
            epoch: Current epoch number
            gradient_accumulation_steps: Number of steps to accumulate gradients
            save_every_n_steps: Save checkpoint every n steps (if None, don't save)
            checkpoint_dir: Directory to save checkpoints
            max_grad_norm: Maximum gradient norm for clipping
            log_every_n_steps: Log detailed metrics every n steps
        """
        if save_every_n_steps and not checkpoint_dir:
            raise ValueError(
                "checkpoint_dir must be provided if save_every_n_steps is set"
            )

        model.train()
        losses = {name: 0.0 for name in self.strategies.keys()}
        total_loss = 0.0

        # Track moving averages for losses
        ema_losses = {name: 0.0 for name in self.strategies.keys()}
        ema_total_loss = 0.0
        ema_decay = 0.99

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        # Get head weights from model configuration
        model_unwrapped = model.module if hasattr(model, "module") else model
        head_weights = model_unwrapped.head_weights

        # Initialize grad scaler status tracking
        grad_scaler_skipped = 0
        nan_detected = 0

        try:
            for batch_idx, batch in enumerate(progress_bar):
                # Move data to device and handle potential key errors
                try:
                    input_ids = batch["inputs"].to(device)
                    targets = {
                        name: batch["targets"][name].to(device)
                        for name in self.strategies.keys()
                    }
                except KeyError as e:
                    print(f"Error in batch data structure: {e}")
                    continue
                except RuntimeError as e:
                    print(f"Error moving batch to device: {e}")
                    continue

                is_accumulation_step = (
                    batch_idx + 1
                ) % gradient_accumulation_steps != 0

                try:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        # Get outputs from all heads
                        outputs = model(input_ids)

                        # Verify outputs match expected heads
                        if not all(name in outputs for name in self.strategies):
                            missing = set(self.strategies) - set(outputs)
                            raise ValueError(f"Missing outputs for heads: {missing}")

                        # Compute losses for each head
                        head_losses = {}
                        for name, strategy in self.strategies.items():
                            try:
                                head_losses[name] = strategy.compute_loss(
                                    outputs[name], targets[name]
                                )

                                # Check for NaN losses
                                if torch.isnan(head_losses[name]):
                                    nan_detected += 1
                                    print(f"NaN loss detected for head {name}")
                                    continue

                            except Exception as e:
                                print(f"Error computing loss for head {name}: {e}")
                                continue

                        # Compute weighted sum of losses
                        loss = sum(
                            head_losses[name] * head_weights[name]
                            for name in head_losses.keys()
                        )
                        loss = loss / gradient_accumulation_steps

                    # Skip backward pass if loss is NaN
                    if torch.isnan(loss):
                        nan_detected += 1
                        print(f"NaN loss detected at step {self.global_step}")
                        continue

                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()

                    if not is_accumulation_step:
                        # Unscale gradients for clipping
                        scaler.unscale_(optimizer)

                        # Clip gradients and check for inf/nan
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )

                        if torch.isfinite(grad_norm):
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            grad_scaler_skipped += 1
                            print(f"Skipped scaler step due to inf/nan gradients")

                        optimizer.zero_grad(
                            set_to_none=True
                        )  # More efficient than zero_grad()
                        scheduler.step()
                        self.global_step += 1

                        # Log metrics to W&B
                        if wandb.run is not None:
                            # Log losses
                            wandb.log(
                                {
                                    "train/total_loss": loss.item(),
                                    "train/learning_rate": scheduler.get_last_lr()[0],
                                    "train/grad_norm": (
                                        grad_norm.item()
                                        if torch.isfinite(grad_norm)
                                        else 0
                                    ),
                                    "train/grad_scaler_skipped": grad_scaler_skipped,
                                    "train/nan_detected": nan_detected,
                                    "train/global_step": self.global_step,
                                },
                                step=self.global_step,
                            )

                            # Log individual head losses
                            wandb.log(
                                {
                                    f"train/{name}_loss": head_losses[name].item()
                                    for name in head_losses.keys()
                                },
                                step=self.global_step,
                            )

                            # Log gradient statistics
                            if self.global_step % log_every_n_steps == 0:
                                grad_stats = {
                                    "train/grad_mean": 0.0,
                                    "train/grad_std": 0.0,
                                    "train/grad_max": 0.0,
                                }
                                for name, param in model.named_parameters():
                                    if param.grad is not None:
                                        grad = param.grad.data
                                        if torch.isfinite(grad).all():
                                            grad_stats[
                                                "train/grad_mean"
                                            ] += grad.mean().item()
                                            grad_stats[
                                                "train/grad_std"
                                            ] += grad.std().item()
                                            grad_stats["train/grad_max"] = max(
                                                grad_stats["train/grad_max"],
                                                grad.abs().max().item(),
                                            )
                                wandb.log(grad_stats, step=self.global_step)

                        # Save checkpoint if needed
                        if (
                            save_every_n_steps
                            and self.global_step % save_every_n_steps == 0
                        ):
                            self.save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch,
                                step=self.global_step,
                                save_dir=checkpoint_dir,
                            )

                except RuntimeError as e:
                    print(f"Error during forward/backward pass: {e}")
                    continue

                # Update metrics and EMA
                total_loss += loss.item() * gradient_accumulation_steps
                ema_total_loss = (
                    ema_decay * ema_total_loss + (1 - ema_decay) * loss.item()
                )

                for name in head_losses:
                    losses[name] += head_losses[name].item()
                    ema_losses[name] = (
                        ema_decay * ema_losses[name]
                        + (1 - ema_decay) * head_losses[name].item()
                    )

                # Update progress bar with EMA values
                if batch_idx % log_every_n_steps == 0:
                    metrics = {
                        "loss": f"{ema_total_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        "step": self.global_step,
                    }
                    metrics.update(
                        {
                            f"{name}_loss": f"{ema_losses[name]:.4f}"
                            for name in self.strategies.keys()
                        }
                    )
                    progress_bar.set_postfix(metrics)

        except Exception as e:
            print(f"Unexpected error during training: {e}")
            raise

        finally:
            # Compute final metrics
            num_batches = len(train_loader)
            metrics = {
                "total_loss": total_loss / num_batches,
            }
            metrics.update(
                {
                    f"{name}_loss": losses[name] / num_batches
                    for name in self.strategies.keys()
                }
            )

            # Log epoch-level metrics to W&B
            if wandb.run is not None:
                wandb.log(
                    {
                        "train/epoch": epoch,
                        "train/epoch_total_loss": metrics["total_loss"],
                        **{
                            f"train/epoch_{name}_loss": metrics[f"{name}_loss"]
                            for name in self.strategies.keys()
                        },
                    }
                )

            return metrics

    def evaluate(self, model, val_loader, device):
        """Evaluate the model using the strategies"""
        model.eval()
        losses = {name: 0.0 for name in self.strategies.keys()}
        total_loss = 0.0

        # Get head weights from model configuration
        model_unwrapped = model.module if hasattr(model, "module") else model
        head_weights = model_unwrapped.head_weights

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch["inputs"].to(device)
                targets = {
                    name: batch["targets"].to(device) for name in self.strategies.keys()
                }

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(input_ids)

                    # Compute losses for each head
                    head_losses = {}
                    for name, strategy in self.strategies.items():
                        head_losses[name] = strategy.compute_loss(
                            outputs[name], targets[name]
                        )

                    # Compute weighted sum of losses
                    loss = sum(
                        head_losses[name] * head_weights[name]
                        for name in self.strategies.keys()
                    )

                total_loss += loss.item()
                for name in self.strategies.keys():
                    losses[name] += head_losses[name].item()

        num_batches = len(val_loader)
        metrics = {
            "total_loss": total_loss / num_batches,
        }
        metrics.update(
            {
                f"{name}_loss": losses[name] / num_batches
                for name in self.strategies.keys()
            }
        )

        return metrics

    def generate(
        self,
        model,
        input_ids,
        max_length,
        temperature=1.0,
        top_p=0.9,
        head_name="primary",
    ):
        """Generate tokens using specified head's strategy"""
        if head_name not in self.strategies:
            raise ValueError(f"Unknown head name: {head_name}")

        return self.strategies[head_name].generate(
            model=model,
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )

    def set_global_step(self, step):
        """Set the global step counter"""
        self.global_step = step


def log_latent_stats(model):
    for name, module in model.named_modules():
        if isinstance(module, MultiLatentAttention):
            # Log norm of latent queries and keys
            latent_q_norm = module.latent_queries.data.norm()
            latent_k_norm = module.latent_keys.data.norm()
            # Add to your logging system
            wandb.log(
                {
                    f"{name}_latent_q_norm": latent_q_norm,
                    f"{name}_latent_k_norm": latent_k_norm,
                }
            )
