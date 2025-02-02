import os, math
import torch
from tqdm.auto import tqdm
import wandb
import torch.distributed as dist
import torch.nn.functional as F


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
        tokenizer=None,
        sample_prompts=None,
        gen_every_n_steps=None,
        max_gen_length=100,
        gradient_accumulation_steps=4,
        save_every_n_steps=None,
        checkpoint_dir=None,
        max_grad_norm=1.0,
        log_every_n_steps=10,
    ):
        """
        Train for one epoch with improved stability and perplexity monitoring
        """
        if gen_every_n_steps and (tokenizer is None or not sample_prompts):
            raise ValueError(
                "tokenizer and sample_prompts must be provided if gen_every_n_steps is set"
            )

        model.train()
        losses = {name: 0.0 for name in self.strategies.keys()}
        perplexities = {name: 0.0 for name in self.strategies.keys()}
        total_loss = 0.0
        total_perplexity = 0.0

        # Exponential moving averages
        ema_losses = {name: 0.0 for name in self.strategies.keys()}
        ema_perplexities = {name: 0.0 for name in self.strategies.keys()}
        ema_total_loss = 0.0
        ema_total_perplexity = 0.0
        ema_decay = 0.99

        # Training monitoring
        grad_scaler_skipped = 0
        nan_detected = 0
        consecutive_nan_count = 0
        max_consecutive_nan = 5

        # Temperature annealing for loss stability
        temperature = max(0.8, 1.0 - epoch * 0.1)  # Gradually decrease temperature

        # Component-specific gradient clipping thresholds
        clip_thresholds = {
            "token_embedding": 0.5,
            "position_embedding": 0.5,
            "blocks": 0.5,
            "ln_fs": 0.1,
            "lm_heads": 0.1,
        }

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        model_unwrapped = model.module if hasattr(model, "module") else model
        head_weights = model_unwrapped.head_weights

        # Initialize metrics
        confidence_metrics = {name: 0.0 for name in self.strategies.keys()}
        head_confidences = {}
        ema_confidence = {name: 0.0 for name in self.strategies.keys()}

        try:
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    input_ids = batch["inputs"].to(device)
                    targets = {
                        name: batch["targets"][name].to(device)
                        for name in self.strategies.keys()
                    }

                    if input_ids.shape[0] == 0:
                        continue

                except (KeyError, RuntimeError) as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue

                is_accumulation_step = (
                    batch_idx + 1
                ) % gradient_accumulation_steps != 0
                debug_freq = 500

                try:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(input_ids)

                        # Check and stabilize model outputs
                        for name in outputs:
                            output = outputs[name]
                            if torch.isnan(output).any() or torch.isinf(output).any():
                                print(
                                    f"Warning: Found nan/inf in model output for {name}"
                                )
                                output = torch.nan_to_num(
                                    output, nan=0.0, posinf=100.0, neginf=-100.0
                                )
                                outputs[name] = output

                        head_losses = {}
                        head_perplexities = {}

                        if batch_idx % debug_freq == 0:
                            for name in self.strategies.keys():
                                print(f"\n=== Debug for head: {name} ===")
                                self._debug_loss_components(
                                    outputs[name], targets[name]
                                )
                            print("=" * 50)

                        for name, strategy in self.strategies.items():
                            try:
                                # Apply temperature scaling
                                scaled_outputs = outputs[name] / temperature

                                loss = strategy.compute_loss(
                                    scaled_outputs, targets[name]
                                )

                                # Validate loss
                                if torch.isnan(loss) or torch.isinf(loss):
                                    print(f"Warning: Invalid loss detected for {name}")
                                    # Use a safe default loss value
                                    loss = torch.tensor(
                                        1.0, device=loss.device, requires_grad=True
                                    )

                                # Compute perplexity with safeguards
                                ppl = self._compute_perplexity(
                                    scaled_outputs, targets[name]
                                )
                                ppl = min(ppl, 1000.0)  # Cap maximum perplexity

                                # Get confidence metrics
                                stats = strategy.get_stats()
                                if stats:
                                    current_confidence = stats["avg_confidence"]
                                    head_confidences[name] = current_confidence
                                    confidence_metrics[name] += current_confidence
                                    ema_confidence[name] = (
                                        ema_decay * ema_confidence[name]
                                        + (1 - ema_decay) * current_confidence
                                    )

                                # Update metrics
                                head_losses[name] = loss.clamp(-100, 100)
                                head_perplexities[name] = ppl
                                perplexities[name] += ppl
                                ema_perplexities[name] = (
                                    ema_decay * ema_perplexities[name]
                                    + (1 - ema_decay) * ppl
                                )

                            except Exception as e:
                                print(f"Error computing metrics for head {name}: {e}")
                                continue

                        if not head_losses:
                            continue

                        # Compute total loss with gradient accumulation
                        loss = (
                            sum(
                                head_losses[name] * head_weights[name]
                                for name in head_losses.keys()
                            )
                            / gradient_accumulation_steps
                        )

                        # Early detection of problematic loss
                        if loss.item() > 100 or torch.isnan(loss) or torch.isinf(loss):
                            nan_detected += 1
                            consecutive_nan_count += 1
                            print(
                                f"Anomalous loss detected: {loss.item()} at step {self.global_step}"
                            )

                            if consecutive_nan_count >= max_consecutive_nan:
                                print(
                                    "Too many consecutive anomalous losses. Implementing recovery..."
                                )
                                optimizer.zero_grad(set_to_none=True)
                                # Reduce learning rate
                                for param_group in optimizer.param_groups:
                                    param_group["lr"] = param_group["lr"] * 0.5
                                # Reset counter
                                consecutive_nan_count = 0
                            continue
                        else:
                            consecutive_nan_count = 0

                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()

                    if not is_accumulation_step:
                        # Gradient stabilization and clipping
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                # Replace nan/inf gradients with zeros
                                if (
                                    torch.isnan(param.grad).any()
                                    or torch.isinf(param.grad).any()
                                ):
                                    param.grad.data.zero_()
                                    continue

                                # Clip extreme gradient values
                                torch.clamp_(param.grad.data, min=-1.0, max=1.0)

                                # Apply component-specific clipping
                                for component, threshold in clip_thresholds.items():
                                    if component in name:
                                        torch.nn.utils.clip_grad_norm_(
                                            [param], threshold
                                        )

                        # Monitor gradients
                        if self.global_step % log_every_n_steps == 0:
                            self._log_gradient_norms(model, max_grad_norm)

                        # Global gradient norm check
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )

                        if torch.isfinite(grad_norm):
                            # Add minimal gradient noise for stability
                            noise_scale = 1e-5 * min(1.0, self.global_step / 1000)
                            for param in model.parameters():
                                if param.grad is not None:
                                    noise = torch.randn_like(param.grad) * noise_scale
                                    param.grad.add_(noise)

                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            grad_scaler_skipped += 1
                            print(
                                f"Skipped scaler step due to inf/nan gradients at step {self.global_step}"
                            )
                            self._log_gradient_issues(model)
                            continue

                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        self.global_step += 1

                        # Log metrics
                        if wandb.run is not None:
                            metrics = {
                                "train/total_loss": loss.item(),
                                "train/total_perplexity": ema_total_perplexity,
                                "train/learning_rate": scheduler.get_last_lr()[0],
                                "train/grad_norm": (
                                    grad_norm.item() if torch.isfinite(grad_norm) else 0
                                ),
                                "train/grad_scaler_skipped": grad_scaler_skipped,
                                "train/nan_detected": nan_detected,
                                "train/global_step": self.global_step,
                                "train/temperature": temperature,
                                **{
                                    f"train/{name}_loss": head_losses[name].item()
                                    for name in head_losses
                                },
                                **{
                                    f"train/{name}_perplexity": ema_perplexities[name]
                                    for name in head_perplexities
                                },
                                **{
                                    f"train/{name}_confidence": ema_confidence[name]
                                    for name in ema_confidence
                                },
                            }
                            wandb.log(metrics, step=self.global_step)

                        # Optional generation and checkpointing
                        if (
                            gen_every_n_steps
                            and self.global_step % gen_every_n_steps == 0
                        ):
                            self._generate_and_log_samples(
                                model, tokenizer, sample_prompts, max_gen_length, device
                            )

                        if (
                            save_every_n_steps
                            and self.global_step % save_every_n_steps == 0
                        ):
                            self.save_checkpoint(
                                model,
                                optimizer,
                                scheduler,
                                epoch,
                                self.global_step,
                                checkpoint_dir,
                            )

                    # Update loss tracking
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

                    # Update progress bar
                    if batch_idx % log_every_n_steps == 0:
                        self._update_progress_bar(
                            progress_bar,
                            ema_total_loss,
                            scheduler,
                            ema_losses,
                            ema_perplexities,
                            ema_confidence,
                        )

                except RuntimeError as e:
                    print(f"Error during training step {self.global_step}: {e}")
                    continue

        except Exception as e:
            print(f"Unexpected training error: {e}")
            raise

        finally:
            # Compute final metrics
            metrics = {
                "epoch": epoch,
                "total_loss": total_loss / len(train_loader),
                "total_perplexity": total_perplexity / len(train_loader),
                **{f"{name}_loss": losses[name] / len(train_loader) for name in losses},
                **{
                    f"{name}_perplexity": perplexities[name] / len(train_loader)
                    for name in perplexities
                },
            }

            if wandb.run is not None:
                wandb.log(
                    {f"epoch/{k}": v for k, v in metrics.items()}, step=self.global_step
                )

            return metrics

    def _debug_loss_components(self, logits, targets):
        with torch.no_grad():
            # Flatten and mask
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)
            mask = flat_targets != -100

            # Get valid examples
            valid_logits = flat_logits[mask]
            valid_targets = flat_targets[mask]

            # Basic stats
            print("\nLoss Components Debug:")
            print(
                f"Valid logits range: [{valid_logits.min().item():.2f}, {valid_logits.max().item():.2f}]"
            )

            # Check probs
            probs = torch.softmax(valid_logits, dim=-1)
            target_probs = probs.gather(1, valid_targets.unsqueeze(1))
            print(
                f"Target probs range: [{target_probs.min().item():.2e}, {target_probs.max().item():.2e}]"
            )

            # Check log probs
            log_probs = torch.log_softmax(valid_logits, dim=-1)
            target_log_probs = log_probs.gather(1, valid_targets.unsqueeze(1))
            print(
                f"Target log probs range: [{target_log_probs.min().item():.2f}, {target_log_probs.max().item():.2f}]"
            )

    def _log_gradient_norms(self, model, max_grad_norm):
        layer_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                layer_norms[name] = norm
                if norm > max_grad_norm * 0.5:  # Alert on high gradients
                    print(f"High gradient in {name}: {norm}")
        return layer_norms

    def _log_gradient_issues(self, model):
        """Detailed logging of problematic gradients"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                if not torch.isfinite(grad).all():
                    print(f"Non-finite gradients in {name}")
                    print(
                        f"Gradient stats - min: {grad.min()}, max: {grad.max()}, mean: {grad.mean()}"
                    )

    def _compute_perplexity(self, logits, targets, ignore_index=-100):
        with torch.no_grad():
            # Add temperature scaling to soften the logits
            temperature = 1.2  # Slightly higher temperature to smooth probabilities
            logits = logits / temperature

            # Flatten and mask
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)
            mask = flat_targets != ignore_index

            if not mask.any():
                return float("inf")  # Return inf if no valid targets

            # Get valid examples
            valid_logits = flat_logits[mask]
            valid_targets = flat_targets[mask]

            # Add small epsilon to prevent log(0)
            epsilon = 1e-6

            # Compute log softmax with better numerical stability
            log_probs = F.log_softmax(valid_logits, dim=-1)
            target_log_probs = log_probs.gather(1, valid_targets.unsqueeze(1))

            # Clamp extreme values
            target_log_probs = torch.clamp(target_log_probs, min=-10, max=0)

            # Average negative log likelihood
            nll = -target_log_probs.mean()

            # Early return for non-finite values
            if not torch.isfinite(nll):
                return float("inf")

            # Compute perplexity with safeguards
            perplexity = torch.exp(torch.clamp(nll, min=0, max=10)).item()

            return min(perplexity, 1000.0)  # Cap maximum perplexity

    def _log_perplexity_debug_info(self, logits, targets, name):
        """
        Log detailed information about perplexity calculation for debugging.

        Args:
            logits: Raw model outputs
            targets: Target indices
            name: Head name for logging
        """
        with torch.no_grad():
            print(f"\nPerplexity debug info for {name}:")

            # Basic stats
            print(f"Logits shape: {logits.shape}")
            print(
                f"Logits - min: {logits.min().item():.2f}, max: {logits.max().item():.2f}, mean: {logits.mean().item():.2f}"
            )

            # Check for any inf/nan in logits
            inf_mask = torch.isinf(logits)
            nan_mask = torch.isnan(logits)
            print(f"Inf values in logits: {inf_mask.sum().item()}")
            print(f"NaN values in logits: {nan_mask.sum().item()}")

            # Compute and check probabilities
            probs = torch.softmax(logits.view(-1, logits.size(-1)), dim=-1)
            zero_probs = (probs == 0).float().mean().item()
            print(f"Proportion of zero probabilities: {zero_probs:.4f}")

            # Target distribution
            valid_mask = targets != -100
            valid_targets = targets[valid_mask]
            if valid_targets.numel() > 0:
                target_probs = probs[valid_mask].gather(1, valid_targets.unsqueeze(1))
                print(
                    f"Target probs - min: {target_probs.min().item():.2e}, max: {target_probs.max().item():.2e}"
                )
                print(
                    f"Log probs - min: {torch.log(target_probs).min().item():.2f}, max: {torch.log(target_probs).max().item():.2f}"
                )

            print("-" * 40)

    def _generate_and_log_samples(
        self, model, tokenizer, sample_prompts, max_gen_length, device
    ):
        # Store original state
        was_training = model.training

        # Clear any cached states and gradients
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log gradient state before generation
        total_norm_before = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm()
                total_norm_before += param_norm.item() ** 2
        total_norm_before = total_norm_before**0.5
        print(f"\nGradient norm before generation: {total_norm_before}")

        try:
            model.eval()
            print("\n" + "=" * 80)
            print(f"Generation samples at step {self.global_step}:")
            print("=" * 80)

            with torch.no_grad():
                generations = self.generate_samples(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=sample_prompts,
                    max_length=max_gen_length,
                    temperature=1.0,
                    device=device,
                )

                for head_name, head_gens in generations.items():
                    print(f"\n{head_name.upper()} HEAD GENERATIONS:")
                    print("-" * 40)
                    for gen in head_gens:
                        print(f"\nPrompt: {gen['prompt']}")
                        print(f"Generated: {gen['generated']}")
                        print("-" * 40)

                print("=" * 80 + "\n")

        finally:
            # Restore original training state
            if was_training:
                model.train()

            # Clear states again after generation
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Reset cached states in attention layers
            for module in model.modules():
                if hasattr(module, "freqs_cis"):
                    module.freqs_cis = None

            # Log gradient state after generation
            total_norm_after = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.norm()
                    total_norm_after += param_norm.item() ** 2
            total_norm_after = total_norm_after**0.5
            print(f"Gradient norm after generation: {total_norm_after}")

            # Check for gradient norm increase
            if total_norm_after > total_norm_before * 1.5:
                print(f"Warning: Generation caused significant gradient norm increase")
                print(f"Before: {total_norm_before:.4f}, After: {total_norm_after:.4f}")
                model.zero_grad(set_to_none=True)

            if wandb.run is not None:
                try:
                    self._log_generations_to_wandb(generations)
                    # Also log gradient norms
                    wandb.log(
                        {
                            "generation/grad_norm_before": total_norm_before,
                            "generation/grad_norm_after": total_norm_after,
                            "generation/grad_norm_ratio": (
                                total_norm_after / total_norm_before
                                if total_norm_before > 0
                                else 0
                            ),
                        },
                        step=self.global_step,
                    )
                except Exception as e:
                    print(f"Warning: Failed to log to W&B: {e}")

    def _log_generations_to_wandb(self, generations):
        gen_logs = {}
        for head_name, head_gens in generations.items():
            for i, gen in enumerate(head_gens):
                gen_logs[f"generation/{head_name}/prompt_{i}"] = wandb.Table(
                    columns=["prompt", "generated"],
                    data=[[gen["prompt"], gen["generated"]]],
                )
        wandb.log(gen_logs, step=self.global_step)

    def _log_training_metrics(
        self,
        loss,
        scheduler,
        grad_norm,
        grad_scaler_skipped,
        nan_detected,
        head_losses,
        model,
        log_every_n_steps,
    ):
        metrics = {
            "train/total_loss": loss.item(),
            "train/learning_rate": scheduler.get_last_lr()[0],
            "train/grad_norm": grad_norm.item() if torch.isfinite(grad_norm) else 0,
            "train/grad_scaler_skipped": grad_scaler_skipped,
            "train/nan_detected": nan_detected,
            "train/global_step": self.global_step,
            **{f"train/{name}_loss": head_losses[name].item() for name in head_losses},
        }

        if self.global_step % log_every_n_steps == 0:
            grad_stats = self._compute_gradient_stats(model)
            metrics.update(grad_stats)

            # Log if gradients are becoming unstable
            if grad_norm > 1.0 or not torch.isfinite(grad_norm):
                print(f"\nHigh gradient norm detected: {grad_norm}")
                self._log_detailed_grad_stats(model)

        wandb.log(metrics, step=self.global_step)

    def _compute_gradient_stats(self, model):
        stats = {
            "train/grad_mean": 0.0,
            "train/grad_std": 0.0,
            "train/grad_max": 0.0,
            "train/params_with_zero_grad": 0,
            "train/params_with_inf_grad": 0,
        }

        total_params = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                total_params += 1
                grad = param.grad.data

                # Count problematic gradients
                if grad.abs().sum().item() == 0:
                    stats["train/params_with_zero_grad"] += 1
                if not torch.isfinite(grad).all():
                    stats["train/params_with_inf_grad"] += 1
                    continue

                stats["train/grad_mean"] += grad.mean().item()
                stats["train/grad_std"] += grad.std().item()
                stats["train/grad_max"] = max(
                    stats["train/grad_max"], grad.abs().max().item()
                )

        # Normalize mean and std by number of parameters
        if total_params > 0:
            stats["train/grad_mean"] /= total_params
            stats["train/grad_std"] /= total_params

        return stats

    def _log_detailed_grad_stats(self, model):
        """Log detailed gradient statistics when issues are detected"""
        print("\n=== Detailed Gradient Statistics ===")

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_norm = grad.norm().item()

                # Only print problematic gradients to avoid cluttering
                if grad_norm > 1.0 or not torch.isfinite(grad_norm):
                    print(f"\nLayer: {name}")
                    print(f"  Gradient norm: {grad_norm:.4f}")
                    print(f"  Gradient mean: {grad.mean().item():.4f}")
                    print(f"  Gradient std: {grad.std().item():.4f}")
                    print(f"  Gradient min: {grad.min().item():.4f}")
                    print(f"  Gradient max: {grad.max().item():.4f}")

                    # Parameter statistics
                    print(f"  Parameter mean: {param.data.mean().item():.4f}")
                    print(f"  Parameter std: {param.data.std().item():.4f}")

                    # Check for common issues
                    zero_grad = grad.abs().sum().item() == 0
                    inf_grad = not torch.isfinite(grad).all()
                    if zero_grad:
                        print("  WARNING: Zero gradient detected")
                    if inf_grad:
                        print("  WARNING: Infinite/NaN gradient detected")

        print("\n===============================")

    def _update_progress_bar(
        self,
        progress_bar,
        ema_loss,
        scheduler,
        head_losses,
        head_perplexities,
        ema_confidence,
    ):
        """Update progress bar with all metrics"""
        postfix_dict = {
            "loss": f"{ema_loss:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        }

        # Add head-specific metrics
        for name in head_losses:
            postfix_dict[f"{name}_loss"] = f"{head_losses[name]:.4f}"
            postfix_dict[f"{name}_ppl"] = f"{head_perplexities[name]:.2f}"
            postfix_dict[f"{name}_conf"] = f"{ema_confidence[name]:.2f}"

        progress_bar.set_postfix(**postfix_dict)

    def _compute_final_metrics(self, train_loader, total_loss, losses):
        num_batches = len(train_loader)
        return {
            "total_loss": total_loss / num_batches,
            **{
                f"{name}_loss": losses[name] / num_batches
                for name in self.strategies.keys()
            },
        }

    def _log_epoch_metrics(self, metrics, epoch):
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

    def evaluate(self, model, val_loader, device):
        model.eval()
        losses = {name: 0.0 for name in self.strategies.keys()}
        total_loss = 0.0

        model_unwrapped = model.module if hasattr(model, "module") else model
        head_weights = model_unwrapped.head_weights

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                try:
                    input_ids = batch["inputs"].to(device)
                    targets = {
                        name: batch["targets"][name].to(device)
                        for name in self.strategies.keys()
                    }

                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(input_ids)
                        head_losses = {
                            name: strategy.compute_loss(outputs[name], targets[name])
                            for name, strategy in self.strategies.items()
                        }
                        loss = sum(
                            head_losses[name] * head_weights[name]
                            for name in self.strategies.keys()
                        )

                    total_loss += loss.item()
                    for name in self.strategies.keys():
                        losses[name] += head_losses[name].item()

                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    continue

        return {
            "total_loss": total_loss / len(val_loader),
            **{
                f"{name}_loss": losses[name] / len(val_loader)
                for name in self.strategies.keys()
            },
        }

    def generate(
        self,
        model,
        input_ids,
        max_length,
        temperature=1.0,
        top_p=0.9,
        head_name="primary",
        eos_token_id=50256,
    ):
        print(f"Temperature type: {type(temperature)}, value: {temperature}")

        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("input_ids must be a torch tensor")

        batch_size = input_ids.size(0)
        cur_len = input_ids.size(1)
        generated = input_ids.clone()
        temp = float(temperature)  # Ensure temperature is float

        with torch.no_grad():
            for _ in range(max_length - cur_len):
                outputs = model(generated)
                logits = outputs[head_name]
                next_token_logits = logits[:, -1, :] / temp

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_keep = cumulative_probs <= top_p
                    sorted_indices_to_keep[..., 0] = True
                    indices_to_keep = sorted_indices_to_keep.scatter(
                        1, sorted_indices, sorted_indices_to_keep
                    )
                    next_token_logits[~indices_to_keep] = float("-inf")

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                if (next_token == eos_token_id).any():
                    eos_positions = (generated == eos_token_id).nonzero()
                    if len(eos_positions) > 0:
                        first_eos_positions = []
                        for batch_idx in range(batch_size):
                            batch_eos = eos_positions[eos_positions[:, 0] == batch_idx]
                            if len(batch_eos) > 0:
                                first_eos_positions.append(batch_eos[0, 1].item())
                            else:
                                first_eos_positions.append(generated.size(1))

                        truncated = [
                            generated[i, :pos]
                            for i, pos in enumerate(first_eos_positions)
                        ]
                        max_len = max(seq.size(0) for seq in truncated)
                        padded = [
                            (
                                torch.cat(
                                    [
                                        seq,
                                        torch.full(
                                            (max_len - seq.size(0),),
                                            eos_token_id,
                                            dtype=seq.dtype,
                                            device=seq.device,
                                        ),
                                    ]
                                ).unsqueeze(0)
                                if seq.size(0) < max_len
                                else seq.unsqueeze(0)
                            )
                            for seq in truncated
                        ]
                        generated = torch.cat(padded, dim=0)
                        break

        return generated

    def generate_samples(
        self,
        model,
        tokenizer,
        prompts,
        max_length,
        temperature=1.0,
        top_p=0.9,
        device="cuda",
    ):
        print(f"Initial temperature type: {type(temperature)}, value: {temperature}")

        """
        Generate text samples from different model heads

        Args:
            model: The multi-head model
            tokenizer: The tiktoken tokenizer
            prompts: List of prompt strings
            max_length: Maximum length of generated sequences
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            device: Device to run generation on

        Returns:
            dict: Dictionary of generated texts for each head
        """
        model.eval()
        generations = {head_name: [] for head_name in self.strategies.keys()}

        with torch.no_grad():
            for prompt in prompts:
                # Use tiktoken's encode method and convert to tensor
                input_tokens = tokenizer.encode(prompt)
                input_ids = (
                    torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
                )

                for head_name in self.strategies.keys():
                    try:
                        output_ids = self.generate(
                            model=model,
                            input_ids=input_ids,
                            max_length=max_length,
                            temperature=temperature,
                            top_p=top_p,
                            head_name=head_name,
                        )

                        # Decode generated tokens
                        generated_text = tokenizer.decode(output_ids[0].tolist())
                        generations[head_name].append(
                            {"prompt": prompt, "generated": generated_text}
                        )
                    except Exception as e:
                        print(f"Error generating from head {head_name}: {e}")
                        generations[head_name].append(
                            {"prompt": prompt, "generated": f"ERROR: {str(e)}"}
                        )

        return generations

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


class SingleHeadTrainer:
    """Trainer for single-head GPT model using specified strategy"""

    def __init__(self, strategy):
        """
        Initialize trainer with a single strategy.

        Args:
            strategy: Training strategy to use
        """
        self.strategy = strategy
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

    def _compute_perplexity(self, logits, targets, ignore_index=-100):
        """
        Compute perplexity with numerical stability safeguards
        """
        with torch.no_grad():
            # Add temperature scaling to soften the logits
            temperature = 1.2
            logits = logits / temperature

            # Flatten and mask
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)
            mask = flat_targets != ignore_index

            if not mask.any():
                return float("inf")

            # Get valid examples
            valid_logits = flat_logits[mask]
            valid_targets = flat_targets[mask]

            # Compute log softmax with better numerical stability
            log_probs = F.log_softmax(valid_logits, dim=-1)
            target_log_probs = log_probs.gather(1, valid_targets.unsqueeze(1))

            # Clamp extreme values
            target_log_probs = torch.clamp(target_log_probs, min=-10, max=0)

            # Average negative log likelihood
            nll = -target_log_probs.mean()

            if not torch.isfinite(nll):
                return float("inf")

            # Compute perplexity with safeguards
            perplexity = torch.exp(torch.clamp(nll, min=0, max=10)).item()

            return min(perplexity, 1000.0)

    def _generate_samples(
        self, model, tokenizer, prompts, max_length, temperature, device
    ):
        """Generate samples from the model"""
        model.eval()
        generations = []

        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer.encode(prompt)
                input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

                # Generate with standard sampling
                output_ids = []
                for _ in range(max_length):
                    outputs = model(input_tensor)
                    next_token_logits = outputs[0, -1, :] / temperature
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    output_ids.append(next_token.item())
                    input_tensor = torch.cat(
                        [input_tensor, next_token.unsqueeze(0)], dim=1
                    )

                generated_text = tokenizer.decode(output_ids)
                generations.append({"prompt": prompt, "generated": generated_text})

        return generations

    def _log_generations_to_wandb(self, generations):
        """Log generated samples to W&B"""
        if wandb.run is not None:
            log_data = []
            for gen in generations:
                log_data.append(
                    f"Prompt: {gen['prompt']}\nGenerated: {gen['generated']}\n"
                )
            wandb.log(
                {
                    "generations": wandb.Table(
                        data=[["\n".join(log_data)]], columns=["Generated Text"]
                    )
                },
                step=self.global_step,
            )

    def train_epoch(
        self,
        model,
        train_loader,
        optimizer,
        scheduler,
        scaler,
        device,
        epoch,
        tokenizer=None,
        sample_prompts=None,
        gen_every_n_steps=None,
        max_gen_length=100,
        gradient_accumulation_steps=4,
        save_every_n_steps=None,
        checkpoint_dir=None,
        max_grad_norm=1.0,
        log_every_n_steps=10,
    ):
        """Memory-optimized training for one epoch"""
        model.train()

        # Use lightweight metric tracking
        running_loss = RunningAverage()
        running_ppl = RunningAverage()

        # Training monitoring
        grad_norm_total = 0
        num_steps = 0

        # Batch accumulation state
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Efficient device transfer
            with torch.cuda.stream(torch.cuda.Stream()):
                input_ids = batch["inputs"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)

            is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps != 0

            # Compute loss with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(input_ids)
                loss = self.strategy.compute_loss(outputs, targets)
                loss = loss / gradient_accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            # Update metrics
            with torch.no_grad():
                running_loss.update(loss.item() * gradient_accumulation_steps)
                running_ppl.update(torch.exp(loss).item() * gradient_accumulation_steps)

            # Gradient update step
            if not is_accumulation_step:
                # Gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                grad_norm_total += grad_norm.item()
                num_steps += 1

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # Reset gradients
                optimizer.zero_grad(set_to_none=True)

                # Log metrics
                if self.global_step % log_every_n_steps == 0 and wandb.run is not None:
                    metrics = {
                        "train/loss": running_loss.get_average(),
                        "train/perplexity": running_ppl.get_average(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/grad_norm_avg": grad_norm_total / max(1, num_steps),
                        "train/global_step": self.global_step,
                    }
                    wandb.log(metrics, step=self.global_step)

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": f"{running_loss.get_average():.4f}",
                        "ppl": f"{running_ppl.get_average():.2f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

                # Sample generations periodically
                if gen_every_n_steps and self.global_step % gen_every_n_steps == 0:
                    self._generate_and_log_samples(
                        model, tokenizer, sample_prompts, max_gen_length, device
                    )

                # Save checkpoint periodically
                if save_every_n_steps and self.global_step % save_every_n_steps == 0:
                    self.save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        self.global_step,
                        checkpoint_dir,
                    )

                self.global_step += 1

            # Explicit cleanup
            del outputs
            del loss

        return {
            "epoch": epoch,
            "final_loss": running_loss.get_average(),
            "final_perplexity": running_ppl.get_average(),
            "grad_norm_avg": grad_norm_total / max(1, num_steps),
        }

    def _log_perplexity_debug_info(self, logits, targets):
        """Log detailed perplexity calculation information"""
        with torch.no_grad():
            print("\nPerplexity debug info:")

            # Basic stats
            print(f"Logits shape: {logits.shape}")
            print(
                f"Logits - min: {logits.min().item():.2f}, max: {logits.max().item():.2f}, mean: {logits.mean().item():.2f}"
            )

            # Check for any inf/nan in logits
            inf_mask = torch.isinf(logits)
            nan_mask = torch.isnan(logits)
            print(f"Inf values in logits: {inf_mask.sum().item()}")
            print(f"NaN values in logits: {nan_mask.sum().item()}")

            # Compute and check probabilities
            probs = torch.softmax(logits.view(-1, logits.size(-1)), dim=-1)
            zero_probs = (probs == 0).float().mean().item()
            print(f"Proportion of zero probabilities: {zero_probs:.4f}")

            # Target distribution
            valid_mask = targets != -100
            valid_targets = targets[valid_mask]
            if valid_targets.numel() > 0:
                target_probs = probs[valid_mask].gather(1, valid_targets.unsqueeze(1))
                print(
                    f"Target probs - min: {target_probs.min().item():.2e}, max: {target_probs.max().item():.2e}"
                )
                print(
                    f"Log probs - min: {torch.log(target_probs).min().item():.2f}, max: {torch.log(target_probs).max().item():.2f}"
                )

            print("-" * 40)

    def _log_gradient_norms(self, model, max_grad_norm):
        """Log gradient norms for monitoring"""
        layer_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                layer_norms[name] = norm
                if norm > max_grad_norm * 0.5:  # Alert on high gradients
                    print(f"High gradient in {name}: {norm}")
        return layer_norms

    def _log_gradient_issues(self, model):
        """Detailed logging of problematic gradients"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                if not torch.isfinite(grad).all():
                    print(f"Non-finite gradients in {name}")
                    print(
                        f"Gradient stats - min: {grad.min()}, max: {grad.max()}, mean: {grad.mean()}"
                    )

    def _generate_and_log_samples(
        self, model, tokenizer, sample_prompts, max_gen_length, device
    ):
        """Generate and log sample outputs"""
        # Store original state
        was_training = model.training

        # Clear cached states and gradients
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            model.eval()
            print("\n" + "=" * 80)
            print(f"Generation samples at step {self.global_step}:")
            print("=" * 80)

            with torch.no_grad():
                generations = []
                for prompt in sample_prompts:
                    input_ids = tokenizer.encode(prompt)
                    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

                    # Generate with temperature sampling
                    output_ids = []
                    temperature = 0.7  # Adjust as needed

                    for _ in range(max_gen_length):
                        outputs = model(input_tensor)
                        next_token_logits = outputs[0, -1, :] / temperature
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        output_ids.append(next_token.item())
                        input_tensor = torch.cat(
                            [input_tensor, next_token.unsqueeze(0)], dim=1
                        )

                    generated_text = tokenizer.decode(output_ids)
                    generations.append({"prompt": prompt, "generated": generated_text})

                    print(f"\nPrompt: {prompt}")
                    print(f"Generated: {generated_text}")
                    print("-" * 40)

                print("=" * 80 + "\n")

                if wandb.run is not None:
                    self._log_generations_to_wandb(generations)

        finally:
            # Restore original state
            if was_training:
                model.train()

            # Clear states
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _log_generations_to_wandb(self, generations):
        """Log generated samples to W&B"""
        if wandb.run is not None:
            try:
                gen_table = wandb.Table(columns=["prompt", "generated"])
                for gen in generations:
                    gen_table.add_data(gen["prompt"], gen["generated"])
                wandb.log({"generations": gen_table}, step=self.global_step)
            except Exception as e:
                print(f"Warning: Failed to log generations to W&B: {e}")

    def _compute_gradient_stats(self, model):
        """Compute detailed gradient statistics"""
        grad_stats = {}
        total_norm = 0.0

        # Collect norms by layer type
        layer_norms = {
            "embedding": [],
            "attention": [],
            "mlp": [],
            "layernorm": [],
            "head": [],
        }

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_norm += grad_norm**2

                # Categorize by layer type
                if "embedding" in name:
                    layer_norms["embedding"].append(grad_norm)
                elif "attn" in name:
                    layer_norms["attention"].append(grad_norm)
                elif "mlp" in name:
                    layer_norms["mlp"].append(grad_norm)
                elif "ln" in name or "norm" in name:
                    layer_norms["layernorm"].append(grad_norm)
                elif "head" in name:
                    layer_norms["head"].append(grad_norm)

        # Compute statistics for each layer type
        for layer_type, norms in layer_norms.items():
            if norms:
                grad_stats[f"grad_norm/{layer_type}/mean"] = np.mean(norms)
                grad_stats[f"grad_norm/{layer_type}/max"] = np.max(norms)
                grad_stats[f"grad_norm/{layer_type}/min"] = np.min(norms)

        grad_stats["grad_norm/total"] = np.sqrt(total_norm)
        return grad_stats

    def _compute_gradient_stats(self, model):
        """Compute detailed gradient statistics"""
        grad_stats = {}
        layer_types = {
            "embedding": [],
            "attention": [],
            "mlp": [],
            "layernorm": [],
            "head": [],
        }

        for name, param in model.named_parameters():
            if param.grad is not None:
                # Categorize gradient by layer type
                if "embedding" in name:
                    layer_types["embedding"].append(param.grad.norm().item())
                elif "attn" in name:
                    layer_types["attention"].append(param.grad.norm().item())
                elif "mlp" in name:
                    layer_types["mlp"].append(param.grad.norm().item())
                elif "ln" in name or "norm" in name:
                    layer_types["layernorm"].append(param.grad.norm().item())
                elif "head" in name:
                    layer_types["head"].append(param.grad.norm().item())

        # Compute statistics for each layer type
        for layer_type, norms in layer_types.items():
            if norms:
                grad_stats[f"grad_norm/{layer_type}/mean"] = np.mean(norms)
                grad_stats[f"grad_norm/{layer_type}/max"] = np.max(norms)
                grad_stats[f"grad_norm/{layer_type}/min"] = np.min(norms)

        return grad_stats

    def set_global_step(self, step):
        """Set the global step counter"""
        self.global_step = step

    def evaluate(self, model, val_loader, device):
        """
        Evaluate model on validation dataset.

        Args:
            model: Model to evaluate
            val_loader: Validation data loader
            device: Device to run evaluation on

        Returns:
            dict: Dictionary containing evaluation metrics including total_loss, primary_loss,
                and auxiliary_loss for compatibility with the training script
        """
        model.eval()
        total_loss = 0.0
        total_perplexity = 0.0
        num_batches = len(val_loader)

        # Store original step
        original_step = self.global_step

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                try:
                    input_ids = batch["inputs"].to(device)
                    targets = batch["targets"].to(device)

                    if input_ids.shape[0] == 0:
                        continue

                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(input_ids)

                        # Apply temperature scaling as in training
                        temperature = 1.2
                        scaled_outputs = outputs / temperature

                        # Compute loss
                        loss = self.strategy.compute_loss(scaled_outputs, targets)

                        # Compute perplexity
                        ppl = self._compute_perplexity(scaled_outputs, targets)

                        if torch.isfinite(loss):
                            total_loss += loss.item()
                            total_perplexity += ppl

                except RuntimeError as e:
                    print(f"Error during evaluation: {e}")
                    continue

        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches

        # Since this is a single-head model, we set primary_loss equal to total_loss
        # and auxiliary_loss to 0 for compatibility with the training script
        metrics = {
            "total_loss": avg_loss,
            "primary_loss": avg_loss,  # Same as total_loss for single-head
            "auxiliary_loss": 0.0,  # No auxiliary loss in single-head model
            "perplexity": avg_perplexity,
        }

        # Log metrics
        if wandb.run is not None:
            wandb.log(
                {
                    "eval/total_loss": metrics["total_loss"],
                    "eval/primary_loss": metrics["primary_loss"],
                    "eval/auxiliary_loss": metrics["auxiliary_loss"],
                    "eval/perplexity": metrics["perplexity"],
                },
                step=original_step,
            )

        return metrics


class RunningAverage:
    """Efficient running average calculation"""

    def __init__(self):
        self.total = 0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1

    def get_average(self):
        return self.total / max(1, self.count)
