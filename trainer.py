import os
import math
from pathlib import Path

import torch
from tqdm.auto import tqdm
import wandb
import warnings
import torch.distributed as dist
import torch.nn.functional as F

# from model import OptimizedMoELayer


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


def get_layer_stats(model):
    """Get parameter and gradient statistics for different layer types"""
    stats = {
        "other_params_mean": 0.0,
        "other_params_std": 0.0,
        "other_grads_mean": 0.0,
        "other_grads_std": 0.0,
        "expert_params_mean": 0.0,
        "expert_params_std": 0.0,
        "expert_grads_mean": 0.0,
        "expert_grads_std": 0.0,
        "router_params_mean": 0.0,
        "router_params_std": 0.0,
        "router_grads_mean": 0.0,
        "router_grads_std": 0.0,
        "head_params_mean": 0.0,
        "head_params_std": 0.0,
        "head_grads_mean": 0.0,
        "head_grads_std": 0.0,
    }

    def calculate_stats(values):
        if not values:
            return 0.0, float("nan")
        mean = sum(values) / len(values)
        squared_diff = [(x - mean) ** 2 for x in values]
        std = (sum(squared_diff) / len(values)) ** 0.5 if len(values) > 1 else 0.0
        return mean, std

    model_to_analyze = model.module if hasattr(model, "module") else model

    # Collect parameters and gradients
    other_params, other_grads = [], []
    expert_params, expert_grads = [], []
    router_params, router_grads = [], []
    head_params, head_grads = [], []

    for name, param in model_to_analyze.named_parameters():
        if param.requires_grad:
            param_val = param.abs().mean().item()
            grad_val = (
                param.grad.abs().mean().item() if param.grad is not None else None
            )

            if "expert" in name.lower():
                expert_params.append(param_val)
                if grad_val is not None:
                    expert_grads.append(grad_val)
            elif "router" in name.lower():
                router_params.append(param_val)
                if grad_val is not None:
                    router_grads.append(grad_val)
            elif "head" in name.lower():
                head_params.append(param_val)
                if grad_val is not None:
                    head_grads.append(grad_val)
            else:
                other_params.append(param_val)
                if grad_val is not None:
                    other_grads.append(grad_val)

    # Calculate statistics
    stats["other_params_mean"], stats["other_params_std"] = calculate_stats(
        other_params
    )
    stats["other_grads_mean"], stats["other_grads_std"] = calculate_stats(other_grads)

    stats["expert_params_mean"], stats["expert_params_std"] = calculate_stats(
        expert_params
    )
    stats["expert_grads_mean"], stats["expert_grads_std"] = calculate_stats(
        expert_grads
    )

    stats["router_params_mean"], stats["router_params_std"] = calculate_stats(
        router_params
    )
    stats["router_grads_mean"], stats["router_grads_std"] = calculate_stats(
        router_grads
    )

    stats["head_params_mean"], stats["head_params_std"] = calculate_stats(head_params)
    stats["head_grads_mean"], stats["head_grads_std"] = calculate_stats(head_grads)

    return stats


def get_expert_usage_metrics(model):
    """Get expert usage statistics from MoE layers"""
    metrics = {}
    model_to_analyze = model.module if hasattr(model, "module") else model

    for name, module in model_to_analyze.named_modules():
        if hasattr(module, "get_expert_usage"):
            try:
                layer_metrics = module.get_expert_usage()
                for metric_name, value in layer_metrics.items():
                    full_name = f"{name}_{metric_name}"
                    if isinstance(value, torch.Tensor):
                        metrics[full_name] = value.item()
                    else:
                        metrics[full_name] = value
            except Exception as e:
                warnings.warn(f"Error getting expert usage for {name}: {str(e)}")
                continue

    # Add aggregate metrics
    if metrics:
        try:
            # Calculate average load balance
            load_balance_metrics = [
                v for k, v in metrics.items() if "load_balance" in k
            ]
            if load_balance_metrics:
                metrics["avg_load_balance"] = sum(load_balance_metrics) / len(
                    load_balance_metrics
                )

            # Calculate average routing consistency
            routing_metrics = [v for k, v in metrics.items() if "routing" in k]
            if routing_metrics:
                metrics["avg_routing_consistency"] = sum(routing_metrics) / len(
                    routing_metrics
                )
        except Exception as e:
            warnings.warn(f"Error calculating aggregate metrics: {str(e)}")

    return metrics


class DualHeadTrainer:
    """Trainer for dual-head GPT model using NextTokenStrategy for both heads"""

    def __init__(self, main_strategy, second_strategy=None):
        """
        Initialize trainer with strategies for both heads.

        Args:
            main_strategy: Strategy for primary head (NextTokenStrategy instance)
            second_strategy: Strategy for second head (if None, uses main_strategy)
        """
        self.main_strategy = main_strategy
        self.second_strategy = second_strategy or main_strategy
        self.global_step = 0
        self._static_graph = False

    def set_static_graph(self, value: bool = True):
        """Set whether to use static graph optimization."""
        self._static_graph = value

    def set_global_step(self, step):
        """Set the global step counter"""
        self.global_step = step

    def save_checkpoint(self, model, optimizer, scheduler, epoch, step, save_dir):
        """Save model checkpoint and training state"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Handle distributed training
        model_to_save = model.module if hasattr(model, "module") else model

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
        }

        save_path = save_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
        torch.save(checkpoint, save_path)

        # Also save as latest checkpoint
        latest_path = save_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

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
        max_grad_norm=0.5,
        initial_step=0,
        num_warmup_steps=2000,
        rank=0,
        world_size=None,
    ):
        model.train()
        self.global_step = initial_step
        scaler_unscaled = False

        if rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        else:
            progress_bar = train_loader

        def normalize_loss(loss, scale_factor=1.0):
            return torch.nn.functional.softplus(loss.clamp(max=20)) * scale_factor

        current_loss = 0
        current_main_loss = 0
        current_second_loss = 0

        for batch_idx, batch in enumerate(progress_bar):
            try:
                scaler_unscaled = False

                input_ids = batch["inputs"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)

                # For the auxiliary task, we'll use the same targets
                # You might want to modify this based on your specific auxiliary task
                aux_targets = batch.get("aux_targets", targets).to(
                    device, non_blocking=True
                )

                batch = None

                is_accumulation_step = (
                    batch_idx + 1
                ) % gradient_accumulation_steps != 0

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    # Forward pass with the new model
                    outputs = model(
                        input_ids=input_ids, labels=targets, aux_labels=aux_targets
                    )

                    # The model now returns a tuple (loss, main_logits, second_logits)
                    # If labels were provided, loss is the first element
                    if len(outputs) >= 3:
                        loss = outputs[0]  # Combined loss from the model
                        main_logits = outputs[1]
                        second_logits = outputs[2]

                        # The model now handles loss computation internally
                        primary_loss = loss  # The model returns combined loss
                        second_loss = torch.tensor(
                            0.0, device=device
                        )  # Included in primary_loss
                    else:
                        main_logits, second_logits = outputs[:2]

                        # Compute losses separately if not provided by model
                        primary_loss = normalize_loss(
                            self.main_strategy.compute_loss(main_logits, targets)
                        )
                        second_loss = normalize_loss(
                            self.second_strategy.compute_loss(
                                second_logits, aux_targets
                            )
                        )
                        loss = 0.5 * (primary_loss + second_loss)

                    scaled_loss = loss / gradient_accumulation_steps

                # Store current loss values
                current_loss = loss.item()
                current_main_loss = primary_loss.item()
                current_second_loss = second_loss.item()

                # Update progress bar
                if rank == 0:
                    progress_bar.set_postfix(
                        loss=f"{current_loss:.4f}",
                        main_loss=f"{current_main_loss:.4f}",
                        second_loss=f"{current_second_loss:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        step=self.global_step,
                    )

                del outputs
                torch.cuda.empty_cache()

                if torch.isnan(scaled_loss):
                    if rank == 0:
                        warnings.warn(f"Skipping batch {batch_idx} due to NaN loss")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(scaled_loss).backward()

                if not is_accumulation_step:
                    try:
                        if not scaler_unscaled:
                            scaler.unscale_(optimizer)
                            scaler_unscaled = True

                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )

                        if torch.isfinite(grad_norm) and grad_norm < max_grad_norm:
                            scaler.step(optimizer)
                            scaler.update()
                            scaler_unscaled = False

                            if self.global_step >= num_warmup_steps:
                                scheduler.step()

                            self.global_step += 1
                        else:
                            if rank == 0:
                                warnings.warn(
                                    f"Skipping step {self.global_step} due to large gradient norm: {grad_norm}"
                                )
                            optimizer.zero_grad(set_to_none=True)
                            scaler_unscaled = False

                    except RuntimeError as e:
                        if "unscale_() has already been called" in str(e):
                            optimizer.zero_grad(set_to_none=True)
                            scaler_unscaled = False
                            continue
                        else:
                            raise e

            except Exception as e:
                if rank == 0:
                    warnings.warn(f"Error in batch {batch_idx}: {str(e)}")
                continue

        return {
            "total_loss": current_loss,
            "main_loss": current_main_loss,
            "second_loss": current_second_loss,
            "final_lr": scheduler.get_last_lr()[0],
            "steps_completed": self.global_step - initial_step,
        }

    def evaluate(self, model, val_loader, device):
        """Evaluate the model on validation data"""
        model.eval()
        total_loss = 0
        total_main_loss = 0
        total_second_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["inputs"].to(device)
                targets = batch["targets"].to(device)
                aux_targets = batch.get("aux_targets", targets).to(device)

                outputs = model(
                    input_ids=input_ids, labels=targets, aux_labels=aux_targets
                )

                if len(outputs) >= 3:
                    loss = outputs[0]
                    main_logits = outputs[1]
                    second_logits = outputs[2]

                    total_loss += loss.item()
                    total_main_loss += (
                        loss.item()
                    )  # Main loss is included in total loss
                else:
                    main_logits, second_logits = outputs[:2]

                    main_loss = self.main_strategy.compute_loss(main_logits, targets)
                    second_loss = self.second_strategy.compute_loss(
                        second_logits, aux_targets
                    )

                    total_main_loss += main_loss.item()
                    total_second_loss += second_loss.item()
                    total_loss += (main_loss + second_loss).item() * 0.5

                num_batches += 1

        return {
            "total_loss": total_loss / num_batches,
            "main_loss": total_main_loss / num_batches,
            "second_loss": total_second_loss / num_batches,
        }

    def generate(
        self, model, input_ids, max_length, temperature=1.0, top_p=0.9, head="primary"
    ):
        """Generate tokens using specified head's strategy"""
        strategy = self.main_strategy if head == "primary" else self.second_strategy
        return strategy.generate(
            model=model,
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )
