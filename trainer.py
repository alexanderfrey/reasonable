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

    def save_checkpoint(self, model, optimizer, scheduler, epoch, step, save_dir):
        """Save model checkpoint and training state"""
        # Handle distributed training
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

        # Also save as latest checkpoint
        latest_path = f"{save_dir}/checkpoint_latest.pt"
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
    ):
        """Train for one epoch using the specified strategies

        Args:
            save_every_n_steps: If not None, save checkpoint every n steps
            checkpoint_dir: Directory to save checkpoints if save_every_n_steps is set
        """
        if save_every_n_steps and not checkpoint_dir:
            raise ValueError(
                "checkpoint_dir must be provided if save_every_n_steps is set"
            )

        model.train()
        total_loss = 0
        total_main_loss = 0
        total_second_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        # Get aux_head_weight from model, handling DDP case
        aux_head_weight = (
            model.module.config.aux_head_weight
            if hasattr(model, "module")
            else model.config.aux_head_weight
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            input_ids = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            targets_second = (
                targets.clone()
            )  # Same targets for both heads in pretraining

            # Check if this is an accumulation step
            is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps != 0

            # Forward pass with autocast
            with torch.amp.autocast("cuda", dtype=torch.float16):
                # Get outputs from both heads
                primary_logits, second_logits, _, _ = model(
                    input_ids, target_second=targets_second
                )

                # Compute losses using strategies
                primary_loss = self.main_strategy.compute_loss(primary_logits, targets)
                second_loss = self.second_strategy.compute_loss(
                    second_logits, targets_second
                )

                # Combined loss with aux_head_weight
                loss = primary_loss + (aux_head_weight * second_loss)
                loss = loss / gradient_accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            if not is_accumulation_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                self.global_step += 1

                # Save checkpoint if needed
                if save_every_n_steps and self.global_step % save_every_n_steps == 0:
                    self.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=self.global_step,
                        save_dir=checkpoint_dir,
                    )

            # Update metrics
            total_loss += loss.item() * gradient_accumulation_steps
            total_main_loss += primary_loss.item()
            total_second_loss += second_loss.item()

            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "main_loss": f"{primary_loss.item():.4f}",
                    "second_loss": f"{second_loss.item():.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": self.global_step,
                }
            )

        # Compute average losses
        num_batches = len(train_loader)
        avg_total_loss = total_loss / num_batches
        avg_main_loss = total_main_loss / num_batches
        avg_second_loss = total_second_loss / num_batches

        return {
            "total_loss": avg_total_loss,
            "main_loss": avg_main_loss,
            "second_loss": avg_second_loss,
        }

    def evaluate(self, model, val_loader, device):
        """Evaluate the model using the strategies"""
        model.eval()
        total_loss = 0
        total_main_loss = 0
        total_second_loss = 0

        # Get aux_head_weight from model, handling DDP case
        aux_head_weight = (
            model.module.config.aux_head_weight
            if hasattr(model, "module")
            else model.config.aux_head_weight
        )

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch["inputs"].to(device)
                targets = batch["targets"].to(device)
                targets_second = targets.clone()

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    primary_logits, second_logits, _, _ = model(
                        input_ids, target_second=targets_second
                    )

                    # Compute losses using strategies
                    primary_loss = self.main_strategy.compute_loss(
                        primary_logits, targets
                    )
                    second_loss = self.second_strategy.compute_loss(
                        second_logits, targets_second
                    )

                    loss = primary_loss + (aux_head_weight * second_loss)

                total_loss += loss.item()
                total_main_loss += primary_loss.item()
                total_second_loss += second_loss.item()

        num_batches = len(val_loader)
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
