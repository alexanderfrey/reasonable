import os
import torch
from tqdm.auto import tqdm
import wandb
import torch.distributed as dist
from strategies import SpanMaskingStrategy


def print_batch_examples(batch, tokenizer, n_examples=3):
    """Print n examples from a batch with their decoded text"""
    print("Batch keys:", batch.keys())
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"{key} shape:", value.shape)

    def safe_decode(tensor):
        try:
            # Convert to list and filter out padding and special tokens
            tokens = tensor.tolist()
            # Only keep tokens within valid range for GPT-2 tokenizer (0-50257)
            valid_tokens = [t for t in tokens if 0 < t < 50257]
            return tokenizer.decode(valid_tokens)
        except Exception as e:
            return f"[Error decoding: {str(e)}]"

    print("\n=== Example Batch Samples ===")
    batch_size = batch["main_inputs"].shape[0]
    for idx in range(min(n_examples, batch_size)):
        print(f"\n--- Example {idx+1} ---")

        # Print first few and last few tokens to help debug
        def print_token_info(tensor, name):
            tokens = tensor[idx].tolist()
            print(f"\n{name} tokens (first 10): {tokens[:10]}")
            print(f"{name} tokens (last 10): {tokens[-10:]}")
            print(f"Text: {safe_decode(tensor[idx])[:500]}...")  # First 500 chars

        print("\nMAIN PATHWAY:")
        print_token_info(batch["main_inputs"], "Input")
        print_token_info(batch["main_targets"], "Target")

        print("\nANALYSIS PATHWAY:")
        print_token_info(batch["analysis_inputs"], "Input")
        print_token_info(batch["analysis_targets"], "Target")

        print("\n" + "=" * 80)


class BilateralTrainer:
    """Trainer class for bilateral models that handles different strategies for each pathway"""

    def __init__(self, main_strategy, analysis_strategy):
        self.main_strategy = main_strategy
        self.analysis_strategy = analysis_strategy
        self.global_step = 0

    def set_global_step(self, step):
        """Set the global step, used when loading from checkpoint"""
        self.global_step = step

    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        step,
        train_metrics,
        val_metrics,
        config,
        path,
        rank=0,
    ):
        """
        Save checkpoint only on rank 0.

        Args:
            model: The bilateral model
            optimizer: The optimizer
            scheduler: The learning rate scheduler
            scaler: The gradient scaler for mixed precision training
            epoch: Current epoch number
            step: Current global step number
            train_metrics: Dictionary containing training metrics
            val_metrics: Dictionary containing validation metrics
            config: BilateralGPTConfig instance
            path: Path to save the checkpoint
            rank: Process rank in distributed training
        """
        if rank == 0:
            # Get the underlying model if using DDP
            model_state_dict = (
                model.module.state_dict()
                if hasattr(model, "module")
                else model.state_dict()
            )

            checkpoint = {
                "epoch": epoch,
                "step": step,  # Add step to checkpoint
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": config.__dict__,
            }

            # Create checkpoint directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save checkpoint
            torch.save(checkpoint, path)

            # Log checkpoint information
            print(f"\nSaved checkpoint to {path}")
            print(f"Checkpoint metrics:")
            print(
                f"- Epoch: {epoch}, Step: {step}\n"
                f"- Train: total_loss={train_metrics['total_loss']:.4f}, "
                f"main_loss={train_metrics['main_loss']:.4f}, "
                f"analysis_loss={train_metrics['analysis_loss']:.4f}"
            )
            print(
                f"- Val: total_loss={val_metrics['total_loss']:.4f}, "
                f"main_loss={val_metrics['main_loss']:.4f}, "
                f"analysis_loss={val_metrics['analysis_loss']:.4f}"
            )

    def train_epoch(
        self,
        model,
        train_loader,
        train_sampler,
        optimizer,
        scheduler,
        scaler,
        device,
        epoch,
        tokenizer=None,
        gen_every_n_steps=None,
        sample_prompts=None,
        rank=0,
        gradient_accumulation_steps=1,
        save_every_n_steps=None,  # New parameter
        checkpoint_dir=None,  # New parameter
        config=None,  # New parameter for saving checkpoints
    ):
        model.train()
        total_loss = 0
        total_main_loss = 0
        total_analysis_loss = 0
        # Initialize global_step to count total steps from start of training
        global_step = self.global_step

        print("global_step ", global_step)

        if rank == 0:
            print(f"\nStarting epoch {epoch} from global step {global_step}")
            print(f"\nGeneration settings:")
            print(f"Tokenizer present: {tokenizer is not None}")
            print(f"gen_every_n_steps: {gen_every_n_steps}")
            print(f"sample_prompts: {sample_prompts}")
            print(f"save_every_n_steps: {save_every_n_steps}")
            # print("\nPrinting examples from first batch:")
            # try:
            #     first_batch = next(iter(train_loader))
            #     # Move batch to CPU for printing
            #     first_batch = {
            #         k: v.cpu() if isinstance(v, torch.Tensor) else v
            #         for k, v in first_batch.items()
            #     }
            #     print_batch_examples(first_batch, tokenizer)
            # except Exception as e:
            #     print(f"Error printing batch examples: {str(e)}")
            #     import traceback

            #     traceback.print_exc()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=rank != 0)

        for batch_idx, batch in enumerate(progress_bar):
            # Handle both tuple and dict batch formats
            if isinstance(batch, tuple):
                input_ids, targets = batch
                input_ids = input_ids.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # Let strategies prepare inputs if needed
                main_inputs, main_targets = (
                    self.main_strategy.prepare_masked_input(input_ids)
                    if hasattr(self.main_strategy, "prepare_masked_input")
                    else (input_ids, targets)
                )
                analysis_inputs, analysis_targets = (
                    self.analysis_strategy.prepare_masked_input(input_ids)
                    if hasattr(self.analysis_strategy, "prepare_masked_input")
                    else (input_ids, targets)
                )
            else:
                # Batch is already prepared by the dataloader
                main_inputs = batch["main_inputs"].to(device, non_blocking=True)
                main_targets = batch["main_targets"].to(device, non_blocking=True)
                analysis_inputs = batch["analysis_inputs"].to(device, non_blocking=True)
                analysis_targets = batch["analysis_targets"].to(
                    device, non_blocking=True
                )

            # Determine if this is an accumulation step
            is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps != 0

            # Forward pass with autocast
            with torch.amp.autocast("cuda", dtype=torch.float16):
                # Get logits and optional embeddings from both pathways
                outputs = model(main_inputs)
                if isinstance(outputs, tuple):
                    main_logits, analysis_logits = outputs
                    embeddings = None
                else:
                    main_logits, analysis_logits, embeddings = outputs

                # Calculate losses using strategies
                main_loss = self.main_strategy.compute_loss(
                    main_logits, main_targets, embeddings=embeddings
                )

                analysis_loss = self.analysis_strategy.compute_loss(
                    analysis_logits, analysis_targets, embeddings=embeddings
                )

                # Combine losses with equal weighting
                loss = (main_loss + analysis_loss) / 2
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

                global_step += 1

                # Step-based checkpoint saving
                if (
                    save_every_n_steps is not None
                    and checkpoint_dir is not None
                    and config is not None
                    and global_step > 0
                    and global_step % save_every_n_steps == 0
                ):
                    # Current training metrics only, no validation
                    train_metrics = {
                        "total_loss": loss.item() * gradient_accumulation_steps,
                        "main_loss": main_loss.item(),
                        "analysis_loss": analysis_loss.item(),
                    }

                    # Save step checkpoint with empty validation metrics
                    step_checkpoint_path = f"{checkpoint_dir}/step_{global_step}.pt"
                    self.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        step=global_step,
                        train_metrics=train_metrics,
                        val_metrics={
                            "total_loss": 0.0,
                            "main_loss": 0.0,
                            "analysis_loss": 0.0,
                        },
                        config=config,
                        path=step_checkpoint_path,
                        rank=rank,
                    )

                    if rank == 0:
                        print(f"\nSaved step checkpoint at global_step {global_step}")

            # Update metrics
            total_loss += loss.item() * gradient_accumulation_steps
            total_main_loss += main_loss.item() * gradient_accumulation_steps
            total_analysis_loss += analysis_loss.item() * gradient_accumulation_steps

            # Sample generation logic
            should_generate = (
                tokenizer is not None
                and gen_every_n_steps is not None
                and sample_prompts is not None
                and global_step > 0
                and global_step % gen_every_n_steps == 0
            )

            if should_generate:
                samples = self._generate_samples(
                    model=model,
                    tokenizer=tokenizer,
                    sample_prompts=sample_prompts,
                    device=device,
                    rank=rank,
                )

                if rank == 0:
                    print(f"\nGenerated samples at step {global_step}:")
                    print(samples)
                    wandb.log(
                        {"generated_samples": wandb.Html(samples.replace("\n", "<br>"))}
                    )

            # Logging (only on rank 0)
            if rank == 0 and not is_accumulation_step:
                current_lr = scheduler.get_last_lr()[0]

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "total_loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                        "main_loss": f"{main_loss.item():.4f}",
                        "analysis_loss": f"{analysis_loss.item():.4f}",
                        "lr": f"{current_lr:.2e}",
                        "step": global_step,
                    }
                )

                # Log to wandb
                wandb.log(
                    {
                        "train/total_loss": loss.item() * gradient_accumulation_steps,
                        "train/main_loss": main_loss.item(),
                        "train/analysis_loss": analysis_loss.item(),
                        "train/learning_rate": current_lr,
                        "train/grad_scale": scaler.get_scale(),
                        "train/global_step": global_step,
                    }
                )

        if dist.is_initialized():
            dist.all_reduce(torch.tensor([total_loss]).to(device))
            total_loss /= dist.get_world_size()

        # Store the global step for next epoch
        self.global_step = global_step

        return {
            "total_loss": total_loss / len(train_loader),
            "main_loss": total_main_loss / len(train_loader),
            "analysis_loss": total_analysis_loss / len(train_loader),
        }

    def evaluate(
        self,
        model,
        val_loader,
        val_sampler,
        device,
        rank=0,
    ):
        """Evaluate the model using the strategies"""
        model.eval()
        total_loss = 0
        total_main_loss = 0
        total_analysis_loss = 0

        if val_sampler is not None:
            val_sampler.set_epoch(0)

        progress_bar = tqdm(val_loader, desc="Evaluating", disable=rank != 0)

        for batch in progress_bar:
            # Handle both tuple and dict batch formats similar to training
            if isinstance(batch, tuple):
                input_ids, targets = batch
                input_ids = input_ids.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                main_inputs, main_targets = (
                    self.main_strategy.prepare_masked_input(input_ids)
                    if hasattr(self.main_strategy, "prepare_masked_input")
                    else (input_ids, targets)
                )
                analysis_inputs, analysis_targets = (
                    self.analysis_strategy.prepare_masked_input(input_ids)
                    if hasattr(self.analysis_strategy, "prepare_masked_input")
                    else (input_ids, targets)
                )
            else:
                main_inputs = batch["main_inputs"].to(device, non_blocking=True)
                main_targets = batch["main_targets"].to(device, non_blocking=True)
                analysis_inputs = batch["analysis_inputs"].to(device, non_blocking=True)
                analysis_targets = batch["analysis_targets"].to(
                    device, non_blocking=True
                )

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    # Get predictions from both pathways
                    outputs = model(main_inputs)
                    if isinstance(outputs, tuple):
                        main_logits, analysis_logits = outputs
                        embeddings = None
                    else:
                        main_logits, analysis_logits, embeddings = outputs

                    # Calculate losses using strategies
                    main_loss = self.main_strategy.compute_loss(
                        main_logits, main_targets, embeddings=embeddings
                    )

                    analysis_loss = self.analysis_strategy.compute_loss(
                        analysis_logits, analysis_targets, embeddings=embeddings
                    )

                    # Combine losses
                    loss = (main_loss + analysis_loss) / 2

            # Accumulate losses
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_analysis_loss += analysis_loss.item()

            if rank == 0:
                progress_bar.set_postfix(
                    {
                        "total_loss": f"{loss.item():.4f}",
                        "main_loss": f"{main_loss.item():.4f}",
                        "analysis_loss": f"{analysis_loss.item():.4f}",
                    }
                )

        # Gather losses from all processes
        if dist.is_initialized():
            losses = torch.tensor(
                [total_loss, total_main_loss, total_analysis_loss], device=device
            )
            dist.all_reduce(losses)
            total_loss, total_main_loss, total_analysis_loss = (
                losses / dist.get_world_size()
            )

        return {
            "total_loss": total_loss / len(val_loader),
            "main_loss": total_main_loss / len(val_loader),
            "analysis_loss": total_analysis_loss / len(val_loader),
        }

    def _generate_samples(self, model, tokenizer, sample_prompts, device, rank):
        """Generate samples using both pathway strategies"""
        was_training = model.training
        model.eval()

        if dist.is_initialized():
            dist.barrier()

        samples = ""
        if rank == 0:
            try:
                sample_outputs = []
                for prompt in sample_prompts:
                    # Tokenize prompt
                    input_ids = torch.tensor(
                        tokenizer.encode(prompt), device=device
                    ).unsqueeze(0)

                    # For pretraining, create a masked version of the prompt
                    if isinstance(self.main_strategy, SpanMaskingStrategy):
                        masked_input_ids, _ = self.main_strategy.prepare_masked_input(
                            input_ids
                        )
                        masked_prompt = tokenizer.decode(masked_input_ids[0].tolist())
                    else:
                        masked_input_ids = input_ids

                    with torch.no_grad():
                        # Generate using main pathway strategy
                        main_output = self.main_strategy.generate(
                            model=model,
                            input_ids=input_ids,  # Use unmasked for continuation
                            max_length=100,
                            temperature=0.7,
                        )

                        # Generate using analysis pathway strategy
                        analysis_output = self.analysis_strategy.generate(
                            model=model,
                            input_ids=masked_input_ids,  # Use masked for analysis
                            max_length=50,
                            temperature=0.7,
                        )

                        # Decode outputs
                        main_text = tokenizer.decode(main_output[0].tolist())
                        analysis_text = tokenizer.decode(analysis_output[0].tolist())

                        # Include masked prompt if used
                        if isinstance(self.main_strategy, SpanMaskingStrategy):
                            sample_output = (
                                f"Prompt: {prompt}\n"
                                f"Masked Prompt: {masked_prompt}\n"
                                f"Main Pathway Output:\n{main_text}\n"
                                f"Analysis Pathway Output:\n{analysis_text}\n"
                                f"{'='*50}\n"
                            )
                        else:
                            sample_output = (
                                f"Prompt: {prompt}\n"
                                f"Main Pathway Output:\n{main_text}\n"
                                f"Analysis Pathway Output:\n{analysis_text}\n"
                                f"{'='*50}\n"
                            )
                        sample_outputs.append(sample_output)

                samples = "\n".join(sample_outputs)

            except Exception as e:
                print(f"Error generating samples: {str(e)}")
                import traceback

                traceback.print_exc()

        if dist.is_initialized():
            dist.barrier()

        if was_training:
            model.train()

        return samples


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
