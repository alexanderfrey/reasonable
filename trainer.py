import os, math, html
import torch
from tqdm.auto import tqdm
import wandb
import torch.distributed as dist
import torch.nn.functional as F
from model import MoELayer, MoETransformerBlock
from typing import List, Dict, Optional, Union, Tuple
from data_utils import CurriculumDataset
from strategies import InstructionFollowingStrategy


class ExpertUsageTracker:
    """Track and analyze expert usage statistics"""

    def __init__(self, model):
        self.model = model
        self.reset_stats()

    def reset_stats(self):
        """Reset all tracking statistics"""
        self.expert_counts = {}
        self.layer_usage = {}
        self.total_tokens = 0

        # Initialize counters for each MoE layer
        for i, block in enumerate(self.model.blocks):
            if hasattr(block, "moe"):
                self.expert_counts[f"layer_{i}"] = torch.zeros(
                    len(block.moe.experts), device=next(block.parameters()).device
                )

    def update(self, router_mask, layer_idx):
        """Update statistics with new batch of router decisions"""
        if f"layer_{layer_idx}" in self.expert_counts:
            # Sum over batch and sequence length dimensions
            expert_assignments = router_mask.sum(dim=[0, 1])
            self.expert_counts[f"layer_{layer_idx}"] += expert_assignments
            self.total_tokens += router_mask.shape[0] * router_mask.shape[1]

    def get_stats(self):
        """Compute summary statistics for logging"""
        stats = {}

        for layer_name, counts in self.expert_counts.items():
            total_layer_calls = counts.sum().item()
            if total_layer_calls > 0:
                # Compute usage percentages
                usage_pct = (counts / total_layer_calls * 100).cpu().numpy()

                # Overall layer stats
                stats[f"{layer_name}/total_calls"] = total_layer_calls
                stats[f"{layer_name}/max_usage_pct"] = usage_pct.max()
                stats[f"{layer_name}/min_usage_pct"] = usage_pct.min()
                stats[f"{layer_name}/usage_std"] = usage_pct.std()

                # Individual expert usage
                for expert_idx, usage in enumerate(usage_pct):
                    stats[f"{layer_name}/expert_{expert_idx}_pct"] = usage

                # Compute Gini coefficient for load balance
                sorted_usage = np.sort(usage_pct)
                n = len(sorted_usage)
                index = np.arange(1, n + 1)
                gini = (np.sum((2 * index - n - 1) * sorted_usage)) / (
                    n * np.sum(sorted_usage)
                )
                stats[f"{layer_name}/gini_coefficient"] = gini

        return stats


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
        """
        Save model checkpoint and training state

        Args:
            model: The model to save
            optimizer: The optimizer state to save
            scheduler: The scheduler state to save
            epoch: Current epoch number
            step: Current step number
            save_dir: Directory to save the checkpoint
        """
        model_to_save = model.module if hasattr(model, "module") else model

        # Get strategy name from class name
        strategy_name = self.strategy.__class__.__name__

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "strategy": strategy_name,  # Add strategy name to checkpoint
        }

        # Include strategy name in the checkpoint filename
        save_path = f"{save_dir}/checkpoint_{strategy_name}_epoch{epoch}_step{step}.pt"
        torch.save(checkpoint, save_path)

        # Also update the latest checkpoint with strategy name
        latest_path = f"{save_dir}/checkpoint_{strategy_name}_latest.pt"
        torch.save(checkpoint, latest_path)

    def _compute_perplexity(self, logits, targets, ignore_index=-100):
        """
        Compute perplexity with numerical stability safeguards
        """
        with torch.no_grad():
            logits = logits

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
        aux_loss_weight=0.1,
        num_aux_heads=0,
        use_moe=True,
        dataset_source=None,
    ):
        """Memory-optimized training with enhanced debugging information"""
        model.train()
        grad_norm_total = 0
        num_steps = 0
        final_total_loss = 0.0
        final_main_loss = 0.0
        final_aux_losses = [0.0] * num_aux_heads

        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        is_instruction_mode = isinstance(self.strategy, InstructionFollowingStrategy)

        for batch_idx, batch in enumerate(progress_bar):
            # Transfer data to device
            with torch.cuda.stream(torch.cuda.Stream()):
                input_ids = batch["inputs"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                future_targets = [
                    t.to(device, non_blocking=True) for t in batch["future_targets"]
                ]

            is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps != 0

            # Debug information logging
            if gen_every_n_steps and self.global_step % gen_every_n_steps == 0:
                # Temporarily set model to eval mode for generation
                model.eval()

                sample_idx = torch.randint(0, input_ids.shape[0], (1,)).item()
                input_tokens = self._remove_padding(
                    input_ids[sample_idx].cpu().tolist()
                )
                target_tokens = self._remove_padding(targets[sample_idx].cpu().tolist())

                print("\n" + "=" * 80)
                print(f"Step {self.global_step} Debug Information:")
                print("=" * 80)

                if is_instruction_mode:
                    input_text = tokenizer.decode(input_tokens)
                    print("Complete input text:", input_text, "\n\n", "-" * 80)

                    system, instruction, response, input_context = (
                        self._parse_instruction_format(input_text)
                    )

                    # Pass input_context to debug logger
                    self._log_instruction_debug(
                        model,
                        tokenizer,
                        input_tokens,
                        instruction,
                        response,
                        system,
                        input_context,
                    )
                else:
                    self._log_next_token_debug(
                        model,
                        tokenizer,
                        input_tokens,
                        target_tokens,
                        self.strategy,
                        future_targets,
                        sample_idx,
                        num_aux_heads,
                    )
                print("=" * 80 + "\n")

                # Set model back to training mode
                model.train()

            # Forward pass and loss computation
            with torch.cuda.amp.autocast():
                outputs = model(input_ids)
                if num_aux_heads > 0:
                    main_logits, aux_outputs = outputs
                else:
                    main_logits = outputs
                    aux_outputs = []

                main_loss = self.strategy.compute_loss(main_logits, targets)
                aux_losses = []
                if num_aux_heads > 0:
                    for aux_output, future_target in zip(aux_outputs, future_targets):
                        aux_loss = self.strategy.compute_loss(aux_output, future_target)
                        aux_losses.append(aux_loss)

                total_loss = main_loss
                if aux_losses:
                    aux_loss_sum = sum(aux_losses) / len(aux_losses)
                    total_loss = main_loss + (aux_loss_weight * aux_loss_sum)

                scaled_loss = total_loss / gradient_accumulation_steps

            # Backward pass
            scaler.scale(scaled_loss).backward()

            # Optimization step
            if not is_accumulation_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                grad_norm_total += grad_norm.item()
                num_steps += 1

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Update progress bar
                progress_bar.set_postfix(
                    self._get_progress_bar_metrics(
                        total_loss, main_loss, scheduler, aux_losses
                    )
                )

                # Regular logging
                if self.global_step % log_every_n_steps == 0:
                    metrics = {
                        "train/main_loss": main_loss.item(),
                        "train/total_loss": total_loss.item(),
                        "train/perplexity": torch.exp(main_loss).item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/grad_norm_avg": grad_norm_total / max(1, num_steps),
                        "train/global_step": self.global_step,
                    }
                    if dataset_source:
                        metrics["train/dataset_source"] = dataset_source
                    if num_aux_heads > 0:
                        for idx, aux_loss in enumerate(aux_losses):
                            metrics[f"train/aux_loss_{idx+1}"] = aux_loss.item()
                    self._log_metrics(metrics, self.global_step)

                # Save checkpoint if needed
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

            # Cleanup
            del outputs, main_loss
            if aux_losses:
                del aux_losses
            del total_loss, scaled_loss

        return {
            "epoch": epoch,
            "grad_norm_avg": grad_norm_total / max(1, num_steps),
            "final_total_loss": total_loss.item(),
            "final_perplexity": torch.exp(main_loss).item(),
            "final_aux_losses": (
                [loss.item() for loss in aux_losses] if num_aux_heads > 0 else None
            ),
        }

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

    def _log_metrics(self, metrics, step):
        """Log metrics to wandb if available"""
        if wandb.run is not None:
            wandb.log(metrics, step=step)

    def _get_progress_bar_metrics(
        self, total_loss, main_loss, scheduler, aux_losses=None
    ):
        """Get metrics for the progress bar"""
        metrics = {
            "loss": f"{total_loss.item():.4f}",
            "ppl": f"{torch.exp(main_loss).item():.2f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        }
        if aux_losses:
            metrics["aux_avg"] = (
                f"{sum(l.item() for l in aux_losses) / len(aux_losses):.4f}"
            )
        return metrics

    def _log_instruction_debug(
        self,
        model,
        tokenizer,
        input_tokens,
        instruction_text,
        response_text=None,
        system_text=None,
        input_context=None,
    ):
        """Log debug information for instruction following mode with consistent formatting"""

        # First, log the complete raw input for debugging
        raw_input = tokenizer.decode(input_tokens)
        print("\nRaw Input Text:")
        print("-" * 40)
        print(raw_input)
        print("-" * 40)

        # Then log the parsed components
        print("\nParsed Components:")
        print("-" * 40)

        if system_text:
            print("System:")
            print(system_text.strip())
            print()

        print("Instruction:")
        print(instruction_text.strip() if instruction_text else "")
        print()

        if input_context:
            print("Input:")
            print(input_context.strip())
            print()

        if response_text:
            print("Response:")
            print(response_text.strip())

        # Token statistics section with better error handling and validation
        print("\nToken Statistics:")
        print("-" * 40)
        try:
            # For instruction tokens
            instruction_token_count = 0
            if instruction_text:
                # Add debug printing
                instruction_tokens = tokenizer.encode(instruction_text.strip())
                instruction_token_count = len(instruction_tokens)
                print(f"Instruction text being counted: '{instruction_text.strip()}'")

            # For response tokens
            response_token_count = 0
            if response_text:
                # Add debug printing
                response_tokens = tokenizer.encode(response_text.strip())
                response_token_count = len(response_tokens)
                print(f"Response text being counted: '{response_text.strip()}'")

            # For input context tokens
            input_context_token_count = 0
            if input_context:
                input_context_tokens = tokenizer.encode(input_context.strip())
                input_context_token_count = len(input_context_tokens)

            # Print all token counts
            print(f"Total tokens (without padding): {len(input_tokens)}")
            print(f"Instruction tokens: {instruction_token_count}")
            if input_context:
                print(f"Input context tokens: {input_context_token_count}")
            print(f"Response tokens: {response_token_count}")

        except Exception as e:
            print(f"Error calculating token statistics: {str(e)}")

        # Generate and log model's response if appropriate
        if response_text:
            print("\nModel Generation:")
            print("-" * 40)
            try:
                with torch.no_grad():
                    input_tensor = torch.tensor(
                        [input_tokens], device=next(model.parameters()).device
                    )
                    generated = self.strategy.generate(
                        model,
                        input_tensor,
                        max_length=100,
                        temperature=0.7,
                        top_p=0.9,
                    )
                    generated_tokens = generated[0].cpu().tolist()
                    generated_text = tokenizer.decode(generated_tokens)

                    # Extract just the model's response part
                    if "[/INST]" in generated_text:
                        generated_response = generated_text.split("[/INST]")[1].strip()
                        generated_response = generated_response.replace(
                            "</s>", ""
                        ).strip()
                        print(f"Generated response:\n{generated_response}")
                    else:
                        print(f"Complete generation:\n{generated_text}")

                    # Log special tokens for debugging
                    print("\nSpecial tokens in generation:")
                    print(f"Contains [/INST]: {'[/INST]' in generated_text}")
                    print(f"Contains </s>: {'</s>' in generated_text}")

            except Exception as e:
                print(f"Generation failed: {str(e)}")

        print("\n" + "=" * 80 + "\n")

    def _log_next_token_debug(
        self,
        model,
        tokenizer,
        input_tokens,
        target_tokens,
        strategy,
        future_targets=None,
        sample_idx=None,
        num_aux_heads=0,
    ):
        """Log debug information for next token prediction mode"""
        print("\nNext Token Prediction Mode Debug:")
        print("-" * 40)
        print(f"\nInput sequence (last 100 tokens decoded):")
        print(tokenizer.decode(input_tokens[-100:]))
        print(f"\nTarget token(s):")
        print(tokenizer.decode(target_tokens[-strategy.predict_last_n :]))

        # Generate model's prediction
        try:
            with torch.no_grad():
                input_tensor = torch.tensor(
                    [input_tokens], device=next(model.parameters()).device
                )
                generated = strategy.generate(
                    model,
                    input_tensor,
                    max_length=strategy.predict_last_n,
                    temperature=0.7,
                    top_p=0.9,
                )
                generated_tokens = generated[0].cpu().tolist()
                print(f"\nModel prediction:")
                print(tokenizer.decode(generated_tokens[-strategy.predict_last_n :]))
        except Exception as e:
            print(f"Generation failed: {str(e)}")

        print("\nPrediction Statistics:")
        print(f"Predicting last {strategy.predict_last_n} tokens")
        print(f"Input sequence length (without padding): {len(input_tokens)}")

        if hasattr(strategy, "confidence_history") and strategy.confidence_history:
            recent_confidence = strategy.confidence_history[-5:]
            print(
                f"Recent prediction confidences: {[f'{c:.3f}' for c in recent_confidence]}"
            )

        # Log auxiliary head information
        if num_aux_heads > 0 and future_targets:
            print("\nAuxiliary Head Predictions:")
            print("-" * 40)
            for i, future_target in enumerate(future_targets):
                fut_tokens = future_target[sample_idx].cpu().tolist()
                fut_tokens = self._remove_padding(fut_tokens)
                print(f"\nHead {i+1} (t+{i+2}):")
                print(f"Target: {tokenizer.decode(fut_tokens[-10:])}")

    @staticmethod
    def _remove_padding(tokens):
        """Remove padding tokens (0) from sequence"""
        try:
            first_pad = tokens.index(0)
            return tokens[:first_pad]
        except ValueError:
            return tokens

    def _parse_instruction_format(self, input_text):
        """Parse instruction format from input text"""
        try:
            # Check for required tokens
            if "[INST]" not in input_text or "[/INST]" not in input_text:
                print(f"Warning: Missing [INST] or [/INST] tokens in input")
                return None, None, None, None

            # Split on [INST] first
            main_content = input_text.split("[INST]", 1)[1]

            # Split on ### Response: to separate instruction and response parts
            if "### Response:" in main_content:
                pre_response, response_part = main_content.split("### Response:", 1)

                # Clean up the response part
                response = response_part.split("[/INST]", 1)[0].strip()

                # Process the instruction part
                if "### Instruction:" in pre_response:
                    instruction = pre_response.split("### Instruction:", 1)[1].strip()
                else:
                    instruction = pre_response.strip()

                # Handle system if present
                system = None
                if "### System:" in pre_response:
                    system_part = pre_response.split("### System:", 1)[1]
                    system = system_part.split("### Instruction:", 1)[0].strip()

                # Handle input if present
                input_context = None
                if "### Input:" in instruction:
                    instruction, input_context = instruction.split("### Input:", 1)
                    instruction = instruction.strip()
                    input_context = input_context.strip()

                # Add debug output
                print("\nParsing Results:")
                print(f"System present: {'Yes' if system else 'No'}")
                print(f"Input context present: {'Yes' if input_context else 'No'}")
                print(f"Instruction length: {len(instruction) if instruction else 0}")
                print(f"Response length: {len(response) if response else 0}")
                print("\nParsed sections preview:")
                print(
                    "Instruction:",
                    (
                        instruction[:100] + "..."
                        if len(instruction) > 100
                        else instruction
                    ),
                )
                print(
                    "Response:",
                    response[:100] + "..." if len(response) > 100 else response,
                )

                return system, instruction, response, input_context
            else:
                print("Warning: No '### Response:' marker found in text")
                return None, None, None, None

        except Exception as e:
            print(f"Error parsing instruction format: {str(e)}")
            print(f"Input text preview: {input_text[:100]}...")
            return None, None, None, None

    def monitor_expert_usage(self, model, input_ids):
        """
        Enhanced monitoring of MoE expert usage with detailed statistics.
        Returns comprehensive metrics about expert utilization, load balancing,
        and routing patterns.
        """
        # Access the underlying model if it's wrapped in DDP
        if hasattr(model, "module"):
            base_model = model.module
        else:
            base_model = model

        # Get embeddings (copied from model's forward pass)
        B, T = input_ids.size()
        total_tokens = B * T
        positions = (
            torch.arange(0, T, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .expand(B, -1)
        )

        tok_emb = base_model.token_embedding(input_ids)
        pos_emb = base_model.position_embedding(positions)
        hidden_states = tok_emb + pos_emb

        moe_metrics = {}
        layer_idx = 0

        # Monitor usage across all MoE layers
        for name, module in base_model.named_modules():
            if isinstance(module, MoELayer):
                layer_metrics = {}

                # Get router probabilities and assignments
                router_logits = module.router(hidden_states)
                router_probs = F.softmax(router_logits, dim=-1)
                expert_assignment = router_probs.argmax(dim=-1)

                # 1. Basic expert counts and utilization
                expert_counts = {
                    f"expert_{i}_count": (expert_assignment == i).sum().item()
                    for i in range(module.num_experts)
                }
                expert_utilization = {
                    f"expert_{i}_utilization": count / total_tokens
                    for i, count in expert_counts.items()
                }

                # 2. Router confidence metrics
                # Handle batch x sequence_length dimensions properly
                top_probs, _ = router_probs.max(dim=-1)  # shape: [batch_size, seq_len]
                router_confidence = {
                    "mean_confidence": top_probs.mean().item(),  # mean over all tokens
                    "min_confidence": top_probs.min().item(),  # min over all tokens
                    "max_confidence": top_probs.max().item(),  # max over all tokens
                }

                # 3. Load balancing metrics
                # Convert expert counts to a 1D tensor for proper statistics
                expert_counts_list = [
                    expert_counts[f"expert_{i}_count"]
                    for i in range(module.num_experts)
                ]
                expert_counts_tensor = torch.tensor(
                    expert_counts_list, device=input_ids.device, dtype=torch.float32
                )

                # Calculate load balancing metrics safely
                mean_count = expert_counts_tensor.mean()
                if mean_count > 0:
                    cv = expert_counts_tensor.std() / mean_count
                    max_count = expert_counts_tensor.max()
                    min_count = expert_counts_tensor.clamp(min=1.0).min()
                    max_to_min = max_count / min_count
                else:
                    cv = torch.tensor(0.0, device=input_ids.device)
                    max_to_min = torch.tensor(1.0, device=input_ids.device)

                load_balancing = {
                    "coefficient_of_variation": cv.item(),
                    "max_to_min_ratio": max_to_min.item(),
                    "mean_tokens_per_expert": mean_count.item(),
                    "max_tokens_per_expert": expert_counts_tensor.max().item(),
                    "min_tokens_per_expert": expert_counts_tensor.min().item(),
                }

                # 4. Routing pattern analysis
                # Calculate entropy while handling dimensions properly
                probs_entropy = -(
                    router_probs * torch.log(router_probs + 1e-10)
                )  # [batch, seq_len, num_experts]
                entropy = probs_entropy.sum(dim=-1)  # Sum over experts dimension
                mean_entropy = entropy.mean()  # Mean over batch and sequence length

                unused_experts = sum(
                    1 for count in expert_counts.values() if count == 0
                )

                routing_patterns = {
                    "router_entropy": mean_entropy.item(),
                    "unused_expert_count": unused_experts,
                    "most_used_expert": expert_counts_tensor.argmax().item(),
                    "least_used_expert": expert_counts_tensor.argmin().item(),
                }

                # 5. Capacity metrics (if using capacity limiting)
                if hasattr(module, "capacity_factor"):
                    capacity = int(module.capacity_factor * total_tokens)
                    over_capacity = {
                        f"expert_{i}_over_capacity": max(
                            0, expert_counts[f"expert_{i}_count"] - capacity
                        )
                        for i in range(module.num_experts)
                    }
                    layer_metrics.update({"capacity_metrics": over_capacity})

                # Combine all metrics for this layer
                layer_metrics.update(
                    {
                        "expert_counts": expert_counts,
                        "expert_utilization": expert_utilization,
                        "router_confidence": router_confidence,
                        "load_balancing": load_balancing,
                        "routing_patterns": routing_patterns,
                    }
                )

                moe_metrics[f"moe_layer_{layer_idx}"] = layer_metrics
                layer_idx += 1

                # Update hidden states for next layer if needed
                hidden_states = module(hidden_states)[
                    0
                ]  # Assuming returns (output, aux_loss)

        return moe_metrics

    def _generate_and_log_samples(
        self, model, tokenizer, sample_prompts, max_gen_length, device
    ):
        """Generate and log sample outputs with tiktoken tokenizer support"""
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
                        # Handle MoE model outputs
                        model_output = model(input_tensor)

                        # Unpack logits from MoE output if necessary
                        if isinstance(model_output, tuple):
                            logits, _ = model_output
                        else:
                            logits = model_output

                        # Get next token logits and apply temperature
                        next_token_logits = logits[0, -1, :] / temperature
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                        # Check for EOT token (50256 for GPT-2)
                        if next_token.item() == 50256:  # tiktoken's EOT token
                            break

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
        log_data = {
            f"generation/sample_{i}": {
                "prompt": gen["prompt"],
                "generated": gen["generated"],
            }
            for i, gen in enumerate(generations)
        }
        wandb.log(log_data, step=self.global_step)

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

    def set_global_step(self, step):
        """Set the global step counter"""
        self.global_step = step

    def evaluate(self, model, val_loader, device):
        """
        Evaluate model on validation dataset with proper handling of auxiliary losses.
        """
        model.eval()
        total_loss = 0.0
        total_aux_loss = 0.0
        total_perplexity = 0.0
        num_batches = len(val_loader)

        # Store original step
        original_step = self.global_step

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                try:
                    input_ids = batch["inputs"].to(device)
                    targets = batch["targets"].to(device)
                    future_targets = [t.to(device) for t in batch["future_targets"]]

                    if input_ids.shape[0] == 0:
                        continue

                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(input_ids)

                        # Check if model returns auxiliary outputs
                        if isinstance(outputs, tuple):
                            main_logits, aux_outputs = outputs
                            # Calculate auxiliary losses
                            aux_losses = []
                            for aux_output, future_target in zip(
                                aux_outputs, future_targets
                            ):
                                aux_loss = self.strategy.compute_loss(
                                    aux_output, future_target
                                )
                                if torch.isfinite(aux_loss):
                                    aux_losses.append(aux_loss)

                            # Average auxiliary losses if any are finite
                            if aux_losses:
                                batch_aux_loss = sum(aux_losses) / len(aux_losses)
                            else:
                                batch_aux_loss = torch.tensor(0.0, device=device)
                        else:
                            main_logits = outputs
                            batch_aux_loss = torch.tensor(0.0, device=device)

                        # Apply temperature scaling to logits
                        temperature = 1.2
                        scaled_logits = main_logits / temperature

                        # Compute main task loss
                        main_loss = self.strategy.compute_loss(scaled_logits, targets)

                        # Compute perplexity using scaled logits
                        ppl = self._compute_perplexity(scaled_logits, targets)

                        if torch.isfinite(main_loss):
                            total_loss += main_loss.item()
                            total_aux_loss += batch_aux_loss.item()
                            total_perplexity += ppl

                except RuntimeError as e:
                    print(f"Error during evaluation: {e}")
                    continue

        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_aux_loss = total_aux_loss / num_batches
        avg_perplexity = total_perplexity / num_batches

        # Store all metrics
        metrics = {
            "total_loss": avg_loss,
            "primary_loss": avg_loss,
            "auxiliary_loss": avg_aux_loss,
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


class CurriculumTrainer(SingleHeadTrainer):
    def __init__(
        self, strategy: "ModelingStrategy", curriculum_schedule: Optional[Dict] = None
    ):
        super().__init__(strategy)
        self.curriculum_schedule = curriculum_schedule or {}
        self.current_curriculum_step = 0

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
        aux_loss_weight=0.1,
    ):
        """Memory-optimized training for one epoch with MoE support"""
        model.train()

        # Training monitoring
        grad_norm_total = 0
        num_steps = 0

        # Track final metrics
        final_total_loss = 0.0
        final_main_loss = 0.0

        # Get dataset for curriculum updates
        dataset = train_loader.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset

        # Batch accumulation state
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Efficient device transfer
            with torch.cuda.stream(torch.cuda.Stream()):
                input_ids = batch["inputs"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)

            is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps != 0

            # Compute loss with mixed precision
            with torch.cuda.amp.autocast():
                # MoE model returns both logits and auxiliary loss
                outputs, aux_loss = model(input_ids)

                # Main task loss
                main_loss = self.strategy.compute_loss(outputs, targets)

                # Combine losses with weighting
                total_loss = main_loss + aux_loss_weight * aux_loss

                # Update final metrics
                final_total_loss = total_loss.item()
                final_main_loss = main_loss.item()

                # Scale for gradient accumulation
                scaled_loss = total_loss / gradient_accumulation_steps

            # Backward pass
            scaler.scale(scaled_loss).backward()

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
                    expert_metrics = self.monitor_expert_usage(model, input_ids)

                    # Base training metrics
                    metrics = {
                        "train/main_loss": main_loss.item(),
                        "train/aux_loss": aux_loss.item(),
                        "train/total_loss": total_loss.item(),
                        "train/perplexity": torch.exp(main_loss).item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/grad_norm_avg": grad_norm_total / max(1, num_steps),
                        "train/global_step": self.global_step,
                    }

                    # Add all expert metrics
                    for layer_name, layer_metrics in expert_metrics.items():
                        for category, category_metrics in layer_metrics.items():
                            for metric_name, value in category_metrics.items():
                                metrics[f"{layer_name}/{category}_{metric_name}"] = (
                                    value
                                )

                    # Add curriculum metrics if applicable
                    if isinstance(dataset, CurriculumDataset):
                        metrics.update(
                            {
                                "curriculum/step": self.current_curriculum_step,
                                "curriculum/active_files": len(dataset.active_files),
                            }
                        )

                    # Log to wandb
                    wandb.log(metrics, step=self.global_step)

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": f"{total_loss.item():.4f}",
                        "aux_loss": f"{aux_loss.item():.4f}",
                        "ppl": f"{torch.exp(main_loss).item():.2f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

                # Check curriculum update
                if isinstance(dataset, CurriculumDataset) and self.curriculum_schedule:
                    steps_per_file = self.curriculum_schedule.get(
                        "steps_per_file", float("inf")
                    )
                    if (
                        self.global_step // steps_per_file
                        > self.current_curriculum_step
                    ):
                        if dataset.add_next_file_pattern():
                            self.current_curriculum_step += 1
                            is_main_process = (
                                not torch.distributed.is_initialized()
                                or torch.distributed.get_rank() == 0
                            )
                            if is_main_process:
                                print(
                                    f"\nUpdated curriculum. Now training on {len(dataset.active_files)} files"
                                )
                            progress_bar.total = len(train_loader)
                            progress_bar.refresh()

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
            del main_loss
            del aux_loss
            del total_loss
            del scaled_loss

        # Calculate final perplexity
        final_perplexity = torch.exp(torch.tensor(final_main_loss)).item()

        return {
            "epoch": epoch,
            "grad_norm_avg": grad_norm_total / max(1, num_steps),
            "final_total_loss": final_total_loss,
            "final_perplexity": final_perplexity,
        }
