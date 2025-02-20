import os, math, html
import torch
from tqdm.auto import tqdm
import wandb
import random
import torch.distributed as dist
import torch.nn.functional as F
from model import MoELayer, MoETransformerBlock
from typing import List, Dict, Optional, Union, Tuple
from strategies import InstructionFollowingStrategy


def safe_decode(tokenizer, tokens, max_length=10000, pad_token_id=0):
    """Safely decode tokens with error handling"""
    try:
        # Convert to list if tensor
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        # Filter out padding and special tokens if needed
        tokens = [t for t in tokens if t != pad_token_id]
        decoded = tokenizer.decode(tokens)
        # Truncate if needed
        return decoded[:max_length] + ("..." if len(decoded) > max_length else "")
    except Exception as e:
        return f"<decode_error: {str(e)}>"


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

    def train_epoch(
        self,
        model,
        train_loader,
        optimizer,
        scheduler,
        scaler,
        device,
        epoch,
        block_size,
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

        # Create persistent streams for data transfer and computation
        compute_stream = torch.cuda.default_stream()

        # Track final metrics
        final_metrics = {"total_loss": None, "main_loss": None, "aux_losses": None}

        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        is_instruction_mode = isinstance(self.strategy, InstructionFollowingStrategy)

        # Pre-allocate tensors for accumulation
        accumulated_loss = torch.zeros(1, device=device)

        for batch_idx, batch in enumerate(progress_bar):
            # Transfer data to device - moved outside stream context for guaranteed access
            input_ids = batch["inputs"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"]
            instruction_mask = batch["instruction_mask"].to(device, non_blocking=True)
            future_targets = [
                t.to(device, non_blocking=True) for t in batch["future_targets"]
            ]

            # Debug logging for first two batches
            # if batch_idx < 2:
            #     print(f"\n--- Batch {batch_idx} ---")
            #     input_tokens_list = batch["inputs"][0].tolist()
            #     target_tokens_list = batch["targets"][0].tolist()
            #     print(
            #         "Input Text (decoded, first example):",
            #         safe_decode(
            #             tokenizer, input_tokens_list, self.strategy.pad_token_id
            #         ),
            #     )
            #     print(
            #         "Target Text (decoded, first example):",
            #         safe_decode(
            #             tokenizer, target_tokens_list, self.strategy.pad_token_id
            #         ),
            #     )

            # Handle generation and debugging if needed
            if gen_every_n_steps and self.global_step % gen_every_n_steps == 0:
                model.eval()
                with torch.no_grad():
                    input_tokens = batch["inputs"][0].cpu().tolist()
                    target_tokens = batch["targets"][0].cpu().tolist()

                    if is_instruction_mode:

                        self._log_instruction_debug(
                            model,
                            tokenizer,
                            input_tokens,
                            attention_mask=attention_mask[0],
                            debug_level="basic",
                        )
                    else:
                        print(
                            "\n=== Debug Logging at Step {} ===".format(
                                self.global_step
                            )
                        )
                        print("Input/Target Token Check:")
                        print("Input tokens (last 5):", input_tokens[-5:])
                        print("Target tokens (last 5):", target_tokens[-5:])
                        print(
                            "Input (last 5) decoded:",
                            tokenizer.decode(input_tokens[-25:]),
                        )
                        print(
                            "Target (last 5) decoded:",
                            tokenizer.decode(target_tokens[-25:]),
                        )

                        self._log_next_token_debug(
                            model,
                            tokenizer,
                            input_tokens,
                            target_tokens,
                            self.strategy,
                            future_targets=(
                                [ft[0] for ft in future_targets]
                                if future_targets
                                else None
                            ),
                            sample_idx=0,
                            num_aux_heads=num_aux_heads,
                            block_size=block_size,
                        )
                model.train()

            # Synchronize data transfer before computation
            # compute_stream.wait_stream(data_stream)

            with torch.cuda.stream(compute_stream):
                is_accumulation_step = (
                    batch_idx + 1
                ) % gradient_accumulation_steps != 0

                # Forward pass and loss computation
                with torch.cuda.amp.autocast():
                    outputs = model(idx=input_ids, attention_mask=attention_mask)

                    if num_aux_heads > 0:
                        main_logits, aux_outputs = outputs
                    else:
                        main_logits = outputs
                        aux_outputs = []

                    main_loss, aux_losses = self.strategy.compute_all_losses(
                        main_logits,
                        aux_outputs,
                        targets,
                        future_targets,
                        instruction_mask=instruction_mask,
                    )

                    total_loss = main_loss
                    if aux_losses:
                        aux_loss_sum = sum(aux_losses) / len(aux_losses)
                        total_loss = main_loss + (aux_loss_weight * aux_loss_sum)

                    # Store metrics for logging
                    final_metrics.update(
                        {
                            "total_loss": total_loss.item(),
                            "main_loss": main_loss.item(),
                            "aux_losses": (
                                [loss.item() for loss in aux_losses]
                                if aux_losses
                                else None
                            ),
                        }
                    )

                    # Scale loss for gradient accumulation
                    scaled_loss = total_loss / gradient_accumulation_steps

                # Backward pass
                scaled_loss = scaler.scale(scaled_loss)
                scaled_loss.backward()

                # Update metrics for logging
                with torch.no_grad():
                    accumulated_loss += total_loss.detach()

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

                    # Reset accumulation
                    accumulated_loss.zero_()

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

                    self.global_step += 1

                # Cleanup
                del outputs, main_loss
                if aux_losses:
                    del aux_losses
                del total_loss, scaled_loss

        # Return metrics using stored values
        return {
            "epoch": epoch,
            "grad_norm_avg": grad_norm_total / max(1, num_steps),
            "final_total_loss": final_metrics["total_loss"],
            "final_perplexity": math.exp(final_metrics["main_loss"]),
            "final_aux_losses": final_metrics["aux_losses"],
        }

    def _compute_perplexity(self, logits, targets):
        """
        Compute perplexity using strategy's compute_loss
        """
        with torch.no_grad():
            loss = self.strategy.compute_loss(logits, targets)
            return torch.exp(loss).item()

    def _log_prediction_analysis(
        self, tokenizer, input_seq, main_pred, aux_preds, targets
    ):
        print("\nPrediction Analysis:")
        print(f"Input context (last 5 tokens): {tokenizer.decode(input_seq[-5:])}")
        print(f"Main head (t+1) prediction: {tokenizer.decode([main_pred])}")
        print(f"Expected t+1: {tokenizer.decode([targets[0]])}")

        for idx, (aux_pred, target) in enumerate(zip(aux_preds, targets[1:])):
            print(
                f"Aux head {idx+1} (t+{idx+2}) prediction: {tokenizer.decode([aux_pred])}"
            )
            print(f"Expected t+{idx+2}: {tokenizer.decode([target])}")

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
        tokenizer,  # tiktoken tokenizer
        input_tokens,
        attention_mask=None,
        debug_level="basic",
    ):
        """
        Debug logging for instruction following mode with tiktoken tokenizer.
        Basic mode shows parsed components and model generation, while detailed mode adds token analysis.

        Args:
            model: The model being used for generation
            tokenizer: tiktoken tokenizer
            input_tokens: Full sequence of input tokens
            instruction: The parsed instruction text
            response: The target response parsed from raw text
            system: The parsed system prompt
            input_context: Any additional context
            attention_mask: Attention mask (optional)
            debug_level: "basic" or "detailed" (default: "basic")
        """
        # In _log_instruction_debug, modify the input preparation:
        if isinstance(input_tokens, list):
            # Remove padding tokens from the end of the list
            while input_tokens and input_tokens[-1] == self.strategy.pad_token_id:
                input_tokens.pop()
            input_tensor = torch.tensor(
                [input_tokens], device=next(model.parameters()).device
            )
        else:
            # If it's already a tensor, remove padding
            padding_mask = input_tokens != self.strategy.pad_token_id
            if not padding_mask.all():
                last_token_pos = padding_mask.nonzero()[-1]
                input_tokens = input_tokens[: last_token_pos + 1]
            input_tensor = (
                input_tokens.unsqueeze(0) if input_tokens.dim() == 1 else input_tokens
            )

        # Debug input shape after padding removal
        print(f"Debug: Input tensor shape after padding removal: {input_tensor.shape}")

        raw_text = tokenizer.decode(input_tokens)
        system, instruction, response, input_context = self._parse_instruction_format(
            raw_text
        )
        print("\n" + "=" * 80)
        print(f"INSTRUCTION DEBUG LOG - {debug_level.upper()} MODE")
        print("=" * 80)

        if debug_level == "basic":
            # Show the parsed components in a clear format
            if system:
                print("\n📋 SYSTEM PROMPT:")
                print("-" * 40)
                print(system.strip())

            print("\n💭 INSTRUCTION:")
            print("-" * 40)
            print(instruction.strip())

            if input_context:
                print("\n📑 CONTEXT:")
                print("-" * 40)
                print(input_context.strip())

            if response:
                print("\n🎯 TARGET RESPONSE:")
                print("-" * 40)
                print(response.strip())

            # Generate model response
            print("\n🤖 MODEL GENERATION:")
            print("-" * 40)
            try:
                with torch.no_grad():
                    # Get the actual model from DDP if needed
                    if hasattr(model, "module"):
                        generation_model = model.module
                    else:
                        generation_model = model

                    # Ensure input_tokens is a tensor
                    if isinstance(input_tokens, list):
                        input_tensor = torch.tensor(
                            [input_tokens], device=next(model.parameters()).device
                        )
                    else:
                        input_tensor = (
                            input_tokens.unsqueeze(0)
                            if input_tokens.dim() == 1
                            else input_tokens
                        )

                    # Debug input shape
                    print(f"Debug: Input tensor shape: {input_tensor.shape}")
                    print(
                        f"Debug: Attention mask shape: {attention_mask.shape if attention_mask is not None else 'None'}"
                    )

                    # Use strategy's generate method with proper parameters
                    outputs = self.strategy.generate(
                        model=generation_model,
                        input_ids=input_tensor,
                        max_length=128,
                        top_p=0.9,
                        temperature=0.7,
                        debug=True,
                    )

                    # Debug output shape
                    print(
                        f"Debug: Output shape: {outputs.shape if isinstance(outputs, torch.Tensor) else len(outputs)}"
                    )

                    # Convert outputs to list if it's a tensor
                    if isinstance(outputs, torch.Tensor):
                        outputs = outputs.cpu().tolist()

                    # If outputs is a list of lists, take the first sequence
                    if isinstance(outputs, list) and isinstance(outputs[0], list):
                        outputs = outputs[0]

                    # Get only the new tokens (excluding input)
                    input_length = (
                        len(input_tokens)
                        if isinstance(input_tokens, list)
                        else input_tokens.shape[-1]
                    )
                    new_tokens = outputs[input_length:]

                    # Debug token counts
                    print(f"Debug: Input length: {input_length}")
                    print(f"Debug: Output length: {len(outputs)}")
                    print(f"Debug: New tokens length: {len(new_tokens)}")

                    # Debug token values
                    print(
                        f"Debug: Last input token: {input_tokens[-1] if isinstance(input_tokens, list) else input_tokens[0, -1].item()}"
                    )
                    if new_tokens:
                        print(f"Debug: First new token: {new_tokens[0]}")

                    # Decode and print the new tokens
                    if new_tokens:
                        generated_text = tokenizer.decode(new_tokens)
                        print("\nGenerated text:")
                        print("-" * 20)
                        print(generated_text.strip())
                    else:
                        print("\nWarning: No new tokens generated!")
                        print(
                            "Debug: Check if input ends with end_token_id or pad_token_id"
                        )
                        print(f"Strategy pad_token_id: {self.strategy.pad_token_id}")
                        print(f"Strategy end_token_id: {self.strategy.end_token_id}")

            except Exception as e:
                print(f"Error during generation: {str(e)}")
                import traceback

                print(f"Full traceback:")
                print(traceback.format_exc())

            print("\n🧮 MODEL INPUT SUMMARY:")
            print("-" * 40)
            total_tokens = len(input_tokens)
            print(f"Total tokens: {total_tokens}")

            print("\n" + "=" * 80 + "\n")
            return

        # Detailed mode token analysis
        raw_input = tokenizer.decode(input_tokens)
        print("\n[1] RAW INPUT & COMPONENTS")
        print("-" * 40)
        print("Complete tokenized input:")
        print(f"```\n{raw_input}\n```")

        # Token Analysis
        print("\n[2] TOKEN ANALYSIS")
        print("-" * 40)
        decoded_tokens = []
        for t in input_tokens:
            try:
                decoded = tokenizer.decode([t])
                decoded = decoded.replace("▁", "_")
                decoded_tokens.append(decoded)
            except Exception as e:
                decoded_tokens.append(f"<error-decoding-{t}>")

        # Show attention patterns if mask provided
        if attention_mask is not None:
            print("\nAttention mask shape:", attention_mask.shape)
            print("\nToken-level visualization with attention patterns:")
            print(
                "Format: Each row shows which tokens the token can attend to (✓=can attend, ×=masked)"
            )

            if len(attention_mask.shape) == 2:
                num_tokens = len(decoded_tokens)
                chunk_size = 50

                for chunk_start in range(0, num_tokens, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_tokens)

                    print("\nTokens {}-{}:".format(chunk_start, chunk_end - 1))
                    print(
                        "\n     | "
                        + "".join(f"{i:3d}" for i in range(chunk_start, chunk_end))
                    )
                    print("-----+" + "---" * (chunk_end - chunk_start))

                    mask_values = attention_mask.cpu()
                    for i in range(num_tokens):
                        row_markers = mask_values[i][chunk_start:chunk_end].tolist()
                        attention_pattern = "".join(
                            "✓  " if m == 1 else "×  " for m in row_markers
                        )
                        print(
                            f"{i:3d} | {attention_pattern} | {repr(decoded_tokens[i])}"
                        )

                    print("\n" + "-" * 80)
            else:
                for i, (token, mask) in enumerate(
                    zip(decoded_tokens, attention_mask.cpu().tolist())
                ):
                    mask_indicator = "✓" if mask == 1 else "×"
                    print(f"{i:3d} | {mask_indicator} | {repr(token)}")
        else:
            print("\nNote: No attention mask provided")
            for i, token in enumerate(decoded_tokens):
                print(f"{i:3d} | - | {repr(token)}")

        # Special Tokens Analysis
        print("\n[3] SPECIAL TOKENS SUMMARY")
        print("-" * 40)
        special_tokens = {
            "[INST]": raw_input.count("[INST]"),
            "[/INST]": raw_input.count("[/INST]"),
            "<s>": raw_input.count("<s>"),
            "</s>": raw_input.count("</s>"),
        }

        # Add EOT token if present
        if hasattr(tokenizer, "eot_token"):
            eot_str = tokenizer.decode([tokenizer.eot_token])
            special_tokens[eot_str] = len(
                [t for t in input_tokens if t == tokenizer.eot_token]
            )

        print("\nSpecial token counts:")
        for token, count in special_tokens.items():
            print(f"{token}: {count}")

        # 4. Token Distribution Analysis
        print("\n[4] TOKEN DISTRIBUTION ANALYSIS")
        print("-" * 40)

        # Count token frequencies
        token_counts = {}
        for token in input_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # Sort by frequency
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

        print("\nTop 20 most frequent tokens:")
        for token_id, count in sorted_tokens[:20]:
            try:
                token_text = tokenizer.decode([token_id])
                print(f"Token {token_id}: {repr(token_text)} - {count} occurrences")
            except Exception as e:
                print(f"Token {token_id}: <decode-error> - {count} occurrences")

        # 5. Sequence Structure Analysis
        print("\n[5] SEQUENCE STRUCTURE ANALYSIS")
        print("-" * 40)

        # Analyze sequence length
        print(f"\nTotal sequence length: {len(input_tokens)}")

        # Find potential instruction boundaries
        inst_starts = [i for i, t in enumerate(decoded_tokens) if "[INST]" in t]
        inst_ends = [i for i, t in enumerate(decoded_tokens) if "[/INST]" in t]

        print("\nInstruction boundaries:")
        for start, end in zip(inst_starts, inst_ends):
            if end > start:
                print(
                    f"Instruction span: tokens {start} to {end} ({end-start+1} tokens)"
                )
                try:
                    inst_text = tokenizer.decode(input_tokens[start : end + 1])
                    print(f"Content: {repr(inst_text)}")
                except Exception as e:
                    print(f"Content: <decode-error>")

        # 6. Model Input Structure
        print("\n[6] MODEL INPUT STRUCTURE")
        print("-" * 40)

        # Analyze input shape and device
        if isinstance(input_tokens, torch.Tensor):
            print(f"\nInput tensor shape: {input_tokens.shape}")
            print(f"Input device: {input_tokens.device}")
            print(f"Input dtype: {input_tokens.dtype}")
        else:
            print("\nInput is not a tensor (type: {type(input_tokens)})")

        # Memory usage estimation
        try:
            if isinstance(input_tokens, torch.Tensor):
                mem_bytes = input_tokens.element_size() * input_tokens.nelement()
                print(f"Approximate memory usage: {mem_bytes/1024:.2f} KB")
        except Exception as e:
            print(f"Could not calculate memory usage: {str(e)}")

        print("\n" + "=" * 80 + "\n")

    @staticmethod
    def _remove_padding(tokens):
        """Remove padding tokens (0) from sequence"""
        # Handle case where tokens is a single integer
        if isinstance(tokens, int):
            return [tokens] if tokens != 0 else []

        # Handle case where tokens is a list or tensor
        try:
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.cpu().tolist()

            first_pad = tokens.index(0)
            return tokens[:first_pad]
        except (ValueError, AttributeError):
            return tokens

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
        block_size=512,  # Added block_size parameter with default value
    ):
        """Log debug information for next token prediction mode with clear token alignment

        Args:
            model: The model to use for prediction
            tokenizer: The tokenizer for encoding/decoding
            input_tokens: List of input token ids
            target_tokens: List of target token ids
            strategy: The prediction strategy
            future_targets: Optional future target tokens for auxiliary heads
            sample_idx: Optional sample index
            num_aux_heads: Number of auxiliary prediction heads (default: 0)
            block_size: Maximum sequence length for the model (default: 512)
        """
        print("\nNext Token Prediction Mode Debug:")
        print("-" * 40)

        # Show the last few tokens of input for context
        context_size = 20  # Show last 20 tokens for context
        print("\nInput context (last few tokens):")
        last_tokens = input_tokens[-context_size:]
        context_text = tokenizer.decode(last_tokens)
        print(f"...{context_text}")

        # Show the exact last input token and its target
        last_input_token = input_tokens[-1]
        target_token = target_tokens[
            -1
        ]  # First target token corresponds to last input token

        print("\nToken-by-token alignment:")
        print(f"Last input token: '{tokenizer.decode([last_input_token])}'")
        print(f"Expected next token: '{tokenizer.decode([target_token])}'")

        try:
            with torch.no_grad():
                input_tensor = torch.tensor(
                    [input_tokens], device=next(model.parameters()).device
                )

                # Generate model's initial prediction
                outputs = model(input_tensor)
                if isinstance(outputs, tuple):
                    logits, _ = outputs
                else:
                    logits = outputs

                # Get predictions for the last position
                last_pos_logits = logits[0, -1, :]
                probs = F.softmax(last_pos_logits, dim=-1)

                # Get top 5 predictions with probabilities
                top_probs, top_tokens = torch.topk(probs, k=5)

                print("\nModel's top 5 predictions for next token:")
                print("-" * 40)
                for prob, token in zip(top_probs.tolist(), top_tokens.tolist()):
                    token_text = tokenizer.decode([token])
                    print(f"'{token_text}': {prob:.4f} probability")
                    if token == target_token:
                        print("  ^ This is the correct token!")

                # Generate next 50 tokens
                print("\nGenerating next 50 tokens:")
                print("-" * 40)

                # Start with a truncated version of input tokens if needed
                tokens_to_generate = 50
                max_context = (
                    block_size - tokens_to_generate
                )  # Reserve space for generation
                generated_tokens = input_tokens[-max_context:].copy()
                generated_text = []

                for _ in range(tokens_to_generate):
                    # Get model prediction for next token
                    current_input = torch.tensor(
                        [generated_tokens], device=next(model.parameters()).device
                    )
                    with torch.no_grad():
                        outputs = model(current_input)
                        if isinstance(outputs, tuple):
                            logits, _ = outputs
                        else:
                            logits = outputs

                        next_token_logits = logits[0, -1, :]

                        # Apply temperature sampling
                        temperature = 0.7  # Adjustable temperature parameter
                        scaled_logits = next_token_logits / temperature
                        probs = F.softmax(scaled_logits, dim=-1)

                        # Sample next token
                        next_token = torch.multinomial(probs, num_samples=1).item()

                        # Decode and store the token
                        token_text = tokenizer.decode([next_token])
                        generated_text.append(token_text)

                        # Add token to generated sequence
                        generated_tokens.append(next_token)

                        # Truncate sequence if needed
                        if len(generated_tokens) >= max_context:
                            generated_tokens = generated_tokens[-max_context:]

                # Print generated sequence as a single flowing text
                print("\nGenerated sequence:")
                print("".join(generated_text))

        except Exception as e:
            print(f"Generation failed: {str(e)}")
            print(f"Current sequence length: {len(generated_tokens)}")
            print(f"Model's maximum sequence length: {block_size}")

        # Show auxiliary head predictions if available
        # if num_aux_heads > 0 and future_targets:
        #     print("\nAuxiliary Head Predictions:")
        #     print("-" * 40)

        #     for i, future_target in enumerate(future_targets, 1):
        #         try:
        #             if isinstance(future_target, torch.Tensor):
        #                 future_target = future_target.cpu().tolist()

        #             print(f"\nHead {i} (predicting token t+{i+1}):")
        #             print(f"Target: '{tokenizer.decode([future_target[0]])}'")

        #             # Show the sequence of predictions
        #             print(
        #                 f"Sequence: Current → {' → '.join(tokenizer.decode([t]) for t in [target_token] + future_targets[:i][0].cpu().tolist())}"
        #             )

        #         except Exception as e:
        #             print(f"Error processing auxiliary head {i}: {str(e)}")

        print("\nPrediction Statistics:")
        print("-" * 40)
        print(f"Original input length: {len(input_tokens)}")
        print(f"Working sequence length: {len(generated_tokens)}")
        print(f"Number of auxiliary heads: {num_aux_heads}")
        print(f"Number of generated tokens: {len(generated_text)}")
        print(f"Block size: {block_size}")

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

                        # Compute main task loss
                        main_loss = self.strategy.compute_loss(main_logits, targets)

                        # Compute perplexity using scaled logits
                        ppl = self._compute_perplexity(main_logits, targets)

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
