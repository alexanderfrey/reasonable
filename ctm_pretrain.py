"""
CTM Pretraining Script.

Trains a CTM (Continuous Thought Machine) language model using the
same data pipeline as pretrain.py but with CTM-specific loss and logging.
"""

import inspect
import torch
import torch.distributed as dist
from torch import amp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
import argparse
from argparse import Namespace
import os
import json
import math
import numpy as np
import time
import logging
import colorsys
from typing import Optional, Dict, Any
import wandb
from tqdm import tqdm

# Enable TF32 for faster training on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import shared utilities from pretrain.py
from pretrain import (
    init_distributed_mode,
    unwrap_model,
    _build_optimizer,
    _build_lr_scheduler,
    load_and_prepare_tokenizer,
    PretokenizedDataset,
    _load_or_tokenize_data,
    compute_default_n_kv_head,
)

# Import CTM model and loss
from ctm_model import CTMConfig, CTMLanguageModel
from ctm_loss import CTMLoss, CTMPerplexity


def setup_environment(args: Namespace):
    """Setup CUDA device and determine AMP settings."""
    if torch.cuda.is_available():
        local_rank = getattr(args, "local_rank", 0)
        if local_rank is None or local_rank < 0:
            local_rank = 0
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available. Training on CPU (will be very slow).")

    # Determine AMP mode
    use_bf16 = getattr(args, "use_bf16", True) and torch.cuda.is_bf16_supported()
    use_amp = getattr(args, "use_amp", False) and not use_bf16

    if use_bf16:
        logger.info("Using BF16 mixed precision (no GradScaler needed).")
    elif use_amp:
        logger.info("Using FP16 mixed precision with GradScaler.")
    else:
        logger.info("Using FP32 precision.")

    return device, use_amp, use_bf16


def prepare_dataloaders(args: Namespace, tokenizer, vocab_size, pad_token_id, eos_token_id):
    """Prepare training and evaluation dataloaders."""
    rank = getattr(args, "rank", 0)
    distributed = getattr(args, "distributed", False)

    # Load or tokenize training data
    train_token_file, train_num_examples = _load_or_tokenize_data(
        corpus_file=args.train_corpus,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        stride=args.stride,
        output_dir=args.output_dir,
        data_type="Training",
        force_retokenize=getattr(args, "force_retokenize", False),
        current_vocab_size=vocab_size,
        current_tokenizer_name=args.tokenizer_name,
        current_pad_token_id=pad_token_id,
        current_eos_token_id=eos_token_id,
        rank=rank,
        distributed=distributed,
        batch_lines=getattr(args, "tokenize_batch_lines", 64),
    )

    if train_token_file is None or train_num_examples <= 0:
        logger.critical("No training data available. Exiting.")
        exit(1)

    train_dataset = PretokenizedDataset(
        train_token_file, train_num_examples, args.max_seq_len, args.stride, "Training"
    )

    # Evaluation data (optional)
    eval_dataloader = None
    eval_sampler = None
    if args.eval_corpus:
        eval_token_file, eval_num_examples = _load_or_tokenize_data(
            corpus_file=args.eval_corpus,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            stride=args.stride,
            output_dir=args.output_dir,
            data_type="Evaluation",
            force_retokenize=getattr(args, "force_retokenize", False),
            current_vocab_size=vocab_size,
            current_tokenizer_name=args.tokenizer_name,
            current_pad_token_id=pad_token_id,
            current_eos_token_id=eos_token_id,
            rank=rank,
            distributed=distributed,
            batch_lines=getattr(args, "tokenize_batch_lines", 64),
        )
        if eval_token_file and eval_num_examples > 0:
            eval_dataset = PretokenizedDataset(
                eval_token_file, eval_num_examples, args.max_seq_len, args.stride, "Evaluation"
            )
            eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if distributed else None
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=eval_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )

    # Training dataloader
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_dataloader, eval_dataloader, train_sampler, eval_sampler


def initialize_ctm_model(args: Namespace, vocab_size: int, device: torch.device):
    """Initialize CTM model."""
    n_kv_head = getattr(args, "n_kv_head", None)
    if n_kv_head is None:
        n_kv_head = compute_default_n_kv_head(args.n_head)

    config = CTMConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_kv_head=n_kv_head,
        n_layer=args.n_layer,
        max_seq_len=args.max_seq_len,
        d_ff=getattr(args, "d_ff", None),
        num_ticks=args.num_ticks,
        nlm_hidden=args.nlm_hidden,
        nlm_depth=args.nlm_depth,
        sync_pairs=args.sync_pairs,
        sync_order=args.sync_order,
        dropout=args.dropout,
        rope_theta=getattr(args, "rope_theta", 500000.0),
        use_gradient_checkpointing=getattr(args, "use_gradient_checkpointing", False),
    )

    model = CTMLanguageModel(config)
    model = model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"CTM Model initialized:")
    logger.info(f"  - d_model: {config.d_model}")
    logger.info(f"  - n_layer: {config.n_layer}")
    logger.info(f"  - n_head: {config.n_head}")
    logger.info(f"  - n_kv_head: {config.n_kv_head}")
    logger.info(f"  - num_ticks: {config.num_ticks}")
    logger.info(f"  - nlm_hidden: {config.nlm_hidden}")
    logger.info(f"  - sync_pairs: {config.sync_pairs}")
    logger.info(f"  - sync_order: {config.sync_order}")
    logger.info(f"  - Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")

    return model, config


def ctm_train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: CTMLoss,
    device: torch.device,
    use_amp: bool,
    use_bf16: bool,
    gradient_accumulation_steps: int,
    num_ticks: Optional[int] = None,
):
    """
    Perform a single CTM training step.

    Returns:
        loss_value: float - unscaled loss value
        metrics: Dict - CTM-specific metrics (per-tick loss, avg tick, etc.)
    """
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)

    # Determine autocast settings
    if use_bf16:
        amp_dtype = torch.bfloat16
        amp_enabled = True
    elif use_amp:
        amp_dtype = torch.float16
        amp_enabled = True
    else:
        amp_dtype = torch.float32
        amp_enabled = False

    from torch.amp import autocast
    with autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
        # Forward pass with all ticks
        _, all_logits = model(input_ids, return_all_ticks=True, num_ticks=num_ticks)

        # Compute loss with tick selection
        loss, metrics = criterion(all_logits, labels)

    # Scale loss for gradient accumulation
    loss_scaled = loss / gradient_accumulation_steps
    loss_scaled.backward()

    return loss.item(), metrics


@torch.no_grad()
def ctm_evaluate(
    model: nn.Module,
    eval_dataloader: DataLoader,
    criterion: CTMLoss,
    device: torch.device,
    use_amp: bool,
    use_bf16: bool,
    args: Namespace,
    num_ticks: Optional[int] = None,
):
    """Evaluate CTM model."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    per_tick_losses = None
    tick_counts = None

    if use_bf16:
        amp_dtype = torch.bfloat16
        amp_enabled = True
    elif use_amp:
        amp_dtype = torch.float16
        amp_enabled = True
    else:
        amp_dtype = torch.float32
        amp_enabled = False

    eval_loader = eval_dataloader
    subset_size = getattr(args, "eval_random_subset_size", 0)
    if subset_size and subset_size > 0:
        dataset = getattr(eval_dataloader, "dataset", None)
        if dataset is not None and len(dataset) > 0:
            max_subset = min(subset_size, len(dataset))
            indices = torch.randperm(len(dataset))[:max_subset].tolist()
            subset = Subset(dataset, indices)
            loader_kwargs = {
                "batch_size": eval_dataloader.batch_size,
                "shuffle": False,
                "num_workers": eval_dataloader.num_workers,
                "pin_memory": eval_dataloader.pin_memory,
                "drop_last": False,
                "collate_fn": eval_dataloader.collate_fn,
            }
            if eval_dataloader.num_workers > 0:
                loader_kwargs["prefetch_factor"] = getattr(eval_dataloader, "prefetch_factor", 2)
                loader_kwargs["persistent_workers"] = getattr(eval_dataloader, "persistent_workers", False)
            pin_device = getattr(eval_dataloader, "pin_memory_device", None)
            if pin_device:
                loader_kwargs["pin_memory_device"] = pin_device
            eval_loader = DataLoader(subset, **loader_kwargs)
            if getattr(args, "is_main_process", True):
                logger.info(f"Evaluating on random subset: {max_subset} examples.")

    show_progress = (not getattr(args, "distributed", False)) or getattr(args, "is_main_process", True)
    eval_iterator = tqdm(eval_loader, desc="Evaluating", leave=False) if show_progress else eval_loader

    max_eval_batches = getattr(args, "max_eval_batches", None)
    num_batches = 0

    from torch.amp import autocast
    for batch_idx, batch in enumerate(eval_iterator):
        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            _, all_logits = model(input_ids, return_all_ticks=True, num_ticks=num_ticks)
            loss, metrics = criterion(all_logits, labels)

        batch_tokens = metrics["num_valid_tokens"].item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        num_batches += 1

        # Accumulate per-tick losses
        if per_tick_losses is None:
            per_tick_losses = metrics["per_tick_loss"].clone() * batch_tokens
            tick_counts = metrics["tick_distribution"].clone() * batch_tokens
        else:
            per_tick_losses += metrics["per_tick_loss"] * batch_tokens
            tick_counts += metrics["tick_distribution"] * batch_tokens

    model.train()

    if total_tokens == 0:
        return float("nan"), float("nan"), {}

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 700 else float("inf")
    avg_per_tick_loss = per_tick_losses / total_tokens
    avg_tick_dist = tick_counts / total_tokens

    return avg_loss, perplexity, {
        "per_tick_loss": avg_per_tick_loss,
        "tick_distribution": avg_tick_dist,
    }


def run_debug_generation(
    args: Namespace,
    model: nn.Module,
    tokenizer,
    device: torch.device,
    use_amp: bool,
    use_bf16: bool,
    global_step: int,
):
    """Run a lightweight debug generation sample."""
    base_model = unwrap_model(model)
    if not (hasattr(base_model, "generate") and callable(getattr(base_model, "generate"))):
        logger.warning("Model does not have a 'generate' method. Skipping debug generation.")
        return

    prompt = args.debug_generate_prompt
    input_ids_list = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids_list:
        logger.warning("Debug prompt encoded to an empty sequence. Skipping generation.")
        return

    input_tensor = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    gen_kwargs = {
        "max_new_tokens": args.debug_max_new_tokens,
        "temperature": args.debug_temperature,
        "top_k": args.debug_top_k if args.debug_top_k > 0 else None,
        "num_ticks": args.debug_num_ticks if args.debug_num_ticks > 0 else None,
    }

    was_training = model.training
    model.eval()
    try:
        amp_enabled = use_amp or use_bf16
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.no_grad(), amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            generated_ids = base_model.generate(input_tensor, **gen_kwargs)

        generated_ids = generated_ids[0].tolist()
        if generated_ids[:len(input_ids_list)] == input_ids_list:
            completion_ids = generated_ids[len(input_ids_list):]
        else:
            completion_ids = generated_ids

        generated_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        logger.info(f"\n--- Debug Generation @ Step {global_step} ---")
        logger.info(f'Prompt: "{prompt}"')
        logger.info(f"Generated Text:\n{generated_text}")
    except Exception as e:
        logger.error(f"Error during debug generation: {e}", exc_info=True)
    finally:
        if was_training:
            model.train()


def log_nlm_diagnostics(
    args: Namespace,
    model: nn.Module,
    batch: Optional[Dict[str, torch.Tensor]],  # Not used, kept for API compatibility
    device: torch.device,
    global_step: int,
):
    """
    Log NLM tick-level dynamics to wandb.

    Visualizations:
    1. Neuron activation trajectories across ticks
    2. FFT frequency spectrum over tick dimension
    3. Sync evolution across ticks
    4. Gate/entropy distributions (NLM internals)
    """
    if getattr(args, "disable_wandb", False) or not wandb.run:
        return

    base_model = unwrap_model(model)
    if not hasattr(base_model, "get_nlm_diagnostics"):
        return

    # Clear memory BEFORE running diagnostics to avoid doubling
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    try:
        # Generate a fresh random input instead of using training batch
        # This avoids holding references to the training batch
        vocab_size = base_model.config.vocab_size
        seq_len = 32  # Short sequence for diagnostics
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

        diagnostics = base_model.get_nlm_diagnostics(input_ids)

        # Immediately free GPU memory
        del input_ids
        torch.cuda.empty_cache()

        if not diagnostics:
            return

        # Move all tensors to CPU immediately to free GPU memory
        diagnostics = {k: v.cpu() for k, v in diagnostics.items()}

        log_data = {}

        def safe_histogram(data, num_bins=16):
            """Create histogram, handling edge cases where data has no variance."""
            try:
                data = np.asarray(data).flatten()
                if len(data) == 0:
                    return None
                # Just log scalar stats if variance is too low
                if data.std() < 1e-6:
                    return None
                return wandb.Histogram(data, num_bins=num_bins)
            except Exception:
                return None

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # === TICK-LEVEL DYNAMICS (the correct frame for CTM) ===
        # tick_activations: (num_ticks, D) - neuron activations at each tick boundary
        tick_activations = diagnostics['tick_activations'].numpy()  # (num_ticks, D)
        tick_sync = diagnostics['tick_sync'].numpy()  # (num_ticks, sync_pairs)
        num_ticks, D = tick_activations.shape

        # 1. Neuron activation trajectories across ticks
        # This shows how each neuron's output evolves during "thinking"
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top-left: Heatmap of ALL neurons across ticks (sorted by variance = "activity")
        neuron_variance = tick_activations.var(axis=0)  # (D,) - how much each neuron changes
        sorted_by_variance = np.argsort(neuron_variance)[::-1]  # Most active first
        sorted_activations = tick_activations[:, sorted_by_variance].T  # (D, num_ticks)

        im = axes[0, 0].imshow(sorted_activations, aspect='auto', cmap='RdBu_r',
                               origin='upper', interpolation='nearest')
        axes[0, 0].set_xlabel('Tick')
        axes[0, 0].set_ylabel('Neuron (sorted by variance: active→stable)')
        axes[0, 0].set_title(f'Neuron Activations Across Ticks @ Step {global_step}')
        axes[0, 0].set_xticks(range(num_ticks))
        plt.colorbar(im, ax=axes[0, 0], label='Activation')

        # Top-right: Distribution of neuron variance (activity levels)
        hist = axes[0, 1].hist(neuron_variance, bins=32, alpha=0.7, color='steelblue')
        axes[0, 1].set_xlabel('Variance across ticks')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Neuron Activity Distribution')
        axes[0, 1].axvline(x=np.median(neuron_variance), color='red', linestyle='--',
                          label=f'Median: {np.median(neuron_variance):.4f}')
        axes[0, 1].legend()

        # Bottom-left: Sample trajectories for most active neurons
        top_active = sorted_by_variance[:8]  # 8 most active
        for i, idx in enumerate(top_active):
            axes[1, 0].plot(range(num_ticks), tick_activations[:, idx],
                           marker='o', alpha=0.7, label=f'n{idx}')
        axes[1, 0].set_xlabel('Tick')
        axes[1, 0].set_ylabel('Activation')
        axes[1, 0].set_title('Most Active Neurons (high variance)')
        axes[1, 0].set_xticks(range(num_ticks))
        axes[1, 0].legend(fontsize=6, ncol=2)
        axes[1, 0].grid(True, alpha=0.3)

        # Bottom-right: Sample trajectories for most stable neurons
        bottom_stable = sorted_by_variance[-8:]  # 8 most stable
        for i, idx in enumerate(bottom_stable):
            axes[1, 1].plot(range(num_ticks), tick_activations[:, idx],
                           marker='o', alpha=0.7, label=f'n{idx}')
        axes[1, 1].set_xlabel('Tick')
        axes[1, 1].set_ylabel('Activation')
        axes[1, 1].set_title('Most Stable Neurons (low variance)')
        axes[1, 1].set_xticks(range(num_ticks))
        axes[1, 1].legend(fontsize=6, ncol=2)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        log_data["nlm/tick_dynamics"] = wandb.Image(fig)
        plt.close(fig)

        # 2. FFT-based frequency spectrum over TICKS (the correct domain!)
        # This shows which neurons oscillate fast vs slow in tick-space
        if num_ticks >= 2:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            # Compute FFT over tick dimension for each neuron
            # Subtract mean to remove DC component for cleaner spectrum
            centered = tick_activations - tick_activations.mean(axis=0, keepdims=True)
            fft_result = np.fft.rfft(centered, axis=0)  # (num_ticks//2+1, D)
            power_spectrum = np.abs(fft_result) ** 2  # Power spectrum
            freqs = np.fft.rfftfreq(num_ticks)  # Normalized frequencies (cycles per tick)

            # Left: Power spectrum heatmap (neurons sorted by dominant frequency)
            if power_spectrum.shape[0] > 1:
                dominant_freq_idx = np.argmax(power_spectrum[1:, :], axis=0)  # Skip DC
            else:
                dominant_freq_idx = np.zeros(D, dtype=int)
            freq_sorted_indices = np.argsort(dominant_freq_idx)
            sorted_power = power_spectrum[:, freq_sorted_indices]  # (freqs, D_sorted)

            # Log scale for better visualization
            sorted_power_log = np.log1p(sorted_power.T)  # (D, freqs)

            im = axes[0].imshow(sorted_power_log, aspect='auto', cmap='magma',
                               origin='upper', interpolation='nearest')
            axes[0].set_xlabel('Frequency bin (cycles/tick)')
            axes[0].set_ylabel('Neuron (sorted by dominant freq)')
            axes[0].set_title('FFT Power Spectrum Over Ticks')
            plt.colorbar(im, ax=axes[0], label='Log power')

            # Middle: Average power spectrum
            avg_power = power_spectrum.mean(axis=1)
            axes[1].bar(range(len(freqs)), avg_power, alpha=0.7, color='steelblue')
            axes[1].set_xlabel('Frequency bin')
            axes[1].set_ylabel('Average Power')
            axes[1].set_title('Average Power Spectrum')
            axes[1].set_xticks(range(len(freqs)))
            axes[1].set_xticklabels([f'{f:.2f}' for f in freqs], rotation=45)

            # Right: Distribution of dominant frequencies
            axes[2].hist(dominant_freq_idx, bins=max(1, len(freqs)-1), alpha=0.7, color='coral')
            axes[2].set_xlabel('Dominant Frequency Bin')
            axes[2].set_ylabel('Count')
            axes[2].set_title('Distribution of Neuron Frequencies')

            plt.tight_layout()
            log_data["nlm/tick_fft_spectrum"] = wandb.Image(fig)
            plt.close(fig)

        # 3. Sync evolution across ticks
        fig, ax = plt.subplots(figsize=(10, 5))
        sync_pairs = tick_sync.shape[1]
        for p in range(min(sync_pairs, 16)):  # Plot up to 16 sync pairs
            ax.plot(range(num_ticks), tick_sync[:, p], marker='o', alpha=0.6, label=f'pair{p}')
        ax.set_xlabel('Tick')
        ax.set_ylabel('Sync Value')
        ax.set_title(f'Synchronization Evolution Across Ticks @ Step {global_step}')
        ax.set_xticks(range(num_ticks))
        ax.legend(fontsize=6, ncol=4, loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        log_data["nlm/sync_evolution"] = wandb.Image(fig)
        plt.close(fig)

        # Log scalar metrics
        log_data["nlm/neuron_variance_mean"] = float(neuron_variance.mean())
        log_data["nlm/neuron_variance_std"] = float(neuron_variance.std())
        log_data["nlm/sync_final_mean"] = float(tick_sync[-1].mean())
        log_data["nlm/sync_change"] = float(tick_sync[-1].mean() - tick_sync[0].mean())

        # 4. Per-layer post-activations across ticks
        # layer_post_acts: (num_ticks, n_layer, D)
        if 'layer_post_acts' in diagnostics:
            layer_post_acts = diagnostics['layer_post_acts'].numpy()  # (num_ticks, n_layer, D)
            n_layer = layer_post_acts.shape[1]

            # Create a grid: each row is a layer, showing its neurons across ticks
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # Top-left: Heatmap of layer mean activations across ticks
            # Average over neurons to get (num_ticks, n_layer)
            layer_means = layer_post_acts.mean(axis=2)  # (num_ticks, n_layer)
            im = axes[0, 0].imshow(layer_means.T, aspect='auto', cmap='RdBu_r',
                                   origin='lower', interpolation='nearest')
            axes[0, 0].set_xlabel('Tick')
            axes[0, 0].set_ylabel('Layer')
            axes[0, 0].set_title('Mean Post-Activation per Layer Across Ticks')
            axes[0, 0].set_xticks(range(num_ticks))
            axes[0, 0].set_yticks(range(n_layer))
            plt.colorbar(im, ax=axes[0, 0], label='Mean activation')

            # Top-right: Layer activation trajectories (line plot)
            for layer_idx in range(n_layer):
                axes[0, 1].plot(range(num_ticks), layer_means[:, layer_idx],
                               marker='o', alpha=0.7, label=f'L{layer_idx}')
            axes[0, 1].set_xlabel('Tick')
            axes[0, 1].set_ylabel('Mean Activation')
            axes[0, 1].set_title('Layer Activation Trajectories')
            axes[0, 1].set_xticks(range(num_ticks))
            axes[0, 1].legend(fontsize=6, ncol=max(1, n_layer // 4))
            axes[0, 1].grid(True, alpha=0.3)

            # Bottom-left: Per-layer neuron variance across ticks
            # Shows which layers have more dynamic neurons
            layer_neuron_var = layer_post_acts.var(axis=0)  # (n_layer, D) - variance over ticks
            layer_var_mean = layer_neuron_var.mean(axis=1)  # (n_layer,) - mean variance per layer
            axes[1, 0].bar(range(n_layer), layer_var_mean, alpha=0.7, color='steelblue')
            axes[1, 0].set_xlabel('Layer')
            axes[1, 0].set_ylabel('Mean Neuron Variance')
            axes[1, 0].set_title('Neuron Dynamics per Layer (variance across ticks)')
            axes[1, 0].set_xticks(range(n_layer))

            # Bottom-right: Detailed heatmap for one layer (layer 0)
            # Show sample neurons across ticks
            layer0_acts = layer_post_acts[:, 0, :]  # (num_ticks, D)
            # Sort neurons by variance
            neuron_var = layer0_acts.var(axis=0)
            sorted_idx = np.argsort(neuron_var)[::-1]
            # Show top 64 most dynamic neurons
            top_neurons = sorted_idx[:min(64, D)]
            sorted_acts = layer0_acts[:, top_neurons].T  # (64, num_ticks)

            im = axes[1, 1].imshow(sorted_acts, aspect='auto', cmap='RdBu_r',
                                   origin='upper', interpolation='nearest')
            axes[1, 1].set_xlabel('Tick')
            axes[1, 1].set_ylabel('Neuron (sorted by variance)')
            axes[1, 1].set_title('Layer 0: Top 64 Dynamic Neurons Across Ticks')
            axes[1, 1].set_xticks(range(num_ticks))
            plt.colorbar(im, ax=axes[1, 1], label='Activation')

            plt.tight_layout()
            log_data["nlm/layer_post_acts"] = wandb.Image(fig)
            plt.close(fig)

            # 5. All-layers neuron grid: show ALL neurons for ALL layers
            # This is the comprehensive view the user asked for
            fig, axes = plt.subplots(1, n_layer, figsize=(4 * n_layer, 8), sharey=True)
            if n_layer == 1:
                axes = [axes]

            for layer_idx in range(n_layer):
                layer_acts = layer_post_acts[:, layer_idx, :]  # (num_ticks, D)
                # Sort by variance
                var = layer_acts.var(axis=0)
                sorted_idx = np.argsort(var)[::-1]
                sorted_acts = layer_acts[:, sorted_idx].T  # (D, num_ticks)

                im = axes[layer_idx].imshow(sorted_acts, aspect='auto', cmap='RdBu_r',
                                            origin='upper', interpolation='nearest')
                axes[layer_idx].set_xlabel('Tick')
                if layer_idx == 0:
                    axes[layer_idx].set_ylabel('Neuron (sorted by variance)')
                axes[layer_idx].set_title(f'Layer {layer_idx}')
                axes[layer_idx].set_xticks(range(num_ticks))

            plt.suptitle(f'All NLM Post-Activations Across Ticks @ Step {global_step}', fontsize=14)
            plt.tight_layout()
            log_data["nlm/all_layers_grid"] = wandb.Image(fig)
            plt.close(fig)

            # 6. Neuron trajectories: X=tick, Y=activation
            # Grid of subplots, one per layer, showing how neurons evolve over ticks
            n_cols = 3
            n_rows = (n_layer + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                                     sharex=True)
            axes = np.atleast_2d(axes)

            # Select neurons to plot: sample evenly across variance spectrum
            overall_var = layer_post_acts.var(axis=0).mean(axis=0)  # (D,)
            sorted_neuron_idx = np.argsort(overall_var)[::-1]

            # Sample neurons: top 10 most dynamic, 10 from middle, 10 most stable
            n_sample = min(10, D // 3)
            top_neurons = sorted_neuron_idx[:n_sample]
            mid_neurons = sorted_neuron_idx[D//2 - n_sample//2 : D//2 + n_sample//2]
            bottom_neurons = sorted_neuron_idx[-n_sample:]
            sample_neurons = np.concatenate([top_neurons, mid_neurons, bottom_neurons])

            # Color map: dynamic=red, mid=green, stable=blue
            colors_top = plt.cm.Reds(np.linspace(0.4, 0.8, n_sample))
            colors_mid = plt.cm.Greens(np.linspace(0.4, 0.8, len(mid_neurons)))
            colors_bottom = plt.cm.Blues(np.linspace(0.4, 0.8, n_sample))

            for layer_idx in range(n_layer):
                row, col = layer_idx // n_cols, layer_idx % n_cols
                ax = axes[row, col]

                # Plot trajectories for sampled neurons
                ticks = np.arange(num_ticks)

                # Dynamic neurons (red)
                for i, neuron_idx in enumerate(top_neurons):
                    trajectory = layer_post_acts[:, layer_idx, neuron_idx]
                    ax.plot(ticks, trajectory, color=colors_top[i], alpha=0.7,
                           linewidth=1.5, marker='o', markersize=4)

                # Mid neurons (green)
                for i, neuron_idx in enumerate(mid_neurons):
                    trajectory = layer_post_acts[:, layer_idx, neuron_idx]
                    ax.plot(ticks, trajectory, color=colors_mid[i], alpha=0.5,
                           linewidth=1, marker='s', markersize=3)

                # Stable neurons (blue)
                for i, neuron_idx in enumerate(bottom_neurons):
                    trajectory = layer_post_acts[:, layer_idx, neuron_idx]
                    ax.plot(ticks, trajectory, color=colors_bottom[i], alpha=0.7,
                           linewidth=1.5, marker='^', markersize=4)

                ax.set_title(f'Layer {layer_idx}', fontsize=11, fontweight='bold')
                ax.set_xlabel('Tick')
                ax.set_ylabel('Activation')
                ax.set_xticks(ticks)
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.grid(True, alpha=0.3)

            # Hide empty subplots
            for idx in range(n_layer, n_rows * n_cols):
                row, col = idx // n_cols, idx % n_cols
                axes[row, col].set_visible(False)

            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', linewidth=2, label='Dynamic neurons'),
                Line2D([0], [0], color='green', linewidth=2, label='Mid neurons'),
                Line2D([0], [0], color='blue', linewidth=2, label='Stable neurons'),
            ]
            fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

            plt.suptitle(f'Neuron Trajectories Over Ticks @ Step {global_step}\n(X=tick, Y=activation)', fontsize=12)
            plt.tight_layout()
            log_data["nlm/neuron_trajectories"] = wandb.Image(fig)
            plt.close(fig)

            # 7. Tick-to-tick DELTAS: shows the change between ticks
            # This reveals dynamics even when absolute values are similar
            if num_ticks > 1:
                # Compute deltas: (num_ticks-1, n_layer, D)
                deltas = np.diff(layer_post_acts, axis=0)

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                                         sharex=True)
                axes = np.atleast_2d(axes)

                for layer_idx in range(n_layer):
                    row, col = layer_idx // n_cols, layer_idx % n_cols
                    ax = axes[row, col]

                    # Sort neurons by delta variance (most changing first)
                    layer_deltas = deltas[:, layer_idx, :]  # (num_ticks-1, D)
                    delta_var = layer_deltas.var(axis=0)
                    sorted_idx = np.argsort(delta_var)[::-1]

                    # Plot top 30 most changing neurons
                    n_show = min(30, D)
                    delta_ticks = np.arange(num_ticks - 1) + 0.5  # Centered between ticks

                    for i in range(n_show):
                        neuron_idx = sorted_idx[i]
                        trajectory = layer_deltas[:, neuron_idx]
                        alpha = 0.8 - (i / n_show) * 0.5  # Fade out less dynamic ones
                        ax.plot(delta_ticks, trajectory, alpha=alpha, linewidth=1.5,
                               marker='o', markersize=4)

                    ax.set_title(f'Layer {layer_idx}', fontsize=11, fontweight='bold')
                    ax.set_xlabel('Tick transition')
                    ax.set_ylabel('Δ Activation')
                    ax.set_xticks(delta_ticks)
                    ax.set_xticklabels([f'{i}→{i+1}' for i in range(num_ticks - 1)])
                    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
                    ax.grid(True, alpha=0.3)

                # Hide empty subplots
                for idx in range(n_layer, n_rows * n_cols):
                    row, col = idx // n_cols, idx % n_cols
                    axes[row, col].set_visible(False)

                plt.suptitle(f'Tick-to-Tick Changes (Δ) @ Step {global_step}\n(How much neurons change between ticks)', fontsize=12)
                plt.tight_layout()
                log_data["nlm/tick_deltas"] = wandb.Image(fig)
                plt.close(fig)

                # Log delta statistics
                mean_abs_delta = np.abs(deltas).mean()
                max_abs_delta = np.abs(deltas).max()
                log_data["nlm/mean_abs_delta"] = float(mean_abs_delta)
                log_data["nlm/max_abs_delta"] = float(max_abs_delta)

        # === NLM INTERNAL DIAGNOSTICS (from last tick) ===
        if 'gate_mean' in diagnostics:
            gate_mean = diagnostics['gate_mean'].mean(dim=0).numpy()  # (D,)
            hist = safe_histogram(gate_mean)
            if hist:
                log_data["nlm/gate_value_hist"] = hist
            log_data["nlm/gate_mean"] = float(gate_mean.mean())
            log_data["nlm/gate_std"] = float(gate_mean.std())

        if 'attn_entropy' in diagnostics:
            attn_entropy = diagnostics['attn_entropy'].mean(dim=0).numpy()  # (D,)
            hist = safe_histogram(attn_entropy)
            if hist:
                log_data["nlm/attn_entropy_hist"] = hist
            log_data["nlm/attn_entropy_mean"] = float(attn_entropy.mean())
            log_data["nlm/attn_entropy_std"] = float(attn_entropy.std())

        wandb.log(log_data, step=global_step)

        # Clean up to free memory
        del diagnostics, log_data
        import gc
        gc.collect()

    except Exception as e:
        logger.warning(f"Error logging NLM diagnostics: {e}")
    finally:
        # Always try to free GPU memory after diagnostics
        torch.cuda.empty_cache()


def _build_tick_palette(num_ticks: int) -> np.ndarray:
    base_palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    if num_ticks <= len(base_palette):
        palette = base_palette[:num_ticks]
    else:
        palette = []
        for i in range(num_ticks):
            r, g, b = colorsys.hsv_to_rgb(i / num_ticks, 0.7, 0.9)
            palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(palette, dtype=np.uint8)


def _selected_ticks_to_rgb(selected_ticks: torch.Tensor, num_ticks: int, max_tokens: int) -> np.ndarray:
    ticks = selected_ticks.detach().to("cpu").numpy()
    if max_tokens > 0 and ticks.shape[1] > max_tokens:
        ticks = ticks[:, :max_tokens]
    palette = _build_tick_palette(num_ticks)
    ticks = np.clip(ticks, 0, num_ticks - 1).astype(np.int64)
    return palette[ticks]


def save_checkpoint(
    args: Namespace,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Optional[GradScaler],
    scheduler,
    epoch: int,
    global_step: int,
    best_eval_loss: float,
    config: CTMConfig,
):
    """Save training checkpoint."""
    if not args.is_main_process:
        return

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_eval_loss": best_eval_loss,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "args": vars(args),
    }

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save latest link
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(checkpoint, latest_path)


def load_checkpoint(args: Namespace, model: nn.Module, optimizer: optim.Optimizer, scaler, device):
    """Load checkpoint if available."""
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    latest_path = os.path.join(checkpoint_dir, "latest.pt")

    if not os.path.exists(latest_path):
        logger.info("No checkpoint found. Starting from scratch.")
        return 0, 0, float("inf"), None

    logger.info(f"Loading checkpoint from {latest_path}")
    checkpoint = torch.load(latest_path, map_location=device)

    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    scheduler_state = checkpoint.get("scheduler_state_dict")

    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))

    logger.info(f"Resumed from epoch {epoch}, step {global_step}")

    return epoch, global_step, best_eval_loss, scheduler_state


def train(args: Namespace):
    """Main CTM training function."""
    init_distributed_mode(args)

    if args.distributed and not args.is_main_process:
        args.disable_wandb = True

    # --- Weights & Biases Setup ---
    if args.is_main_process and not getattr(args, "disable_wandb", False):
        try:
            if wandb.run is None:
                wandb.init(
                    project=args.wandb_project,
                    entity=getattr(args, "wandb_entity", None),
                    name=args.wandb_run_name,
                    config=vars(args),
                    resume="allow",
                )
            if wandb.run:
                logger.info(f"W&B initialized: {wandb.run.get_url()}")
        except Exception as e:
            logger.error(f"Could not initialize W&B: {e}")
            args.disable_wandb = True

    # --- Setup ---
    device, use_amp, use_bf16 = setup_environment(args)

    # --- Tokenizer ---
    tokenizer, vocab_size, pad_token_id, eos_token_id, _, _ = load_and_prepare_tokenizer(args.tokenizer_name)

    # --- Data ---
    train_dataloader, eval_dataloader, train_sampler, _ = prepare_dataloaders(
        args, tokenizer, vocab_size, pad_token_id, eos_token_id
    )

    # --- Model ---
    model, config = initialize_ctm_model(args, vocab_size, device)

    # --- Optimizer ---
    optimizer = _build_optimizer(model, args.learning_rate, args.weight_decay, logger_prefix="CTM ")

    # --- Loss ---
    # Note: ignore_index=-100 (default) means all tokens including EOS are trained on.
    # pad_token_id is kept for potential future use but doesn't affect loss computation.
    criterion = CTMLoss(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        ignore_index=-100,  # Train on all tokens including EOS
        selection=args.tick_selection,
        tau=getattr(args, "tick_selection_tau", 1.0),
        progressive_steepness=getattr(args, "progressive_steepness", 1.0),
    )

    # --- Scaler (FP16 only) ---
    scaler = GradScaler() if use_amp else None

    # --- Load Checkpoint ---
    start_epoch, global_step, best_eval_loss, scheduler_state = load_checkpoint(
        args, model, optimizer, scaler, device
    )

    # --- Compile Model ---
    if getattr(args, "compile_model", False):
        try:
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model, mode="max-autotune", fullgraph=False)
            logger.info("Model compiled successfully.")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    # --- DDP ---
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    # --- Gradient Accumulation ---
    gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {args.batch_size * gradient_accumulation_steps}")

    steps_per_epoch = max(1, math.ceil(len(train_dataloader) / gradient_accumulation_steps))
    total_training_steps = args.epochs * steps_per_epoch

    # --- LR Scheduler ---
    scheduler = None
    scheduler_name = getattr(args, "lr_scheduler", "cosine").lower()
    if scheduler_name != "none":
        scheduler = _build_lr_scheduler(args, optimizer, total_training_steps, start_step=global_step)
        if scheduler_state is not None:
            try:
                scheduler.load_state_dict(scheduler_state)
            except Exception as e:
                logger.warning(f"Could not load scheduler state: {e}")

    # --- Training Loop ---
    logger.info(f"\nStarting CTM training from epoch {start_epoch + 1}...")
    model.train()
    total_loss_accum = 0.0
    micro_steps_count = 0
    can_toggle_sync = hasattr(model, "require_backward_grad_sync")

    # Tick budget warmup (optional)
    tick_warmup_steps = getattr(args, "tick_warmup_steps", 0)
    min_warmup_ticks = getattr(args, "min_warmup_ticks", 2)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if args.is_main_process:
            logger.info(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        progress_iter = enumerate(train_dataloader)
        if args.is_main_process:
            progress_bar = tqdm(
                progress_iter, total=len(train_dataloader),
                desc=f"Epoch {epoch + 1}", leave=True, dynamic_ncols=True
            )
        else:
            progress_bar = progress_iter

        for batch_idx, batch in progress_bar:
            is_final_accum = (batch_idx + 1) % gradient_accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(train_dataloader)
            do_sync = is_final_accum or is_last_batch

            if can_toggle_sync:
                model.require_backward_grad_sync = do_sync

            # Compute tick budget (warmup)
            if tick_warmup_steps > 0 and global_step < tick_warmup_steps:
                progress = global_step / tick_warmup_steps
                num_ticks = max(min_warmup_ticks, int(progress * args.num_ticks))
            else:
                num_ticks = None  # Use default

            # Training step
            step_loss, metrics = ctm_train_step(
                model, batch, criterion, device, use_amp, use_bf16,
                gradient_accumulation_steps, num_ticks=num_ticks
            )

            total_loss_accum += step_loss
            micro_steps_count += 1

            # Optimizer step
            if do_sync:
                if scaler:
                    scaler.unscale_(optimizer)
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                if scheduler:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # --- Logging ---
                if args.log_interval > 0 and global_step % args.log_interval == 0:
                    avg_loss = total_loss_accum / micro_steps_count
                    perplexity = math.exp(avg_loss) if avg_loss < 700 else float("inf")
                    avg_tick = metrics["avg_selected_tick"].item()

                    log_postfix = {
                        "Loss": f"{avg_loss:.4f}",
                        "PPL": f"{perplexity:.2f}",
                        "AvgTick": f"{avg_tick:.2f}",
                        "Step": global_step,
                        "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }

                    if args.is_main_process:
                        if hasattr(progress_bar, "set_postfix"):
                            progress_bar.set_postfix(log_postfix)

                        if not getattr(args, "disable_wandb", False) and wandb.run:
                            log_data = {
                                "train/loss": avg_loss,
                                "train/perplexity": perplexity,
                                "train/learning_rate": optimizer.param_groups[0]["lr"],
                                "train/avg_selected_tick": avg_tick,
                                "epoch": epoch + 1,
                            }
                            # Log per-tick losses
                            per_tick = metrics["per_tick_loss"]
                            for t, tl in enumerate(per_tick):
                                log_data[f"train/tick_{t}_loss"] = tl.item()

                            if (
                                args.tick_heatmap_interval > 0
                                and global_step % args.tick_heatmap_interval == 0
                            ):
                                selected_ticks = metrics.get("selected_ticks")
                                if selected_ticks is not None:
                                    heatmap = _selected_ticks_to_rgb(
                                        selected_ticks,
                                        num_ticks=per_tick.numel(),
                                        max_tokens=args.tick_heatmap_max_tokens,
                                    )
                                    log_data["train/selected_tick_heatmap"] = wandb.Image(
                                        heatmap,
                                        caption=f"Selected tick per token @ step {global_step}",
                                    )

                            # Log tick distribution as bar chart
                            tick_dist = metrics.get("tick_distribution")
                            if tick_dist is not None:
                                dist_data = [[f"tick_{i}", v.item()] for i, v in enumerate(tick_dist)]
                                dist_table = wandb.Table(data=dist_data, columns=["tick", "frequency"])
                                log_data["train/tick_distribution"] = wandb.plot.bar(
                                    dist_table, "tick", "frequency", title="Train Tick Distribution"
                                )

                            wandb.log(log_data, step=global_step)

                    total_loss_accum = 0.0
                    micro_steps_count = 0

                # --- Debug Generation ---
                if (
                    args.is_main_process
                    and args.debug_generate_interval > 0
                    and global_step % args.debug_generate_interval == 0
                    and global_step > 0
                ):
                    run_debug_generation(
                        args, model, tokenizer, device, use_amp, use_bf16, global_step
                    )

                # --- NLM Diagnostics ---
                nlm_diag_interval = getattr(args, "nlm_diagnostics_interval", 500)
                if (
                    args.is_main_process
                    and nlm_diag_interval > 0
                    and global_step % nlm_diag_interval == 0
                    and global_step > 0
                ):
                    # Release batch memory before running diagnostics
                    del batch
                    log_nlm_diagnostics(args, model, None, device, global_step)

                # --- Evaluation ---
                if args.eval_interval > 0 and global_step % args.eval_interval == 0 and eval_dataloader:
                    eval_loss, eval_ppl, eval_metrics = ctm_evaluate(
                        model, eval_dataloader, criterion, device, use_amp, use_bf16, args
                    )

                    if args.is_main_process:
                        logger.info(f"Eval @ step {global_step}: Loss={eval_loss:.4f}, PPL={eval_ppl:.2f}")

                        if not getattr(args, "disable_wandb", False) and wandb.run:
                            eval_log = {
                                "eval/loss": eval_loss,
                                "eval/perplexity": eval_ppl,
                            }
                            tick_dist = eval_metrics.get("tick_distribution")
                            if tick_dist is not None:
                                tick_idx = torch.arange(
                                    tick_dist.numel(), device=tick_dist.device, dtype=tick_dist.dtype
                                )
                                eval_log["eval/avg_selected_tick"] = (tick_idx * tick_dist).sum().item()

                                # Log tick distribution as bar chart
                                dist_data = [[f"tick_{i}", v.item()] for i, v in enumerate(tick_dist)]
                                dist_table = wandb.Table(data=dist_data, columns=["tick", "frequency"])
                                eval_log["eval/tick_distribution"] = wandb.plot.bar(
                                    dist_table, "tick", "frequency", title="Eval Tick Distribution"
                                )

                            if "per_tick_loss" in eval_metrics:
                                for t, tl in enumerate(eval_metrics["per_tick_loss"]):
                                    tick_loss = tl.item()
                                    eval_log[f"eval/tick_{t}_loss"] = tick_loss
                                    eval_log[f"eval/tick_{t}_ppl"] = (
                                        math.exp(tick_loss) if tick_loss < 700 else float("inf")
                                    )
                            wandb.log(eval_log, step=global_step)

                        # Save best model
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            save_checkpoint(
                                args, model, optimizer, scaler, scheduler,
                                epoch, global_step, best_eval_loss, config
                            )

                    model.train()

                # --- Periodic Checkpoint ---
                if args.save_interval > 0 and global_step % args.save_interval == 0:
                    save_checkpoint(
                        args, model, optimizer, scaler, scheduler,
                        epoch, global_step, best_eval_loss, config
                    )

        # End of epoch checkpoint
        if args.is_main_process:
            save_checkpoint(
                args, model, optimizer, scaler, scheduler,
                epoch + 1, global_step, best_eval_loss, config
            )

    logger.info("Training complete!")

    if wandb.run:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="CTM Language Model Pretraining")

    # Data
    parser.add_argument("--train_corpus", type=str, required=True, help="Path to training corpus")
    parser.add_argument("--eval_corpus", type=str, default=None, help="Path to evaluation corpus")
    parser.add_argument("--output_dir", type=str, default="ctm_output", help="Output directory")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--stride", type=int, default=256, help="Stride for sliding window")
    parser.add_argument(
        "--tokenize_batch_lines",
        type=int,
        default=64,
        help="Number of lines to tokenize at once. Reduce if tokenization OOMs or crashes.",
    )

    # Model architecture
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_kv_head", type=int, default=None, help="Number of KV heads (GQA)")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of CTM layers")
    parser.add_argument("--d_ff", type=int, default=None, help="FFN dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    # CTM-specific
    parser.add_argument("--num_ticks", type=int, default=8, help="Number of internal ticks")
    parser.add_argument("--nlm_hidden", type=int, default=64, help="NLM hidden dimension")
    parser.add_argument("--nlm_depth", type=int, default=2, help="NLM depth")
    parser.add_argument("--sync_pairs", type=int, default=512, help="Number of sync pairs")
    parser.add_argument("--sync_order", type=int, default=2,
                        help="Correlation order (2=covariance, faithful to CTM paper)")
    parser.add_argument("--tick_selection", type=str, default="progressive",
                        choices=["progressive", "all", "min_loss", "max_certainty", "weighted", "last"],
                        help="Tick selection strategy (default: 'progressive' encourages multi-tick reasoning)")
    parser.add_argument("--tick_selection_tau", type=float, default=1.0,
                        help="Temperature for weighted tick selection")
    parser.add_argument("--progressive_steepness", type=float, default=1.0,
                        help="Steepness for progressive tick weighting (1=linear, 2=quadratic)")
    parser.add_argument("--tick_warmup_steps", type=int, default=0,
                        help="Steps to warmup tick budget (0 = disabled)")
    parser.add_argument("--min_warmup_ticks", type=int, default=2,
                        help="Minimum ticks during warmup")

    # Training
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["none", "cosine", "linear"], help="LR scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000, help="LR warmup steps")
    parser.add_argument("--lr_min_ratio", type=float, default=0.1, help="Minimum LR ratio")

    # Precision
    parser.add_argument("--use_bf16", action="store_true", default=True, help="Use BF16")
    parser.add_argument("--use_amp", action="store_true", help="Use FP16 AMP")
    parser.add_argument("--use_gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--compile_model", action="store_true", help="torch.compile the model")

    # Logging and checkpointing
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save every N steps")
    parser.add_argument("--max_eval_batches", type=int, default=None, help="Max eval batches")
    parser.add_argument(
        "--eval_random_subset_size",
        type=int,
        default=1000,
        help="Evaluate on a random subset of N examples per eval (0 to disable).",
    )
    parser.add_argument("--tick_heatmap_interval", type=int, default=100,
                        help="Log selected tick heatmap every N steps (0 to disable).")
    parser.add_argument("--tick_heatmap_max_tokens", type=int, default=256,
                        help="Max token positions to include in tick heatmap (0 = no limit).")
    parser.add_argument("--nlm_diagnostics_interval", type=int, default=500,
                        help="Log NLM oscillation patterns every N steps (0 to disable)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--debug_generate_interval", type=int, default=100,
                        help="Run debug generation every N steps (0 to disable)")
    parser.add_argument("--debug_generate_prompt", type=str, default="The meaning of life is")
    parser.add_argument("--debug_max_new_tokens", type=int, default=128)
    parser.add_argument("--debug_temperature", type=float, default=0.7)
    parser.add_argument("--debug_top_k", type=int, default=50)
    parser.add_argument("--debug_num_ticks", type=int, default=0,
                        help="Override num_ticks for debug generation (0 uses default)")

    # Distributed
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed")

    # W&B
    parser.add_argument("--wandb_project", type=str, default="ctm-pretrain", help="W&B project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set default run name
    if args.wandb_run_name is None:
        args.wandb_run_name = f"ctm-d{args.d_model}-l{args.n_layer}-t{args.num_ticks}"

    train(args)


if __name__ == "__main__":
    main()
