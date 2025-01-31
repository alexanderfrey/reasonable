# dual_head_gpt.py
from collections import deque
import torch
import math
import time
import warnings
import threading
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Union, Dict, NamedTuple
from dataclasses import dataclass
from config import DualHeadGPTConfig
from caching import OptimizedKVCache


@dataclass
class ModelOutput:
    primary_logits: torch.Tensor
    primary_loss: Optional[torch.Tensor]


@dataclass
class ModelOutputWithAux(ModelOutput):
    second_logits: torch.Tensor
    second_loss: Optional[torch.Tensor]
    aux_losses: Dict[str, torch.Tensor]


class DualHeadGPT(nn.Module):
    """GPT model with dual heads and improved stability."""

    def __init__(self, config: DualHeadGPTConfig):
        super().__init__()
        self.config = config

        # Core model components
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [self._create_block() for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Initialize second head
        self.second_head = nn.Linear(
            config.n_embd, config.second_vocab_size, bias=config.bias
        )
        self._init_weights(self.second_head)

        # Performance monitoring with bounded storage and thread safety
        self.max_stats_size = 1000
        self._stats_lock = threading.Lock()
        self.perf_stats = {
            "forward_time": deque(maxlen=self.max_stats_size),
            "backward_time": deque(maxlen=self.max_stats_size),
            "memory_used": deque(maxlen=self.max_stats_size),
        }

        # Initialize KV cache
        self.kv_cache = OptimizedKVCache(
            max_seq_len=config.block_size,
            n_head=config.n_head,
            head_dim=config.n_embd // config.n_head,
            device=next(self.parameters()).device,
        )
        self._is_static_graph = False
        self.gradient_checkpointing = False
        self.checkpoint_ratio = 0.5

        # Loss scaling parameters
        self.loss_scale_factor = 1.0
        self.min_loss_scale = 1e-8
        self.max_loss_scale = 1.0
        self.loss_scale_decay = 0.5
        self.max_loss_value = 100.0

        # Gradient clipping
        self.grad_clip_norm = config.grad_clip_norm  # Add to config

        # Debug settings
        self.debug_mode = False

    def _compute_logits(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits for both heads."""
        primary_logits = F.linear(hidden_states, self.wte.weight)
        second_logits = self.second_head(hidden_states)
        return primary_logits, second_logits

    def _forward_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        position: int,
        return_router_logits: bool,
    ) -> torch.Tensor:
        """Forward pass through a single block."""
        return block(
            x, mask=mask, position=position, return_router_logits=return_router_logits
        )

    def _forward_block_with_checkpoint(
        self,
        block: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        position: int,
        return_router_logits: bool,
    ) -> torch.Tensor:
        """Forward pass with gradient checkpointing."""

        def create_custom_forward(module):
            def custom_forward(*args):
                return module(
                    *args, position=position, return_router_logits=return_router_logits
                )

            return custom_forward

        return torch.utils.checkpoint.checkpoint(
            create_custom_forward(block), x, mask, use_reentrant=False
        )

    def _manage_kv_cache(
        self, key: torch.Tensor, value: torch.Tensor, position: int
    ) -> None:
        """Update KV cache with new key-value pairs."""
        try:
            self.kv_cache.update(key, value, position)
        except RuntimeError as e:
            if self.debug_mode:
                self._debug_log(
                    "kv_cache_error",
                    {
                        "error": str(e),
                        "key_shape": key.shape,
                        "value_shape": value.shape,
                        "position": position,
                    },
                )
            # Clear and retry if there's an error
            self.kv_cache.clear()
            self.kv_cache.update(key, value, position)

    def _update_perf_stats(self, start_time: float) -> None:
        """Update performance statistics with thread safety."""
        with self._stats_lock:
            self.perf_stats["forward_time"].append(time.time() - start_time)
            if torch.cuda.is_available():
                self.perf_stats["memory_used"].append(torch.cuda.max_memory_allocated())

    def _clip_gradients(self) -> None:
        """Apply gradient clipping."""
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm)

    def forward(
        self,
        idx: torch.Tensor,
        target_second: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        position: int = 0,
        return_router_logits: bool = False,
    ) -> Union[ModelOutput, ModelOutputWithAux]:
        """Enhanced forward pass with improved validation and error handling."""
        if position < 0:
            raise ValueError("position must be non-negative")

        start_time = time.time()
        batch_size, seq_len = idx.size()

        # Validate and adjust mask
        if mask is not None:
            mask = self._validate_and_adjust_mask(mask, batch_size, seq_len)

        # Validate sequence length
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum block size {self.config.block_size}"
            )

        # Process embeddings
        x = self.wte(idx)
        x = self.drop(x)

        self._debug_log("embeddings", x)

        # Forward through transformer blocks
        aux_losses = {}
        for i, block in enumerate(self.blocks):
            x = self._process_block(block, x, mask, position, return_router_logits, i)

        # Final processing
        x = self.ln_f(x)
        x = self.drop(x)

        # Compute logits and losses
        primary_logits, second_logits = self._compute_logits(x)
        primary_loss, second_loss = self._compute_losses(
            primary_logits, second_logits, idx, target_second
        )

        # Update performance stats
        if self.training:
            self._update_perf_stats(start_time)

        if return_router_logits:
            return primary_logits, second_logits, primary_loss, second_loss, aux_losses
        return primary_logits, second_logits, primary_loss, second_loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        target_head: str = "primary",
    ) -> torch.Tensor:
        """Generate tokens with improved parameter validation and efficiency."""
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        if target_head not in {"primary", "second"}:
            raise ValueError('target_head must be "primary" or "second"')

        self.eval()

        # Clear KV cache only if it exists and isn't empty
        if self.kv_cache and len(self.kv_cache) > 0:
            self.kv_cache.clear()

        vocab_size = (
            self.config.vocab_size
            if target_head == "primary"
            else self.config.second_vocab_size
        )

        if top_k is not None:
            top_k = min(top_k, vocab_size)

        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            # Forward pass
            logits = self._get_next_token_logits(idx_cond, target_head)

            # Apply temperature and sampling
            logits = self._apply_sampling_params(logits, temperature, top_k)
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            # Append and check for EOS
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token_id is not None and (idx_next == eos_token_id).any():
                break

        return idx

    def _validate_and_adjust_mask(
        self, mask: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        """Validate and adjust attention mask dimensions."""
        expected_shape = (batch_size, 1, seq_len, seq_len)

        if mask.shape != expected_shape:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif len(mask.shape) == 3:
                mask = mask.unsqueeze(1)

            if mask.shape != expected_shape:
                raise ValueError(
                    f"Mask shape {mask.shape} cannot be adjusted to expected shape {expected_shape}"
                )

        return mask

    def _process_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        position: int,
        return_router_logits: bool,
        block_idx: int,
    ) -> torch.Tensor:
        """Process a single transformer block with optional checkpointing.

        Args:
            block: Transformer block module
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            mask: Optional attention mask
            position: Current position in sequence
            return_router_logits: Whether to return routing probabilities
            block_idx: Index of current block

        Returns:
            Processed tensor of same shape as input

        Raises:
            ValueError: If block_idx is out of range
            RuntimeError: If checkpoint fails
        """
        if not 0 <= block_idx < len(self.blocks):
            raise ValueError(
                f"Block index {block_idx} out of range [0, {len(self.blocks)})"
            )

        # Cache checkpoint interval calculation
        if not hasattr(self, "_checkpoint_interval"):
            self._checkpoint_interval = max(
                1, int(len(self.blocks) * self.checkpoint_ratio)
            )

        should_checkpoint = (
            self.gradient_checkpointing
            and self.training
            and block_idx % self._checkpoint_interval == 0
        )

        try:
            if should_checkpoint:
                if self.debug_mode:
                    self._debug_log(
                        f"checkpointing_block_{block_idx}",
                        {
                            "input_shape": x.shape,
                            "memory_before": (
                                torch.cuda.memory_allocated()
                                if torch.cuda.is_available()
                                else 0
                            ),
                        },
                    )
                x = self._forward_block_with_checkpoint(
                    block, x, mask, position, return_router_logits
                )
            else:
                x = self._forward_block(block, x, mask, position, return_router_logits)

            if self.debug_mode:
                self._debug_log(
                    f"block_{block_idx}",
                    {
                        "output_shape": x.shape,
                        "output_stats": {
                            "mean": x.mean().item(),
                            "std": x.std().item(),
                            "max": x.max().item(),
                            "min": x.min().item(),
                        },
                    },
                )

            # Check for NaN/inf values
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise ValueError(
                    f"NaN or inf values detected in block {block_idx} output"
                )

            return x

        except RuntimeError as e:
            if "out of memory" in str(e):
                if self.debug_mode:
                    self._debug_log(
                        f"oom_error_block_{block_idx}",
                        {
                            "input_shape": x.shape,
                            "checkpoint_enabled": should_checkpoint,
                        },
                    )
            raise

    def _compute_losses(
        self,
        primary_logits: torch.Tensor,
        second_logits: torch.Tensor,
        idx: torch.Tensor,
        target_second: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute losses for both primary and auxiliary heads.

        Args:
            primary_logits: Logits from primary head (batch_size, seq_len, primary_vocab_size)
            second_logits: Logits from secondary head (batch_size, seq_len, second_vocab_size)
            idx: Target indices for primary head (batch_size, seq_len)
            target_second: Optional target indices for secondary head (batch_size, seq_len)

        Returns:
            Tuple of (primary_loss, secondary_loss), either may be None if not computed

        Notes:
            - Returns (None, None) in eval mode
            - Secondary loss is scaled by aux_head_weight if computed
        """
        if not self.training:
            return None, None

        if primary_logits is None or second_logits is None:
            raise ValueError(
                "Both primary and secondary logits must be provided in training mode"
            )

        primary_loss = None
        second_loss = None

        try:
            # Compute primary loss
            if idx is not None:
                primary_loss = self._compute_loss(primary_logits, idx)
                if self.debug_mode:
                    self._debug_log(
                        "primary_loss",
                        {
                            "value": primary_loss.item(),
                            "logits_shape": primary_logits.shape,
                            "targets_shape": idx.shape,
                        },
                    )
            else:
                warnings.warn("No targets provided for primary head in training mode")

            # Compute secondary loss
            if target_second is not None:
                raw_second_loss = self._compute_loss(second_logits, target_second)
                second_loss = raw_second_loss * self.config.aux_head_weight

                if self.debug_mode:
                    self._debug_log(
                        "second_loss",
                        {
                            "raw_value": raw_second_loss.item(),
                            "scaled_value": second_loss.item(),
                            "scale_factor": self.config.aux_head_weight,
                        },
                    )

            # Validate losses
            for loss_name, loss in [
                ("primary", primary_loss),
                ("secondary", second_loss),
            ]:
                if loss is not None and not torch.isfinite(loss):
                    raise ValueError(f"Non-finite {loss_name} loss detected")

            return primary_loss, second_loss

        except RuntimeError as e:
            if "out of memory" in str(e):
                if self.debug_mode:
                    self._debug_log(
                        "oom_error_losses",
                        {
                            "primary_logits_shape": primary_logits.shape,
                            "second_logits_shape": second_logits.shape,
                        },
                    )
            raise

        finally:
            # Update loss statistics if in training mode
            if self.training and primary_loss is not None:
                self._update_loss_stats(primary_loss.item(), "primary")
            if self.training and second_loss is not None:
                self._update_loss_stats(second_loss.item(), "secondary")

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross entropy loss with improved numerical stability.

        Args:
            logits: Predicted logits of shape (batch_size, seq_len, vocab_size)
            targets: Target indices of shape (batch_size, seq_len)

        Returns:
            Scaled cross entropy loss

        Raises:
            ValueError: If tensor shapes are incompatible
            RuntimeError: If tensors are on different devices
        """
        if logits.device != targets.device:
            raise RuntimeError(
                f"Logits on {logits.device} but targets on {targets.device}"
            )

        if logits.shape[:-1] != targets.shape:
            raise ValueError(
                f"Logits shape {logits.shape} incompatible with targets shape {targets.shape}"
            )

        # Cache the float conversion if needed
        logits_float = logits if logits.dtype == torch.float32 else logits.float()

        try:
            with torch.cuda.amp.autocast(enabled=False):
                loss = F.cross_entropy(
                    logits_float.view(-1, logits_float.size(-1)),
                    targets.view(-1),
                    ignore_index=self.config.ignore_index,  # Move to config
                )

                # Check for invalid values
                if not torch.isfinite(loss):
                    if self.debug_mode:
                        self._debug_log(
                            "invalid_loss",
                            {
                                "loss": loss.item(),
                                "logits_stats": {
                                    "min": logits.min().item(),
                                    "max": logits.max().item(),
                                    "mean": logits.mean().item(),
                                },
                            },
                        )
                    raise ValueError("Loss is not finite")

                return self._scale_loss(loss)

        except RuntimeError as e:
            if "out of memory" in str(e):
                if self.debug_mode:
                    self._debug_log(
                        "oom_error",
                        {
                            "logits_shape": logits.shape,
                            "targets_shape": targets.shape,
                        },
                    )
            raise

    def _scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss with bounds to prevent numerical instability."""
        if loss.item() > self.max_loss_value:
            self.loss_scale_factor = max(
                self.min_loss_scale, self.loss_scale_factor * self.loss_scale_decay
            )
            return loss * self.loss_scale_factor
        return loss

    def _get_next_token_logits(
        self, idx: torch.Tensor, target_head: str
    ) -> torch.Tensor:
        """Get logits for next token generation."""
        primary_logits, second_logits, _, _ = self(idx, position=idx.size(1))
        logits = primary_logits if target_head == "primary" else second_logits
        return logits[:, -1, :]

    def _apply_sampling_params(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
    ) -> torch.Tensor:
        """Apply temperature and top-k sampling to logits."""
        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        return logits

    def _debug_log(self, stage: str, x: torch.Tensor) -> None:
        """Log debug information when debug mode is enabled."""
        if not self.debug_mode:
            return

        print(f"\nDebug Info - {stage}:")
        print(f"Shape: {x.shape}")
        print(f"Device: {x.device}")
        print(
            f"Stats: min={x.min().item():.3f}, max={x.max().item():.3f}, "
            f"mean={x.mean().item():.3f}, std={x.std().item():.3f}"
        )
        has_nan = torch.isnan(x).any()
        has_inf = torch.isinf(x).any()
        if has_nan:
            print("WARNING: NaN values detected!")
        if has_inf:
            print("WARNING: Inf values detected!")

    def _update_perf_stats(self, start_time: float) -> None:
        """Update performance statistics with time and memory usage."""
        self.perf_stats["forward_time"].append(time.time() - start_time)
        if torch.cuda.is_available():
            self.perf_stats["memory_used"].append(torch.cuda.max_memory_allocated())

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics with proper handling of empty stats."""
        if not self.perf_stats["forward_time"]:
            return {}

        stats = {}
        for key, values in self.perf_stats.items():
            if values:
                stats[f"avg_{key}"] = sum(values) / len(values)
                stats[f"max_{key}"] = max(values)

        return stats

    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        for key in self.perf_stats:
            self.perf_stats[key].clear()

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get total number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wte.weight.numel()
        return n_params

    def save_state(self, path: str) -> None:
        """Save model state including performance statistics."""
        state = {
            "model_state": self.state_dict(),
            "config": self.config,
            "perf_stats": dict(self.perf_stats),
        }
        torch.save(state, path)

    def _update_loss_stats(self, loss_value: float, loss_type: str) -> None:
        """Update running statistics for loss values.

        Args:
            loss_value: Current loss value
            loss_type: Type of loss ('primary' or 'secondary')

        Notes:
            - Maintains running statistics including mean, std, min, max
            - Uses bounded storage via deque
            - Skips invalid values (inf/nan)
            - Thread-safe updates
        """
        if not isinstance(loss_value, (int, float)) or not math.isfinite(loss_value):
            if self.debug_mode:
                self._debug_log(
                    f"invalid_loss_stat", {"value": loss_value, "type": loss_type}
                )
            return

        # Initialize stats dict for this loss type if needed
        if not hasattr(self, "_loss_stats"):
            self._loss_stats = {}

        if loss_type not in self._loss_stats:
            self._loss_stats[loss_type] = {
                "values": deque(maxlen=self.max_stats_size),
                "running_mean": 0.0,
                "running_variance": 0.0,
                "count": 0,
                "min_value": float("inf"),
                "max_value": float("-inf"),
                "last_updated": time.time(),
            }

        stats = self._loss_stats[loss_type]

        # Thread-safe update using a lock
        with threading.Lock():
            # Update min/max
            stats["min_value"] = min(stats["min_value"], loss_value)
            stats["max_value"] = max(stats["max_value"], loss_value)

            # Update running mean and variance using Welford's online algorithm
            count = stats["count"]
            old_mean = stats["running_mean"]

            count += 1
            delta = loss_value - old_mean
            new_mean = old_mean + delta / count
            delta2 = loss_value - new_mean

            stats["running_mean"] = new_mean
            stats["running_variance"] += delta * delta2
            stats["count"] = count

            # Store the value
            stats["values"].append(loss_value)
            stats["last_updated"] = time.time()

            # Log if in debug mode
            if self.debug_mode:
                self._debug_log(
                    f"{loss_type}_loss_stats",
                    {
                        "current_value": loss_value,
                        "running_mean": new_mean,
                        "running_std": (
                            math.sqrt(stats["running_variance"] / count)
                            if count > 1
                            else 0
                        ),
                        "min": stats["min_value"],
                        "max": stats["max_value"],
                        "count": count,
                    },
                )

            # Check for concerning patterns
            if count > 10:  # Only check after collecting some samples
                current_std = math.sqrt(stats["running_variance"] / count)

                # Check for loss explosion
                if loss_value > stats["running_mean"] + 5 * current_std:
                    warnings.warn(
                        f"{loss_type} loss value {loss_value:.4f} is unusually high "
                        f"(> 5 std dev from mean {stats['running_mean']:.4f})"
                    )

                # Check for loss collapse
                if loss_value < 1e-8 and stats["running_mean"] > 1e-4:
                    warnings.warn(
                        f"{loss_type} loss value {loss_value:.4e} is unusually low "
                        f"(mean is {stats['running_mean']:.4e})"
                    )

    def get_loss_stats(self, loss_type: str) -> Dict[str, float]:
        """Get current loss statistics.

        Args:
            loss_type: Type of loss to get statistics for

        Returns:
            Dictionary containing current loss statistics
        """
        if not hasattr(self, "_loss_stats") or loss_type not in self._loss_stats:
            return {}

        stats = self._loss_stats[loss_type]
        count = stats["count"]

        if count == 0:
            return {}

        return {
            "mean": stats["running_mean"],
            "std": math.sqrt(stats["running_variance"] / count) if count > 1 else 0,
            "min": stats["min_value"],
            "max": stats["max_value"],
            "count": count,
            "last_updated": stats["last_updated"],
        }
