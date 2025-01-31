# gpt.py
import torch
import math
import time
import warnings
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Union, Dict
import triton
import triton.language as tl
from caching import OptimizedKVCache
from config import GPTConfig
from blocks import GPTBlock


class GPT(nn.Module):
    """Complete GPT model with mixed precision training support."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embeddings with weight tying support
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Initialize weights
        self.apply(self._init_weights)

        # Optional weight quantization
        if config.use_quantization:
            self.quantize_weights()

        # KV cache for efficient inference
        self.kv_cache = (
            None
            if not config.use_kv_cache
            else OptimizedKVCache(
                max_seq_len=config.block_size,
                n_head=config.n_head,
                head_dim=config.n_embd // config.n_head,
                device=next(self.parameters()).device,
            )
        )

        # GradScaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, mask: Optional[torch.Tensor] = None, position: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model with automatic mixed precision support.
        Args:
            idx: Input token indices [batch_size, sequence_length]
            mask: Attention mask [batch_size, sequence_length]
            position: Position for KV cache
        Returns:
            tuple: (logits, loss if training)
        """
        device = idx.device
        b, t = idx.size()

        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"

        # Use autocast for mixed precision computations
        with torch.cuda.amp.autocast():
            # Get token embeddings
            x = self.wte(idx)

            # Forward through transformer blocks
            for block in self.blocks:
                x = block(x, mask=mask, kv_cache=self.kv_cache, position=position)

            # Final layer norm
            x = self.ln_f(x)

            # Get logits by using embedding weight matrix (weight tying)
            logits = F.linear(x, self.wte.weight)

            # Calculate loss if in training mode
            loss = None
            if self.training:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1))

        return logits, loss

    def training_step(
        self, batch: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> torch.Tensor:
        """
        Perform a single training step with mixed precision.
        Args:
            batch: Input batch data
            optimizer: The optimizer instance
        Returns:
            loss: The scaled loss value
        """
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with autocasting
        with torch.cuda.amp.autocast():
            logits, loss = self(batch)

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        # Unscale gradients and check for infs/nans
        self.scaler.unscale_(optimizer)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # Step optimizer and update scaler
        self.scaler.step(optimizer)
        self.scaler.update()

        return loss

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> Tuple[torch.optim.AdamW, torch.cuda.amp.GradScaler]:
        """
        Configure optimizer and grad scaler for mixed precision training.
        """
        # Separate parameters that should and shouldn't have weight decay
        decay = set()
        no_decay = set()

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, (nn.Linear, nn.Embedding)):
                    decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer, self.scaler

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> float:
        """
        Train for one epoch with mixed precision.
        Args:
            dataloader: Training data loader
            optimizer: Optimizer instance
            epoch: Current epoch number
        Returns:
            average_loss: Average loss for the epoch
        """
        self.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(next(self.parameters()).device)
            loss = self.training_step(batch, optimizer)
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        return total_loss / len(dataloader)
