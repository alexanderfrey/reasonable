# blocks.py
from typing import Tuple, Optional
import torch
import torch.nn as nn
from gpt import GPTConfig
from attention import FlashAttention, OptimizedPagedMultiLatentAttention
from caching import OptimizedKVCache
from layer.fused import FusedLinear


class GPTBlock(nn.Module):
    """Transformer block for GPT."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # Choose between standard attention and optimized multi-latent attention
        if config.use_flash_attn and hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        ):
            self.attn = FlashAttention(config)
        else:
            self.attn = OptimizedPagedMultiLatentAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)

        # Choose between MoE and standard MLP
        if config.use_moe:
            self.mlp = OptimizedMoELayer(config)
        else:
            self.mlp = nn.Sequential(
                FusedLinear(config.n_embd, 4 * config.n_embd, bias=config.bias),
                nn.GELU(),
                FusedLinear(
                    4 * config.n_embd,
                    config.n_embd,
                    bias=config.bias,
                    dropout=config.dropout,
                ),
            )

        # Optional activation checkpointing
        self.use_checkpoint = config.use_activation_checkpointing

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[OptimizedKVCache] = None,
        position: int = 0,
    ) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mask, kv_cache, position
            )
        return self._forward(x, mask, kv_cache, position)

    def _forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_cache: Optional[OptimizedKVCache],
        position: int,
    ) -> torch.Tensor:
        # Attention with residual connection
        x = x + self.attn(self.ln_1(x), mask=mask, kv_cache=kv_cache, position=position)
        # MLP/MoE with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x
