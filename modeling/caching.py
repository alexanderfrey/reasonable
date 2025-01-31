# caching.py
from typing import Tuple
import torch
import torch.nn as nn


class OptimizedKVCache:
    def __init__(
        self,
        max_seq_len: int,
        n_head: int,
        head_dim: int,
        device: torch.device,
    ):
        self.max_seq_len = max_seq_len
        self.n_head = n_head
        self.head_dim = head_dim
        self.device = device
        self.current_batch_size = None
        self.cache_k = None
        self.cache_v = None
        self.curr_len = 0
        self.use_half_precision = True

    def _ensure_cache_size(self, batch_size: int):
        """Dynamically resize cache if needed"""
        if self.cache_k is None or self.current_batch_size != batch_size:
            dtype = torch.float16 if self.use_half_precision else torch.float32
            self.current_batch_size = batch_size
            self.cache_k = torch.zeros(
                (batch_size, self.n_head, self.max_seq_len, self.head_dim),
                dtype=dtype,
                pin_memory=True,
                device=self.device,
            )
            self.cache_v = torch.zeros_like(self.cache_k)
            self.curr_len = 0

    def update(self, key: torch.Tensor, value: torch.Tensor, position: int):
        """Update cache with new key-value pairs."""
        batch_size = key.size(0)
        self._ensure_cache_size(batch_size)

        self.cache_k[:, :, position : position + key.size(2)] = key
        self.cache_v[:, :, position : position + value.size(2)] = value
        self.curr_len = max(self.curr_len, position + key.size(2))

    def get_kv(self, position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached key-value pairs."""
        if self.cache_k is None:
            raise RuntimeError("Cache not initialized. Call update() first.")
        return (self.cache_k[:, :, :position], self.cache_v[:, :, :position])

    def clear(self):
        """Clear the cache."""
        self.cache_k = None
        self.cache_v = None
        self.curr_len = 0
        self.current_batch_size = None
