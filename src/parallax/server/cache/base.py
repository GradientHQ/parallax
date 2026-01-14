from abc import ABC, abstractmethod
from typing import Any, Tuple

import mlx.core as mx


class BaseCache(ABC):
    """Abstract base class for layer-level cache."""

    @abstractmethod
    def get_cache(self) -> Any:
        pass

    def is_packed(self) -> bool:
        """Check if this cache uses packed format."""
        key_cache, _ = self.get_cache()
        # KVCache: (1, num_blocks, n_kv_heads, block_size, head_dim)
        # KVCachePacked: (num_blocks, num_kv_heads, head_dim // x, block_size, x)
        return key_cache.ndim == 5 and key_cache.shape[0] != 1

    @abstractmethod
    def read_prefix_kv(
        self,
        block_table: mx.array,
        prefix_len: int,
        num_kv_heads: int,
    ) -> Tuple[mx.array, mx.array]:
        """
        Read prefix KV from cache for a single request.

        Args:
            block_table: (max_blocks,) - Block table for the request
            prefix_len: Number of prefix tokens to read
            num_kv_heads: Number of KV heads

        Returns:
            prefix_k: (num_kv_heads, prefix_len, head_dim) - Prefix keys
            prefix_v: (num_kv_heads, prefix_len, head_dim_v) - Prefix values
        """
