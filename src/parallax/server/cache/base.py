from abc import ABC, abstractmethod
from typing import Any, Tuple

import mlx.core as mx


class BaseCache(ABC):
    """Abstract base class for layer-level cache."""

    @abstractmethod
    def get_cache(self) -> Any:
        pass

    @abstractmethod
    def is_packed(self) -> bool:
        """Check if this cache uses packed format."""

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
