from typing import Tuple

import mlx.core as mx

from parallax.server.cache.base import BaseCache


class KVCache(BaseCache):
    """
    Standard Paged KV Cache for a single layer.
    Shape: (1, num_blocks, num_kv_heads, block_size, head_dim)
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        head_dim_v: int,
        dtype: mx.Dtype,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.dtype = dtype

        self.key_cache = mx.zeros((1, num_blocks, num_kv_heads, block_size, head_dim), dtype=dtype)
        self.value_cache = mx.zeros(
            (1, num_blocks, num_kv_heads, block_size, head_dim_v), dtype=dtype
        )
        mx.eval(self.key_cache, self.value_cache)

    def get_cache(self) -> Tuple[mx.array, mx.array]:
        return self.key_cache, self.value_cache
