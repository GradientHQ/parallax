from typing import Optional, Tuple

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
        indexer_key_head_dim: Optional[int] = None,
        indexer_num_kv_heads: Optional[int] = None,
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
        self.indexer_key_cache = None
        if indexer_key_head_dim is not None and indexer_num_kv_heads is not None:
            self.indexer_key_cache = mx.zeros(
                (
                    1,
                    num_blocks,
                    indexer_num_kv_heads,
                    block_size,
                    indexer_key_head_dim,
                ),
                dtype=dtype,
            )
            mx.eval(self.key_cache, self.value_cache, self.indexer_key_cache)
        else:
            mx.eval(self.key_cache, self.value_cache)

    def get_cache(self) -> Tuple[mx.array, mx.array]:
        return self.key_cache, self.value_cache

    def get_indexer_cache(self) -> Optional[mx.array]:
        return self.indexer_key_cache
