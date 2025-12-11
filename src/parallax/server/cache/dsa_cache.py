from typing import Optional

import mlx.core as mx

from parallax.server.cache.kv_cache import KVCache


class DeepSeekSparseCache(KVCache):
    """
    KVCache with additional indexer cache for DeepSeek Sparse Attention.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        head_dim_v: int,
        dtype: mx.Dtype,
        indexer_key_head_dim: int,
        indexer_num_kv_heads: int,
    ):
        super().__init__(num_blocks, block_size, num_kv_heads, head_dim, head_dim_v, dtype)
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
        mx.eval(self.indexer_key_cache)

    def get_indexer_cache(self) -> Optional[mx.array]:
        return self.indexer_key_cache
