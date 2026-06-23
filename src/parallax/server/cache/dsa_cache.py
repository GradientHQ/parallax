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
        index_head_dim: int,
        index_n_heads: int,
    ):
        super().__init__(num_blocks, block_size, num_kv_heads, head_dim, head_dim_v, dtype)
        self.indexer_key_cache = mx.zeros(
            (
                1,
                num_blocks,
                index_n_heads,
                block_size,
                index_head_dim,
            ),
            dtype=dtype,
        )
        mx.eval(self.indexer_key_cache)

    def get_indexer_cache(self) -> Optional[mx.array]:
        return self.indexer_key_cache

    def read_index_k(
        self,
        block_table: mx.array,
        context_len: int,
    ) -> mx.array:
        """
        Read sparse index keys for one request.

        Returns:
            index_k: (index_heads, context_len, index_head_dim)
        """
        positions = mx.arange(context_len)
        block_indices = positions // self.block_size
        offsets = positions % self.block_size
        physical_blocks = block_table[block_indices]

        index_k = self.indexer_key_cache[0, physical_blocks, :, offsets, :]
        return index_k.transpose(1, 0, 2)
