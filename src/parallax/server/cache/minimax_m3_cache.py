from typing import Tuple

import mlx.core as mx

from parallax.server.cache.kv_cache import KVCachePacked


class MiniMaxM3SparseCache(KVCachePacked):
    """
    Paged KV cache with an additional MiniMax-M3 sparse index-key cache.

    The main K/V cache keeps Parallax's packed layout for existing paged
    attention kernels. The side index cache follows the DeepSeek sparse-cache
    layout so existing index-store utilities can write to it.
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
        super().__init__(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            dtype=dtype,
        )
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_key_cache = mx.zeros(
            (
                1,
                num_blocks,
                index_n_heads,
                block_size,
                index_head_dim,
            ),
            dtype=dtype,
        )
        mx.eval(self.index_key_cache)

    def get_indexer_cache(self) -> mx.array:
        return self.index_key_cache

    def read_index_k(
        self,
        block_table: mx.array,
        context_len: int,
    ) -> mx.array:
        """
        Read index keys for one request.

        Returns:
            index_k: (index_heads, context_len, index_head_dim)
        """
        positions = mx.arange(context_len)
        block_indices = positions // self.block_size
        offsets = positions % self.block_size
        physical_blocks = block_table[block_indices]

        index_k = self.index_key_cache[0, physical_blocks, :, offsets, :]
        return index_k.transpose(1, 0, 2)

    def read_kv(
        self,
        block_table: mx.array,
        context_len: int,
    ) -> Tuple[mx.array, mx.array]:
        return self.read_prefix_kv(block_table, context_len, self.num_kv_heads)
