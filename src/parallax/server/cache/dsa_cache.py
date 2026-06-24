from typing import Optional, Tuple

import mlx.core as mx

from parallax.server.cache.base import BaseCache


class DeepSeekSparseCache(BaseCache):
    """
    Compressed MLA cache with additional indexer cache for DeepSeek/GLM DSA.

    Main attention stores the MLA latent and RoPE key instead of expanded K/V:
    - latent_cache: (1, num_blocks, 1, block_size, kv_lora_rank)
    - rope_cache:   (1, num_blocks, 1, block_size, qk_rope_head_dim)

    The DSA indexer key is shared across query heads for DeepSeek-V3.2/GLM, so
    it can use fewer key heads than the indexer query heads.
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
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        index_key_heads: int = 1,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.dtype = dtype
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_key_heads = index_key_heads

        self.latent_cache = mx.zeros(
            (1, num_blocks, 1, block_size, kv_lora_rank),
            dtype=dtype,
        )
        self.rope_cache = mx.zeros(
            (1, num_blocks, 1, block_size, qk_rope_head_dim),
            dtype=dtype,
        )
        self.indexer_key_cache = mx.zeros(
            (
                1,
                num_blocks,
                index_key_heads,
                block_size,
                index_head_dim,
            ),
            dtype=dtype,
        )
        mx.eval(self.latent_cache, self.rope_cache, self.indexer_key_cache)

    def get_cache(self) -> Tuple[mx.array, mx.array]:
        return self.latent_cache, self.rope_cache

    def is_packed(self) -> bool:
        return False

    def get_indexer_cache(self) -> Optional[mx.array]:
        return self.indexer_key_cache

    def read_prefix_mla(
        self,
        block_table: mx.array,
        prefix_len: int,
    ) -> Tuple[mx.array, mx.array]:
        """
        Read compressed MLA cache for one request.

        Returns:
            latent: (1, prefix_len, kv_lora_rank)
            rope:   (1, prefix_len, qk_rope_head_dim)
        """
        positions = mx.arange(prefix_len)
        block_indices = positions // self.block_size
        offsets = positions % self.block_size
        physical_blocks = block_table[block_indices]

        latent = self.latent_cache[0, physical_blocks, :, offsets, :]
        rope = self.rope_cache[0, physical_blocks, :, offsets, :]
        return latent.transpose(1, 0, 2), rope.transpose(1, 0, 2)

    def read_mla_positions(
        self,
        block_table: mx.array,
        positions: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Read compressed MLA cache at arbitrary logical token positions.

        Returns:
            latent: (1, len(positions), kv_lora_rank)
            rope:   (1, len(positions), qk_rope_head_dim)
        """
        block_indices = positions // self.block_size
        offsets = positions % self.block_size
        physical_blocks = block_table[block_indices]

        latent = self.latent_cache[0, physical_blocks, :, offsets, :]
        rope = self.rope_cache[0, physical_blocks, :, offsets, :]
        return latent.transpose(1, 0, 2), rope.transpose(1, 0, 2)

    def read_prefix_kv(
        self,
        block_table: mx.array,
        prefix_len: int,
        num_kv_heads: int,
    ) -> Tuple[mx.array, mx.array]:
        """Return compressed MLA cache for callers that use the BaseCache API."""
        return self.read_prefix_mla(block_table, prefix_len)

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
