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

    def is_packed(self) -> bool:
        """KVCache uses standard (non-packed) format."""
        return False

    def read_prefix_kv(
        self,
        block_table: mx.array,
        prefix_len: int,
        num_kv_heads: int,
    ) -> Tuple[mx.array, mx.array]:
        """
        Read prefix KV from standard KVCache.

        Args:
            block_table: (max_blocks,) - Block table for the request
            prefix_len: Number of prefix tokens to read
            num_kv_heads: Number of KV heads

        Returns:
            prefix_k: (num_kv_heads, prefix_len, head_dim) - Prefix keys
            prefix_v: (num_kv_heads, prefix_len, head_dim_v) - Prefix values
        """
        # Calculate all block indices and offsets using array operations
        positions = mx.arange(prefix_len)
        block_indices = positions // self.block_size  # (prefix_len,)
        offsets = positions % self.block_size  # (prefix_len,)
        physical_blocks = block_table[block_indices]  # (prefix_len,)

        # Extract all tokens at once using array indexing
        # key_cache: (1, num_blocks, n_kv_heads, block_size, head_dim)
        # value_cache: (1, num_blocks, n_kv_heads, block_size, head_dim_v)
        prefix_k = self.key_cache[
            0, physical_blocks, :, offsets, :
        ]  # (prefix_len, n_kv_heads, head_dim)
        prefix_v = self.value_cache[
            0, physical_blocks, :, offsets, :
        ]  # (prefix_len, n_kv_heads, head_dim_v)

        # Transpose to (n_kv_heads, prefix_len, head_dim)
        prefix_k = prefix_k.transpose(1, 0, 2)  # (n_kv_heads, prefix_len, head_dim)
        prefix_v = prefix_v.transpose(1, 0, 2)  # (n_kv_heads, prefix_len, head_dim_v)

        return prefix_k, prefix_v


def get_packing_factor(dtype: mx.Dtype) -> int:
    """
    Get the packing factor 'x' for the specific dtype.
    Required for Metal Vectorization.
    float32 -> 4
    float16/bfloat16 -> 8
    """
    if dtype == mx.float32:
        return 4
    elif dtype == mx.float16 or dtype == mx.bfloat16:
        return 8
    else:
        raise ValueError(f"Unsupported dtype for KV Cache: {dtype}")


class KVCachePacked(BaseCache):
    """
    Paged KV Cache for a single layer optimized for Parallax Metal Kernels.

    Memory Layouts:
    - Key:   [num_blocks, num_kv_heads, head_dim // x, block_size, x]
    - Value: [num_blocks, num_kv_heads, head_dim_v, block_size]

    Where 'x' is the packing factor (4 for fp32, 8 for fp16).
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

        self.x = get_packing_factor(dtype)

        if head_dim % self.x != 0:
            raise ValueError(
                f"Head dim {head_dim} must be divisible by packing factor {self.x} "
                f"for dtype {dtype}"
            )

        # Shape: [blocks, heads, dim/x, block_size, x]
        self.key_cache = mx.zeros(
            (num_blocks, num_kv_heads, head_dim // self.x, block_size, self.x), dtype=dtype
        )
        # Shape: [blocks, heads, dim, block_size]
        self.value_cache = mx.zeros((num_blocks, num_kv_heads, head_dim_v, block_size), dtype=dtype)

        mx.eval(self.key_cache, self.value_cache)

    def get_cache(self) -> Tuple[mx.array, mx.array]:
        return self.key_cache, self.value_cache

    def is_packed(self) -> bool:
        """KVCachePacked uses packed format."""
        return True

    def read_prefix_kv(
        self,
        block_table: mx.array,
        prefix_len: int,
        num_kv_heads: int,
    ) -> Tuple[mx.array, mx.array]:
        """
        Read prefix KV from packed KVCachePacked.

        Args:
            block_table: (max_blocks,) - Block table for the request
            prefix_len: Number of prefix tokens to read
            num_kv_heads: Number of KV heads

        Returns:
            prefix_k: (num_kv_heads, prefix_len, head_dim) - Prefix keys
            prefix_v: (num_kv_heads, prefix_len, head_dim_v) - Prefix values
        """
        # Calculate all block indices and offsets using array operations
        positions = mx.arange(prefix_len)
        block_indices = positions // self.block_size  # (prefix_len,)
        offsets = positions % self.block_size  # (prefix_len,)
        physical_blocks = block_table[block_indices]  # (prefix_len,)

        # Extract all tokens at once using array indexing
        # KVCachePacked format
        # key_cache: (num_blocks, num_kv_heads, head_dim // x, block_size, x)
        # value_cache: (num_blocks, num_kv_heads, head_dim_v, block_size)
        prefix_k = self.key_cache[
            physical_blocks, :, :, offsets, :
        ]  # (prefix_len, n_kv_heads, head_dim // x, x)
        # Reshape to (prefix_len, n_kv_heads, head_dim)
        prefix_k = prefix_k.reshape(
            prefix_len, num_kv_heads, -1
        )  # (prefix_len, n_kv_heads, head_dim)
        prefix_v = self.value_cache[
            physical_blocks, :, :, offsets
        ]  # (prefix_len, n_kv_heads, head_dim_v)

        # Transpose to (n_kv_heads, prefix_len, head_dim)
        prefix_k = prefix_k.transpose(1, 0, 2)  # (n_kv_heads, prefix_len, head_dim)
        prefix_v = prefix_v.transpose(1, 0, 2)  # (n_kv_heads, prefix_len, head_dim_v)

        return prefix_k, prefix_v
