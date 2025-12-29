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
            (num_blocks, num_kv_heads, head_dim // self.x, block_size, self.x), 
            dtype=dtype
        )
        # Shape: [blocks, heads, dim, block_size]
        self.value_cache = mx.zeros(
            (num_blocks, num_kv_heads, head_dim_v, block_size), 
            dtype=dtype
        )

        mx.eval(self.key_cache, self.value_cache)

    def get_cache(self) -> Tuple[mx.array, mx.array]:
        return self.key_cache, self.value_cache
