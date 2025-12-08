from typing import Tuple, Optional
import mlx.core as mx
from parallax.server.cache.base import BaseCache


class LinearCache(BaseCache):

    def __init__(
        self,
        max_num_seqs: int = 128,
        conv_dim: Optional[int] = None,
        conv_kernel_size: Optional[int] = None,
        linear_k_dim: Optional[int] = None,
        linear_v_dim: Optional[int] = None,
        linear_num_k_heads: Optional[int] = None,
        linear_num_v_heads: Optional[int] = None,
        dtype: mx.Dtype = mx.float16,
    ):
        self.max_num_seqs = max_num_seqs
        self.dtype = dtype

        self.conv_state_cache = None
        self.linear_state_cache = None

        if conv_dim is not None and conv_kernel_size is not None:
            # Conv State: (1, max_num_seqs, conv_dim, kernel_size - 1)
            conv_state_len = conv_kernel_size - 1
            self.conv_state_cache = mx.zeros(
                (1, max_num_seqs, conv_dim, conv_state_len), dtype=dtype
            )
            mx.eval(self.conv_state_cache)

        if (
            linear_k_dim is not None
            and linear_v_dim is not None
            and linear_num_k_heads is not None
            and linear_num_v_heads is not None
        ):
            # Linear State: (1, max_num_seqs, num_heads, k_dim, v_dim)
            self.linear_state_cache = mx.zeros(
                (
                    1,
                    max_num_seqs,
                    linear_num_k_heads,
                    linear_k_dim,
                    linear_v_dim,
                ),
                dtype=dtype,
            )
            mx.eval(self.linear_state_cache)

    def get_cache(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        return self.conv_state_cache, self.linear_state_cache

    def get_indexer_cache(self) -> Optional[mx.array]:
        return None
