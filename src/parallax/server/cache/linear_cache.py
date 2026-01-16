from typing import Optional, Tuple

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
            conv_state_len = conv_kernel_size - 1
            self.conv_state_cache = mx.zeros(
                (1, max_num_seqs, conv_state_len, conv_dim), dtype=dtype
            )
            mx.eval(self.conv_state_cache)

        if (
            linear_k_dim is not None
            and linear_v_dim is not None
            and linear_num_k_heads is not None
            and linear_num_v_heads is not None
        ):
            self.linear_state_cache = mx.zeros(
                (
                    1,
                    max_num_seqs,
                    linear_num_v_heads,
                    linear_v_dim,
                    linear_k_dim,
                ),
                dtype=dtype,
            )
            mx.eval(self.linear_state_cache)

    def get_cache(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        return self.conv_state_cache, self.linear_state_cache

    def get_indexer_cache(self) -> Optional[mx.array]:
        return None

    def read_states(self, slot_mapping: mx.array) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        conv_state_list = []
        linear_state_list = []

        for slot_idx in slot_mapping:
            slot_idx = int(slot_idx)
            if self.conv_state_cache is not None:
                conv_state_slice = self.conv_state_cache[0, slot_idx]
                conv_state_list.append(conv_state_slice[None, :, :])

            if self.linear_state_cache is not None:
                linear_state_slice = self.linear_state_cache[0, slot_idx]
                linear_state_list.append(linear_state_slice[None, :, :, :])

        conv_states = mx.concatenate(conv_state_list, axis=0) if conv_state_list else None
        linear_states = mx.concatenate(linear_state_list, axis=0) if linear_state_list else None

        return conv_states, linear_states

    def write_states(
        self,
        slot_mapping: mx.array,
        conv_states: mx.array,
        linear_states: Optional[mx.array],
    ):
        for i, slot_idx in enumerate(slot_mapping):
            slot_idx = int(slot_idx)
            if self.conv_state_cache is not None:
                self.conv_state_cache[0, slot_idx] = conv_states[i]

            if self.linear_state_cache is not None and linear_states is not None:
                self.linear_state_cache[0, slot_idx] = linear_states[i]

    def is_packed(self) -> bool:
        """LinearCache doesn't use packed format."""
        return False

    def read_prefix_kv(
        self,
        block_table: mx.array,
        prefix_len: int,
        num_kv_heads: int,
    ) -> Tuple[mx.array, mx.array]:
        """
        LinearCache doesn't support prefix KV reading.
        This method should not be called for LinearCache.
        """
        raise NotImplementedError("LinearCache does not support prefix KV reading")
