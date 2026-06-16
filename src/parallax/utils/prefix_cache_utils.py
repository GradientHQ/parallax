"""Utility functions for handling prefix cache in attention layers."""

from typing import Optional

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention

from parallax.server.cache.base import BaseCache


def prepare_attention_with_prefix_cache(
    queries: mx.array,
    keys_new: mx.array,
    values_new: mx.array,
    cache: BaseCache,
    block_tables: mx.array,
    prefix_lens: mx.array,
    target_len: int,
    num_kv_heads: int,
    context_lengths: Optional[mx.array] = None,
    window_size: Optional[int] = None,
):
    """Read prefix KV, concatenate with new KV, and build a prefix-aware mask."""
    batch = queries.shape[0]
    max_prefix_len = int(mx.max(prefix_lens))

    # Prepare new KV in correct shape: (batch, n_kv_heads, target_len, head_dim)
    k_new = keys_new  # (batch, n_kv_heads, target_len, head_dim)
    v_new = values_new  # (batch, n_kv_heads, target_len, head_dim)

    if max_prefix_len > 0:
        # Initialize prefix KV arrays with zeros for padding
        head_dim = k_new.shape[-1]
        value_dim = v_new.shape[-1]
        prefix_k_batch = mx.zeros(
            (batch, num_kv_heads, max_prefix_len, head_dim), dtype=k_new.dtype
        )  # (batch, n_kv_heads, max_prefix_len, head_dim)
        prefix_v_batch = mx.zeros(
            (batch, num_kv_heads, max_prefix_len, value_dim), dtype=v_new.dtype
        )  # (batch, n_kv_heads, max_prefix_len, head_dim)

        # Batch read prefix KV for all requests
        for i in range(batch):
            prefix_len = int(prefix_lens[i])
            if prefix_len > 0:
                block_table_i = block_tables[i : i + 1].reshape(
                    -1
                )  # (max_blocks,) - use slice to avoid copy
                prefix_k, prefix_v = cache.read_prefix_kv(block_table_i, prefix_len, num_kv_heads)
                # prefix_k: (n_kv_heads, prefix_len, head_dim)
                # prefix_v: (n_kv_heads, prefix_len, head_dim)
                prefix_k_batch[i, :, :prefix_len, :] = prefix_k
                prefix_v_batch[i, :, :prefix_len, :] = prefix_v

        # Concatenate prefix and new KV: (batch, n_kv_heads, max_prefix_len + target_len, head_dim)
        k_full = mx.concatenate([prefix_k_batch, k_new], axis=2)
        v_full = mx.concatenate([prefix_v_batch, v_new], axis=2)
    else:
        # No prefix cache, use only new KV
        k_full = k_new
        v_full = v_new

    row_indices = mx.arange(target_len, dtype=mx.int32)
    if context_lengths is None:
        new_lens = mx.full(prefix_lens.shape, target_len, dtype=mx.int32)
    else:
        new_lens = context_lengths - prefix_lens
    q_positions = prefix_lens[:, None] + row_indices[None, :]

    if max_prefix_len > 0:
        prefix_positions = mx.arange(max_prefix_len, dtype=mx.int32)
        prefix_positions = mx.broadcast_to(prefix_positions[None, :], (batch, max_prefix_len))
        prefix_valid = prefix_positions < prefix_lens[:, None]
    else:
        prefix_positions = mx.zeros((batch, 0), dtype=mx.int32)
        prefix_valid = mx.zeros((batch, 0), dtype=mx.bool_)

    new_positions = prefix_lens[:, None] + row_indices[None, :]
    new_valid = row_indices[None, :] < new_lens[:, None]
    key_positions = mx.concatenate([prefix_positions, new_positions], axis=1)
    key_valid = mx.concatenate([prefix_valid, new_valid], axis=1)
    query_valid = new_valid
    causal = key_positions[:, None, :] <= q_positions[:, :, None]
    valid = key_valid[:, None, :] & query_valid[:, :, None] & causal

    if window_size is not None:
        window_start = mx.maximum(0, q_positions[:, :, None] - window_size + 1)
        valid = valid & (key_positions[:, None, :] >= window_start)

    causal_mask = mx.where(valid[:, None], 0.0, -float("inf")).astype(queries.dtype)
    return k_full, v_full, causal_mask, q_positions, key_positions


def compute_attention_with_prefix_cache(
    queries: mx.array,
    keys_new: mx.array,
    values_new: mx.array,
    cache: BaseCache,
    block_tables: mx.array,
    prefix_lens: mx.array,
    target_len: int,
    scale: float,
    num_kv_heads: int,
    mask: Optional[mx.array] = None,
    sinks: Optional[mx.array] = None,
    window_size: Optional[int] = None,
) -> mx.array:
    """
    Compute attention with prefix cache support.

    This function handles the common pattern of:
    1. Reading prefix KV from cache
    2. Concatenating with new KV
    3. Creating causal mask
    4. Computing attention
    """
    batch = queries.shape[0]
    k_full, v_full, causal_mask, _, _ = prepare_attention_with_prefix_cache(
        queries,
        keys_new,
        values_new,
        cache,
        block_tables,
        prefix_lens,
        target_len,
        num_kv_heads,
        window_size=window_size,
    )

    # Batch compute attention
    attention_kwargs = {}
    if sinks is not None:
        attention_kwargs["sinks"] = sinks
    output = scaled_dot_product_attention(
        queries,  # (batch, n_heads, target_len, head_dim)
        k_full,  # (batch, n_kv_heads, full_len, head_dim)
        v_full,  # (batch, n_kv_heads, full_len, head_dim)
        scale=scale,
        mask=causal_mask,
        cache=None,
        **attention_kwargs,
    )
    output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
    return output
