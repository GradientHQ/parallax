"""Utility functions for handling prefix cache in attention layers."""

from typing import Optional

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention

from parallax.server.cache.base import BaseCache


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
    use_batch_processing: bool = False,
) -> mx.array:
    """
    Compute attention with prefix cache support.

    This function handles the common pattern of:
    1. Reading prefix KV from cache
    2. Concatenating with new KV
    3. Creating causal mask
    4. Computing attention

    Args:
        queries: (batch, n_heads, target_len, head_dim) - Query tensors
        keys_new: (batch, n_kv_heads, target_len, head_dim) - New key tensors
        values_new: (batch, n_kv_heads, target_len, head_dim) - New value tensors
        cache: BaseCache object containing the layer cache
        block_tables: (batch, max_blocks) - PagedKV block tables
        prefix_lens: (batch,) - Number of prefix tokens already cached
        target_len: Length of new tokens
        scale: Attention scale factor
        num_kv_heads: Number of KV heads
        mask: Optional attention mask (used when no prefix cache)
        use_batch_processing: If True, use batch processing (like qwen3), else per-request

    Returns:
        output: (batch, target_len, n_heads * head_dim) - Output hidden states
    """
    queries.shape[0]

    if use_batch_processing:
        return _compute_attention_with_prefix_cache_batch(
            queries,
            keys_new,
            values_new,
            cache,
            block_tables,
            prefix_lens,
            target_len,
            scale,
            num_kv_heads,
        )
    else:
        return _compute_attention_with_prefix_cache_per_request(
            queries,
            keys_new,
            values_new,
            cache,
            block_tables,
            prefix_lens,
            target_len,
            scale,
            num_kv_heads,
            mask,
        )


def _compute_attention_with_prefix_cache_batch(
    queries: mx.array,
    keys_new: mx.array,
    values_new: mx.array,
    cache: BaseCache,
    block_tables: mx.array,
    prefix_lens: mx.array,
    target_len: int,
    scale: float,
    num_kv_heads: int,
) -> mx.array:
    """Batch processing version (used by qwen3)."""
    batch = queries.shape[0]
    max_prefix_len = int(mx.max(prefix_lens))

    # Prepare new KV in correct shape: (batch, n_kv_heads, target_len, head_dim)
    k_new = keys_new  # (batch, n_kv_heads, target_len, head_dim)
    v_new = values_new  # (batch, n_kv_heads, target_len, head_dim)

    if max_prefix_len > 0:
        # Initialize prefix KV arrays with zeros for padding
        head_dim = k_new.shape[-1]
        prefix_k_batch = mx.zeros(
            (batch, num_kv_heads, max_prefix_len, head_dim), dtype=k_new.dtype
        )  # (batch, n_kv_heads, max_prefix_len, head_dim)
        prefix_v_batch = mx.zeros(
            (batch, num_kv_heads, max_prefix_len, head_dim), dtype=v_new.dtype
        )  # (batch, n_kv_heads, max_prefix_len, head_dim)

        # Batch read prefix KV for all requests
        for i in range(batch):
            prefix_len = int(prefix_lens[i])
            if prefix_len > 0:
                block_table_i = block_tables[i]  # (max_blocks,)
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

    # Create batch causal mask
    full_len = k_full.shape[2]  # max_prefix_len + target_len

    # Create mask: (batch, target_len, full_len)
    row_indices = mx.arange(target_len)[None, :, None]  # (1, target_len, 1)
    col_indices = mx.arange(full_len)[None, None, :]  # (1, 1, full_len)
    prefix_lens_expanded = prefix_lens[:, None, None]  # (batch, 1, 1)

    # Initialize mask: all positions are allowed by default
    causal_mask = mx.zeros((batch, target_len, full_len), dtype=queries.dtype)

    # Mask 1: Invalid prefix positions for requests with shorter prefix
    invalid_prefix_mask = mx.logical_and(
        col_indices >= prefix_lens_expanded, col_indices < max_prefix_len
    )  # (batch, 1, full_len)
    causal_mask = mx.where(
        invalid_prefix_mask, float("-inf"), causal_mask
    )  # (batch, target_len, full_len)

    # Mask 2: Causal mask for new tokens
    new_token_start = max_prefix_len
    new_token_col_indices = col_indices - new_token_start
    is_new_token_pos = col_indices >= new_token_start
    causal_mask_new = mx.where(
        mx.logical_and(is_new_token_pos, new_token_col_indices > row_indices), float("-inf"), 0.0
    )
    causal_mask = causal_mask + causal_mask_new  # (batch, target_len, full_len)

    # Reshape mask: (batch, 1, target_len, full_len)
    causal_mask = causal_mask[:, None, :, :].astype(queries.dtype)

    # Batch compute attention
    output = scaled_dot_product_attention(
        queries,  # (batch, n_heads, target_len, head_dim)
        k_full,  # (batch, n_kv_heads, full_len, head_dim)
        v_full,  # (batch, n_kv_heads, full_len, head_dim)
        scale=scale,
        mask=causal_mask,
        cache=None,
    )
    output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
    return output


def _compute_attention_with_prefix_cache_per_request(
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
) -> mx.array:
    """Per-request processing version (used by most models)."""
    batch = queries.shape[0]
    output_list = []

    for i in range(batch):
        prefix_len = int(prefix_lens[i])
        q_i = queries[i : i + 1]  # (1, n_heads, target_len, head_dim)
        k_new_i = keys_new[i : i + 1]  # (1, n_kv_heads, target_len, head_dim)
        v_new_i = values_new[i : i + 1]  # (1, n_kv_heads, target_len, head_dim)

        if prefix_len > 0:
            block_table_i = block_tables[i]  # (max_blocks,)
            prefix_k, prefix_v = cache.read_prefix_kv(block_table_i, prefix_len, num_kv_heads)
            # prefix_k: (n_kv_heads, prefix_len, head_dim)
            # prefix_v: (n_kv_heads, prefix_len, head_dim)
            # Reshape to (1, n_kv_heads, prefix_len, head_dim)
            prefix_k = prefix_k[None, ...]  # (1, n_kv_heads, prefix_len, head_dim)
            prefix_v = prefix_v[None, ...]  # (1, n_kv_heads, prefix_len, head_dim)

            # Concatenate prefix and new KV
            k_full = mx.concatenate(
                [prefix_k, k_new_i], axis=2
            )  # (1, n_kv_heads, prefix_len + target_len, head_dim)
            v_full = mx.concatenate(
                [prefix_v, v_new_i], axis=2
            )  # (1, n_kv_heads, prefix_len + target_len, head_dim)
        else:
            k_full = k_new_i
            v_full = v_new_i

        # Compute attention for this request
        full_len = k_full.shape[2]
        # Correct causal mask: position j can attend to positions 0..j
        row_indices = mx.arange(target_len)[:, None] + prefix_len  # actual positions
        col_indices = mx.arange(full_len)[None, :]
        causal_mask = mx.where(col_indices <= row_indices, 0.0, float("-inf"))
        causal_mask = causal_mask[None, None, :, :].astype(
            q_i.dtype
        )  # (1, 1, target_len, full_len)

        out_i = scaled_dot_product_attention(
            q_i,
            k_full,
            v_full,
            scale=scale,
            mask=causal_mask,
            cache=None,
        )
        output_list.append(out_i)

    output = mx.concatenate(output_list, axis=0)
    output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
    return output
