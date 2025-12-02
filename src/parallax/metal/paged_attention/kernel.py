import os
from typing import Dict, List, Optional

import mlx.core as mx

# Cache for compiled kernels
_KERNELS: Dict[str, object] = {}


def _get_metal_source(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "r") as f:
        return f.read()


def _type_to_string(dtype: mx.Dtype) -> str:
    if dtype == mx.float32:
        return "float"
    elif dtype == mx.float16:
        return "half"
    elif dtype == mx.bfloat16:
        # Metal 3.1+ supports bfloat, typically via bfloat16_t or using half
        # For now we map to bfloat16_t assuming compiler support
        return "bfloat16_t"
    else:
        raise ValueError(f"Unsupported dtype for paged attention: {dtype}")


def _get_kernel(
    name: str,
    filename: str,
    input_names: List[str],
    output_names: List[str],
    dtype: mx.Dtype = mx.float32,
):
    type_str = _type_to_string(dtype)
    kernel_key = f"{name}_{type_str}"

    if kernel_key not in _KERNELS:
        source = _get_metal_source(filename)
        # Simple template substitution
        source = source.replace("{{T}}", type_str)

        header = """
#include <metal_stdlib>
using namespace metal;
"""
        _KERNELS[kernel_key] = mx.fast.metal_kernel(
            name=name,  # Internal name for MLX JIT cache (not used for dispatch if we hold the object)
            input_names=input_names,
            output_names=output_names,
            source=source,
            header=header,
        )
    return _KERNELS[kernel_key]


def reshape_and_cache(
    key: mx.array,  # (batch, target_len, num_kv_heads, head_dim)
    value: mx.array,  # ...
    key_cache: mx.array,  # (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
    value_cache: mx.array,
    block_tables: mx.array,  # (batch, max_blocks)
    context_lengths: mx.array,  # (batch,)
    block_size: int,
    layer_idx: int,
    slot_mapping: Optional[mx.array] = None,  # (batch,) or (batch * target_len,)
):
    """
    Writes new keys and values into the Paged KV Cache using a custom Metal kernel.
    NOTE: This performs an in-place update on key_cache/value_cache buffers.

    Supports two modes:
    1. Decode (Single Token): slot_mapping is None. Calculated internally.
       Input shape: (batch, num_kv_heads, 1, head_dim)
    2. Prefill (Batch Tokens): slot_mapping is provided.
       Input shape: (batch, num_kv_heads, target_len, head_dim)
    """
    dtype = key.dtype
    if key_cache.dtype != dtype:
        raise ValueError(f"Key cache dtype {key_cache.dtype} does not match key dtype {dtype}")

    # Handle dimensions based on mode
    if slot_mapping is None:
        # Decode Mode
        batch_size = key.shape[0]
        if key.ndim == 4:
            # (batch, 1, num_kv_heads, head_dim) -> (batch, num_kv_heads, head_dim)
            if key.shape[1] == 1:
                key = key.squeeze(1)
                value = value.squeeze(1)
            elif key.shape[2] == 1:
                # Fallback for old layout (batch, num_kv_heads, 1, head_dim)
                key = key.squeeze(2)
                value = value.squeeze(2)

        num_kv_heads = key.shape[1]
        k_head_dim = key.shape[2]
        v_head_dim = value.shape[2]

        # Compute slot_mapping internally
        indices = context_lengths - 1
        block_indices_in_table = indices // block_size
        offsets = indices % block_size
        batch_indices = mx.arange(batch_size)
        physical_block_numbers = block_tables[batch_indices, block_indices_in_table]
        slot_mapping = physical_block_numbers.astype(mx.int64) * block_size + offsets.astype(
            mx.int64
        )

        num_tokens = batch_size

    else:
        # Prefill Mode
        # Key/Value input shape: (batch, target_len, num_kv_heads, head_dim) = BTHD
        # We need to flatten to: (total_tokens, num_kv_heads, head_dim)
        if key.ndim == 4:
            # Input is (B, T, H, D) from optimized Qwen3
            B, T, H, D = key.shape
            key = key.reshape(B * T, H, D)
            # Value might have different D
            V_D = value.shape[3]
            value = value.reshape(B * T, H, V_D)

        num_tokens = key.shape[0]
        num_kv_heads = key.shape[1]
        k_head_dim = key.shape[2]
        v_head_dim = value.shape[2]

        if slot_mapping.shape[0] != num_tokens:
            raise ValueError(f"Slot mapping length {slot_mapping.shape[0]} != tokens {num_tokens}")

    num_layers = key_cache.shape[0]
    num_blocks = key_cache.shape[1]

    # 2. Prepare Constants
    key_stride = num_kv_heads * k_head_dim
    value_stride = num_kv_heads * v_head_dim

    def mk_int(val):
        return mx.array(val, dtype=mx.int32)

    c_key_stride = mk_int(key_stride)
    c_val_stride = mk_int(value_stride)
    c_num_kv = mk_int(num_kv_heads)
    c_k_head_dim = mk_int(k_head_dim)
    c_v_head_dim = mk_int(v_head_dim)
    c_block_size = mk_int(block_size)
    c_layer_idx = mk_int(layer_idx)
    c_num_layers = mk_int(num_layers)
    c_num_blocks = mk_int(num_blocks)

    # Inputs list
    inputs = [
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        c_key_stride,
        c_val_stride,
        c_num_kv,
        c_k_head_dim,
        c_v_head_dim,
        c_block_size,
        c_layer_idx,
        c_num_layers,
        c_num_blocks,
    ]

    # Input names (just for declaration)
    input_names = [
        "key",
        "value",
        "key_cache",
        "value_cache",
        "slot_mapping",
        "key_stride",
        "value_stride",
        "num_kv_heads",
        "k_head_dim",
        "v_head_dim",
        "block_size",
        "layer_idx",
        "num_layers",
        "num_blocks",
    ]

    # 3. Get and Launch Kernel
    kernel = _get_kernel(
        name="reshape_and_cache_kernel",
        filename="reshape_and_cache.metal",
        input_names=input_names,
        output_names=["dummy_out"],
        dtype=dtype,
    )

    # Grid: (num_kv_heads * max_dim, num_tokens, 1)
    max_dim = max(k_head_dim, v_head_dim)
    grid = (num_kv_heads * max_dim, num_tokens, 1)
    thread_group = (min(1024, num_kv_heads * max_dim), 1, 1)

    # Execute
    # We match output_shapes to the grid dimensions to ensure MLX generates 'index' variable
    # corresponding to (num_tokens, num_kv_heads * max_dim).
    outputs = kernel(
        inputs=inputs,
        grid=grid,
        threadgroup=thread_group,
        output_shapes=[(num_tokens, num_kv_heads * max_dim)],
        output_dtypes=[mx.float32],  # Dummy output dtype usually doesn't matter
        verbose=False,
    )

    mx.eval(outputs)

    return key_cache, value_cache


def paged_attention(
    queries: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    block_size: int,
    scale: float,
    num_kv_heads: int,
    layer_idx: int,
    v_head_dim: Optional[int] = None,
    window_size: Optional[int] = None,
    sinks: Optional[mx.array] = None,
) -> mx.array:
    """
    Paged Attention using Metal Kernel.
    """
    batch_size = queries.shape[0]
    num_heads = queries.shape[1]
    dtype = queries.dtype

    if queries.ndim == 4:
        if queries.shape[2] != 1:
            pass
        queries = queries.squeeze(2)

    k_head_dim = queries.shape[2]
    if v_head_dim is None:
        v_head_dim = k_head_dim

    # Use -1 to represent full attention (infinite window)
    c_window_size_val = window_size if window_size is not None else -1

    num_layers = key_cache.shape[0]
    num_total_blocks = key_cache.shape[1]
    max_blocks = block_tables.shape[1]

    # Prepare Constants
    def mk_int(val):
        return mx.array(val, dtype=mx.int32)

    c_num_heads = mk_int(num_heads)
    c_num_kv_heads = mk_int(num_kv_heads)
    c_k_head_dim = mk_int(k_head_dim)
    c_v_head_dim = mk_int(v_head_dim)
    c_block_size = mk_int(block_size)
    c_max_blocks = mk_int(max_blocks)
    c_layer_idx = mk_int(layer_idx)
    c_num_layers = mk_int(num_layers)
    c_num_total_blocks = mk_int(num_total_blocks)
    c_scale = mx.array(scale, dtype=queries.dtype)
    c_window_size = mk_int(c_window_size_val)

    if sinks is None:
        # Pass -inf if no sinks provided to mask it out
        # Assuming num_heads is enough to cover head_idx access
        c_sinks = mx.full((num_heads,), -float("inf"), dtype=queries.dtype)
    else:
        c_sinks = sinks

    inputs = [
        queries,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        c_num_heads,
        c_num_kv_heads,
        c_k_head_dim,
        c_v_head_dim,
        c_block_size,
        c_max_blocks,
        c_layer_idx,
        c_num_layers,
        c_num_total_blocks,
        c_scale,
        c_window_size,
        c_sinks,
    ]

    input_names = [
        "queries",
        "key_cache",
        "value_cache",
        "block_tables",
        "context_lengths",
        "num_heads",
        "num_kv_heads",
        "k_head_dim",
        "v_head_dim",
        "block_size",
        "max_blocks",
        "layer_idx",
        "num_layers",
        "num_total_blocks",
        "scale",
        "window_size",
        "sinks",
    ]

    # For paged_attention, we don't have explicit T in source,
    # but if we use it in future or if we want to support half specialized logic.
    # Currently paged_attention kernel uses `float` for computation but loads from `queries` (T*).
    # Metal implicitly handles T* access if MLX generated correct input types.
    # However, if we use `reshape_and_cache` style template, we should use it here too.
    # But paged_attention_kernel.metal DOES NOT use {{T}} yet.
    # It uses `float q_vec`.
    # Let's keep it as is for now, as Metal handles implicit conversion on load.
    # The only issue is if we write `output` as `float*` but requested `half` output?
    # In Python we should set output_dtypes=[dtype].

    kernel = _get_kernel(
        name="paged_attention_kernel",
        filename="paged_attention_kernel.metal",
        input_names=input_names,
        output_names=["output"],
        dtype=dtype,  # This will generate paged_attention_kernel_half etc.
    )

    grid = (num_heads * 32, batch_size, 1)
    thread_group = (32, 1, 1)

    outputs = kernel(
        inputs=inputs,
        grid=grid,
        threadgroup=thread_group,
        output_shapes=[(batch_size, num_heads, v_head_dim)],
        output_dtypes=[dtype],  # Output matches input dtype
        verbose=False,
    )

    out = outputs[0]
    return out[:, :, None, :]
