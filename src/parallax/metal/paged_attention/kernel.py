import os
from typing import Dict, List

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
    key: mx.array,  # (batch, num_kv_heads, 1, head_dim)
    value: mx.array,  # (batch, num_kv_heads, 1, head_dim)
    key_cache: mx.array,  # (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
    value_cache: mx.array,
    block_tables: mx.array,  # (batch, max_blocks)
    context_lengths: mx.array,  # (batch,)
    block_size: int,
    layer_idx: int,
):
    """
    Writes new keys and values into the Paged KV Cache using a custom Metal kernel.
    NOTE: This performs an in-place update on key_cache/value_cache buffers.
    """
    batch_size = key.shape[0]
    num_kv_heads = key.shape[1]
    head_dim = key.shape[3]
    num_layers = key_cache.shape[0]
    num_blocks = key_cache.shape[1]

    dtype = key.dtype
    if key_cache.dtype != dtype:
        raise ValueError(f"Key cache dtype {key_cache.dtype} does not match key dtype {dtype}")

    # 1. Prepare inputs
    indices = context_lengths - 1
    block_indices_in_table = indices // block_size
    offsets = indices % block_size

    batch_indices = mx.arange(batch_size)
    physical_block_numbers = block_tables[batch_indices, block_indices_in_table]

    slot_mapping = physical_block_numbers.astype(mx.int64) * block_size + offsets.astype(mx.int64)

    # 2. Prepare Constants
    key_stride = num_kv_heads * head_dim
    value_stride = num_kv_heads * head_dim

    def mk_int(val):
        return mx.array(val, dtype=mx.int32)

    c_key_stride = mk_int(key_stride)
    c_val_stride = mk_int(value_stride)
    c_num_kv = mk_int(num_kv_heads)
    c_head_dim = mk_int(head_dim)
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
        c_head_dim,
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
        "head_dim",
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

    grid = (num_kv_heads * head_dim, batch_size, 1)
    thread_group = (min(1024, num_kv_heads * head_dim), 1, 1)

    # Execute
    outputs = kernel(
        inputs=inputs,
        grid=grid,
        threadgroup=thread_group,
        output_shapes=[(1,)],
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

    head_dim = queries.shape[2]
    num_layers = key_cache.shape[0]
    num_total_blocks = key_cache.shape[1]
    max_blocks = block_tables.shape[1]

    # Prepare Constants
    def mk_int(val):
        return mx.array(val, dtype=mx.int32)

    c_num_heads = mk_int(num_heads)
    c_num_kv_heads = mk_int(num_kv_heads)
    c_head_dim = mk_int(head_dim)
    c_block_size = mk_int(block_size)
    c_max_blocks = mk_int(max_blocks)
    c_layer_idx = mk_int(layer_idx)
    c_num_layers = mk_int(num_layers)
    c_num_total_blocks = mk_int(num_total_blocks)
    c_scale = mx.array(scale, dtype=mx.float32)

    inputs = [
        queries,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        c_num_heads,
        c_num_kv_heads,
        c_head_dim,
        c_block_size,
        c_max_blocks,
        c_layer_idx,
        c_num_layers,
        c_num_total_blocks,
        c_scale,
    ]

    input_names = [
        "queries",
        "key_cache",
        "value_cache",
        "block_tables",
        "context_lengths",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "block_size",
        "max_blocks",
        "layer_idx",
        "num_layers",
        "num_total_blocks",
        "scale",
    ]

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
        output_shapes=[(batch_size, num_heads, head_dim)],
        output_dtypes=[dtype],  # Output matches input dtype
        verbose=False,
    )

    out = outputs[0]
    return out[:, :, None, :]
