import os
from typing import Dict, List, Optional

import mlx.core as mx

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
        return "bfloat16_t"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


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
        source = source.replace("{{T}}", type_str)

        header = """
#include <metal_stdlib>
using namespace metal;
"""
        _KERNELS[kernel_key] = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
            header=header,
        )
    return _KERNELS[kernel_key]


def store_indexer_cache(
    key: mx.array,
    key_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    block_size: int,
    slot_mapping: Optional[mx.array] = None,
):
    dtype = key.dtype
    # key: (batch, target_len, num_heads, head_dim) or flattened

    if slot_mapping is None:
        # Decode Mode
        batch_size = key.shape[0]
        if key.ndim == 4:
            # (batch, 1, num_kv_heads, head_dim) -> (batch, num_kv_heads, head_dim)
            if key.shape[1] == 1:
                key = key.squeeze(1)
            elif key.shape[2] == 1:
                # Fallback for old layout (batch, num_kv_heads, 1, head_dim)
                key = key.squeeze(2)

        num_heads = key.shape[1]
        head_dim = key.shape[2]

        # Compute slot_mapping internally
        indices = context_lengths - 1
        block_indices_in_table = indices // block_size
        offsets = indices % block_size
        batch_indices = mx.arange(batch_size)
        physical_block_numbers = block_tables[batch_indices, block_indices_in_table]
        slot_mapping = physical_block_numbers.astype(mx.int32) * block_size + offsets.astype(
            mx.int32
        )

        num_tokens = batch_size
    else:
        # Prefill Mode
        if key.ndim == 4:
            B, T, H, D = key.shape
            key = key.reshape(B * T, H, D)

        num_tokens = key.shape[0]
        num_heads = key.shape[1]
        head_dim = key.shape[2]

    num_layers = key_cache.shape[0]
    num_blocks = key_cache.shape[1]

    key_stride = num_heads * head_dim

    def mk_int(val):
        return mx.array(val, dtype=mx.int32)

    inputs = [
        key,
        key_cache,
        slot_mapping,
        mk_int(key_stride),
        mk_int(num_heads),
        mk_int(head_dim),
        mk_int(block_size),
        mk_int(num_layers),
        mk_int(num_blocks),
    ]

    input_names = [
        "key",
        "key_cache",
        "slot_mapping",
        "key_stride",
        "num_heads",
        "head_dim",
        "block_size",
        "num_layers",
        "num_blocks",
    ]

    kernel = _get_kernel(
        name="store_key_kernel",
        filename="store_key.metal",
        input_names=input_names,
        output_names=["dummy_out"],
        dtype=dtype,
    )

    grid = (num_heads * head_dim, num_tokens, 1)
    thread_group = (min(1024, num_heads * head_dim), 1, 1)

    outputs = kernel(
        inputs=inputs,
        grid=grid,
        threadgroup=thread_group,
        output_shapes=[(num_tokens, num_heads * head_dim)],  # Dummy output
        output_dtypes=[mx.float32],
        verbose=False,
    )
    mx.eval(outputs)


def q_dot_k(
    q: mx.array,  # (num_heads, head_dim)
    key_cache: mx.array,  # (L, B, H, BS, D)
    block_table: mx.array,  # (max_blocks)
    context_length: mx.array,  # scalar
    block_size: int,
) -> mx.array:

    if q.ndim > 2:
        q = q.squeeze()  # Ensure (H, D)

    num_heads = q.shape[0]
    head_dim = q.shape[1]

    num_layers = key_cache.shape[0]
    num_total_blocks = key_cache.shape[1]
    max_blocks = block_table.shape[0]

    ctx_len = int(context_length.item())

    def mk_int(val):
        return mx.array(val, dtype=mx.int32)

    inputs = [
        q,
        key_cache,
        block_table,
        mk_int(ctx_len),
        mk_int(block_size),
        mk_int(num_heads),
        mk_int(head_dim),
        mk_int(num_layers),
        mk_int(num_total_blocks),
        mk_int(max_blocks),
    ]

    input_names = [
        "q",
        "key_cache",
        "block_table",
        "context_len",
        "block_size",
        "num_heads",
        "head_dim",
        "num_layers",
        "num_total_blocks",
        "max_blocks",
    ]

    kernel = _get_kernel(
        name="q_dot_k_kernel",
        filename="q_dot_k.metal",
        input_names=input_names,
        output_names=["output"],
        dtype=q.dtype,
    )

    # Grid: (block_size, num_heads, 1)
    grid = (block_size, num_heads, 1)
    thread_group = (min(1024, block_size), 1, 1)

    outputs = kernel(
        inputs=inputs,
        grid=grid,
        threadgroup=thread_group,
        output_shapes=[(num_heads, ctx_len)],
        output_dtypes=[mx.float32],  # Score is float32
        verbose=False,
    )

    return outputs[0]
