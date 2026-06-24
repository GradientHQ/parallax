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


def paged_attention(
    queries: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    block_size: int,
    scale: float,
    num_kv_heads: int,
    v_head_dim: Optional[int] = None,
    top_k_indices: Optional[mx.array] = None,
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
    c_num_layers = mk_int(num_layers)
    c_num_total_blocks = mk_int(num_total_blocks)
    c_scale = mx.array(scale, dtype=queries.dtype)

    if top_k_indices is not None:
        index_topk = top_k_indices.shape[1]
        c_index_topk = mk_int(index_topk)

        inputs = [
            queries,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            top_k_indices,
            c_num_heads,
            c_num_kv_heads,
            c_k_head_dim,
            c_v_head_dim,
            c_block_size,
            c_max_blocks,
            c_num_layers,
            c_num_total_blocks,
            c_scale,
            c_index_topk,
        ]

        input_names = [
            "queries",
            "key_cache",
            "value_cache",
            "block_tables",
            "context_lengths",
            "top_k_indices",
            "num_heads",
            "num_kv_heads",
            "k_head_dim",
            "v_head_dim",
            "block_size",
            "max_blocks",
            "num_layers",
            "num_total_blocks",
            "scale",
            "index_topk",
        ]
        kernel_name = "paged_attention_deepseek_v32_kernel"
        filename = "paged_attention_deepseek_v32.metal"
    elif window_size is not None and sinks is not None:
        c_window_size = mk_int(window_size)
        c_sinks = sinks
        inputs = [
            queries,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            c_sinks,
            c_num_heads,
            c_num_kv_heads,
            c_k_head_dim,
            c_v_head_dim,
            c_block_size,
            c_max_blocks,
            c_num_layers,
            c_num_total_blocks,
            c_scale,
            c_window_size,
        ]

        input_names = [
            "queries",
            "key_cache",
            "value_cache",
            "block_tables",
            "context_lengths",
            "sinks",
            "num_heads",
            "num_kv_heads",
            "k_head_dim",
            "v_head_dim",
            "block_size",
            "max_blocks",
            "num_layers",
            "num_total_blocks",
            "scale",
            "window_size",
        ]
        kernel_name = "paged_attention_gpt_oss_kernel"
        filename = "paged_attention_gpt_oss.metal"
    else:
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
            "k_head_dim",
            "v_head_dim",
            "block_size",
            "max_blocks",
            "num_layers",
            "num_total_blocks",
            "scale",
        ]
        kernel_name = "paged_attention_kernel"
        filename = "paged_attention.metal"

    kernel = _get_kernel(
        name=kernel_name,
        filename=filename,
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
