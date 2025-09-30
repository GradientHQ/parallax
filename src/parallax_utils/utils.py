from typing import Optional

import torch

from parallax.server.server_info import HardwareInfo
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


def bytes_per_element(dtype) -> int:
    """Return element size in bytes for supported torch/MLX dtypes."""
    try:
        import mlx.core as mx  # type: ignore
    except Exception:
        mx = None

    if dtype is None:
        return 2
    if dtype in (
        getattr(torch, "float32", None),
        getattr(torch, "bfloat16", None),
        getattr(torch, "float16", None),
        getattr(torch, "half", None),
        getattr(torch, "int8", None),
    ):
        if dtype == torch.float32:
            return 4
        if dtype in (torch.bfloat16, torch.float16, torch.half):
            return 2
        if dtype == torch.int8:
            return 1
    if mx is not None and dtype in (
        getattr(mx, "float32", None),
        getattr(mx, "bfloat16", None),
        getattr(mx, "float16", None),
    ):
        if dtype == mx.float32:
            return 4
        return 2
    return 2


def compute_max_tokens_in_cache(
    *,
    device: str,
    kv_cache_memory_fraction: float,
    num_shard_layers: int,
    num_key_value_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    elem_bytes: int,
    available_cache_bytes: Optional[int] = None,
) -> int:
    """Estimate max tokens storable in KV cache given current free memory and fraction."""
    if available_cache_bytes is not None:
        available_cache_size = int(available_cache_bytes)
    elif device == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        available_cache_size = int(free_bytes * kv_cache_memory_fraction)
    else:
        try:
            import mlx.core as mx
        except Exception:
            mx = None
        hw = HardwareInfo.detect()
        used = mx.get_active_memory() if mx is not None else 0
        available_cache_size = int((hw.total_ram_gb * 1024**3 - used) * kv_cache_memory_fraction)
    per_token_cache_size = (
        num_shard_layers * num_key_value_heads * (head_dim_k + head_dim_v) * elem_bytes
    )
    return max(0, available_cache_size // per_token_cache_size)


def derive_max_batch_size(
    *,
    requested_max_batch_size: Optional[int],
    max_sequence_len: Optional[int],
    max_tokens_in_cache: Optional[int],
) -> int:
    """Derive final max_batch_size clamped by KV capacity if sequence length known."""
    max_batch_capacity: Optional[int] = None
    if max_sequence_len and max_tokens_in_cache:
        max_batch_capacity = max(1, max_tokens_in_cache // int(max_sequence_len))
    if requested_max_batch_size is None:
        if max_batch_capacity is None:
            logger.warning("Overriding max_batch_size to 16 due to no max_sequence_len provided")
            return 16
        return max_batch_capacity
    if max_batch_capacity is not None:
        return min(requested_max_batch_size, max_batch_capacity)
    return requested_max_batch_size


def compute_max_batch_size(
    *,
    requested_max_batch_size: Optional[int],
    max_sequence_len: Optional[int],
    device: Optional[str],
    kv_cache_memory_fraction: float,
    num_shard_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype=None,
    elem_bytes: Optional[int] = None,
    memory_gb: Optional[float] = None,
    head_dim_k: Optional[int] = None,
    head_dim_v: Optional[int] = None,
) -> int:
    """Compute final max_batch_size by chaining dtype->elem_bytes, KV capacity, and clamping.

    If memory_gb is provided, we compute available_cache_bytes from it; otherwise we use device heuristics.
    """
    eb = elem_bytes if elem_bytes is not None else bytes_per_element(dtype)
    available_cache_bytes = None
    if memory_gb is not None:
        available_cache_bytes = int(memory_gb * 1024**3 * kv_cache_memory_fraction)
    ## This is an Error due to kv may have different head_dim
    max_tokens = compute_max_tokens_in_cache(
        device=device or "",  # empty means non-cuda path
        kv_cache_memory_fraction=kv_cache_memory_fraction,
        num_shard_layers=num_shard_layers,
        num_key_value_heads=num_key_value_heads,
        head_dim_k=head_dim_k if head_dim_k is not None else head_dim,
        head_dim_v=head_dim_v if head_dim_v is not None else head_dim,
        elem_bytes=eb,
        available_cache_bytes=available_cache_bytes,
    )
    return derive_max_batch_size(
        requested_max_batch_size=requested_max_batch_size,
        max_sequence_len=max_sequence_len,
        max_tokens_in_cache=max_tokens,
    )
