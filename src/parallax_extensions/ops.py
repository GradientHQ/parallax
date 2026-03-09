import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

import mlx.core as mx


def _build_import_error(original_error: Exception) -> ImportError:
    """Build a helpful error for missing/incompatible prebuilt extension binaries."""
    lib_dir = Path(__file__).resolve().parent / "lib"
    available_bins = sorted(p.name for p in lib_dir.glob("_ext*.so"))
    cache_tag = (
        sys.implementation.cache_tag or f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    )
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    if available_bins:
        available_info = ", ".join(available_bins)
    else:
        available_info = "(none)"

    msg = (
        "Failed to import parallax_extensions native kernels.\n"
        f"- Python: {py_ver}\n"
        f"- Expected binary: _ext.{cache_tag}-*.so (or _ext.abi3.so)\n"
        f"- Found in lib/: {available_info}\n"
        "- If you distribute prebuilt binaries, include one for this Python version.\n"
        "- Or rebuild locally with:\n"
        "  python src/parallax_extensions/setup.py build_ext -j8 --inplace\n"
        f"- Original error: {original_error}"
    )
    return ImportError(msg)


def load_extension_module() -> ModuleType:
    """Load the compiled extension module for the current Python runtime."""
    try:
        # Python's import machinery selects the matching ABI-tagged binary
        # (e.g. _ext.cpython-312-*.so) from parallax_extensions/lib.
        return importlib.import_module("parallax_extensions.lib._ext")
    except Exception as exc:  # pragma: no cover - exercised in env-dependent cases
        raise _build_import_error(exc) from exc


_ext = load_extension_module()
_ext_paged_attention_v1 = _ext.paged_attention_v1
_ext_reshape_and_cache = _ext.reshape_and_cache


def reshape_and_cache(
    key: mx.array,
    value: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    block_size: int,
    slot_mapping: Optional[mx.array] = None,
):
    """
    Wrapper for C++ reshape_and_cache kernel.
    Handles slot_mapping calculation for Decode phase if not provided.
    """

    # Decode Mode
    if slot_mapping is None:
        batch_size = key.shape[0]

        # (B, H, 1, D) -> (B, H, D)
        if key.ndim == 4:
            if key.shape[2] == 1:
                key = key.squeeze(2)
                value = value.squeeze(2)
            elif key.shape[1] == 1:  # (B, 1, H, D) case
                key = key.squeeze(1)
                value = value.squeeze(1)
        key = mx.contiguous(key)
        value = mx.contiguous(value)

        # Calculate Slot Mapping
        indices = context_lengths - 1
        block_indices_in_table = indices // block_size
        offsets = indices % block_size
        # Find Physical Block
        batch_indices = mx.arange(batch_size)
        physical_block_numbers = block_tables[batch_indices, block_indices_in_table]
        # Calculate Physical Slot Index
        slot_mapping = physical_block_numbers * block_size + offsets
        slot_mapping = slot_mapping.astype(mx.int64)

    # Prefill Mode
    else:
        # (B, T, H, D) -> (B*T, H, D)
        if key.ndim == 4:
            B, T, H, D = key.shape
            key = key.reshape(B * T, H, D)
            V_D = value.shape[-1]
            value = value.reshape(B * T, H, V_D)
        key = mx.contiguous(key)
        value = mx.contiguous(value)

        if slot_mapping.dtype != mx.int64:
            slot_mapping = slot_mapping.astype(mx.int64)

    op = _ext_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    mx.async_eval(op)
    return


def paged_attention_v1(
    queries: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    block_size: int,
    scale: float,
    num_kv_heads: int,
    v_head_dim: Optional[int] = None,
    # NOTE: The following parameters are not yet supported by this Kernel.
    top_k_indices: Optional[mx.array] = None,
    window_size: Optional[int] = None,
    sinks: Optional[mx.array] = None,
) -> mx.array:
    """
    Wrapper for paged_attention_v1 kernel in parallax_extensions
    """

    #  (B, H, 1, D) -> (B, H, D)
    if queries.ndim == 4:
        queries = queries.squeeze(2)

    if top_k_indices is not None:
        raise NotImplementedError(
            "DeepSeek-V3 TopK attention is not yet supported in the new C++ kernel."
        )

    if window_size is None:
        window_size = 0

    num_heads = queries.shape[1]

    if sinks is None:
        has_sink = 0
        sinks = mx.zeros((1,), dtype=mx.float32)  # dummy, kernel will ignore
    else:
        has_sink = 1
        if sinks.ndim != 1 or sinks.shape[0] != num_heads:
            raise ValueError("sinks must be shape (num_heads,)")
        if sinks.dtype != mx.float32:
            sinks = sinks.astype(mx.float32)

    max_seq_len = block_tables.shape[1] * block_size

    output = _ext_paged_attention_v1(
        queries,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        num_kv_heads,
        block_size,
        max_seq_len,
        scale,
        window_size,
        sinks,
        has_sink,
    )

    #  (B, H, D) -> (B, H, 1, D)
    return output[:, :, None, :]
