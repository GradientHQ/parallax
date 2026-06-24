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

# MHA, GQA
_ext_paged_attention_v1 = _ext.paged_attention_v1
_ext_paged_attention_v2 = _ext.paged_attention_v2

# MLA
_ext_mla_paged_attention = _ext.mla_paged_attention

# DSA
_ext_dsa_paged_attention = _ext.dsa_paged_attention
_ext_dsa_reshape_and_cache = _ext.dsa_reshape_and_cache
_ext_dsa_indexer_scores_with_update = _ext.dsa_indexer_scores_with_update

# MSA
_ext_msa_paged_attention = _ext.msa_paged_attention
_ext_msa_token_indexer = _ext.msa_token_indexer
_ext_msa_token_indexer_with_update = _ext.msa_token_indexer_with_update

# Cache
_ext_reshape_and_cache = _ext.reshape_and_cache
_ext_store_indexer_cache = _ext.store_indexer_cache

_PAGED_ATTENTION_V1_MAX_LENGTH = 8192


def mla_paged_attention(
    q_latent: mx.array,
    q_pe: mx.array,
    latent_cache: mx.array,
    rope_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    block_size: int,
    scale: float,
) -> mx.array:
    """
    Dense MLA decode attention over compressed paged cache.

    Computes:
      softmax(scale * (q_latent @ latent_cache.T + q_pe @ rope_cache.T)) @ latent_cache
    over the full logical context for each sequence.
    """
    if q_latent.ndim == 4:
        if q_latent.shape[2] != 1:
            raise ValueError("mla_paged_attention only supports one query token.")
        q_latent = q_latent.squeeze(2)
    if q_pe.ndim == 4:
        if q_pe.shape[2] != 1:
            raise ValueError("mla_paged_attention only supports one query token.")
        q_pe = q_pe.squeeze(2)
    if q_latent.ndim != 3 or q_pe.ndim != 3:
        raise ValueError("q_latent and q_pe must be shaped (batch, heads, dim).")
    if q_latent.shape[:2] != q_pe.shape[:2]:
        raise ValueError("q_latent and q_pe batch/head dimensions must match.")
    if latent_cache.ndim != 5 or rope_cache.ndim != 5:
        raise ValueError("latent_cache and rope_cache must be paged cache tensors.")
    if latent_cache.shape[2] != 1 or rope_cache.shape[2] != 1:
        raise ValueError("mla_paged_attention expects one MLA cache head.")
    if latent_cache.shape[-1] != q_latent.shape[-1]:
        raise ValueError("latent cache dim must match q_latent dim.")
    if rope_cache.shape[-1] != q_pe.shape[-1]:
        raise ValueError("rope cache dim must match q_pe dim.")

    output = _ext_mla_paged_attention(
        mx.contiguous(q_latent),
        mx.contiguous(q_pe),
        latent_cache,
        rope_cache,
        mx.contiguous(block_tables.astype(mx.int32)),
        mx.contiguous(context_lengths.astype(mx.int32)),
        block_size,
        scale,
    )
    return output[:, :, None, :]


def store_indexer_cache(
    key: mx.array,
    key_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    block_size: int,
    slot_mapping: mx.array,
):
    """Store index keys into the paged index-key cache."""
    if key_cache.ndim != 5:
        raise ValueError(
            "key_cache must be shaped (1, num_blocks, index_key_heads, block_size, index_dim)."
        )
    if key_cache.shape[0] != 1:
        raise ValueError("store_indexer_cache expects a single-layer index cache.")
    if key_cache.shape[3] != block_size:
        raise ValueError(
            f"block_size={block_size} does not match index cache block size "
            f"{key_cache.shape[3]}."
        )

    if slot_mapping is None:
        raise ValueError("slot_mapping is required for index-cache update.")
    if slot_mapping.dtype != mx.int64:
        slot_mapping = slot_mapping.astype(mx.int64)
    if key.ndim == 4:
        if slot_mapping.shape[0] == key.shape[0] and key.shape[1] == 1:
            key = key.squeeze(1)
        elif slot_mapping.shape[0] == key.shape[0] and key.shape[2] == 1:
            key = key.squeeze(2)
        else:
            batch, target_len, heads, dim = key.shape
            key = key.reshape(batch * target_len, heads, dim)
    if key.ndim != 3:
        raise ValueError(
            "index key must be shaped (tokens, heads, dim), "
            "(batch, tokens, heads, dim), (batch, 1, heads, dim), or "
            "(batch, heads, 1, dim)."
        )

    if key.shape[1] != key_cache.shape[2]:
        raise ValueError(
            "store_indexer_cache requires input key heads to match cache heads; "
            "shared-key models should allocate index_key_heads=1."
        )
    if key.shape[2] != key_cache.shape[4]:
        raise ValueError("index key dim must match index cache dim.")
    if slot_mapping.shape[0] != key.shape[0]:
        raise ValueError("slot_mapping length must match number of index-key tokens.")

    op = _ext_store_indexer_cache(
        mx.contiguous(key),
        key_cache,
        mx.contiguous(slot_mapping),
    )
    mx.async_eval(op)


def dsa_paged_attention(
    q_latent: mx.array,
    q_pe: mx.array,
    latent_cache: mx.array,
    rope_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    topk_indices: mx.array,
    block_size: int,
    scale: float,
) -> mx.array:
    """
    Decode DSA attention over compressed MLA paged cache.

    The kernel computes:
      softmax(scale * (q_latent @ latent_cache.T + q_pe @ rope_cache.T)) @ latent_cache

    over the sparse logical token positions in ``topk_indices``. If a row in
    ``topk_indices`` starts with -1, the kernel attends densely over
    ``range(context_length)`` for that sequence.
    """
    if q_latent.ndim == 4:
        if q_latent.shape[2] != 1:
            raise ValueError("dsa_paged_attention only supports one query token.")
        q_latent = q_latent.squeeze(2)
    if q_pe.ndim == 4:
        if q_pe.shape[2] != 1:
            raise ValueError("dsa_paged_attention only supports one query token.")
        q_pe = q_pe.squeeze(2)
    if q_latent.ndim != 3 or q_pe.ndim != 3:
        raise ValueError("q_latent and q_pe must be shaped (batch, heads, dim).")
    if q_latent.shape[:2] != q_pe.shape[:2]:
        raise ValueError("q_latent and q_pe batch/head dimensions must match.")
    if latent_cache.ndim != 5 or rope_cache.ndim != 5:
        raise ValueError("latent_cache and rope_cache must be paged cache tensors.")
    if latent_cache.shape[2] != 1 or rope_cache.shape[2] != 1:
        raise ValueError("dsa_paged_attention expects one MLA cache head.")
    if latent_cache.shape[-1] != q_latent.shape[-1]:
        raise ValueError("latent cache dim must match q_latent dim.")
    if rope_cache.shape[-1] != q_pe.shape[-1]:
        raise ValueError("rope cache dim must match q_pe dim.")

    if topk_indices.ndim == 3:
        if topk_indices.shape[1] != 1:
            raise ValueError("topk_indices must have singleton query dimension for decode.")
        topk_indices = topk_indices.squeeze(1)
    if topk_indices.ndim != 2:
        raise ValueError("topk_indices must be shaped (batch, positions).")
    if topk_indices.shape[0] != q_latent.shape[0]:
        raise ValueError("topk_indices batch must match q_latent batch.")

    output = _ext_dsa_paged_attention(
        mx.contiguous(q_latent),
        mx.contiguous(q_pe),
        latent_cache,
        rope_cache,
        mx.contiguous(block_tables.astype(mx.int32)),
        mx.contiguous(context_lengths.astype(mx.int32)),
        mx.contiguous(topk_indices.astype(mx.int32)),
        block_size,
        topk_indices.shape[1],
        scale,
    )
    return output[:, :, None, :]


def dsa_indexer_scores_with_update(
    index_queries: mx.array,
    index_key_update: mx.array,
    index_key_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    index_weights: mx.array,
    max_context_len: int,
    slot_mapping: mx.array,
) -> mx.array:
    """Store the decode index key and compute GLM/DeepSeek DSA token scores."""
    if index_queries.ndim == 4:
        if index_queries.shape[2] != 1:
            raise ValueError("dsa_indexer_scores_with_update only supports one query token.")
        index_queries = index_queries.squeeze(2)
    if index_queries.ndim != 3:
        raise ValueError(
            "index_queries must be shaped (batch, index_heads, dim) or "
            "(batch, index_heads, 1, dim)."
        )

    if index_key_update.ndim == 4:
        if index_key_update.shape[2] == 1:
            index_key_update = index_key_update.squeeze(2)
        elif index_key_update.shape[1] == 1:
            index_key_update = index_key_update.squeeze(1)
        else:
            raise ValueError("index_key_update must have a singleton decode dimension.")
    if index_key_update.ndim != 3:
        raise ValueError(
            "index_key_update must be shaped (batch, index_key_heads, dim) or "
            "(batch, index_key_heads, 1, dim)."
        )
    if index_key_update.shape[0] != index_queries.shape[0]:
        raise ValueError("index_key_update batch must match index_queries.")
    if index_key_update.shape[-1] != index_queries.shape[-1]:
        raise ValueError("index_key_update dim must match index_queries dim.")

    if index_key_cache.ndim != 5:
        raise ValueError(
            "index_key_cache must be shaped "
            "(1, num_blocks, index_key_heads, block_size, index_dim)."
        )
    if index_key_cache.shape[-1] != index_queries.shape[-1]:
        raise ValueError("index_key_cache dim must match index_queries dim.")
    if index_key_cache.shape[2] != index_key_update.shape[1]:
        raise ValueError("index_key_update heads must match index_key_cache heads.")

    if index_weights.ndim == 3:
        if index_weights.shape[1] != 1:
            raise ValueError("index_weights must have a singleton decode dimension.")
        index_weights = index_weights.squeeze(1)
    if index_weights.ndim != 2:
        raise ValueError("index_weights must be shaped (batch, index_heads).")
    if index_weights.shape != index_queries.shape[:2]:
        raise ValueError("index_weights must match index query batch/head dimensions.")
    if max_context_len <= 0:
        raise ValueError("max_context_len must be positive.")
    if slot_mapping is None:
        raise ValueError("slot_mapping is required for decode index-cache update.")
    if slot_mapping.dtype != mx.int64:
        slot_mapping = slot_mapping.astype(mx.int64)
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != index_queries.shape[0]:
        raise ValueError("slot_mapping must be shaped (batch,).")

    return _ext_dsa_indexer_scores_with_update(
        mx.contiguous(index_queries),
        mx.contiguous(index_key_update),
        index_key_cache,
        mx.contiguous(block_tables.astype(mx.int32)),
        mx.contiguous(context_lengths.astype(mx.int32)),
        mx.contiguous(index_weights.astype(mx.float32)),
        mx.contiguous(slot_mapping),
        max_context_len,
    )


def dsa_token_indexer_with_update(
    index_queries: mx.array,
    index_key_update: mx.array,
    index_key_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    index_weights: mx.array,
    index_topk: int,
    slot_mapping: mx.array,
) -> mx.array:
    """
    Decode helper for GLM/DeepSeek DSA indexer.

    The extension updates the paged index-key cache and computes token scores.
    Top-k selection uses MLX argpartition, avoiding the previous Python loop
    while keeping MLX's optimized large-k implementation.
    """
    if index_topk <= 0:
        raise ValueError("index_topk must be positive.")

    block_size = index_key_cache.shape[3]
    max_context_len = block_tables.shape[1] * block_size
    if max_context_len <= index_topk:
        batch = index_queries.shape[0]
        return mx.full((batch, index_topk), -1, dtype=mx.int32)

    scores = dsa_indexer_scores_with_update(
        index_queries,
        index_key_update,
        index_key_cache,
        block_tables,
        context_lengths,
        index_weights,
        max_context_len,
        slot_mapping=slot_mapping,
    )
    topk = mx.argpartition(scores, kth=-index_topk, axis=-1)[:, -index_topk:].astype(mx.int32)
    dense_rows = context_lengths.astype(mx.int32) <= index_topk
    return mx.where(
        dense_rows[:, None],
        mx.full(topk.shape, -1, dtype=mx.int32),
        topk,
    )


def reshape_and_cache(
    key: mx.array,
    value: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    block_size: int,
    slot_mapping: mx.array,
):
    """
    Wrapper for C++ reshape_and_cache kernel.
    """

    dsa_cache_layout = key_cache.ndim == 5 and value_cache.ndim == 5

    if slot_mapping is None:
        raise ValueError("slot_mapping is required for KV-cache update.")
    if slot_mapping.dtype != mx.int64:
        slot_mapping = slot_mapping.astype(mx.int64)
    if key.ndim == 4:
        if slot_mapping.shape[0] == key.shape[0] and key.shape[2] == 1:
            key = key.squeeze(2)
            value = value.squeeze(2)
        elif slot_mapping.shape[0] == key.shape[0] and key.shape[1] == 1:
            key = key.squeeze(1)
            value = value.squeeze(1)
        else:
            B, T, H, D = key.shape
            key = key.reshape(B * T, H, D)
            V_D = value.shape[-1]
            value = value.reshape(B * T, H, V_D)
    key = mx.contiguous(key)
    value = mx.contiguous(value)

    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != key.shape[0]:
        raise ValueError("slot_mapping length must match number of KV tokens.")

    if dsa_cache_layout:
        op = _ext_dsa_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    else:
        op = _ext_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    mx.async_eval(op)
    return


def _prepare_paged_attention_inputs(
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
):
    """Normalize inputs shared by paged attention wrappers."""

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

    return queries, sinks, has_sink, window_size, max_seq_len


def paged_attention_v2(
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
    """Wrapper for partitioned paged_attention_v2 kernel in parallax_extensions."""

    queries, sinks, has_sink, window_size, max_seq_len = _prepare_paged_attention_inputs(
        queries,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        block_size,
        scale,
        num_kv_heads,
        v_head_dim,
        top_k_indices,
        window_size,
        sinks,
    )
    if has_sink and max_seq_len > 512:
        raise NotImplementedError(
            "paged_attention_v2 does not yet support attention sinks across multiple partitions"
        )

    output = _ext_paged_attention_v2(
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
    Wrapper for paged_attention_v1 kernel in parallax_extensions.

    Long decode contexts are dispatched to the partitioned v2 kernel to avoid
    v1's context-length-sized threadgroup logits buffer.
    """

    queries, sinks, has_sink, window_size, max_seq_len = _prepare_paged_attention_inputs(
        queries,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        block_size,
        scale,
        num_kv_heads,
        v_head_dim,
        top_k_indices,
        window_size,
        sinks,
    )

    effective_logits_len = max_seq_len
    if window_size > 0:
        effective_logits_len = min(max_seq_len, window_size + block_size)

    if has_sink == 0 and effective_logits_len > _PAGED_ATTENTION_V1_MAX_LENGTH:
        output = _ext_paged_attention_v2(
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
        return output[:, :, None, :]

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


def msa_paged_attention(
    queries: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    token_positions: mx.array,
    token_positions_valid: Optional[mx.array],
    block_size: int,
    scale: float,
    num_kv_heads: int,
) -> mx.array:
    """
    MSA paged attention in parallax_extensions.

    The sparse pattern is expressed directly as logical token positions. The
    native kernel maps those positions through block_tables into the packed KV
    cache and computes exact token-level attention over the selected tokens.
    """
    if queries.ndim == 4:
        if queries.shape[2] != 1:
            raise ValueError("msa_paged_attention only supports one query token.")
        queries = queries.squeeze(2)
    if queries.ndim != 3:
        raise ValueError("queries must be shaped (batch, heads, dim) or (batch, heads, 1, dim).")
    if key_cache.ndim != 5 or value_cache.ndim != 4:
        raise ValueError("msa_paged_attention requires packed paged KV cache tensors.")
    if value_cache.shape[2] != queries.shape[2]:
        raise ValueError("msa_paged_attention requires value head dim to match query head dim.")

    if token_positions.ndim == 3:
        if token_positions.shape[1] != 1:
            raise ValueError("token_positions must have singleton query dimension for decode.")
        token_positions = token_positions.squeeze(1)
    if token_positions.ndim != 2:
        raise ValueError("token_positions must be shaped (batch, positions).")

    if token_positions_valid is None:
        token_positions_valid = mx.ones(token_positions.shape, dtype=mx.int32)
    elif token_positions_valid.ndim == 3:
        if token_positions_valid.shape[1] != 1:
            raise ValueError(
                "token_positions_valid must have singleton query dimension for decode."
            )
        token_positions_valid = token_positions_valid.squeeze(1)
    if token_positions_valid.shape != token_positions.shape:
        raise ValueError("token_positions_valid must match token_positions shape.")

    queries = mx.contiguous(queries)
    block_tables = mx.contiguous(block_tables.astype(mx.int32))
    context_lengths = mx.contiguous(context_lengths.astype(mx.int32))
    token_positions = mx.contiguous(token_positions.astype(mx.int32))
    token_positions_valid = mx.contiguous(token_positions_valid.astype(mx.int32))
    max_num_positions = token_positions.shape[1]

    output = _ext_msa_paged_attention(
        queries,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        token_positions,
        token_positions_valid,
        num_kv_heads,
        block_size,
        max_num_positions,
        scale,
    )
    return output[:, :, None, :]


def msa_token_indexer(
    index_queries: mx.array,
    index_key_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    max_context_len: int,
    sparse_block_size: int,
    sparse_topk_blocks: int,
    sparse_init_blocks: int,
    sparse_local_blocks: int,
    scale: float,
) -> mx.array:
    """
    Select sparse index blocks and expand them to logical token positions.

    Invalid slots are returned as -1. The sparse attention kernel treats
    negative positions as masked, so callers can pass the returned positions
    with ``token_positions_valid=None``.
    """
    if index_queries.ndim == 4:
        if index_queries.shape[2] != 1:
            raise ValueError("msa_token_indexer only supports one query token.")
        index_queries = index_queries.squeeze(2)
    if index_queries.ndim != 3:
        raise ValueError(
            "index_queries must be shaped (batch, index_heads, dim) or "
            "(batch, index_heads, 1, dim)."
        )
    if index_key_cache.ndim != 5:
        raise ValueError(
            "index_key_cache must be shaped "
            "(1, num_blocks, index_key_heads, block_size, index_dim)."
        )
    if index_key_cache.shape[-1] != index_queries.shape[-1]:
        raise ValueError("index_key_cache dim must match index_queries dim.")
    if sparse_topk_blocks <= 0:
        raise ValueError("sparse_topk_blocks must be positive.")
    if sparse_block_size <= 0:
        raise ValueError("sparse_block_size must be positive.")
    if max_context_len <= 0:
        raise ValueError("max_context_len must be positive.")

    return _ext_msa_token_indexer(
        mx.contiguous(index_queries),
        index_key_cache,
        mx.contiguous(block_tables.astype(mx.int32)),
        mx.contiguous(context_lengths.astype(mx.int32)),
        max_context_len,
        sparse_block_size,
        sparse_topk_blocks,
        sparse_init_blocks,
        sparse_local_blocks,
        scale,
    )


def msa_token_indexer_with_update(
    index_queries: mx.array,
    index_key_update: mx.array,
    index_key_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    max_context_len: int,
    sparse_block_size: int,
    sparse_topk_blocks: int,
    sparse_init_blocks: int,
    sparse_local_blocks: int,
    scale: float,
    slot_mapping: mx.array,
) -> mx.array:
    """
    Decode helper that stores the current index key and returns sparse token positions.

    Invalid slots are returned as -1. The update and index selection are encoded
    in a single extension primitive to avoid a separate Python Metal-kernel
    dispatch on the decode path.
    """
    if index_queries.ndim == 4:
        if index_queries.shape[2] != 1:
            raise ValueError("msa_token_indexer_with_update only supports one query token.")
        index_queries = index_queries.squeeze(2)
    if index_queries.ndim != 3:
        raise ValueError(
            "index_queries must be shaped (batch, index_heads, dim) or "
            "(batch, index_heads, 1, dim)."
        )

    if index_key_update.ndim == 4:
        if index_key_update.shape[2] == 1:
            index_key_update = index_key_update.squeeze(2)
        elif index_key_update.shape[1] == 1:
            index_key_update = index_key_update.squeeze(1)
        else:
            raise ValueError("index_key_update must have a singleton decode dimension.")
    if index_key_update.ndim != 3:
        raise ValueError(
            "index_key_update must be shaped (batch, index_key_heads, dim) or "
            "(batch, index_key_heads, 1, dim)."
        )
    if index_key_update.shape[0] != index_queries.shape[0]:
        raise ValueError("index_key_update batch must match index_queries.")
    if index_key_update.shape[-1] != index_queries.shape[-1]:
        raise ValueError("index_key_update dim must match index_queries dim.")

    if index_key_cache.ndim != 5:
        raise ValueError(
            "index_key_cache must be shaped "
            "(1, num_blocks, index_key_heads, block_size, index_dim)."
        )
    if index_key_cache.shape[-1] != index_queries.shape[-1]:
        raise ValueError("index_key_cache dim must match index_queries dim.")
    if index_key_cache.shape[2] != index_key_update.shape[1]:
        raise ValueError("index_key_update heads must match index_key_cache heads.")
    if sparse_topk_blocks <= 0:
        raise ValueError("sparse_topk_blocks must be positive.")
    if sparse_block_size <= 0:
        raise ValueError("sparse_block_size must be positive.")
    if max_context_len <= 0:
        raise ValueError("max_context_len must be positive.")
    if slot_mapping is None:
        raise ValueError("slot_mapping is required for decode index-cache update.")
    if slot_mapping.dtype != mx.int64:
        slot_mapping = slot_mapping.astype(mx.int64)
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != index_queries.shape[0]:
        raise ValueError("slot_mapping must be shaped (batch,).")

    return _ext_msa_token_indexer_with_update(
        mx.contiguous(index_queries),
        mx.contiguous(index_key_update),
        index_key_cache,
        mx.contiguous(block_tables.astype(mx.int32)),
        mx.contiguous(context_lengths.astype(mx.int32)),
        mx.contiguous(slot_mapping),
        max_context_len,
        sparse_block_size,
        sparse_topk_blocks,
        sparse_init_blocks,
        sparse_local_blocks,
        scale,
    )
