from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.mlx


@pytest.fixture(scope="module")
def m3_deps():
    import mlx.core as mx
    import numpy as np

    from parallax.models.minimax_m3 import (
        MiniMaxAttention,
        ModelArgs,
        ParallaxMiniMaxM3Block,
    )
    from mlx_lm.models.base import scaled_dot_product_attention
    from parallax.server.cache.minimax_m3_cache import MiniMaxM3SparseCache
    from parallax.utils.utils import combine_padding_and_causal_masks, create_causal_mask
    from parallax_extensions.ops import sparse_token_indexer, sparse_token_indexer_with_update

    return SimpleNamespace(
        mx=mx,
        np=np,
        MiniMaxAttention=MiniMaxAttention,
        ModelArgs=ModelArgs,
        ParallaxMiniMaxM3Block=ParallaxMiniMaxM3Block,
        MiniMaxM3SparseCache=MiniMaxM3SparseCache,
        combine_padding_and_causal_masks=combine_padding_and_causal_masks,
        create_causal_mask=create_causal_mask,
        scaled_dot_product_attention=scaled_dot_product_attention,
        sparse_token_indexer=sparse_token_indexer,
        sparse_token_indexer_with_update=sparse_token_indexer_with_update,
    )


def _tiny_args(deps):
    return deps.ModelArgs(
        hidden_size=16,
        intermediate_size=8,
        dense_intermediate_size=32,
        shared_intermediate_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=1,
        max_position_embeddings=32,
        vocab_size=32,
        num_local_experts=2,
        num_experts_per_tok=1,
        moe_layer_freq=[1],
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_index_dim": 4,
            "sparse_num_index_heads": 2,
            "sparse_topk_blocks": 1,
            "sparse_block_size": 2,
            "sparse_init_block": 0,
            "sparse_local_block": 0,
            "sparse_attention_freq": [1],
        },
    )


def _cache(deps, args, num_blocks=2, block_size=8):
    return deps.MiniMaxM3SparseCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        head_dim_v=args.head_dim,
        dtype=deps.mx.float32,
        index_head_dim=args.sparse_attention_config["sparse_index_dim"],
        index_n_heads=1,
    )


def _sparse_decode_dense_reference(
    deps,
    attention,
    queries,
    idx_queries,
    cache,
    block_tables,
    context_lengths,
):
    mx = deps.mx
    B = queries.shape[0]
    max_len = int(mx.max(context_lengths))
    keys = mx.zeros(
        (B, attention.num_key_value_heads, max_len, attention.head_dim),
        dtype=queries.dtype,
    )
    values = mx.zeros(
        (B, attention.num_key_value_heads, max_len, cache.head_dim_v),
        dtype=queries.dtype,
    )
    idx_keys = mx.zeros(
        (B, cache.index_n_heads, max_len, attention.index_dim),
        dtype=queries.dtype,
    )

    for i in range(B):
        context_len = int(context_lengths[i])
        k_i, v_i = cache.read_kv(block_tables[i], context_len)
        idx_i = cache.read_index_k(block_tables[i], context_len)
        keys[i, :, :context_len, :] = k_i
        values[i, :, :context_len, :] = v_i
        idx_keys[i, :, :context_len, :] = idx_i

    q_positions = context_lengths - 1
    sparse_mask = attention._build_sparse_mask(idx_queries, idx_keys, q_positions)
    valid = mx.arange(max_len)[None, None, None, :] < context_lengths[:, None, None, None]
    output = deps.scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=attention.scale,
        mask=sparse_mask & valid,
    )
    return output.transpose(0, 2, 1, 3).reshape(B, 1, -1)


def test_sparse_mask_selects_topk_blocks_not_topk_tokens(m3_deps):
    mx = m3_deps.mx
    args = _tiny_args(m3_deps)
    attention = m3_deps.MiniMaxAttention(args, layer_idx=0)

    idx_queries = mx.ones((1, 2, 1, 4), dtype=mx.float32)
    idx_keys = mx.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0, 0.0],
                    [9.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ]
        ],
        dtype=mx.float32,
    )

    sparse_mask = attention._build_sparse_mask(
        idx_queries,
        idx_keys,
        q_positions=mx.array([5], dtype=mx.int32),
    )
    mx.eval(sparse_mask)
    allowed = m3_deps.np.array(sparse_mask[0, 0, 0]).tolist()

    assert allowed == [False, False, True, True, False, False]


def test_sparse_token_indexer_expands_selected_blocks_to_token_positions(m3_deps):
    mx = m3_deps.mx
    args = _tiny_args(m3_deps)
    attention = m3_deps.MiniMaxAttention(args, layer_idx=0)
    cache = _cache(m3_deps, args, num_blocks=2, block_size=4)
    block_tables = mx.array([[0, 1]], dtype=mx.int32)
    context_lengths = mx.array([6], dtype=mx.int32)

    index_cache = m3_deps.np.zeros((1, 2, 1, 4, 4), dtype=m3_deps.np.float32)
    index_cache[0, 0, 0, 2, 0] = 10.0
    index_cache[0, 0, 0, 3, 0] = 9.0
    index_cache[0, 1, 0, 0, 0] = 1.0
    index_cache[0, 1, 0, 1, 0] = 1.0
    cache.index_key_cache = mx.array(index_cache, dtype=mx.float32)

    idx_queries = mx.ones((1, attention.index_heads, 1, attention.index_dim), dtype=mx.float32)
    token_positions = m3_deps.sparse_token_indexer(
        idx_queries,
        cache.get_indexer_cache(),
        block_tables,
        context_lengths,
        max_context_len=6,
        sparse_block_size=attention.sparse_block_size,
        sparse_topk_blocks=attention.sparse_topk_blocks,
        sparse_init_blocks=attention.sparse_init_blocks,
        sparse_local_blocks=attention.sparse_local_blocks,
        scale=attention.scale,
    )
    mx.eval(token_positions)

    assert m3_deps.np.array(token_positions[0]).tolist() == [2, 3]


def test_sparse_token_indexer_with_update_stores_current_index_key(m3_deps):
    mx = m3_deps.mx
    args = _tiny_args(m3_deps)
    attention = m3_deps.MiniMaxAttention(args, layer_idx=0)
    cache = _cache(m3_deps, args, num_blocks=2, block_size=4)
    block_tables = mx.array([[0, 1]], dtype=mx.int32)
    context_lengths = mx.array([6], dtype=mx.int32)

    index_cache = m3_deps.np.zeros((1, 2, 1, 4, 4), dtype=m3_deps.np.float32)
    index_cache[0, 0, 0, 2, 0] = 10.0
    index_cache[0, 0, 0, 3, 0] = 9.0
    cache.index_key_cache = mx.array(index_cache, dtype=mx.float32)

    idx_queries = mx.ones((1, attention.index_heads, 1, attention.index_dim), dtype=mx.float32)
    idx_key_update = mx.array([[[12.0, 0.0, 0.0, 0.0]]], dtype=mx.float32)
    token_positions = m3_deps.sparse_token_indexer_with_update(
        idx_queries,
        idx_key_update,
        cache.get_indexer_cache(),
        block_tables,
        context_lengths,
        max_context_len=6,
        sparse_block_size=attention.sparse_block_size,
        sparse_topk_blocks=attention.sparse_topk_blocks,
        sparse_init_blocks=attention.sparse_init_blocks,
        sparse_local_blocks=attention.sparse_local_blocks,
        scale=attention.scale,
    )
    written = cache.read_index_k(block_tables[0], 6)
    mx.eval(token_positions, written)

    assert m3_deps.np.array(token_positions[0]).tolist() == [4, 5]
    assert m3_deps.np.array(written[0, 5]).tolist() == [12.0, 0.0, 0.0, 0.0]


def test_sparse_token_positions_marks_context_tail_padding_invalid(m3_deps):
    mx = m3_deps.mx
    args = _tiny_args(m3_deps)
    attention = m3_deps.MiniMaxAttention(args, layer_idx=0)
    cache = _cache(m3_deps, args, num_blocks=1, block_size=4)
    block_tables = mx.array([[0]], dtype=mx.int32)
    context_lengths = mx.array([1], dtype=mx.int32)

    cache.index_key_cache = mx.ones(cache.index_key_cache.shape, dtype=mx.float32)
    idx_queries = mx.ones((1, attention.index_heads, 1, attention.index_dim), dtype=mx.float32)

    token_positions = m3_deps.sparse_token_indexer(
        idx_queries,
        cache.get_indexer_cache(),
        block_tables,
        context_lengths,
        max_context_len=1,
        sparse_block_size=attention.sparse_block_size,
        sparse_topk_blocks=attention.sparse_topk_blocks,
        sparse_init_blocks=attention.sparse_init_blocks,
        sparse_local_blocks=attention.sparse_local_blocks,
        scale=attention.scale,
    )
    mx.eval(token_positions)

    assert m3_deps.np.array(token_positions[0]).tolist() == [0, -1]


def test_sparse_mask_uses_real_key_positions_for_prefix_blocks(m3_deps):
    mx = m3_deps.mx
    args = _tiny_args(m3_deps)
    attention = m3_deps.MiniMaxAttention(args, layer_idx=0)

    idx_queries = mx.ones((1, 2, 1, 4), dtype=mx.float32)
    idx_keys = mx.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0, 0.0],
                    [9.0, 0.0, 0.0, 0.0],
                ]
            ]
        ],
        dtype=mx.float32,
    )

    sparse_mask = attention._build_sparse_mask(
        idx_queries,
        idx_keys,
        q_positions=mx.array([[5]], dtype=mx.int32),
        key_positions=mx.array([[0, 1, 4, 5]], dtype=mx.int32),
    )
    mx.eval(sparse_mask)
    allowed = m3_deps.np.array(sparse_mask[0, 0, 0]).tolist()

    assert allowed == [False, False, True, True]


def test_sparse_decode_kernel_matches_dense_reference(m3_deps):
    mx = m3_deps.mx
    args = _tiny_args(m3_deps)
    attention = m3_deps.MiniMaxAttention(args, layer_idx=0)
    cache = _cache(m3_deps, args, num_blocks=2, block_size=4)
    block_tables = mx.array([[0, 1]], dtype=mx.int32)

    x = mx.random.normal((1, 6, args.hidden_size)).astype(mx.float32)
    prefill = attention(
        x,
        mask=m3_deps.create_causal_mask(6, 6, mx.float32),
        cache=cache,
        block_tables=block_tables,
        context_lengths=mx.array([6], dtype=mx.int32),
        slot_mapping=mx.array([0, 1, 2, 3, 4, 5], dtype=mx.int64),
    )
    mx.eval(prefill, cache.get_cache()[0], cache.get_cache()[1], cache.get_indexer_cache())

    q = mx.random.normal(
        (1, attention.num_attention_heads, 1, attention.head_dim),
        dtype=mx.float32,
    )
    idx_q = mx.random.normal((1, attention.index_heads, 1, attention.index_dim), dtype=mx.float32)
    idx_k = cache.read_index_k(block_tables[0], 6)[None, :, -1:, :]
    context_lengths = mx.array([6], dtype=mx.int32)

    fast = attention._sparse_decode_from_cache(
        q,
        idx_q,
        idx_k,
        cache,
        block_tables,
        context_lengths,
    )
    dense = _sparse_decode_dense_reference(
        m3_deps,
        attention,
        q,
        idx_q,
        cache,
        block_tables,
        context_lengths,
    )
    mx.eval(fast, dense)

    assert fast.shape == dense.shape == (1, 1, args.hidden_size)
    assert mx.allclose(fast, dense, atol=1e-4).item()


def test_attention_prefill_and_decode_write_kv_and_index_cache(m3_deps):
    mx = m3_deps.mx
    args = _tiny_args(m3_deps)
    attention = m3_deps.MiniMaxAttention(args, layer_idx=0)
    cache = _cache(m3_deps, args)

    x = mx.ones((1, 5, args.hidden_size), dtype=mx.float32)
    out = attention(
        x,
        mask=m3_deps.create_causal_mask(5, 5, mx.float32),
        cache=cache,
        block_tables=mx.array([[0]], dtype=mx.int32),
        context_lengths=mx.array([5], dtype=mx.int32),
        slot_mapping=mx.array([0, 1, 2, 3, 4], dtype=mx.int64),
    )
    mx.eval(out, cache.get_cache()[0], cache.get_indexer_cache())

    assert out.shape == (1, 5, args.hidden_size)
    assert float(mx.sum(mx.abs(cache.get_cache()[0]))) > 0
    assert float(mx.sum(mx.abs(cache.get_indexer_cache()))) > 0

    decode = attention(
        mx.ones((1, 1, args.hidden_size), dtype=mx.float32),
        cache=cache,
        block_tables=mx.array([[0]], dtype=mx.int32),
        context_lengths=mx.array([6], dtype=mx.int32),
    )
    mx.eval(decode)

    assert decode.shape == (1, 1, args.hidden_size)
    assert cache.read_index_k(mx.array([0], dtype=mx.int32), 6).shape == (1, 6, 4)


def test_attention_prefill_reads_prefix_kv_and_index_cache_for_chunked_path(m3_deps):
    mx = m3_deps.mx
    args = _tiny_args(m3_deps)
    attention = m3_deps.MiniMaxAttention(args, layer_idx=0)
    cache = _cache(m3_deps, args)
    block_tables = mx.array([[0]], dtype=mx.int32)

    prefix = attention(
        mx.ones((1, 4, args.hidden_size), dtype=mx.float32),
        mask=m3_deps.create_causal_mask(4, 4, mx.float32),
        cache=cache,
        block_tables=block_tables,
        context_lengths=mx.array([4], dtype=mx.int32),
        slot_mapping=mx.array([0, 1, 2, 3], dtype=mx.int64),
    )
    mx.eval(prefix, cache.get_indexer_cache())

    chunk = attention(
        mx.full((1, 2, args.hidden_size), 2.0, dtype=mx.float32),
        mask=m3_deps.create_causal_mask(2, 2, mx.float32),
        cache=cache,
        block_tables=block_tables,
        context_lengths=mx.array([6], dtype=mx.int32),
        slot_mapping=mx.array([4, 5], dtype=mx.int64),
        prefix_lens=mx.array([4], dtype=mx.int32),
    )
    mx.eval(chunk, cache.get_cache()[0], cache.get_indexer_cache())

    assert prefix.shape == (1, 4, args.hidden_size)
    assert chunk.shape == (1, 2, args.hidden_size)
    assert cache.read_kv(mx.array([0], dtype=mx.int32), 6)[0].shape == (2, 6, 4)
    assert cache.read_index_k(mx.array([0], dtype=mx.int32), 6).shape == (1, 6, 4)


def test_attention_prefix_prefill_handles_mixed_prefix_lengths(m3_deps):
    mx = m3_deps.mx
    args = _tiny_args(m3_deps)
    attention = m3_deps.MiniMaxAttention(args, layer_idx=0)
    cache = _cache(m3_deps, args, num_blocks=2)
    block_tables = mx.array([[0], [1]], dtype=mx.int32)

    padding = mx.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=mx.float32)[:, None, None, :]
    prefix_mask = m3_deps.combine_padding_and_causal_masks(
        padding,
        m3_deps.create_causal_mask(4, 4, mx.float32),
        mx.float32,
    )
    prefix = attention(
        mx.ones((2, 4, args.hidden_size), dtype=mx.float32),
        mask=prefix_mask,
        cache=cache,
        block_tables=block_tables,
        context_lengths=mx.array([2, 4], dtype=mx.int32),
        slot_mapping=mx.array([0, 1, -1, -1, 8, 9, 10, 11], dtype=mx.int64),
    )
    mx.eval(prefix, cache.get_indexer_cache())

    chunk = attention(
        mx.full((2, 2, args.hidden_size), 2.0, dtype=mx.float32),
        mask=m3_deps.create_causal_mask(2, 2, mx.float32),
        cache=cache,
        block_tables=block_tables,
        context_lengths=mx.array([4, 6], dtype=mx.int32),
        slot_mapping=mx.array([2, 3, 12, 13], dtype=mx.int64),
        prefix_lens=mx.array([2, 4], dtype=mx.int32),
    )
    mx.eval(chunk)

    assert chunk.shape == (2, 2, args.hidden_size)
    assert cache.read_index_k(mx.array([0], dtype=mx.int32), 4).shape == (1, 4, 4)
    assert cache.read_index_k(mx.array([1], dtype=mx.int32), 6).shape == (1, 6, 4)


def test_block_forward_runs_attention_and_moe(m3_deps):
    mx = m3_deps.mx
    args = _tiny_args(m3_deps)
    block = m3_deps.ParallaxMiniMaxM3Block(args, layer_idx=0, local_layer_idx=0)
    cache = _cache(m3_deps, args)

    out = block(
        mx.ones((1, 5, args.hidden_size), dtype=mx.float32),
        mask=m3_deps.create_causal_mask(5, 5, mx.float32),
        cache=[cache],
        block_tables=mx.array([[0]], dtype=mx.int32),
        context_lengths=mx.array([5], dtype=mx.int32),
        slot_mapping=mx.array([0, 1, 2, 3, 4], dtype=mx.int64),
    )
    mx.eval(out)

    assert out.shape == (1, 5, args.hidden_size)
