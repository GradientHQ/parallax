import sys

import mlx.core as mx
import numpy as np
import pytest

from parallax.server.cache.dsa_cache import DeepSeekSparseCache
from parallax_extensions.ops import dsa_token_indexer_with_update, store_indexer_cache

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="MLX tests require macOS")


def _make_index_cache(initial_keys, *, block_size=4):
    seq_len = initial_keys.shape[0]
    cache = DeepSeekSparseCache(
        num_blocks=2,
        block_size=block_size,
        num_kv_heads=1,
        head_dim=initial_keys.shape[-1],
        head_dim_v=initial_keys.shape[-1],
        dtype=mx.float32,
        index_head_dim=initial_keys.shape[-1],
        index_n_heads=1,
        kv_lora_rank=initial_keys.shape[-1],
        qk_rope_head_dim=2,
        index_key_heads=1,
    )
    block_tables = mx.array([[0, 1]], dtype=mx.int32)
    store_indexer_cache(
        initial_keys[None, :, None, :],
        cache.get_indexer_cache(),
        block_tables,
        mx.array([seq_len], dtype=mx.int32),
        block_size,
        slot_mapping=mx.arange(seq_len, dtype=mx.int64),
    )
    mx.eval(cache.get_indexer_cache())
    return cache, block_tables


def test_store_indexer_cache_rejects_implicit_head_repeat():
    cache = DeepSeekSparseCache(
        num_blocks=1,
        block_size=4,
        num_kv_heads=1,
        head_dim=4,
        head_dim_v=4,
        dtype=mx.float32,
        index_head_dim=4,
        index_n_heads=2,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        index_key_heads=2,
    )
    with pytest.raises(ValueError, match="input key heads to match cache heads"):
        store_indexer_cache(
            mx.ones((1, 1, 1, 4), dtype=mx.float32),
            cache.get_indexer_cache(),
            mx.array([[0]], dtype=mx.int32),
            mx.array([1], dtype=mx.int32),
            block_size=4,
            slot_mapping=mx.array([0], dtype=mx.int64),
        )


def _reference_topk(index_query, keys, weights, index_topk):
    dots = index_query @ keys.swapaxes(0, 1)
    scores = mx.sum(mx.maximum(dots, 0) * weights[:, None], axis=0)
    return mx.argpartition(scores, kth=-index_topk, axis=-1)[-index_topk:].astype(mx.int32)


def test_dsa_token_indexer_with_update_matches_reference_and_updates_cache():
    initial_keys = mx.array(
        [
            [0.2, 0.1, -0.1, 0.4],
            [0.5, -0.2, 0.3, 0.1],
            [-0.4, 0.6, 0.2, -0.3],
            [0.7, 0.2, -0.5, 0.2],
            [0.1, -0.7, 0.5, 0.3],
        ],
        dtype=mx.float32,
    )
    cache, block_tables = _make_index_cache(initial_keys)

    index_query = mx.array(
        [[[0.3, -0.1, 0.4, 0.2], [-0.5, 0.6, 0.1, -0.2]]],
        dtype=mx.float32,
    )
    key_update = mx.array([[[0.9, 0.2, -0.1, 0.5]]], dtype=mx.float32)
    weights = mx.array([[0.8, -0.25]], dtype=mx.float32)
    context_lengths = mx.array([6], dtype=mx.int32)
    index_topk = 3

    topk = dsa_token_indexer_with_update(
        index_query,
        key_update,
        cache.get_indexer_cache(),
        block_tables,
        context_lengths,
        weights,
        index_topk,
        slot_mapping=mx.array([5], dtype=mx.int64),
    )
    keys = mx.concatenate([initial_keys, key_update[0]], axis=0)
    ref_topk = _reference_topk(index_query[0], keys, weights[0], index_topk)
    read_back = cache.read_index_k(block_tables[0], 6)
    mx.eval(topk, ref_topk, read_back)

    assert sorted(np.array(topk[0]).tolist()) == sorted(np.array(ref_topk).tolist())
    assert mx.allclose(read_back[0, -1], key_update[0, 0])


def test_dsa_token_indexer_with_update_returns_dense_row_for_short_context():
    initial_keys = mx.array(
        [[0.2, 0.1, -0.1, 0.4], [0.5, -0.2, 0.3, 0.1]],
        dtype=mx.float32,
    )
    cache, block_tables = _make_index_cache(initial_keys)

    topk = dsa_token_indexer_with_update(
        mx.ones((1, 2, 4), dtype=mx.float32),
        mx.array([[[0.7, -0.1, 0.2, 0.4]]], dtype=mx.float32),
        cache.get_indexer_cache(),
        block_tables,
        mx.array([3], dtype=mx.int32),
        mx.ones((1, 2), dtype=mx.float32),
        4,
        slot_mapping=mx.array([2], dtype=mx.int64),
    )
    read_back = cache.read_index_k(block_tables[0], 3)
    mx.eval(topk, read_back)

    assert bool(mx.all(topk == -1).item())
    assert mx.allclose(read_back[0, -1], mx.array([0.7, -0.1, 0.2, 0.4]))


def test_dsa_token_indexer_with_update_glm_index_dimensions_smoke():
    heads = 32
    dim = 128
    seq_len = 5
    initial_keys = (mx.arange(seq_len * dim, dtype=mx.float32) / 1000).reshape(seq_len, dim)
    cache, block_tables = _make_index_cache(initial_keys)
    index_query = (mx.arange(heads * dim, dtype=mx.float32) / 2000).reshape(1, heads, dim)
    key_update = (mx.arange(dim, dtype=mx.float32) / 3000).reshape(1, 1, dim)

    topk = dsa_token_indexer_with_update(
        index_query,
        key_update,
        cache.get_indexer_cache(),
        block_tables,
        mx.array([seq_len + 1], dtype=mx.int32),
        mx.ones((1, heads), dtype=mx.float32),
        2,
        slot_mapping=mx.array([seq_len], dtype=mx.int64),
    )
    mx.eval(topk)

    assert topk.shape == (1, 2)
    assert bool(mx.all((topk >= 0) & (topk < seq_len + 1)).item())
