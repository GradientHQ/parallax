import sys

import mlx.core as mx
import pytest

from parallax.server.cache.dsa_cache import DeepSeekSparseCache
from parallax_extensions.ops import dsa_paged_attention, mla_paged_attention, reshape_and_cache

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="MLX tests require macOS")


def _softmax(x):
    x = x - mx.max(x, axis=-1, keepdims=True)
    ex = mx.exp(x)
    return ex / mx.sum(ex, axis=-1, keepdims=True)


def _make_cache(latent_seq, rope_seq, *, block_size=4):
    seq_len, latent_dim = latent_seq.shape
    rope_dim = rope_seq.shape[-1]
    cache = DeepSeekSparseCache(
        num_blocks=2,
        block_size=block_size,
        num_kv_heads=1,
        head_dim=latent_dim,
        head_dim_v=latent_dim,
        dtype=mx.float32,
        index_head_dim=rope_dim,
        index_n_heads=1,
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        index_key_heads=1,
    )
    block_tables = mx.array([[0, 1]], dtype=mx.int32)
    slot_mapping = mx.arange(seq_len, dtype=mx.int64)
    reshape_and_cache(
        latent_seq[None, :, None, :],
        rope_seq[None, :, None, :],
        cache.latent_cache,
        cache.rope_cache,
        block_tables,
        mx.array([seq_len], dtype=mx.int32),
        block_size,
        slot_mapping=slot_mapping,
    )
    mx.eval(cache.latent_cache, cache.rope_cache)
    return cache, block_tables


def _reference(q_latent, q_pe, latent_seq, rope_seq, positions, scale):
    latent_selected = latent_seq[positions]
    rope_selected = rope_seq[positions]
    scores = scale * (q_latent[0] @ latent_selected.T + q_pe[0] @ rope_selected.T)
    weights = _softmax(scores)
    return weights @ latent_selected


def test_dsa_paged_attention_sparse_positions_match_reference():
    latent_seq = mx.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5, 1.6],
            [1.7, 1.8, 1.9, 2.0],
            [2.1, 2.2, 2.3, 2.4],
        ],
        dtype=mx.float32,
    )
    rope_seq = mx.array(
        [
            [0.2, -0.1],
            [0.1, 0.0],
            [-0.2, 0.3],
            [0.4, 0.2],
            [-0.3, 0.5],
            [0.6, -0.4],
        ],
        dtype=mx.float32,
    )
    cache, block_tables = _make_cache(latent_seq, rope_seq)

    q_latent = mx.array(
        [[[0.3, -0.1, 0.2, 0.5], [-0.4, 0.2, 0.1, 0.3]]],
        dtype=mx.float32,
    )
    q_pe = mx.array([[[0.2, 0.7], [-0.5, 0.4]]], dtype=mx.float32)
    topk = mx.array([[5, 1, 3, 0]], dtype=mx.int32)
    scale = 0.7

    out = dsa_paged_attention(
        q_latent,
        q_pe,
        cache.latent_cache,
        cache.rope_cache,
        block_tables,
        mx.array([6], dtype=mx.int32),
        topk,
        cache.block_size,
        scale,
    ).squeeze(2)
    ref = _reference(q_latent, q_pe, latent_seq, rope_seq, topk[0], scale)
    mx.eval(out, ref)

    assert mx.allclose(out[0], ref, rtol=1e-4, atol=1e-4)


def test_dsa_paged_attention_dense_minus_one_row_match_reference():
    latent_seq = mx.array(
        [
            [0.1, -0.2, 0.3, -0.4],
            [0.5, -0.6, 0.7, -0.8],
            [0.9, -1.0, 1.1, -1.2],
        ],
        dtype=mx.float32,
    )
    rope_seq = mx.array([[0.2, 0.1], [-0.3, 0.4], [0.5, -0.6]], dtype=mx.float32)
    cache, block_tables = _make_cache(latent_seq, rope_seq)

    q_latent = mx.array([[[0.2, 0.1, -0.5, 0.4]]], dtype=mx.float32)
    q_pe = mx.array([[[0.3, -0.2]]], dtype=mx.float32)
    topk = mx.array([[-1, -1, -1, -1]], dtype=mx.int32)
    scale = 0.5

    out = dsa_paged_attention(
        q_latent,
        q_pe,
        cache.latent_cache,
        cache.rope_cache,
        block_tables,
        mx.array([3], dtype=mx.int32),
        topk,
        cache.block_size,
        scale,
    ).squeeze(2)
    ref = _reference(
        q_latent,
        q_pe,
        latent_seq,
        rope_seq,
        mx.array([0, 1, 2], dtype=mx.int32),
        scale,
    )
    mx.eval(out, ref)

    assert mx.allclose(out[0], ref, rtol=1e-4, atol=1e-4)


def test_mla_paged_attention_dense_matches_reference():
    latent_seq = mx.array(
        [
            [0.1, -0.2, 0.3, -0.4],
            [0.5, -0.6, 0.7, -0.8],
            [0.9, -1.0, 1.1, -1.2],
            [1.3, -1.4, 1.5, -1.6],
            [1.7, -1.8, 1.9, -2.0],
            [2.1, -2.2, 2.3, -2.4],
        ],
        dtype=mx.float32,
    )
    rope_seq = mx.array(
        [
            [0.2, 0.1],
            [-0.3, 0.4],
            [0.5, -0.6],
            [0.7, 0.8],
            [-0.9, 1.0],
            [1.1, -1.2],
        ],
        dtype=mx.float32,
    )
    block_size = 4
    cache = DeepSeekSparseCache(
        num_blocks=2,
        block_size=block_size,
        num_kv_heads=1,
        head_dim=latent_seq.shape[-1],
        head_dim_v=latent_seq.shape[-1],
        dtype=mx.float32,
        kv_lora_rank=latent_seq.shape[-1],
        qk_rope_head_dim=rope_seq.shape[-1],
    )
    block_tables = mx.array([[0, 1]], dtype=mx.int32)
    context_lengths = mx.array([5], dtype=mx.int32)
    reshape_and_cache(
        latent_seq[None, :, None, :],
        rope_seq[None, :, None, :],
        cache.latent_cache,
        cache.rope_cache,
        block_tables,
        mx.array([latent_seq.shape[0]], dtype=mx.int32),
        block_size,
        slot_mapping=mx.arange(latent_seq.shape[0], dtype=mx.int64),
    )
    mx.eval(cache.latent_cache, cache.rope_cache)

    q_latent = mx.array(
        [[[0.2, 0.1, -0.5, 0.4], [-0.3, 0.5, 0.2, -0.1]]],
        dtype=mx.float32,
    )
    q_pe = mx.array([[[0.3, -0.2], [0.6, 0.1]]], dtype=mx.float32)
    scale = 0.5

    out = mla_paged_attention(
        q_latent,
        q_pe,
        cache.latent_cache,
        cache.rope_cache,
        block_tables,
        context_lengths,
        block_size,
        scale,
    ).squeeze(2)
    ref = _reference(
        q_latent,
        q_pe,
        latent_seq,
        rope_seq,
        mx.arange(int(context_lengths[0]), dtype=mx.int32),
        scale,
    )
    mx.eval(out, ref)

    assert cache.get_indexer_cache() is None
    assert mx.allclose(out[0], ref, rtol=1e-4, atol=1e-4)


def test_dsa_paged_attention_glm_dimensions_smoke():
    latent_dim = 512
    rope_dim = 64
    seq_len = 4
    latent_seq = (mx.arange(seq_len * latent_dim, dtype=mx.float32) / 1000).reshape(
        seq_len, latent_dim
    )
    rope_seq = (mx.arange(seq_len * rope_dim, dtype=mx.float32) / 2000).reshape(seq_len, rope_dim)
    cache, block_tables = _make_cache(latent_seq, rope_seq)

    q_latent = (mx.arange(latent_dim, dtype=mx.float32) / 3000).reshape(1, 1, latent_dim)
    q_pe = (mx.arange(rope_dim, dtype=mx.float32) / 4000).reshape(1, 1, rope_dim)
    topk = mx.array([[3, 1]], dtype=mx.int32)
    scale = 0.25

    out = dsa_paged_attention(
        q_latent,
        q_pe,
        cache.latent_cache,
        cache.rope_cache,
        block_tables,
        mx.array([seq_len], dtype=mx.int32),
        topk,
        cache.block_size,
        scale,
    ).squeeze(2)
    ref = _reference(q_latent, q_pe, latent_seq, rope_seq, topk[0], scale)
    mx.eval(out, ref)

    assert out.shape == (1, 1, latent_dim)
    assert mx.allclose(out[0], ref, rtol=1e-4, atol=1e-4)


def test_dsa_reshape_and_cache_supports_mla_dims():
    latent_seq = mx.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5, 1.6],
            [1.7, 1.8, 1.9, 2.0],
        ],
        dtype=mx.float32,
    )
    rope_seq = mx.array(
        [
            [0.2, -0.1],
            [0.1, 0.0],
            [-0.2, 0.3],
            [0.4, 0.2],
            [-0.3, 0.5],
        ],
        dtype=mx.float32,
    )
    cache = DeepSeekSparseCache(
        num_blocks=2,
        block_size=4,
        num_kv_heads=1,
        head_dim=latent_seq.shape[-1],
        head_dim_v=latent_seq.shape[-1],
        dtype=mx.float32,
        index_head_dim=rope_seq.shape[-1],
        index_n_heads=1,
        kv_lora_rank=latent_seq.shape[-1],
        qk_rope_head_dim=rope_seq.shape[-1],
        index_key_heads=1,
    )
    block_tables = mx.array([[0, 1]], dtype=mx.int32)

    reshape_and_cache(
        latent_seq[:4][None, :, None, :],
        rope_seq[:4][None, :, None, :],
        cache.latent_cache,
        cache.rope_cache,
        block_tables,
        mx.array([4], dtype=mx.int32),
        cache.block_size,
        slot_mapping=mx.arange(4, dtype=mx.int64),
    )
    reshape_and_cache(
        latent_seq[4][None, None, :],
        rope_seq[4][None, None, :],
        cache.latent_cache,
        cache.rope_cache,
        block_tables,
        mx.array([5], dtype=mx.int32),
        cache.block_size,
    )
    mx.eval(cache.latent_cache, cache.rope_cache)
    mx.synchronize()

    latent_back, rope_back = cache.read_prefix_mla(block_tables[0], 5)
    mx.eval(latent_back, rope_back)

    assert mx.allclose(latent_back[0], latent_seq)
    assert mx.allclose(rope_back[0], rope_seq)
