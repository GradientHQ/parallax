import sys

import mlx.core as mx
import pytest

from parallax.models.deepseek_v32 import (
    ModelArgs,
    ParallaxDeepSeekV32Attention,
    ParallaxDeepSeekV32Block,
    derive_indexer_types,
)
from parallax.server.cache.dsa_cache import DeepSeekSparseCache
from parallax.server.cache_manager import CacheManager
from parallax.utils.utils import create_causal_mask

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="MLX tests require macOS")


def _tiny_args():
    return ModelArgs(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=4,
        index_head_dim=4,
        index_n_heads=2,
        index_topk=4,
        num_hidden_layers=1,
        max_position_embeddings=16,
    )


def _tiny_cache(args, *, num_blocks=1, block_size=8):
    return DeepSeekSparseCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=args.num_key_value_heads,
        head_dim=args.qk_nope_head_dim + args.qk_rope_head_dim,
        head_dim_v=args.v_head_dim,
        dtype=mx.float32,
        index_head_dim=args.index_head_dim,
        index_n_heads=args.index_n_heads,
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        index_key_heads=1,
    )


def test_attention_decode_forward_uses_glm_style_kv_cache():
    args = _tiny_args()
    attention = ParallaxDeepSeekV32Attention(args)
    cache = _tiny_cache(args)

    output, topk = attention(
        mx.zeros((1, 1, args.hidden_size), dtype=mx.float32),
        cache=cache,
        block_tables=mx.array([[0]], dtype=mx.int32),
        context_lengths=mx.array([1], dtype=mx.int32),
    )
    mx.eval(output)

    assert output.shape == (1, 1, args.hidden_size)
    assert topk.shape == (1, args.index_topk)


def test_attention_prefix_prefill_reads_prefix_index_cache():
    args = _tiny_args()
    attention = ParallaxDeepSeekV32Attention(args)
    cache = _tiny_cache(args)
    block_tables = mx.array([[0]], dtype=mx.int32)

    prefix, _ = attention(
        mx.ones((1, 4, args.hidden_size), dtype=mx.float32),
        mask=create_causal_mask(4, 4, mx.float32),
        cache=cache,
        block_tables=block_tables,
        context_lengths=mx.array([4], dtype=mx.int32),
        slot_mapping=mx.array([0, 1, 2, 3], dtype=mx.int64),
    )
    mx.eval(prefix, cache.get_indexer_cache())

    index_k = cache.read_index_k(mx.array([0], dtype=mx.int32), 4)
    mx.eval(index_k)
    assert index_k.shape == (1, 4, args.index_head_dim)
    assert float(mx.sum(mx.abs(index_k[0]))) > 0

    chunk, topk = attention(
        mx.full((1, 2, args.hidden_size), 2.0, dtype=mx.float32),
        mask=create_causal_mask(2, 2, mx.float32),
        cache=cache,
        block_tables=block_tables,
        context_lengths=mx.array([6], dtype=mx.int32),
        slot_mapping=mx.array([4, 5], dtype=mx.int64),
        prefix_lens=mx.array([4], dtype=mx.int32),
    )
    mx.eval(chunk, topk, cache.get_indexer_cache())

    assert chunk.shape == (1, 2, args.hidden_size)
    assert topk.shape == (1, 2, args.index_topk)
    assert bool(mx.all((topk == -1) | ((topk >= 0) & (topk < 6))).item())


def test_attention_decode_after_prefill_uses_sparse_mla_cache():
    args = _tiny_args()
    attention = ParallaxDeepSeekV32Attention(args)
    cache = _tiny_cache(args, num_blocks=2, block_size=8)
    block_tables = mx.array([[0, 1]], dtype=mx.int32)

    prefill, _ = attention(
        mx.ones((1, 5, args.hidden_size), dtype=mx.float32),
        mask=create_causal_mask(5, 5, mx.float32),
        cache=cache,
        block_tables=block_tables,
        context_lengths=mx.array([5], dtype=mx.int32),
        slot_mapping=mx.array([0, 1, 2, 3, 4], dtype=mx.int64),
    )
    mx.eval(prefill, cache.get_cache()[0], cache.get_cache()[1], cache.get_indexer_cache())

    output, topk = attention(
        mx.full((1, 1, args.hidden_size), 2.0, dtype=mx.float32),
        cache=cache,
        block_tables=block_tables,
        context_lengths=mx.array([6], dtype=mx.int32),
    )
    mx.eval(output, topk)

    assert output.shape == (1, 1, args.hidden_size)
    assert topk.shape == (1, args.index_topk)
    assert not bool(mx.all(topk == -1).item())
    assert bool(mx.all((topk >= 0) & (topk < 6)).item())


def test_deepseek_sparse_cache_uses_compressed_mla_shapes():
    args = _tiny_args()
    cache = _tiny_cache(args, num_blocks=2, block_size=4)
    latent_cache, rope_cache = cache.get_cache()

    assert latent_cache.shape == (1, 2, 1, 4, args.kv_lora_rank)
    assert rope_cache.shape == (1, 2, 1, 4, args.qk_rope_head_dim)
    assert cache.get_indexer_cache().shape == (1, 2, 1, 4, args.index_head_dim)


def test_cache_manager_counts_compressed_mla_and_unexpanded_indexer_bytes(monkeypatch):
    monkeypatch.setattr(
        mx.metal,
        "device_info",
        lambda: {"max_recommended_working_set_size": 1024 * 1024},
    )
    monkeypatch.setattr(mx, "get_active_memory", lambda: 0)

    manager = CacheManager(
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        head_dim_v=4,
        dtype=mx.float32,
        block_size=4,
        cache_memory_fraction=1.0,
        index_head_dim=4,
        index_n_heads=2,
        index_key_heads=1,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
    )

    dtype_size = 4
    expected = 2 * 4 * (4 + 2 + 4) * dtype_size
    assert manager._calculate_kv_block_bytes(dtype_size) == expected


def test_indexer_types_default_to_all_full():
    assert derive_indexer_types(4) == ["full", "full", "full", "full"]


def test_indexer_types_match_glm_5_2_config_pattern():
    assert derive_indexer_types(
        12,
        index_topk_freq=4,
        first_k_dense_replace=3,
        index_skip_topk_offset=3,
    ) == [
        "full",
        "full",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
        "shared",
    ]


def test_indexer_types_explicit_config_wins():
    indexer_types = ["full", "shared", "full"]

    assert derive_indexer_types(
        3,
        index_topk_freq=4,
        indexer_types=indexer_types,
        first_k_dense_replace=3,
        index_skip_topk_offset=3,
    ) == indexer_types


def test_block_marks_shared_indexer_layers_from_config():
    args = _tiny_args()
    args.num_hidden_layers = 4
    args.indexer_types = ["full", "shared", "full", "shared"]

    full_block = ParallaxDeepSeekV32Block(args, layer_idx=0, local_layer_idx=0)
    shared_block = ParallaxDeepSeekV32Block(args, layer_idx=1, local_layer_idx=1)

    assert full_block.is_full_indexer_layer is True
    assert full_block.self_attn.is_full is True
    assert shared_block.is_full_indexer_layer is False
    assert shared_block.self_attn.is_full is False
