import sys

import mlx.core as mx
import pytest

from parallax.models.deepseek_v32 import ModelArgs, ParallaxDeepSeekV32Attention
from parallax.server.cache.dsa_cache import DeepSeekSparseCache

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


def test_attention_decode_forward_uses_glm_style_kv_cache():
    args = _tiny_args()
    attention = ParallaxDeepSeekV32Attention(args)
    cache = DeepSeekSparseCache(
        num_blocks=1,
        block_size=8,
        num_kv_heads=args.num_key_value_heads,
        head_dim=args.qk_nope_head_dim + args.qk_rope_head_dim,
        head_dim_v=args.v_head_dim,
        dtype=mx.float32,
        index_head_dim=args.index_head_dim,
        index_n_heads=args.index_n_heads,
    )

    output = attention(
        mx.zeros((1, 1, args.hidden_size), dtype=mx.float32),
        cache=cache,
        block_tables=mx.array([[0]], dtype=mx.int32),
        context_lengths=mx.array([1], dtype=mx.int32),
    )
    mx.eval(output)

    assert output.shape == (1, 1, args.hidden_size)
