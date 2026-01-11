"""
hidden_dimefines the Qwen3 model.
"""

from typing import Any, List, Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.qwen3 import Attention as MLXQwen3Attention
from mlx_lm.models.qwen3 import ModelArgs
from mlx_lm.models.qwen3 import TransformerBlock as MLXQwen3Block

from parallax.server.cache.base import BaseCache
from parallax_extensions.ops import paged_attention_v1, reshape_and_cache
from mlx.nn.layers.distributed import shard_linear
import time


class ParallaxQwen3Attention(MLXQwen3Attention):
    """A custom attention module for Parallax, extending the Qwen3 Attention class.

    We apply explicit KV cache handling and passing in `offset` directly from Request.
    This version returns the new K and V states for external caching.
    """

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[BaseCache] = None,
        block_tables: Optional[mx.array] = None,
        context_lengths: Optional[mx.array] = None,
        slot_mapping: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """
        Attention forward pass with explicit KV cache handling.

        Args:
            x: (batch, target_len, hidden_dim) - Input hidden states for the current query segment.
            mask: (batch, n_q_heads, target_len, source_len)
            cache: BaseCache object containing the layer cache.
            block_tables: (batch, max_blocks) - PagedKV block tables.
            context_lengths: (batch,) - PagedKV sequence lengths.
            slot_mapping: (batch * target_len,) - Flattened slot mapping.
            prefix_lens: (batch,) - Number of prefix tokens already cached (for RoPE offset).

        Returns:
            output: (batch, target_len, hidden_dim) - Output hidden states.
        """
        batch, target_len, _ = x.shape
        queries_new = self.q_proj(x)
        keys_new = self.k_proj(x)
        values_new = self.v_proj(x)


        queries_new = self.q_norm(
            queries_new.reshape(batch, target_len, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys_new = self.k_norm(keys_new.reshape(batch, target_len, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values_new = values_new.reshape(batch, target_len, self.n_kv_heads, -1)

        key_cache_global, value_cache_global = cache.get_cache()

        if target_len == 1:
            current_pos = context_lengths - 1
        else:
            current_pos = 0
        queries_rotated = self.rope(queries_new, offset=current_pos)
        keys_rotated = self.rope(keys_new, offset=current_pos)

        block_size = key_cache_global.shape[3]

        reshape_and_cache(
            keys_rotated.transpose(0, 2, 1, 3),
            values_new,
            key_cache_global,
            value_cache_global,
            block_tables,
            context_lengths,
            block_size,
            slot_mapping=slot_mapping,
        )

        # 3. Compute Attention
        if target_len == 1:
            # Decode Phase: Use Paged Attention Kernel
            output = paged_attention_v1(
                queries_rotated,
                key_cache_global,
                value_cache_global,
                block_tables,
                context_lengths,
                block_size,
                self.scale,
                self.n_kv_heads,
            )
            output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
        else:
            # No prefix cache, use standard self-attention on local data only
            output = scaled_dot_product_attention(
                queries_rotated,
                keys_rotated,
                values_new.transpose(0, 2, 1, 3),
                scale=self.scale,
                mask=mask,
                cache=None,
            )
            output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)

        return self.o_proj(output)


class ParallaxQwen3Block(MLXQwen3Block):
    """A custom transformer block for Parallax, extending the Qwen3 Block class.
    This version handles the KV cache explicitly and returns new K and V states.
    """

    def __init__(self, args: ModelArgs, layer_idx: int, local_layer_idx: int):
        super().__init__(args)
        self.self_attn = ParallaxQwen3Attention(args)
        self.layer_idx = layer_idx
        self.local_layer_idx = local_layer_idx

    def test_mlp(self, x: mx.array):
        mx.eval(x)
        start_time = time.time()
        for _ in range(100):
            x = self.mlp(x)
        mx.eval(x)
        logger.warning(f"test_mlp done, avg time: {(time.time() - start_time) / 100 * 1000:.3f} ms")
        return x

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
        block_tables: Optional[mx.array] = None,
        context_lengths: Optional[mx.array] = None,
        slot_mapping: Optional[mx.array] = None,
        **kwargs,
    ):
        start_time = time.time()
        r = self.self_attn(
            self.input_layernorm(x),
            mask,
            cache[self.local_layer_idx],
            block_tables=block_tables,
            context_lengths=context_lengths,
            slot_mapping=slot_mapping,
            **kwargs,
        )
        # mx.eval(r)
        # logger.warning(f"self attention done, time: {(time.time() - start_time) * 1000:.3f} ms")
        # start_time = time.time()
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        # mx.eval(out)
        # logger.warning(f"mlp done, time: {(time.time() - start_time) * 1000:.3f} ms")
        # self.test_mlp(out)
        return out
    
    def shard(self):
        group = mx.distributed.init()
        N = group.size()
        # Shard the self attention
        self.self_attn.q_proj = shard_linear(
            self.self_attn.q_proj, "all-to-sharded", group=group
        )
        self.self_attn.k_proj = shard_linear(
            self.self_attn.k_proj, "all-to-sharded", group=group
        )
        self.self_attn.v_proj = shard_linear(
            self.self_attn.v_proj, "all-to-sharded", group=group
        )
        self.self_attn.o_proj = shard_linear(
            self.self_attn.o_proj, "sharded-to-all", group=group
        )
        self.self_attn.n_heads //= N
        self.self_attn.n_kv_heads //= N

        # Shard the MLP
        self.mlp.gate_proj = shard_linear(
            self.mlp.gate_proj, "all-to-sharded", group=group
        )
        self.mlp.up_proj = shard_linear(
            self.mlp.up_proj, "all-to-sharded", group=group
        )
        self.mlp.down_proj = shard_linear(
            self.mlp.down_proj, "sharded-to-all", group=group
        )

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "Qwen3ForCausalLM"


EntryClass = ParallaxQwen3Block
