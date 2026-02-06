"""
Defines the Step3.5 model wrapper for Parallax.

This module adapts MLX Step3p5 attention to explicitly handle KV cache and
PagedAttention.
"""

from typing import Any, List, Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

import mlx.core as mx
from mlx.nn.layers.distributed import shard_inplace, shard_linear
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.step3p5 import ModelArgs
from mlx_lm.models.step3p5 import Step3p5Attention as MLXStep3p5Attention
from mlx_lm.models.step3p5 import Step3p5DecoderLayer as MLXStep3p5Block
from mlx_lm.models.step3p5 import Step3p5MLP, Step3p5MoE

from parallax.server.cache.base import BaseCache
from parallax_extensions.ops import paged_attention_v1, reshape_and_cache


class ParallaxStep3p5Attention(MLXStep3p5Attention):
    """Custom attention for Step3.5, with explicit KV cache handling."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__(args, layer_idx)
        self.sliding_window = args.sliding_window

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[BaseCache] = None,
        block_tables: Optional[mx.array] = None,
        context_lengths: Optional[mx.array] = None,
        slot_mapping: Optional[mx.array] = None,
        prefix_lens: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """
        Attention forward pass with explicit KV cache handling.

        Args:
            x: (batch, target_len, hidden_dim) - Input hidden states for the current query segment.
            mask: (batch, 1, target_len, source_len) - Causal + padding mask for prefill.
            cache: BaseCache object containing the layer cache.
            block_tables: (batch, max_blocks) - PagedKV block tables.
            context_lengths: (batch,) - PagedKV sequence lengths.
            slot_mapping: (batch * target_len,) - Flattened slot mapping.
            prefix_lens: (batch,) - Number of prefix tokens already cached (unused).
        """
        batch, target_len, _ = x.shape

        queries_new = self.q_proj(x)
        keys_new = self.k_proj(x)
        values_new = self.v_proj(x)

        queries_new = self.q_norm(
            queries_new.reshape(batch, target_len, self.num_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys_new = self.k_norm(
            keys_new.reshape(batch, target_len, self.num_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        values_new = values_new.reshape(batch, target_len, self.num_kv_heads, -1)

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

        window_size = self.sliding_window if self.is_sliding else None

        if target_len == 1:
            output = paged_attention_v1(
                queries_rotated,
                key_cache_global,
                value_cache_global,
                block_tables,
                context_lengths,
                block_size,
                self.scale,
                self.num_kv_heads,
                window_size=window_size,
            )
            output = output.transpose(0, 2, 1, 3)  # (B, 1, H, D)
        else:
            if window_size is not None:
                row = mx.arange(target_len)[None, :, None]
                col = mx.arange(target_len)[None, None, :]
                window_start = mx.maximum(0, row - window_size + 1)
                in_window = (col >= window_start) & (col <= row)
                window_mask = mx.where(in_window, 0.0, float("-inf"))
                window_mask = window_mask[:, None, :, :]
                if mask is not None:
                    mask = mask + window_mask
                else:
                    mask = window_mask

            if mask is not None:
                mask = mask.astype(queries_rotated.dtype)

            output = scaled_dot_product_attention(
                queries_rotated,
                keys_rotated,
                values_new.transpose(0, 2, 1, 3),
                scale=self.scale,
                mask=mask,
                cache=None,
            )
            output = output.transpose(0, 2, 1, 3)

        if self.use_head_wise_attn_gate:
            output = output * mx.sigmoid(self.g_proj(x))[..., None]

        return self.o_proj(output.reshape(batch, target_len, -1))


class ParallaxStep3p5Block(MLXStep3p5Block):
    """Transformer block wrapper returning explicit KV cache updates."""

    def __init__(self, args: ModelArgs, layer_idx: int, local_layer_idx: int):
        super().__init__(args, layer_idx)
        self.self_attn = ParallaxStep3p5Attention(args, layer_idx)
        self.is_sliding = self.self_attn.is_sliding
        self.layer_idx = layer_idx
        self.local_layer_idx = local_layer_idx

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
        r = self.self_attn(
            self.input_layernorm(x),
            mask,
            cache[self.local_layer_idx],
            block_tables=block_tables,
            context_lengths=context_lengths,
            slot_mapping=slot_mapping,
            **kwargs,
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out

    def shard(self):
        group = mx.distributed.init()
        N = group.size()

        self.self_attn.q_proj = shard_linear(self.self_attn.q_proj, "all-to-sharded", group=group)
        self.self_attn.k_proj = shard_linear(self.self_attn.k_proj, "all-to-sharded", group=group)
        self.self_attn.v_proj = shard_linear(self.self_attn.v_proj, "all-to-sharded", group=group)
        self.self_attn.o_proj = shard_linear(self.self_attn.o_proj, "sharded-to-all", group=group)
        self.self_attn.num_heads //= N
        self.self_attn.num_kv_heads //= N

        if self.self_attn.use_head_wise_attn_gate:
            self.self_attn.g_proj = shard_linear(
                self.self_attn.g_proj, "all-to-sharded", group=group
            )

        if isinstance(self.mlp, Step3p5MLP):
            self.mlp.gate_proj = shard_linear(self.mlp.gate_proj, "all-to-sharded", group=group)
            self.mlp.up_proj = shard_linear(self.mlp.up_proj, "all-to-sharded", group=group)
            self.mlp.down_proj = shard_linear(self.mlp.down_proj, "sharded-to-all", group=group)
        elif isinstance(self.mlp, Step3p5MoE):
            self.mlp.sharding_group = group
            shard_inplace(self.mlp.share_expert.gate_proj, "all-to-sharded", group=group)
            shard_inplace(self.mlp.share_expert.up_proj, "all-to-sharded", group=group)
            shard_inplace(self.mlp.share_expert.down_proj, "sharded-to-all", group=group)
            shard_inplace(self.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(self.mlp.switch_mlp.up_proj, "all-to-sharded", group=group)
            shard_inplace(self.mlp.switch_mlp.down_proj, "sharded-to-all", group=group)

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "Step3p5ForCausalLM"


EntryClass = ParallaxStep3p5Block
