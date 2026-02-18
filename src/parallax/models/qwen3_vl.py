"""
Defines the Qwen3VL model for Parallax.

This module reuses components from mlx-vlm and adds PagedAttention support
for distributed inference.
"""

from typing import Any, List, Optional

import mlx.core as mx
from mlx import nn

# Import from mlx-vlm
from mlx_vlm.models.qwen3_vl.language import MLP
from mlx_vlm.models.qwen3_vl.language import Attention as MLXQwen3VLAttention
from mlx_vlm.models.qwen3_vl.language import apply_multimodal_rotary_pos_emb

from parallax.server.cache.base import BaseCache
from parallax_extensions.ops import paged_attention_v1, reshape_and_cache
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class ParallaxQwen3VLAttention(MLXQwen3VLAttention):
    """Qwen3VL Attention with PagedAttention support for Parallax."""

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[BaseCache] = None,
        block_tables: Optional[mx.array] = None,
        context_lengths: Optional[mx.array] = None,
        slot_mapping: Optional[mx.array] = None,
        prefix_lens: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, self.head_dim)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim)

        # Get KV cache
        key_cache_global, value_cache_global = cache.get_cache()

        # Compute RoPE position
        if L == 1:
            # Decode phase: use context_lengths - 1 as offset
            current_pos = context_lengths - 1
            pos_ids = mx.broadcast_to(current_pos[:, None], (B, L))
            pos_ids = mx.broadcast_to(pos_ids[None, :, :], (3, B, L))
        elif position_ids is not None:
            # Prefill with MRoPE position_ids
            pos_ids = position_ids
        elif prefix_lens is not None:
            # Prefill with prefix cache
            pos_ids = mx.arange(L)[None, :] + prefix_lens[:, None]
            pos_ids = mx.broadcast_to(pos_ids[None, :, :], (3, B, L))
        else:
            # Standard prefill
            pos_ids = mx.arange(L)[None, :]
            pos_ids = mx.broadcast_to(pos_ids, (B, L))
            pos_ids = mx.broadcast_to(pos_ids[None, :, :], (3, B, L))

        cos, sin = self.rotary_emb(values, pos_ids)
        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        # Ensure dtype consistency with cache (RoPE may output float32)
        cache_dtype = key_cache_global.dtype
        if keys.dtype != cache_dtype:
            keys = keys.astype(cache_dtype)
        if values.dtype != cache_dtype:
            values = values.astype(cache_dtype)
        if queries.dtype != cache_dtype:
            queries = queries.astype(cache_dtype)

        # Cache update with PagedAttention
        block_size = key_cache_global.shape[3]
        reshape_and_cache(
            keys.transpose(0, 2, 1, 3),
            values,
            key_cache_global,
            value_cache_global,
            block_tables,
            context_lengths,
            block_size,
            slot_mapping=slot_mapping,
        )

        # Compute attention
        if L == 1:
            # Decode: use PagedAttention
            output = paged_attention_v1(
                queries,
                key_cache_global,
                value_cache_global,
                block_tables,
                context_lengths,
                block_size,
                self.scale,
                self.n_kv_heads,
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        else:
            # Prefill: standard attention
            from mlx_lm.models.base import scaled_dot_product_attention

            output = scaled_dot_product_attention(
                queries,
                keys,
                values.transpose(0, 2, 1, 3),
                scale=self.scale,
                mask=mask,
                cache=None,
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output)


class ParallaxQwen3VLBlock(nn.Module):
    """Qwen3VL Transformer block with PagedAttention support."""

    def __init__(self, args, layer_idx: int, local_layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.self_attn = ParallaxQwen3VLAttention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
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

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "Qwen3VLForConditionalGeneration"


EntryClass = ParallaxQwen3VLBlock
