"""
Defines the KimiVL model for Parallax.

KimiVL uses a DeepSeek-V3 based language model with MoE and a MoonViT vision encoder.
This module reuses components from mlx-vlm and adds PagedAttention support
for distributed inference.
"""

from typing import Any, List, Optional

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention

# Import from mlx-vlm kimi_vl language module
from mlx_vlm.models.kimi_vl.language import DeepseekV3Attention as MLXKimiVLAttention
from mlx_vlm.models.kimi_vl.language import (
    DeepseekV3DecoderLayer as MLXKimiVLDecoderLayer,
)

from parallax.metal.paged_attention.kernel import paged_attention, reshape_and_cache
from parallax.server.cache.base import BaseCache
from parallax.utils.prefix_cache_utils import compute_attention_with_prefix_cache
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class ParallaxKimiVLAttention(MLXKimiVLAttention):
    """KimiVL (DeepSeek-V3) Attention with PagedAttention support for Parallax.

    This extends the MLX-VLM KimiVL attention (DeepseekV3Attention) with:
    - Paged KV cache support for efficient memory management
    - Block-table based attention for decode phase
    - Prefix cache support for prefill phase
    """

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[BaseCache] = None,
        offset: int = 0,
        lengths: Optional[mx.array] = None,
        block_tables: Optional[mx.array] = None,
        context_lengths: Optional[mx.array] = None,
        slot_mapping: Optional[mx.array] = None,
        prefix_lens: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        batch, target_len, _ = x.shape

        # Q projection (with optional LoRA)
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(batch, target_len, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        # KV projection (with MQA compression)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(batch, target_len, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))

        kv = kv.reshape(batch, target_len, self.num_heads, -1)
        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)
        k_nope = k_nope.transpose(0, 2, 1, 3)

        # Get KV cache
        key_cache_global, value_cache_global = cache.get_cache()

        # Compute RoPE offsets
        if target_len == 1:
            # Decode phase: position is context_length - 1
            current_pos = context_lengths - 1
        elif prefix_lens is not None:
            # Prefill phase with prefix cache
            current_pos = prefix_lens
        else:
            # Prefill phase without prefix cache
            current_pos = 0

        # Apply RoPE
        q_pe = self.rope(q_pe, offset=current_pos)
        k_pe = self.rope(k_pe, offset=current_pos)

        k_pe = mx.repeat(k_pe, self.num_heads, axis=1)
        queries = mx.concatenate([q_nope, q_pe], axis=-1)
        keys = mx.concatenate([k_nope, k_pe], axis=-1)

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

        if target_len == 1:
            # Decode phase: Use Paged Attention
            output = paged_attention(
                queries,
                key_cache_global,
                value_cache_global,
                block_tables,
                context_lengths,
                block_size,
                self.scale,
                self.num_heads,
                v_head_dim=values.shape[-1],
            )
            output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
        else:
            # Prefill phase
            has_prefix_cache = prefix_lens is not None and bool(mx.any(prefix_lens > 0))

            if has_prefix_cache:
                k_new = keys
                v_new = values.transpose(0, 2, 1, 3)
                output = compute_attention_with_prefix_cache(
                    queries,
                    k_new,
                    v_new,
                    cache,
                    block_tables,
                    prefix_lens,
                    target_len,
                    self.scale,
                    self.num_heads,
                    mask=mask,
                )
            else:
                # Standard self-attention
                if mask is not None:
                    mask = mx.array(mask, dtype=queries.dtype)

                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values.transpose(0, 2, 1, 3),
                    scale=self.scale,
                    mask=mask,
                    cache=None,
                )
                output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)

        return self.o_proj(output)


class ParallaxKimiVLBlock(MLXKimiVLDecoderLayer):
    """KimiVL Transformer block with PagedAttention support.

    Extends the MLX-VLM KimiVL decoder layer to use ParallaxKimiVLAttention
    and pass through paged attention arguments.
    """

    def __init__(self, args, layer_idx: int, local_layer_idx: int):
        super().__init__(args, layer_idx=layer_idx)
        # Replace attention with Parallax version
        self.self_attn = ParallaxKimiVLAttention(args)
        self.layer_idx = layer_idx
        self.local_layer_idx = local_layer_idx

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
        lengths: Optional[mx.array] = None,
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
        return "KimiVLForConditionalGeneration"


EntryClass = ParallaxKimiVLBlock
