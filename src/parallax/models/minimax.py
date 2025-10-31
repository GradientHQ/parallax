# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, scaled_dot_product_attention
from mlx_lm.models.minimax import MiniMaxAttention as MLXMiniMaxAttention
from mlx_lm.models.minimax import MiniMaxDecoderLayer as MLXMiniMaxBlock
from mlx_lm.models.minimax import ModelArgs


class ParallaxMiniMaxAttention(MLXMiniMaxAttention):

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        offset: int = 0,
        lengths: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:

        batch, target_len, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.reshape(batch, target_len, self.num_attention_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(batch, target_len, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(batch, target_len, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        # for batch, rope offset is not correct due to padding in batch
        queries_rotated = self.rope(queries, offset=offset)
        keys_rotated = self.rope(keys, offset=offset)

        if cache is not None:
            past_k, past_v = cache
            if past_k is not None and past_v is not None:
                if past_k.shape[2] != offset:
                    raise ValueError(
                        f"ParallaxAttention: Expected past_k sequence length {past_k.shape[2]} "
                        f"to match RoPE offset {offset} (S_past_padded)."
                    )
                final_keys_for_attn = mx.concatenate([past_k, keys_rotated], axis=2)
                final_values_for_attn = mx.concatenate([past_v, values], axis=2)
            else:
                raise ValueError("cache was provided but one of k/v was None.")
        else:
            final_keys_for_attn = keys_rotated
            final_values_for_attn = values

        output = scaled_dot_product_attention(
            queries_rotated,
            final_keys_for_attn,
            final_values_for_attn,
            scale=self.scale,
            mask=mask,
            cache=None,
        )

        output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
        return self.o_proj(output), (keys_rotated, values)


class ParallaxMiniMaxBlock(MLXMiniMaxBlock):
    """A custom transformer block for Parallax, extending the MiniMax Block class.
    This version handles the KV cache explicitly and returns new K and V states.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__(args)
        self.self_attn = ParallaxMiniMaxAttention(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        offset: int = 0,
        lengths: Optional[mx.array] = None,
    ):
        r, (k_cache, v_cache) = self.self_attn(self.input_layernorm(x), mask, cache, offset=offset)
        h = x + r
        r = self.block_sparse_moe(self.post_attention_layernorm(h))
        out = h + r
        return out, (k_cache, v_cache)

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "MiniMaxM2ForCausalLM"


EntryClass = ParallaxMiniMaxBlock
