# Copyright © 2025 Apple Inc.

from typing import Optional, Tuple

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.minimax import MiniMaxAttention as MLXMiniMaxAttention
from mlx_lm.models.minimax import MiniMaxDecoderLayer as MLXMiniMaxBlock
from mlx_lm.models.minimax import ModelArgs

from parallax.metal.paged_attention.kernel import paged_attention, reshape_and_cache


class ParallaxMiniMaxAttention(MLXMiniMaxAttention):

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        block_tables: Optional[mx.array] = None,
        context_lengths: Optional[mx.array] = None,
        slot_mapping: Optional[mx.array] = None,
        layer_idx: int = 0,
    ) -> mx.array:

        batch, target_len, _ = x.shape

        queries_new = self.q_proj(x)
        keys_new = self.k_proj(x)
        values = self.v_proj(x)

        if self.use_qk_norm:
            queries_new = self.q_norm(queries_new)
            keys_new = self.k_norm(keys_new)

        queries_new = queries_new.reshape(
            batch, target_len, self.num_attention_heads, -1
        ).transpose(0, 2, 1, 3)
        keys_new = keys_new.reshape(batch, target_len, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )
        values_new = values.reshape(batch, target_len, self.num_key_value_heads, -1)

        key_cache_global, value_cache_global = cache

        queries_rotated_list = []
        keys_rotated_list = []

        for i in range(batch):
            current_pos = int(context_lengths[i]) - 1 if target_len == 1 else 0
            q_slice = queries_new[i : i + 1]
            k_slice = keys_new[i : i + 1]
            q_rot = self.rope(q_slice, offset=current_pos)
            k_rot = self.rope(k_slice, offset=current_pos)
            queries_rotated_list.append(q_rot)
            keys_rotated_list.append(k_rot)

        queries_rotated = mx.concatenate(queries_rotated_list, axis=0)
        keys_rotated = mx.concatenate(keys_rotated_list, axis=0)

        block_size = key_cache_global.shape[3]

        reshape_and_cache(
            keys_rotated.transpose(0, 2, 1, 3),
            values_new,
            key_cache_global,
            value_cache_global,
            block_tables,
            context_lengths,
            block_size,
            layer_idx,
            slot_mapping=slot_mapping,
        )

        # 3. Compute Attention
        if target_len == 1:
            # Decode Phase: Use Paged Attention Kernel
            output = paged_attention(
                queries_rotated,
                key_cache_global,
                value_cache_global,
                block_tables,
                context_lengths,
                block_size,
                self.scale,
                self.num_key_value_heads,
                layer_idx,
            )
            output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
        else:
            # Prefill Phase: Use Standard Self-Attention on local data
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


class ParallaxMiniMaxBlock(MLXMiniMaxBlock):
    """A custom transformer block for Parallax, extending the MiniMax Block class.
    This version handles the KV cache explicitly and returns new K and V states.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__(args)
        self.self_attn = ParallaxMiniMaxAttention(args)
        self.layer_idx = layer_idx

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        block_tables: Optional[mx.array] = None,
        context_lengths: Optional[mx.array] = None,
        slot_mapping: Optional[mx.array] = None,
        **kwargs,
    ):
        r = self.self_attn(
            self.input_layernorm(x),
            mask,
            cache,
            block_tables=block_tables,
            context_lengths=context_lengths,
            slot_mapping=slot_mapping,
            layer_idx=self.layer_idx,
        )
        h = x + r
        r = self.block_sparse_moe(self.post_attention_layernorm(h))
        out = h + r
        return out

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "MiniMaxM2ForCausalLM"


EntryClass = ParallaxMiniMaxBlock
