"""
hidden_dimefines the Qwen3 model.
"""

from typing import Optional, Tuple

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.deepseek_v2 import DeepseekV2Attention as MLXDeepseekV2Attention
from mlx_lm.models.deepseek_v2 import DeepseekV2DecoderLayer as MLXDeepseekV2Block
from mlx_lm.models.deepseek_v2 import ModelArgs

from parallax.metal.paged_attention.kernel import paged_attention, reshape_and_cache


class ParallaxDeepSeekV2Attention(MLXDeepseekV2Attention):
    """A custom attention module for Parallax, extending the DeepseekV2 Attention class.

    We apply explicit KV cache handling and passing in `offset` directly from Request.
    This version returns the new K and V states for external caching.
    """

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
        """
        Attention forward pass with explicit KV cache handling.

        Args:
            x: (batch, target_len, hidden_dim) - Input hidden states for the current query segment.
            mask: (batch, n_q_heads, target_len, source_len)
            cache: contains (key_cache, value_cache) global.
            block_tables: (batch, max_blocks) - PagedKV block tables.
            context_lengths: (batch,) - PagedKV sequence lengths.
            layer_idx: Layer index for PagedKV access.

        Returns:
            output_h: (batch, target_len, hidden_dim) - Output hidden states.
        """
        batch, target_len, _ = x.shape
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(batch, target_len, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(batch, target_len, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.reshape(batch, target_len, self.num_heads, -1)

        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)
        k_nope = k_nope.transpose(0, 2, 1, 3)

        # q_pe = self.rope(q_pe, offset=offset)
        # k_pe = self.rope(k_pe, offset=offset)
        key_cache_global, value_cache_global = cache
        q_pe_list = []
        k_pe_list = []
        for i in range(batch):
            current_pos = int(context_lengths[i]) - 1 if target_len == 1 else 0
            q_slice = q_pe[i : i + 1]
            k_slice = k_pe[i : i + 1]
            q_rot = self.rope(q_slice, offset=current_pos)
            k_rot = self.rope(k_slice, offset=current_pos)
            q_pe_list.append(q_rot)
            k_pe_list.append(k_rot)
        q_pe = mx.concatenate(q_pe_list, axis=0)
        k_pe = mx.concatenate(k_pe_list, axis=0)

        k_pe = mx.repeat(k_pe, self.num_heads, axis=1)
        queries = mx.concatenate([q_nope, q_pe], axis=-1)

        # Construct full keys
        keys = mx.concatenate([k_nope, k_pe], axis=-1)

        block_size = key_cache_global.shape[3]

        reshape_and_cache(
            keys.transpose(0, 2, 1, 3),
            values,
            key_cache_global,
            value_cache_global,
            block_tables,
            context_lengths,
            block_size,
            layer_idx=layer_idx,
            slot_mapping=slot_mapping,
        )

        if target_len == 1:
            output = paged_attention(
                queries,
                key_cache_global,
                value_cache_global,
                block_tables,
                context_lengths,
                block_size,
                self.scale,
                self.num_heads,  # num_kv_heads (MQA/MLA, here num_heads == num_kv_heads effectively after repeat?)
                layer_idx,
                v_head_dim=values.shape[-1],
            )
            output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
        else:
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


class ParallaxDeepSeekV2Block(MLXDeepseekV2Block):
    """A custom transformer block for Parallax, extending the Qwen3 Block class.
    This version handles the KV cache explicitly and returns new K and V states.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__(args, layer_idx=layer_idx)
        self.self_attn = ParallaxDeepSeekV2Attention(args)
        self.layer_idx = layer_idx

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        block_tables: Optional[mx.array] = None,
        context_lengths: Optional[mx.array] = None,
        slot_mapping: Optional[mx.array] = None,
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
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "DeepseekV2ForCausalLM"


EntryClass = ParallaxDeepSeekV2Block
