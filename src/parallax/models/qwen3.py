"""
hidden_dimefines the Qwen3 model.
"""

from typing import Optional, Tuple

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.qwen3 import Attention as MLXQwen3Attention
from mlx_lm.models.qwen3 import ModelArgs
from mlx_lm.models.qwen3 import TransformerBlock as MLXQwen3Block

from parallax.metal.paged_attention.kernel import paged_attention, reshape_and_cache


class ParallaxQwen3Attention(MLXQwen3Attention):
    """A custom attention module for Parallax, extending the Qwen3 Attention class.

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

        key_cache_global, value_cache_global = cache

        queries_rotated_list = []
        keys_rotated_list = []

        for i in range(batch):
            # TODO: for chunked Prefill, we need to pass in the offset for each chunk.
            # Currently, we assume the offset is 0 for all chunks.
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
                self.n_kv_heads,
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


class ParallaxQwen3Block(MLXQwen3Block):
    """A custom transformer block for Parallax, extending the Qwen3 Block class.
    This version handles the KV cache explicitly and returns new K and V states.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__(args)
        self.self_attn = ParallaxQwen3Attention(args)
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
        return "Qwen3ForCausalLM"


EntryClass = ParallaxQwen3Block
