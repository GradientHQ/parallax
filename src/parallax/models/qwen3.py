"""
hidden_dimefines the Qwen3 model.
"""

from typing import Optional, Tuple

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.qwen3 import Attention as MLXQwen3Attention
from mlx_lm.models.qwen3 import ModelArgs
from mlx_lm.models.qwen3 import TransformerBlock as MLXQwen3Block


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
        offset: int = 0,
        lengths: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Attention forward pass with explicit KV cache handling.

        Args:
            x: (batch, target_len, hidden_dim) - Input hidden states for the current query segment.
            mask: (batch, n_q_heads, target_len, source_len)
            cache: Optional tuple (past_k, past_v).
                   shape: (batch, n_kv_heads, S_past_padded, head_dim)
            offset: source_len_padded (scalar, used for RoPE calculation).

        Returns:
            output_h: (batch, target_len, hidden_dim) - Output hidden states.
            new_k: (batch, n_kv_heads, target_len, head_dim) - New keys for this segment.
            new_v: (batch, n_kv_heads, target_len, head_dim) - New values for this segment.
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
        values_new = values_new.reshape(batch, target_len, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        # for batch, rope offset is not correct due to padding in batch

        if cache is not None:
            queries_rotated_list = []
            keys_rotated_list = []
            for i in range(batch):
                individual_offset = int(lengths[i])
                query_single = queries_new[i : i + 1]
                key_single = keys_new[i : i + 1]
                query_rotated_single = self.rope(query_single, offset=individual_offset)
                key_rotated_single = self.rope(key_single, offset=individual_offset)
                queries_rotated_list.append(query_rotated_single)
                keys_rotated_list.append(key_rotated_single)
            queries_rotated = mx.concatenate(queries_rotated_list, axis=0)
            keys_rotated = mx.concatenate(keys_rotated_list, axis=0)
            past_k, past_v = cache
            if past_k is not None and past_v is not None:
                if past_k.shape[2] != offset:
                    raise ValueError(
                        f"ParallaxAttention: Expected past_k sequence length {past_k.shape[2]} "
                        f"to match RoPE offset {offset} (S_past_padded)."
                    )
                final_keys_for_attn = mx.concatenate([past_k, keys_rotated], axis=2)
                final_values_for_attn = mx.concatenate([past_v, values_new], axis=2)
            else:
                raise ValueError("cache was provided but one of k/v was None.")
        else:
            queries_rotated = queries_new
            keys_rotated = keys_new
            for i in range(batch):
                seq_len = int(lengths[i])
                q_slice = queries_new[i, :, :seq_len, :]
                k_slice = keys_new[i, :, :seq_len, :]
                q_rotated_slice = self.rope(q_slice)
                k_rotated_slice = self.rope(k_slice)
                queries_rotated[i, :, :seq_len, :] = q_rotated_slice
                keys_rotated[i, :, :seq_len, :] = k_rotated_slice
            final_keys_for_attn = keys_rotated
            final_values_for_attn = values_new

        output = scaled_dot_product_attention(
            queries_rotated,
            final_keys_for_attn,
            final_values_for_attn,
            scale=self.scale,
            mask=mask,
            cache=None,
        )

        output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
        return self.o_proj(output), (keys_rotated, values_new)


class ParallaxQwen3Block(MLXQwen3Block):
    """A custom transformer block for Parallax, extending the Qwen3 Block class.
    This version handles the KV cache explicitly and returns new K and V states.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__(args)
        self.self_attn = ParallaxQwen3Attention(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        offset: int = 0,
        lengths: Optional[mx.array] = None,
    ):
        r, (k_cache, v_cache) = self.self_attn(
            self.input_layernorm(x), mask, cache, offset=offset, lengths=lengths
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, (k_cache, v_cache)

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "Qwen3ForCausalLM"


EntryClass = ParallaxQwen3Block
