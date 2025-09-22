"""
hidden_dimefines the Qwen3 model.
"""

from typing import Optional, Tuple

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.qwen3_moe import Attention as MLXQwen3MoeAttention
from mlx_lm.models.qwen3_moe import ModelArgs
from mlx_lm.models.qwen3_moe import Qwen3MoeDecoderLayer as MLXQwen3MoeBlock


class ParallaxQwen3MoeAttention(MLXQwen3MoeAttention):
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
        queries_rotated = self.rope(queries_new, offset=offset)
        keys_rotated = self.rope(keys_new, offset=offset)

        if cache is not None:
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


class ParallaxQwen3MoeBlock(MLXQwen3MoeBlock):
    """A custom transformer block for Parallax, extending the Qwen3 Block class.
    This version handles the KV cache explicitly and returns new K and V states.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__(args, layer_idx)
        self.self_attn = ParallaxQwen3MoeAttention(args, layer_idx)

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
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, (k_cache, v_cache)

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "Qwen3MoeForCausalLM"


EntryClass = ParallaxQwen3MoeBlock
