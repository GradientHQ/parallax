"""
hidden_dimefines the Qwen3 model.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_next import ModelArgs
from mlx_lm.models.qwen3_next import Qwen3NextAttention as MLXQwen3NextAttention
from mlx_lm.models.qwen3_next import Qwen3NextDecoderLayer as MLXQwen3NextBlock
from mlx_lm.models.qwen3_next import Qwen3NextGatedDeltaNet as MLXQwen3NextGatedDeltaNet


class ParallaxQwen3NextAttention(MLXQwen3NextAttention):
    """A custom attention module for Parallax, extending the Qwen3 Attention class.

    We apply explicit KV cache handling and passing in `offset` directly from Request.
    This version returns the new K and V states for external caching.
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.hidden_size = args.hidden_size
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        offset: int = 0,
        state_cache: Optional[Tuple[mx.array, mx.array]] = None,
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
        # print("inputs shape:", x.shape)
        # print(f"x.value --- IGNORE --- {x}")

        queries_new = self.q_proj(x)
        keys_new = self.k_proj(x)
        values_new = self.v_proj(x)

        queries_new, gate = mx.split(
            queries_new.reshape(batch, target_len, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(batch, target_len, -1)
        queries_new = self.q_norm(queries_new).transpose(0, 2, 1, 3)
        keys_new = self.k_norm(
            keys_new.reshape(batch, target_len, self.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values_new = values_new.reshape(batch, target_len, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

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

        return self.o_proj(output * mx.sigmoid(gate)), (
            keys_rotated,
            values_new,
            (
                state_cache[0]
                if (state_cache is not None)
                else mx.zeros((batch, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
            ),
            (
                state_cache[1]
                if (state_cache is not None)
                else mx.zeros(
                    (batch, self.num_v_heads, self.head_k_dim, self.head_v_dim), dtype=x.dtype
                )
            ),
        )


class ParallaxQwen3NextGatedDeltaNet(MLXQwen3NextGatedDeltaNet):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim

    def __call__(
        self,
        inputs,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        state_cache: Optional[Tuple[mx.array, mx.array]] = None,
    ):
        B, S, _ = inputs.shape
        # print(f"inputs.value --- IGNORE --- {inputs}")
        q, k, v, z, b, a = self.fix_query_key_value_ordering(
            self.in_proj_qkvz(inputs), self.in_proj_ba(inputs)
        )

        if state_cache is not None and state_cache[0] is not None:
            conv_state = state_cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        mixed_qkv = mx.concatenate(
            [q.reshape(B, S, -1), k.reshape(B, S, -1), v.reshape(B, S, -1)], axis=-1
        )
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)

        state0 = conv_input[:, -(self.conv_kernel_size - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        if state_cache is not None:
            state1 = state_cache[1]
        else:
            state1 = None

        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state1 = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, state1)

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1)), (
            (
                cache[0][..., :S, :]
                if cache is not None
                else mx.zeros((B, self.num_key_value_heads, S, self.head_dim), dtype=inputs.dtype)
            ),
            (
                cache[1][..., :S, :]
                if cache is not None
                else mx.zeros((B, self.num_key_value_heads, S, self.head_dim), dtype=inputs.dtype)
            ),
            state0,
            state1,
        )


class ParallaxQwen3NextBlock(MLXQwen3NextBlock):
    """A custom transformer block for Parallax, extending the Qwen3 Block class.
    This version handles the KV cache explicitly and returns new K and V states.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__(args, layer_idx)
        if self.is_linear:
            self.linear_attn = ParallaxQwen3NextGatedDeltaNet(args)
        else:
            self.self_attn = ParallaxQwen3NextAttention(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        offset: int = 0,
        lengths: Optional[mx.array] = None,
        state_cache: Optional[Tuple[mx.array, mx.array]] = None,
    ):
        if self.is_linear:
            r, (k_cache, v_cache, state0, state1) = self.linear_attn(
                self.input_layernorm(x), cache, state_cache
            )
        else:
            r, (k_cache, v_cache, state0, state1) = self.self_attn(
                self.input_layernorm(x), mask, cache, offset, state_cache
            )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, (k_cache, v_cache, state0, state1)

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "Qwen3NextForCausalLM"


EntryClass = ParallaxQwen3NextBlock
