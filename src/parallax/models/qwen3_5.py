"""
Defines the Qwen3.5 text block for Parallax.
"""

from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_5 import DecoderLayer as MLXQwen35Block
from mlx_lm.models.qwen3_5 import GatedDeltaNet as MLXQwen35GatedDeltaNet
from mlx_lm.models.qwen3_5 import TextModelArgs

from parallax.models.qwen3_next import ParallaxQwen3NextAttention
from parallax.server.cache.base import BaseCache


class ParallaxQwen35GatedDeltaNet(MLXQwen35GatedDeltaNet):
    def __call__(
        self,
        x: mx.array,
        cache: Optional[BaseCache] = None,
        state_slot_mapping: Optional[mx.array] = None,
        **kwargs,
    ):
        batch, target_len, _ = x.shape

        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(batch, target_len, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        if target_len == 1:
            conv_state, state = cache.read_states(state_slot_mapping)
        else:
            conv_state = mx.zeros(
                (batch, self.conv_kernel_size - 1, self.conv_dim),
                dtype=x.dtype,
            )
            state = None

        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        next_conv_state = conv_input[:, -(self.conv_kernel_size - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(batch, target_len, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            use_kernel=not self.training,
        )

        cache.write_states(state_slot_mapping, next_conv_state, state)

        out = self.norm(out, z)
        return self.out_proj(out.reshape(batch, target_len, -1))


class ParallaxQwen35Block(MLXQwen35Block):
    def __init__(self, args: TextModelArgs, layer_idx: int, local_layer_idx: int):
        super().__init__(args, layer_idx)
        self.layer_idx = layer_idx
        self.local_layer_idx = local_layer_idx
        if self.is_linear:
            self.linear_attn = ParallaxQwen35GatedDeltaNet(args)
        else:
            self.self_attn = ParallaxQwen3NextAttention(args)

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
        if self.is_linear:
            state_slot_mapping = kwargs.pop("state_slot_mapping", None)
            r = self.linear_attn(
                self.input_layernorm(x),
                cache[self.local_layer_idx],
                state_slot_mapping,
                **kwargs,
            )
        else:
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
        return h + self.mlp(self.post_attention_layernorm(h))

    @classmethod
    def get_architecture(cls):
        return "Qwen3_5ForConditionalGeneration"


EntryClass = ParallaxQwen35Block
