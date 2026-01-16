"""
hidden_dimefines the Qwen3 model.
"""

from typing import Any, List, Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_next import ModelArgs
from mlx_lm.models.qwen3_next import Qwen3NextAttention as MLXQwen3NextAttention
from mlx_lm.models.qwen3_next import Qwen3NextDecoderLayer as MLXQwen3NextBlock
from mlx_lm.models.qwen3_next import Qwen3NextGatedDeltaNet as MLXQwen3NextGatedDeltaNet

from parallax.server.cache.base import BaseCache
from parallax.utils.prefix_cache_utils import compute_attention_with_prefix_cache
from parallax_extensions.ops import paged_attention_v1, reshape_and_cache


class ParallaxQwen3NextAttention(MLXQwen3NextAttention):
    """A custom attention module for Parallax, extending the Qwen3Next Attention class.

    We apply explicit KV cache handling and passing in `offset` directly from Request.
    This version returns the new K and V states for external caching.
    """

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[BaseCache] = None,
        block_tables: Optional[mx.array] = None,
        context_lengths: Optional[mx.array] = None,
        slot_mapping: Optional[mx.array] = None,
        prefix_lens: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """
        Attention forward pass with explicit KV cache handling.

        Args:
            x: (batch, target_len, hidden_dim) - Input hidden states for the current query segment.
            mask: (batch, n_q_heads, target_len, source_len)
            cache: BaseCache object containing the layer cache.
            block_tables: (batch, max_blocks) - PagedKV block tables.
            context_lengths: (batch,) - PagedKV sequence lengths.
            slot_mapping: (batch * target_len,) - Flattened slot mapping.
            prefix_lens: (batch,) - Number of prefix tokens already cached (for RoPE offset).

        Returns:
            output: (batch, target_len, hidden_dim) - Output hidden states.
        """
        batch, target_len, _ = x.shape

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
        values_new = values_new.reshape(batch, target_len, self.num_key_value_heads, -1)

        key_cache_global, value_cache_global = cache.get_cache()

        if target_len == 1:
            current_pos = context_lengths - 1
        elif prefix_lens is not None:
            current_pos = prefix_lens
        else:
            current_pos = 0
        queries_rotated = self.rope(queries_new, offset=current_pos)
        keys_rotated = self.rope(keys_new, offset=current_pos)

        block_size = key_cache_global.shape[3]
        reshape_and_cache(
            keys_rotated.transpose(0, 2, 1, 3),
            values_new,
            key_cache_global,
            value_cache_global,
            block_tables,
            context_lengths,
            block_size,
            slot_mapping=slot_mapping,
        )
        if target_len == 1:
            # Decode Phase: Use Paged Attention Kernel
            output = paged_attention_v1(
                queries_rotated,
                key_cache_global,
                value_cache_global,
                block_tables,
                context_lengths,
                block_size,
                self.scale,
                self.num_key_value_heads,
            )
            output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
        else:
            # Prefill Phase: Need to attend to both cached prefix and new tokens
            # Check if any request has prefix cache
            has_prefix_cache = prefix_lens is not None and bool(mx.any(prefix_lens > 0))

            if has_prefix_cache:
                # Use shared prefix cache handling with batch processing
                k_new = keys_rotated  # (batch, num_key_value_heads, target_len, head_dim)
                v_new = values_new.transpose(
                    0, 2, 1, 3
                )  # (batch, num_key_value_heads, target_len, head_dim)
                output = compute_attention_with_prefix_cache(
                    queries_rotated,  # (batch, num_attention_heads, target_len, head_dim)
                    k_new,
                    v_new,
                    cache,
                    block_tables,
                    prefix_lens,
                    target_len,
                    self.scale,
                    self.num_key_value_heads,
                    mask=mask,
                )
            else:
                # No prefix cache, use standard self-attention on local data only
                if mask is not None:
                    mask = mx.array(mask, dtype=queries_rotated.dtype)

                output = scaled_dot_product_attention(
                    queries_rotated,
                    keys_rotated,
                    values_new.transpose(0, 2, 1, 3),
                    scale=self.scale,
                    mask=mask,
                    cache=None,
                )
                output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)

        return self.o_proj(output * mx.sigmoid(gate))


class ParallaxQwen3NextGatedDeltaNet(MLXQwen3NextGatedDeltaNet):
    def __call__(
        self,
        x: mx.array,
        cache: Optional[BaseCache] = None,
        state_slot_mapping: Optional[mx.array] = None,
        **kwargs,
    ):
        batch, target_len, _ = x.shape
        q, k, v, z, b, a = self.fix_query_key_value_ordering(
            self.in_proj_qkvz(x), self.in_proj_ba(x)
        )

        if target_len == 1:
            conv_state, state1 = cache.read_states(state_slot_mapping)
        else:
            conv_state = mx.zeros(
                (batch, self.conv_kernel_size - 1, self.conv_dim),
                dtype=x.dtype,
            )
            state1 = None

        mixed_qkv = mx.concatenate(
            [
                q.reshape(batch, target_len, -1),
                k.reshape(batch, target_len, -1),
                v.reshape(batch, target_len, -1),
            ],
            axis=-1,
        )
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)

        state0 = conv_input[:, -(self.conv_kernel_size - 1) :]
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
        out, state1 = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, state1)
        out = self.norm(out, z)

        cache.write_states(state_slot_mapping, state0, state1)

        return self.out_proj(out.reshape(batch, target_len, -1))


class ParallaxQwen3NextBlock(MLXQwen3NextBlock):

    def __init__(self, args: ModelArgs, layer_idx: int, local_layer_idx: int):
        super().__init__(args, layer_idx)
        self.layer_idx = layer_idx
        self.local_layer_idx = local_layer_idx
        if self.is_linear:
            self.linear_attn = ParallaxQwen3NextGatedDeltaNet(args)
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
                self.input_layernorm(x), cache[self.local_layer_idx], state_slot_mapping, **kwargs
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
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "Qwen3NextForCausalLM"


EntryClass = ParallaxQwen3NextBlock
