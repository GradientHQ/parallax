"""
hidden_dimefines the Qwen3 model.
"""

from typing import Any, List, Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.deepseek_v2 import DeepseekV2Attention as MLXDeepseekV2Attention
from mlx_lm.models.deepseek_v2 import DeepseekV2DecoderLayer as MLXDeepseekV2Block
from mlx_lm.models.deepseek_v2 import ModelArgs

from parallax.metal.paged_attention.kernel import paged_attention, reshape_and_cache
from parallax.server.cache.base import BaseCache


class ParallaxDeepSeekV2Attention(MLXDeepseekV2Attention):
    """A custom attention module for Parallax, extending the DeepseekV2 Attention class.

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

        key_cache_global, value_cache_global = cache.get_cache()
        q_pe_list = []
        k_pe_list = []
        for i in range(batch):
            # For decode phase: position is context_length - 1
            # For prefill phase: position starts at prefix_len (skip cached prefix tokens)
            if target_len == 1:
                # Decode phase
                current_pos = int(context_lengths[i]) - 1
            else:
                # Prefill phase - start from prefix_len if using prefix cache
                current_pos = int(prefix_lens[i]) if prefix_lens is not None else 0
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
            slot_mapping=slot_mapping,
        )

        if target_len == 1:
            # Decode Phase: Use Paged Attention Kernel
            output = paged_attention(
                queries,
                key_cache_global,
                value_cache_global,
                block_tables,
                context_lengths,
                block_size,
                self.scale,
                self.num_heads,  # num_kv_heads (MQA/MLA, here num_heads == num_kv_heads effectively after repeat?)
                v_head_dim=values.shape[-1],
            )
            output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
        else:
            # Prefill Phase: Need to attend to both cached prefix and new tokens
            # Check if any request has prefix cache
            has_prefix_cache = prefix_lens is not None and bool(mx.any(prefix_lens > 0))

            logger.debug("Prefill phase: prefix_lens=%s", prefix_lens)
            logger.debug("Prefill phase: has_prefix_cache=%s", has_prefix_cache)

            if has_prefix_cache:
                # Read cached prefix KV from paged cache and concatenate with new KV
                # key_cache_global: (num_layers, num_blocks, num_heads, head_dim, block_size)
                # value_cache_global: (num_layers, num_blocks, num_heads, block_size, head_dim)
                output_list = []
                for i in range(batch):
                    prefix_len = int(prefix_lens[i])
                    q_i = queries[i : i + 1]  # (1, num_heads, target_len, head_dim)
                    k_new_i = keys[i : i + 1]  # (1, num_heads, target_len, head_dim)
                    v_new_i = values[i : i + 1].transpose(
                        0, 2, 1, 3
                    )  # (1, num_heads, target_len, head_dim)

                    if prefix_len > 0:
                        # Read prefix KV from cache using block_table
                        block_table_i = block_tables[i]  # (max_blocks,)

                        # Gather prefix tokens from paged cache
                        prefix_k_list = []
                        prefix_v_list = []
                        for pos in range(prefix_len):
                            block_idx = pos // block_size
                            offset_in_block = pos % block_size
                            physical_block = int(block_table_i[block_idx])
                            # key_cache_global[0]: (num_blocks, num_heads, block_size, head_dim)
                            # value_cache_global[0]: (num_blocks, num_heads, block_size, head_dim_v)
                            k_token = key_cache_global[
                                0, physical_block, :, offset_in_block, :
                            ]  # (num_heads, head_dim)
                            v_token = value_cache_global[
                                0, physical_block, :, offset_in_block, :
                            ]  # (num_heads, head_dim_v)
                            prefix_k_list.append(k_token)
                            prefix_v_list.append(v_token)

                        # Stack prefix KV: (prefix_len, num_heads, head_dim)
                        prefix_k = mx.stack(
                            prefix_k_list, axis=0
                        )  # (prefix_len, num_heads, head_dim)
                        prefix_v = mx.stack(
                            prefix_v_list, axis=0
                        )  # (prefix_len, num_heads, head_dim)

                        # Reshape and transpose for attention
                        prefix_k = prefix_k.transpose(1, 0, 2)[
                            None, ...
                        ]  # (1, num_heads, prefix_len, head_dim)
                        prefix_v = prefix_v.transpose(1, 0, 2)[
                            None, ...
                        ]  # (1, num_heads, prefix_len, head_dim)

                        # Concatenate prefix and new KV
                        k_full = mx.concatenate(
                            [prefix_k, k_new_i], axis=2
                        )  # (1, num_heads, prefix_len + target_len, head_dim)
                        v_full = mx.concatenate(
                            [prefix_v, v_new_i], axis=2
                        )  # (1, num_heads, prefix_len + target_len, head_dim)
                    else:
                        k_full = k_new_i
                        v_full = v_new_i

                    # Compute attention for this request
                    # Need to create proper causal mask for the full sequence
                    full_len = k_full.shape[2]
                    # Correct causal mask: position j can attend to positions 0..j
                    row_indices = mx.arange(target_len)[:, None] + prefix_len  # actual positions
                    col_indices = mx.arange(full_len)[None, :]
                    causal_mask = mx.where(col_indices <= row_indices, 0.0, float("-inf"))
                    causal_mask = causal_mask[None, None, :, :].astype(
                        q_i.dtype
                    )  # (1, 1, target_len, full_len)

                    out_i = scaled_dot_product_attention(
                        q_i,
                        k_full,
                        v_full,
                        scale=self.scale,
                        mask=causal_mask,
                        cache=None,
                    )
                    output_list.append(out_i)

                output = mx.concatenate(output_list, axis=0)
                output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
            else:
                # No prefix cache, use standard self-attention on local data only
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

    def __init__(self, args: ModelArgs, layer_idx: int, local_layer_idx: int):
        super().__init__(args, layer_idx=layer_idx)
        self.self_attn = ParallaxDeepSeekV2Attention(args)
        self.layer_idx = layer_idx
        self.local_layer_idx = local_layer_idx

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
        return "DeepseekV2ForCausalLM"


EntryClass = ParallaxDeepSeekV2Block
