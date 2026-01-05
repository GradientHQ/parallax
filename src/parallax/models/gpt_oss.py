"""
hidden_dimefines the Qwen3 model.
"""

from typing import Any, List, Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask, scaled_dot_product_attention
from mlx_lm.models.gpt_oss import AttentionBlock as MLXGPTOSSAttention
from mlx_lm.models.gpt_oss import ModelArgs
from mlx_lm.models.gpt_oss import TransformerBlock as MLXGPTOSSBlock

from parallax.metal.paged_attention.kernel import paged_attention, reshape_and_cache
from parallax.server.cache.base import BaseCache


class ParallaxGPTOSSAttention(MLXGPTOSSAttention):
    """A custom attention module for Parallax, extending the Qwen3 Attention class.

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
        window_size: Optional[int] = None,
        prefix_lens: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """
        Attention forward pass with PagedAttention integration.

        Args:
            x: (batch, target_len, hidden_dim) - Input hidden states for the current query segment.
            mask: (batch, n_q_heads, target_len, source_len)
            cache: BaseCache object containing the layer cache.
            block_tables: (batch, max_blocks) - PagedKV block tables.
            context_lengths: (batch,) - PagedKV sequence lengths.
            slot_mapping: (batch * target_len,) - Flattened slot mapping.
            window_size: Optional window size for sliding window attention.
            prefix_lens: (batch,) - Number of prefix tokens already cached (for RoPE offset).
        """
        batch, target_len, _ = x.shape

        queries_new = self.q_proj(x)
        keys_new = self.k_proj(x)
        values_new = self.v_proj(x)

        queries_new = queries_new.reshape(
            batch, target_len, self.num_attention_heads, -1
        ).transpose(0, 2, 1, 3)
        keys_new = keys_new.reshape(batch, target_len, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )
        values_new = values_new.reshape(batch, target_len, self.num_key_value_heads, -1)

        key_cache_global, value_cache_global = cache.get_cache()

        queries_rotated_list = []
        keys_rotated_list = []
        for i in range(batch):
            # For decode phase: position is context_length - 1
            # For prefill phase: position starts at prefix_len (skip cached prefix tokens)
            if target_len == 1:
                # Decode phase
                current_pos = int(context_lengths[i]) - 1
            else:
                # Prefill phase - start from prefix_len if using prefix cache
                current_pos = int(prefix_lens[i]) if prefix_lens is not None else 0
            q_slice = queries_new[i : i + 1]
            k_slice = keys_new[i : i + 1]
            q_rot = self.rope(q_slice, offset=current_pos)
            k_rot = self.rope(k_slice, offset=current_pos)
            queries_rotated_list.append(q_rot)
            keys_rotated_list.append(k_rot)

        queries_rotated = mx.concatenate(queries_rotated_list, axis=0)
        keys_rotated = mx.concatenate(keys_rotated_list, axis=0)

        # Update Paged Cache
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

        # Compute Attention
        if target_len == 1:
            # Decode Phase: Use Paged Attention Kernel
            output = paged_attention(
                queries_rotated,
                key_cache_global,
                value_cache_global,
                block_tables,
                context_lengths,
                block_size,
                self.sm_scale,
                self.num_key_value_heads,
                window_size=window_size,
                sinks=self.sinks,
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
                # key_cache_global: (num_layers, num_blocks, num_key_value_heads, head_dim, block_size)
                # value_cache_global: (num_layers, num_blocks, num_key_value_heads, block_size, head_dim)
                output_list = []
                for i in range(batch):
                    prefix_len = int(prefix_lens[i])
                    q_i = queries_rotated[
                        i : i + 1
                    ]  # (1, num_attention_heads, target_len, head_dim)
                    k_new_i = keys_rotated[
                        i : i + 1
                    ]  # (1, num_key_value_heads, target_len, head_dim)
                    v_new_i = values_new[i : i + 1].transpose(
                        0, 2, 1, 3
                    )  # (1, num_key_value_heads, target_len, head_dim)

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
                            # key_cache_global[0]: (num_blocks, num_key_value_heads, block_size, head_dim)
                            # value_cache_global[0]: (num_blocks, num_key_value_heads, block_size, head_dim_v)
                            k_token = key_cache_global[
                                0, physical_block, :, offset_in_block, :
                            ]  # (num_key_value_heads, head_dim)
                            v_token = value_cache_global[
                                0, physical_block, :, offset_in_block, :
                            ]  # (num_key_value_heads, head_dim_v)
                            prefix_k_list.append(k_token)
                            prefix_v_list.append(v_token)

                        # Stack prefix KV: (prefix_len, num_key_value_heads, head_dim)
                        prefix_k = mx.stack(
                            prefix_k_list, axis=0
                        )  # (prefix_len, num_key_value_heads, head_dim)
                        prefix_v = mx.stack(
                            prefix_v_list, axis=0
                        )  # (prefix_len, num_key_value_heads, head_dim)

                        # Reshape and transpose for attention
                        prefix_k = prefix_k.transpose(1, 0, 2)[
                            None, ...
                        ]  # (1, num_key_value_heads, prefix_len, head_dim)
                        prefix_v = prefix_v.transpose(1, 0, 2)[
                            None, ...
                        ]  # (1, num_key_value_heads, prefix_len, head_dim)

                        # Concatenate prefix and new KV
                        k_full = mx.concatenate(
                            [prefix_k, k_new_i], axis=2
                        )  # (1, num_key_value_heads, prefix_len + target_len, head_dim)
                        v_full = mx.concatenate(
                            [prefix_v, v_new_i], axis=2
                        )  # (1, num_key_value_heads, prefix_len + target_len, head_dim)
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

                    # Apply window_size mask if needed
                    if window_size is not None:
                        # For prefix_cache case with window_size:
                        # - Can attend to all prefix tokens (col < prefix_len)
                        # - For new tokens, apply sliding window: can attend to positions >= (actual_row_pos - window_size)
                        # Create window mask: (target_len, full_len)
                        row_positions = (
                            mx.arange(target_len, dtype=mx.int32)[:, None] + prefix_len
                        )  # (target_len, 1)
                        col_positions = mx.arange(full_len, dtype=mx.int32)[
                            None, :
                        ]  # (1, full_len)

                        # All prefix tokens are accessible
                        prefix_mask = col_positions < prefix_len

                        # For new tokens, apply sliding window
                        # TODO: check if this is correct
                        window_start = mx.maximum(
                            prefix_len, row_positions - window_size
                        )  # (target_len, 1)
                        in_window = (col_positions >= window_start) & (
                            col_positions <= row_positions
                        )

                        # Combine: accessible if in prefix OR in window
                        window_mask = prefix_mask | in_window
                        window_mask = mx.where(window_mask, 0.0, float("-inf"))
                        window_mask = window_mask[None, None, :, :]
                        causal_mask = causal_mask + window_mask

                    out_i = scaled_dot_product_attention(
                        q_i,
                        k_full,
                        v_full,
                        scale=self.sm_scale,
                        mask=causal_mask,
                        cache=None,
                        sinks=self.sinks,
                    )
                    output_list.append(out_i)

                output = mx.concatenate(output_list, axis=0)
                output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
            else:
                # No prefix cache, use standard self-attention on local data only
                if window_size is not None:
                    mask_prefill = create_causal_mask(target_len, offset=0, window_size=window_size)
                    mask_prefill = (1 - mask_prefill) * -1e9
                    if mask is not None:
                        mask = mask + mask_prefill
                    else:
                        mask = mask_prefill

                if mask is not None:
                    mask = mask.astype(queries_rotated.dtype)

                output = scaled_dot_product_attention(
                    queries_rotated,
                    keys_rotated,
                    values_new.transpose(0, 2, 1, 3),
                    scale=self.sm_scale,
                    mask=mask,
                    cache=None,
                    sinks=self.sinks,
                )
                output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)

        return self.o_proj(output)


class ParallaxGPTOSSBlock(MLXGPTOSSBlock):
    """A custom transformer block for Parallax, extending the GptOss Block class.
    This version handles the KV cache explicitly and returns new K and V states.
    """

    def __init__(self, args: ModelArgs, layer_idx: int, local_layer_idx: int):
        super().__init__(args)
        self.self_attn = ParallaxGPTOSSAttention(args)
        self.sliding_window = args.sliding_window
        self.layer_idx = layer_idx
        self.local_layer_idx = local_layer_idx
        if args.layer_types:
            self.layer_type = args.layer_types[layer_idx]
        else:
            self.layer_type = "sliding_attention" if layer_idx % 2 == 0 else "full_attention"

    def get_window_size(self):
        return self.sliding_window - 1

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
        # Determine window size for this layer
        if self.layer_type == "sliding_attention":
            window_size = self.get_window_size()
        else:
            window_size = None

        r = self.self_attn(
            self.input_layernorm(x),
            mask=mask,
            cache=cache[self.local_layer_idx],
            block_tables=block_tables,
            context_lengths=context_lengths,
            slot_mapping=slot_mapping,
            window_size=window_size,
            **kwargs,
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "GptOssForCausalLM"


EntryClass = ParallaxGPTOSSBlock
