"""
hidden_dimefines the Qwen3 model.
"""

from typing import Optional, Tuple

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask, scaled_dot_product_attention
from mlx_lm.models.gpt_oss import AttentionBlock as MLXGPTOSSAttention
from mlx_lm.models.gpt_oss import ModelArgs
from mlx_lm.models.gpt_oss import TransformerBlock as MLXGPTOSSBlock
from parallax.metal.paged_attention.kernel import paged_attention, reshape_and_cache


class ParallaxGPTOSSAttention(MLXGPTOSSAttention):
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
        window_size: Optional[int] = None,
    ) -> mx.array:
        """
        Attention forward pass with PagedAttention integration.
        """
        batch, target_len, _ = x.shape

        print(f"Layer {layer_idx} x min/max: {x.min().item()}, {x.max().item()}")
        queries_new = self.q_proj(x)
        keys_new = self.k_proj(x)
        values_new = self.v_proj(x)
        print(
            f"Layer {layer_idx} queries_new min/max: {queries_new.min().item()}, {queries_new.max().item()}"
        )
        print(
            f"Layer {layer_idx} keys_new min/max: {keys_new.min().item()}, {keys_new.max().item()}"
        )
        print(
            f"Layer {layer_idx} values_new min/max: {values_new.min().item()}, {values_new.max().item()}"
        )

        queries_new = queries_new.reshape(
            batch, target_len, self.num_attention_heads, -1
        ).transpose(0, 2, 1, 3)
        keys_new = keys_new.reshape(batch, target_len, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )
        values_new = values_new.reshape(batch, target_len, self.num_key_value_heads, -1)

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

        # Update Paged Cache
        block_size = key_cache_global.shape[3]

        # print(f"keys_rotated: {keys_rotated}")
        # print(f"values_new: {values_new}")

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

        # Compute Attention
        if target_len == 1:
            output = paged_attention(
                queries_rotated,
                key_cache_global,
                value_cache_global,
                block_tables,
                context_lengths,
                block_size,
                self.sm_scale,
                self.num_key_value_heads,
                layer_idx,
                window_size=window_size,
            )
            output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
            print(f"decode output: {output}")
        else:

            if window_size is not None:
                mask_prefill = create_causal_mask(target_len, offset=0, window_size=window_size)
                # Ensure mask is additive (0, -inf)
                if mask_prefill.max() > 0:
                    # Assume it is (1, 0) or similar, convert to (0, -1e9)
                    mask_prefill = (1 - mask_prefill) * -1e9

                if mask is not None:
                    mask = mask + mask_prefill
                else:
                    mask = mask_prefill

            # Debug Prints
            if mask is not None:
                mx.eval(mask)
                print(f"Layer {layer_idx} Mask min/max: {mask.min().item()}, {mask.max().item()}")

            mx.eval(queries_rotated, keys_rotated, values_new)
            print(
                f"Layer {layer_idx} Q min/max: {queries_rotated.min().item()}, {queries_rotated.max().item()}"
            )
            print(
                f"Layer {layer_idx} K min/max: {keys_rotated.min().item()}, {keys_rotated.max().item()}"
            )
            print(
                f"Layer {layer_idx} V min/max: {values_new.min().item()}, {values_new.max().item()}"
            )

            # print(f"mask: {mask}")
            # print(f"queries_rotated: {queries_rotated}")
            # print(f"keys_rotated: {keys_rotated}")
            # print(f"values_new: {values_new}")

            if mask is not None:
                mask = mask.astype(queries_rotated.dtype)

            output = scaled_dot_product_attention(
                queries_rotated,
                keys_rotated,
                values_new.transpose(0, 2, 1, 3),
                scale=self.sm_scale,
                mask=mask,
                cache=None,
            ).astype(queries_rotated.dtype)

            if mx.isnan(output).any():
                print(f"!!! Layer {layer_idx} Output contains NaN !!!")

            print(f"prefill output: {output}")
            output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)

        return self.o_proj(output)


class ParallaxGPTOSSBlock(MLXGPTOSSBlock):
    """A custom transformer block for Parallax, extending the GptOss Block class.
    This version handles the KV cache explicitly and returns new K and V states.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__(args)
        self.self_attn = ParallaxGPTOSSAttention(args)
        self.mlp = ParallaxMLPBlock(args)
        self.sliding_window = args.sliding_window
        self.layer_idx = layer_idx
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
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        block_tables: Optional[mx.array] = None,
        context_lengths: Optional[mx.array] = None,
        slot_mapping: Optional[mx.array] = None,
    ):
        # Determine window size for this layer
        if self.layer_type == "sliding_attention":
            window_size = self.get_window_size()
        else:
            window_size = None

        r = self.self_attn(
            self.input_layernorm(x),
            mask=mask,
            cache=cache,
            block_tables=block_tables,
            context_lengths=context_lengths,
            slot_mapping=slot_mapping,
            layer_idx=self.layer_idx,
            window_size=window_size,
        )
        h = x + r
        print(f"Layer {self.layer_idx} h min/max: {h.min().item()}, {h.max().item()}")
        r = self.mlp(self.post_attention_layernorm(h))
        print(f"Layer {self.layer_idx} r min/max: {r.min().item()}, {r.max().item()}")
        out = h + r
        print(f"Layer {self.layer_idx} out min/max: {out.min().item()}, {out.max().item()}")
        return out

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "GptOssForCausalLM"


EntryClass = ParallaxGPTOSSBlock
