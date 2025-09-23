"""
hidden_dimefines the Qwen3 model.
"""

from typing import Optional, Tuple

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask, scaled_dot_product_attention
from mlx_lm.models.gpt_oss import AttentionBlock as MLXGPTOSSAttention
from mlx_lm.models.gpt_oss import ModelArgs
from mlx_lm.models.gpt_oss import TransformerBlock as MLXGPTOSSBlock


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
        offset: int = 0,
        length: Optional[mx.array] = None,
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

        queries_new = queries_new.reshape(
            batch, target_len, self.num_attention_heads, -1
        ).transpose(0, 2, 1, 3)
        keys_new = keys_new.reshape(batch, target_len, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )
        values_new = values_new.reshape(batch, target_len, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries_rotated_list = []
            keys_rotated_list = []
            for i in range(batch):
                individual_offset = int(length[i])
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
                # zeros for sinks are already added to kv cache
                final_keys_for_attn = mx.concatenate([past_k, keys_rotated], axis=2)
                final_values_for_attn = mx.concatenate([past_v, values_new], axis=2)
            else:
                raise ValueError("cache was provided but one of k/v was None.")
        else:
            queries_rotated = queries_new
            keys_rotated = keys_new
            for i in range(batch):
                seq_len = int(length[i])
                q_slice = queries_new[i, :, :seq_len, :]
                k_slice = keys_new[i, :, :seq_len, :]
                q_rotated_slice = self.rope(q_slice)
                k_rotated_slice = self.rope(k_slice)
                queries_rotated[i, :, :seq_len, :] = q_rotated_slice
                keys_rotated[i, :, :seq_len, :] = k_rotated_slice

            zeros = mx.zeros(
                (batch, self.num_key_value_heads, 1, self.head_dim), dtype=keys_rotated.dtype
            )
            final_keys_for_attn = mx.concatenate([zeros, keys_rotated], axis=2)
            final_values_for_attn = mx.concatenate([zeros, values_new], axis=2)

        v_hat = scaled_dot_product_attention(
            queries_rotated,
            final_keys_for_attn,
            final_values_for_attn,
            scale=self.sm_scale,
            mask=mask,
            cache=None,
        )
        v_hat = v_hat.swapaxes(1, 2).reshape(batch, target_len, -1)

        return self.o_proj(v_hat), (keys_rotated, values_new)

    def get_causal_mask(self, x, offset):
        _, L, _ = x.shape

        def _make_mask(L, offset):
            zero = mx.array(0, dtype=x.dtype)
            neginf = mx.array(-mx.inf, dtype=x.dtype)
            mask = mx.where(create_causal_mask(L, offset), zero, neginf)
            mask = mask.reshape(1, 1, L, -1)
            mask = mx.tile(mask, (1, self.num_attention_heads, 1, 1))
            sinks = mx.tile(self.sinks.reshape(1, -1, 1, 1), (1, 1, L, 1))
            mask = mx.concatenate([sinks, mask], axis=-1)
            return mask

        # When training re-create the mask so that gradients flow to the sinks.
        # When L is large then recreate the mask because otherwise it will take
        # a pretty significant chunk of memory.
        if self.training or L > 8:
            self._previous_mask = None
            return _make_mask(L, offset)

        # Create the mask once and try to reuse it. For this reason we round up
        # to the closest multiple of 512 so we can reuse the mask several times.
        length = ((L + offset + 511) // 512) * 512
        if (
            self._previous_mask is None
            or self._previous_mask.shape[-1] < length
            or self._previous_mask.shape[-2] != L
        ):
            self._previous_mask = _make_mask(L, length - L)

        return self._previous_mask[..., : L + offset + 1]

    def get_sliding_window_mask(self, x, offset, window_size):
        _, L, _ = x.shape

        def _make_mask(L, offset):
            zero = mx.array(0, dtype=x.dtype)
            neginf = mx.array(-mx.inf, dtype=x.dtype)
            mask = create_causal_mask(L, offset, window_size)
            mask = mx.where(mask, zero, neginf)
            mask = mask.reshape(1, 1, L, -1)
            mask = mx.tile(mask, (1, self.num_attention_heads, 1, 1))
            sinks = mx.tile(self.sinks.reshape(1, -1, 1, 1), (1, 1, L, 1))
            mask = mx.concatenate([sinks, mask], axis=-1)
            return mask

        # If we are training then simply re-create the mask every time to make
        # sure gradients flow to the sinks.
        #
        # For simplicity also re-create the mask if we have more than 1 query
        # for now.
        if self.training or L > 1:
            self._previous_mask = None
            return _make_mask(L, min(window_size + 1, offset))

        # We are in inference so cache the mask and try to reuse it
        if self._previous_mask is None:
            self._previous_mask = _make_mask(L, window_size + 1)

        return self._previous_mask[..., : min(L + offset + 1, window_size + 2)]

    def get_mask(self, x, offset, window_size):
        if window_size is not None:
            return self.get_sliding_window_mask(x, offset, window_size)
        else:
            return self.get_causal_mask(x, offset)

    def create_pad_mask(self, padlen, L, dtype, offset):
        batch_size = padlen.shape[0]
        positions = mx.arange(L)
        pos_expanded = positions[None, :]
        padlen_expanded = padlen[:, None]
        pad_mask = mx.ones((batch_size, L), dtype=dtype)
        condition = pos_expanded >= (L - padlen_expanded)
        pad_mask = mx.where(condition, 0.0, pad_mask)
        if offset == 0:
            extra_column = mx.ones((batch_size, 1), dtype=dtype)
            pad_mask = mx.concatenate([extra_column, pad_mask], axis=1)
        else:
            extra_column = mx.ones((batch_size, 1), dtype=dtype)
            pad_mask = mx.concatenate([pad_mask, extra_column], axis=1)

        pad_mask = pad_mask.reshape(batch_size, 1, 1, L + 1)

        return pad_mask

    def get_mask_with_pad(self, x, offset, window_size, pad_lengths, mx_length, dtype):
        causal_mask = self.get_mask(x, offset, window_size)
        pad_mask = self.create_pad_mask(pad_lengths, mx_length, dtype, offset)
        pad_mask = (pad_mask - 1) * 1e9
        pad_mask = pad_mask.astype(x.dtype)
        return causal_mask + pad_mask


class ParallaxGPTOSSBlock(MLXGPTOSSBlock):
    """A custom transformer block for Parallax, extending the GptOss Block class.
    This version handles the KV cache explicitly and returns new K and V states.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__(args)
        self.self_attn = ParallaxGPTOSSAttention(args)
        self.sliding_window = args.sliding_window
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
        offset: int = 0,
        lengths: Optional[mx.array] = None,
    ):

        batch, target_len, _ = x.shape
        mx_length = offset if offset != 0 else target_len - 1
        pad_length = mx_length - lengths
        if offset == 0:
            pad_length = pad_length + 1

        if self.layer_type == "sliding_attention":
            if cache is not None:
                past_k, past_v = cache
                new_k, new_v = [], []
                for i in range(batch):
                    if lengths[i] >= self.get_window_size():
                        start_index = (lengths[i] - self.get_window_size()).item()
                        end_index = lengths[i].item()
                        new_k.append(past_k[i][:, start_index:end_index, :])
                        new_v.append(past_v[i][:, start_index:end_index, :])
                        pad_length[i] = 0
                    else:
                        new_k.append(past_k[i][:, : min(offset, self.get_window_size()), :])
                        new_v.append(past_v[i][:, : min(offset, self.get_window_size()), :])
                        pad_length[i] = min(offset, self.get_window_size()) - lengths[i]

                cache = (mx.stack(new_k, axis=0), mx.stack(new_v, axis=0))
                mx_length = min(mx_length, self.get_window_size())
            mask = self.self_attn.get_mask_with_pad(
                x,
                offset,
                window_size=self.get_window_size(),
                pad_lengths=pad_length,
                mx_length=mx_length + 1,
                dtype=x.dtype,
            )

        else:
            mask = self.self_attn.get_mask_with_pad(
                x,
                offset,
                window_size=None,
                pad_lengths=pad_length,
                mx_length=mx_length + 1,
                dtype=x.dtype,
            )
        # add sink token if cache is not None
        if cache is not None:
            past_k, past_v = cache
            batch, n_kv_heads, _, head_dim = past_k.shape
            sink_k = mx.zeros((batch, n_kv_heads, 1, head_dim), dtype=past_k.dtype)
            sink_v = mx.zeros((batch, n_kv_heads, 1, head_dim), dtype=past_v.dtype)
            cache = (
                mx.concatenate([sink_k, past_k], axis=2),
                mx.concatenate([sink_v, past_v], axis=2),
            )

        r, (k_cache, v_cache) = self.self_attn(
            self.input_layernorm(x), mask, cache, offset=offset, length=lengths
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r.reshape(h.shape)
        return out, (k_cache, v_cache)

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "GptOssForCausalLM"


EntryClass = ParallaxGPTOSSBlock
