"""
hidden_dimefines the Qwen3 model.
"""

from typing import Any, List, Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.deepseek_v3 import DeepseekV3Attention as MLXDeepseekV3Attention
from mlx_lm.models.deepseek_v3 import DeepseekV3DecoderLayer as MLXDeepseekV3Block
from mlx_lm.models.deepseek_v3 import ModelArgs

from parallax.server.cache.base import BaseCache
from parallax.server.cache.dsa_cache import DeepSeekSparseCache
from parallax_extensions.ops import mla_paged_attention, reshape_and_cache


class ParallaxDeepSeekV3Attention(MLXDeepseekV3Attention):
    """A custom attention module for Parallax, extending the DeepseekV3 Attention class.

    We apply explicit KV cache handling and passing in `offset` directly from Request.
    This version returns the new K and V states for external caching.
    """

    def _read_mla_batch(
        self,
        cache: DeepSeekSparseCache,
        block_tables: mx.array,
        lengths: mx.array,
        max_len: int,
        dtype: mx.Dtype,
    ) -> tuple[mx.array, mx.array]:
        batch = lengths.shape[0]
        latent = mx.zeros((batch, 1, max_len, self.kv_lora_rank), dtype=dtype)
        rope = mx.zeros((batch, 1, max_len, self.qk_rope_head_dim), dtype=dtype)

        for i in range(batch):
            length = int(lengths[i])
            if length <= 0:
                continue
            latent_i, rope_i = cache.read_prefix_mla(block_tables[i], length)
            latent[i, :, :length, :] = latent_i
            rope[i, :, :length, :] = rope_i
        return latent, rope

    def _mla_attention(
        self,
        q_nope: mx.array,
        q_pe: mx.array,
        kv_latent: mx.array,
        k_pe: mx.array,
        mask: Optional[mx.array],
    ) -> mx.array:
        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = pe_scores + mask.astype(pe_scores.dtype)

        if q_nope.shape[2] == 1:
            q_latent = self.embed_q(q_nope)
            output = scaled_dot_product_attention(
                q_latent,
                kv_latent,
                kv_latent,
                scale=self.scale,
                mask=pe_scores,
                cache=None,
            )
            return self.unembed_out(output)

        k_nope = self.embed_q(kv_latent, transpose=False)
        values = self.unembed_out(kv_latent)
        return scaled_dot_product_attention(
            q_nope,
            k_nope,
            values,
            scale=self.scale,
            mask=pe_scores,
            cache=None,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[BaseCache] = None,
        offset: int = 0,
        lengths: Optional[mx.array] = None,
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
        if not isinstance(cache, DeepSeekSparseCache):
            raise TypeError("ParallaxDeepSeekV3Attention requires DeepSeekSparseCache.")

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(batch, target_len, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(batch, target_len, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)
        kv_latent = kv_latent[:, None, :, :]

        # Compute RoPE offsets using array operations instead of loops
        if target_len == 1:
            # Decode phase: position is context_length - 1
            current_pos = context_lengths - 1
        elif prefix_lens is not None:
            # Prefill phase - start from prefix_len if using prefix cache
            current_pos = prefix_lens
        else:
            # Prefill phase - no prefix cache
            current_pos = 0

        # Apply RoPE to q_pe and k_pe with batch processing
        q_pe = self.rope(q_pe, offset=current_pos)
        k_pe = self.rope(k_pe, offset=current_pos)

        latent_cache, rope_cache = cache.get_cache()

        reshape_and_cache(
            kv_latent.transpose(0, 2, 1, 3),
            k_pe.transpose(0, 2, 1, 3),
            latent_cache,
            rope_cache,
            block_tables,
            context_lengths,
            cache.block_size,
            slot_mapping=slot_mapping,
        )

        if target_len == 1:
            q_latent = self.embed_q(q_nope)
            output = mla_paged_attention(
                q_latent,
                q_pe,
                latent_cache,
                rope_cache,
                block_tables,
                context_lengths,
                cache.block_size,
                self.scale,
            )
            output = self.unembed_out(output)
            output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
        else:
            has_prefix_cache = prefix_lens is not None and bool(mx.any(prefix_lens > 0))

            if has_prefix_cache:
                max_prefix_len = int(mx.max(prefix_lens))
                if max_prefix_len > 0:
                    prefix_latent, prefix_k_pe = self._read_mla_batch(
                        cache,
                        block_tables,
                        prefix_lens,
                        max_prefix_len,
                        kv_latent.dtype,
                    )
                    kv_full = mx.concatenate([prefix_latent, kv_latent], axis=2)
                    k_pe_full = mx.concatenate([prefix_k_pe, k_pe], axis=2)
                else:
                    kv_full = kv_latent
                    k_pe_full = k_pe

                row_indices = mx.arange(target_len, dtype=mx.int32)
                new_lens = context_lengths - prefix_lens
                q_positions = prefix_lens[:, None] + row_indices[None, :]

                if max_prefix_len > 0:
                    prefix_positions = mx.arange(max_prefix_len, dtype=mx.int32)
                    prefix_positions = mx.broadcast_to(
                        prefix_positions[None, :], (batch, max_prefix_len)
                    )
                    prefix_valid = prefix_positions < prefix_lens[:, None]
                else:
                    prefix_positions = mx.zeros((batch, 0), dtype=mx.int32)
                    prefix_valid = mx.zeros((batch, 0), dtype=mx.bool_)

                new_positions = prefix_lens[:, None] + row_indices[None, :]
                new_valid = row_indices[None, :] < new_lens[:, None]
                key_positions = mx.concatenate([prefix_positions, new_positions], axis=1)
                key_valid = mx.concatenate([prefix_valid, new_valid], axis=1)
                valid = key_valid[:, None, :] & new_valid[:, :, None]
                valid = valid & (key_positions[:, None, :] <= q_positions[:, :, None])
                mask = mx.where(valid[:, None, :, :], 0.0, -float("inf")).astype(q_nope.dtype)

                output = self._mla_attention(q_nope, q_pe, kv_full, k_pe_full, mask)
                output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
            else:
                if mask is not None:
                    if mask.ndim == 2:
                        mask = mask[None, None, :, :]
                    elif mask.ndim == 3:
                        mask = mask[:, None, :, :]
                    if mask.dtype == mx.bool_:
                        mask = mx.where(mask, 0.0, -float("inf"))
                    mask = mask.astype(q_nope.dtype)

                output = self._mla_attention(q_nope, q_pe, kv_latent, k_pe, mask)
                output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)

        return self.o_proj(output)


class ParallaxDeepSeekV3Block(MLXDeepseekV3Block):
    """A custom transformer block for Parallax, extending the Qwen3 Block class.
    This version handles the KV cache explicitly and returns new K and V states.
    """

    def __init__(self, args: ModelArgs, layer_idx: int, local_layer_idx: int):
        super().__init__(args, layer_idx=layer_idx)
        self.self_attn = ParallaxDeepSeekV3Attention(args)
        self.layer_idx = layer_idx
        self.local_layer_idx = local_layer_idx

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
        lengths: Optional[mx.array] = None,
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
        return "DeepseekV3ForCausalLM"


EntryClass = ParallaxDeepSeekV3Block
