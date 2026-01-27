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

from parallax.metal.paged_attention.kernel import paged_attention, reshape_and_cache
from parallax.server.cache.base import BaseCache
from parallax.utils.prefix_cache_utils import compute_attention_with_prefix_cache


class ParallaxDeepSeekV3Attention(MLXDeepseekV3Attention):
    """A custom attention module for Parallax, extending the DeepseekV3 Attention class.

    We apply explicit KV cache handling and passing in `offset` directly from Request.
    This version returns the new K and V states for external caching.
    """

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

        k_pe = mx.repeat(k_pe, self.num_heads, axis=1)
        queries = mx.concatenate([q_nope, q_pe], axis=-1)
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
            # Decode phase: Use Paged Attention
            output = paged_attention(
                queries,
                key_cache_global,
                value_cache_global,
                block_tables,
                context_lengths,
                block_size,
                self.scale,
                self.num_heads,
                v_head_dim=values.shape[-1],
            )
            output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
        else:
            # Prefill Phase: Need to attend to both cached prefix and new tokens
            # Check if any request has prefix cache
            has_prefix_cache = prefix_lens is not None and bool(mx.any(prefix_lens > 0))

            if has_prefix_cache:
                # Use shared prefix cache handling with batch processing
                # keys: (batch, num_heads, target_len, head_dim)
                # values: (batch, target_len, num_heads, head_dim) -> transpose to (batch, num_heads, target_len, head_dim)
                k_new = keys  # (batch, num_heads, target_len, head_dim)
                v_new = values.transpose(0, 2, 1, 3)  # (batch, num_heads, target_len, head_dim)
                output = compute_attention_with_prefix_cache(
                    queries,  # (batch, num_heads, target_len, head_dim)
                    k_new,
                    v_new,
                    cache,
                    block_tables,
                    prefix_lens,
                    target_len,
                    self.scale,
                    self.num_heads,  # In deepseek_v3, num_heads equals n_kv_heads after MQA processing
                    mask=mask,
                )
            else:
                # No prefix cache, use standard self-attention on local data only
                if mask is not None:
                    mask = mx.array(mask, dtype=queries.dtype)

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
