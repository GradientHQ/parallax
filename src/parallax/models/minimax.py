# Copyright © 2025 Apple Inc.

from typing import Any, List, Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.minimax import MiniMaxAttention as MLXMiniMaxAttention
from mlx_lm.models.minimax import MiniMaxDecoderLayer as MLXMiniMaxBlock
from mlx_lm.models.minimax import ModelArgs

from parallax.server.cache.base import BaseCache
from parallax.utils.prefix_cache_utils import compute_attention_with_prefix_cache
from parallax_extensions.ops import paged_attention_v1, reshape_and_cache


class ParallaxMiniMaxAttention(MLXMiniMaxAttention):

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
        values = self.v_proj(x)

        if self.use_qk_norm:
            queries_new = self.q_norm(queries_new)
            keys_new = self.k_norm(keys_new)

        queries_new = queries_new.reshape(
            batch, target_len, self.num_attention_heads, -1
        ).transpose(0, 2, 1, 3)
        keys_new = keys_new.reshape(batch, target_len, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )
        values_new = values.reshape(batch, target_len, self.num_key_value_heads, -1)

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

        # 3. Compute Attention
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

        return self.o_proj(output)


class ParallaxMiniMaxBlock(MLXMiniMaxBlock):
    """A custom transformer block for Parallax, extending the MiniMax Block class.
    This version handles the KV cache explicitly and returns new K and V states.
    """

    def __init__(self, args: ModelArgs, layer_idx: int, local_layer_idx: int):
        super().__init__(args)
        self.self_attn = ParallaxMiniMaxAttention(args)
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
        r = self.block_sparse_moe(self.post_attention_layernorm(h))
        out = h + r
        return out

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "MiniMaxM2ForCausalLM"


EntryClass = ParallaxMiniMaxBlock
