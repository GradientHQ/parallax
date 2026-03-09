"""
hidden_dimefines the Qwen3 model.
"""

from typing import Any, List, Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

import mlx.core as mx
from mlx.nn.layers.distributed import shard_inplace, shard_linear
from mlx_lm.models.base import create_causal_mask, scaled_dot_product_attention
from mlx_lm.models.gpt_oss import AttentionBlock as MLXGPTOSSAttention
from mlx_lm.models.gpt_oss import ModelArgs
from mlx_lm.models.gpt_oss import TransformerBlock as MLXGPTOSSBlock

from parallax.server.cache.base import BaseCache
from parallax.utils.prefix_cache_utils import compute_attention_with_prefix_cache
from parallax_extensions.ops import paged_attention_v1, reshape_and_cache


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

        if target_len == 1:
            current_pos = context_lengths - 1
        elif prefix_lens is not None:
            current_pos = prefix_lens
        else:
            current_pos = 0
        queries_rotated = self.rope(queries_new, offset=current_pos)
        keys_rotated = self.rope(keys_new, offset=current_pos)

        block_size = key_cache_global.shape[3]

        # Update Paged Cache before attention computation
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
            output = paged_attention_v1(
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
                # Use shared prefix cache handling with batch processing
                k_new = keys_rotated  # (batch, n_kv_heads, target_len, head_dim)
                v_new = values_new.transpose(
                    0, 2, 1, 3
                )  # (batch, n_kv_heads, target_len, head_dim)
                output = compute_attention_with_prefix_cache(
                    queries_rotated,  # (batch, n_heads, target_len, head_dim)
                    k_new,
                    v_new,
                    cache,
                    block_tables,
                    prefix_lens,
                    target_len,
                    self.sm_scale,
                    self.num_key_value_heads,
                    mask=mask,
                    sinks=self.sinks,
                    window_size=window_size,
                )
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

    def shard(self):
        group = mx.distributed.init()
        N = group.size()
        r = group.rank()
        # Shard the self attention
        self.self_attn.q_proj = shard_linear(self.self_attn.q_proj, "all-to-sharded", group=group)
        self.self_attn.k_proj = shard_linear(self.self_attn.k_proj, "all-to-sharded", group=group)
        self.self_attn.v_proj = shard_linear(self.self_attn.v_proj, "all-to-sharded", group=group)
        self.self_attn.o_proj = shard_linear(self.self_attn.o_proj, "sharded-to-all", group=group)
        num_attention_heads = self.self_attn.num_attention_heads // N
        self.self_attn.sinks = self.self_attn.sinks[
            num_attention_heads * r : num_attention_heads * (r + 1)
        ]
        self.self_attn.num_attention_heads = num_attention_heads
        self.self_attn.num_key_value_heads = self.self_attn.num_key_value_heads // N

        # Shard the MLP
        shard_inplace(self.mlp.experts.gate_proj, "all-to-sharded", group=group)
        shard_inplace(self.mlp.experts.up_proj, "all-to-sharded", group=group)
        shard_inplace(self.mlp.experts.down_proj, "sharded-to-all", group=group)
        if r > 0:
            # set the bias to 0 for the down proj on the non-zero ranks so that bias only be added once.
            self.mlp.experts.down_proj.bias = mx.zeros_like(self.mlp.experts.down_proj.bias)

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "GptOssForCausalLM"


EntryClass = ParallaxGPTOSSBlock
