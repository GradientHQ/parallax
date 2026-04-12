from typing import Any, List, Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.glm4_moe_lite import Glm4MoeLiteAttention as MLXGLM4MoeLiteAttention
from mlx_lm.models.glm4_moe_lite import Glm4MoeLiteDecoderLayer as MLXGLM4MoeLiteBlock
from mlx_lm.models.glm4_moe_lite import ModelArgs

from parallax.metal.paged_attention.kernel import paged_attention, reshape_and_cache
from parallax.server.cache.base import BaseCache
from parallax.utils.prefix_cache_utils import compute_attention_with_prefix_cache


class ParallaxGLM4MoeLiteAttention(MLXGLM4MoeLiteAttention):
    """A custom attention module for Parallax, extending the GLM4 MoE Lite Attention class.

    GLM4 MoE Lite uses Multi-head Latent Attention (MLA) similar to DeepSeek V3, but
    instead of kv_b_proj, it uses embed_q and unembed_out (MultiLinear):
      - embed_q: transforms q_nope from qk_nope_head_dim -> kv_lora_rank (per head)
      - unembed_out: transforms attention output from kv_lora_rank -> v_head_dim (per head)
      - keys = [kv_latent, k_pe] with 1 KV head (MQA-style)
      - values = kv_latent with 1 KV head
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
            x: (batch, target_len, hidden_dim) - Input hidden states.
            mask: (batch, n_q_heads, target_len, source_len)
            cache: BaseCache object containing the layer cache.
            block_tables: (batch, max_blocks) - PagedKV block tables.
            context_lengths: (batch,) - PagedKV sequence lengths.
            slot_mapping: (batch * target_len,) - Flattened slot mapping.
            prefix_lens: (batch,) - Number of prefix tokens already cached.

        Returns:
            output: (batch, target_len, hidden_dim) - Output hidden states.
        """
        batch, target_len, _ = x.shape

        # Q projection (with optional LoRA)
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

        if target_len == 1:
            current_pos = context_lengths - 1
        elif prefix_lens is not None:
            current_pos = prefix_lens
        else:
            current_pos = 0

        q_pe = self.rope(q_pe, offset=current_pos)
        k_pe = self.rope(k_pe, offset=current_pos)

        # Transform q_nope into kv_lora_rank space via embed_q (per-head MultiLinear)
        kv_latent_expanded = mx.expand_dims(kv_latent, axis=1)
        # kv_latent_expanded: (batch, 1, target_len, kv_lora_rank)

        q_nope = self.embed_q(q_nope)
        # q_nope: (batch, num_heads, target_len, kv_lora_rank)

        # Construct queries, keys, values
        queries = mx.concatenate([q_nope, q_pe], axis=-1)
        # queries: (batch, num_heads, target_len, kv_lora_rank + qk_rope_head_dim)

        keys = mx.concatenate([kv_latent_expanded, k_pe], axis=-1)
        # keys: (batch, 1, target_len, kv_lora_rank + qk_rope_head_dim)

        # Values = kv_latent (the non-rope part of keys)
        # For reshape_and_cache, values shape: (batch, target_len, num_kv_heads=1, kv_lora_rank)
        values = mx.expand_dims(kv_latent, axis=2)
        # values: (batch, target_len, 1, kv_lora_rank)

        key_cache_global, value_cache_global = cache.get_cache()
        block_size = key_cache_global.shape[3]

        # Store keys and values in paged cache
        reshape_and_cache(
            keys.transpose(0, 2, 1, 3),  # (batch, target_len, 1, key_head_dim)
            values,  # (batch, target_len, 1, kv_lora_rank)
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
                1,  # num_kv_heads = 1 (MQA via latent attention)
                v_head_dim=self.kv_lora_rank,
            )
            # output: (batch, num_heads, 1, kv_lora_rank)
            output = self.unembed_out(output)
            # output: (batch, num_heads, 1, v_head_dim)
            output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
        else:
            # Prefill phase
            has_prefix_cache = prefix_lens is not None and bool(mx.any(prefix_lens > 0))

            if has_prefix_cache:
                k_new = keys  # (batch, 1, target_len, key_head_dim)
                v_new = values.transpose(0, 2, 1, 3)  # (batch, 1, target_len, kv_lora_rank)
                output = compute_attention_with_prefix_cache(
                    queries,
                    k_new,
                    v_new,
                    cache,
                    block_tables,
                    prefix_lens,
                    target_len,
                    self.scale,
                    1,  # num_kv_heads = 1
                    mask=mask,
                    unembed_out=True,  # Skip reshape, we need to apply unembed_out first
                )
                # output: (batch, num_heads, target_len, kv_lora_rank)
                output = self.unembed_out(output)
                # output: (batch, num_heads, target_len, v_head_dim)
                output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
            else:
                # No prefix cache, standard self-attention
                if mask is not None:
                    mask = mx.array(mask, dtype=queries.dtype)

                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values.transpose(0, 2, 1, 3),  # (batch, 1, target_len, kv_lora_rank)
                    scale=self.scale,
                    mask=mask,
                    cache=None,
                )
                # output: (batch, num_heads, target_len, kv_lora_rank)
                output = self.unembed_out(output)
                # output: (batch, num_heads, target_len, v_head_dim)
                output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)

        return self.o_proj(output)


class ParallaxGLM4MoeLiteBlock(MLXGLM4MoeLiteBlock):
    """A custom transformer block for Parallax, extending GLM4 MoE Lite DecoderLayer."""

    def __init__(self, args: ModelArgs, layer_idx: int, local_layer_idx: int):
        super().__init__(args, layer_idx)
        self.self_attn = ParallaxGLM4MoeLiteAttention(args)
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
        return "Glm4MoeLiteForCausalLM"


EntryClass = ParallaxGLM4MoeLiteBlock
