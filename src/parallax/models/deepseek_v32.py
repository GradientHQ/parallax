# Copyright © 2025 Apple Inc.
from typing import Any, List, Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.deepseek_v32 import DeepseekV32Attention as MLXDeepseekV32Attention
from mlx_lm.models.deepseek_v32 import DeepseekV32DecoderLayer as MLXDeepseekV32Block
from mlx_lm.models.deepseek_v32 import Indexer as MLXDeepseekV32Indexer
from mlx_lm.models.deepseek_v32 import ModelArgs

from parallax.metal.indexer.kernel import q_dot_k, store_indexer_cache
from parallax.metal.paged_attention.kernel import paged_attention, reshape_and_cache
from parallax.server.cache.base import BaseCache


class ParallaxDeepSeekV32Indexer(MLXDeepseekV32Indexer):
    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any] = None,
        block_tables: Optional[mx.array] = None,
        context_lengths: Optional[mx.array] = None,
        block_size: int = 1024,
        slot_mapping: Optional[mx.array] = None,
        prefix_lens: Optional[mx.array] = None,
        **kwargs,
    ):
        # Computes top_k indices for attention
        batch, target_len, _ = x.shape
        q = self.wq_b(qr)
        q = q.reshape(batch, target_len, self.n_heads, self.head_dim).swapaxes(1, 2)
        q_pe, q_nope = mx.split(q, [self.rope_head_dim], axis=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k = mx.reshape(k, (batch, 1, target_len, self.head_dim))
        k_pe, k_nope = mx.split(k, [self.rope_head_dim], axis=-1)

        # Compute current_pos for all batches using array operations
        if target_len == 1:
            current_pos = context_lengths - 1
        elif prefix_lens is not None:
            current_pos = prefix_lens
        else:
            current_pos = 0
        q_pe = self.rope(q_pe, offset=current_pos)
        k_pe = self.rope(k_pe, offset=current_pos)
        q = mx.concatenate([q_pe, q_nope], axis=-1)
        k = mx.concatenate([k_pe, k_nope], axis=-1)

        store_indexer_cache(
            k.transpose(0, 2, 1, 3),
            cache,
            block_tables,
            context_lengths,
            block_size=block_size,
            slot_mapping=slot_mapping,
        )

        if target_len == 1:
            topk_list = []
            for i in range(batch):
                current_pos = int(context_lengths[i]) - 1
                if current_pos < self.index_topk:
                    topk_list.append([-1] * self.index_topk)
                else:
                    score = q_dot_k(
                        q[i],
                        k[i],
                        block_size=block_size,
                        block_table=block_tables[i],
                        context_length=context_lengths[i],
                    )  # shape: (n_heads, context_len)
                    score = score[:, None, :]  # shape: (n_heads, 1, context_len)
                    score = mx.maximum(score, 0)
                    weight = self.weights_proj(x[i : i + 1]) * (
                        self.n_heads**-0.5
                    )  # shape: (1, 1, n_heads)
                    weight = (weight * self.softmax_scale).swapaxes(-1, -2)[
                        ..., None
                    ]  # shape: (1, n_heads, 1, 1)
                    score = score * weight.squeeze(0)  # shape: (n_heads, 1, context_len)
                    score = score.sum(axis=0)  # shape: (1, context_len)
                    score = score.squeeze(0)  # shape: (context_len,)
                    topk_indices = mx.argpartition(score, kth=-self.index_topk, axis=-1)[
                        -self.index_topk :
                    ]
                    topk_list.append(topk_indices)
            return mx.array(topk_list)
        else:
            if target_len < self.index_topk:
                return mx.full((batch, target_len, self.index_topk), -1, dtype=mx.int32)
            scores = q @ k.swapaxes(-1, -2)
            scores = mx.maximum(scores, 0)
            weights = self.weights_proj(x) * (self.n_heads**-0.5)
            weights = (weights * self.softmax_scale).swapaxes(-1, -2)[..., None]
            scores = scores * weights
            scores = scores.sum(axis=1)
            if mask is not None:
                scores = mx.where(mask, scores, -float("inf"))
            return mx.argpartition(scores, kth=-self.index_topk, axis=-1)[..., -self.index_topk :]


class ParallaxDeepSeekV32Attention(MLXDeepseekV32Attention):

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.indexer = ParallaxDeepSeekV32Indexer(args)

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
        batch, target_len, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
            qr = None
        else:
            qr = self.q_a_layernorm(self.q_a_proj(x))
            q = self.q_b_proj(qr)

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
        indexer_cache = cache.get_indexer_cache()

        # Compute current_pos for all batches using array operations
        if target_len == 1:
            current_pos = context_lengths - 1
        elif prefix_lens is not None:
            current_pos = prefix_lens
        else:
            current_pos = 0
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

        topk_indices = self.indexer(
            x,
            qr,
            mask,
            cache=indexer_cache,
            block_tables=block_tables,
            context_lengths=context_lengths,
            block_size=block_size,
            slot_mapping=slot_mapping,
            prefix_lens=prefix_lens,
        )

        if target_len == 1:
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
                top_k_indices=topk_indices,
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
                # Use batch processing similar to qwen3, but handle topk_indices separately
                max_prefix_len = int(mx.max(prefix_lens))

                # Prepare new KV in correct shape: (batch, num_heads, target_len, head_dim)
                k_new = keys  # (batch, num_heads, target_len, head_dim)
                v_new = values.transpose(0, 2, 1, 3)  # (batch, num_heads, target_len, head_dim)

                if max_prefix_len > 0:
                    # Initialize prefix KV arrays with zeros for padding
                    head_dim = k_new.shape[-1]
                    prefix_k_batch = mx.zeros(
                        (batch, self.num_heads, max_prefix_len, head_dim), dtype=k_new.dtype
                    )  # (batch, num_heads, max_prefix_len, head_dim)
                    prefix_v_batch = mx.zeros(
                        (batch, self.num_heads, max_prefix_len, head_dim), dtype=v_new.dtype
                    )  # (batch, num_heads, max_prefix_len, head_dim)

                    # Batch read prefix KV for all requests using cache.read_prefix_kv
                    for i in range(batch):
                        prefix_len = int(prefix_lens[i])
                        if prefix_len > 0:
                            block_table_i = block_tables[i]  # (max_blocks,)
                            prefix_k, prefix_v = cache.read_prefix_kv(
                                block_table_i, prefix_len, self.num_heads
                            )
                            # prefix_k: (num_heads, prefix_len, head_dim)
                            # prefix_v: (num_heads, prefix_len, head_dim)
                            prefix_k_batch[i, :, :prefix_len, :] = prefix_k
                            prefix_v_batch[i, :, :prefix_len, :] = prefix_v

                    # Concatenate prefix and new KV: (batch, num_heads, max_prefix_len + target_len, head_dim)
                    k_full = mx.concatenate([prefix_k_batch, k_new], axis=2)
                    v_full = mx.concatenate([prefix_v_batch, v_new], axis=2)
                else:
                    # No prefix cache, use only new KV
                    k_full = k_new
                    v_full = v_new

                # Create batch causal mask
                full_len = k_full.shape[2]  # max_prefix_len + target_len

                # Create mask: (batch, target_len, full_len)
                row_indices = mx.arange(target_len)[None, :, None]  # (1, target_len, 1)
                col_indices = mx.arange(full_len)[None, None, :]  # (1, 1, full_len)
                prefix_lens_expanded = prefix_lens[:, None, None]  # (batch, 1, 1)

                # Initialize mask: all positions are allowed by default
                causal_mask = mx.zeros((batch, target_len, full_len), dtype=queries.dtype)

                # Mask 1: Invalid prefix positions for requests with shorter prefix
                invalid_prefix_mask = mx.logical_and(
                    col_indices >= prefix_lens_expanded, col_indices < max_prefix_len
                )  # (batch, 1, full_len)
                causal_mask = mx.where(
                    invalid_prefix_mask, float("-inf"), causal_mask
                )  # (batch, target_len, full_len)

                # Mask 2: Causal mask for new tokens
                new_token_start = max_prefix_len
                new_token_col_indices = col_indices - new_token_start
                is_new_token_pos = col_indices >= new_token_start
                causal_mask_new = mx.where(
                    mx.logical_and(is_new_token_pos, new_token_col_indices > row_indices),
                    float("-inf"),
                    0.0,
                )
                causal_mask = causal_mask + causal_mask_new  # (batch, target_len, full_len)

                # Reshape mask: (batch, 1, target_len, full_len)
                causal_mask = causal_mask[:, None, :, :].astype(queries.dtype)

                # Apply sparse attention mask if topk_indices is available
                if topk_indices is not None:
                    # Process topk_indices for all batches
                    k_seq = target_len
                    sparse_mask = mx.zeros((batch, target_len, k_seq), dtype=mx.bool_)
                    sparse_mask = mx.put_along_axis(
                        sparse_mask, topk_indices, mx.array(True), axis=-1
                    )
                    all_minus_one = (topk_indices == -1).all(axis=-1, keepdims=True)
                    sparse_mask = mx.where(all_minus_one, True, sparse_mask)

                    # Expand sparse_mask to include prefix: (batch, target_len, max_prefix_len + target_len)
                    # For prefix positions, allow all (True), for new positions, use sparse_mask
                    prefix_sparse_mask = mx.ones(
                        (batch, target_len, max_prefix_len), dtype=mx.bool_
                    )
                    full_sparse_mask = mx.concatenate([prefix_sparse_mask, sparse_mask], axis=2)
                    full_sparse_mask = full_sparse_mask[
                        :, None, :, :
                    ]  # (batch, 1, target_len, full_len)

                    # Combine causal mask with sparse mask
                    causal_mask = mx.where(full_sparse_mask, causal_mask, float("-inf"))

                # Batch compute attention
                output = scaled_dot_product_attention(
                    queries,  # (batch, num_heads, target_len, head_dim)
                    k_full,  # (batch, num_heads, full_len, head_dim)
                    v_full,  # (batch, num_heads, full_len, head_dim)
                    scale=self.scale,
                    mask=causal_mask,
                    cache=None,
                )
                output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
            else:
                # No prefix cache, use standard self-attention on local data only
                if topk_indices is not None:
                    k_seq = target_len
                    sparse_mask = mx.zeros((batch, target_len, k_seq), dtype=mx.bool_)
                    sparse_mask = mx.put_along_axis(
                        sparse_mask, topk_indices, mx.array(True), axis=-1
                    )
                    all_minus_one = (topk_indices == -1).all(axis=-1, keepdims=True)
                    sparse_mask = mx.where(all_minus_one, True, sparse_mask)
                    sparse_mask = sparse_mask[:, None, :, :]
                    if mask is not None:
                        mask = mask + (1 - sparse_mask) * -1e9
                        mask = mask.astype(queries.dtype)
                    else:
                        mask = sparse_mask
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


class ParallaxDeepSeekV32Block(MLXDeepseekV32Block):
    def __init__(self, args: ModelArgs, layer_idx: int, local_layer_idx: int):
        super().__init__(args, layer_idx=layer_idx)
        self.self_attn = ParallaxDeepSeekV32Attention(args)
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
        return "DeepseekV32ForCausalLM"


EntryClass = ParallaxDeepSeekV32Block
