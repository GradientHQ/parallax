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

                    # Apply sparse attention mask if topk_indices is available
                    if topk_indices is not None:
                        topk_i = topk_indices[i]  # (target_len, index_topk)
                        k_seq = target_len
                        sparse_mask = mx.zeros((target_len, k_seq), dtype=mx.bool_)
                        sparse_mask = mx.put_along_axis(sparse_mask, topk_i, mx.array(True), axis=-1)
                        all_minus_one = (topk_i == -1).all(axis=-1, keepdims=True)
                        sparse_mask = mx.where(all_minus_one, True, sparse_mask)
                        # Expand sparse_mask to include prefix: (target_len, prefix_len + target_len)
                        # For prefix positions, allow all (True), for new positions, use sparse_mask
                        prefix_sparse_mask = mx.ones((target_len, prefix_len), dtype=mx.bool_)
                        full_sparse_mask = mx.concatenate([prefix_sparse_mask, sparse_mask], axis=1)
                        full_sparse_mask = full_sparse_mask[None, None, :, :]  # (1, 1, target_len, full_len)
                        # Combine causal mask with sparse mask
                        causal_mask = mx.where(full_sparse_mask, causal_mask, float("-inf"))

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
                if topk_indices is not None:
                    k_seq = target_len
                    sparse_mask = mx.zeros((batch, target_len, k_seq), dtype=mx.bool_)
                    sparse_mask = mx.put_along_axis(sparse_mask, topk_indices, mx.array(True), axis=-1)
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
