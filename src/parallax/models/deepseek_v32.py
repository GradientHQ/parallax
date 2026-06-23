# Copyright © 2025 Apple Inc.
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.deepseek_v32 import DeepseekV32Attention as MLXDeepseekV32Attention
from mlx_lm.models.deepseek_v32 import DeepseekV32DecoderLayer as MLXDeepseekV32Block
from mlx_lm.models.deepseek_v32 import Indexer as MLXDeepseekV32Indexer
from mlx_lm.models.deepseek_v32 import ModelArgs
from mlx_lm.models.rope_utils import initialize_rope

from parallax.metal.indexer.kernel import q_dot_k, store_indexer_cache
from parallax.metal.paged_attention.kernel import paged_attention, reshape_and_cache
from parallax.server.cache.base import BaseCache
from parallax.server.cache.dsa_cache import DeepSeekSparseCache
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


def derive_indexer_types(
    num_layers: int,
    index_topk_freq: int = 1,
    indexer_types: Optional[Any] = None,
    first_k_dense_replace: int = 0,
    index_skip_topk_offset: Optional[int] = None,
) -> List[str]:
    """Return per-layer DSA indexer modes.

    "full" layers run the indexer; "shared" layers reuse the previous full
    layer's top-k. The default keeps DeepSeek-V3.2 behavior: every layer is full.
    """
    if indexer_types is not None:
        return list(indexer_types)

    if index_topk_freq <= 1:
        return ["full"] * num_layers

    if index_skip_topk_offset is None:
        index_skip_topk_offset = index_topk_freq - 1

    return [
        "full"
        if (
            i < first_k_dense_replace
            or (i - first_k_dense_replace) % index_topk_freq
            == index_skip_topk_offset
        )
        else "shared"
        for i in range(num_layers)
    ]


GLM_MOE_DSA_DEFAULTS = {
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "moe_layer_freq": 1,
    "index_topk_freq": 1,
    "indexer_types": None,
    "index_skip_topk_offset": None,
    "indexer_rope_traditional": False,
    "indexer_norm_eps": 1e-6,
}

GLM_MOE_DSA_EXTRA_ARG_KEYS = (
    "index_topk_freq",
    "indexer_types",
    "index_skip_topk_offset",
    "indexer_rope_traditional",
    "indexer_norm_eps",
)


def _is_glm_moe_dsa_config(config: dict) -> bool:
    return config.get("model_type") == "glm_moe_dsa"


class ParallaxDeepSeekV32Indexer(MLXDeepseekV32Indexer):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.k_norm = nn.LayerNorm(
            self.head_dim,
            eps=getattr(args, "indexer_norm_eps", 1e-5),
        )
        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=getattr(args, "indexer_rope_traditional", True),
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

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

        indexer_cache = cache.get_indexer_cache() if isinstance(cache, DeepSeekSparseCache) else cache
        if indexer_cache is not None:
            store_indexer_cache(
                k.transpose(0, 2, 1, 3),
                indexer_cache,
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
                        indexer_cache,
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
                    ].astype(mx.int32)
                    topk_list.append(topk_indices)
            return mx.array(topk_list)
        else:
            has_prefix_cache = (
                isinstance(cache, DeepSeekSparseCache)
                and prefix_lens is not None
                and bool(mx.any(prefix_lens > 0))
            )
            if has_prefix_cache:
                max_prefix_len = int(mx.max(prefix_lens))
                k_full = k
                if max_prefix_len > 0:
                    prefix_k = mx.zeros(
                        (batch, self.n_heads, max_prefix_len, self.head_dim),
                        dtype=k.dtype,
                    )
                    for i in range(batch):
                        prefix_len = int(prefix_lens[i])
                        if prefix_len <= 0:
                            continue
                        index_k_i = cache.read_index_k(block_tables[i], prefix_len)
                        prefix_k[i, :, :prefix_len, :] = index_k_i
                    k_local = k
                    if k_local.shape[1] == 1 and self.n_heads > 1:
                        k_local = mx.repeat(k_local, self.n_heads, axis=1)
                    k_full = mx.concatenate([prefix_k, k_local], axis=2)

                full_len = k_full.shape[2]
                if full_len <= self.index_topk:
                    return mx.full((batch, target_len, self.index_topk), -1, dtype=mx.int32)

                scores = q @ k_full.swapaxes(-1, -2)
                scores = mx.maximum(scores, 0)
                weights = self.weights_proj(x) * (self.n_heads**-0.5)
                weights = (weights * self.softmax_scale).swapaxes(-1, -2)[..., None]
                scores = (scores * weights).sum(axis=1)

                row_indices = mx.arange(target_len, dtype=mx.int32)
                prefix_positions = mx.arange(max_prefix_len, dtype=mx.int32)
                prefix_positions = mx.broadcast_to(
                    prefix_positions[None, :], (batch, max_prefix_len)
                )
                prefix_valid = prefix_positions < prefix_lens[:, None]
                new_positions = prefix_lens[:, None] + row_indices[None, :]
                new_lens = context_lengths - prefix_lens
                new_valid = row_indices[None, :] < new_lens[:, None]
                key_positions = mx.concatenate([prefix_positions, new_positions], axis=1)
                key_valid = mx.concatenate([prefix_valid, new_valid], axis=1)
                q_positions = prefix_lens[:, None] + row_indices[None, :]
                valid = key_valid[:, None, :] & (key_positions[:, None, :] <= q_positions[:, :, None])
                valid = valid & new_valid[:, :, None]

                scores = mx.where(valid, scores, -float("inf"))
                topk_indices = mx.argpartition(scores, kth=-self.index_topk, axis=-1)[
                    ..., -self.index_topk :
                ].astype(mx.int32)
                valid_count = valid.sum(axis=-1)
                return mx.where(
                    valid_count[..., None] <= self.index_topk,
                    mx.full(topk_indices.shape, -1, dtype=mx.int32),
                    topk_indices,
                )

            if target_len < self.index_topk:
                return mx.full((batch, target_len, self.index_topk), -1, dtype=mx.int32)
            scores = q @ k.swapaxes(-1, -2)
            scores = mx.maximum(scores, 0)
            weights = self.weights_proj(x) * (self.n_heads**-0.5)
            weights = (weights * self.softmax_scale).swapaxes(-1, -2)[..., None]
            scores = scores * weights
            scores = scores.sum(axis=1)
            if mask is not None:
                if mask.ndim == 4:
                    mask = mask[:, 0, :, :]
                if mask.dtype == mx.bool_:
                    scores = mx.where(mask, scores, -float("inf"))
                else:
                    scores = scores + mask.astype(scores.dtype)
            return mx.argpartition(scores, kth=-self.index_topk, axis=-1)[
                ..., -self.index_topk :
            ].astype(mx.int32)


class ParallaxDeepSeekV32Attention(MLXDeepseekV32Attention):

    def __init__(self, args: ModelArgs, is_full: bool = True):
        super().__init__(args)
        self.is_full = is_full
        self.indexer = ParallaxDeepSeekV32Indexer(args) if is_full else None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[BaseCache] = None,
        block_tables: Optional[mx.array] = None,
        context_lengths: Optional[mx.array] = None,
        slot_mapping: Optional[mx.array] = None,
        prefix_lens: Optional[mx.array] = None,
        prev_topk: Optional[mx.array] = None,
        **kwargs,
    ):
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
        kv_latent = self.kv_a_layernorm(compressed_kv)
        kv_latent = kv_latent[:, None, :, :]
        k_nope = self.embed_q(kv_latent, transpose=False)
        values = self.unembed_out(kv_latent).transpose(0, 2, 1, 3)
        key_cache_global, value_cache_global = cache.get_cache()

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

        if self.is_full:
            topk_indices = self.indexer(
                x,
                qr,
                mask,
                cache=cache,
                block_tables=block_tables,
                context_lengths=context_lengths,
                block_size=block_size,
                slot_mapping=slot_mapping,
                prefix_lens=prefix_lens,
            )
        else:
            if prev_topk is None:
                raise ValueError(
                    "DSA shared layer requires top-k from a previous full layer in the same shard."
                )
            topk_indices = prev_topk

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
                    # topk_indices are in the full prefix+new key space.
                    k_seq = full_len
                    sparse_mask = mx.zeros((batch, target_len, k_seq), dtype=mx.bool_)
                    sparse_mask = mx.put_along_axis(
                        sparse_mask, topk_indices, mx.array(True), axis=-1
                    )
                    all_minus_one = (topk_indices == -1).all(axis=-1, keepdims=True)
                    sparse_mask = mx.where(all_minus_one, True, sparse_mask)
                    full_sparse_mask = sparse_mask[:, None, :, :]

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
        return self.o_proj(output), topk_indices


class ParallaxDeepSeekV32Block(MLXDeepseekV32Block):
    @classmethod
    def prepare_mlx_lm_config(cls, config: dict) -> dict:
        if not _is_glm_moe_dsa_config(config):
            return config

        prepared = dict(config)
        for key, value in GLM_MOE_DSA_DEFAULTS.items():
            prepared.setdefault(key, value)
        return prepared

    @classmethod
    def attach_mlx_lm_model_args(cls, config: dict, model_args: Any) -> None:
        if not _is_glm_moe_dsa_config(config):
            return

        for key in GLM_MOE_DSA_EXTRA_ARG_KEYS:
            setattr(model_args, key, config.get(key, GLM_MOE_DSA_DEFAULTS[key]))

    @classmethod
    def validate_shard_start(cls, config: dict, start_layer: int) -> None:
        if not _is_glm_moe_dsa_config(config) or start_layer == 0:
            return

        indexer_types = derive_indexer_types(
            config.get("num_hidden_layers", 0),
            config.get("index_topk_freq", GLM_MOE_DSA_DEFAULTS["index_topk_freq"]),
            config.get("indexer_types"),
            config.get("first_k_dense_replace", 0),
            config.get("index_skip_topk_offset"),
        )
        if start_layer >= len(indexer_types):
            return

        if indexer_types[start_layer] != "full":
            raise ValueError(
                "GLM DSA shard starts must be layer 0 or a full indexer layer because "
                "Parallax does not transfer DSA top-k across nodes. "
                f"Got start_layer={start_layer} with indexer_type={indexer_types[start_layer]!r}."
            )

    def __init__(self, args: ModelArgs, layer_idx: int, local_layer_idx: int):
        super().__init__(args, layer_idx=layer_idx)
        indexer_types = derive_indexer_types(
            args.num_hidden_layers,
            getattr(args, "index_topk_freq", 1),
            getattr(args, "indexer_types", None),
            getattr(args, "first_k_dense_replace", 0),
            getattr(args, "index_skip_topk_offset", None),
        )
        self.is_full_indexer_layer = indexer_types[layer_idx] == "full"
        self.self_attn = ParallaxDeepSeekV32Attention(
            args,
            is_full=self.is_full_indexer_layer,
        )
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
        prev_topk: Optional[mx.array] = None,
        **kwargs,
    ):

        r, topk = self.self_attn(
            self.input_layernorm(x),
            mask,
            cache[self.local_layer_idx],
            block_tables=block_tables,
            context_lengths=context_lengths,
            slot_mapping=slot_mapping,
            prev_topk=prev_topk,
            **kwargs,
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, topk

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "DeepseekV32ForCausalLM"


EntryClass = ParallaxDeepSeekV32Block
