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

from parallax.server.cache.base import BaseCache
from parallax.server.cache.dsa_cache import DeepSeekSparseCache
from parallax_extensions.ops import (
    dsa_paged_attention,
    dsa_token_indexer_with_update,
    reshape_and_cache,
    store_indexer_cache,
)
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
        (
            "full"
            if (
                i < first_k_dense_replace
                or (i - first_k_dense_replace) % index_topk_freq == index_skip_topk_offset
            )
            else "shared"
        )
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

        indexer_cache = (
            cache.get_indexer_cache() if isinstance(cache, DeepSeekSparseCache) else cache
        )
        if target_len == 1:
            weights = self.weights_proj(x).squeeze(1)
            weights = weights * (self.n_heads**-0.5 * self.softmax_scale)
            return dsa_token_indexer_with_update(
                q,
                k,
                indexer_cache,
                block_tables,
                context_lengths,
                weights,
                self.index_topk,
                slot_mapping=slot_mapping,
            )
        else:
            if indexer_cache is not None:
                store_indexer_cache(
                    k.transpose(0, 2, 1, 3),
                    indexer_cache,
                    block_tables,
                    context_lengths,
                    block_size=block_size,
                    slot_mapping=slot_mapping,
                )

            has_prefix_cache = (
                isinstance(cache, DeepSeekSparseCache)
                and prefix_lens is not None
                and bool(mx.any(prefix_lens > 0))
            )
            if has_prefix_cache:
                max_prefix_len = int(mx.max(prefix_lens))
                k_full = k
                if max_prefix_len > 0:
                    index_key_heads = cache.index_key_heads
                    prefix_k = mx.zeros(
                        (batch, index_key_heads, max_prefix_len, self.head_dim),
                        dtype=k.dtype,
                    )
                    for i in range(batch):
                        prefix_len = int(prefix_lens[i])
                        if prefix_len <= 0:
                            continue
                        index_k_i = cache.read_index_k(block_tables[i], prefix_len)
                        prefix_k[i, :, :prefix_len, :] = index_k_i
                    k_full = mx.concatenate([prefix_k, k], axis=2)

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
                valid = key_valid[:, None, :] & (
                    key_positions[:, None, :] <= q_positions[:, :, None]
                )
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
        block_tables: Optional[mx.array] = None,
        context_lengths: Optional[mx.array] = None,
        slot_mapping: Optional[mx.array] = None,
        prefix_lens: Optional[mx.array] = None,
        prev_topk: Optional[mx.array] = None,
        **kwargs,
    ):
        batch, target_len, _ = x.shape
        if not isinstance(cache, DeepSeekSparseCache):
            raise TypeError("ParallaxDeepSeekV32Attention requires DeepSeekSparseCache.")

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

        # Compute current_pos for all batches using array operations
        if target_len == 1:
            current_pos = context_lengths - 1
        elif prefix_lens is not None:
            current_pos = prefix_lens
        else:
            current_pos = 0
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

        if self.is_full:
            topk_indices = self.indexer(
                x,
                qr,
                mask,
                cache=cache,
                block_tables=block_tables,
                context_lengths=context_lengths,
                block_size=cache.block_size,
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
            q_latent = self.embed_q(q_nope)
            output = dsa_paged_attention(
                q_latent,
                q_pe,
                latent_cache,
                rope_cache,
                block_tables,
                context_lengths,
                topk_indices,
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

                if topk_indices is not None:
                    full_len = kv_full.shape[2]
                    topk_for_mask = topk_indices
                    if topk_for_mask.ndim == 2:
                        topk_for_mask = topk_for_mask[:, None, :]
                    valid_topk = topk_for_mask >= 0
                    safe_topk = mx.where(valid_topk, topk_for_mask, 0)
                    sparse_mask = mx.zeros((batch, target_len, full_len), dtype=mx.bool_)
                    sparse_mask = mx.put_along_axis(sparse_mask, safe_topk, valid_topk, axis=-1)
                    dense_rows = (topk_for_mask == -1).all(axis=-1, keepdims=True)
                    sparse_mask = mx.where(dense_rows, True, sparse_mask)
                    sparse_mask = mx.where(sparse_mask[:, None, :, :], 0.0, -float("inf")).astype(
                        q_nope.dtype
                    )
                    mask = mask + sparse_mask

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

                if topk_indices is not None:
                    topk_for_mask = topk_indices
                    if topk_for_mask.ndim == 2:
                        topk_for_mask = topk_for_mask[:, None, :]
                    valid_topk = topk_for_mask >= 0
                    safe_topk = mx.where(valid_topk, topk_for_mask, 0)
                    sparse_mask = mx.zeros((batch, target_len, target_len), dtype=mx.bool_)
                    sparse_mask = mx.put_along_axis(sparse_mask, safe_topk, valid_topk, axis=-1)
                    dense_rows = (topk_for_mask == -1).all(axis=-1, keepdims=True)
                    sparse_mask = mx.where(dense_rows, True, sparse_mask)
                    sparse_mask = mx.where(sparse_mask[:, None, :, :], 0.0, -float("inf")).astype(
                        q_nope.dtype
                    )
                    mask = sparse_mask if mask is None else mask + sparse_mask

                output = self._mla_attention(q_nope, q_pe, kv_latent, k_pe, mask)
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
