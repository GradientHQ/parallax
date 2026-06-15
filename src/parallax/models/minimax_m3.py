from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx_lm.models.base import BaseModelArgs, scaled_dot_product_attention
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU

from parallax.metal.indexer.kernel import store_indexer_cache
from parallax.server.cache.base import BaseCache
from parallax.server.cache.minimax_m3_cache import MiniMaxM3SparseCache
from parallax.utils.prefix_cache_utils import prepare_attention_with_prefix_cache
from parallax_extensions.ops import (
    sparse_block_indexer,
    sparse_paged_attention,
    paged_attention_v1,
    reshape_and_cache,
)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "minimax_m3"
    hidden_size: int = 6144
    intermediate_size: int = 3072
    dense_intermediate_size: int = 12288
    shared_intermediate_size: int = 3072
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: Optional[int] = 128
    num_hidden_layers: int = 60
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000
    rotary_dim: Optional[int] = None
    partial_rotary_factor: float = 0.5
    rope_scaling: Optional[Dict[str, Any]] = None
    max_position_embeddings: int = 1048576
    vocab_size: int = 200064
    tie_word_embeddings: bool = False
    hidden_act: str = "swigluoai"
    swiglu_alpha: float = 1.702
    swiglu_beta: float = 1.0
    swiglu_limit: float = 7.0
    use_qk_norm: bool = True
    use_gemma_norm: bool = True
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    scoring_func: str = "sigmoid"
    use_routing_bias: bool = True
    routed_scaling_factor: float = 2.0
    moe_layer_freq: List[int] = field(default_factory=list)
    mlp_layer_types: Optional[List[str]] = None
    sparse_attention_config: Optional[Dict[str, Any]] = None
    layer_types: Optional[List[str]] = None
    index_n_heads: Optional[int] = None
    index_head_dim: Optional[int] = None
    index_block_size: Optional[int] = None
    index_topk_blocks: Optional[int] = None
    index_local_blocks: Optional[int] = None
    architectures: Optional[List[str]] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rotary_dim is None:
            self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if isinstance(self.rope_scaling, dict) and "type" not in self.rope_scaling:
            self.rope_scaling = dict(self.rope_scaling)
            if "rope_type" in self.rope_scaling:
                self.rope_scaling["type"] = self.rope_scaling["rope_type"]

        if not self.moe_layer_freq:
            if self.mlp_layer_types is not None:
                self.moe_layer_freq = [
                    1 if layer_type == "sparse" else 0 for layer_type in self.mlp_layer_types
                ]
            else:
                self.moe_layer_freq = self._default_sparse_frequency()

        sparse_freq = self._sparse_frequency_from_layer_types()
        if self.sparse_attention_config is None:
            if sparse_freq is None:
                sparse_freq = self._default_sparse_frequency()
            self.sparse_attention_config = {
                "use_sparse_attention": True,
                "sparse_index_dim": self.index_head_dim or 128,
                "sparse_num_index_heads": self.index_n_heads or 4,
                "sparse_topk_blocks": self.index_topk_blocks or 16,
                "sparse_block_size": self.index_block_size or 128,
                "sparse_disable_index_value": sparse_freq.copy(),
                "sparse_score_type": "max",
                "sparse_init_block": 0,
                "sparse_local_block": (
                    self.index_local_blocks if self.index_local_blocks is not None else 1
                ),
                "sparse_attention_freq": sparse_freq,
            }
        else:
            self.sparse_attention_config = dict(self.sparse_attention_config)
            if sparse_freq is not None:
                self.sparse_attention_config.setdefault("sparse_attention_freq", sparse_freq)
                self.sparse_attention_config.setdefault("use_sparse_attention", True)
            if self.sparse_attention_config.get("sparse_attention_freq") is None and isinstance(
                self.sparse_attention_config.get("sparse_disable_index_value"), list
            ):
                self.sparse_attention_config["sparse_attention_freq"] = list(
                    self.sparse_attention_config["sparse_disable_index_value"]
                )
                self.sparse_attention_config.setdefault("use_sparse_attention", True)
            self._apply_sparse_aliases()

        self.index_head_dim = self.sparse_attention_config.get("sparse_index_dim")
        self.index_n_heads = self.sparse_attention_config.get("sparse_num_index_heads")

    def _default_sparse_frequency(self) -> List[int]:
        dense_layers = min(3, self.num_hidden_layers)
        return [0] * dense_layers + [1] * (self.num_hidden_layers - dense_layers)

    def _sparse_frequency_from_layer_types(self) -> Optional[List[int]]:
        if self.layer_types is None:
            return None
        return [1 if layer_type == "minimax_m3_sparse" else 0 for layer_type in self.layer_types]

    def _apply_sparse_aliases(self):
        aliases = {
            "sparse_index_dim": self.index_head_dim,
            "sparse_num_index_heads": self.index_n_heads,
            "sparse_topk_blocks": self.index_topk_blocks,
            "sparse_block_size": self.index_block_size,
            "sparse_local_block": self.index_local_blocks,
        }
        for key, value in aliases.items():
            if value is not None and key not in self.sparse_attention_config:
                self.sparse_attention_config[key] = value

    def is_moe_layer(self, layer_idx: int) -> bool:
        if layer_idx >= len(self.moe_layer_freq):
            return True
        return bool(self.moe_layer_freq[layer_idx])

    def has_sparse_index(self, layer_idx: int) -> bool:
        if not self.sparse_attention_config.get("use_sparse_attention", False):
            return False
        freq = self.sparse_attention_config.get("sparse_attention_freq")
        if isinstance(freq, list) and layer_idx < len(freq):
            return bool(freq[layer_idx])
        return False


@mx.compile
def _minimax_moe_select(
    gates: mx.array,
    correction_bias: mx.array,
    k: int,
    routed_scaling_factor: float,
    scoring_func: str,
):
    gates = gates.astype(mx.float32)
    if scoring_func == "sigmoid":
        scores = mx.sigmoid(gates)
    else:
        scores = mx.softmax(gates, axis=-1, precise=True)

    biased_scores = scores + correction_bias
    inds = mx.argpartition(-biased_scores, kth=k - 1, axis=-1)[..., :k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    return inds, weights * routed_scaling_factor


@mx.compile
def _swiglu_oai(x_linear, x_glu, alpha: float, limit: float, beta: float):
    x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
    x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
    return x_glu * mx.sigmoid(alpha * x_glu) * (x_linear + beta)


class MiniMaxSwiGLUOAI(nn.Module):
    def __init__(self, alpha: float = 1.702, limit: float = 7.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.limit = limit
        self.beta = beta

    def __call__(self, x, gate):
        return _swiglu_oai(x, gate, self.alpha, self.limit, self.beta)


class MiniMaxRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6, gemma: bool = True):
        super().__init__()
        self.weight = mx.zeros((dims,)) if gemma else mx.ones((dims,))
        self.eps = eps
        self.gemma = gemma

    def __call__(self, x):
        weight = self.weight + 1 if self.gemma else self.weight
        return mx.fast.rms_norm(x, weight.astype(mx.float32), self.eps).astype(x.dtype)


class MiniMaxMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        alpha: float,
        limit: float,
        beta: float,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = MiniMaxSwiGLUOAI(alpha, limit, beta)

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x), self.gate_proj(x)))


class MiniMaxSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.routed_scaling_factor = args.routed_scaling_factor
        self.scoring_func = args.scoring_func

        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        activation = MiniMaxSwiGLUOAI(args.swiglu_alpha, args.swiglu_limit, args.swiglu_beta)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.intermediate_size,
            args.num_local_experts,
            activation=activation,
        )
        self.shared_experts = (
            MiniMaxMLP(
                args.hidden_size,
                args.shared_intermediate_size,
                args.swiglu_alpha,
                args.swiglu_limit,
                args.swiglu_beta,
                bias=False,
            )
            if args.n_shared_experts
            else None
        )
        self.e_score_correction_bias = (
            mx.zeros((args.num_local_experts,)) if args.use_routing_bias else None
        )
        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        gates = self.gate(x.astype(mx.float32))
        if self.e_score_correction_bias is not None:
            inds, scores = _minimax_moe_select(
                gates,
                self.e_score_correction_bias,
                self.num_experts_per_tok,
                self.routed_scaling_factor,
                self.scoring_func,
            )
            scores = scores.astype(x.dtype)
        else:
            if self.scoring_func == "sigmoid":
                scores = mx.sigmoid(gates)
            else:
                scores = mx.softmax(gates, axis=-1, precise=True)
            inds = mx.argpartition(-scores, kth=self.num_experts_per_tok - 1, axis=-1)[
                ..., : self.num_experts_per_tok
            ]
            scores = mx.take_along_axis(scores, inds, axis=-1)
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
            scores = (scores * self.routed_scaling_factor).astype(x.dtype)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


def _broadcast_mask(mask: mx.array, B: int, L: int, K: int) -> mx.array:
    if mask.ndim == 2:
        if mask.shape == (L, K):
            mask = mask[None, None, :, :]
        elif mask.shape == (B, K):
            mask = mask[:, None, None, :]
        else:
            mask = mask[None, None, :, :]
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]
    if mask.shape[0] == 1 and B != 1:
        mask = mx.broadcast_to(mask, (B, *mask.shape[1:]))
    if mask.shape[-2] == 1 and L != 1:
        mask = mx.broadcast_to(mask, (*mask.shape[:-2], L, mask.shape[-1]))
    return mask


def _mask_to_valid(mask: Optional[mx.array], B: int, L: int, K: int) -> Optional[mx.array]:
    if mask is None or isinstance(mask, str):
        return None
    mask = _broadcast_mask(mask, B, L, K)
    if mask.dtype == mx.bool_:
        return mask
    return mask >= 0


class MiniMaxAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.use_qk_norm = args.use_qk_norm

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        if self.use_qk_norm:
            self.q_norm = MiniMaxRMSNorm(
                self.head_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )
            self.k_norm = MiniMaxRMSNorm(
                self.head_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )

        self.has_sparse_index = args.has_sparse_index(layer_idx)
        if self.has_sparse_index:
            sparse_config = args.sparse_attention_config
            self.sparse_block_size = sparse_config.get("sparse_block_size", 128)
            self.sparse_topk_blocks = sparse_config.get("sparse_topk_blocks", 16)
            self.sparse_init_blocks = sparse_config.get("sparse_init_block", 0)
            self.sparse_local_blocks = sparse_config.get("sparse_local_block", 1)
            self.index_dim = sparse_config.get("sparse_index_dim", self.head_dim)
            self.index_heads = sparse_config.get("sparse_num_index_heads", 4)
            self.index_q_proj = nn.Linear(
                args.hidden_size, self.index_heads * self.index_dim, bias=False
            )
            self.index_k_proj = nn.Linear(args.hidden_size, self.index_dim, bias=False)
            self.index_q_norm = MiniMaxRMSNorm(
                self.index_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )
            self.index_k_norm = MiniMaxRMSNorm(
                self.index_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )

        self.rope = initialize_rope(
            args.rotary_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def _build_sparse_mask(
        self,
        idx_queries: mx.array,
        idx_keys: mx.array,
        q_positions: mx.array,
        mask: Optional[mx.array] = None,
        key_positions: Optional[mx.array] = None,
    ) -> mx.array:
        B, H_idx, L, _ = idx_queries.shape
        total_len = idx_keys.shape[2]
        scores = mx.matmul(
            idx_queries.astype(mx.float32),
            idx_keys.astype(mx.float32).swapaxes(-1, -2),
        )
        scores = scores * self.scale

        if q_positions.ndim == 0:
            qpos = mx.broadcast_to(q_positions.reshape(1, 1), (B, L))
        elif q_positions.ndim == 1:
            if q_positions.shape[0] == B and L == 1:
                qpos = q_positions[:, None]
            else:
                qpos = mx.broadcast_to(q_positions[None, :], (B, L))
        else:
            qpos = q_positions
            if qpos.shape[-1] != L:
                qpos = qpos[:, -L:]

        if key_positions is None:
            kpos = mx.arange(total_len)
            causal = kpos[None, None, :] <= qpos[:, :, None]
        else:
            if key_positions.ndim == 1:
                key_positions = mx.broadcast_to(key_positions[None, :], (B, total_len))
            kpos = key_positions.astype(mx.int32)
            causal = kpos[:, None, :] <= qpos[:, :, None]
        scores = mx.where(causal[:, None], scores, -float("inf"))

        valid = _mask_to_valid(mask, B, L, total_len)
        if valid is not None:
            scores = mx.where(valid, scores, -float("inf"))

        if key_positions is None:
            num_blocks = (total_len + self.sparse_block_size - 1) // self.sparse_block_size
            pad = num_blocks * self.sparse_block_size - total_len
            if pad:
                scores = mx.concatenate(
                    [
                        scores,
                        mx.full((*scores.shape[:-1], pad), -float("inf"), dtype=scores.dtype),
                    ],
                    axis=-1,
                )

            scores_by_block = scores.reshape(B, H_idx, L, num_blocks, self.sparse_block_size)
            block_scores = mx.max(mx.max(scores_by_block, axis=-1), axis=1)
        else:
            max_position = int(mx.max(kpos)) + 1
            num_blocks = max(
                1, (max_position + self.sparse_block_size - 1) // self.sparse_block_size
            )
            blocks = mx.arange(num_blocks)
            key_blocks_for_scores = (kpos // self.sparse_block_size).astype(mx.int32)
            block_members = (
                key_blocks_for_scores[:, None, None, :, None] == blocks[None, None, None, None, :]
            )
            expanded_scores = mx.where(
                block_members,
                scores[..., None],
                mx.full((*scores.shape, num_blocks), -float("inf"), dtype=scores.dtype),
            )
            block_scores = mx.max(mx.max(expanded_scores, axis=3), axis=1)
        block_scores = mx.where(block_scores == block_scores, block_scores, -float("inf"))

        blocks = mx.arange(num_blocks)
        cur_block = qpos // self.sparse_block_size
        causal_block = blocks[None, None, :] <= cur_block[:, :, None]
        selected_scores = mx.where(causal_block, block_scores, -float("inf"))

        if self.sparse_init_blocks > 0:
            init_blocks = blocks[None, None, :] < self.sparse_init_blocks
            selected_scores = mx.where(
                init_blocks & causal_block,
                mx.array(1e30, dtype=selected_scores.dtype),
                selected_scores,
            )
        if self.sparse_local_blocks > 0:
            local_start = mx.maximum(cur_block - self.sparse_local_blocks + 1, 0)
            local_blocks = (blocks[None, None, :] >= local_start[:, :, None]) & causal_block
            selected_scores = mx.where(
                local_blocks,
                mx.array(1e29, dtype=selected_scores.dtype),
                selected_scores,
            )

        topk = min(self.sparse_topk_blocks, num_blocks)
        topk_idx = mx.argpartition(-selected_scores, kth=topk - 1, axis=-1)[..., :topk]
        block_selected = mx.any(topk_idx[..., None] == blocks, axis=-2) & causal_block

        if key_positions is None:
            key_blocks = (kpos // self.sparse_block_size).astype(mx.int32)
            key_blocks = mx.broadcast_to(key_blocks[None, None, :], (B, L, total_len))
        else:
            key_blocks = (kpos // self.sparse_block_size).astype(mx.int32)
            key_blocks = mx.broadcast_to(key_blocks[:, None, :], (B, L, total_len))
        key_selected = mx.take_along_axis(block_selected, key_blocks, axis=-1)
        sparse_mask = key_selected[:, None] & causal[:, None]
        if valid is not None:
            sparse_mask = sparse_mask & valid
        return sparse_mask

    def _build_sparse_block_indices(
        self,
        idx_queries: mx.array,
        idx_keys: mx.array,
        q_positions: mx.array,
    ) -> tuple[mx.array, mx.array]:
        B, H_idx, L, _ = idx_queries.shape
        if L != 1:
            raise ValueError("MiniMax-M3 sparse block indices are only used for decode.")

        total_len = idx_keys.shape[2]
        scores = mx.matmul(
            idx_queries.astype(mx.float32),
            idx_keys.astype(mx.float32).swapaxes(-1, -2),
        )
        scores = scores * self.scale

        if q_positions.ndim == 0:
            qpos = mx.broadcast_to(q_positions.reshape(1, 1), (B, L))
        elif q_positions.ndim == 1:
            qpos = q_positions[:, None] if q_positions.shape[0] == B else q_positions[None, :]
        else:
            qpos = q_positions
            if qpos.shape[-1] != L:
                qpos = qpos[:, -L:]

        kpos = mx.arange(total_len)
        causal = kpos[None, None, :] <= qpos[:, :, None]
        scores = mx.where(causal[:, None], scores, -float("inf"))

        num_blocks = max(1, (total_len + self.sparse_block_size - 1) // self.sparse_block_size)
        pad = num_blocks * self.sparse_block_size - total_len
        if pad:
            scores = mx.concatenate(
                [
                    scores,
                    mx.full((*scores.shape[:-1], pad), -float("inf"), dtype=scores.dtype),
                ],
                axis=-1,
            )

        scores_by_block = scores.reshape(B, H_idx, L, num_blocks, self.sparse_block_size)
        block_scores = mx.max(mx.max(scores_by_block, axis=-1), axis=1)
        block_scores = mx.where(block_scores == block_scores, block_scores, -float("inf"))

        blocks = mx.arange(num_blocks)
        cur_block = qpos // self.sparse_block_size
        causal_block = blocks[None, None, :] <= cur_block[:, :, None]
        selected_scores = mx.where(causal_block, block_scores, -float("inf"))

        if self.sparse_init_blocks > 0:
            init_blocks = blocks[None, None, :] < self.sparse_init_blocks
            selected_scores = mx.where(
                init_blocks & causal_block,
                mx.array(1e30, dtype=selected_scores.dtype),
                selected_scores,
            )
        if self.sparse_local_blocks > 0:
            local_start = mx.maximum(cur_block - self.sparse_local_blocks + 1, 0)
            local_blocks = (blocks[None, None, :] >= local_start[:, :, None]) & causal_block
            selected_scores = mx.where(
                local_blocks,
                mx.array(1e29, dtype=selected_scores.dtype),
                selected_scores,
            )

        topk = max(1, min(self.sparse_topk_blocks, num_blocks))
        topk_idx = mx.argpartition(-selected_scores, kth=topk - 1, axis=-1)[..., :topk]
        topk_scores = mx.take_along_axis(selected_scores, topk_idx, axis=-1)
        topk_valid = topk_scores > mx.array(-float("inf"), dtype=topk_scores.dtype)
        return topk_idx.astype(mx.int32), topk_valid.astype(mx.int32)

    def _build_sparse_token_positions(
        self,
        idx_queries: mx.array,
        idx_keys: mx.array,
        q_positions: mx.array,
    ) -> tuple[mx.array, mx.array]:
        sparse_block_indices, sparse_block_valid = self._build_sparse_block_indices(
            idx_queries,
            idx_keys,
            q_positions,
        )
        return self._sparse_blocks_to_token_positions(
            sparse_block_indices,
            sparse_block_valid,
            q_positions,
        )

    def _sparse_blocks_to_token_positions(
        self,
        sparse_block_indices: mx.array,
        sparse_block_valid: mx.array,
        q_positions: mx.array,
    ) -> tuple[mx.array, mx.array]:
        B = sparse_block_indices.shape[0]
        if q_positions.ndim == 0:
            qpos = mx.broadcast_to(q_positions.reshape(1, 1), (B, 1))
        elif q_positions.ndim == 1:
            qpos = (
                q_positions[:, None]
                if q_positions.shape[0] == B
                else mx.broadcast_to(q_positions[None, -1:], (B, 1))
            )
        else:
            qpos = q_positions[:, -1:]

        invalid_block = mx.full(
            sparse_block_indices.shape,
            2_000_000_000 // max(self.sparse_block_size, 1),
            dtype=mx.int32,
        )
        sparse_block_indices = mx.sort(
            mx.where(sparse_block_valid != 0, sparse_block_indices, invalid_block),
            axis=-1,
        )

        offsets = mx.arange(self.sparse_block_size, dtype=mx.int32)
        token_positions = (
            sparse_block_indices[..., None] * self.sparse_block_size
            + offsets[None, None, None, :]
        )
        token_positions = token_positions.reshape(B, -1).astype(mx.int32)
        token_valid = token_positions <= qpos

        valid_counts = mx.sum(token_valid.astype(mx.int32), axis=-1)
        max_valid_count = max(1, int(mx.max(valid_counts)))
        token_positions = token_positions[:, :max_valid_count]
        token_valid = token_valid[:, :max_valid_count]
        token_positions = mx.where(token_valid, token_positions, mx.zeros_like(token_positions))

        return token_positions, token_valid.astype(mx.int32)

    def _read_prefix_index_keys(
        self,
        idx_keys_new: mx.array,
        cache: MiniMaxM3SparseCache,
        block_tables: mx.array,
        prefix_lens: mx.array,
    ) -> mx.array:
        B = idx_keys_new.shape[0]
        max_prefix_len = int(mx.max(prefix_lens))
        if max_prefix_len <= 0:
            return idx_keys_new

        prefix_idx = mx.zeros(
            (B, cache.index_n_heads, max_prefix_len, self.index_dim),
            dtype=idx_keys_new.dtype,
        )
        for i in range(B):
            prefix_len = int(prefix_lens[i])
            if prefix_len <= 0:
                continue
            idx_i = cache.read_index_k(block_tables[i], prefix_len)
            prefix_idx[i, :, :prefix_len, :] = idx_i
        return mx.concatenate([prefix_idx, idx_keys_new], axis=2)

    def _dense_decode_from_cache(
        self,
        queries: mx.array,
        cache: BaseCache,
        block_tables: mx.array,
        context_lengths: mx.array,
    ) -> mx.array:
        key_cache_global, value_cache_global = cache.get_cache()
        block_size = key_cache_global.shape[3]
        output = paged_attention_v1(
            queries,
            key_cache_global,
            value_cache_global,
            block_tables,
            context_lengths,
            block_size,
            self.scale,
            self.num_key_value_heads,
        )
        return output.transpose(0, 2, 1, 3).reshape(queries.shape[0], 1, -1)

    def _sparse_decode_from_cache_dense(
        self,
        queries: mx.array,
        idx_queries: mx.array,
        cache: MiniMaxM3SparseCache,
        block_tables: mx.array,
        context_lengths: mx.array,
    ) -> mx.array:
        B = queries.shape[0]
        max_len = int(mx.max(context_lengths))
        keys = mx.zeros((B, self.num_key_value_heads, max_len, self.head_dim), dtype=queries.dtype)
        values = mx.zeros(
            (B, self.num_key_value_heads, max_len, cache.head_dim_v), dtype=queries.dtype
        )
        idx_keys = mx.zeros((B, cache.index_n_heads, max_len, self.index_dim), dtype=queries.dtype)

        for i in range(B):
            context_len = int(context_lengths[i])
            k_i, v_i = cache.read_kv(block_tables[i], context_len)
            idx_i = cache.read_index_k(block_tables[i], context_len)
            keys[i, :, :context_len, :] = k_i
            values[i, :, :context_len, :] = v_i
            idx_keys[i, :, :context_len, :] = idx_i

        q_positions = context_lengths - 1
        sparse_mask = self._build_sparse_mask(idx_queries, idx_keys, q_positions)
        valid = mx.arange(max_len)[None, None, None, :] < context_lengths[:, None, None, None]
        sparse_mask = sparse_mask & valid
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=None,
            scale=self.scale,
            mask=sparse_mask,
        )
        return output.transpose(0, 2, 1, 3).reshape(B, 1, -1)

    def _sparse_decode_from_cache(
        self,
        queries: mx.array,
        idx_queries: mx.array,
        cache: MiniMaxM3SparseCache,
        block_tables: mx.array,
        context_lengths: mx.array,
    ) -> mx.array:
        B = queries.shape[0]
        max_len = int(mx.max(context_lengths))

        q_positions = context_lengths - 1
        sparse_block_indices, sparse_block_valid = sparse_block_indexer(
            idx_queries,
            cache.get_indexer_cache(),
            block_tables,
            context_lengths,
            max_len,
            self.sparse_block_size,
            self.sparse_topk_blocks,
            self.sparse_init_blocks,
            self.sparse_local_blocks,
            self.scale,
        )
        token_positions, token_positions_valid = self._sparse_blocks_to_token_positions(
            sparse_block_indices,
            sparse_block_valid,
            q_positions,
        )

        key_cache_global, value_cache_global = cache.get_cache()
        block_size = key_cache_global.shape[3]
        output = sparse_paged_attention(
            queries,
            key_cache_global,
            value_cache_global,
            block_tables,
            context_lengths,
            token_positions,
            token_positions_valid,
            block_size=block_size,
            scale=self.scale,
            num_kv_heads=self.num_key_value_heads,
        )
        return output.transpose(0, 2, 1, 3).reshape(B, 1, -1)

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
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.num_attention_heads, self.head_dim)
        keys = self.k_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)

        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        if L == 1:
            current_pos = context_lengths - 1
        elif prefix_lens is not None:
            current_pos = prefix_lens
        else:
            current_pos = 0

        queries = self.rope(queries.transpose(0, 2, 1, 3), offset=current_pos)
        keys = self.rope(keys.transpose(0, 2, 1, 3), offset=current_pos)

        idx_queries = None
        idx_keys = None
        if self.has_sparse_index:
            idx_queries = self.index_q_proj(x).reshape(B, L, self.index_heads, self.index_dim)
            idx_keys = self.index_k_proj(x).reshape(B, L, 1, self.index_dim)
            idx_queries = self.index_q_norm(idx_queries).transpose(0, 2, 1, 3)
            idx_keys = self.index_k_norm(idx_keys).transpose(0, 2, 1, 3)
            idx_queries = self.rope(idx_queries, offset=current_pos)
            idx_keys = self.rope(idx_keys, offset=current_pos)

        key_cache_global, value_cache_global = cache.get_cache()
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

        if self.has_sparse_index:
            if not isinstance(cache, MiniMaxM3SparseCache):
                raise TypeError("MiniMax-M3 sparse layers require MiniMaxM3SparseCache.")
            store_indexer_cache(
                idx_keys.transpose(0, 2, 1, 3),
                cache.get_indexer_cache(),
                block_tables,
                context_lengths,
                block_size=block_size,
                slot_mapping=slot_mapping,
            )

        if L == 1:
            if self.has_sparse_index:
                output = self._sparse_decode_from_cache(
                    queries, idx_queries, cache, block_tables, context_lengths
                )
            else:
                output = self._dense_decode_from_cache(
                    queries, cache, block_tables, context_lengths
                )
        else:
            has_prefix_cache = prefix_lens is not None and bool(mx.any(prefix_lens > 0))
            if has_prefix_cache:
                if not isinstance(cache, MiniMaxM3SparseCache):
                    raise TypeError("MiniMax-M3 prefix cache requires MiniMaxM3SparseCache.")
                (
                    keys_full,
                    values_full,
                    prefix_mask,
                    q_positions,
                    key_positions,
                ) = prepare_attention_with_prefix_cache(
                    queries,
                    keys,
                    values.transpose(0, 2, 1, 3),
                    cache,
                    block_tables,
                    prefix_lens,
                    L,
                    self.num_key_value_heads,
                    context_lengths=context_lengths,
                )
                attn_mask = prefix_mask
                if self.has_sparse_index:
                    idx_keys_full = self._read_prefix_index_keys(
                        idx_keys,
                        cache,
                        block_tables,
                        prefix_lens,
                    )
                    sparse_key_positions = key_positions
                    if bool(mx.all(prefix_lens == int(mx.max(prefix_lens)))):
                        sparse_key_positions = None
                    attn_mask = self._build_sparse_mask(
                        idx_queries,
                        idx_keys_full,
                        q_positions,
                        prefix_mask,
                        key_positions=sparse_key_positions,
                    )
                output = scaled_dot_product_attention(
                    queries,
                    keys_full,
                    values_full,
                    cache=None,
                    scale=self.scale,
                    mask=attn_mask,
                )
            else:
                attn_mask = mask
                if self.has_sparse_index and L > self.sparse_block_size * self.sparse_topk_blocks:
                    q_positions = mx.arange(L, dtype=mx.int32)
                    attn_mask = self._build_sparse_mask(idx_queries, idx_keys, q_positions, mask)
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values.transpose(0, 2, 1, 3),
                    cache=None,
                    scale=self.scale,
                    mask=attn_mask,
                )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output)

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        n = group.size()
        self.q_proj = shard_linear(self.q_proj, "all-to-sharded", group=group)
        self.k_proj = shard_linear(self.k_proj, "all-to-sharded", group=group)
        self.v_proj = shard_linear(self.v_proj, "all-to-sharded", group=group)
        self.o_proj = shard_linear(self.o_proj, "sharded-to-all", group=group)
        self.num_attention_heads //= n
        self.num_key_value_heads //= n
        if self.has_sparse_index:
            if self.index_heads % n != 0:
                raise ValueError(
                    "MiniMax-M3 sparse index heads must be divisible by tensor parallel size."
                )
            self.index_q_proj = shard_linear(self.index_q_proj, "all-to-sharded", group=group)
            self.index_heads //= n


class ParallaxMiniMaxM3Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int, local_layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.local_layer_idx = local_layer_idx
        self.self_attn = MiniMaxAttention(args, layer_idx)
        self.input_layernorm = MiniMaxRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
        )
        self.post_attention_layernorm = MiniMaxRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
        )
        self.is_moe_layer = args.is_moe_layer(layer_idx)
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxSparseMoeBlock(args)
        else:
            self.mlp = MiniMaxMLP(
                args.hidden_size,
                args.dense_intermediate_size,
                args.swiglu_alpha,
                args.swiglu_limit,
                args.swiglu_beta,
                bias=False,
            )

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
        mlp = self.block_sparse_moe if self.is_moe_layer else self.mlp
        return h + mlp(self.post_attention_layernorm(h))

    def shard(self):
        self.self_attn.shard()
        if self.is_moe_layer:
            group = mx.distributed.init()
            shard_inplace(self.block_sparse_moe.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(self.block_sparse_moe.switch_mlp.up_proj, "all-to-sharded", group=group)
            shard_inplace(
                self.block_sparse_moe.switch_mlp.down_proj,
                "sharded-to-all",
                group=group,
            )
            if self.block_sparse_moe.shared_experts is not None:
                self.block_sparse_moe.shared_experts.gate_proj = shard_linear(
                    self.block_sparse_moe.shared_experts.gate_proj,
                    "all-to-sharded",
                    group=group,
                )
                self.block_sparse_moe.shared_experts.up_proj = shard_linear(
                    self.block_sparse_moe.shared_experts.up_proj,
                    "all-to-sharded",
                    group=group,
                )
                self.block_sparse_moe.shared_experts.down_proj = shard_linear(
                    self.block_sparse_moe.shared_experts.down_proj,
                    "sharded-to-all",
                    group=group,
                )
            self.block_sparse_moe.sharding_group = group

    @classmethod
    def get_architecture(cls):
        return "MiniMaxM3SparseForCausalLM"

    @classmethod
    def make_final_norm(cls, args: ModelArgs):
        return MiniMaxRMSNorm(args.hidden_size, eps=args.rms_norm_eps, gemma=args.use_gemma_norm)


def _pack_uint8_weight(weight: mx.array) -> mx.array:
    if weight.dtype != mx.uint8 or weight.shape[-1] % 4 != 0:
        return weight
    shape = (*weight.shape[:-1], weight.shape[-1] // 4, 4)
    weight = weight.reshape(shape).astype(mx.uint32)
    shifts = mx.array([0, 8, 16, 24], dtype=mx.uint32)
    return mx.sum(weight << shifts, axis=-1)


def _sanitize_moe_weights(weights: Dict[str, mx.array], args: ModelArgs):
    mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

    for layer_idx in range(args.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}.block_sparse_moe"
        for suffix in ("weight", "scales", "biases", "bias"):
            for hf_name, mlx_name in mapping.items():
                expert_keys = [
                    f"{prefix}.experts.{expert}.{hf_name}.{suffix}"
                    for expert in range(args.num_local_experts)
                ]
                if all(key in weights for key in expert_keys):
                    weights[f"{prefix}.switch_mlp.{mlx_name}.{suffix}"] = mx.stack(
                        [weights.pop(key) for key in expert_keys]
                    )


class Model(nn.Module):
    def sanitize(self, weights):
        sanitized_weights = {}
        for key, value in weights.items():
            if key.startswith("language_model."):
                key = key.replace("language_model.", "", 1)
            if key.startswith("model.language_model."):
                key = key.replace("model.language_model.", "model.", 1)
            sanitized_weights[key] = value
        weights.clear()

        scale_keys = {
            key.replace(".weight_scale_inv", ".weight")
            for key in sanitized_weights
            if key.endswith(".weight_scale_inv")
        }
        for weight_key in scale_keys:
            weight = sanitized_weights.get(weight_key)
            if weight is not None:
                sanitized_weights[weight_key] = _pack_uint8_weight(weight)

        for key in list(sanitized_weights):
            if key.endswith(".weight_scale_inv"):
                sanitized_weights[key.replace(".weight_scale_inv", ".scales")] = (
                    sanitized_weights.pop(key)
                )

        _sanitize_moe_weights(sanitized_weights, self.args)
        return sanitized_weights


EntryClass = ParallaxMiniMaxM3Block
