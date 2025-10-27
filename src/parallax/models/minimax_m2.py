# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, scaled_dot_product_attention
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    num_experts_per_tok: int
    num_local_experts: int
    shared_intermediate_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    vocab_size: int
    block_size: int = 256
    tie_word_embeddings: bool = False
    shared_moe_mode: str = "sigmoid"
    full_attn_alpha_factor: float = 3.5565588200778455
    full_attn_beta_factor: float = 1.0
    linear_attn_alpha_factor: float = 3.5565588200778455
    linear_attn_beta_factor: float = 1.0
    mlp_alpha_factor: float = 3.5565588200778455
    mlp_beta_factor: float = 1.0
    layer_types: List[str] = None
    head_dim: Optional[int] = None
    use_qk_norm: bool = True


class MLXMiniMaxM2Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_dim = hidden_size = args.hidden_size

        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = head_dim = (
            hidden_size // args.num_attention_heads if args.head_dim is None else args.head_dim
        )
        self.scale = head_dim**-0.5

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

        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim * self.num_attention_heads, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(head_dim * self.num_key_value_heads, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(head_dim, traditional=False, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output)


class MiniMaxSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok

        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.intermediate_size, args.num_local_experts
        )
        self.e_score_correction_bias = mx.zeros((args.num_local_experts,))

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x.astype(mx.float32))

        scores = mx.sigmoid(gates)
        orig_scores = scores
        scores = scores + self.e_score_correction_bias

        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(orig_scores, inds, axis=-1)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y


class MLXMiniMaxM2Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.mlp_alpha_factor = args.mlp_alpha_factor
        self.mlp_beta_factor = args.mlp_beta_factor

        self.self_attn = MLXMiniMaxM2Attention(args)
        self.attn_alpha_factor = args.full_attn_alpha_factor
        self.attn_beta_factor = args.full_attn_beta_factor

        self.block_sparse_moe = MiniMaxSparseMoeBlock(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = (
            self.input_layernorm(x) * self.attn_alpha_factor
            + self.self_attn(x, mask, cache) * self.attn_beta_factor
        )
        r = (
            self.block_sparse_moe(self.post_attention_layernorm(x)) * self.mlp_alpha_factor
            + r * self.mlp_beta_factor
        )
        return r


class ParallaxMiniMaxAttention(MLXMiniMaxM2Attention):

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        offset: int = 0,
        lengths: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Attention forward pass with explicit KV cache handling.

        Args:
            x: (batch, target_len, hidden_dim) - Input hidden states for the current query segment.
            mask: (batch, n_q_heads, target_len, source_len)
            cache: Optional tuple (past_k, past_v).
                   shape: (batch, n_kv_heads, S_past_padded, head_dim)
            offset: source_len_padded (scalar, used for RoPE calculation).

        Returns:
            output_h: (batch, target_len, hidden_dim) - Output hidden states.
            new_k: (batch, n_kv_heads, target_len, head_dim) - New keys for this segment.
            new_v: (batch, n_kv_heads, target_len, head_dim) - New values for this segment.
        """
        batch, target_len, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(batch, target_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch, target_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(batch, target_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # for batch, rope offset is not correct due to padding in batch
        queries_rotated = self.rope(queries, offset=offset)
        keys_rotated = self.rope(keys, offset=offset)

        if cache is not None:
            past_k, past_v = cache
            if past_k is not None and past_v is not None:
                if past_k.shape[2] != offset:
                    raise ValueError(
                        f"ParallaxAttention: Expected past_k sequence length {past_k.shape[2]} "
                        f"to match RoPE offset {offset} (S_past_padded)."
                    )
                final_keys_for_attn = mx.concatenate([past_k, keys_rotated], axis=2)
                final_values_for_attn = mx.concatenate([past_v, values], axis=2)
            else:
                raise ValueError("cache was provided but one of k/v was None.")
        else:
            final_keys_for_attn = keys_rotated
            final_values_for_attn = values

        output = scaled_dot_product_attention(
            queries_rotated,
            final_keys_for_attn,
            final_values_for_attn,
            scale=self.scale,
            mask=mask,
            cache=None,
        )

        output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
        return self.o_proj(output), (keys_rotated, values)


class ParallaxMiniMaxM2Block(MLXMiniMaxM2Block):
    """A custom transformer block for Parallax, extending the MiniMaxM2 Block class.
    This version handles the KV cache explicitly and returns new K and V states.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__(args)
        self.self_attn = ParallaxMiniMaxAttention(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        offset: int = 0,
        lengths: Optional[mx.array] = None,
    ):
        r, (k_cache, v_cache) = self.self_attn(self.input_layernorm(x), mask, cache, offset=offset)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, (k_cache, v_cache)

    @classmethod
    def get_architecture(cls):
        """Get the architecture name for the block."""
        return "MiniMaxM2CausalLM"


EntryClass = ParallaxMiniMaxM2Block
