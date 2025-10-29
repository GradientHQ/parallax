from typing import Optional, Tuple

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.glm4_moe import Attention as MLXGLM4MoeAttention
from mlx_lm.models.glm4_moe import DecoderLayer as MLXGLM4MoeBlock
from mlx_lm.models.glm4_moe import ModelArgs


class ParallaxGLM4MoeAttention(MLXGLM4MoeAttention):
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        offset: int = 0,
        lengths: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)

        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values_new = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

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
                final_values_for_attn = mx.concatenate([past_v, values_new], axis=2)
            else:
                raise ValueError("cache was provided but one of k/v was None.")
        else:
            final_keys_for_attn = keys_rotated
            final_values_for_attn = values_new

        output = scaled_dot_product_attention(
            queries_rotated,
            final_keys_for_attn,
            final_values_for_attn,
            scale=self.scale,
            mask=mask,
            cache=None,
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys_rotated, values_new)


class ParallaxGLM4MoeBlock(MLXGLM4MoeBlock):

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__(args, layer_idx)
        self.self_attn = ParallaxGLM4MoeAttention(args)

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
        return "Glm4MoeForCausalLM"


EntryClass = ParallaxGLM4MoeBlock
