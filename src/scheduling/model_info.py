"""
ModelInfo class for scheduling.

Abstraction of transformer models containing key architectural configurations
used for FLOPs and I/O estimation. This information is later consumed by
nodes within the scheduling system to make informed layer allocation
and performance estimation decisions.
"""

from dataclasses import dataclass
from typing import Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """
    Abstraction of transformer model architecture for scheduling decisions.
    Assuming all decoder layers have uniform computational and memory requirements
    """

    model_name: str
    mlx_model_name: str
    head_size: int
    hidden_dim: int
    intermediate_dim: int
    num_attention_heads: int
    num_kv_heads: int
    vocab_size: int
    num_layers: int
    ffn_num_projections: int = 3
    num_local_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_intermediate_dim: Optional[int] = None
    tie_embedding: bool = False
    # Default int8
    param_bytes_per_element: float = 1
    mlx_param_bytes_per_element: float = 1
    cache_bytes_per_element: int = 1
    embedding_bytes_per_element: int = 1

    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    head_size_k: int = None
    head_size_v: int = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.qk_nope_head_dim is not None and self.qk_rope_head_dim is not None:
            self.head_size_k = self.qk_nope_head_dim + self.qk_rope_head_dim
        else:
            self.head_size_k = self.head_size
        self.head_size_v = self.head_size

    @property
    def q_dim(self) -> int:
        """Return query head dim."""
        return self.num_attention_heads * self.head_size

    @property
    def v_dim(self) -> int:
        """Return key and value head dim."""
        return self.num_kv_heads * self.head_size_v

    @property
    def k_dim(self) -> int:
        """Return key head dim."""
        return self.num_kv_heads * self.head_size_k

    @property
    def mlx_bit_factor(self) -> float:
        return self.mlx_param_bytes_per_element / self.param_bytes_per_element

    @property
    def embedding_io_bytes(self) -> int:
        """Estimate memory for input_embeddings / or lm_head."""
        return self.embedding_bytes_per_element * self.vocab_size * self.hidden_dim

    @property
    def per_token_per_layer_kv_size(self) -> int:
        """Return bytes per token for KV cache."""
        return self.cache_bytes_per_element * (self.k_dim + self.v_dim)

    def per_layer_kv_cache_size(self, *, batch_size: int = 1, source_seq_len: int = 256) -> int:
        """Return size of KV cache in bytes for given request dimensions."""
        return self.per_token_per_layer_kv_size * batch_size * source_seq_len

    def expected_num_activated_experts(
        self, *, batch_size: int = 1, target_seq_len: int = 1
    ) -> Optional[int]:
        """Return expected number of activated experts for a request size."""
        num_tokens = batch_size * target_seq_len
        if self.num_local_experts is not None and self.num_experts_per_tok is not None:
            return int(
                self.num_local_experts
                * (1 - (1 - self.num_experts_per_tok / self.num_local_experts) ** num_tokens)
            )
        return None

    def decoder_layer_flops(
        self,
        *,
        batch_size: int = 1,
        target_seq_len: int = 1,
        source_seq_len: int = 256,
    ) -> int:
        """
        Estimate FLOPs per decoder layer including attention and FFN components.
        For GEMM: (M, K) @ (K, N) = (M, N), flops would be 2 * M * K * N;
        """
        # Attention
        # Q/O projections: (T, hidden_dim) @ (hidden_dim, hidden_dim)
        qo_flops = 2 * 2 * target_seq_len * self.hidden_dim * self.hidden_dim
        # K/V projections: (T, hidden_dim) @ (hidden_dim, kv_dim)
        kv_flops = 2 * target_seq_len * self.hidden_dim * (self.k_dim + self.v_dim)
        projection_flops = qo_flops + kv_flops

        # 'roof' estimation for GQA
        # score: (T, hidden_dim) @ (hidden_dim, S)
        score_flops = 2 * self.hidden_dim * target_seq_len * source_seq_len
        # weight: (T, S) @ (S, hidden_dim)
        attn_v_flops = 2 * self.hidden_dim * target_seq_len * source_seq_len

        attention_flops = projection_flops + score_flops + attn_v_flops

        # Dense FFN
        ffn_flops = (
            2 * self.ffn_num_projections * target_seq_len * self.hidden_dim * self.intermediate_dim
        )
        # Sparse MoE FFN, if applicable
        expected_experts = self.expected_num_activated_experts(
            batch_size=batch_size, target_seq_len=target_seq_len
        )
        if expected_experts is not None:
            ffn_flops *= expected_experts

        return batch_size * (attention_flops + ffn_flops)

    def decoder_layer_io_bytes(
        self,
        roofline: Optional[bool] = None,
        *,
        batch_size: int = 1,
        target_seq_len: int = 1,
        source_seq_len: int = 256,
    ) -> int:
        """
        Estimate memory per decoder layer in bytes including params and kv cache.

        Args:
            roofline: True if calculation is for roofline io latency, otherwise for param size estimation.
            batch_size: Request batch size
            target_seq_len: Target sequence length (tokens to generate)
            source_seq_len: Source sequence length (prompt tokens)
        """
        # Attention params
        qo_params = self.param_bytes_per_element * self.hidden_dim * self.q_dim * 2
        kv_params = self.param_bytes_per_element * self.hidden_dim * (self.k_dim + self.v_dim)
        attention_params = qo_params + kv_params

        # FFN params
        ffn_params = self.param_bytes_per_element * self.ffn_num_projections * self.hidden_dim
        if self.moe_intermediate_dim is not None:
            ffn_params *= self.moe_intermediate_dim
        else:
            ffn_params *= self.intermediate_dim

        if roofline:
            expected_experts = self.expected_num_activated_experts(
                batch_size=batch_size, target_seq_len=target_seq_len
            )
            if expected_experts is not None:
                ffn_params *= expected_experts
            kv_cache_size = self.per_layer_kv_cache_size(
                batch_size=batch_size, source_seq_len=source_seq_len
            )
        else:
            if self.num_local_experts is not None:
                ffn_params *= self.num_local_experts
            kv_cache_size = 0

        return round(ffn_params + kv_cache_size + attention_params)

    def lm_head_flops(self, target_seq_len: int = 1) -> int:
        """Estimate FLOPs for lm_head (last layer GEMM) for a sequence length."""
        return 2 * target_seq_len * self.hidden_dim * self.vocab_size
