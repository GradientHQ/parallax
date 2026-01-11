"""
Defines the ShardedModel class for distributing MLX models across multiple devices.
"""

from typing import Any, List, Optional, Type

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import BaseModelArgs

from parallax.server.sampling.sampler import Sampler, SamplingBatchInfo
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class ShardedModel(nn.Module):
    """A general class for MLX sharded model, adapted for Parallax KV cache.

    Assumes self.layers are composed of modules (e.g., TransformerBlocks)
    that internally use ParallaxAttention and their __call__ method returns
    (hidden_state, new_k_for_layer, new_v_for_layer).
    """

    def __init__(
        self,
        config: BaseModelArgs,
        model_id: str,
        start_layer: int,
        end_layer: int,
        block_class: Type[nn.Module],
        *,
        has_norm_in: bool = False,
        dtype: Optional[mx.Dtype] = None,
    ):
        super().__init__()
        self.config = config
        self.model_id = model_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.block_class = block_class
        self.has_norm_in = has_norm_in
        self.dtype = dtype if dtype is not None else mx.float16

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        self.is_first_shard = start_layer == 0
        self.is_last_shard = end_layer == config.num_hidden_layers
        self.n_layers_in_shard = end_layer - start_layer

        if self.is_first_shard:
            self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
            if has_norm_in:
                self.norm_in = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        else:
            self.embed_tokens = None
            self.norm_in = None

        self.layers = [
            block_class(config, layer_idx, layer_idx - start_layer)
            for layer_idx in range(start_layer, end_layer)
        ]

        if self.is_last_shard:
            self.norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        else:
            self.norm = None
            self.lm_head = None

    def shard_layers(self):
        group = mx.distributed.init()
        tp_size = group.size()
        if tp_size > 1:
            for layer in self.layers:
                if hasattr(layer, "shard"):
                    layer.shard()
                else:
                    logger.error(
                        f"Model {layer.__class__.__name__} does not have a shard method, does not support tensor parallelism"
                    )
                    exit(1)

    def logits_to_tokens(
        self,
        logits: mx.array,
        lengths: Optional[mx.array] = None,
        sampling_info: Optional[SamplingBatchInfo] = None,
    ) -> mx.array:
        """Convert logits to token IDs with greedy decoding.

        Args:
            logits: (batch, target_len_padded, vocab_size), logits from final lm_head
            lengths: (batch,), int array of true lengths
            sampling_info: sampling info of the batched requests

        Return:
            Generated tokens of shape (batch,).
        """
        if logits.ndim != 3:
            raise ValueError(f"Logits must be 3D, but got shape {logits.shape}")

        if lengths is not None:
            # To select the logit vector for the last valid token of each sequence,
            # we need to provide indices for both the batch and sequence dimensions.
            batch_indices = mx.arange(logits.shape[0])
            last_token_indices = lengths - 1
            last_token_logits = logits[batch_indices, last_token_indices, :]
        else:
            # If no lengths are provided, assume all sequences are of max length
            # and we are interested in the very last token's logits.
            last_token_logits = logits[:, -1, :]

        # last_token_logits now has shape (batch_size, vocab_size)
        if sampling_info is None:
            next_token_ids = mx.argmax(last_token_logits, axis=-1)
        else:
            sampler = Sampler()
            next_token_ids = sampler(last_token_logits, sampling_info)
        return next_token_ids

    def __call__(
        self,
        h_or_tokens: mx.array,
        cache: Optional[List[Any]] = None,
        mask: Optional[mx.array] = None,
        block_tables: Optional[mx.array] = None,
        context_lengths: Optional[mx.array] = None,
        slot_mapping: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """
        Args:
            h_or_tokens:
                (batch, target_len_padded, D) or (batch, target_len_padded) for prefill,
                (batch, 1, D) or (batch, 1) for decode.
            cache: List of layer caches (KVCache or LinearCache).
                   Legacy mode: (key_cache_global, value_cache_global) tuple.
            lengths: (batch,) true lengths of each sequence in batch.
            mask: Optional causal mask for the current segment.
            window_size: Optional int, if provided, will use a sliding window attention mask.
            block_tables: (batch, max_blocks) for PagedAttention.
            context_lengths: (batch,) for PagedAttention.
            slot_mapping: (total_tokens,) for PagedAttention.
        """
        h = h_or_tokens
        target_len = h.shape[1]

        if self.is_first_shard:
            if self.embed_tokens is None:
                raise ValueError("embed_tokens is None for the first shard.")
            h = self.embed_tokens(h)
            if self.has_norm_in and self.norm_in:
                h = self.norm_in(h)

        if target_len > 1 and mask is None:
            raise ValueError("ShardedModel: mask cannot be None for prefill.")

        for _, layer_module in enumerate(self.layers):
            h = layer_module(
                h,
                mask=mask,
                cache=cache,
                block_tables=block_tables,
                context_lengths=context_lengths,
                slot_mapping=slot_mapping,
                **kwargs,
            )

        if self.is_last_shard:
            if self.norm is None or self.lm_head is None:
                raise ValueError("ShardedModel: norm or lm_head is None for the last shard.")
            h = self.norm(h)
            h = self.lm_head(h)

        return h
