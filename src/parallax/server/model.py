"""
Defines the ShardedModel class for distributing MLX models across multiple devices.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import BaseModelArgs

from parallax.server.sampling.sampler import Sampler, SamplingBatchInfo
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class VisionConfig:
    """Dynamic configuration for vision models in VLM.

    This class dynamically accepts all parameters from config.json's vision_config,
    making it compatible with different VLM architectures (Qwen-VL, LLaVA, etc.).
    """

    def __init__(self, **kwargs):
        # Set all provided parameters as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Set common defaults if not provided
        if not hasattr(self, "model_type"):
            self.model_type = "clip_vision_model"
        if not hasattr(self, "hidden_size"):
            self.hidden_size = 1024

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "VisionConfig":
        """Create VisionConfig from a dictionary, accepting all parameters."""
        if params is None:
            return None
        return cls(**params)


@dataclass
class InputEmbeddingsOutput:
    """Output from get_input_embeddings method."""

    inputs_embeds: mx.array
    attention_mask: Optional[mx.array] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs_embeds": self.inputs_embeds,
            "attention_mask": self.attention_mask,
        }


class ShardedModel(nn.Module):
    """A general class for MLX sharded model, adapted for Parallax KV cache.

    Assumes self.layers are composed of modules (e.g., TransformerBlocks)
    that internally use ParallaxAttention and their __call__ method returns
    (hidden_state, new_k_for_layer, new_v_for_layer).

    Supports VLM (Vision Language Models) by optionally loading vision_tower
    and multi_modal_projector on the first shard.
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
        # VLM support
        vision_config: Optional[Dict[str, Any]] = None,
        vision_tower_class: Optional[Type[nn.Module]] = None,
        multi_modal_projector_class: Optional[Type[nn.Module]] = None,
        image_token_index: Optional[int] = None,
        vision_feature_layer: Optional[int] = -2,
        vision_feature_select_strategy: str = "default",
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

        # VLM configuration
        self.is_vlm = vision_config is not None and vision_tower_class is not None
        self.vision_config = VisionConfig.from_dict(vision_config) if vision_config else None
        self.image_token_index = image_token_index
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy

        if self.is_first_shard:
            self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
            if has_norm_in:
                self.norm_in = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

            if self.is_vlm:
                logger.info(
                    f"Initializing VLM components: vision_tower ({self.vision_config.model_type})"
                )
                self.vision_tower = vision_tower_class(self.vision_config)
                if multi_modal_projector_class is not None:
                    try:
                        self.multi_modal_projector = multi_modal_projector_class(config)
                    except (TypeError, AttributeError):
                        combined_config = type(
                            "CombinedConfig",
                            (),
                            {
                                "vision_config": self.vision_config,
                                "text_config": config,
                            },
                        )()
                        self.multi_modal_projector = multi_modal_projector_class(combined_config)
                        logger.info("Initialized projector with combined vision+text config")
                else:
                    self.multi_modal_projector = None
                    logger.info(
                        "No separate projector class - projector is integrated into VisionModel"
                    )
            else:
                self.vision_tower = None
                self.multi_modal_projector = None
        else:
            self.embed_tokens = None
            self.norm_in = None
            self.vision_tower = None
            self.multi_modal_projector = None

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

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsOutput:
        """Get input embeddings, optionally with vision features merged in.

        This method handles:
        1. Text-only inputs: Simply embed tokens
        2. VLM inputs: Embed tokens, encode images, and merge vision features

        Args:
            input_ids: (batch, seq_len) Token IDs
            pixel_values: (batch, C, H, W) or (num_patches, C, H, W) Image pixel values
            **kwargs: Additional arguments (e.g., image_grid_thw for Qwen2-VL)

        Returns:
            InputEmbeddingsOutput with merged embeddings
        """
        if not self.is_first_shard:
            raise ValueError("get_input_embeddings should only be called on the first shard")

        inputs_embeds = self.embed_tokens(input_ids)
        if pixel_values is None or not self.is_vlm:
            return InputEmbeddingsOutput(inputs_embeds=inputs_embeds)
        image_features = self._encode_images(pixel_values, **kwargs)
        final_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )

        return InputEmbeddingsOutput(inputs_embeds=final_embeds)

    def _encode_images(
        self,
        pixel_values: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """Encode images through vision tower and projector."""
        if self.vision_tower is None:
            raise ValueError("Vision tower not initialized for this model")

        model_type = getattr(self.vision_config, "model_type", "") if self.vision_config else ""
        is_qwen_vl = "qwen" in model_type.lower() and "vl" in model_type.lower()
        is_moonvit = model_type.lower() == "moonvit"
        uses_grid_thw = is_qwen_vl or is_moonvit

        # Ensure correct dtype
        if hasattr(self.vision_tower, "patch_embed") and hasattr(
            self.vision_tower.patch_embed, "proj"
        ):
            pixel_values = pixel_values.astype(self.vision_tower.patch_embed.proj.weight.dtype)
        else:
            pixel_values = pixel_values.astype(self.dtype)

        if uses_grid_thw and image_grid_thw is not None:
            if is_moonvit:
                # MoonViT (KimiVL) expects NHWC input
                if pixel_values.ndim == 4 and pixel_values.shape[1] in [1, 3, 4]:
                    pixel_values = pixel_values.transpose(0, 2, 3, 1)
                vision_outputs = self.vision_tower(
                    pixel_values, grid_thw=image_grid_thw, output_hidden_states=True
                )
            else:
                # Qwen-VL expects flat patches
                vision_outputs = self.vision_tower(pixel_values, image_grid_thw)

            if isinstance(vision_outputs, tuple):
                selected_features = vision_outputs[0]
            elif isinstance(vision_outputs, list):
                selected_features = vision_outputs
            else:
                selected_features = vision_outputs
        else:
            # CLIP/SigLIP style: NCHW -> NHWC
            if pixel_values.ndim == 4 and pixel_values.shape[1] in [1, 3, 4]:
                pixel_values = pixel_values.transpose(0, 2, 3, 1)

            vision_outputs = self.vision_tower(pixel_values, output_hidden_states=True)

            if isinstance(vision_outputs, tuple):
                if len(vision_outputs) >= 3:
                    hidden_states = vision_outputs[2]
                    if isinstance(self.vision_feature_layer, int):
                        selected_features = hidden_states[self.vision_feature_layer]
                        if self.vision_feature_select_strategy == "default":
                            selected_features = selected_features[:, 1:]
                    else:
                        hs_pool = [hidden_states[idx] for idx in self.vision_feature_layer]
                        if self.vision_feature_select_strategy == "default":
                            hs_pool = [hs[:, 1:] for hs in hs_pool]
                        selected_features = mx.concatenate(hs_pool, axis=-1)
                else:
                    selected_features = vision_outputs[1]
                    if self.vision_feature_select_strategy == "default":
                        selected_features = selected_features[:, 1:]
            else:
                selected_features = vision_outputs

        if self.multi_modal_projector is not None:
            image_features = self.multi_modal_projector(selected_features)
        else:
            image_features = selected_features

        return image_features

    def _merge_input_ids_with_image_features(
        self,
        image_features: mx.array,
        inputs_embeds: mx.array,
        input_ids: mx.array,
    ) -> mx.array:
        """Replace <image> placeholder tokens with actual image feature embeddings."""
        if self.image_token_index is None:
            logger.warning("image_token_index not set, cannot merge image features")
            return inputs_embeds

        batch_size, seq_len, hidden_dim = inputs_embeds.shape
        image_positions = input_ids == self.image_token_index

        if image_features.ndim == 3:
            image_features = image_features.reshape(-1, image_features.shape[-1])

        image_features = image_features.astype(inputs_embeds.dtype)

        batch_outputs = []
        feature_start_idx = 0

        for batch_idx in range(batch_size):
            batch_mask = image_positions[batch_idx]
            num_positions = int(mx.sum(batch_mask).item())

            if num_positions > 0:
                batch_features = image_features[
                    feature_start_idx : feature_start_idx + num_positions
                ]

                if batch_features.shape[0] != num_positions:
                    raise ValueError(
                        f"Image token positions ({num_positions}) does not match "
                        f"image features ({batch_features.shape[0]}) for batch {batch_idx}"
                    )

                cumsum = mx.cumsum(batch_mask.astype(mx.int32))
                feature_indices = mx.where(batch_mask, cumsum - 1, 0)
                gathered_features = batch_features[feature_indices]
                batch_mask_expanded = mx.expand_dims(batch_mask, axis=-1)
                batch_output = mx.where(
                    batch_mask_expanded, gathered_features, inputs_embeds[batch_idx]
                )
                feature_start_idx += num_positions
            else:
                batch_output = inputs_embeds[batch_idx]

            batch_outputs.append(batch_output)

        return mx.stack(batch_outputs, axis=0)

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
        pixel_values: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """
        Forward pass through the sharded model.

        Args:
            h_or_tokens:
                (batch, target_len_padded, D) or (batch, target_len_padded) for prefill,
                (batch, 1, D) or (batch, 1) for decode.
            cache: List of layer caches (KVCache or LinearCache).
                   Legacy mode: (key_cache_global, value_cache_global) tuple.
            mask: Optional causal mask for the current segment.
            block_tables: (batch, max_blocks) for PagedAttention.
            context_lengths: (batch,) for PagedAttention.
            slot_mapping: (total_tokens,) for PagedAttention.
            pixel_values: (batch, C, H, W) or (num_patches, C, H, W) for VLM.
                          Image pixel values to be processed by vision tower.
            inputs_embeds: (batch, seq_len, hidden_dim) Pre-computed embeddings.
                          If provided, skips embedding and vision processing.
            **kwargs: Additional model-specific arguments (e.g., image_grid_thw).

        Returns:
            For last shard: logits (batch, seq_len, vocab_size)
            For other shards: hidden states (batch, seq_len, hidden_dim)
        """
        h = h_or_tokens
        target_len = h.shape[1]

        if self.is_first_shard:
            if inputs_embeds is not None:
                # Use pre-computed embeddings (e.g., from get_input_embeddings)
                h = inputs_embeds
            else:
                if self.embed_tokens is None:
                    raise ValueError("embed_tokens is None for the first shard.")

                # Check if we need to process vision inputs
                if pixel_values is not None and self.is_vlm:
                    # Use get_input_embeddings for VLM processing
                    embed_output = self.get_input_embeddings(h, pixel_values, **kwargs)
                    h = embed_output.inputs_embeds
                else:
                    # Standard text embedding
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
