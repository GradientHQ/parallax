"""Qwen3Hybrid model configuration monkey patch"""

import enum

from transformers.utils import logging

logger = logging.get_logger(__name__)


class HybridLayerType(enum.Enum):
    full_attention = "attention"
    swa_attention = "swa_attention"
    linear_attention = "linear_attention"
    mamba2 = "mamba"


def apply_qwen3_next_config_monkey_patch():
    """
    Applies a monkey patch to the Qwen3NextConfig class to correctly handle the `layers_block_type` property.
    This makes it aware of pipeline parallelism overrides and fixes a bug where the property ignores
    the `layers_block_type` list loaded from the model's config.json.
    """
    from sglang.srt.configs.qwen3_next import Qwen3NextConfig

    # Store the original property's getter function
    original_layers_block_type_property = Qwen3NextConfig.layers_block_type

    @property
    def patched_layers_block_type(self):
        """
        A patched property that correctly determines the layer block types.
        1. Checks for a `_layers_block_type_override` attribute for pipeline parallelism.
        2. Checks for the `layers_block_type` attribute set by HuggingFace from config.json.
        3. Falls back to the original sglang logic if neither is found.
        """
        # 1. Our override for pipeline parallelism takes highest priority.
        if hasattr(self, "_layers_block_type_override"):
            return self._layers_block_type_override

        # 2. Respect the 'layers_block_type' from the original config.json if it exists.
        # The original @property on the class hides the instance attribute, so we check __dict__.
        if "layers_block_type" in self.__dict__:
            return self.__dict__["layers_block_type"]

        # 3. Fallback to the original sglang logic (periodic attention).
        if original_layers_block_type_property and hasattr(
            original_layers_block_type_property, "fget"
        ):
            return original_layers_block_type_property.fget(self)

        # Fallback in case the property is somehow not a property object
        return []

    Qwen3NextConfig.layers_block_type = patched_layers_block_type
    logger.debug("Applied monkey patch to Qwen3NextConfig.layers_block_type")
