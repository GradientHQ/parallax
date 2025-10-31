"""Qwen3Hybrid model configuration"""

import enum

from transformers.utils import logging

logger = logging.get_logger(__name__)


# NOTE: HybridLayerType
class HybridLayerType(enum.Enum):
    full_attention = "attention"
    swa_attention = "swa_attention"
    linear_attention = "linear_attention"
    mamba2 = "mamba"


## overwirite due to pipeline parallelism
@property
def monkey_patch_linear_layer_ids(self):
    """Return linear-attention layer ids restricted to the PP slice.

    This is intended to be bound as a property on
    `sglang.srt.configs.qwen3_next.Qwen3NextConfig`.
    """
    lst = [
        i
        for i, type_value in enumerate(self.layers_block_type)
        if type_value == HybridLayerType.linear_attention.value
        and i >= self.start_layer
        and i < self.end_layer
    ]
    ## If no matching layer id, return at least [-1]
    ## It is for memory pool calcuate tokens
    return lst if lst else [-1]


## overwirite due to pipeline parallelism
@property
def monkey_patch_full_attention_layer_ids(self):
    """Return full-attention layer ids restricted to the PP slice.

    This is intended to be bound as a property on
    `sglang.srt.configs.qwen3_next.Qwen3NextConfig`.
    """
    lst = [
        i
        for i, type_value in enumerate(self.layers_block_type)
        if type_value == HybridLayerType.full_attention.value
        and i >= self.start_layer
        and i < self.end_layer
    ]
    ## If no matching layer id, return at least [-1]
    ## It is for memory pool calcuate tokens
    return lst if lst else [-1]


def apply_qwen3_next_config_monkey_patch():
    """Bind monkey-patch helpers to the upstream Qwen3NextConfig class.

    We attach the two helpers above as properties so callers can access
    `config.linear_layer_ids` / `config.full_attention_layer_ids` the same
    way upstream expects.
    """

    import sglang.srt.configs.qwen3_next as s

    s.Qwen3NextConfig.linear_layer_ids = monkey_patch_linear_layer_ids
    s.Qwen3NextConfig.full_attention_layer_ids = monkey_patch_full_attention_layer_ids
