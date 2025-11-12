"""
Monkey patches for vLLM to support Parallax pipeline parallelism.

This module provides a unified entry point for applying all vLLM-related monkey patches
required for Parallax's distributed inference with pipeline parallelism.
"""

from parallax.vllm.monkey_patch_utils.weight_loader import (
    apply_vllm_weight_loader_patch,
    set_vllm_pipeline_stage,
)


## Here are patch functions for vLLM
## Hopefully, when vLLM supports pipeline parallelism natively in the way we need,
## we can remove these patches
def apply_parallax_vllm_monkey_patch(is_first_stage: bool, is_last_stage: bool):
    """
    Apply all Parallax monkey patches for vLLM.

    Args:
        is_first_stage: Whether this is the first pipeline stage.
        is_last_stage: Whether this is the last pipeline stage. This affects
                      whether lm_head weights are expected to be loaded.
    """
    set_vllm_pipeline_stage(is_first_stage, is_last_stage)
    apply_vllm_weight_loader_patch()
