"""
Monkey patch for vLLM weight loading to skip lm_head weights on non-last pipeline stages.
This is similar to the approach used in sglang monkey patches.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_vllm_patch_applied = False
_is_last_stage = True  # Default to True for safety


def set_vllm_pipeline_stage(is_last_stage: bool):
    """Set whether this is the last pipeline stage."""
    global _is_last_stage
    _is_last_stage = is_last_stage
    logger.debug(f"Set vLLM pipeline stage: is_last_stage={is_last_stage}")


def apply_vllm_weight_loader_patch():
    """
    Apply monkey patch to vLLM's default loader to skip lm_head initialization check
    when not on the last pipeline stage.

    This patch intercepts ValueError exceptions during weight loading and checks if they
    are related to lm_head.weight not being initialized. If this occurs on a non-last
    pipeline stage, the error is suppressed as expected behavior. Otherwise, the error
    is re-raised.
    """
    global _vllm_patch_applied

    if _vllm_patch_applied:
        logger.debug("vLLM weight loader patch already applied, skipping")
        return

    try:
        from vllm.model_executor.model_loader import default_loader

        original_load_weights = default_loader.DefaultModelLoader.load_weights

        def patched_load_weights(self, model: Any, model_config: Any):
            """Patched load_weights that handles lm_head for pipeline parallelism."""
            global _is_last_stage

            try:
                # Call original load_weights
                original_load_weights(self, model, model_config)
            except ValueError as e:
                error_msg = str(e)
                # Check if this is the lm_head initialization error
                if "lm_head.weight" in error_msg and "not initialized from checkpoint" in error_msg:
                    if not _is_last_stage:
                        # Expected behavior for non-last pipeline stages
                        logger.info(
                            "Skipping lm_head.weight initialization check on non-last pipeline stage"
                        )
                        return
                    else:
                        # This is the last stage, lm_head should be initialized
                        logger.error(
                            "lm_head.weight not initialized on last pipeline stage, this is an error"
                        )
                        raise
                else:
                    # Different error, re-raise
                    raise

        # Apply the patch
        default_loader.DefaultModelLoader.load_weights = patched_load_weights
        _vllm_patch_applied = True
        logger.info("Successfully applied vLLM weight loader patch for pipeline parallelism")

    except ImportError as e:
        logger.warning(f"Could not apply vLLM weight loader patch: {e}")
    except Exception as e:
        logger.error(f"Error applying vLLM weight loader patch: {e}")
        raise
