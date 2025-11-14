"""
Monkey patch for vLLM weight loading to skip non-existent weights on different pipeline stages.
This is similar to the approach used in sglang monkey patches.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_vllm_patch_applied = False
_is_first_stage = False  # Default to False
_is_last_stage = True  # Default to True for safety


def set_vllm_pipeline_stage(is_first_stage: bool, is_last_stage: bool):
    """Set whether this is the first and/or last pipeline stage."""
    global _is_first_stage, _is_last_stage
    _is_first_stage = is_first_stage
    _is_last_stage = is_last_stage
    logger.debug(
        f"Set vLLM pipeline stage: is_first_stage={_is_first_stage}, is_last_stage={_is_last_stage}"
    )


def apply_vllm_weight_loader_patch():
    """
    Apply monkey patch to vLLM's default loader to skip initialization checks
    for weights that are not expected on certain pipeline stages.

    - Skips `embed_tokens` check on non-first stages.
    - Skips `lm_head` check on non-last stages.
    """
    global _vllm_patch_applied

    if _vllm_patch_applied:
        logger.debug("vLLM weight loader patch already applied, skipping")
        return

    try:
        from vllm.model_executor.model_loader import default_loader

        original_load_weights = default_loader.DefaultModelLoader.load_weights

        def patched_load_weights(self, model: Any, model_config: Any):
            """Patched load_weights that handles embed_tokens and lm_head for pipeline parallelism."""
            global _is_first_stage, _is_last_stage

            try:
                # Call original load_weights
                original_load_weights(self, model, model_config)
            except ValueError as e:
                error_msg = str(e)
                uninitialized_weights = "not initialized from checkpoint" in error_msg

                # Case 1: embed_tokens.weight not found
                if "model.embed_tokens.weight" in error_msg and uninitialized_weights:
                    if not _is_first_stage:
                        # Expected behavior for non-first pipeline stages
                        logger.info(
                            "Skipping embed_tokens.weight initialization check on non-first pipeline stage"
                        )
                    else:
                        # This is the first stage, embed_tokens should be initialized
                        logger.error(
                            "embed_tokens.weight not initialized on first pipeline stage, this is an error"
                        )
                        raise

                # Case 2: lm_head.weight not found
                elif "lm_head.weight" in error_msg and uninitialized_weights:
                    if not _is_last_stage:
                        # Expected behavior for non-last pipeline stages
                        logger.info(
                            "Skipping lm_head.weight initialization check on non-last pipeline stage"
                        )
                    else:
                        # This is the last stage, lm_head should be initialized
                        logger.error(
                            "lm_head.weight not initialized on last pipeline stage, this is an error"
                        )
                        raise

                # Case 3: Other errors
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
