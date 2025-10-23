"""
Monkey patch for attention registry to fix FlashAttentionBackend issues on SM89.
This patch forces the use of TritonAttnBackend instead of FlashAttentionBackend
for hybrid linear attention models on SM89 architecture.
"""

import logging
from typing import TYPE_CHECKING

from sglang.srt.utils import is_blackwell, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def monkey_patch_attn_backend_wrapper(runner: "ModelRunner", full_attn_backend: "AttentionBackend"):
    """
    Patched wrapper for special models like hybrid GDN to use TritonAttnBackend
    instead of FlashAttentionBackend on SM89 architecture.
    This follows the exact solution provided by the user.
    """
    assert not (
        runner.hybrid_gdn_config is not None and runner.use_mla_backend
    ), "hybrid_gdn can only be used with non-MLA models."

    if cfg := runner.mambaish_config:
        from sglang.srt.layers.attention.fla.utils import check_environments
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            GDNAttnBackend,
            HybridLinearAttnBackend,
            Mamba2AttnBackend,
        )

        check_environments()
        if runner.hybrid_gdn_config is not None:
            if is_blackwell():
                assert (
                    runner.server_args.attention_backend == "triton"
                ), "triton backend is the only supported backend on Blackwell GPUs for hybrid GDN models, use --attention-backend triton to specify the backend."
            if is_npu():
                assert (
                    runner.server_args.attention_backend == "ascend"
                ), "ascend backend is the only supported backend on NPU for hybrid GDN models, use --attention-backend ascend to specify the backend."
            logger.debug(f"Using hybrid linear attention backend for hybrid GDN models.")
            linear_attn_backend = GDNAttnBackend(runner)
        elif runner.mamba2_config is not None:
            linear_attn_backend = Mamba2AttnBackend(runner)
        else:
            raise ValueError("Expected hybrid GDN or NemotronH models, but got unknown model.")
        full_attn_layers = cfg.full_attention_layer_ids

        # For hybrid models, we need to ensure the full_attn_backend is properly configured
        # The full_attn_backend is already created and initialized by the caller
        # We just need to ensure it's the right type for SM89 architecture

        # Check if we need to replace FlashAttentionBackend with TritonAttnBackend for SM89
        if (
            not is_blackwell()
            and not is_npu()
            and hasattr(full_attn_backend, "__class__")
            and "FlashAttentionBackend" in str(full_attn_backend.__class__)
        ):

            # Replace FlashAttentionBackend with TritonAttnBackend for SM89
            from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

            # Apply our triton backend monkey patch first to fix get_v_head_dim issue
            from parallax.sglang.monkey_patch.triton_backend import (
                apply_triton_backend_init_monkey_patch,
            )

            apply_triton_backend_init_monkey_patch()

            logger.debug(
                "Replacing FlashAttentionBackend with TritonAttnBackend for SM89 compatibility"
            )
            full_attn_backend = TritonAttnBackend(runner)

        return HybridLinearAttnBackend(full_attn_backend, linear_attn_backend, full_attn_layers)

    return full_attn_backend


def apply_attention_registry_monkey_patch():
    """Apply the monkey patch to fix SM89 FlashAttentionBackend issues."""
    import sglang.srt.layers.attention.attention_registry as attention_registry

    # Replace the original function with our patched version
    attention_registry.attn_backend_wrapper = monkey_patch_attn_backend_wrapper
    logger.debug("Applied attention registry monkey patch for SM89 FlashAttentionBackend fix")
