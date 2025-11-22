"""
Here is some patch func for sglang
Hopefully, when sglang support pipeline parallelism natively, we can remove these patches
"""


def apply_parallax_sglang_monkey_patch():
    from parallax.sglang.monkey_patch_utils.model_parallel import (
        apply_model_parallel_monkey_patch,
    )

    apply_model_parallel_monkey_patch()

    from parallax.sglang.monkey_patch_utils.triton_backend import (
        apply_triton_backend_init_monkey_patch,
    )

    apply_triton_backend_init_monkey_patch()

    from parallax.sglang.monkey_patch_utils.weight_loader_filter import (
        apply_weight_loader_filter_patch,
    )

    apply_weight_loader_filter_patch()

    from parallax.sglang.monkey_patch_utils.qwen3_next_config import (
        apply_qwen3_next_config_monkey_patch,
    )

    apply_qwen3_next_config_monkey_patch()

    from parallax.sglang.monkey_patch_utils.qwen3_next_model import (
        apply_qwen3_next_monkey_patch,
    )

    apply_qwen3_next_monkey_patch()

    from parallax.sglang.monkey_patch_utils.gpt_oss_model import (
        apply_gpt_oss_monkey_patch,
    )

    apply_gpt_oss_monkey_patch()

    from parallax.sglang.monkey_patch_utils.minimax_m2_model import (
        apply_minimax_m2_monkey_patch,
    )

    apply_minimax_m2_monkey_patch()

    from parallax.sglang.monkey_patch_utils.glm4_moe_model import (
        apply_glm4_moe_monkey_patch,
    )

    apply_glm4_moe_monkey_patch()
