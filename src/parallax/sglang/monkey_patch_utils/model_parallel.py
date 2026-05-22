"""Parallax model-parallel monkey patches for sglang.

Summary:
- ParallaxGroupCoordinator (subclasses sglang.srt.distributed.parallel_state.GroupCoordinator):
    adds pp_start_layer, pp_end_layer, hidden_layers and redefines is_first_rank/is_last_rank to use
    layer ranges.
- monkey_patch_init_model_parallel_group: replaces
    sglang.srt.distributed.parallel_state.init_model_parallel_group to return ParallaxGroupCoordinator.
- monkey_patch_initialize_model_parallel: replaces
    sglang.srt.distributed.parallel_state.initialize_model_parallel and passes PP layer bounds when
    creating pipeline-parallel groups.
- monkey_patch_make_layers: replaces sglang.srt.utils.make_layers; uses
    get_pp_group().pp_start_layer/end_layer to instantiate local layers and PPMissingLayer placeholders
    for non-local layers.

These are minimal, reversible patches to support decentralized per-layer pipeline parallelism. Remove
when upstream sglang provides native support.
"""

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import sglang
import sglang.srt.distributed
import sglang.srt.distributed.parallel_state
import torch
from sglang.srt.distributed import get_world_group
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator as SGLGroupCoordinator,
)
from sglang.srt.utils import (
    LayerFn,
    add_prefix,
    is_npu,
    is_xpu,
)
from torch.distributed import Backend

# from parallax.sglang.monkey_patch.model_runner import ModelRunner as SGLModelRunner

logger = logging.getLogger(__name__)

_is_npu = is_npu()
_is_xpu = is_xpu()
_sgl_initialize_model_parallel = (
    sglang.srt.distributed.parallel_state.initialize_model_parallel
)


class ParallaxGroupCoordinator(SGLGroupCoordinator):
    """
    Parallax GroupCoordinator module.
    pp_start_layer, pp_end_layer, hidden_layers are necessary for decentralized inference.
    Also change the definition of first_rank/last_rank.
    """

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_pynccl: bool,
        use_pymscclpp: bool,
        use_custom_allreduce: bool,
        use_hpu_communicator: bool,
        use_xpu_communicator: bool,
        use_npu_communicator: bool,
        use_torch_symm_mem: bool = False,
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
        gloo_timeout: timedelta = timedelta(seconds=120 * 60),
        recovered_rank: bool = False,
        pp_start_layer: int = 0,
        pp_end_layer: int = 0,
        hidden_layers: int = 0,
    ):
        """Add pp_start_layer, pp_end_layer, hidden_layers for decentralized model"""
        super().__init__(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=torch_distributed_backend,
            use_pynccl=use_pynccl,
            use_pymscclpp=use_pymscclpp,
            use_custom_allreduce=use_custom_allreduce,
            use_hpu_communicator=use_hpu_communicator,
            use_xpu_communicator=use_xpu_communicator,
            use_npu_communicator=use_npu_communicator,
            use_torch_symm_mem_all_reduce=use_torch_symm_mem,
            use_message_queue_broadcaster=use_message_queue_broadcaster,
            group_name=group_name,
            gloo_timeout=gloo_timeout,
            recovered_rank=recovered_rank,
        )
        self.pp_start_layer = pp_start_layer
        self.pp_end_layer = pp_end_layer
        self.hidden_layers = hidden_layers

    @property
    def is_first_rank(self):
        """Return whether the caller is the first process in the group"""
        return self.pp_start_layer == 0

    @property
    def is_last_rank(self):
        """Return whether the caller is the last process in the group"""
        return self.pp_end_layer == self.hidden_layers


def monkey_patch_init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    use_pynccl: Optional[bool] = None,
    use_custom_allreduce: Optional[bool] = None,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
    use_mscclpp_allreduce: Optional[bool] = None,
    use_torch_symm_mem_allreduce: Optional[bool] = None,
    recovered_rank: bool = False,
    pp_start_layer: int = 0,
    pp_end_layer: int = 0,
    hidden_layers: int = 0,
) -> SGLGroupCoordinator:
    """A monkey patch to replace sglang.srt.distributed.parallel_state.init_model_parallel_group"""
    if use_custom_allreduce is None:
        use_custom_allreduce = sglang.srt.distributed.parallel_state._ENABLE_CUSTOM_ALL_REDUCE
    if use_mscclpp_allreduce is None:
        use_mscclpp_allreduce = sglang.srt.distributed.parallel_state._ENABLE_MSCCLPP_ALL_REDUCE
    if use_torch_symm_mem_allreduce is None:
        use_torch_symm_mem_allreduce = (
            sglang.srt.distributed.parallel_state._ENABLE_TORCH_SYMM_MEM_ALL_REDUCE
        )
    return ParallaxGroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=(
            not (_is_npu or _is_xpu or backend == "mooncake")
            if use_pynccl is None
            else use_pynccl
        ),
        use_pymscclpp=use_mscclpp_allreduce,
        use_custom_allreduce=use_custom_allreduce,
        use_torch_symm_mem=use_torch_symm_mem_allreduce,
        use_hpu_communicator=True,
        use_xpu_communicator=True,
        use_npu_communicator=True,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
        recovered_rank=recovered_rank,
        pp_start_layer=pp_start_layer,
        pp_end_layer=pp_end_layer,
        hidden_layers=hidden_layers,
    )


def monkey_patch_initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    attention_data_parallel_size: int = 1,
    attention_context_model_parallel_size: int = 1,
    moe_data_model_parallel_size: int = 1,
    backend: Optional[str] = None,
    duplicate_tp_group: bool = False,
    enable_symm_mem: bool = False,
    recovered_rank: bool = False,
    pp_start_layer: int = 0,
    pp_end_layer: int = 0,
    hidden_layers: int = 0,
) -> None:
    """A monkey patch to replace sglang.srt.distributed.parallel_state.initialize_model_parallel"""
    parallel_state = sglang.srt.distributed.parallel_state
    if any(
        getattr(parallel_state, group_name) is not None
        for group_name in (
            "_TP",
            "_PP",
            "_MOE_EP",
            "_MOE_TP",
            "_ATTN_CP",
            "_ATTN_TP",
            "_MOE_DP",
            "_PDMUX_PREFILL_TP_GROUP",
        )
    ):
        parallel_state.destroy_model_parallel()

    _sgl_initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        attention_data_parallel_size=attention_data_parallel_size,
        attention_context_model_parallel_size=attention_context_model_parallel_size,
        moe_data_model_parallel_size=moe_data_model_parallel_size,
        backend=backend,
        duplicate_tp_group=duplicate_tp_group,
        enable_symm_mem=enable_symm_mem,
        recovered_rank=recovered_rank,
    )

    pp_group = parallel_state._PP
    pp_group.pp_start_layer = pp_start_layer
    pp_group.pp_end_layer = pp_end_layer
    pp_group.hidden_layers = hidden_layers


def monkey_patch_make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    pp_rank: Optional[int] = None,
    pp_size: Optional[int] = None,
    prefix: str = "",
    return_tuple: bool = False,
    offloader_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.nn.ModuleList, int, int]:
    """A monkey patch to replace sglang.srt.utils.make_layers"""
    # circula imports
    from sglang.srt.distributed import get_pp_group
    from sglang.srt.layers.utils import PPMissingLayer
    from sglang.srt.utils.offloader import get_offloader

    assert not pp_size or num_hidden_layers >= pp_size
    start_layer, end_layer = get_pp_group().pp_start_layer, get_pp_group().pp_end_layer

    modules = torch.nn.ModuleList(
        [PPMissingLayer(return_tuple=return_tuple) for _ in range(start_layer)]
        + get_offloader().wrap_modules(
            (
                layer_fn(idx=idx, prefix=add_prefix(idx, prefix))
                for idx in range(start_layer, end_layer)
            ),
            **(offloader_kwargs or {}),
        )
        + [PPMissingLayer(return_tuple=return_tuple) for _ in range(end_layer, num_hidden_layers)]
    )
    if pp_rank is None or pp_size is None:
        return modules
    return modules, start_layer, end_layer


def apply_model_parallel_monkey_patch():
    sglang.srt.distributed.parallel_state.init_model_parallel_group = (
        monkey_patch_init_model_parallel_group
    )
    sglang.srt.distributed.parallel_state.initialize_model_parallel = (
        monkey_patch_initialize_model_parallel
    )
    sglang.srt.distributed.init_model_parallel_group = monkey_patch_init_model_parallel_group
    sglang.srt.distributed.initialize_model_parallel = monkey_patch_initialize_model_parallel
    sglang.srt.utils.make_layers = monkey_patch_make_layers
    import sglang.srt.utils.common as utils_common

    utils_common.make_layers = monkey_patch_make_layers
