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
from typing import Any, Dict, List, Optional, Tuple, Union

import sglang
import sglang.srt.distributed.parallel_state
import torch
from sglang.srt.distributed import get_world_group
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator as SGLGroupCoordinator,
)
from sglang.srt.utils import (
    LayerFn,
    add_prefix,
    cpu_has_amx_support,
    get_bool_env_var,
    is_npu,
)
from torch.distributed import Backend

# from parallax.sglang.monkey_patch.model_runner import ModelRunner as SGLModelRunner

logger = logging.getLogger(__name__)

_is_cpu_amx_available = cpu_has_amx_support()


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
    use_custom_allreduce: Optional[bool] = None,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
    use_mscclpp_allreduce: Optional[bool] = None,
    pp_start_layer: int = 0,
    pp_end_layer: int = 0,
    hidden_layers: int = 0,
) -> SGLGroupCoordinator:
    """A monkey patch to replace sglang.srt.distributed.parallel_state.init_model_parallel_group"""
    if use_custom_allreduce is None:
        use_custom_allreduce = sglang.srt.distributed.parallel_state._ENABLE_CUSTOM_ALL_REDUCE
    if use_mscclpp_allreduce is None:
        use_mscclpp_allreduce = sglang.srt.distributed.parallel_state._ENABLE_MSCCLPP_ALL_REDUCE
    return ParallaxGroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=not is_npu(),
        use_pymscclpp=use_mscclpp_allreduce,
        use_custom_allreduce=use_custom_allreduce,
        use_hpu_communicator=True,
        use_xpu_communicator=True,
        use_npu_communicator=True,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
        pp_start_layer=pp_start_layer,
        pp_end_layer=pp_end_layer,
        hidden_layers=hidden_layers,
    )


def monkey_patch_initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
    duplicate_tp_group: bool = False,
    pp_start_layer: int = 0,
    pp_end_layer: int = 0,
    hidden_layers: int = 0,
) -> None:
    """A monkey patch to replace sglang.srt.distributed.parallel_state.initialize_model_parallel"""
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    # Build the tensor model-parallel groups.
    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    ############################################################################
    ## This is a patch code for sgalng
    ## Ignore parallel state already set alert
    # assert (
    #     sglang.srt.distributed.parallel_state._TP is None
    # ), "tensor model parallel group is already initialized"
    ## End of patch
    ############################################################################
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size))
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    sglang.srt.distributed.parallel_state._TP = (
        sglang.srt.distributed.parallel_state.init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=get_bool_env_var(
                "SGLANG_USE_MESSAGE_QUEUE_BROADCASTER", "true"
            ),
            group_name="tp",
        )
    )

    if duplicate_tp_group:
        global _PDMUX_PREFILL_TP_GROUP
        assert (
            _PDMUX_PREFILL_TP_GROUP is None
        ), "tensor model parallel group for PD-Multiplexing Prefill is already initialized"
        _PDMUX_PREFILL_TP_GROUP = sglang.srt.distributed.parallel_state.init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=get_bool_env_var(
                "SGLANG_USE_MESSAGE_QUEUE_BROADCASTER", "true"
            ),
            group_name="pdmux_prefill_tp",
        )
        sglang.srt.distributed.parallel_state._TP.pynccl_comm.disabled = False
        _PDMUX_PREFILL_TP_GROUP.pynccl_comm.disabled = False

    moe_ep_size = expert_model_parallel_size

    moe_tp_size = tensor_model_parallel_size // moe_ep_size
    ############################################################################
    ## This is a patch code for sgalng
    ## Ignore parallel state already set alert
    # assert (
    #     sglang.srt.distributed.parallel_state._MOE_EP is None
    # ), "expert model parallel group is already initialized"
    ## End of patch
    ############################################################################
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        for j in range(moe_tp_size):
            st = i * tensor_model_parallel_size + j
            en = (i + 1) * tensor_model_parallel_size + j
            ranks = list(range(st, en, moe_tp_size))
            group_ranks.append(ranks)

    sglang.srt.distributed.parallel_state._MOE_EP = (
        sglang.srt.distributed.parallel_state.init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_custom_allreduce=False,
            group_name="moe_ep",
        )
    )

    ############################################################################
    ## This is a patch code for sgalng
    ## Ignore parallel state already set alert
    # assert (
    #     sglang.srt.distributed.parallel_state._MOE_TP is None
    # ), "expert model parallel group is already initialized"
    ## End of patch
    ############################################################################
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        for j in range(moe_ep_size):
            st = i * tensor_model_parallel_size + j * moe_tp_size
            en = i * tensor_model_parallel_size + (j + 1) * moe_tp_size
            ranks = list(range(st, en))
            group_ranks.append(ranks)

    sglang.srt.distributed.parallel_state._MOE_TP = (
        sglang.srt.distributed.parallel_state.init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_custom_allreduce=False,
            group_name="moe_tp",
        )
    )

    # Build the pipeline model-parallel groups.
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    ############################################################################
    ## This is a patch code for sgalng
    ## Ignore parallel state already set alert
    # assert (
    #     sglang.srt.distributed.parallel_state._PP is None
    # ), "pipeline model parallel group is already initialized"
    ## End of patch
    ############################################################################
    group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce
    sglang.srt.distributed.parallel_state._PP = (
        sglang.srt.distributed.parallel_state.init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_custom_allreduce=False,
            group_name="pp",
            pp_start_layer=pp_start_layer,
            pp_end_layer=pp_end_layer,
            hidden_layers=hidden_layers,
        )
    )


def monkey_patch_make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    pp_rank: Optional[int] = None,
    pp_size: Optional[int] = None,
    prefix: str = "",
    return_tuple: bool = True,
    offloader_kwargs: Dict[str, Any] = {},
) -> Tuple[int, int, torch.nn.ModuleList]:
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
            **offloader_kwargs,
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
    sglang.srt.utils.make_layers = monkey_patch_make_layers
