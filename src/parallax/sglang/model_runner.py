"""
Imports sglang ModelRunner related modules and wrap them into create functions.
We use monkey patch to modify sglang originated methods. The main purpose is to pass
arguments needed by decentralized inference.
"""

import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import sglang
import sglang.srt.distributed.parallel_state
import torch
from mlx_lm.utils import get_model_path, load_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
)
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator as SGLGroupCoordinator,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    initialize_dp_attention,
)
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.model_executor.model_runner import ModelRunner as SGLModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    LayerFn,
    add_prefix,
    cpu_has_amx_support,
    get_available_gpu_memory,
    get_bool_env_var,
    is_npu,
    monkey_patch_p2p_access_check,
)
from torch.distributed import Backend

from parallax.utils.tokenizer_utils import load_tokenizer

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
            use_torch_symm_mem=use_torch_symm_mem,
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


class ParallaxModelRunner(SGLModelRunner):
    """
    Parallax ModelRunner module.
    pp_start_layer and pp_end_layer are passed to initialize states of distribution.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        pp_start_layer: int,
        pp_end_layer: int,
    ):
        """Add pp_start_layer and pp_end_layer for decentralized model"""
        self.pp_start_layer = pp_start_layer
        self.pp_end_layer = pp_end_layer
        super().__init__(
            model_config=model_config,
            mem_fraction_static=mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=moe_ep_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )

    def init_torch_distributed(self):
        """
        Modifies init_torch_distributed in sglang.
        The only difference is to replace initialize_model_parallel.
        """
        logger.info("Init torch distributed begin.")

        try:
            torch.get_device_module(self.device).set_device(self.gpu_id)
        except Exception:
            logger.warning(
                f"Context: {self.device=} {self.gpu_id=} {os.environ.get('CUDA_VISIBLE_DEVICES')=} \
                {self.tp_rank=} {self.tp_size=}"
            )
            raise

        if self.device == "cuda":
            backend = "nccl"
        elif self.device == "xpu":
            backend = "xccl"
        elif self.device == "hpu":
            backend = "hccl"
        elif self.device == "cpu":
            backend = "gloo"
        elif self.device == "npu":
            backend = "hccl"

        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if not self.server_args.enable_p2p_check:
            monkey_patch_p2p_access_check()

        if self.server_args.dist_init_addr:
            dist_init_method = f"tcp://{self.server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{self.dist_port}"
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        set_mscclpp_all_reduce(self.server_args.enable_mscclpp)

        if not self.is_draft_worker:
            if self.device == "cpu":
                if _is_cpu_amx_available:
                    # Bind OpenMP threads to CPU cores
                    torch.ops.sgl_kernel.init_cpu_threads_env(self.local_omp_cpuid)

                    # Set local size to hint SGLang to use shared memory based AllReduce
                    os.environ["LOCAL_SIZE"] = str(self.tp_size)
                    torch.ops.sgl_kernel.initialize(self.tp_size, self.tp_rank)
                else:
                    logger.warning(
                        "init_cpu_threads_env and shared memory based AllReduce is disabled \
                         since intel amx backend is not available"
                    )

            # Only initialize the distributed environment on the target model worker.
            init_distributed_environment(
                backend=backend,
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
            )

            # Use monkey patch modified function
            sglang.srt.distributed.parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                pipeline_model_parallel_size=self.pp_size,
                expert_model_parallel_size=self.moe_ep_size,
                duplicate_tp_group=self.server_args.enable_pdmux,
                pp_start_layer=self.pp_start_layer,
                pp_end_layer=self.pp_end_layer,
                hidden_layers=self.model_config.num_hidden_layers,
            )

            initialize_dp_attention(
                self.server_args,
                self.model_config,
            )

        min_per_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        self.tp_group = get_tp_group()
        self.attention_tp_group = get_attention_tp_group()

        # Check memory for tensor parallelism
        local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if self.tp_size > 1 and not self.is_draft_worker:
            if min_per_gpu_memory < local_gpu_memory * 0.9:
                if get_bool_env_var("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"):
                    logger.warning(
                        "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                        f"{min_per_gpu_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                    )
                else:
                    raise ValueError(
                        "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                        f"{min_per_gpu_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                    )

        logger.info(
            f"Init torch distributed ends. mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
        )

        # This is a hack for initializing CudaGraphRunner
        self.server_args.pp_size = 2

        return min_per_gpu_memory


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
    assert (
        sglang.srt.distributed.parallel_state._TP is None
    ), "tensor model parallel group is already initialized"
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
    assert (
        sglang.srt.distributed.parallel_state._MOE_EP is None
    ), "expert model parallel group is already initialized"
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

    assert (
        sglang.srt.distributed.parallel_state._MOE_TP is None
    ), "expert model parallel group is already initialized"
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
    assert (
        sglang.srt.distributed.parallel_state._PP is None
    ), "pipeline model parallel group is already initialized"
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


## TODO: Move this when sgalang supports qwen3_next pipeline parallelism
def monkey_patch_qwen3_next():
    from parallax.sglang.monkey_patch.qwen3_next_config import (
        apply_qwen3_next_config_monkey_patch,
    )
    from parallax.sglang.monkey_patch.qwen3_next_model import (
        apply_qwen3_next_monkey_patch,
    )

    apply_qwen3_next_monkey_patch()
    apply_qwen3_next_config_monkey_patch()


## TODO: Move this when sgalang supports gpt_oss pipeline parallelism
def monkey_patch_gpt_oss():
    from parallax.sglang.monkey_patch.gpt_oss_model import apply_gpt_oss_monkey_patch

    apply_gpt_oss_monkey_patch()


## TODO: Move this when sgalang supports triton backend pipeline parallelism
def monkey_patch_triton_backend_init():
    from parallax.sglang.monkey_patch.triton_backend import (
        apply_triton_backend_init_monkey_patch,
    )

    apply_triton_backend_init_monkey_patch()


def monkey_patch_minimax_m2_model():
    from parallax.sglang.monkey_patch.minimax_m2_model import (
        apply_minimax_m2_monkey_patch,
    )

    apply_minimax_m2_monkey_patch()


def monkey_patch_glm4_moe_model():
    from parallax.sglang.monkey_patch.glm4_moe_model import apply_glm4_moe_monkey_patch

    apply_glm4_moe_monkey_patch()


def form_sgl_server_args(
    model_path: str,
    dtype: str = "bfloat16",
    attention_backend: str = "flashinfer",
    kv_block_size: int = 64,
    moe_runner_backend="auto",
):
    """Creates a SGL ServerArgs object"""
    sgl_server_args = ServerArgs(
        model_path=model_path,
        dtype=dtype,
        attention_backend=attention_backend,
        page_size=kv_block_size,
        mem_fraction_static=0.85,
        moe_runner_backend=moe_runner_backend,
    )
    return sgl_server_args


def apply_parallax_monkey_patch():
    """Apply all monkey patch"""
    # Function patch
    sglang.srt.distributed.parallel_state.init_model_parallel_group = (
        monkey_patch_init_model_parallel_group
    )
    sglang.srt.distributed.parallel_state.initialize_model_parallel = (
        monkey_patch_initialize_model_parallel
    )
    sglang.srt.utils.make_layers = monkey_patch_make_layers
    monkey_patch_qwen3_next()
    monkey_patch_gpt_oss()
    monkey_patch_triton_backend_init()
    monkey_patch_minimax_m2_model()
    monkey_patch_glm4_moe_model()


def initialize_sgl_model_runner(
    original_model_path: str,
    start_layer: int,
    end_layer: int,
    kv_cache_memory_fraction: float,
    attention_backend: str,
    kv_block_size: int,
    moe_runner_backend: str,
):
    """
    Creates a SGL ModelRunner object.
    Returns:
      - model_runner: SGL model runner
      - config: model config driven by mlx-lm
      - tokenizer: tokenizer driven by mlx-lm
    """
    apply_parallax_monkey_patch()
    model_path = get_model_path(original_model_path)[0]
    config = load_config(model_path)
    tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id", None))
    dtype = config.get("torch_dtype", "bfloat16")
    nccl_port = random.randint(4000, 5000)

    # Handling mxfp4 arguments
    quant_method = config.get("quant_method", None)
    quantization_config = config.get("quantization_config", None)
    if quant_method is None and quantization_config is not None:
        quant_method = quantization_config.get("quant_method", None)
    if quant_method == "mxfp4":
        attention_backend = "triton"
        moe_runner_backend = "triton_kernel"

    architectures = config.get("architectures", [])
    if architectures and any("Qwen3Next" in arch for arch in architectures):
        logger.debug(f"Qwen3-Next model detected, setting kv_block_size to 1")
        kv_block_size = 1

    server_args = form_sgl_server_args(
        original_model_path,
        dtype,
        attention_backend,
        kv_block_size,
        moe_runner_backend,
    )
    initialize_moe_config(server_args)
    quant_method = None
    if (quantization_config := config.get("quantization_config", None)) is not None:
        quant_method = quantization_config.get("quant_method")
    model_config = ModelConfig(
        model_path=original_model_path,
        model_override_args="{}",
        dtype=dtype,
        quantization=quant_method,
    )
    # TODO: Fix me
    model_config.hf_config.tie_word_embeddings = False
    model_config.hf_config.start_layer = start_layer
    model_config.hf_config.end_layer = end_layer

    logger.debug(f"model_start_layer: {model_config.hf_config.start_layer}")
    logger.debug(f"model_end_layer: {model_config.hf_config.end_layer}")

    model_runner = ParallaxModelRunner(
        model_config=model_config,
        mem_fraction_static=kv_cache_memory_fraction,
        gpu_id=0,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        nccl_port=nccl_port,
        server_args=server_args,
        pp_start_layer=start_layer,
        pp_end_layer=end_layer,
    )
    return model_runner, config, tokenizer
