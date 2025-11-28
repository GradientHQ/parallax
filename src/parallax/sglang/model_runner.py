"""
Imports sglang ModelRunner related modules and wrap them into create functions.
We use monkey patch to modify sglang originated methods. The main purpose is to pass
arguments needed by decentralized inference.
"""

import logging
import os
import random
from typing import List, Optional

import sglang
import sglang.srt.distributed.parallel_state
import torch
from mlx_lm.utils import load_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    initialize_dp_attention,
)
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.model_executor.model_runner import ModelRunner as SGLModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_available_gpu_memory,
    get_bool_env_var,
    monkey_patch_p2p_access_check,
)

from parallax.sglang.monkey_patch import apply_parallax_sglang_monkey_patch
from parallax.sglang.monkey_patch_utils.weight_loader_filter import (
    set_layer_range_for_filtering,
)
from parallax.utils.tokenizer_utils import load_tokenizer

logger = logging.getLogger(__name__)

_is_cpu_amx_available = cpu_has_amx_support()


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
        num_hidden_layers = model_config.hf_config.num_hidden_layers
        set_layer_range_for_filtering(pp_start_layer, pp_end_layer, num_hidden_layers)

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


def form_sgl_server_args(
    model_path: str,
    dtype: str = "bfloat16",
    kv_cache_memory_fraction: float = 0.85,
    tp_size: int = 1,
    attention_backend: str = "flashinfer",
    kv_block_size: int = 64,
    moe_runner_backend="auto",
    enable_lora: Optional[bool] = False,
    max_lora_rank: Optional[int] = None,
    lora_target_modules: Optional[List[str]] = None,
    lora_paths: Optional[List[str]] = None,
    max_loras_per_batch: Optional[int] = None,
    max_loaded_loras: Optional[int] = None,
    lora_eviction_policy: Optional[str] = "lru",
    lora_backend: Optional[str] = "triton",
    max_lora_chunk_size: Optional[int] = 128,
):
    """Creates a SGL ServerArgs object"""
    sgl_server_args = ServerArgs(
        model_path=model_path,
        dtype=dtype,
        attention_backend=attention_backend,
        page_size=kv_block_size,
        mem_fraction_static=kv_cache_memory_fraction,
        moe_runner_backend=moe_runner_backend,
        tp_size=tp_size,
        trust_remote_code=True,
        enable_lora=enable_lora,
        max_lora_rank=max_lora_rank,
        lora_target_modules=lora_target_modules,
        lora_paths=lora_paths,
        max_loras_per_batch=max_loras_per_batch,
        max_loaded_loras=max_loaded_loras,
        lora_eviction_policy=lora_eviction_policy,
        lora_backend=lora_backend,
        max_lora_chunk_size=max_lora_chunk_size,
    )
    return sgl_server_args


def initialize_sgl_model_runner(
    model_repo: str,
    start_layer: int,
    end_layer: int,
    kv_cache_memory_fraction: float,
    attention_backend: str,
    kv_block_size: int,
    moe_runner_backend: str,
    max_num_tokens_per_batch: int = 1024,
    enable_lora: Optional[bool] = False,
    max_lora_rank: Optional[int] = None,
    lora_target_modules: Optional[List[str]] = None,
    lora_paths: Optional[List[str]] = None,
    max_loras_per_batch: Optional[int] = None,
    max_loaded_loras: Optional[int] = None,
    lora_eviction_policy: Optional[str] = "lru",
    lora_backend: Optional[str] = "triton",
    max_lora_chunk_size: Optional[int] = 128,
    **kwargs,
):
    """
    Creates a SGL ModelRunner object.
    Returns:
      - model_runner: SGL model runner
      - config: model config driven by mlx-lm
      - tokenizer: tokenizer driven by mlx-lm
    """
    apply_parallax_sglang_monkey_patch()

    # Extract TP-related parameters from kwargs or use defaults
    tp_rank = kwargs.get("tp_rank", 0)
    tp_size = kwargs.get("tp_size", 1)
    use_hfcache = kwargs.get("use_hfcache", False)
    nccl_port = kwargs.get("nccl_port", None)
    # Use selective download for GPU models to save bandwidth and disk space
    from parallax.utils.selective_download import get_model_path_with_selective_download

    logger.info(
        f"Downloading model with selective weight files for layers [{start_layer}, {end_layer})"
    )
    model_path = get_model_path_with_selective_download(
        model_repo, start_layer=start_layer, end_layer=end_layer, local_files_only=use_hfcache
    )

    config = load_config(model_path)
    tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id", None))
    dtype = config.get("torch_dtype", "bfloat16")

    if nccl_port is None:
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
        str(model_path),
        dtype,
        kv_cache_memory_fraction,
        tp_size,
        attention_backend,
        kv_block_size,
        moe_runner_backend,
        enable_lora,
        max_lora_rank,
        lora_target_modules,
        lora_paths,
        max_loras_per_batch,
        max_loaded_loras,
        lora_eviction_policy,
        lora_backend,
        max_lora_chunk_size,
    )
    initialize_moe_config(server_args)
    quant_method = None
    if (quantization_config := config.get("quantization_config", None)) is not None:
        quant_method = quantization_config.get("quant_method")
    model_config = ModelConfig(
        model_path=str(model_path),
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
        gpu_id=tp_rank,  # Currently reuse tp_rank to only support TP.
        tp_rank=tp_rank,
        tp_size=tp_size,
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
