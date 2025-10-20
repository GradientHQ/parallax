"""
Imports vLLM ModelRunner related modules and wrap them into create functions.
We use monkey patch to modify vLLM originated methods. The main purpose is to pass
arguments needed by decentralized inference with pipeline parallelism.
"""

import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from mlx_lm.utils import get_model_path, load_config
from vllm import EngineArgs, LLMEngine
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    LoRAConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
)
from vllm.executor.ray_gpu_executor import RayGPUExecutor
from vllm.model_executor.layers.sampler import Sampler
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import get_distributed_init_method, get_ip, get_open_port

from parallax.utils.tokenizer_utils import load_tokenizer

logger = logging.getLogger(__name__)


class ParallaxVLLMEngine:
    """
    Wrapper around vLLM Engine that supports pipeline parallelism for decentralized inference.
    This class handles the sharding of layers across different nodes.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        pp_start_layer: int,
        pp_end_layer: int,
        **kwargs,
    ):
        """
        Initialize ParallaxVLLMEngine with pipeline parallelism support.

        Args:
            model_config: vLLM model configuration
            cache_config: vLLM cache configuration
            parallel_config: vLLM parallel configuration
            scheduler_config: vLLM scheduler configuration
            device_config: vLLM device configuration
            load_config: vLLM load configuration
            lora_config: Optional LoRA configuration
            pp_start_layer: Starting layer index for this shard (inclusive)
            pp_end_layer: Ending layer index for this shard (exclusive)
        """
        self.pp_start_layer = pp_start_layer
        self.pp_end_layer = pp_end_layer
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.load_config = load_config
        self.lora_config = lora_config

        # Modify model config to only load specified layers
        self.model_config.hf_config.start_layer = pp_start_layer
        self.model_config.hf_config.end_layer = pp_end_layer

        # Initialize the vLLM engine
        # Note: vLLM doesn't natively support arbitrary layer sharding,
        # so we need to monkey patch the model loading
        from vllm.worker.model_runner import ModelRunner

        self.model_runner = None
        self.is_first_peer = pp_start_layer == 0
        self.is_last_peer = pp_end_layer == model_config.hf_config.num_hidden_layers

        logger.info(
            f"Initialized ParallaxVLLMEngine: layers [{pp_start_layer}, {pp_end_layer}), "
            f"is_first={self.is_first_peer}, is_last={self.is_last_peer}"
        )

    def initialize_model(self):
        """Initialize the model with the specified layer range."""
        # Import here to avoid circular dependency
        from vllm.worker.worker import Worker

        # Create worker with modified configuration
        worker = Worker(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            local_rank=0,
            rank=0,
            distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
        )

        # Initialize worker
        worker.init_device()
        worker.load_model()

        self.model_runner = worker.model_runner
        logger.info("vLLM model loaded successfully")

    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[torch.Tensor],
    ) -> SamplerOutput:
        """
        Execute the model on the given sequences.

        Args:
            seq_group_metadata_list: List of sequence group metadata
            kv_caches: List of KV cache tensors

        Returns:
            SamplerOutput containing logits or sampled tokens
        """
        if self.model_runner is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        return self.model_runner.execute_model(
            seq_group_metadata_list=seq_group_metadata_list, kv_caches=kv_caches
        )


def form_vllm_engine_args(
    model_path: str,
    dtype: str = "bfloat16",
    kv_block_size: int = 16,
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 256,
    max_model_len: Optional[int] = None,
    enforce_eager: bool = False,
    **kwargs,
) -> EngineArgs:
    """
    Creates vLLM EngineArgs object with Parallax-specific configurations.

    Args:
        model_path: Path or name of the model
        dtype: Data type for model weights (e.g., "bfloat16", "float16")
        kv_block_size: Block size for paged attention KV cache
        gpu_memory_utilization: Fraction of GPU memory to use
        max_num_seqs: Maximum number of sequences to process
        max_model_len: Maximum model context length
        enforce_eager: Whether to enforce eager execution (disable CUDA graphs)

    Returns:
        EngineArgs: vLLM engine arguments
    """
    engine_args = EngineArgs(
        model=model_path,
        dtype=dtype,
        tokenizer=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        block_size=kv_block_size,
        enforce_eager=enforce_eager,
        # Disable tensor parallelism for now (will be handled by Parallax)
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        **kwargs,
    )
    return engine_args


def initialize_vllm_model_runner(
    original_model_path: str,
    start_layer: int,
    end_layer: int,
    kv_cache_memory_fraction: float,
    kv_block_size: int,
    max_num_seqs: int = 256,
    max_model_len: Optional[int] = None,
    enforce_eager: bool = False,
) -> Tuple[ParallaxVLLMEngine, Dict[str, Any], Any]:
    """
    Creates a Parallax vLLM Engine object for decentralized inference.

    Args:
        original_model_path: Original model path or name
        start_layer: Starting layer index (inclusive)
        end_layer: Ending layer index (exclusive)
        kv_cache_memory_fraction: Fraction of memory for KV cache
        kv_block_size: Block size for paged attention
        max_num_seqs: Maximum number of sequences
        max_model_len: Maximum model context length
        enforce_eager: Whether to disable CUDA graphs

    Returns:
        Tuple of (vllm_engine, config_dict, tokenizer)
    """
    # Load model configuration
    model_path = get_model_path(original_model_path)[0]
    config = load_config(model_path)
    tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id", None))

    # Get dtype from config
    dtype = str(config.get("torch_dtype", "bfloat16")).replace("torch.", "")

    # Create engine args
    engine_args = form_vllm_engine_args(
        model_path=original_model_path,
        dtype=dtype,
        kv_block_size=kv_block_size,
        gpu_memory_utilization=kv_cache_memory_fraction,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
    )

    # Create model, cache, parallel, scheduler, and device configs
    model_config = ModelConfig(
        model=original_model_path,
        tokenizer=original_model_path,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype=dtype,
        seed=0,
        max_model_len=max_model_len,
    )

    cache_config = CacheConfig(
        block_size=kv_block_size,
        gpu_memory_utilization=kv_cache_memory_fraction,
        swap_space=4,  # GB
        cache_dtype=dtype,
    )

    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        worker_use_ray=False,
        max_parallel_loading_workers=None,
    )

    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=None,
        max_num_seqs=max_num_seqs,
        max_model_len=model_config.max_model_len,
    )

    device_config = DeviceConfig(device="cuda")

    load_config = LoadConfig(
        load_format="auto",
        download_dir=None,
        model_loader_extra_config=None,
    )

    # Create Parallax vLLM Engine
    logger.info(
        f"Creating ParallaxVLLMEngine: model={original_model_path}, "
        f"layers=[{start_layer}, {end_layer}), dtype={dtype}"
    )

    vllm_engine = ParallaxVLLMEngine(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config,
        lora_config=None,
        pp_start_layer=start_layer,
        pp_end_layer=end_layer,
    )

    # Initialize the model
    vllm_engine.initialize_model()

    logger.info(f"vLLM model runner initialized for layers [{start_layer}, {end_layer})")

    return vllm_engine, config, tokenizer
