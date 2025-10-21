"""
vLLM Model Runner wrapper for Parallax with Pipeline Parallelism support.

Integrates vLLM v1 GPUModelRunner for CUDA backend.
Uses vLLM's native Pipeline Parallelism mechanism to load only required layers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer

from vllm.config import (
    CacheConfig,
    DecodingConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
)
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.config import VllmConfig
from vllm.distributed import (
    initialize_model_parallel,
    get_pp_group,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager

from parallax.server.request import Request
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class ParallaxVLLMModelRunner(GPUModelRunner):
    """
    Extended vLLM GPUModelRunner that leverages vLLM's native Pipeline Parallelism.

    This class uses vLLM's PPMissingLayer mechanism to load only the required layers
    during model initialization, avoiding the need to load and then prune the full model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        device: str,
        start_layer: int,
        end_layer: int,
        num_hidden_layers: int,
    ):
        """
        Args:
            vllm_config: vLLM configuration object
            kv_cache_config: KV cache configuration
            device: Device to run on (e.g., "cuda")
            start_layer: First layer index to load (inclusive)
            end_layer: Last layer index to load (exclusive)
            num_hidden_layers: Total number of layers in the full model
        """
        # Store layer information before calling super().__init__
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.num_hidden_layers = num_hidden_layers
        self.num_shard_layers = end_layer - start_layer

        self.is_first_peer = start_layer == 0
        self.is_last_peer = end_layer == num_hidden_layers

        # Calculate PP rank and size for vLLM
        # We simulate a PP setup where each Parallax peer is a PP rank
        self.pp_rank = 0  # Will be updated based on layer range
        self.pp_size = 1  # Single node, but with layer slicing

        # Call parent init
        super().__init__(vllm_config=vllm_config, device=torch.device(device))
        self.kv_cache_config = kv_cache_config

        logger.info(
            f"ParallaxVLLMModelRunner initialized: layers [{start_layer}, {end_layer}), "
            f"is_first={self.is_first_peer}, is_last={self.is_last_peer}"
        )

    def initialize_kv_cache_manager(self, max_model_len: int) -> KVCacheManager:
        """
        Initialize vLLM's native KVCacheManager.

        This should be called after the model is loaded to properly set up
        the KV cache management system.

        Args:
            max_model_len: Maximum sequence length the model can handle

        Returns:
            Initialized KVCacheManager instance
        """
        logger.info("Initializing vLLM KVCacheManager...")

        kv_cache_manager = KVCacheManager(
            kv_cache_config=self.kv_cache_config,
            max_model_len=max_model_len,
            enable_caching=True,  # Enable prefix caching
            use_eagle=False,  # Not using EAGLE speculative decoding
            log_stats=True,  # Enable stats logging
            enable_kv_cache_events=False,  # Disable KV cache events for now
            dcp_world_size=1,  # Decode Context Parallelism world size
        )

        self.kv_cache_manager = kv_cache_manager
        logger.info(
            f"KVCacheManager initialized: block_size={kv_cache_manager.block_size}, "
            f"usage={kv_cache_manager.usage:.2%}"
        )

        return kv_cache_manager

    def load_model(self) -> None:
        """
        Load model using vLLM's native layer loading mechanism.

        This method uses vLLM's make_layers function which creates PPMissingLayer
        placeholders for layers outside [start_layer, end_layer), ensuring only
        the required layers are actually loaded from checkpoint.
        """
        logger.info(f"Loading vLLM model with layers [{self.start_layer}, {self.end_layer})...")

        # Temporarily override vLLM's PP configuration for this peer
        # This allows us to use vLLM's layer skipping mechanism
        import vllm.distributed.parallel_state as parallel_state
        from vllm.distributed.utils import get_pp_indices

        # Monkey-patch get_pp_indices to return our custom layer range
        original_get_pp_indices = parallel_state.get_pp_indices

        def custom_get_pp_indices(num_layers: int, rank: int, world_size: int):
            """Return our custom layer range instead of vLLM's calculated range."""
            logger.debug(
                f"custom_get_pp_indices called: num_layers={num_layers}, "
                f"returning [{self.start_layer}, {self.end_layer})"
            )
            return self.start_layer, self.end_layer

        # Temporarily replace the function
        import vllm.distributed.utils

        vllm.distributed.utils.get_pp_indices = custom_get_pp_indices

        try:
            # Now call the parent load_model, which will use our custom layer range
            super().load_model()
            logger.info(
                f"Successfully loaded {self.num_shard_layers} layers "
                f"[{self.start_layer}:{self.end_layer}]"
            )
        finally:
            # Restore original function
            vllm.distributed.utils.get_pp_indices = original_get_pp_indices

        logger.info("Model loaded successfully with partial layers")


def initialize_vllm_model_runner(
    model_repo: str,
    start_layer: int,
    end_layer: int,
    kv_cache_memory_fraction: float,
    attention_backend: str,
    kv_block_size: int,
    dtype: str = "float16",
) -> Tuple[ParallaxVLLMModelRunner, Dict, Any]:
    """Initialize vLLM GPUModelRunner with true partial layer loading.

        This function leverages vLLM's native Pipeline Parallelism mechanism to load
        only the required layers, avoiding the memory overhead of loading the full model.

        The key insight is to monkey-patch vLLM's get_pp_indices function during model
        loading, which allows us to control exactly which layers are loaded. Layers
        outside the [start_layer, end_layer) range are replaced with PPMissingLayer
        placeholders that consume minimal memory.

        Args:
            model_repo: HuggingFace model repo path
            start_layer: Start layer index (inclusive)
            end_layer: End layer index (exclusive)
            kv_cache_memory_fraction: Fraction of GPU memory for KV cache
            attention_backend: Attention backend (e.g., "flash_attn")
            kv_block_size: KV cache block size
            dtype: Model dtype

        Returns:
            (model_runner, config_dict, tokenizer)

        Example:
            >>> # Load only layers 8-16 of a 32-layer model
            >>> runner, config, tok = initialize_vllm_model_runner(
            ...     "meta-llama/Llama-2-7b-hf", 8, 16, 0.8, "flash_attn", 64
            ... )
            >>> # Only 8 layers are actually loaded into memory
    ```
    """
    logger.info(
        f"Initializing vLLM model runner for {model_repo}, " f"layers=[{start_layer}, {end_layer})"
    )

    # Load HuggingFace config and tokenizer
    hf_config = AutoConfig.from_pretrained(model_repo, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)

    num_hidden_layers = hf_config.num_hidden_layers

    if end_layer > num_hidden_layers:
        raise ValueError(
            f"end_layer ({end_layer}) cannot be greater than "
            f"num_hidden_layers ({num_hidden_layers})"
        )

    # Build vLLM configs
    model_config = ModelConfig(
        model=model_repo,
        tokenizer=model_repo,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype=dtype,
        seed=0,
        max_model_len=getattr(hf_config, "max_position_embeddings", 4096),
    )

    cache_config = CacheConfig(
        block_size=kv_block_size,
        gpu_memory_utilization=kv_cache_memory_fraction,
        swap_space=0,
        cache_dtype="auto",
    )

    # Configure PP for layer slicing
    # We set pp_size > 1 to enable vLLM's layer skipping mechanism
    # but use our custom get_pp_indices to control which layers to load
    is_first_peer = start_layer == 0
    is_last_peer = end_layer == num_hidden_layers

    # Calculate a virtual PP size that makes sense
    # For example, if we have 32 layers and loading [8, 16), we're in the "middle"
    # Set pp_size=2 to enable PP mode, and we'll override the layer calculation
    virtual_pp_size = 2 if not (is_first_peer and is_last_peer) else 1

    parallel_config = ParallelConfig(
        pipeline_parallel_size=virtual_pp_size,
        tensor_parallel_size=1,
        distributed_executor_backend=None,
    )

    device_config = DeviceConfig(device="cuda")
    load_config = LoadConfig(load_format="auto")

    # Minimal scheduler config (we bypass vLLM scheduler)
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=8192,
        max_num_seqs=256,
        max_model_len=model_config.max_model_len,
    )

    decoding_config = DecodingConfig()

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config,
        lora_config=None,
        speculative_config=None,
        decoding_config=decoding_config,
        observability_config=None,
        prompt_adapter_config=None,
        quant_config=None,
        compilation_config=None,
    )

    # Determine KV cache blocks
    kv_cache_config = KVCacheConfig(
        block_size=kv_block_size,
        num_gpu_blocks=None,  # Will be calculated by model runner
    )

    # Initialize our custom ParallaxVLLMModelRunner
    model_runner = ParallaxVLLMModelRunner(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        device="cuda",
        start_layer=start_layer,
        end_layer=end_layer,
        num_hidden_layers=num_hidden_layers,
    )

    # Load model with partial layers
    logger.info("Loading vLLM model (partial layers)...")
    model_runner.load_model()
    logger.info("vLLM model loaded successfully")

    # Initialize KV Cache Manager after model is loaded
    logger.info("Initializing KV Cache Manager...")
    model_runner.initialize_kv_cache_manager(max_model_len=model_config.max_model_len)
    logger.info("KV Cache Manager initialized successfully")

    # Return config as dict for compatibility with Parallax executor
    config_dict = {
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_key_value_heads": getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        ),
        "head_dim": getattr(
            hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
        ),
    }

    return model_runner, config_dict, tokenizer
