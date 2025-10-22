"""
vLLM Model Runner wrapper for Parallax with Pipeline Parallelism support.

Integrates vLLM v1 GPUModelRunner for CUDA backend.
Uses vLLM's native Pipeline Parallelism mechanism to load only required layers.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    VllmConfig,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

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
        kv_cache_config: Optional[KVCacheConfig],
        device: str,
        start_layer: int,
        end_layer: int,
        num_hidden_layers: int,
    ):
        """
        Args:
            vllm_config: vLLM configuration object
            kv_cache_config: KV cache configuration (can be None, will be created by KVCacheManager)
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

        self.request_block_hasher: Optional[Callable[[Any], List[Any]]] = None
        self.enable_prefix_caching: bool = True

        # Call parent init
        super().__init__(vllm_config=vllm_config, device=torch.device(device))
        # KV cache config will be created by KVCacheManager during initialization
        self.kv_cache_config = kv_cache_config

        logger.info(
            f"ParallaxVLLMModelRunner initialized: layers [{start_layer}, {end_layer}), "
            f"is_first={self.is_first_peer}, is_last={self.is_last_peer}"
        )

    def _create_kv_cache_config(self) -> KVCacheConfig:
        """
        Create KV cache configuration from the loaded model.

        This method leverages vLLM's native KV cache configuration generation
        by extracting KV cache specs from the model's attention layers and
        using vLLM's utilities to generate the proper configuration.

        Returns:
            KVCacheConfig: Properly configured KV cache configuration
        """
        logger.info("Generating KV cache configuration from model...")

        # Get KV cache specs from model's attention layers
        # This method is provided by vLLM's GPUModelRunner
        kv_cache_specs = self.model.get_kv_cache_spec()

        # Get available GPU memory for KV cache
        # Use vLLM's memory profiling utilities
        from vllm.utils import get_gpu_memory

        free_memory, _ = get_gpu_memory(self.device.index or 0)

        # Calculate available memory for KV cache based on cache_config
        gpu_memory_utilization = self.cache_config.gpu_memory_utilization
        available_memory = int(free_memory * gpu_memory_utilization)

        logger.info(
            f"Available GPU memory for KV cache: "
            f"{available_memory / (1024**3):.2f} GB "
            f"({gpu_memory_utilization:.1%} of {free_memory / (1024**3):.2f} GB)"
        )

        # Use vLLM's utility to generate KV cache config
        # This handles all the complexity of different attention types,
        # hybrid models, sliding windows, etc.
        kv_cache_configs = get_kv_cache_configs(
            vllm_config=self.vllm_config,
            kv_cache_specs=[kv_cache_specs],  # Single worker
            available_memory=[available_memory],
        )

        # For scheduler (single worker case), we can use the first config
        kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)

        logger.info(
            f"KV cache config generated: "
            f"num_blocks={kv_cache_config.num_blocks}, "
            f"num_groups={len(kv_cache_config.kv_cache_groups)}"
        )

        return kv_cache_config

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

        # Generate KV cache config from model if not already provided
        if self.kv_cache_config is None:
            self.kv_cache_config = self._create_kv_cache_config()

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
        cache_config = self.vllm_config.cache_config
        enable_prefix = cache_config.enable_prefix_caching
        if enable_prefix is None:
            enable_prefix = True
        self.enable_prefix_caching = enable_prefix

        self.request_block_hasher = None
        if enable_prefix and kv_cache_manager.block_size is not None:
            try:
                hashing_mod = importlib.import_module("vllm.utils.hashing")
                get_hash_fn_by_name: Callable[[str], Callable[[Any], bytes]] = getattr(
                    hashing_mod, "get_hash_fn_by_name"
                )
                hash_fn = get_hash_fn_by_name(cache_config.prefix_caching_hash_algo)
            except (ModuleNotFoundError, AttributeError) as exc:
                logger.warning("Unable to initialize prefix cache hashing: %s", exc)
            else:
                init_none_hash(hash_fn)
                block_size = kv_cache_manager.block_size
                if block_size is None and self.kv_cache_config.kv_cache_groups:
                    block_size = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
                if block_size is not None:
                    self.request_block_hasher = get_request_block_hasher(block_size, hash_fn)
                    logger.info(
                        "Initialized prefix cache block hasher with block_size=%d", block_size
                    )

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

    # Note: KVCacheConfig will be created by vLLM's KVCacheManager during initialization
    # We don't need to manually create it here as it requires complex layer-specific information
    # The KVCacheManager will handle this based on the model's architecture

    # Initialize our custom ParallaxVLLMModelRunner
    model_runner = ParallaxVLLMModelRunner(
        vllm_config=vllm_config,
        kv_cache_config=None,  # Will be created by KVCacheManager
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
