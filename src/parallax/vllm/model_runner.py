"""
vLLM Model Runner wrapper for Parallax with Pipeline Parallelism support.

Integrates vLLM v1 GPUModelRunner for CUDA backend.
Uses vLLM's native Pipeline Parallelism mechanism to load only required layers.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple
import vllm
import torch
from transformers import AutoConfig, AutoTokenizer
from vllm.config import (
    CacheConfig,
    CompilationConfig,
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
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from parallax.utils.tokenizer_utils import load_tokenizer

from parallax_utils.logging_config import get_logger
from mlx_lm.utils import get_model_path, load_config

logger = get_logger(__name__)


def _create_kv_cache_config_from_specs(
    kv_cache_group: KVCacheGroupSpec,
    attn_layers: List[str],
    kv_cache_memory_fraction: float,
) -> KVCacheConfig:
    """
    Create KV cache configuration from KV cache group specs and attention layers.

    This is a standalone function that can be used by both the model runner's
    _create_kv_cache_config method and the initialize_vllm_model_runner function.

    Args:
        kv_cache_group: KV cache group specification
        attn_layers: List of attention layer names
        kv_cache_memory_fraction: Fraction of GPU memory to use for KV cache

    Returns:
        KVCacheConfig: Properly configured KV cache configuration
    """
    import torch

    # Calculate available GPU memory for KV cache
    free_memory, total_memory = torch.cuda.mem_get_info(0)
    available_memory = int(free_memory * kv_cache_memory_fraction)

    logger.info(
        f"Available GPU memory for KV cache: "
        f"{available_memory / (1024**3):.2f} GB "
        f"({kv_cache_memory_fraction:.1%} of {free_memory / (1024**3):.2f} GB)"
    )

    # Calculate page_size_bytes for proper tensor sizing
    page_size_bytes = kv_cache_group.kv_cache_spec.page_size_bytes

    # Calculate reasonable number of blocks based on available memory
    # Each block needs page_size_bytes, so we can fit this many blocks
    max_blocks_by_memory = available_memory // page_size_bytes

    # Use a conservative estimate (80% of max possible blocks)
    # But ensure we don't exceed available memory
    num_blocks = max(100, min(1000, int(max_blocks_by_memory * 0.8)))

    logger.info(f"Calculated KV cache blocks: {num_blocks} (max possible: {max_blocks_by_memory})")

    # Ensure tensor size is divisible by page_size_bytes
    tensor_size_bytes = page_size_bytes * num_blocks

    # Ensure KVCacheTensor.shared_by covers all attention layers; otherwise
    # vLLM will assert that some layers are not initialized.
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=tensor_size_bytes,
                shared_by=attn_layers,
            )
        ],
        kv_cache_groups=[kv_cache_group],
    )

    return kv_cache_config


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

    def _create_kv_cache_config(self, kv_cache_memory_fraction: float = None) -> KVCacheConfig:
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
        # Try to access the method directly, bypassing cudagraph wrapper if needed
        try:
            kv_cache_specs = self.model.get_kv_cache_spec()
        except AttributeError:
            # If cudagraph wrapper is blocking access, try to get specs from the underlying model
            logger.warning(
                "Cannot access get_kv_cache_spec due to cudagraph wrapper, using fallback method"
            )
            # Use a simplified approach - let KVCacheManager handle the details
            kv_cache_specs = None

        # Get available GPU memory for KV cache
        # Use PyTorch's native memory info function
        import torch

        free_memory, total_memory = torch.cuda.mem_get_info(self.device.index or 0)

        # Calculate available memory for KV cache
        # Use provided fraction or fall back to cache_config
        memory_fraction = (
            kv_cache_memory_fraction
            if kv_cache_memory_fraction is not None
            else self.cache_config.gpu_memory_utilization
        )
        available_memory = int(free_memory * memory_fraction)

        logger.info(
            f"Available GPU memory for KV cache: "
            f"{available_memory / (1024**3):.2f} GB "
            f"({memory_fraction:.1%} of {free_memory / (1024**3):.2f} GB)"
        )

        # Use vLLM's utility to generate KV cache config
        # This handles all the complexity of different attention types,
        # hybrid models, sliding windows, etc.
        if kv_cache_specs is not None:
            kv_cache_configs = get_kv_cache_configs(
                vllm_config=self.vllm_config,
                kv_cache_specs=[kv_cache_specs],  # Single worker
                available_memory=[available_memory],
            )
            # For scheduler (single worker case), we can use the first config
            kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        else:
            # Fallback: create a basic KV cache config
            logger.info("Using fallback KV cache configuration")

            # Try to get model info from the loaded model to create a more accurate config
            
                # Get model architecture info from the loaded model
            model = self.model
            hf_config = model.model.config
            num_attention_heads = getattr(hf_config, "num_attention_heads", 8)
            hidden_size = getattr(hf_config, "hidden_size", 1024)
            head_size = hidden_size // num_attention_heads
            

            # Create a basic KV cache group with the block size from cache config
            from vllm.v1.kv_cache_interface import KVCacheGroupSpec, FullAttentionSpec

            # Get the correct dtype from the model config to match query/key dtypes
            model_dtype = self.vllm_config.model_config.dtype
            if isinstance(model_dtype, str):
                from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

                model_dtype = STR_DTYPE_TO_TORCH_DTYPE.get(model_dtype, torch.float16)

            kv_cache_group = KVCacheGroupSpec(
                layer_names=[
                    f"model.layers.{i}" for i in range(self.start_layer, self.end_layer)
                ],  # Only loaded layers
                kv_cache_spec=FullAttentionSpec(
                    block_size=self.cache_config.block_size,
                    num_kv_heads=num_attention_heads,  # Use actual model info
                    head_size=head_size,  # Use actual model info
                    dtype=model_dtype,  # Use model dtype instead of hardcoded float16
                ),
            )

            # Use the extracted function to create KV cache config
            # Get layer names for the loaded layers
            layer_names = [f"model.layers.{i}" for i in range(self.start_layer, self.end_layer)]

            kv_cache_config = _create_kv_cache_config_from_specs(
                kv_cache_group=kv_cache_group,
                attn_layers=layer_names,
                kv_cache_memory_fraction=memory_fraction,
            )

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

        self.enable_prefix_caching = False

        self.request_block_hasher = None
        if enable_prefix and kv_cache_manager.block_size is not None:
            try:
                hashing_mod = importlib.import_module("vllm.utils.hashing")
                get_hash_fn_by_name: Callable[[str], Callable[[Any], bytes]] = getattr(
                    hashing_mod, "get_hash_fn_by_name"
                )
                hash_fn = get_hash_fn_by_name(cache_config.prefix_caching_hash_algo)
                init_none_hash(hash_fn)
            except (ModuleNotFoundError, AttributeError) as exc:
                logger.warning("Unable to initialize prefix cache hashing: %s", exc)

                # Use a simple fallback hash function
                def simple_hash_fn(obj: Any) -> bytes:
                    return str(hash(str(obj))).encode("utf-8")

                hash_fn = simple_hash_fn
                logger.info("Using simple fallback hash function for prefix caching")

            # Initialize block hasher regardless of whether we got the hash function from vLLM or fallback
            block_size = kv_cache_manager.block_size
            if block_size is None and self.kv_cache_config.kv_cache_groups:
                block_size = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
            if block_size is not None:
                self.request_block_hasher = get_request_block_hasher(block_size, hash_fn)
                logger.info("Initialized prefix cache block hasher with block_size=%d", block_size)

        # Add detailed debugging information
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
        from vllm.distributed.utils import get_pp_indices

        # Monkey-patch get_pp_indices to return our custom layer range
        original_get_pp_indices = get_pp_indices

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
    max_num_tokens_per_batch: int = 1024,
    dtype: str = "float16",
    **kwargs,
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

    model_path = get_model_path(model_repo)[0]
    config = load_config(model_path)
    tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id", None))
    dtype = config.get("torch_dtype", "bfloat16")

    # Calculate virtual PP size (needed for both configs)
    num_hidden_layers = getattr(config, "num_hidden_layers", 28)
    is_first_peer = start_layer == 0
    is_last_peer = end_layer == num_hidden_layers
    virtual_pp_size = 2 if not (is_first_peer and is_last_peer) else 1

    # Initialize vLLM distributed environment for pipeline parallelism
    # This is required for vLLM's pipeline parallel mechanism to work
    import vllm.distributed.parallel_state as parallel_state
    import os

    # Initialize distributed environment if not already initialized
    if not parallel_state.model_parallel_is_initialized():
        logger.info("Initializing vLLM distributed environment...")

        # Set required environment variables for single GPU scenario
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"

        try:
            parallel_state.init_distributed_environment()
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,  # Single GPU
                pipeline_model_parallel_size=virtual_pp_size,  # Match ParallelConfig
            )
            logger.info(f"vLLM distributed environment initialized with pp_size={virtual_pp_size}")
        except Exception as e:
            logger.warning(f"Failed to initialize distributed environment: {e}")
            logger.info("Continuing without distributed initialization...")

    

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
        max_model_len=getattr(config, "max_position_embeddings", 4096),
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
    # virtual_pp_size is already calculated above

    parallel_config = ParallelConfig(
        pipeline_parallel_size=virtual_pp_size,
        tensor_parallel_size=1,
        distributed_executor_backend=None,
    )

    device_config = DeviceConfig(device="cuda")
    load_config_for_config = LoadConfig(load_format="auto")

    # Minimal scheduler config (we bypass vLLM scheduler)
    # Ensure max_num_batched_tokens is at least as large as max_model_len
    # Use the provided max_num_tokens_per_batch parameter
    max_batched_tokens = max(max_num_tokens_per_batch, model_config.max_model_len)
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=max_batched_tokens,
        max_num_seqs=256,
        max_model_len=model_config.max_model_len,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config_for_config,
        lora_config=None,
        speculative_config=None,
        observability_config=None,
        prompt_adapter_config=None,
        quant_config=None,
        compilation_config=CompilationConfig(),
        kv_transfer_config=None,
        kv_events_config=None,
        additional_config={},
        instance_id="",
    )

    # Initialize runner first; we'll build KV cache config after model load
    model_runner = ParallaxVLLMModelRunner(
        vllm_config=vllm_config,
        kv_cache_config=None,
        device="cuda",
        start_layer=start_layer,
        end_layer=end_layer,
        num_hidden_layers=num_hidden_layers,
    )

    # Load model with partial layers
    logger.info("Loading vLLM model (partial layers)...")
    model_runner.load_model()
    logger.info("vLLM model loaded successfully")

    # Let vLLM automatically generate KV cache configuration
    # This ensures proper shape and format compatibility
    logger.info("Letting vLLM automatically generate KV cache configuration...")

    # Get KV cache specs from the loaded model
    kv_cache_specs = model_runner.get_kv_cache_spec()

    if not kv_cache_specs:
        raise RuntimeError("No KV cache specs found in the loaded model")

    # Calculate available memory for KV cache
    import torch

    free_memory, total_memory = torch.cuda.mem_get_info(0)
    available_memory = int(free_memory * kv_cache_memory_fraction)

    logger.info(
        f"Available GPU memory for KV cache: "
        f"{available_memory / (1024**3):.2f} GB "
        f"({kv_cache_memory_fraction:.1%} of {free_memory / (1024**3):.2f} GB)"
    )

    # Use vLLM's utility to generate KV cache config
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs, generate_scheduler_kv_cache_config

    kv_cache_configs = get_kv_cache_configs(
        vllm_config=model_runner.vllm_config,
        kv_cache_specs=[kv_cache_specs],  # Single worker
        available_memory=[available_memory],
    )

    # For single worker case, use the first config
    kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)

    model_runner.kv_cache_config = kv_cache_config

    # Initialize GPU-side KV cache (creates attn_groups, block tables, etc.)
    logger.info("Initializing GPUModelRunner KV cache...")
    model_runner.initialize_kv_cache(kv_cache_config)
    logger.info("GPUModelRunner KV cache initialized successfully")

    # Initialize KV Cache Manager after model is loaded
    logger.info("Initializing KV Cache Manager...")
    model_runner.initialize_kv_cache_manager(max_model_len=model_config.max_model_len)
    logger.info("KV Cache Manager initialized successfully")

    # Return config as dict for compatibility with Parallax executor
    

    return model_runner, config, tokenizer
