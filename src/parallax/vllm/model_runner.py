
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
    import torch

    free_memory, total_memory = torch.cuda.mem_get_info(0)
    available_memory = int(free_memory * kv_cache_memory_fraction)

    logger.info(
        f"Available GPU memory for KV cache: "
        f"{available_memory / (1024**3):.2f} GB "
        f"({kv_cache_memory_fraction:.1%} of {free_memory / (1024**3):.2f} GB)"
    )

    page_size_bytes = kv_cache_group.kv_cache_spec.page_size_bytes

    max_blocks_by_memory = available_memory // page_size_bytes

    num_blocks = max(100, min(1000, int(max_blocks_by_memory * 0.8)))

    logger.info(f"Calculated KV cache blocks: {num_blocks} (max possible: {max_blocks_by_memory})")

    tensor_size_bytes = page_size_bytes * num_blocks

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

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: Optional[KVCacheConfig],
        device: str,
        start_layer: int,
        end_layer: int,
        num_hidden_layers: int,
    ):
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.num_hidden_layers = num_hidden_layers
        self.num_shard_layers = end_layer - start_layer

        self.is_first_peer = start_layer == 0
        self.is_last_peer = end_layer == num_hidden_layers

        self.pp_rank = 0
        self.pp_size = 1

        self.request_block_hasher: Optional[Callable[[Any], List[Any]]] = None
        self.enable_prefix_caching: bool = True

        super().__init__(vllm_config=vllm_config, device=torch.device(device))
        self.kv_cache_config = kv_cache_config

        logger.info(
            f"ParallaxVLLMModelRunner initialized: layers [{start_layer}, {end_layer}), "
            f"is_first={self.is_first_peer}, is_last={self.is_last_peer}"
        )

    def _create_kv_cache_config(self, kv_cache_memory_fraction: float = None) -> KVCacheConfig:
        logger.info("Generating KV cache configuration from model...")

        try:
            kv_cache_specs = self.model.get_kv_cache_spec()
        except AttributeError:
            logger.warning(
                "Cannot access get_kv_cache_spec due to cudagraph wrapper, using fallback method"
            )
            kv_cache_specs = None

        import torch

        free_memory, total_memory = torch.cuda.mem_get_info(self.device.index or 0)

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

        if kv_cache_specs is not None:
            kv_cache_configs = get_kv_cache_configs(
                vllm_config=self.vllm_config,
                kv_cache_specs=[kv_cache_specs],
                available_memory=[available_memory],
            )
            kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        else:
            logger.info("Using fallback KV cache configuration")

            model = self.model
            hf_config = model.model.config
            num_attention_heads = getattr(hf_config, "num_attention_heads", 8)
            hidden_size = getattr(hf_config, "hidden_size", 1024)
            head_size = hidden_size // num_attention_heads

            from vllm.v1.kv_cache_interface import KVCacheGroupSpec, FullAttentionSpec

            model_dtype = self.vllm_config.model_config.dtype
            if isinstance(model_dtype, str):
                from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

                model_dtype = STR_DTYPE_TO_TORCH_DTYPE.get(model_dtype, torch.float16)

            kv_cache_group = KVCacheGroupSpec(
                layer_names=[
                    f"model.layers.{i}" for i in range(self.start_layer, self.end_layer)
                ],
                kv_cache_spec=FullAttentionSpec(
                    block_size=self.cache_config.block_size,
                    num_kv_heads=num_attention_heads,
                    head_size=head_size,
                    dtype=model_dtype,
                ),
            )

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
        logger.info("Initializing vLLM KVCacheManager...")

        if self.kv_cache_config is None:
            self.kv_cache_config = self._create_kv_cache_config()

        kv_cache_manager = KVCacheManager(
            kv_cache_config=self.kv_cache_config,
            max_model_len=max_model_len,
            enable_caching=True,
            use_eagle=False,
            log_stats=True,
            enable_kv_cache_events=False,
            dcp_world_size=1,
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

                def simple_hash_fn(obj: Any) -> bytes:
                    return str(hash(str(obj))).encode("utf-8")

                hash_fn = simple_hash_fn
                logger.info("Using simple fallback hash function for prefix caching")

            block_size = kv_cache_manager.block_size
            if block_size is None and self.kv_cache_config.kv_cache_groups:
                block_size = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
            if block_size is not None:
                self.request_block_hasher = get_request_block_hasher(block_size, hash_fn)
                logger.info("Initialized prefix cache block hasher with block_size=%d", block_size)

        logger.info(
            f"KVCacheManager initialized: block_size={kv_cache_manager.block_size}, "
            f"usage={kv_cache_manager.usage:.2%}"
        )

        return kv_cache_manager

    def load_model(self) -> None:
        logger.info(f"Loading vLLM model with layers [{self.start_layer}, {self.end_layer})...")

        from vllm.distributed.utils import get_pp_indices

        original_get_pp_indices = get_pp_indices

        def custom_get_pp_indices(num_layers: int, rank: int, world_size: int):
            logger.debug(
                f"custom_get_pp_indices called: num_layers={num_layers}, "
                f"returning [{self.start_layer}, {self.end_layer})"
            )
            return self.start_layer, self.end_layer

        import vllm.distributed.utils

        vllm.distributed.utils.get_pp_indices = custom_get_pp_indices

        try:
            super().load_model()
            logger.info(
                f"Successfully loaded {self.num_shard_layers} layers "
                f"[{self.start_layer}:{self.end_layer}]"
            )
        finally:
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
    from parallax.utils.selective_download import get_model_path_with_selective_download
    logger.info(
        f"Initializing vLLM model runner for {model_repo}, " f"layers=[{start_layer}, {end_layer})"
    )

    model_path = get_model_path_with_selective_download(
        model_repo,
        start_layer=start_layer,
        end_layer=end_layer,
    )

    config = load_config(model_path)
    tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id", None))
    dtype = config.get("torch_dtype", "bfloat16")

    num_hidden_layers = getattr(config, "num_hidden_layers", 28)
    is_first_peer = start_layer == 0
    is_last_peer = end_layer == num_hidden_layers
    virtual_pp_size = 2 if not (is_first_peer and is_last_peer) else 1

    import vllm.distributed.parallel_state as parallel_state
    import os

    if not parallel_state.model_parallel_is_initialized():
        logger.info("Initializing vLLM distributed environment...")

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
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=virtual_pp_size,
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

    model_config = ModelConfig(
        model=model_path,
        tokenizer=model_path,
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

    parallel_config = ParallelConfig(
        pipeline_parallel_size=virtual_pp_size,
        tensor_parallel_size=1,
        distributed_executor_backend=None,
    )

    device_config = DeviceConfig(device="cuda")
    load_config_for_config = LoadConfig(load_format="auto")

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

    model_runner = ParallaxVLLMModelRunner(
        vllm_config=vllm_config,
        kv_cache_config=None,
        device="cuda",
        start_layer=start_layer,
        end_layer=end_layer,
        num_hidden_layers=num_hidden_layers,
    )

    logger.info("Loading vLLM model (partial layers)...")
    model_runner.load_model()
    logger.info("vLLM model loaded successfully")

    logger.info("Letting vLLM automatically generate KV cache configuration...")

    kv_cache_specs = model_runner.get_kv_cache_spec()

    if not kv_cache_specs:
        raise RuntimeError("No KV cache specs found in the loaded model")

    import torch

    free_memory, total_memory = torch.cuda.mem_get_info(0)
    available_memory = int(free_memory * kv_cache_memory_fraction)

    logger.info(
        f"Available GPU memory for KV cache: "
        f"{available_memory / (1024**3):.2f} GB "
        f"({kv_cache_memory_fraction:.1%} of {free_memory / (1024**3):.2f} GB)"
    )

    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs, generate_scheduler_kv_cache_config

    kv_cache_configs = get_kv_cache_configs(
        vllm_config=model_runner.vllm_config,
        kv_cache_specs=[kv_cache_specs],
        available_memory=[available_memory],
    )

    kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)

    model_runner.kv_cache_config = kv_cache_config

    logger.info("Initializing GPUModelRunner KV cache...")
    model_runner.initialize_kv_cache(kv_cache_config)
    logger.info("GPUModelRunner KV cache initialized successfully")

    logger.info("Initializing KV Cache Manager...")
    model_runner.initialize_kv_cache_manager(max_model_len=model_config.max_model_len)
    logger.info("KV Cache Manager initialized successfully")

    return model_runner, config, tokenizer
