"""
vLLM Model Runner wrapper for Parallax.

Integrates vLLM v1 GPUModelRunner for CUDA backend.
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

from parallax.server.request import Request
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


def initialize_vllm_model_runner(
    model_repo: str,
    start_layer: int,
    end_layer: int,
    kv_cache_memory_fraction: float,
    attention_backend: str,
    kv_block_size: int,
    dtype: str = "float16",
) -> Tuple[GPUModelRunner, Dict, Any]:
    """Initialize vLLM GPUModelRunner.

    Args:
        model_repo: HuggingFace model repo path
        start_layer: Start layer index (for PP)
        end_layer: End layer index (for PP)
        kv_cache_memory_fraction: Fraction of GPU memory for KV cache
        attention_backend: Attention backend (e.g., "flash_attn")
        kv_block_size: KV cache block size
        dtype: Model dtype

    Returns:
        (model_runner, config_dict, tokenizer)
    """
    logger.info(f"Initializing vLLM model runner for {model_repo}")

    # Load HuggingFace config and tokenizer
    hf_config = AutoConfig.from_pretrained(model_repo, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)

    num_hidden_layers = hf_config.num_hidden_layers

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

    # For single-node in Parallax, we don't use vLLM's internal PP
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
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

    # Initialize GPUModelRunner
    model_runner = GPUModelRunner(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        device="cuda",
    )

    # Load model
    logger.info("Loading vLLM model...")
    model_runner.load_model()
    logger.info("vLLM model loaded successfully")

    # Return config as dict for compatibility with Parallax executor
    config_dict = {
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_key_value_heads": getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        ),
    }

    return model_runner, config_dict, tokenizer


class VLLMKVCacheManager:
    """Simple KV cache block manager for vLLM integration."""

    def __init__(self, model_runner: GPUModelRunner, block_size: int):
        self.model_runner = model_runner
        self.block_size = block_size
        self.request_blocks: Dict[str, List[int]] = {}
        self.next_block_id = 0

        # Get available blocks from model runner
        self.total_blocks = model_runner.kv_cache_config.num_gpu_blocks
        self.free_blocks = list(range(self.total_blocks))

    def allocate(self, request_id: str, num_tokens: int) -> Tuple[List[int], ...]:
        """Allocate KV cache blocks for a request.

        Returns:
            block_ids: Tuple of lists of block IDs (one per KV cache layer)
        """
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError(
                f"Not enough KV cache blocks. Needed: {num_blocks_needed}, Available: {len(self.free_blocks)}"
            )

        # Allocate blocks
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop(0)
            allocated.append(block_id)

        self.request_blocks[request_id] = allocated

        # vLLM expects tuple of lists (one per layer group)
        # For simplicity, we use the same blocks for all layers
        return (allocated,)

    def free(self, request_id: str):
        """Free KV cache blocks for a request."""
        if request_id in self.request_blocks:
            blocks = self.request_blocks.pop(request_id)
            self.free_blocks.extend(blocks)

    def get_blocks(self, request_id: str) -> Tuple[List[int], ...]:
        """Get allocated blocks for a request."""
        return (self.request_blocks.get(request_id, []),)


from __future__ import annotations

from typing import Any, Tuple

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


def initialize_vllm_model_runner(
    model_repo: str,
    start_layer: int,
    end_layer: int,
    kv_cache_memory_fraction: float,
    attention_backend: str,
    kv_block_size: int,
    dtype: str = "float16",
) -> Tuple[Any, dict, Any]:
    """Initialize vLLM GPUModelRunner (scaffold).

    This function is a placeholder and documents the expected return values:
    - model_runner: An object exposing execute_model() compatible with vLLM v1.
    - config: A dict-like model config with at least num_hidden_layers.
    - tokenizer: Tokenizer instance used by executor.
    """
    raise NotImplementedError(
        "vLLM backend scaffolding is present, but the actual model runner "
        "initialization is not implemented yet. Please implement "
        "parallax.vllm.model_runner.initialize_vllm_model_runner() per the plan."
    )
