"""
Creates executor from factory for different backends.
"""

import argparse
from typing import Any, List, Optional

from parallax.utils.utils import get_current_device
from parallax_utils.logging_config import get_logger, set_log_level

logger = get_logger(__name__)


def create_executor_config(args: argparse.Namespace, shared_state=None, conn=None):
    """Create executor configuration from command line arguments."""

    config = {
        "model_repo": args.model_path,
        "start_layer": args.start_layer,
        "end_layer": args.end_layer,
        "dtype": args.dtype,
        "max_sequence_length": args.max_sequence_length if "max_sequence_length" in args else None,
        "max_batch_size": args.max_batch_size if "max_batch_size" in args else None,
        "kv_block_size": args.kv_block_size,
        "kv_cache_memory_fraction": args.kv_cache_memory_fraction,
        "enable_prefix_cache": args.enable_prefix_cache,
        "max_num_tokens_per_batch": args.max_num_tokens_per_batch,
        "prefill_priority": args.prefill_priority,
        "micro_batch_ratio": args.micro_batch_ratio,
        "scheduler_wait_ms": args.scheduler_wait_ms,
        "send_to_peer_addr": args.send_to_peer_addr if "send_to_peer_addr" in args else None,
        "recv_from_peer_addr": args.recv_from_peer_addr if "recv_from_peer_addr" in args else None,
        "executor_input_ipc_addr": args.executor_input_ipc,
        "executor_output_ipc_addr": args.executor_output_ipc,
        "attention_backend": args.attention_backend,
        "enable_dp_attention": getattr(args, "enable_dp_attention", False),
        "moe_runner_backend": args.moe_runner_backend,
        "tp_rank": args.tp_rank,
        "tp_size": args.tp_size,
        "dp_rank": getattr(args, "dp_rank", 0),
        "dp_size": getattr(args, "dp_size", 1),
        "nccl_port": args.nccl_port,
        "shared_state": shared_state,
        "conn": conn,
        "use_hfcache": args.use_hfcache,
        "enable_lora": args.enable_lora,
        "max_lora_rank": args.max_lora_rank,
        "lora_paths": args.lora_paths,
        "max_loras_per_batch": args.max_loras_per_batch,
        "max_loaded_loras": args.max_loaded_loras,
        "enable_weight_refit": args.enable_weight_refit,
        "weight_refit_mode": args.weight_refit_mode,
    }

    if args.gpu_backend == "sglang":
        config.update(
            {
                "lora_target_modules": args.lora_target_modules,
                "lora_eviction_policy": args.lora_eviction_policy,
                "lora_backend": args.lora_backend,
                "max_lora_chunk_size": args.max_lora_chunk_size,
            }
        )
    elif args.gpu_backend == "vllm":
        config.update(
            {
                "fully_sharded_loras": getattr(args, "fully_sharded_loras", False),
                "enable_return_routed_experts": getattr(
                    args, "enable_return_routed_experts", False
                ),
            }
        )

    return config


def create_from_args(
    args,
    shared_state: Optional[dict] = None,
    conn: Optional[List[Any]] = None,
    device: Optional[str] = None,
):
    """
    Creat executor for different backend.
    Lazy import here since CUDA modules cannot be import withough hardware support.
    """
    config = create_executor_config(args, shared_state, conn)
    if device is None:
        device = get_current_device()
    if device is not None and device.startswith("cuda"):
        if args.gpu_backend == "sglang":
            from parallax.server.executor.sglang_executor import SGLExecutor

            executor = SGLExecutor(**config)
        elif args.gpu_backend == "vllm":
            from parallax.server.executor.vllm_executor import VLLMExecutor

            executor = VLLMExecutor(**config)
        else:
            raise ValueError(f"Unsupported GPU backend type: {args.gpu_backend}")
    elif device == "mlx":
        from parallax.server.executor.mlx_executor import MLXExecutor

        executor = MLXExecutor(**config)
    else:
        raise ValueError(f"Unsupported device type: {device}")
    return executor


def run_executor_process(args, shared_state=None, conn=None):
    """Run executor as a subprocess"""
    set_log_level(args.log_level)
    executor = None
    try:
        executor = create_from_args(args, shared_state, conn)
        executor.run_loop()
    except KeyboardInterrupt:
        logger.debug("Executor received interrupt signal, shutting down...")
    except Exception as e:
        logger.exception(e)
    finally:
        if executor is not None:
            executor.shutdown()


def stop_executor_process(executor_process):
    """Kill a subprocess"""
    logger.debug("Terminating executor subprocess...")
    try:
        executor_process.kill()
        executor_process.join()
    except Exception as e:
        logger.error(f"Failed to terminate executor subprocess: {e}")
