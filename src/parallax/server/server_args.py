"""
CLI argument parser for Parallax server.

This module provides argument parsing functionality for the Parallax executor,
supporting various configuration options for model loading, layer sharding,
and performance tuning.
"""

import argparse

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the Parallax executor.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Parallax Executor - Distributed LLM Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # HTTP server configuration
    parser.add_argument("--host", type=str, default="localhost", help="Host of the HTTP server.")
    parser.add_argument("--port", type=int, default=3000, help="Port of the HTTP server")
    parser.add_argument(
        "--node-chat-port", type=int, default=3002, help="Port of the node chat HTTP server"
    )

    # Lattica configuration
    parser.add_argument("--initial-peers", nargs="+", default=[], help="List of initial DHT peers")
    parser.add_argument("--scheduler-addr", type=str, default=None, help="Scheduler address")
    parser.add_argument("--relay-servers", nargs="+", default=[], help="List of relay DHT peers")
    parser.add_argument("--tcp-port", type=int, default=0, help="Port for Lattica TCP listening")
    parser.add_argument("--udp-port", type=int, default=0, help="Port for Lattica UDP listening")
    parser.add_argument(
        "--announce-maddrs", nargs="+", default=[], help="List of multiaddresses to announce"
    )
    parser.add_argument("--dht-prefix", type=str, default="gradient", help="Prefix for DHT keys")
    parser.add_argument(
        "--notify-url", type=str, default=None, help="URL to notify when a request is finished"
    )

    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the model repository or model name (e.g., 'mlx-community/Qwen3-0.6B-bf16')",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="Maximum sequence length for the model",
    )

    parser.add_argument(
        "--param-mem-ratio",
        type=float,
        default=0.65,
        help="Ratio of GPU memory to use for parameter hosting",
    )

    parser.add_argument(
        "--kvcache-mem-ratio",
        type=float,
        default=0.25,
        help="Ratio of GPU memory to use for KV cache",
    )

    parser.add_argument(
        "--start-layer",
        type=int,
        default=None,
        help="Starting layer index for this shard (inclusive)",
    )

    parser.add_argument(
        "--end-layer", type=int, default=None, help="Ending layer index for this shard (exclusive)"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights and computations",
    )

    # KV Cache configuration
    parser.add_argument(
        "--kv-cache-memory-fraction",
        type=float,
        default=0.8,
        help="Fraction of available memory to use for KV cache (0.0 to 1.0)",
    )

    parser.add_argument(
        "--kv-block-size", type=int, default=64, help="Block size for KV cache management"
    )

    parser.add_argument(
        "--enable-prefix-cache", action="store_true", help="Enable prefix cache reuse"
    )

    # Scheduler configuration
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum batch size for processing requests",
    )

    parser.add_argument(
        "--max-num-tokens-per-batch",
        type=int,
        default=1024,
        help="Maximum number of tokens in a batch",
    )

    parser.add_argument(
        "--prefill-priority",
        type=int,
        default=0,
        choices=[0, 1],
        help="Priority for prefill requests (0 or 1)",
    )

    parser.add_argument(
        "--micro-batch-ratio", type=int, default=2, help="Micro batch ratio for scheduling"
    )

    parser.add_argument(
        "--scheduler-wait-ms", type=int, default=500, help="Scheduler wait time in milliseconds"
    )

    parser.add_argument(
        "--request-timeout-s",
        type=int,
        default=600,
        help="Per-request timeout in seconds before automatic abort",
    )

    # GPU/SGLang specialized configuration
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="flashinfer",
        choices=["torch_native", "flashinfer", "triton", "fa3"],
        help="Choose the GPU attention kernels",
    )

    parser.add_argument(
        "--moe-runner-backend",
        type=str,
        default="auto",
        choices=[
            "auto",
            "triton",
            "triton_kernel",
            "flashinfer_trtllm",
            "flashinfer_cutlass",
            "flashinfer_mxfp4",
        ],
        help="Choose the GPU moe kernels",
    )

    parser.add_argument(
        "--enable-lora", action="store_true", help="Enable LoRA adapter support for SGLang backend"
    )

    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=None,
        help="The maximum rank of LoRA adapters. If not specified, it will be automatically inferred from the adapters provided in --lora-paths.",
    )

    parser.add_argument(
        "--lora-target-modules",
        nargs="*",
        type=str,
        default=None,
        help="The union set of all target modules where LoRA should be applied. If not specified, it will be automatically inferred from the adapters provided in --lora-paths. If 'all' is specified, all supported modules will be targeted.",
    )

    parser.add_argument(
        "--lora-paths",
        nargs="*",
        type=str,
        default=None,
        help="The list of LoRA adapters to load. Each adapter must be specified in one of the following formats: <PATH> | <NAME>=<PATH> | JSON with schema {'lora_name':str,'lora_path':str,'pinned':bool}.",
    )

    parser.add_argument(
        "--max-loras-per-batch",
        type=int,
        default=8,
        help="Maximum number of adapters for a running batch, include base-only request.",
    )

    parser.add_argument(
        "--max-loaded-loras",
        type=int,
        default=None,
        help="If specified, it limits the maximum number of LoRA adapters loaded in CPU memory at a time. The value must be greater than or equal to --max-loras-per-batch.",
    )

    parser.add_argument(
        "--lora-eviction-policy",
        choices=["lru", "fifo"],
        default="lru",
        help="LoRA adapter eviction policy when memory pool is full. 'lru': Least Recently Used (default, better cache efficiency). 'fifo': First-In-First-Out.",
    )

    parser.add_argument(
        "--lora-backend",
        choices=["triton", "csgmv"],
        default="triton",
        help="Choose the kernel backend for multi-LoRA serving.",
    )

    parser.add_argument(
        "--max-lora-chunk-size",
        choices=[16, 32, 64, 128],
        type=int,
        default=16,
        help="Maximum chunk size for the ChunkedSGMV LoRA backend. Only used when --lora-backend is 'csgmv'. Choosing a larger value might improve performance.",
    )

    # Tensor parallel configuration
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")

    parser.add_argument(
        "--nccl-port",
        type=int,
        default=None,
        help="The port for NCCL distributed environment setup",
    )

    # Logging and debugging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--gpu-backend",
        type=str,
        default="sglang",
        choices=["sglang", "vllm"],
        help="GPU backend to use",
    )

    parser.add_argument(
        "--use-hfcache",
        action="store_true",
        default=False,
        help="Use local Hugging Face cache only (no network download)",
    )

    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    return args


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate parsed arguments.

    Args:
        args: Parsed arguments namespace

    Raises:
        ValueError: If arguments are invalid
    """
    # Validate layer indices
    if args.start_layer is not None and args.start_layer < 0:
        raise ValueError("start_layer must be non-negative")

    if args.end_layer is not None and args.end_layer <= args.start_layer:
        raise ValueError("end_layer must be greater than start_layer")

    # Validate memory fraction
    if not 0.0 <= args.kv_cache_memory_fraction <= 1.0:
        raise ValueError("kv_cache_memory_fraction must be between 0.0 and 1.0")

    # Validate batch sizes
    if getattr(args, "max_batch_size", None) is not None and args.max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")

    max_seq_len = getattr(args, "max_sequence_length", None)
    if max_seq_len is not None and max_seq_len <= 0:
        raise ValueError("max_sequence_len must be positive")

    if max_seq_len is None and getattr(args, "max_batch_size", None) is None:
        raise ValueError("max_sequence_len or max_batch_size must be provided")

    if args.max_num_tokens_per_batch <= 0:
        raise ValueError("max_num_tokens_per_batch must be positive")

    if args.kv_block_size <= 0:
        raise ValueError("kv_block_size must be positive")

    if args.micro_batch_ratio <= 0:
        raise ValueError("micro_batch_ratio must be positive")

    if args.scheduler_wait_ms < 0:
        raise ValueError("scheduler_wait_ms must be non-negative")

    if getattr(args, "request_timeout_s", None) is not None and args.request_timeout_s <= 0:
        raise ValueError("request_timeout_s must be positive")

    # Validate supported dtypes
    dtype_list = [
        "float16",
        "bfloat16",
        "float32",
    ]
    if args.dtype not in dtype_list:
        raise ValueError(f"Unsupported dtype: {args.dtype}. Supported dtypes: {dtype_list}")
