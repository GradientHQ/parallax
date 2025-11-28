"""
Tests for the server_args module.
"""

import argparse
from unittest.mock import patch

import pytest

from parallax.server.executor.factory import create_executor_config
from parallax.server.server_args import parse_args, validate_args


class TestValidateArgs:
    """Test argument validation functionality."""

    def test_valid_layer_indices(self):
        """Test valid layer indices."""
        args = argparse.Namespace(
            start_layer=0,
            end_layer=10,
            dtype="bfloat16",
            kv_cache_memory_fraction=0.5,
            max_batch_size=16,
            max_num_tokens_per_batch=1024,
            kv_block_size=16,
            micro_batch_ratio=2,
            scheduler_wait_ms=500,
        )

        # Should not raise any exception
        validate_args(args)

    def test_invalid_start_layer(self):
        """Test invalid start layer."""
        args = argparse.Namespace(
            start_layer=-1,
            end_layer=10,
            dtype="bfloat16",
            kv_cache_memory_fraction=0.5,
            max_batch_size=16,
            max_num_tokens_per_batch=1024,
            kv_block_size=16,
            micro_batch_ratio=2,
            scheduler_wait_ms=500,
        )

        with pytest.raises(ValueError, match="start_layer must be non-negative"):
            validate_args(args)

    def test_invalid_end_layer(self):
        """Test invalid end layer."""
        args = argparse.Namespace(
            start_layer=10,
            end_layer=5,
            dtype="bfloat16",
            kv_cache_memory_fraction=0.5,
            max_batch_size=16,
            max_num_tokens_per_batch=1024,
            kv_block_size=16,
            micro_batch_ratio=2,
            scheduler_wait_ms=500,
        )

        with pytest.raises(ValueError, match="end_layer must be greater than start_layer"):
            validate_args(args)


class TestCreateExecutorConfig:
    """Test executor configuration creation."""

    def test_create_config(self):
        """Test creating executor configuration."""
        args = argparse.Namespace(
            model_path="mlx-community/Qwen3-0.6B-bf16",
            start_layer=0,
            end_layer=10,
            dtype="float16",
            gpu_backend="sglang",
            max_sequence_length=2048,
            max_batch_size=8,
            kv_block_size=64,
            kv_cache_memory_fraction=0.8,
            enable_prefix_cache=False,
            max_num_tokens_per_batch=1024,
            prefill_priority=0,
            micro_batch_ratio=2,
            scheduler_wait_ms=500,
            send_to_peer_addr=None,
            recv_from_peer_addr=None,
            executor_input_ipc="ipc://test_input",
            executor_output_ipc="ipc://test_output",
            attention_backend="flashinfer",
            moe_runner_backend="auto",
            tp_rank=0,
            tp_size=1,
            nccl_port=4000,
            use_hfcache=False,
            enable_lora=False,
            max_lora_rank=None,
            lora_target_modules=None,
            lora_paths=None,
            max_loras_per_batch=1,
            max_loaded_loras=8,
            lora_eviction_policy="lru",
            lora_backend="triton",
            max_lora_chunk_size=128,
        )
        args = argparse.Namespace(
            model_path="mlx-community/Qwen3-0.6B-bf16",
            start_layer=0,
            end_layer=14,
            dtype="bfloat16",
            max_batch_size=16,
            kv_block_size=16,
            kv_cache_memory_fraction=0.8,
            max_num_tokens_per_batch=1024,
            prefill_priority=0,
            micro_batch_ratio=2,
            scheduler_wait_ms=500,
            enable_prefix_cache=True,
            executor_input_ipc="///ipc/1",
            executor_output_ipc="///ipc/2",
            attention_backend="torch_native",
            moe_runner_backend="auto",
            tp_rank=0,
            tp_size=1,
            nccl_port=4001,
            use_hfcache=False,
            enable_lora=False,
            max_lora_rank=None,
            lora_target_modules=None,
            lora_paths=None,
            max_loras_per_batch=1,
            max_loaded_loras=8,
            lora_eviction_policy="lru",
            lora_backend="triton",
            max_lora_chunk_size=128,
        )

        config = create_executor_config(args)

        assert config["model_repo"] == "mlx-community/Qwen3-0.6B-bf16"
        assert config["start_layer"] == 0
        assert config["end_layer"] == 14
        assert config["dtype"] == "bfloat16"
        assert config["kv_cache_memory_fraction"] == 0.8


class TestParseArgs:
    """Test argument parsing with mocked sys.argv."""

    @patch(
        "sys.argv",
        [
            "test_server_args.py",
            "--model-path",
            "mlx-community/Qwen3-0.6B-bf16",
            "--start-layer",
            "0",
            "--end-layer",
            "14",
            "--dtype",
            "bfloat16",
            "--kv-cache-memory-fraction",
            "0.8",
            "--max-sequence-length",
            "1024",
        ],
    )
    def test_parse_valid_args(self):
        """Test parsing valid arguments."""
        args = parse_args()

        assert args.model_path == "mlx-community/Qwen3-0.6B-bf16"
        assert args.start_layer == 0
        assert args.end_layer == 14
        assert args.dtype == "bfloat16"
        assert args.kv_cache_memory_fraction == 0.8

    @patch(
        "sys.argv",
        ["test_server_args.py", "--model-path", "test", "--start-layer", "10", "--end-layer", "5"],
    )
    def test_parse_invalid_args(self):
        """Test parsing invalid arguments."""
        with pytest.raises(ValueError, match="end_layer must be greater than start_layer"):
            parse_args()
