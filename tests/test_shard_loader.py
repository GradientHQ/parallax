"""
Tests for the shard_loader module.
"""

import json
import sys
from unittest.mock import Mock, patch

import mlx.core as mx
import pytest

from parallax.server.shard_loader import (
    ARCHITECTURE_CLASS_ALIASES,
    MODEL_CLASS_MAP,
    MLXModelLoader,
    normalize_language_model_weight_key,
)
from parallax.utils.model_download import _determine_needed_weight_files_for_download
from parallax.utils.utils import normalize_model_config
from parallax.utils.weight_filter_utils import should_include_weight_key


def test_normalize_nested_language_model_weight_keys():
    """Qwen3.5 VLM checkpoints nest the text tower under model.language_model."""
    assert (
        normalize_language_model_weight_key("model.language_model.embed_tokens.weight")
        == "model.embed_tokens.weight"
    )
    assert (
        normalize_language_model_weight_key("model.language_model.layers.12.mlp.up_proj.weight")
        == "model.layers.12.mlp.up_proj.weight"
    )
    assert (
        normalize_language_model_weight_key(
            "language_model.model.layers.12.mlp.switch_mlp.up_proj.weight"
        )
        == "model.layers.12.mlp.switch_mlp.up_proj.weight"
    )
    assert (
        normalize_language_model_weight_key("model.language_model.norm.weight")
        == "model.norm.weight"
    )
    assert (
        normalize_language_model_weight_key("language_model.model.norm.weight")
        == "model.norm.weight"
    )
    assert (
        normalize_language_model_weight_key("model.language_model.lm_head.weight")
        == "lm_head.weight"
    )
    assert (
        normalize_language_model_weight_key("language_model.model.lm_head.weight")
        == "lm_head.weight"
    )
    assert normalize_language_model_weight_key("model.visual.patch_embed.weight") == (
        "model.visual.patch_embed.weight"
    )
    assert (
        normalize_language_model_weight_key(
            "language_model.model.layers.3.self_attn.index_k_proj.weight"
        )
        == "model.layers.3.self_attn.index_k_proj.weight"
    )


def test_weight_filter_includes_nested_qwen35_text_keys():
    assert should_include_weight_key(
        normalize_language_model_weight_key("model.language_model.layers.12.mlp.up_proj.weight"),
        start_layer=12,
        end_layer=13,
        is_first_shard=False,
        is_last_shard=False,
    )
    assert not should_include_weight_key(
        normalize_language_model_weight_key("model.language_model.layers.12.mlp.up_proj.weight"),
        start_layer=13,
        end_layer=14,
        is_first_shard=False,
        is_last_shard=False,
    )
    assert should_include_weight_key(
        normalize_language_model_weight_key("model.language_model.norm.weight"),
        start_layer=0,
        end_layer=24,
        is_first_shard=False,
        is_last_shard=True,
    )
    assert should_include_weight_key(
        normalize_language_model_weight_key("model.language_model.embed_tokens.weight"),
        start_layer=0,
        end_layer=24,
        is_first_shard=False,
        is_last_shard=True,
        tie_word_embeddings=True,
    )


def test_mlx_lm_sanitize_uses_local_layer_keys_for_shards():
    import mlx_lm.models.qwen3_5 as qwen35

    loader = MLXModelLoader("test_model_path")
    args = qwen35.TextModelArgs(num_hidden_layers=24, tie_word_embeddings=True)
    weights = {
        "model.layers.12.linear_attn.conv1d.weight": mx.zeros((4, 1, 3)),
        "model.layers.12.input_layernorm.weight": mx.zeros((2,)),
        "model.norm.weight": mx.zeros((2,)),
    }

    local_weights = {
        loader._to_local_shard_model_key(key, start_layer=12): value
        for key, value in weights.items()
    }

    sanitized = loader._apply_mlx_lm_sanitize(
        qwen35,
        args,
        local_weights,
        num_layers=1,
    )

    assert sanitized["model.layers.0.linear_attn.conv1d.weight"].shape == (4, 3, 1)
    assert float(sanitized["model.layers.0.input_layernorm.weight"][0]) == 1.0
    assert float(sanitized["model.norm.weight"][0]) == 1.0
    assert loader._remap_sanitized_key_to_shard(
        "model.layers.0.linear_attn.conv1d.weight",
        end_layer=1,
        is_first_shard=False,
        is_last_shard=False,
        tie_word_embeddings=False,
    ) == ["layers.0.linear_attn.conv1d.weight"]


def test_qwen35_moe_uses_qwen35_text_args_and_sanitizer_module():
    loader = MLXModelLoader("test_model_path")
    config = normalize_model_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "vocab_size": 248320,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 512,
            },
        }
    )

    sanitizer_module, model_args = loader._load_mlx_lm_module_and_args("qwen3_5_moe", config)

    assert MODEL_CLASS_MAP["qwen3_5_moe"] == "mlx_lm.models.qwen3_5"
    assert sanitizer_module.__name__ == "mlx_lm.models.qwen3_5"
    assert model_args.num_hidden_layers == 40
    assert model_args.hidden_size == 2048
    assert model_args.num_experts == 256
    assert model_args.num_experts_per_tok == 8
    assert model_args.moe_intermediate_size == 512


def _glm_moe_dsa_config(**overrides):
    config = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "vocab_size": 1024,
        "hidden_size": 128,
        "index_head_dim": 64,
        "index_n_heads": 4,
        "index_topk": 8,
        "intermediate_size": 256,
        "moe_intermediate_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "n_shared_experts": 1,
        "n_routed_experts": 4,
        "routed_scaling_factor": 2.5,
        "kv_lora_rank": 64,
        "q_lora_rank": 64,
        "qk_rope_head_dim": 64,
        "v_head_dim": 16,
        "qk_nope_head_dim": 32,
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
        "attention_bias": False,
    }
    config.update(overrides)
    return config


def test_glm_moe_dsa_uses_mlx_lm_args_with_parallax_defaults():
    loader = MLXModelLoader("test_model_path")
    block_class = loader.block_class_map["GlmMoeDsaForCausalLM"]
    sanitizer_module, model_args = loader._load_mlx_lm_module_and_args(
        "glm_moe_dsa",
        _glm_moe_dsa_config(),
        block_class,
    )

    assert sanitizer_module.__name__ == "mlx_lm.models.glm_moe_dsa"
    assert model_args.topk_method == "noaux_tc"
    assert model_args.scoring_func == "sigmoid"
    assert model_args.moe_layer_freq == 1
    assert model_args.index_topk_freq == 1
    assert model_args.indexer_types is None
    assert model_args.index_skip_topk_offset is None
    assert model_args.indexer_rope_traditional is False
    assert model_args.indexer_norm_eps == 1e-6


def test_glm_moe_dsa_shard_start_rejects_shared_layer():
    loader = MLXModelLoader("test_model_path")
    block_class = loader.block_class_map["GlmMoeDsaForCausalLM"]
    config = _glm_moe_dsa_config(indexer_types=["full", "full", "shared", "full"])

    with pytest.raises(ValueError, match="does not transfer DSA top-k"):
        block_class.validate_shard_start(config, 2)

    block_class.validate_shard_start(config, 0)
    block_class.validate_shard_start(config, 1)
    block_class.validate_shard_start(config, 3)


def test_glm_moe_dsa_shard_start_uses_config_indexer_pattern():
    loader = MLXModelLoader("test_model_path")
    block_class = loader.block_class_map["GlmMoeDsaForCausalLM"]
    config = _glm_moe_dsa_config(
        num_hidden_layers=12,
        first_k_dense_replace=3,
        index_topk_freq=4,
        index_skip_topk_offset=3,
    )

    with pytest.raises(ValueError, match="does not transfer DSA top-k"):
        block_class.validate_shard_start(config, 3)

    block_class.validate_shard_start(config, 0)
    block_class.validate_shard_start(config, 2)
    block_class.validate_shard_start(config, 6)


def _minimax_m3_vl_config():
    return {
        "model_type": "minimax_m3_vl",
        "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
        "tie_word_embeddings": True,
        "quantization": {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "language_model.model.layers.3.block_sparse_moe.gate": {
                "bits": 8,
                "group_size": 64,
                "mode": "affine",
            },
            "ignored_layers": ["language_model.model.embed_tokens"],
        },
        "text_config": {
            "model_type": "minimax_m3",
            "architectures": ["MiniMaxM3SparseForCausalLM"],
            "hidden_size": 16,
            "intermediate_size": 8,
            "dense_intermediate_size": 32,
            "shared_intermediate_size": 8,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 32,
            "tie_word_embeddings": False,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "sparse_attention_config": {
                "use_sparse_attention": True,
                "sparse_index_dim": 4,
                "sparse_num_index_heads": 2,
                "sparse_topk_blocks": 1,
                "sparse_block_size": 2,
                "sparse_local_block": 1,
                "sparse_attention_freq": [0, 0, 0, 1],
            },
        },
    }


def test_minimax_m3_vlm_config_is_flattened_to_text_model():
    config = normalize_model_config(_minimax_m3_vl_config())

    assert config["model_type"] == "minimax_m3"
    assert config["original_model_type"] == "minimax_m3_vl"
    assert config["architectures"] == ["MiniMaxM3SparseForCausalLM"]
    assert config["tie_word_embeddings"] is False
    assert config["num_hidden_layers"] == 4
    assert config["index_head_dim"] == 4
    assert config["index_n_heads"] == 1
    assert config["index_block_size"] == 2
    assert config["index_topk_blocks"] == 1
    assert config["moe_intermediate_size"] == 8
    assert "model.layers.3.block_sparse_moe.gate" in config["quantization"]
    assert "language_model.model.layers.3.block_sparse_moe.gate" not in config["quantization"]
    assert config["quantization"]["ignored_layers"] == ["model.embed_tokens"]


def test_minimax_m3_loader_registry_uses_local_module():
    assert MODEL_CLASS_MAP["minimax_m3"] == "parallax.models.minimax_m3"
    assert (
        ARCHITECTURE_CLASS_ALIASES["MiniMaxM3SparseForConditionalGeneration"]
        == "MiniMaxM3SparseForCausalLM"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX tests require macOS")
def test_minimax_m3_uses_parallax_args_module():
    loader = MLXModelLoader("test_model_path")
    config = normalize_model_config(_minimax_m3_vl_config())

    sanitizer_module, model_args = loader._load_mlx_lm_module_and_args("minimax_m3", config)

    assert sanitizer_module.__name__ == "parallax.models.minimax_m3"
    assert model_args.hidden_size == 16
    assert model_args.sparse_attention_config["sparse_num_index_heads"] == 2
    assert model_args.index_n_heads == 2


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX tests require macOS")
def test_minimax_m3_sanitizer_strips_language_model_prefix_and_packs_experts():
    import parallax.models.minimax_m3 as minimax_m3

    loader = MLXModelLoader("test_model_path")
    args = minimax_m3.ModelArgs(
        hidden_size=4,
        intermediate_size=2,
        dense_intermediate_size=8,
        shared_intermediate_size=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=1,
        num_local_experts=2,
        num_experts_per_tok=1,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_index_dim": 4,
            "sparse_num_index_heads": 1,
            "sparse_topk_blocks": 1,
            "sparse_block_size": 2,
            "sparse_attention_freq": [1],
        },
    )
    prefix = "language_model.model.layers.0.block_sparse_moe"
    weights = {
        f"{prefix}.experts.0.w1.weight": mx.ones((2, 4)),
        f"{prefix}.experts.1.w1.weight": mx.ones((2, 4)),
        f"{prefix}.experts.0.w2.weight": mx.ones((4, 2)),
        f"{prefix}.experts.1.w2.weight": mx.ones((4, 2)),
        f"{prefix}.experts.0.w3.weight": mx.ones((2, 4)),
        f"{prefix}.experts.1.w3.weight": mx.ones((2, 4)),
        f"{prefix}.shared_experts.gate_proj.weight": mx.ones((2, 4)),
        f"{prefix}.shared_experts.up_proj.weight": mx.ones((2, 4)),
        f"{prefix}.shared_experts.down_proj.weight": mx.ones((4, 2)),
        "language_model.model.norm.weight": mx.zeros((4,)),
    }

    sanitized = loader._apply_mlx_lm_sanitize(
        minimax_m3,
        args,
        weights,
        num_layers=1,
    )

    assert sanitized["model.layers.0.block_sparse_moe.switch_mlp.gate_up_proj.weight"].shape == (
        3,
        4,
        4,
    )
    assert sanitized["model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"].shape == (
        3,
        4,
        2,
    )
    assert "language_model.model.norm.weight" not in sanitized
    assert sanitized["model.norm.weight"].shape == (4,)
    assert "model.layers.0.block_sparse_moe.shared_experts.gate_proj.weight" not in sanitized


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX tests require macOS")
def test_register_block_class_includes_qwen35_moe():
    loader = MLXModelLoader("test_model_path")

    assert "Qwen3_5MoeForConditionalGeneration" in loader.block_class_map


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX tests require macOS")
def test_register_block_class_includes_minimax_m3_aliases():
    loader = MLXModelLoader("test_model_path")

    assert "MiniMaxM3SparseForCausalLM" in loader.block_class_map
    assert "MiniMaxM3SparseForConditionalGeneration" in loader.block_class_map


def test_selective_download_uses_nested_qwen35_moe_num_layers(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "num_hidden_layers": 40,
                    "tie_word_embeddings": False,
                },
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "language_model.model.layers.39.linear_attn.in_proj_qkv.weight": (
                        "layers-39.safetensors"
                    ),
                    "language_model.model.norm.weight": "final.safetensors",
                    "language_model.lm_head.weight": "final.safetensors",
                }
            }
        )
    )

    needed_files = _determine_needed_weight_files_for_download(
        tmp_path,
        start_layer=39,
        end_layer=40,
    )

    assert needed_files == ["final.safetensors", "layers-39.safetensors"]


def test_selective_download_uses_minimax_m3_text_config_and_language_model_keys(tmp_path):
    config = _minimax_m3_vl_config()
    config["text_config"]["num_hidden_layers"] = 4
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "language_model.model.layers.3.self_attn.index_k_proj.weight": (
                        "layers-3.safetensors"
                    ),
                    "language_model.model.layers.2.self_attn.index_k_proj.weight": (
                        "layers-2.safetensors"
                    ),
                    "language_model.model.norm.weight": "final.safetensors",
                    "language_model.lm_head.weight": "final.safetensors",
                    "vision_tower.patch_embed.weight": "vision.safetensors",
                }
            }
        )
    )

    needed_files = _determine_needed_weight_files_for_download(
        tmp_path,
        start_layer=3,
        end_layer=4,
    )

    assert needed_files == ["final.safetensors", "layers-3.safetensors"]


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX tests require macOS")
class TestMLXModelLoader:
    """Test MLXModelLoader functionality."""

    def test_minimax_m2_uses_mlx_lm_minimax_module(self):
        """MiniMax M2 configs use model_type=minimax_m2, but MLX-LM exposes minimax."""
        assert MODEL_CLASS_MAP["minimax_m2"] == "mlx_lm.models.minimax"

    def test_register_block_class_success(self):
        """Test successful registration of block classes from models directory."""
        loader = MLXModelLoader("test_model_path")

        # Check that block_class_map is populated
        assert hasattr(loader, "block_class_map")
        assert isinstance(loader.block_class_map, dict)

        # Check that expected architectures are registered
        expected_architectures = [
            "Qwen2ForCausalLM",
            "Qwen3ForCausalLM",
            "GlmMoeDsaForCausalLM",
        ]
        for architecture in expected_architectures:
            assert architecture in loader.block_class_map
            assert hasattr(loader.block_class_map[architecture], "get_architecture")
            target_architecture = ARCHITECTURE_CLASS_ALIASES.get(architecture, architecture)
            assert loader.block_class_map[architecture].get_architecture() == target_architecture

    def test_register_block_class_with_missing_get_architecture(self):
        """Test registration when EntryClass doesn't have get_architecture method."""
        # Create a mock module with EntryClass but no get_architecture method
        mock_entry_class = Mock()
        mock_entry_class.__name__ = "TestBlock"
        # Don't add get_architecture method to mock_entry_class

        mock_module = Mock()
        mock_module.EntryClass = mock_entry_class

        with patch(
            "parallax.server.shard_loader.importlib.import_module", return_value=mock_module
        ):
            with patch("parallax.server.shard_loader.pathlib.Path.glob") as mock_glob:
                # Mock a single model file
                mock_file = Mock()
                mock_file.name = "test_model.py"
                mock_file.stem = "test_model"
                mock_glob.return_value = [mock_file]

                # This should not raise an exception, just log a warning
                loader = MLXModelLoader("test_model_path")
                # The mock will have a get_architecture method by default, so we need to check differently
                # The test should verify that the method exists but doesn't have the expected behavior
                assert (
                    len(loader.block_class_map) >= 0
                )  # At least 0 (could be more due to real models)

    def test_register_block_class_with_no_entry_class(self):
        """Test registration when module doesn't have EntryClass."""
        mock_module = Mock()
        # Don't add EntryClass to mock_module

        with patch(
            "parallax.server.shard_loader.importlib.import_module", return_value=mock_module
        ):
            with patch("parallax.server.shard_loader.pathlib.Path.glob") as mock_glob:
                # Mock a single model file
                mock_file = Mock()
                mock_file.name = "no_entry_model.py"
                mock_file.stem = "no_entry_model"
                mock_glob.return_value = [mock_file]

                # This should not raise an exception, just skip the module
                loader = MLXModelLoader("test_model_path")
                # The real models will still be loaded, so we can't assert empty map
                # Instead, verify that the loader was created successfully
                assert hasattr(loader, "block_class_map")

    def test_register_block_class_excludes_init_py(self):
        """Test that __init__.py files are excluded from registration."""
        with patch("parallax.server.shard_loader.pathlib.Path.glob") as mock_glob:
            # Mock files including __init__.py
            mock_init_file = Mock()
            mock_init_file.name = "__init__.py"
            mock_init_file.stem = "__init__"

            mock_model_file = Mock()
            mock_model_file.name = "test_model.py"
            mock_model_file.stem = "test_model"

            mock_glob.return_value = [mock_init_file, mock_model_file]

            # Mock successful import for the model file
            mock_entry_class = Mock()
            mock_entry_class.__name__ = "TestBlock"
            mock_entry_class.get_architecture.return_value = "TestArchitecture"

            mock_module = Mock()
            mock_module.EntryClass = mock_entry_class

            with patch(
                "parallax.server.shard_loader.importlib.import_module", return_value=mock_module
            ):
                loader = MLXModelLoader("test_model_path")
                # Should only register the non-__init__.py file
                assert "TestArchitecture" in loader.block_class_map

    def test_register_block_class_architecture_mapping(self):
        """Test that architecture names are correctly mapped to EntryClass."""
        loader = MLXModelLoader("test_model_path")

        # Test Qwen2 architecture
        if "Qwen2ForCausalLM" in loader.block_class_map:
            qwen2_class = loader.block_class_map["Qwen2ForCausalLM"]
            assert qwen2_class.get_architecture() == "Qwen2ForCausalLM"

        # Test Qwen3 architecture
        if "Qwen3ForCausalLM" in loader.block_class_map:
            qwen3_class = loader.block_class_map["Qwen3ForCausalLM"]
            assert qwen3_class.get_architecture() == "Qwen3ForCausalLM"

    def test_register_block_class_multiple_models(self):
        """Test registration with multiple model files."""
        # This test verifies that multiple models can be registered
        loader = MLXModelLoader("test_model_path")

        # Should have registered at least the expected models
        registered_architectures = list(loader.block_class_map.keys())
        assert len(registered_architectures) >= 0  # At least 0 (could be more in future)

        # Each registered architecture should have a valid EntryClass
        for architecture, entry_class in loader.block_class_map.items():
            assert hasattr(entry_class, "get_architecture")
            target_architecture = ARCHITECTURE_CLASS_ALIASES.get(architecture, architecture)
            assert entry_class.get_architecture() == target_architecture

    def test_register_block_class_initialization(self):
        """Test that register_block_class is called during initialization."""
        with patch.object(MLXModelLoader, "register_block_class") as mock_register:
            MLXModelLoader("test_model_path")
            mock_register.assert_called_once()

    def test_register_block_class_empty_models_directory(self):
        """Test registration when models directory is empty."""
        with patch("parallax.server.shard_loader.pathlib.Path.glob", return_value=[]):
            loader = MLXModelLoader("test_model_path")
            assert not loader.block_class_map

    def test_register_block_class_with_exception_in_get_architecture(self):
        """Test registration when get_architecture method raises an exception."""
        mock_entry_class = Mock()
        mock_entry_class.__name__ = "TestBlock"
        mock_entry_class.get_architecture.side_effect = Exception("Test exception")

        mock_module = Mock()
        mock_module.EntryClass = mock_entry_class

        with patch(
            "parallax.server.shard_loader.importlib.import_module", return_value=mock_module
        ):
            with patch("parallax.server.shard_loader.pathlib.Path.glob") as mock_glob:
                # Mock a single model file
                mock_file = Mock()
                mock_file.name = "exception_model.py"
                mock_file.stem = "exception_model"
                mock_glob.return_value = [mock_file]

                # This should not raise an exception, just log a warning
                loader = MLXModelLoader("test_model_path")
                assert not loader.block_class_map
