from backend.server import static_config
from backend.server.static_config import MODELS, get_model_info
from parallax.utils.utils import normalize_model_config


def test_glm_5_1_uses_mlx_community_model():
    assert MODELS["zai-org/GLM-5.1"] == "mlx-community/GLM-5.1"


def test_qwen3_6_mxfp4_is_scheduler_supported():
    assert MODELS["mlx-community/Qwen3.6-27B-mxfp4"] == "mlx-community/Qwen3.6-27B-mxfp4"


def test_qwen3_6_mxfp4_model_info_uses_text_config(monkeypatch):
    def fake_load_config_only(model_name, local_files_only=False):
        assert model_name == "mlx-community/Qwen3.6-27B-mxfp4"
        return normalize_model_config(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "quantization_config": {"bits": 4, "mode": "mxfp4"},
                "text_config": {
                    "num_hidden_layers": 64,
                    "head_dim": 256,
                    "hidden_size": 5120,
                    "intermediate_size": 17408,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 4,
                    "vocab_size": 248320,
                },
            }
        )

    monkeypatch.setattr(static_config, "load_config_only", fake_load_config_only)

    model_info = get_model_info("mlx-community/Qwen3.6-27B-mxfp4")

    assert model_info.num_layers == 64
    assert model_info.head_size == 256
    assert model_info.hidden_dim == 5120
    assert model_info.num_attention_heads == 24
    assert model_info.num_kv_heads == 4
    assert model_info.param_bytes_per_element == 0.5
    assert model_info.mlx_param_bytes_per_element == 0.5
