import json
from unittest.mock import patch

from parallax.utils import model_download


def test_selective_download_fetches_needed_weights_in_one_snapshot_call(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps({"num_hidden_layers": 4}))
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": "layer-0.safetensors",
                    "model.layers.1.self_attn.q_proj.weight": "layer-1-attn.safetensors",
                    "model.layers.1.mlp.up_proj.weight": "layer-1-mlp.safetensors",
                    "model.layers.2.self_attn.q_proj.weight": "layer-2.safetensors",
                }
            }
        )
    )

    with (
        patch(
            "parallax.utils.model_download.download_model_snapshot",
            return_value=model_path,
        ) as download_snapshot,
        patch("parallax.utils.model_download.download_model_file") as download_file,
    ):
        result = model_download.selective_model_download(
            "remote-org/remote-model",
            start_layer=1,
            end_layer=2,
        )

    assert result == model_path
    assert download_snapshot.call_count == 2
    assert download_snapshot.call_args_list[1].kwargs["allow_patterns"] == [
        "layer-1-attn.safetensors",
        "layer-1-mlp.safetensors",
    ]
    download_file.assert_not_called()


def test_selective_download_includes_tied_embeddings_for_last_shard(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 4, "tie_word_embeddings": True})
    )
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "embed.safetensors",
                    "model.layers.3.self_attn.q_proj.weight": "layer-3.safetensors",
                    "model.norm.weight": "final.safetensors",
                }
            }
        )
    )

    needed_files = model_download._determine_needed_weight_files_for_download(
        model_path,
        start_layer=3,
        end_layer=4,
    )

    assert needed_files == [
        "embed.safetensors",
        "final.safetensors",
        "layer-3.safetensors",
    ]
