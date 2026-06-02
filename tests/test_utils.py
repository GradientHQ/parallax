from types import SimpleNamespace
from unittest.mock import patch

from parallax.utils import utils


def test_is_metal_available_uses_mlx_metal_is_available(monkeypatch):
    fake_mx = SimpleNamespace(metal=SimpleNamespace(is_available=lambda: True))

    monkeypatch.setattr(utils, "mx", fake_mx)

    assert utils.is_metal_available() is True


def test_is_metal_available_returns_false_when_metal_api_missing(monkeypatch):
    fake_mx = SimpleNamespace()

    monkeypatch.setattr(utils, "mx", fake_mx)

    assert utils.is_metal_available() is False


def test_get_current_device_prefers_mlx_when_metal_available(monkeypatch):
    monkeypatch.setattr(utils, "is_cuda_available", lambda: False)
    monkeypatch.setattr(utils, "is_metal_available", lambda: True)

    assert utils.get_current_device() == "mlx"


def test_get_current_device_prefers_mlx_when_both_backends_report_available(monkeypatch):
    monkeypatch.setattr(utils, "is_cuda_available", lambda: True)
    monkeypatch.setattr(utils, "is_metal_available", lambda: True)

    assert utils.get_current_device() == "mlx"


def test_load_config_only_reads_local_config(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"num_hidden_layers": 2}')

    config = utils.load_config_only(str(model_path))
    assert config == {"num_hidden_layers": 2}


def test_load_config_only_downloads_config_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"num_hidden_layers": 78}')

    with patch("parallax.utils.utils.download_model_file", return_value=config_path) as download:
        config = utils.load_config_only("mlx-community/GLM-5.1", local_files_only=True)

    assert config == {"num_hidden_layers": 78}
    download.assert_called_once_with(
        repo_id="mlx-community/GLM-5.1",
        filename="config.json",
        local_files_only=True,
    )
