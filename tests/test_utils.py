from types import SimpleNamespace

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
