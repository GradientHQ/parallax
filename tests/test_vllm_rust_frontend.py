import json
from types import SimpleNamespace

import pytest

from parallax.server import vllm_rust_frontend


def test_resolve_vllm_rs_binary_prefers_path(monkeypatch, tmp_path):
    binary = tmp_path / "vllm-rs"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    monkeypatch.setattr(vllm_rust_frontend.shutil, "which", lambda name: str(binary))

    assert vllm_rust_frontend.resolve_vllm_rs_binary() == str(binary)


def test_resolve_vllm_rs_binary_falls_back_to_python_bin(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    other_bin_dir = tmp_path / "other-bin"
    bin_dir.mkdir()
    other_bin_dir.mkdir()
    python = bin_dir / "python"
    binary = bin_dir / "vllm-rs"
    python.write_text("")
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    monkeypatch.setattr(vllm_rust_frontend.shutil, "which", lambda name: None)
    monkeypatch.setattr(vllm_rust_frontend.sysconfig, "get_path", lambda name: str(other_bin_dir))
    monkeypatch.setattr(vllm_rust_frontend.sys, "executable", str(python))

    assert vllm_rust_frontend.resolve_vllm_rs_binary() == str(binary)


def test_resolve_vllm_rs_binary_falls_back_to_scripts_dir(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    scripts_dir = tmp_path / "scripts"
    bin_dir.mkdir()
    scripts_dir.mkdir()
    python = bin_dir / "python"
    binary = scripts_dir / "vllm-rs"
    python.write_text("")
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    monkeypatch.setattr(vllm_rust_frontend.shutil, "which", lambda name: None)
    monkeypatch.setattr(vllm_rust_frontend.sysconfig, "get_path", lambda name: str(scripts_dir))
    monkeypatch.setattr(vllm_rust_frontend.sys, "executable", str(python))

    assert vllm_rust_frontend.resolve_vllm_rs_binary() == str(binary)


def test_resolve_vllm_rs_binary_raises_when_missing(monkeypatch, tmp_path):
    python = tmp_path / "python"
    scripts_dir = tmp_path / "scripts"
    python.write_text("")
    scripts_dir.mkdir()

    monkeypatch.setattr(vllm_rust_frontend.shutil, "which", lambda name: None)
    monkeypatch.setattr(vllm_rust_frontend.sysconfig, "get_path", lambda name: str(scripts_dir))
    monkeypatch.setattr(vllm_rust_frontend.sys, "executable", str(python))

    with pytest.raises(vllm_rust_frontend.VllmRustFrontendNotFound, match="./install.sh"):
        vllm_rust_frontend.resolve_vllm_rs_binary()


def test_runtime_args_default_to_language_model_only():
    args = SimpleNamespace(model_path="mlx-community/MiniMax-M3-4bit", max_sequence_length=None)

    runtime_args = json.loads(vllm_rust_frontend._runtime_args_json(args))

    assert runtime_args == {
        "model_tag": "mlx-community/MiniMax-M3-4bit",
        "language_model_only": True,
    }


def test_runtime_args_include_max_model_len_when_configured():
    args = SimpleNamespace(model_path="Qwen/Qwen3-0.6B", max_sequence_length=4096)

    runtime_args = json.loads(vllm_rust_frontend._runtime_args_json(args))

    assert runtime_args == {
        "model_tag": "Qwen/Qwen3-0.6B",
        "language_model_only": True,
        "max_model_len": 4096,
    }
