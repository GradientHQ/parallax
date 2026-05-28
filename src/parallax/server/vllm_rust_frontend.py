"""Subprocess wrapper for the official vLLM Rust frontend binary."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class VllmRustFrontendNotFound(RuntimeError):
    """Raised when `vllm-rs` cannot be resolved from PATH."""


@dataclass
class VllmRustFrontendProcess:
    process: subprocess.Popen
    listen_fd: int
    host: str
    port: int

    def is_alive(self) -> bool:
        return self.process.poll() is None


def resolve_vllm_rs_binary() -> str:
    """Resolve the official vLLM Rust frontend binary from PATH."""
    binary = shutil.which("vllm-rs")
    if binary is not None:
        return binary

    candidates = [
        Path(sysconfig.get_path("scripts")) / "vllm-rs",
        Path(sys.executable).parent / "vllm-rs",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    raise VllmRustFrontendNotFound(
        "Unable to find `vllm-rs` on PATH or next to the active Python interpreter. "
        "Run `./install.sh`, then activate `.venv` or add `.venv/bin` to PATH."
    )


def _bind_listener_socket(host: str, port: int) -> socket.socket:
    addrinfos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    last_error: Optional[OSError] = None
    for family, socktype, proto, _, sockaddr in addrinfos:
        sock = socket.socket(family, socktype, proto)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(sockaddr)
            sock.set_inheritable(True)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
    assert last_error is not None
    raise last_error


def _runtime_args_json(args) -> str:
    runtime_args = {
        "model_tag": args.model_path,
    }
    if getattr(args, "max_sequence_length", None) is not None:
        runtime_args["max_model_len"] = int(args.max_sequence_length)
    return json.dumps(runtime_args, separators=(",", ":"))


def launch_vllm_rust_frontend(args) -> VllmRustFrontendProcess:
    """Launch `vllm-rs frontend` as Parallax's only HTTP frontend."""
    binary = resolve_vllm_rs_binary()
    listener = _bind_listener_socket(args.host, args.port)
    listen_fd = listener.fileno()
    bound_port = listener.getsockname()[1]
    runtime_args = _runtime_args_json(args)

    cmd = [
        binary,
        "frontend",
        "--listen-fd",
        str(listen_fd),
        "--input-address",
        args.executor_input_ipc,
        "--output-address",
        args.executor_output_ipc,
        "--engine-count",
        "1",
        "--args-json",
        runtime_args,
    ]

    logger.info(
        "Launching vLLM Rust frontend on %s:%s with input=%s output=%s",
        args.host,
        bound_port,
        args.executor_input_ipc,
        args.executor_output_ipc,
    )
    process = subprocess.Popen(
        cmd,
        pass_fds=(listen_fd,),
        env=os.environ.copy(),
    )

    # The child inherited the listener fd; close the parent's copy so shutdown
    # fully releases the port when the Rust frontend exits.
    listener.close()
    time.sleep(0.05)
    if process.poll() is not None:
        raise RuntimeError(f"vLLM Rust frontend exited early with code {process.returncode}")

    if bound_port != args.port:
        args.port = bound_port

    return VllmRustFrontendProcess(
        process=process,
        listen_fd=listen_fd,
        host=args.host,
        port=bound_port,
    )


def stop_vllm_rust_frontend(frontend_process: Optional[VllmRustFrontendProcess]):
    """Terminate the Rust frontend subprocess."""
    if frontend_process is None:
        return None

    process = frontend_process.process
    if process.poll() is not None:
        return frontend_process

    logger.debug("Terminating vLLM Rust frontend subprocess %s", process.pid)
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning("vLLM Rust frontend did not exit after SIGTERM; killing it")
        try:
            process.send_signal(signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    return frontend_process
