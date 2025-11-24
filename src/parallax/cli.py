#!/usr/bin/env python3
"""
Parallax CLI - Command line interface for Parallax distributed LLM serving.

This module provides the main CLI entry point for Parallax, supporting
commands like 'run' and 'join' that mirror the functionality of the
bash scripts.
"""

import argparse
import base64
import json
import os
import signal
import subprocess
import sys

import requests

from parallax_utils.file_util import get_project_root
from parallax_utils.logging_config import get_logger
from parallax_utils.version_check import get_current_version

logger = get_logger("parallax.cli")

PUBLIC_INITIAL_PEERS = [
    "/dns4/bootstrap-lattica.gradient.network/udp/18080/quic-v1/p2p/12D3KooWJHXvu8TWkFn6hmSwaxdCLy4ZzFwr4u5mvF9Fe2rMmFXb",
    "/dns4/bootstrap-lattica.gradient.network/tcp/18080/p2p/12D3KooWJHXvu8TWkFn6hmSwaxdCLy4ZzFwr4u5mvF9Fe2rMmFXb",
    "/dns4/bootstrap-lattica-us.gradient.network/udp/18080/quic-v1/p2p/12D3KooWFD8NoyHfmVxLVCocvXJBjwgE9RZ2bgm2p5WAWQax4FoQ",
    "/dns4/bootstrap-lattica-us.gradient.network/tcp/18080/p2p/12D3KooWFD8NoyHfmVxLVCocvXJBjwgE9RZ2bgm2p5WAWQax4FoQ",
    "/dns4/bootstrap-lattica-eu.gradient.network/udp/18080/quic-v1/p2p/12D3KooWCNuEF4ro95VA4Lgq4NvjdWfJFoTcvWsBA7Z6VkBByPtN",
    "/dns4/bootstrap-lattica-eu.gradient.network/tcp/18080/p2p/12D3KooWCNuEF4ro95VA4Lgq4NvjdWfJFoTcvWsBA7Z6VkBByPtN",
]

PUBLIC_RELAY_SERVERS = [
    "/dns4/relay-lattica.gradient.network/udp/18080/quic-v1/p2p/12D3KooWDaqDAsFupYvffBDxjHHuWmEAJE4sMDCXiuZiB8aG8rjf",
    "/dns4/relay-lattica.gradient.network/tcp/18080/p2p/12D3KooWDaqDAsFupYvffBDxjHHuWmEAJE4sMDCXiuZiB8aG8rjf",
    "/dns4/relay-lattica-us.gradient.network/udp/18080/quic-v1/p2p/12D3KooWHMXi6SCfaQzLcFt6Th545EgRt4JNzxqmDeLs1PgGm3LU",
    "/dns4/relay-lattica-us.gradient.network/tcp/18080/p2p/12D3KooWHMXi6SCfaQzLcFt6Th545EgRt4JNzxqmDeLs1PgGm3LU",
    "/dns4/relay-lattica-eu.gradient.network/udp/18080/quic-v1/p2p/12D3KooWRAuR7rMNA7Yd4S1vgKS6akiJfQoRNNexTtzWxYPiWfG5",
    "/dns4/relay-lattica-eu.gradient.network/tcp/18080/p2p/12D3KooWRAuR7rMNA7Yd4S1vgKS6akiJfQoRNNexTtzWxYPiWfG5",
]


def check_python_version():
    """Check if Python version is 3.11 or higher."""
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        logger.info(
            f"Error: Python 3.11 or higher and less than 3.14 is required. Current version is {sys.version_info.major}.{sys.version_info.minor}."
        )
        sys.exit(1)


def _flag_present(args_list: list[str], flag_names: list[str]) -> bool:
    """Return True if any of the given flags is present in args_list.

    Supports forms: "--flag value", "--flag=value", "-f value", "-f=value".
    """
    if not args_list:
        return False
    flags_set = set(flag_names)
    for i, token in enumerate(args_list):
        if token in flags_set:
            return True
        for flag in flags_set:
            if token.startswith(flag + "="):
                return True
    return False


def _find_flag_value(args_list: list[str], flag_names: list[str]) -> str | None:
    """Find the value for the first matching flag in args_list, if present.

    Returns the associated value for forms: "--flag value" or "--flag=value" or
    "-f value" or "-f=value". Returns None if not found or value is missing.
    """
    if not args_list:
        return None
    flags_set = set(flag_names)
    for i, token in enumerate(args_list):
        if token in flags_set:
            # expect value in next token if exists and is not another flag
            if i + 1 < len(args_list) and not args_list[i + 1].startswith("-"):
                return args_list[i + 1]
            return None
        for flag in flags_set:
            prefix = flag + "="
            if token.startswith(prefix):
                return token[len(prefix) :]
    return None


def _execute_with_graceful_shutdown(cmd: list[str], env: dict[str, str] | None = None) -> None:
    """Execute a command in a subprocess and handle graceful shutdown on Ctrl-C.

    This centralizes the common Popen + signal handling logic shared by
    run_command and join_command.
    """
    logger.info(f"Running command: {' '.join(cmd)}")

    sub_process = None
    try:
        # Start in a new session so we can signal the entire process group
        sub_process = subprocess.Popen(cmd, env=env, start_new_session=True)
        # Wait for the subprocess to finish
        return_code = sub_process.wait()
        if return_code != 0:
            logger.error(f"Command failed with exit code {return_code}")
            sys.exit(return_code)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")

        # If another Ctrl-C arrives during cleanup, force-kill the whole group immediately
        def _force_kill_handler(signum, frame):
            try:
                os.killpg(sub_process.pid, signal.SIGKILL)
            except Exception:
                try:
                    sub_process.kill()
                except Exception:
                    pass
            os._exit(130)

        try:
            signal.signal(signal.SIGINT, _force_kill_handler)
        except Exception:
            pass

        if sub_process is not None:
            try:
                logger.info("Terminating subprocess group...")
                # Gracefully terminate the entire process group
                try:
                    os.killpg(sub_process.pid, signal.SIGINT)
                except Exception:
                    # Fall back to signaling just the child process
                    sub_process.send_signal(signal.SIGINT)

                logger.info("Waiting for subprocess to exit...")
                # Wait for the subprocess to exit gracefully
                try:
                    sub_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.info("SIGINT timeout; sending SIGTERM to process group...")
                    try:
                        os.killpg(sub_process.pid, signal.SIGTERM)
                    except Exception:
                        sub_process.terminate()
                    try:
                        sub_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.info("SIGTERM timeout; forcing SIGKILL on process group...")
                        try:
                            os.killpg(sub_process.pid, signal.SIGKILL)
                        except Exception:
                            sub_process.kill()
                        sub_process.wait()
                logger.info("Subprocess exited.")
            except Exception as e:
                logger.error(f"Failed to terminate subprocess: {e}")
        else:
            logger.info("Subprocess not found, skipping shutdown...")
        sys.exit(0)


def _get_relay_params():
    return [
        "--relay-servers",
        *PUBLIC_RELAY_SERVERS,
        "--initial-peers",
        *PUBLIC_INITIAL_PEERS,
    ]


def run_command(args, passthrough_args: list[str] | None = None):
    """Run the scheduler (equivalent to scripts/start.sh)."""
    if not args.skip_upload:
        update_package_info()

    check_python_version()

    project_root = get_project_root()
    backend_main = project_root / "src" / "backend" / "main.py"

    if not backend_main.exists():
        logger.info(f"Error: Backend main.py not found at {backend_main}")
        sys.exit(1)

    # Build the command to run the backend main.py
    passthrough_args = passthrough_args or []
    cmd = [sys.executable, str(backend_main)]
    if not _flag_present(passthrough_args, ["--port"]):
        cmd.extend(["--port", "3001"])

    # Add optional arguments if provided
    if args.model_name:
        cmd.extend(["--model-name", args.model_name])
    if args.init_nodes_num:
        cmd.extend(["--init-nodes-num", str(args.init_nodes_num)])
    if args.use_relay:
        cmd.extend(_get_relay_params())
        logger.info(
            "Using public relay server to help nodes and the scheduler establish a connection (remote mode). Your IP address will be reported to the relay server to help establish the connection."
        )

    # Append any passthrough args (unrecognized by this CLI) directly to the command
    if passthrough_args:
        cmd.extend(passthrough_args)

    _execute_with_graceful_shutdown(cmd)


def join_command(args, passthrough_args: list[str] | None = None):
    """Join a distributed cluster (equivalent to scripts/join.sh)."""
    if not args.skip_upload:
        update_package_info()

    check_python_version()

    project_root = get_project_root()
    launch_script = project_root / "src" / "parallax" / "launch.py"

    if not launch_script.exists():
        logger.info(f"Error: Launch script not found at {launch_script}")
        sys.exit(1)

    # Set environment variable for the subprocess
    env = os.environ.copy()
    env["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"

    # Build the command to run the launch.py script
    passthrough_args = passthrough_args or []

    cmd = [sys.executable, str(launch_script)]
    if not _flag_present(passthrough_args, ["--max-num-tokens-per-batch"]):
        cmd.extend(["--max-num-tokens-per-batch", "4096"])
    if not _flag_present(passthrough_args, ["--max-sequence-length"]):
        cmd.extend(["--max-sequence-length", "2048"])
    if not _flag_present(passthrough_args, ["--max-batch-size"]):
        cmd.extend(["--max-batch-size", "8"])
    if not _flag_present(passthrough_args, ["--kv-block-size"]):
        cmd.extend(["--kv-block-size", "1024"])

    # The scheduler address is now taken directly from the parsed arguments.
    cmd.extend(["--scheduler-addr", args.scheduler_addr])

    if args.enable_lora:
        logger.info("LoRA adapter support enabled for SGLang backend")
        cmd.append("--enable-lora")

    if args.max_lora_rank is not None: 
        logger.info(f"Setting maximum LoRA rank to: {args.max_lora_rank}")
        cmd.extend(["--max-lora-rank", str(args.max_lora_rank)])

    # 添加 lora-target-modules 参数处理
    # 检查是否提供了目标模块列表
    if args.lora_target_modules is not None:
        # nargs='*' 会返回一个列表，即使只有一个元素或没有元素
        if args.lora_target_modules: # 检查列表是否非空
            # 将列表中的模块名用空格连接成一个字符串
            modules_str = " ".join(args.lora_target_modules)
            logger.info(f"Setting LoRA target modules to: {modules_str}")
            # SGLang 可能期望模块名之间用空格分隔，作为单个参数值传递
            # 或者需要多次传递 --lora-target-modules <module>
            # 根据 SGLang 实际要求调整。假设是空格分隔作为一个参数：
            cmd.extend(["--lora-target-modules", modules_str])
        else:
            # 如果列表为空 (例如用户只写了 --lora-target-modules 但没给值)
            # 可能需要特殊处理或忽略，取决于 SGLang 的行为
            # 这里选择忽略
            logger.warning("--lora-target-modules was provided but no modules were listed. Ignoring.")

    # 添加 lora-paths 参数处理
    # 检查是否提供了 LoRA 路径列表
    if args.lora_paths is not None:
        if args.lora_paths: # 检查列表是否非空
            # 将列表中的路径用空格连接成一个字符串 (假设 SGLang 接受这种方式)
            paths_str = " ".join(args.lora_paths)
            logger.info(f"Loading LoRA adapters from paths: {paths_str}")
            cmd.extend(["--lora-paths", paths_str])
        else:
            # 如果列表为空
            logger.warning("--lora-paths was provided but no paths were listed. Ignoring.")

    if args.max_loras_per_batch is not None:
        logger.info(f"Setting maximum LoRA adapters per batch to: {args.max_loras_per_batch}")
        cmd.extend(["--max-loras-per-batch", str(args.max_loras_per_batch)])

    # 添加 max-loaded-loras 参数处理
    if args.max_loaded_loras is not None:
        logger.info(f"Setting maximum loaded LoRA adapters to: {args.max_loaded_loras}")
        cmd.extend(["--max-loaded-loras", str(args.max_loaded_loras)])

    # 添加 lora-eviction-policy 参数处理
    if args.lora_eviction_policy: # 'lru' 或 'fifo' 都为真
        logger.info(f"Setting LoRA eviction policy to: {args.lora_eviction_policy}")
        cmd.extend(["--lora-eviction-policy", args.lora_eviction_policy])

    if args.lora_backend: # 'triton' 或 'csgmv' 都为真
        logger.info(f"Setting LoRA backend to: {args.lora_backend}")
        cmd.extend(["--lora-backend", args.lora_backend])

    if args.max_lora_chunk_size: # 16, 32, 64, 128 都为真
        # 注意：这个参数可能只在 --lora-backend csgmv 时有效，根据 SGLang 文档
        logger.info(f"Setting maximum LoRA chunk size to: {args.max_lora_chunk_size}")
        cmd.extend(["--max-lora-chunk-size", str(args.max_lora_chunk_size)]) # 转换为字符串

    # Relay logic based on effective scheduler address
    if args.use_relay or (
        args.scheduler_addr != "auto" and not str(args.scheduler_addr).startswith("/")
    ):
        cmd.extend(_get_relay_params())
        logger.info(
            "Using public relay server to help nodes and the scheduler establish a connection (remote mode). Your IP address will be reported to the relay server to help establish the connection."
        )

    # Append any passthrough args (unrecognized by this CLI) directly to the command
    if passthrough_args:
        cmd.extend(passthrough_args)

    logger.info(f"Scheduler address: {args.scheduler_addr}")
    _execute_with_graceful_shutdown(cmd, env=env)


def chat_command(args, passthrough_args: list[str] | None = None):
    """Start the Parallax chat server (equivalent to scripts/chat.sh)."""
    check_python_version()

    project_root = get_project_root()
    launch_script = project_root / "src" / "parallax" / "launch_chat.py"

    if not launch_script.exists():
        logger.info(f"Error: Launch chat script not found at {launch_script}")
        sys.exit(1)

    # Build the command to run the launch_chat.py script
    passthrough_args = passthrough_args or []
    cmd = [sys.executable, str(launch_script)]

    cmd.extend(["--scheduler-addr", args.scheduler_addr])

    # Relay logic based on effective scheduler address
    if args.use_relay or (
        args.scheduler_addr != "auto" and not str(args.scheduler_addr).startswith("/")
    ):
        cmd.extend(_get_relay_params())
        logger.info(
            "Using public relay server to help chat client and the scheduler establish a connection (remote mode). Your IP address will be reported to the relay server to help establish the connection."
        )

    # Append any passthrough args (unrecognized by this CLI) directly to the command
    if passthrough_args:
        cmd.extend(passthrough_args)

    logger.info(f"Scheduler address: {args.scheduler_addr}")
    _execute_with_graceful_shutdown(cmd)


def update_package_info():
    """Update package information."""
    version = get_current_version()

    try:
        package_info = load_package_info()
        if package_info is not None and package_info["version"] == version:
            return

        save_package_info({"version": version})
    except Exception:
        pass


def load_package_info():
    """Load package information."""
    try:
        project_root = get_project_root()
        if not (project_root / ".cache" / "tmp_key.txt").exists():
            return None
        with open(project_root / ".cache" / "tmp_key.txt", "r") as f:
            return json.loads(reversible_decode_string(f.read()))
    except Exception:
        return None


def save_package_info(usage_info: dict):
    """Save package information."""
    project_root = get_project_root()
    os.makedirs(project_root / ".cache", exist_ok=True)
    with open(project_root / ".cache" / "tmp_key.txt", "w") as f:
        f.write(reversible_encode_string(json.dumps(usage_info)))

    upload_package_info(usage_info)


def upload_package_info(usage_info: dict):
    post_url = "https://chatbe-dev.gradient.network/api/v1/parallax/upload"
    headers = {
        "Content-Type": "application/json",
    }
    try:
        requests.post(post_url, headers=headers, json=usage_info, timeout=5)
        return
    except Exception:
        return


def reversible_encode_string(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("utf-8")


def reversible_decode_string(encoded: str) -> str:
    return base64.urlsafe_b64decode(encoded.encode("utf-8")).decode("utf-8")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Parallax - A fully decentralized inference engine developed by Gradient Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  parallax run                                                          # Start scheduler with frontend
  parallax run -m {model-name} -n {number-of-worker-nodes}              # Start scheduler without frontend
  parallax run -m Qwen/Qwen3-0.6B -n 2                                  # example
  parallax join                                                         # Join cluster in local network
  parallax join -s {scheduler-address}                                  # Join cluster in public network
  parallax join -s 12D3KooWLX7MWuzi1Txa5LyZS4eTQ2tPaJijheH8faHggB9SxnBu # example
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add 'run' command parser
    run_parser = subparsers.add_parser(
        "run", help="Start the Parallax scheduler (equivalent to scripts/start.sh)"
    )
    run_parser.add_argument("-n", "--init-nodes-num", type=int, help="Number of initial nodes")
    run_parser.add_argument("-m", "--model-name", type=str, help="Model name")
    run_parser.add_argument(
        "-r", "--use-relay", action="store_true", help="Use public relay servers"
    )
    run_parser.add_argument(
        "-u", "--skip-upload", action="store_true", help="Skip upload package info"
    )

    # Add 'join' command parser
    join_parser = subparsers.add_parser(
        "join", help="Join a distributed cluster (equivalent to scripts/join.sh)"
    )
    join_parser.add_argument(
        "-s",
        "--scheduler-addr",
        default="auto",
        type=str,
        help="Scheduler address (required)",
    )
    join_parser.add_argument(
        "-r", "--use-relay", action="store_true", help="Use public relay servers"
    )
    join_parser.add_argument(
        "-u", "--skip-upload", action="store_true", help="Skip upload package info"
    )


    join_parser.add_argument('--enable-lora', action='store_true',
                        help='Enable LoRA support for the model. This argument is automatically set to True if `--lora-paths` is provided.')


    join_parser.add_argument('--max-lora-rank', type=int, default=None,
                        help='The maximum rank of LoRA adapters. If not specified, it will be automatically inferred from the adapters provided in --lora-paths.')


    join_parser.add_argument('--lora-target-modules', nargs='*', type=str, default=None,
                        help='The union set of all target modules where LoRA should be applied. If not specified, it will be automatically inferred from the adapters provided in --lora-paths. If \'all\' is specified, all supported modules will be targeted.')

    join_parser.add_argument('--lora-paths', nargs='*', type=str, default=None,
                        help='The list of LoRA adapters to load. Each adapter must be specified in one of the following formats: <PATH> | <NAME>=<PATH> | JSON with schema {"lora_name":str,"lora_path":str,"pinned":bool}.')

    join_parser.add_argument('--max-loras-per-batch', type=int, default=8,
                        help='Maximum number of adapters for a running batch, include base-only request.')

    join_parser.add_argument('--max-loaded-loras', type=int, default=None,
                        help='If specified, it limits the maximum number of LoRA adapters loaded in CPU memory at a time. The value must be greater than or equal to `--max-loras-per-batch`.')

    join_parser.add_argument('--lora-eviction-policy', choices=['lru', 'fifo'], default='lru',
                        help='LoRA adapter eviction policy when memory pool is full. \'lru\': Least Recently Used (default, better cache efficiency). \'fifo\': First-In-First-Out.')

    join_parser.add_argument('--lora-backend', choices=['triton', 'csgmv'], default='triton',
                        help='Choose the kernel backend for multi-LoRA serving.')

    join_parser.add_argument('--max-lora-chunk-size', choices=[16, 32, 64, 128], default=16,
                        help='Maximum chunk size for the ChunkedSGMV LoRA backend. Only used when --lora-backend is \'csgmv\'. Choosing a larger value might improve performance.')

    chat_parser = subparsers.add_parser(
        "chat", help="Start the Parallax chat server (equivalent to scripts/chat.sh)"
    )
    chat_parser.add_argument(
        "-s",
        "--scheduler-addr",
        default="auto",
        type=str,
        help="Scheduler address (required)",
    )
    chat_parser.add_argument(
        "-r", "--use-relay", action="store_true", help="Use public relay servers"
    )

    # Accept unknown args and pass them through to the underlying python command
    args, passthrough_args = parser.parse_known_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        run_command(args, passthrough_args)
    elif args.command == "join":
        join_command(args, passthrough_args)
    elif args.command == "chat":
        chat_command(args, passthrough_args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
