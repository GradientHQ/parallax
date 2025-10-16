#!/usr/bin/env python3
"""
Parallax CLI - Command line interface for Parallax distributed LLM serving.

This module provides the main CLI entry point for Parallax, supporting
commands like 'run' and 'join' that mirror the functionality of the
bash scripts.
"""

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

from common.static_config import get_relay_params
from parallax_utils.logging_config import get_logger

logger = get_logger("parallax.cli")


def check_python_version():
    """Check if Python version is 3.11 or higher."""
    if sys.version_info < (3, 11):
        print(
            f"Error: Python 3.11 or higher is required. Current version is {sys.version_info.major}.{sys.version_info.minor}."
        )
        sys.exit(1)


def get_project_root():
    """Get the project root directory."""
    # Search for the project root by looking for pyproject.toml in parent directories
    current_dir = Path(__file__).parent
    while current_dir != current_dir.parent:
        if (current_dir / "pyproject.toml").exists():
            return current_dir
        current_dir = current_dir.parent

    # If not found, fallback to current working directory
    return Path.cwd()


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


def run_command(args, passthrough_args: list[str] | None = None):
    """Run the scheduler (equivalent to scripts/start.sh)."""
    check_python_version()

    project_root = get_project_root()
    backend_main = project_root / "src" / "backend" / "main.py"

    if not backend_main.exists():
        print(f"Error: Backend main.py not found at {backend_main}")
        sys.exit(1)

    # Build the command to run the backend main.py
    passthrough_args = passthrough_args or []
    cmd = [sys.executable, str(backend_main)]
    if not _flag_present(passthrough_args, ["--dht-port"]):
        cmd.extend(["--dht-port", "5001"])
    if not _flag_present(passthrough_args, ["--port"]):
        cmd.extend(["--port", "3001"])

    # Add optional arguments if provided
    if args.model_name:
        cmd.extend(["--model-name", args.model_name])
    if args.init_nodes_num:
        cmd.extend(["--init-nodes-num", str(args.init_nodes_num)])
    if args.use_relay:
        cmd.extend(get_relay_params())

    # Append any passthrough args (unrecognized by this CLI) directly to the command
    if passthrough_args:
        cmd.extend(passthrough_args)

    _execute_with_graceful_shutdown(cmd)


def join_command(args, passthrough_args: list[str] | None = None):
    """Join a distributed cluster (equivalent to scripts/join.sh)."""
    check_python_version()

    project_root = get_project_root()
    launch_script = project_root / "src" / "parallax" / "launch.py"

    if not launch_script.exists():
        print(f"Error: Launch script not found at {launch_script}")
        sys.exit(1)

    # Set environment variable for the subprocess
    env = os.environ.copy()
    env["SGL_ENABLE_JIT_DEEPGEMM"] = "0"

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

    # Relay logic based on effective scheduler address
    if args.use_relay or (
        args.scheduler_addr != "auto" and not str(args.scheduler_addr).startswith("/")
    ):
        logger.info("Using public relay servers")
        cmd.extend(get_relay_params())

    # Append any passthrough args (unrecognized by this CLI) directly to the command
    if passthrough_args:
        cmd.extend(passthrough_args)

    logger.info(f"Scheduler address: {args.scheduler_addr}")
    _execute_with_graceful_shutdown(cmd, env=env)


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

    # Accept unknown args and pass them through to the underlying python command
    args, passthrough_args = parser.parse_known_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        run_command(args, passthrough_args)
    elif args.command == "join":
        join_command(args, passthrough_args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
