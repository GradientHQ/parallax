#!/usr/bin/env python3
"""
Parallax CLI - Command line interface for Parallax distributed LLM serving.

This module provides the main CLI entry point for Parallax, supporting
commands like 'run' and 'join' that mirror the functionality of the
bash scripts.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

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
    # Find the project root by looking for pyproject.toml
    current_dir = Path(__file__).parent
    while current_dir != current_dir.parent:
        if (current_dir / "pyproject.toml").exists():
            return current_dir
        current_dir = current_dir.parent

    # Fallback to current working directory
    return Path.cwd()


def run_command(args):
    """Run the scheduler (equivalent to scripts/start.sh)."""
    check_python_version()

    project_root = get_project_root()
    backend_main = project_root / "src" / "backend" / "main.py"

    if not backend_main.exists():
        print(f"Error: Backend main.py not found at {backend_main}")
        sys.exit(1)

    # Build the command
    cmd = [sys.executable, str(backend_main), "--dht-port", "5001", "--port", "3001"]

    # Add optional arguments
    if args.model_name:
        cmd.extend(["--model-name", args.model_name])
    if args.init_nodes_num:
        cmd.extend(["--init-nodes-num", str(args.init_nodes_num)])

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        sys.exit(0)


def join_command(args):
    """Join a distributed cluster (equivalent to scripts/join.sh)."""
    check_python_version()

    if not args.scheduler_addr:
        print("Error: Scheduler address is required. Use -s or --scheduler-addr")
        sys.exit(1)

    project_root = get_project_root()
    launch_script = project_root / "src" / "parallax" / "launch.py"

    if not launch_script.exists():
        print(f"Error: Launch script not found at {launch_script}")
        sys.exit(1)

    # Set environment variable
    env = os.environ.copy()
    env["SGL_ENABLE_JIT_DEEPGEMM"] = "0"

    # Build the command
    cmd = [
        sys.executable,
        str(launch_script),
        "--max-num-tokens-per-batch",
        "4096",
        "--max-sequence-length",
        "2048",
        "--max-batch-size",
        "8",
        "--kv-block-size",
        "1024",
        "--host",
        "0.0.0.0",
        "--port",
        "3000",
        "--scheduler-addr",
        args.scheduler_addr,
    ]

    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Scheduler address: {args.scheduler_addr}")

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        sys.exit(0)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Parallax - Decentralized pipeline-parallel LLM serving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  parallax run                           # Start scheduler with frontend
  parallax run -m Qwen/Qwen3-0.6B -n 2  # Start scheduler without frontend
  parallax join -s /ip4/192.168.1.2/tcp/5001/p2p/xxxxxxxxxxxx  # Join cluster
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser(
        "run", help="Start the Parallax scheduler (equivalent to scripts/start.sh)"
    )
    run_parser.add_argument("-n", "--init-nodes-num", type=int, help="Number of initial nodes")
    run_parser.add_argument("-m", "--model-name", type=str, help="Model name")

    # Join command
    join_parser = subparsers.add_parser(
        "join", help="Join a distributed cluster (equivalent to scripts/join.sh)"
    )
    join_parser.add_argument(
        "-s", "--scheduler-addr", default="auto", type=str, help="Scheduler address (required)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        run_command(args)
    elif args.command == "join":
        join_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
