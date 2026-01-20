"""
Environment diagnostic checks for Parallax.

This module implements the environment validation checks specified in issue #307:
    - Python version compatibility
    - CUDA/Metal availability
    - Dependencies validation
    - WSL path issues detection

Each check returns a CheckResult with status (pass/warn/fail/skip) and details.
"""

import importlib.metadata
import os
import platform
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple


class CheckStatus(Enum):
    """Status of a diagnostic check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a diagnostic check.

    Attributes:
        name: Human-readable name of the check.
        status: Status indicating pass, warn, fail, or skip.
        message: Brief summary message.
        details: Optional detailed information for verbose output.
    """

    name: str
    status: CheckStatus
    message: str
    details: Optional[str] = None


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _colorize(text: str, color: str) -> str:
    """Apply ANSI color to text."""
    return f"{color}{text}{Colors.RESET}"


def _status_icon(status: CheckStatus) -> str:
    """Get status icon with color."""
    icons = {
        CheckStatus.PASS: _colorize("✓", Colors.GREEN),
        CheckStatus.WARN: _colorize("⚠", Colors.YELLOW),
        CheckStatus.FAIL: _colorize("✗", Colors.RED),
        CheckStatus.SKIP: _colorize("○", Colors.BLUE),
    }
    return icons.get(status, "?")


def check_python_version() -> CheckResult:
    """Check if Python version is compatible.

    Parallax requires Python >=3.11 and <3.14.

    Returns:
        CheckResult with PASS if version is compatible, FAIL otherwise.
    """
    version = sys.version_info
    # Use index access to support both real version_info and test mocks (plain tuples)
    major, minor, micro = version[0], version[1], version[2]
    version_str = f"{major}.{minor}.{micro}"

    if (major, minor) < (3, 11):
        return CheckResult(
            name="Python Version",
            status=CheckStatus.FAIL,
            message=f"Python {version_str} is too old",
            details="Parallax requires Python >=3.11 and <3.14. Please upgrade.",
        )
    elif (major, minor) >= (3, 14):
        return CheckResult(
            name="Python Version",
            status=CheckStatus.FAIL,
            message=f"Python {version_str} is too new",
            details="Parallax requires Python >=3.11 and <3.14.",
        )
    else:
        return CheckResult(
            name="Python Version",
            status=CheckStatus.PASS,
            message=f"Python {version_str}",
        )


def check_cuda_availability() -> CheckResult:
    """Check if CUDA is available for GPU acceleration.

    Returns:
        CheckResult with PASS if CUDA is available, WARN if not, SKIP if
        PyTorch is not installed.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            cuda_version = torch.version.cuda or "Unknown"
            return CheckResult(
                name="CUDA",
                status=CheckStatus.PASS,
                message=f"Available (CUDA {cuda_version}, {device_count} GPU(s))",
                details=f"Primary GPU: {device_name}",
            )
        else:
            return CheckResult(
                name="CUDA",
                status=CheckStatus.WARN,
                message="Not available",
                details="PyTorch installed but CUDA not available.",
            )
    except ImportError:
        return CheckResult(
            name="CUDA",
            status=CheckStatus.SKIP,
            message="PyTorch not installed",
            details="Install with 'pip install -e .[gpu]' for CUDA support.",
        )
    except Exception as e:
        return CheckResult(
            name="CUDA",
            status=CheckStatus.WARN,
            message="Error checking CUDA",
            details=str(e),
        )


def check_metal_availability() -> CheckResult:
    """Check if Metal (Apple Silicon MLX) is available.

    Returns:
        CheckResult with PASS if Metal is available, SKIP if not on macOS
        or not Apple Silicon, WARN if MLX is not installed.
    """
    if platform.system() != "Darwin":
        return CheckResult(
            name="Metal",
            status=CheckStatus.SKIP,
            message="Not on macOS",
        )

    if not platform.machine().startswith("arm"):
        return CheckResult(
            name="Metal",
            status=CheckStatus.SKIP,
            message="Not Apple Silicon",
        )

    try:
        import mlx.core as mx

        device_info = mx.metal.device_info()
        device_name = device_info.get("device_name", "Unknown")
        return CheckResult(
            name="Metal",
            status=CheckStatus.PASS,
            message=f"Available ({device_name})",
        )
    except ImportError:
        return CheckResult(
            name="Metal",
            status=CheckStatus.WARN,
            message="MLX not installed",
            details="Install with 'pip install -e .[mac]' for Metal support.",
        )
    except (RuntimeError, AttributeError) as e:
        return CheckResult(
            name="Metal",
            status=CheckStatus.WARN,
            message="Metal not accessible",
            details=str(e),
        )


CORE_DEPENDENCIES = [
    "msgpack",
    "safetensors",
    "huggingface-hub",
    "transformers",
    "jinja2",
    "numpy",
    "pyzmq",
    "psutil",
    "httpx",
    "aiohttp",
    "uvicorn",
    "uvloop",
    "fastapi",
    "pydantic",
    "protobuf",
    "dijkstar",
    "lattica",
    "orjson",
]


def check_dependencies() -> CheckResult:
    """Check if required dependencies are installed.

    Validates core dependencies from pyproject.toml.

    Returns:
        CheckResult with PASS if all installed, FAIL if any missing.
    """
    missing = []
    installed_count = 0

    for dep in CORE_DEPENDENCIES:
        try:
            # Handle package names with hyphens (pip uses hyphens, importlib uses underscores)
            pkg_name = dep.replace("-", "_")
            try:
                importlib.metadata.version(dep)
            except importlib.metadata.PackageNotFoundError:
                importlib.metadata.version(pkg_name)
            installed_count += 1
        except Exception:
            missing.append(dep)

    if missing:
        return CheckResult(
            name="Dependencies",
            status=CheckStatus.FAIL,
            message=f"{len(missing)} missing package(s)",
            details=f"Missing: {', '.join(missing)}. Run 'pip install -e .' to install.",
        )
    else:
        return CheckResult(
            name="Dependencies",
            status=CheckStatus.PASS,
            message=f"All {installed_count} core packages installed",
        )


def _is_wsl() -> bool:
    """Detect if running inside Windows Subsystem for Linux."""
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except FileNotFoundError:
        return False


def check_wsl_paths() -> CheckResult:
    """Check for WSL path issues.

    Detects common path problems when running in WSL:
    - Running from Windows filesystem (/mnt/c/...)
    - Spaces in path

    Returns:
        CheckResult with PASS if no issues, WARN if issues detected,
        SKIP if not running in WSL.
    """
    if not _is_wsl():
        return CheckResult(
            name="WSL Paths",
            status=CheckStatus.SKIP,
            message="Not running in WSL",
        )

    cwd = os.getcwd()
    issues = []

    # Check if running from Windows filesystem via WSL
    if cwd.startswith("/mnt/"):
        issues.append(
            "Running from Windows filesystem (/mnt/...) may cause performance issues. "
            "Consider cloning to Linux filesystem (e.g., ~/parallax)."
        )

    # Check for spaces in path
    if " " in cwd:
        issues.append(f"Path contains spaces which may cause issues: '{cwd}'")

    if issues:
        return CheckResult(
            name="WSL Paths",
            status=CheckStatus.WARN,
            message=f"{len(issues)} issue(s) detected",
            details="\n".join(f"• {issue}" for issue in issues),
        )
    else:
        return CheckResult(
            name="WSL Paths",
            status=CheckStatus.PASS,
            message="No path issues detected",
        )


def get_all_checks() -> List[Callable[[], CheckResult]]:
    """Get the list of all diagnostic checks.

    Returns:
        List of check functions as specified in issue #307.
    """
    return [
        check_python_version,
        check_cuda_availability,
        check_metal_availability,
        check_dependencies,
        check_wsl_paths,
    ]


def run_all_checks(verbose: bool = False) -> Tuple[List[CheckResult], bool]:
    """Run all diagnostic checks and display results.

    Args:
        verbose: If True, print detailed output for each check.

    Returns:
        Tuple of (list of CheckResults, success boolean).
        Success is True if no checks failed.
    """
    results = []
    checks = get_all_checks()

    print(f"\n{Colors.BOLD}Parallax Doctor{Colors.RESET}")
    print("=" * 50)
    print(f"Running {len(checks)} diagnostic checks...\n")

    for check_fn in checks:
        try:
            result = check_fn()
        except Exception as e:
            result = CheckResult(
                name=check_fn.__name__.replace("check_", "").replace("_", " ").title(),
                status=CheckStatus.FAIL,
                message="Check raised exception",
                details=str(e),
            )
        results.append(result)

        icon = _status_icon(result.status)
        print(f"  {icon} {result.name}: {result.message}")

        if verbose and result.details:
            for line in result.details.split("\n"):
                print(f"      {_colorize(line, Colors.CYAN)}")

    print("\n" + "=" * 50)
    passed = sum(1 for r in results if r.status == CheckStatus.PASS)
    warned = sum(1 for r in results if r.status == CheckStatus.WARN)
    failed = sum(1 for r in results if r.status == CheckStatus.FAIL)
    skipped = sum(1 for r in results if r.status == CheckStatus.SKIP)

    print(f"{Colors.BOLD}Summary:{Colors.RESET}")
    print(
        f"  {_colorize(f'{passed} passed', Colors.GREEN)}, "
        f"{_colorize(f'{warned} warnings', Colors.YELLOW)}, "
        f"{_colorize(f'{failed} failed', Colors.RED)}, "
        f"{_colorize(f'{skipped} skipped', Colors.BLUE)}"
    )

    if failed > 0:
        print(f"\n{_colorize('Some checks failed. Please fix the issues above.', Colors.RED)}")
        success = False
    elif warned > 0:
        print(f"\n{_colorize('Warnings detected. Review the issues above.', Colors.YELLOW)}")
        success = True
    else:
        print(f"\n{_colorize('All checks passed. Parallax is ready.', Colors.GREEN)}")
        success = True

    print()
    return results, success
