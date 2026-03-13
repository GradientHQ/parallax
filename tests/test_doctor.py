"""
Tests for the parallax doctor module.

Tests the environment diagnostic checks as specified in issue #307:
- Python version validation
- CUDA/Metal availability detection
- Dependencies validation
- WSL path issues detection
"""

import sys
from unittest.mock import patch

from parallax.doctor.checks import (
    CheckResult,
    CheckStatus,
    check_cuda_availability,
    check_dependencies,
    check_metal_availability,
    check_python_version,
    check_wsl_paths,
    get_all_checks,
    run_all_checks,
)


class TestCheckPythonVersion:
    """Tests for Python version check."""

    def test_python_version_pass(self):
        """Test that current Python version passes if in valid range."""
        result = check_python_version()
        version = sys.version_info

        if (3, 11) <= version < (3, 14):
            assert result.status == CheckStatus.PASS
            assert f"{version.major}.{version.minor}" in result.message
        else:
            assert result.status == CheckStatus.FAIL

    @patch("parallax.doctor.checks.sys")
    def test_python_version_too_old(self, mock_sys):
        """Test that Python < 3.11 fails."""
        mock_sys.version_info = (3, 10, 0)
        result = check_python_version()
        assert result.status == CheckStatus.FAIL
        assert "too old" in result.message

    @patch("parallax.doctor.checks.sys")
    def test_python_version_too_new(self, mock_sys):
        """Test that Python >= 3.14 fails."""
        mock_sys.version_info = (3, 14, 0)
        result = check_python_version()
        assert result.status == CheckStatus.FAIL
        assert "too new" in result.message


class TestCheckCudaAvailability:
    """Tests for CUDA availability check."""

    def test_cuda_no_torch(self):
        """Test when PyTorch is not installed."""
        with patch.dict("sys.modules", {"torch": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = check_cuda_availability()
                assert result.status == CheckStatus.SKIP
                assert "not installed" in result.message

    @patch("parallax.doctor.checks.torch", create=True)
    def test_cuda_available(self, mock_torch):
        """Test when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        mock_torch.version.cuda = "12.1"

        # Need to patch the import
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = check_cuda_availability()
            assert result.status == CheckStatus.PASS
            assert "CUDA" in result.message

    @patch("parallax.doctor.checks.torch", create=True)
    def test_cuda_not_available(self, mock_torch):
        """Test when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = check_cuda_availability()
            assert result.status == CheckStatus.WARN


class TestCheckMetalAvailability:
    """Tests for Metal availability check."""

    @patch("parallax.doctor.checks.platform")
    def test_metal_not_macos(self, mock_platform):
        """Test when not on macOS."""
        mock_platform.system.return_value = "Linux"
        result = check_metal_availability()
        assert result.status == CheckStatus.SKIP
        assert "Not on macOS" in result.message

    @patch("parallax.doctor.checks.platform")
    def test_metal_not_apple_silicon(self, mock_platform):
        """Test when on macOS but not Apple Silicon."""
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "x86_64"
        result = check_metal_availability()
        assert result.status == CheckStatus.SKIP
        assert "Not Apple Silicon" in result.message


class TestCheckDependencies:
    """Tests for dependencies check."""

    def test_dependencies_some_installed(self):
        """Test dependencies check returns valid result."""
        result = check_dependencies()
        assert isinstance(result, CheckResult)
        assert result.name == "Dependencies"
        # Status depends on what's installed
        assert result.status in [CheckStatus.PASS, CheckStatus.FAIL]

    @patch("parallax.doctor.checks.importlib.metadata.version")
    def test_dependencies_all_installed(self, mock_version):
        """Test when all dependencies are installed."""
        mock_version.return_value = "1.0.0"
        result = check_dependencies()
        assert result.status == CheckStatus.PASS

    @patch("parallax.doctor.checks.importlib.metadata.version")
    def test_dependencies_missing(self, mock_version):
        """Test when dependencies are missing."""
        mock_version.side_effect = Exception("Not found")
        result = check_dependencies()
        assert result.status == CheckStatus.FAIL
        assert "missing" in result.message


class TestCheckWslPaths:
    """Tests for WSL path issues check."""

    @patch("parallax.doctor.checks._is_wsl")
    def test_wsl_not_in_wsl(self, mock_is_wsl):
        """Test when not running in WSL."""
        mock_is_wsl.return_value = False
        result = check_wsl_paths()
        assert result.status == CheckStatus.SKIP
        assert "Not running in WSL" in result.message

    @patch("parallax.doctor.checks._is_wsl")
    @patch("parallax.doctor.checks.os.getcwd")
    def test_wsl_windows_path(self, mock_getcwd, mock_is_wsl):
        """Test when running from Windows filesystem in WSL."""
        mock_is_wsl.return_value = True
        mock_getcwd.return_value = "/mnt/c/Users/test/parallax"
        result = check_wsl_paths()
        assert result.status == CheckStatus.WARN
        assert "issue" in result.message

    @patch("parallax.doctor.checks._is_wsl")
    @patch("parallax.doctor.checks.os.getcwd")
    def test_wsl_linux_path(self, mock_getcwd, mock_is_wsl):
        """Test when running from Linux filesystem in WSL."""
        mock_is_wsl.return_value = True
        mock_getcwd.return_value = "/home/user/parallax"
        result = check_wsl_paths()
        assert result.status == CheckStatus.PASS

    @patch("parallax.doctor.checks._is_wsl")
    @patch("parallax.doctor.checks.os.getcwd")
    def test_wsl_path_with_spaces(self, mock_getcwd, mock_is_wsl):
        """Test when path contains spaces."""
        mock_is_wsl.return_value = True
        mock_getcwd.return_value = "/home/user/my project"
        result = check_wsl_paths()
        assert result.status == CheckStatus.WARN
        assert "spaces" in result.details


class TestGetAllChecks:
    """Tests for get_all_checks function."""

    def test_returns_list_of_callables(self):
        """Test that get_all_checks returns a list of callable functions."""
        checks = get_all_checks()
        assert isinstance(checks, list)
        assert len(checks) == 5  # As specified in issue #307
        for check in checks:
            assert callable(check)

    def test_all_checks_return_check_result(self):
        """Test that all checks return CheckResult objects."""
        checks = get_all_checks()
        for check_fn in checks:
            result = check_fn()
            assert isinstance(result, CheckResult)
            assert isinstance(result.status, CheckStatus)
            assert isinstance(result.name, str)
            assert isinstance(result.message, str)


class TestRunAllChecks:
    """Tests for run_all_checks function."""

    def test_run_all_checks_returns_tuple(self, capsys):
        """Test that run_all_checks returns (results, success) tuple."""
        results, success = run_all_checks(verbose=False)
        assert isinstance(results, list)
        assert isinstance(success, bool)
        assert len(results) == 5

    def test_run_all_checks_verbose(self, capsys):
        """Test verbose output."""
        results, success = run_all_checks(verbose=True)
        captured = capsys.readouterr()
        assert "Parallax Doctor" in captured.out
        assert "Summary" in captured.out

    def test_run_all_checks_output(self, capsys):
        """Test that output is printed."""
        run_all_checks(verbose=False)
        captured = capsys.readouterr()
        assert "Parallax Doctor" in captured.out
        assert "diagnostic checks" in captured.out


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_check_result_required_fields(self):
        """Test CheckResult with required fields only."""
        result = CheckResult(
            name="Test",
            status=CheckStatus.PASS,
            message="Test passed",
        )
        assert result.name == "Test"
        assert result.status == CheckStatus.PASS
        assert result.message == "Test passed"
        assert result.details is None

    def test_check_result_with_details(self):
        """Test CheckResult with optional details."""
        result = CheckResult(
            name="Test",
            status=CheckStatus.FAIL,
            message="Test failed",
            details="Detailed error message",
        )
        assert result.details == "Detailed error message"


class TestCheckStatus:
    """Tests for CheckStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert CheckStatus.PASS.value == "pass"
        assert CheckStatus.WARN.value == "warn"
        assert CheckStatus.FAIL.value == "fail"
        assert CheckStatus.SKIP.value == "skip"
