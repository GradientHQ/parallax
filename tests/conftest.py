"""
Pytest configuration and fixtures for parallax tests.
"""

import sys

import pytest

from parallax.utils.utils import get_current_device, is_metal_available


def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "metal: mark test as requiring Metal backend")
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA backend")
    config.addinivalue_line("markers", "mlx: mark test as requiring MLX backend (Metal on macOS)")


def pytest_ignore_collect(collection_path, config):
    """Skip collecting parallax_extensions tests on non-macOS platforms"""
    # Check if this is a parallax_extensions test file
    if "parallax_extensions_tests" in str(collection_path):
        # Skip on non-macOS platforms (parallax_extensions requires Metal/MLX)
        if sys.platform != "darwin":
            return True
    return False


@pytest.fixture(scope="session")
def metal_available():
    """Fixture to check if Metal backend is available"""
    return is_metal_available()


@pytest.fixture(scope="session")
def current_device():
    """Fixture to get current device"""
    return get_current_device()


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests that require Metal if Metal is not available"""
    metal_available = is_metal_available()
    current_device = get_current_device()

    for item in items:
        # Skip tests marked with @pytest.mark.metal if Metal is not available
        if item.get_closest_marker("metal") and not metal_available:
            skip_marker = pytest.mark.skip(
                reason="Metal backend not available (requires macOS with Metal support)"
            )
            item.add_marker(skip_marker)

        # Skip tests marked with @pytest.mark.mlx if not on MLX device
        if item.get_closest_marker("mlx") and current_device != "mlx":
            skip_marker = pytest.mark.skip(
                reason=f"MLX device not available (current device: {current_device})"
            )
            item.add_marker(skip_marker)

        # Skip tests marked with @pytest.mark.cuda if CUDA is not available
        if item.get_closest_marker("cuda") and current_device != "cuda":
            skip_marker = pytest.mark.skip(
                reason=f"CUDA device not available (current device: {current_device})"
            )
            item.add_marker(skip_marker)
