"""
Tests for Apple Silicon hardware detection in server_info.

Covers the M5-family table entries and the graceful fallback for unknown
Apple silicon chips (regression for the crash reported in issue #439).
"""

from unittest.mock import patch

from parallax.server.server_info import AppleSiliconHardwareInfo


def _check_output_for(brand: str, gpu_cores=None):
    """Build a subprocess.check_output side_effect simulating a given Mac.

    Args:
        brand: chip name returned for the cpu brand_string query, e.g. "M5 Max".
        gpu_cores: GPU core count returned by system_profiler, or None to
            simulate it being unavailable (e.g. CI / virtualized hosts).
    """

    def _side_effect(cmd, *args, **kwargs):
        if "machdep.cpu.brand_string" in cmd:
            return f"Apple {brand}"
        if "hw.memsize" in cmd:
            return str(64 * 2**30)  # 64 GB
        if "SPDisplaysDataType" in cmd:
            if gpu_cores is None:
                raise FileNotFoundError("system_profiler unavailable")
            return f"Graphics/Displays:\n    Total Number of Cores: {gpu_cores}\n"
        raise ValueError(f"unexpected command: {cmd}")

    return _side_effect


class TestAppleSiliconDetect:
    """AppleSiliconHardwareInfo.detect() chip resolution and fallback."""

    def test_known_chip_uses_table(self):
        """A chip in the table resolves to its tabulated FP16 value."""
        with (
            patch("parallax.server.server_info.psutil", None),
            patch("subprocess.check_output", side_effect=_check_output_for("M5 Max")),
        ):
            info = AppleSiliconHardwareInfo.detect()
        assert info.chip == "Apple M5 Max"
        assert info.tflops_fp16 == AppleSiliconHardwareInfo._APPLE_PEAK_FP16["M5 Max"]

    def test_unknown_chip_estimates_from_gpu_cores(self):
        """An unknown chip is estimated from GPU core count, not crashed on."""
        with (
            patch("parallax.server.server_info.psutil", None),
            patch("subprocess.check_output", side_effect=_check_output_for("M6 Max", gpu_cores=40)),
        ):
            info = AppleSiliconHardwareInfo.detect()
        expected = round(40 * AppleSiliconHardwareInfo._FALLBACK_FP16_PER_CORE, 2)
        assert info.tflops_fp16 == expected

    def test_unknown_chip_without_core_count_uses_default(self):
        """When GPU core count can't be read, a conservative default is used."""
        with (
            patch("parallax.server.server_info.psutil", None),
            patch(
                "subprocess.check_output", side_effect=_check_output_for("M6 Ultra", gpu_cores=None)
            ),
        ):
            info = AppleSiliconHardwareInfo.detect()
        assert info.tflops_fp16 == AppleSiliconHardwareInfo._FALLBACK_FP16_DEFAULT

    def test_unknown_chip_does_not_raise(self):
        """Regression for #439: an unknown chip must not raise."""
        with (
            patch("parallax.server.server_info.psutil", None),
            patch(
                "subprocess.check_output", side_effect=_check_output_for("M99 Ultra", gpu_cores=24)
            ),
        ):
            info = AppleSiliconHardwareInfo.detect()  # must not raise
        assert info.tflops_fp16 > 0

    def test_m5_family_entries_present(self):
        """The M5 family is tabulated (issue #439)."""
        table = AppleSiliconHardwareInfo._APPLE_PEAK_FP16
        assert table["M5"] == 8.08
        assert table["M5 Pro"] == 16.16
        assert table["M5 Max"] == 32.32
