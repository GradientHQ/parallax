"""
ServerInfo that will be announce to DHT and used for client's routing.
    HardwareInfo: Detects and summarizes hardware information, RAM and FLOPs
We haven't used other info, will wait until DHT implemented.
"""

import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Dict, Optional

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_reduce
from mlx_lm.tuner.utils import get_total_parameters

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in some envs
    torch = None

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class HardwareInfo:
    """Generic hardware summary for a peer."""

    total_ram_gb: float
    chip: str
    tflops_fp16: float
    num_gpus: int

    def dumps(self) -> Dict[str, Any]:
        """Serializes the HardwareInfo object to a dictionary."""
        return asdict(self)

    @classmethod
    def loads(cls, obj: Dict[str, Any]) -> "HardwareInfo":
        """Deserializes a dictionary into a HardwareInfo object."""
        return cls(**obj)

    @staticmethod
    def detect() -> "HardwareInfo":
        """Dispatch to the correct subclass for the current machine.

        Prefers CUDA when available; falls back to Apple silicon on macOS.
        """
        if torch is not None and torch.cuda.is_available():
            return NvidiaHardwareInfo.detect()
        if platform.system() == "Darwin" and platform.machine().startswith("arm"):
            return AppleSiliconHardwareInfo.detect()
        raise NotImplementedError("Unsupported hardware; add a subclass.")


@dataclass
class AppleSiliconHardwareInfo(HardwareInfo):
    """HardwareInfo specialised for Apple silicon (M-series)."""

    # From cpu-monkey.com
    _APPLE_PEAK_FP16: ClassVar[Dict[str, float]] = {
        "M1": 4.58,
        "M1 Pro": 10.6,
        "M1 Max": 21.2,
        "M2": 7.1,
        "M2 Pro": 11.36,
        "M2 Max": 26.98,
        "M2 Ultra": 53.96,
        "M3": 7.1,
        "M3 Pro": 9.94,
        "M3 Max": 28.4,
        "M4": 8.52,
        "M4 Pro": 17.04,
        "M4 Max": 34.08,
    }

    @classmethod
    def detect(cls) -> "AppleSiliconHardwareInfo":
        if psutil:
            total_gb = psutil.virtual_memory().total / 2**30
        else:
            total_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30

        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()

        short_name = chip.rsplit("Apple ", maxsplit=1)[-1]
        # For github action, we need to remove the "(Virtual)" suffix
        if short_name.endswith(" (Virtual)"):
            short_name = short_name.rsplit(" (Virtual)", maxsplit=1)[0]
        try:
            flops = cls._APPLE_PEAK_FP16[short_name]
        except KeyError as e:
            raise RuntimeError(
                f"Unknown Apple silicon chip '{short_name}' detected. "
                "Please add it to the _APPLE_PEAK_FP16 dictionary."
            ) from e

        return cls(num_gpus=1, total_ram_gb=round(total_gb, 1), chip=chip, tflops_fp16=flops)


@dataclass
class NvidiaHardwareInfo(HardwareInfo):
    """HardwareInfo specialised for NVIDIA CUDA devices.

    Captures peak FP16 TFLOPS and memory bandwidth using a best-effort mapping
    from device name. VRAM is reported via CUDA device properties.
    """

    vram_gb: float = 0.0
    memory_bandwidth_gbps: float = 0.0

    # Best-effort device database; can be extended as needed
    _GPU_DB: ClassVar[Dict[str, Dict[str, float]]] = {
        # key: substring to match in CUDA device name (case-insensitive)
        "a100-80g": {"tflops_fp16": 312.0, "bandwidth_gbps": 2039.0},
        "a100 80": {"tflops_fp16": 312.0, "bandwidth_gbps": 2039.0},
        "a100-40g": {"tflops_fp16": 312.0, "bandwidth_gbps": 1935.0},
        "a100 40": {"tflops_fp16": 312.0, "bandwidth_gbps": 1935.0},
        "rtx 5090": {"tflops_fp16": 104.8, "bandwidth_gbps": 1792.0},
        "rtx 4090": {"tflops_fp16": 82.6, "bandwidth_gbps": 1008.0},
    }

    @classmethod
    def _match_gpu_specs(cls, name: str, vram_gb: float) -> Dict[str, float]:
        key = name.lower()
        # Specialize A100 by VRAM size when name is generic
        if "a100" in key and ("80" in key or vram_gb >= 60):
            return cls._GPU_DB.get("a100-80g", {"tflops_fp16": 312.0, "bandwidth_gbps": 2039.0})
        if "a100" in key and ("40" in key or vram_gb < 60):
            return cls._GPU_DB.get("a100-40g", {"tflops_fp16": 312.0, "bandwidth_gbps": 1935.0})
        for sub, spec in cls._GPU_DB.items():
            if sub in key:
                return spec
        # Conservative fallback when unknown
        return {"tflops_fp16": 50.0, "bandwidth_gbps": 600.0}

    @classmethod
    def detect(cls) -> "NvidiaHardwareInfo":
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; cannot detect NVIDIA hardware")

        device_count = torch.cuda.device_count()
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        name = getattr(props, "name", f"cuda:{device_index}")
        total_vram_gb = round(props.total_memory / (1024**3), 1)

        # Host RAM (for completeness)
        if psutil:
            total_gb = psutil.virtual_memory().total / 2**30
        else:
            total_gb = 0.0

        spec = cls._match_gpu_specs(name, total_vram_gb)
        return cls(
            num_gpus=device_count,
            total_ram_gb=round(total_gb, 1),
            chip=name,
            tflops_fp16=float(spec["tflops_fp16"]),
            vram_gb=total_vram_gb,
            memory_bandwidth_gbps=float(spec["bandwidth_gbps"]),
        )


def detect_node_hardware(node_id: Optional[str]) -> Dict[str, Any]:
    """Detect local hardware and return a dict for scheduling.

    Returns a dictionary with keys compatible with `NodeHardwareInfo` builder:
    - node_id: The peer/node id
    - tflops_fp16: Peak FP16 TFLOPS
    - memory_gb: Device memory in GB (VRAM for CUDA, total RAM for Apple)
    - memory_bandwidth_gbps: Estimated memory bandwidth in GB/s
    """
    try:
        hw = HardwareInfo.detect()
    except NotImplementedError:
        # Fallback to a conservative default
        return {
            "node_id": node_id,
            "num_gpus": 1,
            "tflops_fp16": 50.0,
            "gpu_name": "Unknown",
            "memory_gb": 16.0,
            "memory_bandwidth_gbps": 100.0,
            "device": "Unknown",
        }

    if isinstance(hw, NvidiaHardwareInfo):
        return {
            "node_id": node_id,
            "num_gpus": hw.num_gpus,
            "tflops_fp16": hw.tflops_fp16,
            "gpu_name": hw.chip,
            "memory_gb": hw.vram_gb,
            "memory_bandwidth_gbps": hw.memory_bandwidth_gbps,
            "device": "cuda",
        }
    if isinstance(hw, AppleSiliconHardwareInfo):
        # Use unified memory size as memory_gb; bandwidth rough estimate per family
        est_bandwidth = 100.0
        return {
            "node_id": node_id,
            "num_gpus": hw.num_gpus,
            "tflops_fp16": hw.tflops_fp16,
            "gpu_name": hw.chip,
            "memory_gb": hw.total_ram_gb,
            "memory_bandwidth_gbps": est_bandwidth,
            "device": "mlx",
        }
    # Generic fallback
    return {
        "node_id": node_id,
        "num_gpus": hw.num_gpus,
        "tflops_fp16": hw.tflops_fp16,
        "gpu_name": "Unknown",
        "memory_gb": 16.0,
        "memory_bandwidth_gbps": 100.0,
        "device": "Unknown",
    }


@dataclass
class ShardedModelInfo:
    """
    Detailed information about the specific model shard hosted by a server.
    """

    model_name: str
    start_layer: int
    end_layer: int
    parameter_count: int = 0
    memory_consumption_mb: float = 0.0

    def dumps(self) -> Dict[str, Any]:
        """Serializes the HardwareInfo object to a dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def loads(cls, data: Dict[str, Any]) -> "ShardedModelInfo":
        """Deserializes a dictionary into a HardwareInfo object."""
        return cls(**data)

    @classmethod
    def from_sharded_model(
        cls, sharded_model_instance: nn.Module  # Instance of your ShardedModel
    ) -> "ShardedModelInfo":
        """
        Constructs ShardedModelInfo from a loaded ShardedModel instance.
        Assumes sharded_model_instance has start_layer, end_layer, and model_id_original attributes.
        """
        # Calculate parameter count
        count = get_total_parameters(sharded_model_instance)

        model_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc,
            sharded_model_instance.parameters(),
            0,
        )
        memory_mb = round(model_bytes / (1024 * 1024), 2)

        return cls(
            model_name=sharded_model_instance.model_id,  # Use the cleaned name
            start_layer=sharded_model_instance.start_layer,
            end_layer=sharded_model_instance.end_layer,
            parameter_count=count,
            memory_consumption_mb=memory_mb,
        )
