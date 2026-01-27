"""
Utilities for optional PyTorch profiling in Parallax.

Design goals:
- Opt-in via environment variables (no runtime overhead when disabled)
- Safe/best-effort: never crash serving if profiler fails
- Multi-process friendly: write traces to per-process subdirectories

Environment variables (defaults in parentheses):
- PARALLAX_TORCH_PROFILER_DIR (required to enable): output base directory for traces
  - Compat: VLLM_TORCH_PROFILER_DIR (also accepted)
- PARALLAX_TORCH_PROFILER_WAIT (1)
- PARALLAX_TORCH_PROFILER_WARMUP (1)
- PARALLAX_TORCH_PROFILER_ACTIVE (4)
- PARALLAX_TORCH_PROFILER_REPEAT (1)
- PARALLAX_TORCH_PROFILER_STEP_ON ("batch"): "batch" | "decode"
- PARALLAX_TORCH_PROFILER_RECORD_SHAPES (1)
- PARALLAX_TORCH_PROFILER_PROFILE_MEMORY (1)
- PARALLAX_TORCH_PROFILER_WITH_STACK (0)
- PARALLAX_TORCH_PROFILER_WITH_FLOPS (0)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return str(val)


@dataclass(frozen=True)
class TorchProfilerConfig:
    trace_root_dir: str
    wait: int = 1
    warmup: int = 1
    active: int = 4
    repeat: int = 1
    step_on: str = "batch"  # "batch" | "decode"
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    with_flops: bool = False

    @property
    def total_steps(self) -> int:
        return (max(0, self.wait) + max(0, self.warmup) + max(1, self.active)) * max(1, self.repeat)

    @property
    def step_on_decode_only(self) -> bool:
        return self.step_on.lower() in ("decode", "decode_only")


def load_torch_profiler_config_from_env() -> Optional[TorchProfilerConfig]:
    trace_dir = os.environ.get("PARALLAX_TORCH_PROFILER_DIR") or os.environ.get(
        "VLLM_TORCH_PROFILER_DIR"
    )
    if not trace_dir:
        return None

    wait = max(0, _env_int("PARALLAX_TORCH_PROFILER_WAIT", 1))
    warmup = max(0, _env_int("PARALLAX_TORCH_PROFILER_WARMUP", 1))
    active = max(1, _env_int("PARALLAX_TORCH_PROFILER_ACTIVE", 4))
    repeat = max(1, _env_int("PARALLAX_TORCH_PROFILER_REPEAT", 1))
    step_on = _env_str("PARALLAX_TORCH_PROFILER_STEP_ON", "batch")

    return TorchProfilerConfig(
        trace_root_dir=trace_dir,
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
        step_on=step_on,
        record_shapes=_env_bool("PARALLAX_TORCH_PROFILER_RECORD_SHAPES", True),
        profile_memory=_env_bool("PARALLAX_TORCH_PROFILER_PROFILE_MEMORY", True),
        with_stack=_env_bool("PARALLAX_TORCH_PROFILER_WITH_STACK", False),
        with_flops=_env_bool("PARALLAX_TORCH_PROFILER_WITH_FLOPS", False),
    )


class TorchProfilerController:
    """
    A small controller that wraps torch.profiler.profile with best-effort start/step/stop.
    """

    def __init__(
        self,
        *,
        config: Optional[TorchProfilerConfig],
        device: Optional[str],
        start_layer: int,
        end_layer: int,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        self._cfg = config
        self._device = device
        self._start_layer = start_layer
        self._end_layer = end_layer
        self._tp_rank = tp_rank
        self._tp_size = tp_size

        self._prof = None
        self._started = False
        self._steps_done = 0
        self._trace_dir: Optional[str] = None

        if self._cfg is None:
            return

        # Only support CUDA devices (target use-case for micro-batch debugging).
        if not (self._device and str(self._device).startswith("cuda")):
            logger.info("Torch profiler requested but executor is not on CUDA; skipping.")
            self._cfg = None
            return

        try:
            import torch  # noqa: F401

            if not torch.cuda.is_available():
                logger.info("Torch profiler requested but CUDA is unavailable; skipping.")
                self._cfg = None
                return
        except Exception as e:
            logger.info(f"Torch profiler requested but unavailable: {e}")
            self._cfg = None
            return

        # Create a unique subdir per executor process to avoid trace writer contention.
        subdir = (
            f"layers_{self._start_layer}_{self._end_layer}"
            f"-tp_{self._tp_rank}_{self._tp_size}"
            f"-pid_{os.getpid()}"
        )
        out_dir = os.path.join(self._cfg.trace_root_dir, subdir)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            logger.info(f"Failed to create profiler output dir {out_dir}: {e}")
            self._cfg = None
            return

        self._trace_dir = out_dir

        try:
            from torch.profiler import (
                ProfilerActivity,
                profile,
                schedule,
                tensorboard_trace_handler,
            )

            self._prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(
                    wait=self._cfg.wait,
                    warmup=self._cfg.warmup,
                    active=self._cfg.active,
                    repeat=self._cfg.repeat,
                ),
                on_trace_ready=tensorboard_trace_handler(out_dir),
                record_shapes=self._cfg.record_shapes,
                profile_memory=self._cfg.profile_memory,
                with_stack=self._cfg.with_stack,
                with_flops=self._cfg.with_flops,
            )
            logger.info(
                "Torch profiler enabled. Traces will be written under "
                f"{out_dir} (total_steps={self._cfg.total_steps}, step_on={self._cfg.step_on})."
            )
        except Exception as e:
            logger.info(f"Failed to initialize torch profiler: {e}")
            self._cfg = None
            self._prof = None

    @property
    def enabled(self) -> bool:
        return self._cfg is not None and self._prof is not None and self._trace_dir is not None

    @property
    def started(self) -> bool:
        return self._started

    @property
    def trace_dir(self) -> Optional[str]:
        return self._trace_dir

    def maybe_start(self) -> None:
        if not self.enabled or self._started:
            return
        try:
            self._prof.__enter__()
            self._started = True
            self._steps_done = 0
            logger.info(f"Torch profiler started (trace_dir={self._trace_dir}).")
        except Exception as e:
            logger.info(f"Failed to start torch profiler: {e}")
            self._started = False

    def should_step_for_batch_type(self, batch_type: str) -> bool:
        if not self.enabled or not self._started:
            return False
        if self._cfg is None:
            return False
        if self._cfg.step_on_decode_only:
            return batch_type == "decode_batch"
        return True

    def step(self) -> None:
        if not self.enabled or not self._started:
            return
        try:
            self._prof.step()
            self._steps_done += 1
            if (
                self._cfg is not None
                and self._cfg.total_steps > 0
                and self._steps_done >= self._cfg.total_steps
            ):
                self.stop()
        except Exception:
            # strictly best-effort; never break serving loop
            pass

    def record_function(self, label: str):
        """
        Return a context manager equivalent to torch.profiler.record_function(label).
        If profiler isn't active, returns a no-op context manager.
        """
        if not self.enabled or not self._started:
            return _NullCtx()
        try:
            import torch

            return torch.profiler.record_function(label)
        except Exception:
            return _NullCtx()

    def stop(self) -> None:
        if not self.enabled or not self._started:
            return
        try:
            self._prof.__exit__(None, None, None)
        except Exception:
            pass
        finally:
            logger.info(f"Torch profiler stopped (trace_dir={self._trace_dir}).")
            self._started = False


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
