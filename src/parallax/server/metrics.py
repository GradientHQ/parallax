"""
Metrics registry for executor-node telemetry.

Exposes functions to update and retrieve per-node metrics that are consumed by
the P2P server announcements (e.g., current_requests, layer_latency_ms).

Metrics are shared via a shared_state dict for inter-process communication.
The shared_state is process-safe (managed by multiprocessing.Manager), so no thread locks are needed.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Union

# Import SharedState for type hints (avoid circular import)
try:
    from parallax.utils.shared_state import SharedState
except ImportError:
    SharedState = Any  # type: ignore

# Optional publisher for pushing updates to a backend (e.g., central scheduler)
_publisher: Optional[Callable[[Dict[str, Any]], None]] = None

# Shared state dict for inter-process communication
_shared_state: Optional[Dict[str, Any]] = None


def _get_metrics_dict() -> Dict[str, Any]:
    """Get the metrics dictionary to update/read from.

    Returns shared_state["metrics"] which should be a Manager().dict() for proper inter-process sharing.
    """
    global _shared_state
    if _shared_state is not None and "metrics" in _shared_state:
        return _shared_state["metrics"]
    raise RuntimeError("shared_state['metrics'] not initialized. Call set_shared_state() first.")


def update_metrics(
    *,
    current_requests: Optional[int] = None,
    layer_latency_ms_sample: Optional[float] = None,
    ewma_alpha: float = 0.2,
) -> None:
    """Update metrics with optional fields and EWMA smoothing for latency.

    Args:
        current_requests: Number of in-flight requests on this node.
        layer_latency_ms_sample: A new sample of per-layer latency in ms.
        ewma_alpha: Smoothing factor in [0, 1] for latency EWMA.
    """
    metrics = _get_metrics_dict()

    # Update metrics
    if current_requests is not None:
        metrics["current_requests"] = int(current_requests)
    if layer_latency_ms_sample is not None:
        prev = metrics.get("layer_latency_ms")
        if prev is None:
            metrics["layer_latency_ms"] = float(layer_latency_ms_sample)
        else:
            metrics["layer_latency_ms"] = float(
                (1.0 - ewma_alpha) * float(prev) + ewma_alpha * float(layer_latency_ms_sample)
            )
    metrics["_last_update_ts"] = time.time()

    # Publish snapshot if publisher is set
    if _publisher is not None:
        try:
            # Create a snapshot by explicitly accessing each key for Manager().dict()
            snapshot = {k: metrics[k] for k in metrics.keys()}
            _publisher(snapshot)
        except Exception:
            # Best-effort; logging is avoided here to keep this utility lightweight
            pass


def get_metrics() -> Dict[str, Any]:
    """Return a shallow copy of current metrics suitable for JSON serialization.

    For Manager().dict(), we need to access each key explicitly to ensure proper reading.
    """
    metrics_dict = _get_metrics_dict()
    # For Manager().dict(), create a copy by accessing each key explicitly
    return {k: metrics_dict[k] for k in metrics_dict.keys()}


def set_metrics_publisher(publisher: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Register a callback to publish metric snapshots after each update.
    Args:
        publisher: Callable receiving a metrics dict. Set to None to disable publishing.
    """
    global _publisher
    _publisher = publisher


def set_shared_state(shared_state: Union[Dict[str, Any], "SharedState"]) -> None:
    """Set shared state for inter-process metrics sharing.

    Metrics updates will be written to shared_state["metrics"].
    This allows executor processes to share metrics with P2P server processes.

    Args:
        shared_state: SharedState instance or shared dictionary (e.g., from multiprocessing.Manager().dict()).
                     Must contain a "metrics" key with a Manager().dict() value.
    """
    global _shared_state
    # Auto-extract dict from SharedState if needed
    if hasattr(shared_state, "dict"):
        _shared_state = shared_state.dict
    else:
        _shared_state = shared_state
    if "metrics" not in _shared_state:
        raise ValueError("shared_state must contain a 'metrics' key")
