"""
Metrics registry for executor-node telemetry.

Exposes functions to update and retrieve per-node metrics that are consumed by
the P2P server announcements (e.g., current_requests, layer_latency_ms).

When running in subprocess mode, metrics are shared via a shared_state dict
for inter-process communication. The shared_state is process-safe (managed by
multiprocessing.Manager), so no thread locks are needed.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

# Optional publisher for pushing updates to a backend (e.g., central scheduler)
_publisher: Optional[Callable[[Dict[str, Any]], None]] = None

# Optional shared state dict for inter-process communication (when running in subprocess mode)
_shared_state: Optional[Dict[str, Any]] = None

# Fallback in-process metrics (for backward compatibility when not using shared_state)
_metrics: Dict[str, Any] = {
    "current_requests": 0,
    "layer_latency_ms": None,
    "_last_update_ts": 0.0,
}


def _get_metrics_dict() -> Dict[str, Any]:
    """Get the metrics dictionary to update/read from.

    Returns shared_state["metrics"] if available, otherwise returns local _metrics.
    Automatically initializes shared_state["metrics"] if needed.
    """
    global _metrics, _shared_state
    if _shared_state is not None:
        try:
            if "metrics" not in _shared_state:
                _shared_state["metrics"] = {
                    "current_requests": 0,
                    "layer_latency_ms": None,
                    "_last_update_ts": 0.0,
                }
            return _shared_state["metrics"]
        except Exception:
            # Fallback to local _metrics if shared_state access fails
            return _metrics
    return _metrics


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
    snapshot = dict(metrics)

    # Publish snapshot if publisher is set
    if _publisher is not None:
        try:
            _publisher(snapshot)
        except Exception:
            # Best-effort; logging is avoided here to keep this utility lightweight
            pass


def get_metrics() -> Dict[str, Any]:
    """Return a shallow copy of current metrics suitable for JSON serialization."""
    return dict(_get_metrics_dict())


def set_metrics_publisher(publisher: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Register a callback to publish metric snapshots after each update.
    Args:
        publisher: Callable receiving a metrics dict. Set to None to disable publishing.
    """
    global _publisher
    _publisher = publisher


def set_shared_state(shared_state: Optional[Dict[str, Any]]) -> None:
    """Set shared state dict for inter-process metrics sharing.

    When provided, metrics updates will be written to shared_state["metrics"].
    This allows executor processes to share metrics with P2P server processes.
    The metrics dict will be automatically initialized on first access.

    Args:
        shared_state: Optional shared dictionary (e.g., from multiprocessing.Manager().dict()).
                     Set to None to disable inter-process sharing.
    """
    global _shared_state
    _shared_state = shared_state
