"""
Parallax HTTP router.

This module provides a HTTP router for the Parallax system.
It is responsible for routing requests to the appropriate downstream endpoint.
It is used to balance the load between the downstream endpoints.

Start the router with:
    python src/router/main.py --host 0.0.0.0 --port 8081
"""

import argparse
import asyncio
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, get_args

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from parallax_utils.logging_config import get_logger

try:
    # Prefer package import for tooling/type-checkers.
    from router.lb_strategy import PerformanceConfig, StrategyName, make_strategy
except Exception:  # pragma: no cover
    # Fallback for running this file directly (python src/router/main.py).
    from .lb_strategy import PerformanceConfig, StrategyName, make_strategy

logger = get_logger("router.main")

SUPPORTED_STRATEGIES: Tuple[str, ...] = tuple(get_args(StrategyName))

MAX_REQUEST_SAMPLES = 1000
THROUGHPUT_HISTORY_SEC = 3600
THROUGHPUT_DISPLAY_LAG_SEC = 3


@dataclass
class RouterConfig:
    # Load balancing strategy.
    strategy: StrategyName = "round_robin"

    # Performance-strategy config (also owns the scorer-related knobs).
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # Endpoint readiness check (queried from downstream).
    status_check_path: str = "/cluster/status_json"
    status_check_ttl_sec: float = 3.0
    status_check_timeout_sec: float = 3.0


def load_router_config() -> RouterConfig:
    # Configuration uses defaults and can be updated at runtime via HTTP APIs.
    return RouterConfig()


@dataclass
class EndpointMetrics:
    inflight: int = 0
    total_requests: int = 0
    total_errors: int = 0
    last_error_ts: Optional[float] = None

    # Exponential moving averages in milliseconds
    ema_ttft_ms: Optional[float] = None
    ema_tpot_ms: Optional[float] = None

    # For visibility/debugging
    last_ttft_ms: Optional[float] = None
    last_tpot_ms: Optional[float] = None

    # Downstream readiness status (queried from /cluster/status/onetime)
    last_status_ok: Optional[bool] = None
    last_status: Optional[str] = None
    last_status_ts: Optional[float] = None
    last_status_error: Optional[str] = None

    # Capacity hint from downstream cluster status.
    max_running_request: Optional[int] = None

    # Recent per-request samples for simple UI charting.
    # Each entry: {"ts": float, "ttft_ms": float|None, "tpot_ms": float|None, "output_units": int|None}
    request_samples: deque = field(default_factory=lambda: deque(maxlen=MAX_REQUEST_SAMPLES))


@dataclass
class Endpoint:
    endpoint_id: str
    base_url: str
    enabled: bool = True
    created_ts: float = field(default_factory=time.time)
    metrics: EndpointMetrics = field(default_factory=EndpointMetrics)


class EndpointRegistry:
    def __init__(self, *, config: RouterConfig) -> None:
        self._lock = asyncio.Lock()
        # Use base_url as the single source of truth to avoid state divergence.
        self._endpoints: Dict[str, Endpoint] = {}
        self._client: Optional[httpx.AsyncClient] = None
        self._config = config
        self._strategy = make_strategy(config.strategy, performance_cfg=config.performance)
        # Rolling throughput buckets: base_url -> deque[(unix_sec, count)]
        self._throughput_buckets: Dict[str, deque] = {}

    async def mark_ttft(self, base_url: str, *, ttft_ms: float) -> None:
        async with self._lock:
            ep = self._endpoints.get(base_url)
            if ep is None:
                return
            ep.metrics.last_ttft_ms = float(ttft_ms)
            ep.metrics.ema_ttft_ms = self._ema(ep.metrics.ema_ttft_ms, float(ttft_ms))

    async def mark_throughput_bucket(self, base_url: str, *, sec: int, count: int) -> None:
        if count <= 0:
            return
        async with self._lock:
            # Endpoint might have been unregistered mid-stream.
            if base_url not in self._endpoints:
                return
            self._record_output_buckets(base_url, {int(sec): int(count)})

    def _record_output_buckets(self, base_url: str, buckets: Dict[int, int]) -> None:
        if not buckets:
            return
        q = self._throughput_buckets.get(base_url)
        if q is None:
            q = deque()
            self._throughput_buckets[base_url] = q
        for sec in sorted(buckets.keys()):
            cnt = int(buckets[sec])
            if cnt <= 0:
                continue
            if q and int(q[-1][0]) == int(sec):
                q[-1] = (int(sec), int(q[-1][1]) + cnt)
            else:
                q.append((int(sec), cnt))
        cutoff = int(time.time() - THROUGHPUT_HISTORY_SEC)
        while q and int(q[0][0]) < cutoff:
            q.popleft()

    def _get_throughput_series_1h(self, base_url: str, *, end_sec: int) -> Tuple[int, List[int]]:
        q = self._throughput_buckets.get(base_url)
        end_sec = int(end_sec)
        start_sec = end_sec - (THROUGHPUT_HISTORY_SEC - 1)
        if not q:
            return start_sec, [0 for _ in range(THROUGHPUT_HISTORY_SEC)]

        # Prune lazily.
        while q and int(q[0][0]) < start_sec:
            q.popleft()

        by_sec: Dict[int, int] = {}
        for s, cnt in q:
            si = int(s)
            if si < start_sec or si > end_sec:
                continue
            by_sec[si] = int(by_sec.get(si, 0)) + int(cnt)

        series: List[int] = []
        for sec in range(start_sec, end_sec + 1):
            series.append(int(by_sec.get(sec, 0)))

        # Ensure fixed length.
        if len(series) < THROUGHPUT_HISTORY_SEC:
            series = ([0] * (THROUGHPUT_HISTORY_SEC - len(series))) + series
        elif len(series) > THROUGHPUT_HISTORY_SEC:
            series = series[-THROUGHPUT_HISTORY_SEC:]

        return start_sec, series

    async def get_balancer_config(self) -> Dict[str, Any]:
        async with self._lock:
            c = self._config
            prob = {
                "status_check_path": c.status_check_path,
                "status_check_ttl_sec": c.status_check_ttl_sec,
                "status_check_timeout_sec": c.status_check_timeout_sec,
            }
            cfg: Dict[str, Any] = {}
            if c.strategy == "performance":
                cfg = asdict(c.performance)
            return {
                "strategy": c.strategy,
                "available_strategies": list(SUPPORTED_STRATEGIES),
                "prob": prob,
                "config": cfg,
            }

    async def set_balancer_config(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        def _as_float(v: Any, name: str) -> float:
            if isinstance(v, (int, float)):
                return float(v)
            raise ValueError(f"{name} must be a number")

        def _as_int(v: Any, name: str) -> int:
            if isinstance(v, bool):
                raise ValueError(f"{name} must be an integer")
            if isinstance(v, int):
                return int(v)
            if isinstance(v, float) and float(v).is_integer():
                return int(v)
            raise ValueError(f"{name} must be an integer")

        async with self._lock:
            c = self._config
            # Support both nested and flat payloads.
            strategy_val = patch.get("strategy")
            prob_patch = patch.get("prob") or patch.get("common")
            config_patch = patch.get("config")
            if any(
                k in patch
                for k in ("status_check_path", "status_check_ttl_sec", "status_check_timeout_sec")
            ):
                prob_patch = prob_patch or {}
                for k in ("status_check_path", "status_check_ttl_sec", "status_check_timeout_sec"):
                    if k in patch:
                        prob_patch[k] = patch[k]
            if any(k in patch for k in asdict(c.performance).keys()):
                config_patch = config_patch or {}
                for k in asdict(c.performance).keys():
                    if k in patch:
                        config_patch[k] = patch[k]

            if strategy_val is not None:
                if not isinstance(strategy_val, str):
                    raise ValueError("strategy must be a string")

                if strategy_val not in SUPPORTED_STRATEGIES:
                    raise ValueError(f"strategy must be one of: {', '.join(SUPPORTED_STRATEGIES)}")
                c.strategy = strategy_val  # type: ignore[assignment]

            if prob_patch is not None:
                if not isinstance(prob_patch, dict):
                    raise ValueError("prob must be an object")
                for k, v in prob_patch.items():
                    if k == "status_check_path":
                        if not isinstance(v, str) or not v.strip():
                            raise ValueError("status_check_path must be a non-empty string")
                        c.status_check_path = v.strip()
                    elif k == "status_check_ttl_sec":
                        x = _as_float(v, k)
                        if x < 0:
                            raise ValueError("status_check_ttl_sec must be >= 0")
                        c.status_check_ttl_sec = x
                    elif k == "status_check_timeout_sec":
                        x = _as_float(v, k)
                        if x <= 0:
                            raise ValueError("status_check_timeout_sec must be > 0")
                        c.status_check_timeout_sec = x
                    else:
                        raise ValueError(f"Unknown prob key: {k}")

            if config_patch is not None:
                if not isinstance(config_patch, dict):
                    raise ValueError("config must be an object")
                if c.strategy != "performance":
                    raise ValueError("config is only supported for performance strategy")
                for k, v in config_patch.items():
                    if k == "ema_alpha":
                        x = _as_float(v, k)
                        if not (0.0 < x <= 1.0):
                            raise ValueError("ema_alpha must be in (0, 1]")
                        c.performance.ema_alpha = x
                    elif k == "default_ttft_ms":
                        x = _as_float(v, k)
                        if x <= 0:
                            raise ValueError("default_ttft_ms must be > 0")
                        c.performance.default_ttft_ms = x
                    elif k == "default_tpot_ms":
                        x = _as_float(v, k)
                        if x < 0:
                            raise ValueError("default_tpot_ms must be >= 0")
                        c.performance.default_tpot_ms = x
                    elif k == "inflight_penalty_ms":
                        x = _as_float(v, k)
                        if x < 0:
                            raise ValueError("inflight_penalty_ms must be >= 0")
                        c.performance.inflight_penalty_ms = x
                    elif k == "err_rate_penalty_ms":
                        x = _as_float(v, k)
                        if x < 0:
                            raise ValueError("err_rate_penalty_ms must be >= 0")
                        c.performance.err_rate_penalty_ms = x
                    elif k == "recent_error_window_sec":
                        x = _as_float(v, k)
                        if x < 0:
                            raise ValueError("recent_error_window_sec must be >= 0")
                        c.performance.recent_error_window_sec = x
                    elif k == "recent_error_penalty_ms":
                        x = _as_float(v, k)
                        if x < 0:
                            raise ValueError("recent_error_penalty_ms must be >= 0")
                        c.performance.recent_error_penalty_ms = x
                    elif k == "tpot_weight":
                        x = _as_float(v, k)
                        if x < 0:
                            raise ValueError("tpot_weight must be >= 0")
                        c.performance.tpot_weight = x
                    elif k == "top_k":
                        x = _as_int(v, k)
                        if x < 1:
                            raise ValueError("top_k must be >= 1")
                        c.performance.top_k = x
                    elif k == "explore_ratio":
                        x = _as_float(v, k)
                        if x < 0.0 or x >= 1.0:
                            raise ValueError("explore_ratio must be in [0, 1)")
                        c.performance.explore_ratio = x
                    else:
                        raise ValueError(f"Unknown performance config key: {k}")

            # Rebuild strategy when config changes.
            self._strategy = make_strategy(c.strategy, performance_cfg=c.performance)

            prob = {
                "status_check_path": c.status_check_path,
                "status_check_ttl_sec": c.status_check_ttl_sec,
                "status_check_timeout_sec": c.status_check_timeout_sec,
            }
            cfg: Dict[str, Any] = {}
            if c.strategy == "performance":
                cfg = asdict(c.performance)
            return {
                "updated": list(patch.keys()),
                "strategy": c.strategy,
                "prob": prob,
                "config": cfg,
            }

    async def aclose(self) -> None:
        async with self._lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._lock:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=httpx.Timeout(20 * 60))
            return self._client

    async def register(
        self,
        base_url: str,
        *,
        status_ok: Optional[bool] = None,
        status_val: Optional[str] = None,
        max_running_request: Optional[int] = None,
        status_error: Optional[str] = None,
        status_ts: Optional[float] = None,
    ) -> Endpoint:
        base_url = base_url.strip().rstrip("/")
        if not base_url.startswith("http://") and not base_url.startswith("https://"):
            raise ValueError("base_url must start with http:// or https://")
        async with self._lock:
            existing = self._endpoints.get(base_url)
            if existing is not None:
                existing.enabled = True
                if status_ok is not None:
                    existing.metrics.last_status_ok = status_ok
                    existing.metrics.last_status = status_val
                    existing.metrics.last_status_ts = (
                        time.time() if status_ts is None else status_ts
                    )
                    existing.metrics.last_status_error = status_error
                return existing

            endpoint_id = str(uuid.uuid4())
            ep = Endpoint(endpoint_id=endpoint_id, base_url=base_url, enabled=True)
            if status_ok is not None:
                ep.metrics.last_status_ok = status_ok
                ep.metrics.last_status = status_val
                ep.metrics.last_status_ts = time.time() if status_ts is None else status_ts
                ep.metrics.last_status_error = status_error
                ep.metrics.max_running_request = max_running_request
            self._endpoints[base_url] = ep
            self._throughput_buckets.setdefault(base_url, deque())
            return ep

    async def unregister(self, *, base_url: str) -> int:
        async with self._lock:
            base_url = base_url.strip().rstrip("/")
            removed = 1 if self._endpoints.pop(base_url, None) is not None else 0
            self._throughput_buckets.pop(base_url, None)
            return removed

    async def set_endpoint_enabled(self, *, base_url: str, enabled: bool) -> Endpoint:
        base_url = base_url.strip().rstrip("/")
        async with self._lock:
            ep = self._endpoints.get(base_url)
            if ep is None:
                raise KeyError("Endpoint not found")
            ep.enabled = bool(enabled)
            return ep

    async def list_endpoints(self) -> List[Dict[str, Any]]:
        endpoints = await self._snapshot_endpoints()
        await self.refresh_statuses_if_needed(endpoints)

        out: List[Dict[str, Any]] = []
        end_sec = int(time.time()) - int(THROUGHPUT_DISPLAY_LAG_SEC)
        for ep in endpoints:
            start_sec, series = self._get_throughput_series_1h(ep.base_url, end_sec=end_sec)
            tps = float(series[-1]) if series else 0.0
            item: Dict[str, Any] = {
                "endpoint_id": ep.endpoint_id,
                "base_url": ep.base_url,
                "created_ts": ep.created_ts,
                "enabled": ep.enabled,
                "metrics": {
                    **{k: v for k, v in asdict(ep.metrics).items() if k != "request_samples"},
                    "request_samples": list(ep.metrics.request_samples),
                    "throughput_tok_s": tps,
                    "throughput_series_1h": {"start_sec": start_sec, "tok_s": series},
                },
            }
            if self._config.strategy == "performance":
                item["score"] = float(self._strategy.score(ep))
            out.append(item)
        return out

    def _ema(self, prev: Optional[float], value: float) -> float:
        if prev is None:
            return value
        alpha = self._config.performance.ema_alpha
        return prev * (1 - alpha) + value * alpha

    async def _snapshot_endpoints(self) -> List[Endpoint]:
        async with self._lock:
            return list(self._endpoints.values())

    async def probe_endpoint_status(
        self, base_url: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Probe downstream readiness via GET {base_url}{status_check_path}.

        Returns:
          (ok, status_value, error_message)
        """
        cfg = self._config
        client = await self._get_client()
        url = _join_url(base_url, cfg.status_check_path)
        try:
            resp = await client.get(url, timeout=httpx.Timeout(cfg.status_check_timeout_sec))
            if resp.status_code != 200:
                return False, None, f"Non-200 status: {resp.status_code}"
            data = resp.json()
            status_val = data.get("data", {}).get("status") if isinstance(data, dict) else None
            max_running_request = (
                data.get("data", {}).get("max_running_request") if isinstance(data, dict) else None
            )
            ok = status_val == "available"
            if not ok:
                return False, status_val, 0, f"Not available: {status_val}"
            return True, status_val, max_running_request, None
        except Exception as e:
            return False, None, 0, str(e)

    async def _refresh_endpoint_status_if_needed(self, ep: Endpoint, *, now_ts: float) -> None:
        cfg = self._config
        last_ts = ep.metrics.last_status_ts
        if last_ts is not None and (now_ts - last_ts) < cfg.status_check_ttl_sec:
            return

        ok_raw, status_val, max_running_request, err = await self.probe_endpoint_status(ep.base_url)
        ok: Optional[bool] = bool(ok_raw)

        async with self._lock:
            cur = self._endpoints.get(ep.base_url)
            if cur is None:
                return
            cur.metrics.last_status_ok = ok
            cur.metrics.last_status = status_val
            cur.metrics.max_running_request = max_running_request
            cur.metrics.last_status_ts = now_ts
            cur.metrics.last_status_error = err

    async def refresh_statuses_if_needed(self, endpoints: List[Endpoint]) -> None:
        now_ts = time.time()
        await asyncio.gather(
            *[self._refresh_endpoint_status_if_needed(ep, now_ts=now_ts) for ep in endpoints],
            return_exceptions=False,
        )

    async def broadcast_raw(
        self,
        *,
        path: str,
        headers: Dict[str, str],
        body: bytes,
    ) -> List[Dict[str, Any]]:
        endpoints = await self._snapshot_endpoints()
        if not endpoints:
            raise HTTPException(status_code=503, detail="No downstream endpoints registered")

        client = await self._get_client()

        async def _one(ep: Endpoint) -> Dict[str, Any]:
            url = _join_url(ep.base_url, path)
            try:
                logger.info(f"Broadcasting raw to {url}")
                resp = await client.post(url, headers=headers, content=body)
                content_type = resp.headers.get("content-type", "")
                if "application/json" in content_type.lower():
                    body_out: Any = resp.json()
                else:
                    body_out = resp.text
                return {
                    "endpoint_id": ep.endpoint_id,
                    "base_url": ep.base_url,
                    "ok": 200 <= resp.status_code < 300,
                    "status_code": resp.status_code,
                    "response": body_out,
                }
            except Exception as e:
                return {
                    "endpoint_id": ep.endpoint_id,
                    "base_url": ep.base_url,
                    "ok": False,
                    "status_code": None,
                    "error": str(e),
                }

        results = await asyncio.gather(*[_one(ep) for ep in endpoints], return_exceptions=False)
        return results

    async def choose_best(self) -> Endpoint:
        endpoints = await self._snapshot_endpoints()
        if not endpoints:
            raise HTTPException(status_code=503, detail="No downstream endpoints registered")

        endpoints = [ep for ep in endpoints if ep.enabled]
        if not endpoints:
            raise HTTPException(status_code=503, detail="No enabled downstream endpoints")

        await self.refresh_statuses_if_needed(endpoints)
        healthy = [ep for ep in endpoints if ep.metrics.last_status_ok is True]
        unknown = [ep for ep in endpoints if ep.metrics.last_status_ok is None]
        if healthy:
            endpoints = healthy
        elif unknown:
            endpoints = unknown
        else:
            raise HTTPException(status_code=503, detail="No healthy downstream endpoints")

        try:
            return self._strategy.select(endpoints)
        except ValueError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Strategy error: {e}") from e

    async def mark_start(self, base_url: str) -> None:
        async with self._lock:
            ep = self._endpoints.get(base_url)
            if ep is None:
                return
            ep.metrics.inflight += 1
            ep.metrics.total_requests += 1

    async def mark_error(self, base_url: str) -> None:
        async with self._lock:
            ep = self._endpoints.get(base_url)
            if ep is None:
                return
            ep.metrics.total_errors += 1
            ep.metrics.last_error_ts = time.time()

    async def mark_finish(
        self,
        base_url: str,
        *,
        ttft_ms: Optional[float],
        tpot_ms: Optional[float],
        output_units: Optional[int],
        output_buckets: Optional[Dict[int, int]],
        sample_ttft_ms: Optional[float] = None,
    ) -> None:
        async with self._lock:
            ep = self._endpoints.get(base_url)
            if ep is None:
                return
            ep.metrics.inflight = max(ep.metrics.inflight - 1, 0)

            if ttft_ms is not None:
                ep.metrics.last_ttft_ms = ttft_ms
                ep.metrics.ema_ttft_ms = self._ema(ep.metrics.ema_ttft_ms, ttft_ms)
            if tpot_ms is not None:
                ep.metrics.last_tpot_ms = tpot_ms
                ep.metrics.ema_tpot_ms = self._ema(ep.metrics.ema_tpot_ms, tpot_ms)
            if (
                isinstance(output_units, int)
                and output_units > 0
                and isinstance(output_buckets, dict)
            ):
                self._record_output_buckets(base_url, output_buckets)
            # Store a per-request sample for UI charting.
            ep.metrics.request_samples.append(
                {
                    "ts": time.time(),
                    "ttft_ms": sample_ttft_ms if sample_ttft_ms is not None else ttft_ms,
                    "tpot_ms": tpot_ms,
                    "output_units": output_units,
                }
            )


router_config = load_router_config()
registry = EndpointRegistry(config=router_config)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        await registry.aclose()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _filter_forward_headers(raw_headers: Dict[str, str]) -> Dict[str, str]:
    allowed = {
        "authorization",
        "content-type",
        "accept",
        "user-agent",
        "x-request-id",
    }
    out: Dict[str, str] = {}
    for k, v in raw_headers.items():
        lk = k.lower()
        if lk in allowed:
            out[k] = v
    if "Accept" not in out and "accept" not in out:
        out["Accept"] = "application/json"
    return out


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _extract_sse_data_lines(buffer: bytearray) -> List[bytes]:
    # Very small SSE parser: split by \n, extract lines that start with "data: ".
    # This is sufficient for TTFT/throughput estimation without fully parsing JSON.
    lines: List[bytes] = []
    while True:
        idx = buffer.find(b"\n")
        if idx < 0:
            break
        line = bytes(buffer[:idx]).rstrip(b"\r")
        del buffer[: idx + 1]
        if line.startswith(b"data:"):
            lines.append(line)
    return lines


def _is_done_sse_data_line(line: bytes) -> bool:
    # Accept both: b"data: [DONE]" and with extra spaces.
    tail = line[len(b"data:") :].strip()
    return tail == b"[DONE]"


def _is_contentful_sse_data_line(line: bytes) -> bool:
    # Heuristic: treat any non-DONE data line as a "token event".
    return line.startswith(b"data:") and not _is_done_sse_data_line(line)


async def _proxy_chat_completions_stream(
    *,
    endpoint: Endpoint,
    url: str,
    headers: Dict[str, str],
    request_json: Dict[str, Any],
) -> Tuple[AsyncIterator[bytes], Callable[[], Dict[str, Any]]]:
    start_ts = time.time()
    first_token_ts: Optional[float] = None
    last_token_ts: Optional[float] = None
    token_events = 0
    event_buckets: Dict[int, int] = {}
    cur_sec: Optional[int] = None
    cur_sec_count: int = 0
    cur_sec_flushed: int = 0
    last_flush_ts: float = 0.0

    buffer = bytearray()

    async def gen() -> AsyncIterator[bytes]:
        nonlocal first_token_ts, last_token_ts, token_events, event_buckets, cur_sec, cur_sec_count
        nonlocal cur_sec_flushed, last_flush_ts
        client = await registry._get_client()
        async with client.stream("POST", url, headers=headers, json=request_json) as upstream:
            async for chunk in upstream.aiter_bytes():
                if chunk:
                    buffer.extend(chunk)
                    for line in _extract_sse_data_lines(buffer):
                        if _is_contentful_sse_data_line(line):
                            now = time.time()
                            if first_token_ts is None:
                                first_token_ts = now
                                await registry.mark_ttft(
                                    endpoint.base_url, ttft_ms=(now - start_ts) * 1000.0
                                )
                            last_token_ts = now
                            token_events += 1
                            sec = int(now)
                            event_buckets[sec] = int(event_buckets.get(sec, 0)) + 1
                            if cur_sec is None:
                                cur_sec = sec
                            if sec != cur_sec:
                                # Flush remaining delta for the previous second.
                                delta = int(cur_sec_count - cur_sec_flushed)
                                if delta > 0:
                                    await registry.mark_throughput_bucket(
                                        endpoint.base_url, sec=int(cur_sec), count=delta
                                    )
                                cur_sec = sec
                                cur_sec_count = 0
                                cur_sec_flushed = 0
                                last_flush_ts = now
                            cur_sec_count += 1
                            # Flush at most once per second during streaming so UI sees updates
                            # without waiting for request completion.
                            if (now - last_flush_ts) >= 1.0:
                                delta = int(cur_sec_count - cur_sec_flushed)
                                if delta > 0 and cur_sec is not None:
                                    await registry.mark_throughput_bucket(
                                        endpoint.base_url, sec=int(cur_sec), count=delta
                                    )
                                    cur_sec_flushed += delta
                                last_flush_ts = now
                yield chunk

    def finalize_metrics() -> Dict[str, Any]:
        end_ts = time.time()
        total_ms = (end_ts - start_ts) * 1000.0
        ttft_ms = None if first_token_ts is None else (first_token_ts - start_ts) * 1000.0
        output_units: Optional[int] = int(token_events) if token_events > 0 else None
        if ttft_ms is None or token_events <= 0:
            tpot_ms = None
        else:
            # Approximate TPOT by "per content SSE event" rather than tokenizer tokens.
            gen_ms = max(total_ms - ttft_ms, 0.0)
            tpot_ms = gen_ms / max(token_events, 1)
        return {
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "output_units": output_units,
            # Only include the remaining unflushed delta for the current second to avoid double counting
            # with live updates.
            "output_buckets": (
                {int(cur_sec): int(cur_sec_count - cur_sec_flushed)}
                if cur_sec is not None and (cur_sec_count - cur_sec_flushed) > 0
                else {}
            ),
        }

    # Return a callable so metrics are computed after the stream finishes.
    return gen(), finalize_metrics


@app.get("/health")
async def health() -> JSONResponse:
    """
    Example:
      curl -sS http://127.0.0.1:8081/health
    """
    return JSONResponse(
        content={
            "status": "ok",
            "apis": [
                "/health",
                "/balancer/config",
                "/endpoint/enabled",
                "/register",
                "/unregister",
                "/endpoints",
                "/v1/chat/completions",
                "/weight/refit",
            ],
        }
    )


@app.post("/endpoint/enabled")
async def set_endpoint_enabled(raw_request: Request) -> JSONResponse:
    """
    Enable/disable an endpoint without unregistering it.

    Example:
      curl -sS -X POST http://127.0.0.1:8081/endpoint/enabled \
        -H 'Content-Type: application/json' \
        -d '{"base_url":"http://127.0.0.1:3001","enabled":false}'
    """
    payload = await raw_request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")
    base_url = payload.get("base_url") or payload.get("endpoint")
    if not base_url:
        raise HTTPException(status_code=400, detail="Missing base_url")
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="enabled must be a boolean")
    try:
        ep = await registry.set_endpoint_enabled(base_url=str(base_url), enabled=enabled)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return JSONResponse(content={"base_url": ep.base_url, "enabled": ep.enabled})


@app.get("/balancer/config")
async def get_balancer_config() -> JSONResponse:
    """
    Example:
      curl -sS http://127.0.0.1:8081/balancer/config
    """
    return JSONResponse(content={"config": await registry.get_balancer_config()})


@app.post("/balancer/config")
async def set_balancer_config(raw_request: Request) -> JSONResponse:
    """
    Example:
      curl -sS -X POST http://127.0.0.1:8081/balancer/config \
        -H 'Content-Type: application/json' \
        -d '{"ema_alpha":0.3,"top_k":2}'
    """
    payload = await raw_request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")
    try:
        result = await registry.set_balancer_config(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return JSONResponse(content=result)


@app.post("/register")
async def register(raw_request: Request) -> JSONResponse:
    """
    Example:
      curl -sS -X POST http://127.0.0.1:8081/register \
        -H 'Content-Type: application/json' \
        -d '{"base_url":"http://127.0.0.1:3001"}'
    """
    payload = await raw_request.json()
    base_url = payload.get("endpoint") or payload.get("base_url")
    if not base_url:
        raise HTTPException(status_code=400, detail="Missing endpoint/base_url")
    base_url = str(base_url).strip().rstrip("/")

    # Readiness check before registering (use the same path/logic as routing).
    ok, status_val, max_running_request, err = await registry.probe_endpoint_status(base_url)
    if status_val is None:
        detail = err if err is not None else f"Not available: {status_val}"
        raise HTTPException(status_code=400, detail=f"Endpoint not ready: {detail}")

    try:
        ep = await registry.register(
            base_url,
            status_ok=ok,
            status_val=status_val,
            max_running_request=max_running_request,
            status_error=err,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return JSONResponse(content={"endpoint_id": ep.endpoint_id, "base_url": ep.base_url})


@app.post("/unregister")
async def unregister(raw_request: Request) -> JSONResponse:
    """
    Example:
      curl -sS -X POST http://127.0.0.1:8081/unregister \
        -H 'Content-Type: application/json' \
        -d '{"base_url":"http://127.0.0.1:3001"}'
    """
    payload = await raw_request.json()
    base_url = payload.get("endpoint") or payload.get("base_url")
    if not base_url:
        raise HTTPException(status_code=400, detail="Missing endpoint/base_url")
    removed = await registry.unregister(base_url=str(base_url))
    return JSONResponse(content={"removed": removed})


@app.get("/endpoints")
async def endpoints() -> JSONResponse:
    """
    Example:
      curl -sS http://127.0.0.1:8081/endpoints
    """
    return JSONResponse(content={"endpoints": await registry.list_endpoints()})


@app.post("/weight/refit")
async def weight_refit(raw_request: Request) -> JSONResponse:
    """
    Example:
      curl -sS -X POST http://127.0.0.1:3001/weight/refit \
        -H 'Content-Type: application/json' \
        -d '{
          "time_stamp": "2025-12-17T00:00:00Z",
          "cid": ["cid1","cid2"],
          "index_map": {"weight_a":"cid1"},
          "echo_peer_id": "peer_id",
          "version": "v1"
        }'
    """
    headers = _filter_forward_headers(dict(raw_request.headers))
    body = await raw_request.body()
    results = await registry.broadcast_raw(path="/weight/refit", headers=headers, body=body)
    ok = all(r.get("ok") is True for r in results)
    return JSONResponse(
        status_code=200 if ok else 207,
        content={
            "ok": ok,
            "broadcast_count": len(results),
            "results": results,
        },
    )


@app.post("/v1/chat/completions")
async def v1_chat_completions(raw_request: Request):
    """
    Example:
      curl -sS -X POST http://127.0.0.1:8081/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{
          "model": "your-model",
          "messages": [{"role":"user","content":"Hello"}],
          "stream": true
        }'
    """
    request_json = await raw_request.json()
    is_stream = bool(request_json.get("stream", False))

    ep = await registry.choose_best()
    await registry.mark_start(ep.base_url)
    logger.info(
        "Forwarding /v1/chat/completions to endpoint_id=%s base_url=%s stream=%s",
        ep.endpoint_id,
        ep.base_url,
        is_stream,
    )

    url = _join_url(ep.base_url, "/v1/chat/completions")
    headers = _filter_forward_headers(dict(raw_request.headers))

    if is_stream:
        try:
            stream_iter, metrics_final = await _proxy_chat_completions_stream(
                endpoint=ep,
                url=url,
                headers=headers,
                request_json=request_json,
            )

            async def wrapped() -> AsyncIterator[bytes]:
                try:
                    async for chunk in stream_iter:
                        yield chunk
                except Exception:
                    await registry.mark_error(ep.base_url)
                    raise
                finally:
                    m = metrics_final()
                    await registry.mark_finish(
                        ep.base_url,
                        # TTFT already updated on first token during streaming.
                        ttft_ms=None,
                        tpot_ms=m.get("tpot_ms"),
                        output_units=m.get("output_units"),
                        output_buckets=m.get("output_buckets"),
                        sample_ttft_ms=m.get("ttft_ms"),
                    )

            return StreamingResponse(
                wrapped(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
            )
        except HTTPException:
            await registry.mark_error(ep.base_url)
            await registry.mark_finish(
                ep.base_url, ttft_ms=None, tpot_ms=None, output_units=None, output_buckets=None
            )
            raise
        except Exception as e:
            await registry.mark_error(ep.base_url)
            await registry.mark_finish(
                ep.base_url, ttft_ms=None, tpot_ms=None, output_units=None, output_buckets=None
            )
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e

    start_ts = time.time()
    client = await registry._get_client()
    try:
        resp = await client.post(url, headers=headers, json=request_json)
        latency_ms = (time.time() - start_ts) * 1000.0
        # For non-stream, treat TTFT as full latency.
        await registry.mark_finish(
            ep.base_url, ttft_ms=latency_ms, tpot_ms=None, output_units=None, output_buckets=None
        )
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception as e:
        await registry.mark_error(ep.base_url)
        await registry.mark_finish(
            ep.base_url, ttft_ms=None, tpot_ms=None, output_units=None, output_buckets=None
        )
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e


@app.get("/")
async def endpoints_ui() -> HTMLResponse:
    """
    Simple UI for viewing registered endpoints and their metrics.

    Example:
      open http://127.0.0.1:8081
    """
    ui_path = Path(__file__).resolve().parent / "ui" / "endpoints.html"
    return HTMLResponse(content=ui_path.read_text(encoding="utf-8"), status_code=200)


if __name__ == "__main__":

    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Parallax HTTP router")
        parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen host")
        parser.add_argument("--port", type=int, default=8081, help="Listen port")
        return parser.parse_args()

    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
