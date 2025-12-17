import asyncio
from parallax_utils.logging_config import get_logger
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse


logger = get_logger("router.main")


@dataclass(frozen=True)
class RouterConfig:
    # EMA smoothing factor in (0, 1]. Higher means "more real-time".
    ema_alpha: float = 0.2

    # Defaults used for cold-start endpoints without any metrics yet.
    default_ttft_ms: float = 3000.0
    default_e2el_ms: float = 6000.0

    # Scoring weights (all in milliseconds).
    inflight_penalty_ms: float = 1000.0
    err_rate_penalty_ms: float = 5000.0
    recent_error_window_sec: float = 30.0
    recent_error_penalty_ms: float = 2000.0

    # Selection: pick randomly from top-k (by score). k=1 == strict best.
    top_k: int = 1

    # Exploration ratio in [0, 1). With probability p, pick a random endpoint.
    explore_ratio: float = 0.0

    # Endpoint readiness check (queried from downstream).
    status_check_path: str = "/cluster/status/onetime"
    status_check_ttl_sec: float = 2.0
    status_check_timeout_sec: float = 2.0


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_router_config() -> RouterConfig:
    alpha = _get_env_float("ROUTER_EMA_ALPHA", 0.2)
    # Keep alpha in a sane range to avoid silent misconfiguration.
    if not (0.0 < alpha <= 1.0):
        alpha = 0.2

    top_k = _get_env_int("ROUTER_TOP_K", 1)
    if top_k < 1:
        top_k = 1

    explore_ratio = _get_env_float("ROUTER_EXPLORE_RATIO", 0.0)
    if explore_ratio < 0.0:
        explore_ratio = 0.0
    if explore_ratio >= 1.0:
        explore_ratio = 0.99

    return RouterConfig(
        ema_alpha=alpha,
        default_ttft_ms=_get_env_float("ROUTER_DEFAULT_TTFT_MS", 3000.0),
        default_e2el_ms=_get_env_float("ROUTER_DEFAULT_E2EL_MS", 6000.0),
        inflight_penalty_ms=_get_env_float("ROUTER_INFLIGHT_PENALTY_MS", 1000.0),
        err_rate_penalty_ms=_get_env_float("ROUTER_ERR_RATE_PENALTY_MS", 5000.0),
        recent_error_window_sec=_get_env_float("ROUTER_RECENT_ERROR_WINDOW_SEC", 30.0),
        recent_error_penalty_ms=_get_env_float("ROUTER_RECENT_ERROR_PENALTY_MS", 2000.0),
        top_k=top_k,
        explore_ratio=explore_ratio,
        status_check_path=(os.getenv("ROUTER_STATUS_CHECK_PATH", "/cluster/status/onetime").strip()
        or "/cluster/status/onetime"),
        status_check_ttl_sec=max(0.0, _get_env_float("ROUTER_STATUS_CHECK_TTL_SEC", 2.0)),
        status_check_timeout_sec=max(0.1, _get_env_float("ROUTER_STATUS_CHECK_TIMEOUT_SEC", 2.0)),
    )


@dataclass
class EndpointMetrics:
    inflight: int = 0
    total_requests: int = 0
    total_errors: int = 0
    last_error_ts: Optional[float] = None

    # Exponential moving averages in milliseconds
    ema_ttft_ms: Optional[float] = None
    ema_tpot_ms: Optional[float] = None
    ema_itl_ms: Optional[float] = None
    ema_e2el_ms: Optional[float] = None

    # For visibility/debugging
    last_ttft_ms: Optional[float] = None
    last_tpot_ms: Optional[float] = None
    last_itl_ms: Optional[float] = None
    last_e2el_ms: Optional[float] = None

    # Downstream readiness status (queried from /cluster/status/onetime)
    last_status_ok: Optional[bool] = None
    last_status: Optional[str] = None
    last_status_ts: Optional[float] = None
    last_status_error: Optional[str] = None


@dataclass
class Endpoint:
    endpoint_id: str
    base_url: str
    created_ts: float = field(default_factory=time.time)
    metrics: EndpointMetrics = field(default_factory=EndpointMetrics)


class EndpointRegistry:
    def __init__(self, *, config: RouterConfig) -> None:
        self._lock = asyncio.Lock()
        self._endpoints: Dict[str, Endpoint] = {}
        self._client: Optional[httpx.AsyncClient] = None
        self._config = config

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

    async def register(self, base_url: str) -> Endpoint:
        base_url = base_url.strip().rstrip("/")
        if not base_url.startswith("http://") and not base_url.startswith("https://"):
            raise ValueError("base_url must start with http:// or https://")
        endpoint_id = str(uuid.uuid4())
        ep = Endpoint(endpoint_id=endpoint_id, base_url=base_url)
        async with self._lock:
            self._endpoints[endpoint_id] = ep
        return ep

    async def unregister(self, *, endpoint_id: Optional[str] = None, base_url: Optional[str] = None) -> int:
        async with self._lock:
            if endpoint_id is not None:
                return 1 if self._endpoints.pop(endpoint_id, None) is not None else 0
            if base_url is not None:
                base_url = base_url.strip().rstrip("/")
                removed = [k for k, v in self._endpoints.items() if v.base_url == base_url]
                for k in removed:
                    self._endpoints.pop(k, None)
                return len(removed)
        return 0

    async def list_endpoints(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return [
                {
                    "endpoint_id": ep.endpoint_id,
                    "base_url": ep.base_url,
                    "created_ts": ep.created_ts,
                    "metrics": asdict(ep.metrics),
                }
                for ep in self._endpoints.values()
            ]

    def _ema(self, prev: Optional[float], value: float) -> float:
        if prev is None:
            return value
        alpha = self._config.ema_alpha
        return prev * (1 - alpha) + value * alpha

    async def _snapshot_endpoints(self) -> List[Endpoint]:
        async with self._lock:
            return list(self._endpoints.values())

    async def _refresh_endpoint_status_if_needed(self, ep: Endpoint, *, now_ts: float) -> None:
        cfg = self._config
        last_ts = ep.metrics.last_status_ts
        if last_ts is not None and (now_ts - last_ts) < cfg.status_check_ttl_sec:
            return

        client = await self._get_client()
        url = _join_url(ep.base_url, cfg.status_check_path)
        ok: Optional[bool] = None
        status_val: Optional[str] = None
        err: Optional[str] = None
        try:
            resp = await client.get(url, timeout=httpx.Timeout(cfg.status_check_timeout_sec))
            if resp.status_code != 200:
                ok = False
                err = f"Non-200 status: {resp.status_code}"
            else:
                data = resp.json()
                if isinstance(data, dict):
                    status_val = data.get("data", {}).get("status")
                ok = status_val == "available"
                if not ok:
                    err = f"Not available: {status_val}"
        except Exception as e:
            ok = False
            err = str(e)

        async with self._lock:
            cur = self._endpoints.get(ep.endpoint_id)
            if cur is None:
                return
            cur.metrics.last_status_ok = ok
            cur.metrics.last_status = status_val
            cur.metrics.last_status_ts = now_ts
            cur.metrics.last_status_error = err

    async def refresh_statuses_if_needed(self, endpoints: List[Endpoint]) -> None:
        now_ts = time.time()
        await asyncio.gather(
            *[self._refresh_endpoint_status_if_needed(ep, now_ts=now_ts) for ep in endpoints],
            return_exceptions=False,
        )

    async def broadcast_json(
        self,
        *,
        path: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        endpoints = await self._snapshot_endpoints()
        if not endpoints:
            raise HTTPException(status_code=503, detail="No downstream endpoints registered")

        client = await self._get_client()

        async def _one(ep: Endpoint) -> Dict[str, Any]:
            url = _join_url(ep.base_url, path)
            try:
                resp = await client.post(url, headers=headers, json=payload)
                content_type = resp.headers.get("content-type", "")
                if "application/json" in content_type.lower():
                    body: Any = resp.json()
                else:
                    body = resp.text
                return {
                    "endpoint_id": ep.endpoint_id,
                    "base_url": ep.base_url,
                    "ok": 200 <= resp.status_code < 300,
                    "status_code": resp.status_code,
                    "response": body,
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

        await self.refresh_statuses_if_needed(endpoints)
        healthy = [ep for ep in endpoints if ep.metrics.last_status_ok is True]
        unknown = [ep for ep in endpoints if ep.metrics.last_status_ok is None]
        if healthy:
            endpoints = healthy
        elif unknown:
            endpoints = unknown
        else:
            raise HTTPException(status_code=503, detail="No healthy downstream endpoints")

        def score(ep: Endpoint) -> float:
            m = ep.metrics
            cfg = self._config
            inflight_penalty = float(m.inflight) * float(cfg.inflight_penalty_ms)
            ttft = m.ema_ttft_ms if m.ema_ttft_ms is not None else float(cfg.default_ttft_ms)
            e2el = m.ema_e2el_ms if m.ema_e2el_ms is not None else float(cfg.default_e2el_ms)
            err_rate = (m.total_errors / max(m.total_requests, 1)) * float(cfg.err_rate_penalty_ms)
            recent_err_penalty = 0.0
            if (
                m.last_error_ts is not None
                and (time.time() - m.last_error_ts) < float(cfg.recent_error_window_sec)
            ):
                recent_err_penalty = float(cfg.recent_error_penalty_ms)
            return inflight_penalty + ttft + 0.5 * e2el + err_rate + recent_err_penalty

        # Optional exploration to keep metrics fresh and avoid pathological lock-in.
        if self._config.explore_ratio > 0.0 and random.random() < self._config.explore_ratio:
            return random.choice(endpoints)

        # Pick randomly from top-k best endpoints by score to avoid thundering herd.
        ranked = sorted(endpoints, key=score)
        k = min(max(int(self._config.top_k), 1), len(ranked))
        if k == 1:
            return ranked[0]
        return random.choice(ranked[:k])

    async def mark_start(self, endpoint_id: str) -> None:
        async with self._lock:
            ep = self._endpoints.get(endpoint_id)
            if ep is None:
                return
            ep.metrics.inflight += 1
            ep.metrics.total_requests += 1

    async def mark_error(self, endpoint_id: str) -> None:
        async with self._lock:
            ep = self._endpoints.get(endpoint_id)
            if ep is None:
                return
            ep.metrics.total_errors += 1
            ep.metrics.last_error_ts = time.time()

    async def mark_finish(
        self,
        endpoint_id: str,
        *,
        ttft_ms: Optional[float],
        tpot_ms: Optional[float],
        itl_ms: Optional[float],
        e2el_ms: Optional[float],
    ) -> None:
        async with self._lock:
            ep = self._endpoints.get(endpoint_id)
            if ep is None:
                return
            ep.metrics.inflight = max(ep.metrics.inflight - 1, 0)

            if ttft_ms is not None:
                ep.metrics.last_ttft_ms = ttft_ms
                ep.metrics.ema_ttft_ms = self._ema(ep.metrics.ema_ttft_ms, ttft_ms)
            if tpot_ms is not None:
                ep.metrics.last_tpot_ms = tpot_ms
                ep.metrics.ema_tpot_ms = self._ema(ep.metrics.ema_tpot_ms, tpot_ms)
            if itl_ms is not None:
                ep.metrics.last_itl_ms = itl_ms
                ep.metrics.ema_itl_ms = self._ema(ep.metrics.ema_itl_ms, itl_ms)
            if e2el_ms is not None:
                ep.metrics.last_e2el_ms = e2el_ms
                ep.metrics.ema_e2el_ms = self._ema(ep.metrics.ema_e2el_ms, e2el_ms)


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
    # This is sufficient for TTFT/ITL estimation without fully parsing JSON.
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
) -> Tuple[AsyncIterator[bytes], Callable[[], Dict[str, Optional[float]]]]:
    start_ts = time.time()
    first_token_ts: Optional[float] = None
    last_token_ts: Optional[float] = None
    prev_token_ts: Optional[float] = None
    itl_samples_ms: List[float] = []
    token_events = 0

    buffer = bytearray()

    async def gen() -> AsyncIterator[bytes]:
        nonlocal first_token_ts, last_token_ts, prev_token_ts, token_events
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
                            if prev_token_ts is not None:
                                itl_samples_ms.append((now - prev_token_ts) * 1000.0)
                            prev_token_ts = now
                            last_token_ts = now
                            token_events += 1
                yield chunk

    def finalize_metrics() -> Dict[str, Optional[float]]:
        end_ts = time.time()
        e2el_ms = (end_ts - start_ts) * 1000.0
        ttft_ms = None if first_token_ts is None else (first_token_ts - start_ts) * 1000.0
        itl_ms = None if not itl_samples_ms else sum(itl_samples_ms) / len(itl_samples_ms)
        if ttft_ms is None or token_events <= 0:
            tpot_ms = None
        else:
            # Approximate TPOT by "per content SSE event" rather than tokenizer tokens.
            gen_ms = max(e2el_ms - ttft_ms, 0.0)
            tpot_ms = gen_ms / max(token_events, 1)
        return {"ttft_ms": ttft_ms, "itl_ms": itl_ms, "tpot_ms": tpot_ms, "e2el_ms": e2el_ms}

    # Return a callable so metrics are computed after the stream finishes.
    return gen(), finalize_metrics


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})


@app.post("/register")
async def register(raw_request: Request) -> JSONResponse:
    payload = await raw_request.json()
    base_url = payload.get("endpoint") or payload.get("base_url")
    if not base_url:
        raise HTTPException(status_code=400, detail="Missing endpoint/base_url")
    try:
        ep = await registry.register(str(base_url))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return JSONResponse(content={"endpoint_id": ep.endpoint_id, "base_url": ep.base_url})


@app.post("/unregister")
async def unregister(raw_request: Request) -> JSONResponse:
    payload = await raw_request.json()
    endpoint_id = payload.get("endpoint_id")
    base_url = payload.get("endpoint") or payload.get("base_url")
    removed = await registry.unregister(endpoint_id=endpoint_id, base_url=base_url)
    return JSONResponse(content={"removed": removed})


@app.get("/endpoints")
async def endpoints() -> JSONResponse:
    return JSONResponse(content={"endpoints": await registry.list_endpoints()})


@app.post("/weight/refit")
async def weight_refit(raw_request: Request) -> JSONResponse:
    headers = _filter_forward_headers(dict(raw_request.headers))
    body = await raw_request.body()
    eps = await registry._snapshot_endpoints()
    logger.info(
        "Broadcasting /weight/refit to %d endpoints: %s",
        len(eps),
        [e.base_url for e in eps],
    )
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
    request_json = await raw_request.json()
    is_stream = bool(request_json.get("stream", False))

    ep = await registry.choose_best()
    await registry.mark_start(ep.endpoint_id)
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
                    await registry.mark_error(ep.endpoint_id)
                    raise
                finally:
                    m = metrics_final()
                    await registry.mark_finish(
                        ep.endpoint_id,
                        ttft_ms=m.get("ttft_ms"),
                        tpot_ms=m.get("tpot_ms"),
                        itl_ms=m.get("itl_ms"),
                        e2el_ms=m.get("e2el_ms"),
                    )

            return StreamingResponse(
                wrapped(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
            )
        except HTTPException:
            await registry.mark_error(ep.endpoint_id)
            await registry.mark_finish(ep.endpoint_id, ttft_ms=None, tpot_ms=None, itl_ms=None, e2el_ms=None)
            raise
        except Exception as e:
            await registry.mark_error(ep.endpoint_id)
            await registry.mark_finish(ep.endpoint_id, ttft_ms=None, tpot_ms=None, itl_ms=None, e2el_ms=None)
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e

    start_ts = time.time()
    client = await registry._get_client()
    try:
        resp = await client.post(url, headers=headers, json=request_json)
        e2el_ms = (time.time() - start_ts) * 1000.0
        # For non-stream, treat TTFT as full latency.
        await registry.mark_finish(ep.endpoint_id, ttft_ms=e2el_ms, tpot_ms=None, itl_ms=None, e2el_ms=e2el_ms)
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception as e:
        await registry.mark_error(ep.endpoint_id)
        await registry.mark_finish(ep.endpoint_id, ttft_ms=None, tpot_ms=None, itl_ms=None, e2el_ms=None)
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")

