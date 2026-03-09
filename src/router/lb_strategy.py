"""
Load balancing strategies for the router.

All code/comments/logs are in English by project convention.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List, Literal, Protocol, TypeVar

T = TypeVar("T")

StrategyName = Literal["round_robin", "performance", "random"]


class Strategy(Protocol[T]):
    name: StrategyName

    def select(self, candidates: List[T]) -> T: ...


@dataclass
class PerformanceConfig:
    # EMA smoothing factor in (0, 1]. Higher means "more real-time".
    ema_alpha: float = 0.1

    # Defaults used for cold-start endpoints without any metrics yet.
    default_ttft_ms: float = 3000.0
    default_tpot_ms: float = 50.0

    # Scoring weights (all in milliseconds).
    inflight_penalty_ms: float = 1000.0
    err_rate_penalty_ms: float = 5000.0
    recent_error_window_sec: float = 30.0
    recent_error_penalty_ms: float = 2000.0
    tpot_weight: float = 1.0

    # Selection: pick randomly from top-k (by score). k=1 == strict best.
    top_k: int = 1

    # Exploration ratio in [0, 1). With probability p, pick a random endpoint.
    explore_ratio: float = 0.0


class PerformanceStrategy:
    name: StrategyName = "performance"

    def __init__(self, cfg: PerformanceConfig) -> None:
        self._cfg = cfg

    def score(self, candidate: T) -> float:
        """
        Compute a performance score for a candidate.

        Lower is better. This assumes the candidate has a `metrics` attribute with:
          - inflight: int
          - ema_ttft_ms: float|None
          - ema_tpot_ms: float|None
          - total_errors: int
          - total_requests: int
          - last_error_ts: float|None
        """
        m = getattr(candidate, "metrics")
        inflight_penalty = float(getattr(m, "inflight")) * float(self._cfg.inflight_penalty_ms)
        ttft = (
            float(getattr(m, "ema_ttft_ms"))
            if getattr(m, "ema_ttft_ms") is not None
            else float(self._cfg.default_ttft_ms)
        )
        tpot = (
            float(getattr(m, "ema_tpot_ms"))
            if getattr(m, "ema_tpot_ms") is not None
            else float(self._cfg.default_tpot_ms)
        )

        total_errors = float(getattr(m, "total_errors"))
        total_requests = float(getattr(m, "total_requests"))
        err_rate = (total_errors / max(total_requests, 1.0)) * float(self._cfg.err_rate_penalty_ms)

        recent_err_penalty = 0.0
        last_error_ts = getattr(m, "last_error_ts")
        if last_error_ts is not None:
            if (time.time() - float(last_error_ts)) < float(self._cfg.recent_error_window_sec):
                recent_err_penalty = float(self._cfg.recent_error_penalty_ms)

        return (
            inflight_penalty
            + ttft
            + float(self._cfg.tpot_weight) * tpot
            + err_rate
            + recent_err_penalty
        )

    def select(self, candidates: List[T]) -> T:
        if not candidates:
            raise ValueError("No candidates")

        if self._cfg.explore_ratio > 0.0 and random.random() < self._cfg.explore_ratio:
            return random.choice(candidates)

        ranked = sorted(candidates, key=self.score)
        k = min(max(int(self._cfg.top_k), 1), len(ranked))
        if k == 1:
            return ranked[0]
        return random.choice(ranked[:k])


class RandomStrategy:
    name: StrategyName = "random"

    def select(self, candidates: List[T]) -> T:
        if not candidates:
            raise ValueError("No candidates")
        return random.choice(candidates)


class RoundRobinStrategy:
    name: StrategyName = "round_robin"

    def __init__(self) -> None:
        self._cursor: int = 0

    def select(self, candidates: List[T]) -> T:
        if not candidates:
            raise ValueError("No candidates")

        # Use stable ordering for predictable rotation.
        ordered = list(candidates)
        # Stable ordering: prefer `base_url` if present, otherwise fallback to str().
        ordered.sort(key=lambda x: getattr(x, "base_url", str(x)))

        # "Round robin" here means: always pick the endpoint with the smallest
        # normalized load (inflight / max_running_request). Even if inflight exceeds
        # max_running_request, we still allow selection as requested.
        #
        # Tie-breaker uses cursor-based rotation for fairness.
        n = len(ordered)
        start = self._cursor % n

        best_idx: int = start
        best_score: float = float("inf")

        for step in range(n):
            idx = (start + step) % n
            cand = ordered[idx]
            m = getattr(cand, "metrics", None)
            inflight = int(getattr(m, "inflight", 0)) if m is not None else 0
            maxr = getattr(m, "max_running_request", None) if m is not None else None
            denom = 1
            if isinstance(maxr, int) and maxr > 0:
                denom = maxr
            score = float(inflight) / float(denom)
            if score < best_score:
                best_score = score
                best_idx = idx

        self._cursor = best_idx + 1
        return ordered[best_idx]


def make_strategy(name: StrategyName, *, performance_cfg: PerformanceConfig) -> Strategy:
    if name == "performance":
        return PerformanceStrategy(performance_cfg)
    if name == "random":
        return RandomStrategy()
    if name == "round_robin":
        return RoundRobinStrategy()
    raise ValueError(f"Unknown strategy: {name}")
