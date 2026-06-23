#!/usr/bin/env python3
"""Reproduce MLX idle materialization latency without Parallax.

This script keeps a configurable amount of MLX/Metal memory resident, waits for
an idle gap, then times a scalar eval and hidden-state-shaped zero tensors.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx


GIB = 1024**3


def parse_dtype(name: str) -> mx.Dtype:
    try:
        return {
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
            "float32": mx.float32,
        }[name]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {name}") from exc


def dtype_nbytes(dtype: mx.Dtype) -> int:
    if dtype in (mx.float16, mx.bfloat16):
        return 2
    if dtype == mx.float32:
        return 4
    raise ValueError(f"unsupported dtype: {dtype}")


def memory_gib() -> str:
    active = mx.get_active_memory() / GIB
    cache = mx.get_cache_memory() / GIB
    peak = mx.get_peak_memory() / GIB
    return f"active={active:.3f}GiB cache={cache:.3f}GiB peak={peak:.3f}GiB"


def chunk_shape(elements: int) -> tuple[int, ...]:
    # Large 1D dimensions can be rejected by MLX's Python binding. Split large
    # resident buffers into a simple 2D shape while keeping the same element count
    # for the default 4GiB bf16 chunks.
    cols = 65536
    if elements > cols and elements % cols == 0:
        return (elements // cols, cols)
    return (elements,)


def timed_eval(name: str, fn) -> float:
    # mx.synchronize()
    start = time.perf_counter()
    out = fn()
    mx.eval(out)
    # mx.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"{name}: {elapsed_ms:.3f}ms, new_size: {out.nbytes / 2**20}MiB, {memory_gib()}", flush=True)
    return elapsed_ms


def reserve_memory(reserve_gib: float, chunk_gib: float, dtype: mx.Dtype) -> list[mx.array]:
    resident = []
    if reserve_gib <= 0:
        return resident

    total_bytes = int(reserve_gib * GIB)
    chunk_bytes = int(chunk_gib * GIB)
    if chunk_bytes <= 0:
        raise ValueError("--chunk-gib must be positive")

    remaining = total_bytes
    chunk_idx = 0
    item_nbytes = dtype_nbytes(dtype)
    while remaining > 0:
        this_chunk_bytes = min(chunk_bytes, remaining)
        elements = max(1, this_chunk_bytes // item_nbytes)
        arr = mx.zeros(chunk_shape(elements), dtype=dtype)
        mx.eval(arr)
        resident.append(arr)
        remaining -= elements * item_nbytes
        chunk_idx += 1
        print(
            f"reserved chunk={chunk_idx} bytes={elements * item_nbytes / GIB:.3f}GiB "
            f"shape={arr.shape} {memory_gib()}",
            flush=True,
        )

    mx.synchronize()
    print(f"reserved total={reserve_gib:.3f}GiB {memory_gib()}", flush=True)
    return resident


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal MLX-only repro for idle hidden-state materialization latency."
    )
    parser.add_argument("--reserve-gib", type=float, default=32.0)
    parser.add_argument("--chunk-gib", type=float, default=1.0)
    parser.add_argument("--idle-s", type=float, default=3.0)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--dtype", type=parse_dtype, default=mx.bfloat16)
    parser.add_argument("--skip-wired-limit", action="store_true")
    args = parser.parse_args()

    info = mx.device_info()
    print(f"device_info={info}", flush=True)
    if not args.skip_wired_limit:
        try:
            mx.set_wired_limit(info["max_recommended_working_set_size"] - 2**30)
            print(f"wired_limit={info['max_recommended_working_set_size'] / GIB:.3f}GiB", flush=True)
        except Exception as exc:
            print(f"set_wired_limit failed: {exc}", flush=True)

    resident = reserve_memory(args.reserve_gib, args.chunk_gib, args.dtype)
    print(f"resident_buffers={len(resident)}", flush=True)

    print("warmup", flush=True)
    timed_eval(
        "warm_zero",
        lambda: mx.zeros((args.hidden_size), dtype=args.dtype),
    )
    timed_eval(
        "zero_immediate",
        lambda: mx.zeros((args.hidden_size), dtype=args.dtype),
    )

    for _ in range(3):
        print(f"idle sleep {args.idle_s:.3f}s", flush=True)
        time.sleep(args.idle_s)

        print("after_idle", flush=True)
        timed_eval(
            "zero_after_idle",
            lambda: mx.zeros((args.hidden_size), dtype=args.dtype),
        )
        timed_eval(
            "zero_immediate",
            lambda: mx.zeros((args.hidden_size), dtype=args.dtype),
        )


if __name__ == "__main__":
    main()
