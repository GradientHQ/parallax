#!/usr/bin/env python3
import argparse
import math
import time

import mlx.core as mx
import numpy as np

from parallax.metal.paged_attention.kernel import paged_attention as old_paged_attention
from parallax_extensions.ops import paged_attention_v1 as new_paged_attention


def parse_dtype(name: str) -> mx.Dtype:
    name = name.lower()
    if name in ("float16", "fp16", "f16"):
        return mx.float16
    if name in ("float32", "fp32", "f32"):
        return mx.float32
    if name in ("bfloat16", "bf16"):
        return mx.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def packing_factor(dtype: mx.Dtype) -> int:
    if dtype == mx.float32:
        return 4
    if dtype == mx.float16 or dtype == mx.bfloat16:
        return 8
    return 4


def make_block_tables(batch_size: int, num_blocks_per_seq: int) -> mx.array:
    total_blocks = batch_size * num_blocks_per_seq
    tables = np.arange(total_blocks, dtype=np.int32).reshape(batch_size, num_blocks_per_seq)
    return mx.array(tables, dtype=mx.int32)


def bench_old(
    q_old: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    block_size: int,
    scale: float,
    num_kv_heads: int,
    iters: int,
    warmup: int,
    window_size: int | None,
    sinks: mx.array | None,
) -> float:
    kwargs = {}
    if window_size is not None and sinks is not None:
        kwargs["window_size"] = window_size
        kwargs["sinks"] = sinks
    for _ in range(warmup):
        out = old_paged_attention(
            q_old,
            k_cache,
            v_cache,
            block_tables,
            context_lengths,
            block_size,
            scale,
            num_kv_heads,
            **kwargs,
        )
        mx.eval(out)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = old_paged_attention(
            q_old,
            k_cache,
            v_cache,
            block_tables,
            context_lengths,
            block_size,
            scale,
            num_kv_heads,
            **kwargs,
        )
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def bench_new(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    block_tables: mx.array,
    context_lengths: mx.array,
    block_size: int,
    scale: float,
    num_kv_heads: int,
    iters: int,
    warmup: int,
    window_size: int,
    sinks: mx.array | None,
) -> float:
    for _ in range(warmup):
        out = new_paged_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lengths,
            block_size,
            scale,
            num_kv_heads,
            window_size=window_size,
            sinks=sinks,
        )
        mx.eval(out)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = new_paged_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lengths,
            block_size,
            scale,
            num_kv_heads,
            window_size=window_size,
            sinks=sinks,
        )
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark old vs new paged attention kernels.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=0)
    parser.add_argument("--use-sink", action="store_true")
    parser.add_argument("--sink-min", type=float, default=-0.5)
    parser.add_argument("--sink-max", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    dtype = parse_dtype(args.dtype)
    scale = 1.0 / math.sqrt(args.head_dim)
    num_blocks_per_seq = (args.seq_len + args.block_size - 1) // args.block_size

    block_tables = make_block_tables(args.batch_size, num_blocks_per_seq)
    context_lengths = mx.full((args.batch_size,), args.seq_len, dtype=mx.int32)

    q = mx.random.normal((args.batch_size, args.num_heads, args.head_dim)).astype(dtype)
    q_old = q[:, :, None, :]

    x = packing_factor(dtype)
    total_blocks = args.batch_size * num_blocks_per_seq
    old_k_cache = mx.random.normal(
        (1, total_blocks, args.num_kv_heads, args.block_size, args.head_dim)
    ).astype(dtype)
    old_v_cache = mx.random.normal(
        (1, total_blocks, args.num_kv_heads, args.block_size, args.head_dim)
    ).astype(dtype)
    new_k_cache = mx.random.normal(
        (total_blocks, args.num_kv_heads, args.head_dim // x, args.block_size, x)
    ).astype(dtype)
    new_v_cache = mx.random.normal(
        (total_blocks, args.num_kv_heads, args.head_dim, args.block_size)
    ).astype(dtype)
    mx.eval(q, q_old, old_k_cache, old_v_cache, new_k_cache, new_v_cache)

    sinks = None
    if args.use_sink:
        sinks = mx.array(
            np.random.uniform(args.sink_min, args.sink_max, size=(args.num_heads,)),
            dtype=mx.float32,
        )
    elif args.window_size > 0:
        sinks = mx.full((args.num_heads,), -float("inf"), dtype=mx.float32)

    print(
        "Benchmark config:",
        f"BS={args.batch_size}",
        f"heads={args.num_heads}",
        f"kv_heads={args.num_kv_heads}",
        f"D={args.head_dim}",
        f"seq_len={args.seq_len}",
        f"block_size={args.block_size}",
        f"dtype={args.dtype}",
        f"window={args.window_size}",
        f"sink={'on' if args.use_sink else 'off'}",
    )

    if args.window_size == 0 and not args.use_sink:
        old_ms = bench_old(
            q_old,
            old_k_cache,
            old_v_cache,
            block_tables,
            context_lengths,
            args.block_size,
            scale,
            args.num_kv_heads,
            args.iters,
            args.warmup,
            window_size=None,
            sinks=None,
        )
        print(f"Old kernel (baseline): {old_ms:.3f} ms")
    else:
        old_ms = bench_old(
            q_old,
            old_k_cache,
            old_v_cache,
            block_tables,
            context_lengths,
            args.block_size,
            scale,
            args.num_kv_heads,
            args.iters,
            args.warmup,
            window_size=args.window_size,
            sinks=sinks,
        )
        print(f"Old kernel (gpt_oss): {old_ms:.3f} ms")

    new_ms = bench_new(
        q,
        new_k_cache,
        new_v_cache,
        block_tables,
        context_lengths,
        args.block_size,
        scale,
        args.num_kv_heads,
        args.iters,
        args.warmup,
        args.window_size,
        sinks,
    )
    print(f"New kernel: {new_ms:.3f} ms")

    if old_ms is not None:
        print(f"Speedup: {old_ms / new_ms:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
