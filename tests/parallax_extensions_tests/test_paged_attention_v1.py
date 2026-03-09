import math
import time

import mlx.core as mx
import numpy as np
import pytest

from parallax.metal.paged_attention.kernel import paged_attention as old_paged_attention
from parallax.metal.paged_attention.kernel import (
    reshape_and_cache as old_reshape_and_cache,
)
from parallax_extensions.ops import paged_attention_v1, reshape_and_cache


def get_packing_factor(dtype):
    if dtype == mx.float32:
        return 4
    elif dtype == mx.float16 or dtype == mx.bfloat16:
        return 8
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def ref_masked_attention(q, k, v, seq_lens, scale):
    """Reference implementation (Logical Layout)."""
    # q: [B, H, 1, D]
    q_expanded = q if q.ndim == 4 else q[:, :, None, :]

    # Scores: [B, H, 1, L]
    scores = (q_expanded @ k.transpose(0, 1, 3, 2)) * scale

    # Create mask: [B, 1, 1, L]
    seq_len = k.shape[2]
    indices = mx.arange(seq_len)[None, None, None, :]
    mask = indices < seq_lens[:, None, None, None]

    scores = mx.where(mask, scores, mx.array(-1e4, dtype=scores.dtype))
    probs = mx.softmax(scores, axis=-1)
    probs = mx.where(mask, probs, mx.array(0.0, dtype=probs.dtype))

    # [B, H, 1, L] @ [B, H, L, D] -> [B, H, 1, D]
    output = probs @ v
    return output  # Keep (B, H, 1, D) for consistency with new API


def ref_attention_gqa(q, k, v, seq_lens, scale):
    """Reference for GQA."""
    bs, n_heads, _, dim = q.shape
    n_kv_heads = k.shape[1]

    if n_kv_heads != n_heads:
        n_rep = n_heads // n_kv_heads
        k = mx.repeat(k[:, :, None, :, :], n_rep, axis=2).reshape(bs, n_heads, k.shape[2], dim)
        v = mx.repeat(v[:, :, None, :, :], n_rep, axis=2).reshape(bs, n_heads, v.shape[2], dim)

    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    seq_len = k.shape[2]
    indices = mx.arange(seq_len)[None, None, None, :]
    mask = indices < seq_lens[:, None, None, None]

    scores = mx.where(mask, scores, mx.array(-1e4, dtype=scores.dtype))
    probs = mx.softmax(scores, axis=-1)
    probs = mx.where(mask, probs, mx.array(0.0, dtype=probs.dtype))
    output = probs @ v
    return output


def reference_paged_attention(
    q,
    k,
    v,
    scale,
    num_kv_heads,
    context_lengths=None,
    window_size=0,
    sinks=None,
):
    """Reference paged attention implementation with window/sink support."""
    if q.ndim == 3:
        q = q[:, :, None, :]
    batch_size, num_heads, _, head_dim = q.shape
    _, _, seq_len, _ = k.shape

    n_rep = num_heads // num_kv_heads
    if n_rep > 1:
        k = mx.repeat(k[:, :, None, :, :], n_rep, axis=2).reshape(
            batch_size, num_heads, seq_len, head_dim
        )
        v = mx.repeat(v[:, :, None, :, :], n_rep, axis=2).reshape(
            batch_size, num_heads, seq_len, head_dim
        )

    if context_lengths is None:
        context_lengths = mx.full((batch_size,), seq_len, dtype=mx.int32)

    q_f32 = q.astype(mx.float32)
    k_f32 = k.astype(mx.float32)
    v_f32 = v.astype(mx.float32)

    scores = (q_f32 @ k_f32.transpose(0, 1, 3, 2)) * scale

    positions = mx.arange(seq_len)[None, None, None, :]
    ctx = context_lengths[:, None, None, None]
    scores = mx.where(positions >= ctx, -float("inf"), scores)

    if window_size > 0:
        window_start = mx.maximum(context_lengths - 1 - window_size, 0)
        win = window_start[:, None, None, None]
        scores = mx.where(positions < win, -float("inf"), scores)

    if sinks is not None:
        if sinks.ndim != 1 or sinks.shape[0] != num_heads:
            raise ValueError("sinks must be shape (num_heads,)")
        sink_scores = sinks.astype(mx.float32)[None, :, None]
        max_scores = mx.maximum(mx.max(scores, axis=-1, keepdims=True), sink_scores)
        exp_scores = mx.exp(scores - max_scores)
        exp_sum = mx.sum(exp_scores, axis=-1, keepdims=True) + mx.exp(sink_scores - max_scores)
        attn_weights = exp_scores / exp_sum
    else:
        attn_weights = mx.softmax(scores, axis=-1)

    output = attn_weights @ v_f32
    if output.ndim == 3:
        output = output[:, :, None, :]

    return output


def _bench_old(
    q_old,
    k_cache,
    v_cache,
    block_tables,
    context_lengths,
    block_size,
    scale,
    num_kv_heads,
    iters,
    warmup,
    window_size=None,
    sinks=None,
):
    kwargs = {}
    if window_size is not None:
        kwargs["window_size"] = window_size
    if sinks is not None:
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
    start = time.perf_counter()
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
    return (time.perf_counter() - start) / iters * 1000.0


def _bench_new(
    q,
    k_cache,
    v_cache,
    block_tables,
    context_lengths,
    block_size,
    scale,
    num_kv_heads,
    iters,
    warmup,
    window_size=0,
    sinks=None,
):
    for _ in range(warmup):
        out = paged_attention_v1(
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
    start = time.perf_counter()
    for _ in range(iters):
        out = paged_attention_v1(
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
    return (time.perf_counter() - start) / iters * 1000.0


class TestPagedAttentionV1:

    @pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
    def test_basic_functionality(self, dtype):
        mx.random.seed(42)
        np.random.seed(42)

        # Constants
        BATCH_SIZE = 2
        NUM_HEADS = 4
        NUM_KV_HEADS = 4
        HEAD_DIM = 32
        BLOCK_SIZE = 16
        NUM_BLOCKS = 16
        SCALE = 1.0 / math.sqrt(HEAD_DIM)

        x = get_packing_factor(dtype)
        head_dim_x = HEAD_DIM // x
        atol = 1e-3 if dtype == mx.float32 else 1e-2

        # Setup Cache
        key_cache = mx.zeros((NUM_BLOCKS, NUM_KV_HEADS, head_dim_x, BLOCK_SIZE, x), dtype=dtype)
        value_cache = mx.zeros((NUM_BLOCKS, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE), dtype=dtype)

        # Block Tables
        block_tables_np = np.zeros((BATCH_SIZE, 2), dtype=np.int32)
        block_tables_np[0, :] = [0, 1]
        block_tables_np[1, 0] = 2
        block_tables_np[1, 1] = -1
        block_tables = mx.array(block_tables_np)

        # Seq Lens (Context lengths)
        seq_lens = mx.array([20, 5], dtype=mx.int32)

        # --- Write to Cache (Decode Mode - automatic slot mapping) ---
        k_new = mx.random.normal(shape=(BATCH_SIZE, NUM_KV_HEADS, HEAD_DIM)).astype(dtype)
        v_new = mx.random.normal(shape=(BATCH_SIZE, NUM_KV_HEADS, HEAD_DIM)).astype(dtype)

        reshape_and_cache(
            key=k_new,
            value=v_new,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lengths=seq_lens,
            block_size=BLOCK_SIZE,
        )

        # --- Verification 1: Cache Write ---
        # Check Seq 1 (Length 5 -> Block 2, Offset 4)
        k_phys_block_2 = key_cache[2]
        k_phys_unpacked = k_phys_block_2.transpose(0, 2, 1, 3).reshape(
            NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM
        )
        k_cached_val = k_phys_unpacked[:, 4, :]
        assert mx.allclose(k_cached_val, k_new[1], atol=atol).item()

        # --- Read from Cache ---
        q = mx.random.normal(shape=(BATCH_SIZE, NUM_HEADS, HEAD_DIM)).astype(dtype)

        attn_out = paged_attention_v1(
            queries=q,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lengths=seq_lens,
            block_size=BLOCK_SIZE,
            scale=SCALE,
            num_kv_heads=NUM_KV_HEADS,
        )
        mx.eval(attn_out)

        # --- Verification 2: Reference ---
        max_seq_len = block_tables.shape[1] * BLOCK_SIZE
        k_logical_hist = mx.zeros((BATCH_SIZE, NUM_KV_HEADS, max_seq_len, HEAD_DIM), dtype=dtype)
        v_logical_hist = mx.zeros((BATCH_SIZE, NUM_KV_HEADS, max_seq_len, HEAD_DIM), dtype=dtype)

        k_logical_hist[0, :, 19, :] = k_new[0]
        v_logical_hist[0, :, 19, :] = v_new[0]
        k_logical_hist[1, :, 4, :] = k_new[1]
        v_logical_hist[1, :, 4, :] = v_new[1]

        ref_out = ref_masked_attention(
            q.astype(mx.float32),
            k_logical_hist.astype(mx.float32),
            v_logical_hist.astype(mx.float32),
            seq_lens,
            SCALE,
        )

        diff = mx.abs(attn_out.astype(mx.float32) - ref_out).max().item()
        assert diff < atol, f"Max diff: {diff}"

    @pytest.mark.parametrize(
        "params",
        [
            {"bs": 4, "len": 128, "heads": 32, "kv_heads": 32, "dim": 128, "desc": "MHA Float16"},
            {"bs": 2, "len": 256, "heads": 32, "kv_heads": 8, "dim": 128, "desc": "GQA Float16"},
        ],
    )
    def test_large_scale_correctness(self, params):
        mx.random.seed(42)
        np.random.seed(42)
        bs = params["bs"]
        seq_len = params["len"]
        n_heads = params["heads"]
        n_kv_heads = params["kv_heads"]
        dim = params["dim"]
        block_size = 16
        dtype = mx.float16
        scale = 1.0 / (dim**0.5)

        x = get_packing_factor(dtype)
        num_blocks_per_seq = (seq_len + block_size - 1) // block_size
        total_blocks = num_blocks_per_seq * bs

        key_cache = mx.zeros((total_blocks, n_kv_heads, dim // x, block_size, x), dtype=dtype)
        value_cache = mx.zeros((total_blocks, n_kv_heads, dim, block_size), dtype=dtype)

        all_blocks = np.arange(total_blocks, dtype=np.int32).reshape(bs, num_blocks_per_seq)
        block_tables = mx.array(all_blocks)

        # Generate K/V in logical layout for reference: (B, H, T, D)
        k_hist_logical = mx.random.normal(shape=(bs, n_kv_heads, seq_len, dim)).astype(dtype)
        v_hist_logical = mx.random.normal(shape=(bs, n_kv_heads, seq_len, dim)).astype(dtype)

        # For reshape_and_cache in prefill mode, need (B, T, H, D)
        k_hist_prefill = k_hist_logical.transpose(0, 2, 1, 3)  # (B, H, T, D) -> (B, T, H, D)
        v_hist_prefill = v_hist_logical.transpose(0, 2, 1, 3)

        slots_np = []
        for b in range(bs):
            for t in range(seq_len):
                logical_block = t // block_size
                block_off = t % block_size
                phys_block = all_blocks[b, logical_block]
                slots_np.append(phys_block * block_size + block_off)
        slot_mapping = mx.array(np.array(slots_np, dtype=np.int64))

        context_lens = mx.full((bs,), seq_len, dtype=mx.int32)
        reshape_and_cache(
            key=k_hist_prefill,
            value=v_hist_prefill,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lengths=context_lens,
            block_size=block_size,
            slot_mapping=slot_mapping,
        )

        q = mx.random.normal(shape=(bs, n_heads, dim)).astype(dtype)

        attn_out = paged_attention_v1(
            queries=q,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lengths=context_lens,
            block_size=block_size,
            scale=scale,
            num_kv_heads=n_kv_heads,
        )
        mx.eval(attn_out)

        # Run Reference (use logical layout)
        q_ref = q[:, :, None, :]
        ref_out = ref_attention_gqa(
            q_ref.astype(mx.float32),
            k_hist_logical.astype(mx.float32),
            v_hist_logical.astype(mx.float32),
            context_lens,
            scale,
        )

        diff = mx.abs(attn_out.astype(mx.float32) - ref_out).max().item()
        assert diff < 5e-2, f"Large scale mismatch ({params['desc']}). Max diff: {diff}"

    def test_reference_cases(self):
        mx.random.seed(42)
        np.random.seed(42)

        batch_size = 1
        num_heads = 2
        num_kv_heads = 2
        head_dim = 64
        block_size = 16
        seq_len = 32
        dtype = mx.float16
        scale = 1.0 / math.sqrt(head_dim)

        q = mx.random.uniform(shape=(batch_size, num_heads, 1, head_dim)).astype(dtype)
        k_seq = mx.random.uniform(shape=(batch_size, num_kv_heads, seq_len, head_dim)).astype(dtype)
        v_seq = mx.random.uniform(shape=(batch_size, num_kv_heads, seq_len, head_dim)).astype(dtype)

        context_lengths = mx.array([seq_len], dtype=mx.int32)
        num_blocks = (seq_len + block_size - 1) // block_size
        block_tables = mx.arange(num_blocks, dtype=mx.int32)[None, :]

        slots = []
        block_tables_np = np.array(block_tables)
        for t in range(seq_len):
            block_idx = t // block_size
            block_offset = t % block_size
            physical_block = int(block_tables_np[0, block_idx])
            slots.append(physical_block * block_size + block_offset)
        slot_mapping = mx.array(np.array(slots, dtype=np.int64))

        k_prefill = k_seq.transpose(0, 2, 1, 3)
        v_prefill = v_seq.transpose(0, 2, 1, 3)

        old_key_cache = mx.zeros((1, num_blocks, num_kv_heads, block_size, head_dim), dtype=dtype)
        old_value_cache = mx.zeros((1, num_blocks, num_kv_heads, block_size, head_dim), dtype=dtype)

        old_reshape_and_cache(
            k_prefill,
            v_prefill,
            old_key_cache,
            old_value_cache,
            block_tables,
            context_lengths,
            block_size,
            slot_mapping=slot_mapping,
        )

        x = get_packing_factor(dtype)
        new_key_cache = mx.zeros(
            (num_blocks, num_kv_heads, head_dim // x, block_size, x), dtype=dtype
        )
        new_value_cache = mx.zeros((num_blocks, num_kv_heads, head_dim, block_size), dtype=dtype)

        reshape_and_cache(
            k_prefill,
            v_prefill,
            new_key_cache,
            new_value_cache,
            block_tables,
            context_lengths,
            block_size,
            slot_mapping=slot_mapping,
        )

        mx.eval(old_key_cache, old_value_cache, new_key_cache, new_value_cache)
        mx.synchronize()

        sinks = mx.array(np.random.uniform(-0.5, 0.5, size=(num_heads,)), dtype=mx.float32)
        window_sizes = [block_size // 2, block_size]
        cases = [{"name": "baseline", "window_size": 0, "sinks": None, "use_old": True}]
        for ws in window_sizes:
            cases.append({"name": f"window-{ws}", "window_size": ws, "sinks": None})
        cases.append({"name": "sink-only", "window_size": 0, "sinks": sinks})
        for ws in window_sizes:
            cases.append({"name": f"window-{ws}+sink", "window_size": ws, "sinks": sinks})

        for case in cases:
            window_size = case["window_size"]
            sinks_case = case["sinks"]
            use_old = case.get("use_old", False)

            ref_output = reference_paged_attention(
                q,
                k_seq,
                v_seq,
                scale,
                num_kv_heads,
                context_lengths=context_lengths,
                window_size=window_size,
                sinks=sinks_case,
            )
            mx.eval(ref_output)

            atol = 1e-2 if sinks_case is not None else 1e-3
            if use_old:
                old_output = old_paged_attention(
                    q,
                    old_key_cache,
                    old_value_cache,
                    block_tables,
                    context_lengths,
                    block_size,
                    scale,
                    num_kv_heads,
                )
                mx.eval(old_output)
                diff_old = mx.abs(ref_output.astype(mx.float32) - old_output.astype(mx.float32))
                max_diff_old = mx.max(diff_old).item()
                assert mx.allclose(
                    ref_output.astype(mx.float32), old_output.astype(mx.float32), atol=1e-3
                ).item(), f"Old kernel mismatch (max diff {max_diff_old:.6f})"

            new_output = paged_attention_v1(
                q,
                new_key_cache,
                new_value_cache,
                block_tables,
                context_lengths,
                block_size,
                scale,
                num_kv_heads,
                window_size=window_size,
                sinks=sinks_case,
            )
            mx.eval(new_output)
            diff_new = mx.abs(ref_output.astype(mx.float32) - new_output.astype(mx.float32))
            max_diff_new = mx.max(diff_new).item()
            assert mx.allclose(
                ref_output.astype(mx.float32), new_output.astype(mx.float32), atol=atol
            ).item(), f"New kernel mismatch (case={case['name']}, max diff {max_diff_new:.6f})"

    def test_benchmark_paged_vs_native(self):
        mx.random.seed(42)
        bs = 64
        n_heads = 32
        n_kv_heads = 32
        dim = 128
        seq_len = 1024
        block_size = 16
        dtype = mx.float16
        scale = 1.0 / (dim**0.5)

        x = get_packing_factor(dtype)
        num_blocks = (seq_len + block_size - 1) // block_size
        total_blocks = num_blocks * bs

        key_cache = mx.zeros((total_blocks, n_kv_heads, dim // x, block_size, x), dtype=dtype)
        value_cache = mx.zeros((total_blocks, n_kv_heads, dim, block_size), dtype=dtype)
        mx.eval(key_cache, value_cache)

        block_tables = mx.array(np.arange(total_blocks, dtype=np.int32).reshape(bs, num_blocks))
        seq_lens = mx.full((bs,), seq_len, dtype=mx.int32)

        q = mx.random.normal(shape=(bs, n_heads, dim)).astype(dtype)
        k_cont = mx.random.normal(shape=(bs, n_kv_heads, seq_len, dim)).astype(dtype)
        v_cont = mx.random.normal(shape=(bs, n_kv_heads, seq_len, dim)).astype(dtype)
        q_old = q[:, :, None, :]

        old_k_cache = mx.random.normal((1, total_blocks, n_kv_heads, block_size, dim)).astype(dtype)
        old_v_cache = mx.random.normal((1, total_blocks, n_kv_heads, block_size, dim)).astype(dtype)
        mx.eval(old_k_cache, old_v_cache)

        print(f"\n[Benchmark BS={bs}, Len={seq_len}, Float16]")

        if hasattr(mx.fast, "scaled_dot_product_attention"):
            q_native = q[:, :, None, :]
            for _ in range(10):
                _ = mx.fast.scaled_dot_product_attention(q_native, k_cont, v_cont, scale=scale)
            mx.eval(_)
            mx.synchronize()
            time.perf_counter()
            for _ in range(100):
                out = mx.fast.scaled_dot_product_attention(q_native, k_cont, v_cont, scale=scale)
                mx.eval(out)
            mx.synchronize()
            time.perf_counter()
            # print(f"Native SDPA:   {(end-start)*10:.3f} ms")

        sinks = mx.array(np.random.uniform(-0.5, 0.5, size=(n_heads,)), dtype=mx.float32)
        window_sizes = [64, 128, 256]
        cases = [
            {"name": f"window-{ws}+sink", "window_size": ws, "sinks": sinks} for ws in window_sizes
        ]

        for case in cases:
            ws = case["window_size"]
            sinks_case = case["sinks"]
            label = case["name"]

            old_ws = None if ws == 0 else ws
            old_ms = _bench_old(
                q_old,
                old_k_cache,
                old_v_cache,
                block_tables,
                seq_lens,
                block_size,
                scale,
                n_kv_heads,
                iters=100,
                warmup=10,
                window_size=old_ws,
                sinks=sinks_case,
            )
            new_ms = _bench_new(
                q,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                block_size,
                scale,
                n_kv_heads,
                iters=100,
                warmup=10,
                window_size=ws,
                sinks=sinks_case,
            )
            speedup = old_ms / new_ms if new_ms > 0 else float("inf")
            print(f"Old kernel ({label}): {old_ms:.3f} ms")
            print(f"Paged Attn V1 ({label}): {new_ms:.3f} ms")
            print(f"Speedup ({label}): {speedup:.2f}x")
