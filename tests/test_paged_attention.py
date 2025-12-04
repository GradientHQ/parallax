import math
import time

import mlx.core as mx
import numpy as np
import pytest

from parallax.metal.paged_attention.kernel import paged_attention, reshape_and_cache


def ref_masked_attention(q, k, v, scale):
    """
    Reference implementation of attention for verification.
    Used for basic tests.
    q: (batch, n_heads, head_dim)
    k: (batch, n_heads, seq_len, head_dim)
    v: (batch, n_heads, seq_len, head_dim)
    """
    # (batch, n_heads, 1, head_dim) @ (batch, n_heads, head_dim, seq_len) -> (batch, n_heads, 1, seq_len)
    scores = (q[:, :, None, :] @ k.transpose(0, 1, 3, 2)) * scale
    probs = mx.softmax(scores, axis=-1)
    # (batch, n_heads, 1, seq_len) @ (batch, n_heads, seq_len, head_dim) -> (batch, n_heads, 1, head_dim)
    output = probs @ v
    return output.squeeze(2)  # (batch, n_heads, head_dim)


def ref_attention_large(q, k, v, scale):
    """
    Reference implementation handling GQA/MQA.
    Used for large scale tests.
    q: (batch, n_heads, 1, head_dim)
    k: (batch, n_kv_heads, seq_len, head_dim)
    v: (batch, n_kv_heads, seq_len, head_dim)
    """
    n_heads = q.shape[1]
    n_kv_heads = k.shape[1]
    n_rep = n_heads // n_kv_heads

    if n_rep > 1:
        k = mx.repeat(k[:, :, None, :, :], n_rep, axis=2).reshape(
            k.shape[0], n_heads, k.shape[2], k.shape[3]
        )
        v = mx.repeat(v[:, :, None, :, :], n_rep, axis=2).reshape(
            v.shape[0], n_heads, v.shape[2], v.shape[3]
        )

    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    probs = mx.softmax(scores, axis=-1)
    output = probs @ v
    return output.squeeze(2)  # (B, H, D)


class TestPagedAttention:

    @pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
    def test_basic_functionality(self, dtype):
        """Test reshape_and_cache and paged_attention with different dtypes on small data."""
        # Check for bfloat16 support
        if dtype == mx.bfloat16:
            try:
                mx.array(1.0, dtype=mx.bfloat16)
            except ValueError:
                pytest.skip("bfloat16 not supported")

        # Constants
        BATCH_SIZE = 2
        NUM_HEADS = 4
        NUM_KV_HEADS = 4
        HEAD_DIM = 32
        BLOCK_SIZE = 16
        NUM_LAYERS = 1
        NUM_BLOCKS = 1024
        LAYER_IDX = 0
        SCALE = 1.0 / math.sqrt(HEAD_DIM)
        atol = 1e-2 if dtype != mx.float32 else 1e-4

        # Setup Memory
        key_cache = mx.zeros(
            (NUM_LAYERS, NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM), dtype=dtype
        )
        value_cache = mx.zeros(
            (NUM_LAYERS, NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM), dtype=dtype
        )

        # Mock Block Tables
        max_blocks_per_req = 2
        block_tables_np = np.zeros((BATCH_SIZE, max_blocks_per_req), dtype=np.int32)
        block_tables_np[0, :] = [0, 1]
        block_tables_np[1, 0] = 2
        block_tables = mx.array(block_tables_np)

        # Context Lengths
        context_lengths = mx.array([20, 5], dtype=mx.int32)

        # --- Step 1: Test reshape_and_cache ---
        k_new = mx.random.uniform(shape=(BATCH_SIZE, NUM_KV_HEADS, 1, HEAD_DIM)).astype(dtype)
        v_new = mx.random.uniform(shape=(BATCH_SIZE, NUM_KV_HEADS, 1, HEAD_DIM)).astype(dtype)

        new_k_cache, new_v_cache = reshape_and_cache(
            k_new,
            v_new,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            BLOCK_SIZE,
            LAYER_IDX,
        )
        mx.eval(new_k_cache, new_v_cache)

        # Verify Data in Cache
        # Req 0 (len 20) -> Block 1, Offset 3
        cached_k_0 = new_k_cache[0, 1, :, 3, :]
        input_k_0 = k_new[0].squeeze(1)
        assert mx.allclose(cached_k_0, input_k_0, atol=atol).item(), "Cache update failed for Req 0"

        # Req 1 (len 5) -> Block 2, Offset 4
        cached_k_1 = new_k_cache[0, 2, :, 4, :]
        input_k_1 = k_new[1].squeeze(1)
        assert mx.allclose(
            cached_k_1, input_k_1, atol=atol
        ).item(), "Cache update failed for Req 1 (Key)"

        cached_v_1 = new_v_cache[0, 2, :, 4, :]
        input_v_1 = v_new[1].squeeze(1)
        assert mx.allclose(
            cached_v_1, input_v_1, atol=atol
        ).item(), "Cache update failed for Req 1 (Value)"

        # --- Step 2: Test paged_attention ---
        q = mx.random.uniform(shape=(BATCH_SIZE, NUM_HEADS, 1, HEAD_DIM)).astype(dtype)

        output = paged_attention(
            q,
            new_k_cache,
            new_v_cache,
            block_tables,
            context_lengths,
            BLOCK_SIZE,
            SCALE,
            NUM_KV_HEADS,
            LAYER_IDX,
        )
        mx.eval(output)

        # Verify against Reference
        # Construct inputs for reference
        k_part1 = mx.zeros((NUM_KV_HEADS, 4, HEAD_DIM), dtype=dtype)
        k_part2 = k_new[1]
        k_full_1 = mx.concatenate([k_part1, k_part2], axis=1)

        v_part1 = mx.zeros((NUM_KV_HEADS, 4, HEAD_DIM), dtype=dtype)
        v_part2 = v_new[1]
        v_full_1 = mx.concatenate([v_part1, v_part2], axis=1)

        k_ref_input = k_full_1[None, :, :, :]
        v_ref_input = v_full_1[None, :, :, :]
        q_input = q[1:2].squeeze(2)

        ref_out = ref_masked_attention(
            q_input.astype(mx.float32),
            k_ref_input.astype(mx.float32),
            v_ref_input.astype(mx.float32),
            SCALE,
        )
        kernel_out_1 = output[1].squeeze(1).astype(mx.float32)

        assert mx.allclose(
            kernel_out_1, ref_out[0], atol=atol
        ).item(), f"Paged Attention mismatch for Req 1 with {dtype}"

    @pytest.mark.parametrize(
        "params",
        [
            {
                "bs": 8,
                "len": 2048,
                "heads": 32,
                "kv_heads": 32,
                "dim": 128,
                "desc": "MHA Llama-2-7B Style",
            },
            {
                "bs": 4,
                "len": 2048,
                "heads": 32,
                "kv_heads": 8,
                "dim": 128,
                "desc": "GQA Custom Style",
            },
        ],
    )
    def test_large_scale_correctness(self, params):
        """
        Test paged_attention correctness on larger scales with MHA/GQA.
        Uses float16 for reasonable memory usage/precision check.
        """
        batch_size = params["bs"]
        seq_len = params["len"]
        num_heads = params["heads"]
        num_kv_heads = params["kv_heads"]
        head_dim = params["dim"]
        block_size = 16
        dtype = mx.float16

        scale = 1.0 / (head_dim**0.5)
        num_blocks_per_req = (seq_len + block_size - 1) // block_size
        total_blocks = num_blocks_per_req * batch_size

        # Setup Cache
        key_cache = mx.zeros((1, total_blocks, num_kv_heads, block_size, head_dim), dtype=dtype)
        value_cache = mx.zeros((1, total_blocks, num_kv_heads, block_size, head_dim), dtype=dtype)

        all_blocks = np.arange(total_blocks, dtype=np.int32).reshape(batch_size, num_blocks_per_req)
        block_tables = mx.array(all_blocks)
        context_lengths = mx.array([seq_len] * batch_size, dtype=mx.int32)

        # Generate Data
        k_history = mx.random.uniform(shape=(batch_size, num_kv_heads, seq_len, head_dim)).astype(
            dtype
        )
        v_history = mx.random.uniform(shape=(batch_size, num_kv_heads, seq_len, head_dim)).astype(
            dtype
        )
        q = mx.random.uniform(shape=(batch_size, num_heads, 1, head_dim)).astype(dtype)

        # Populate Cache (Simulated via reshape/assign)
        padded_len = num_blocks_per_req * block_size
        if padded_len > seq_len:
            padding = mx.zeros(
                (batch_size, num_kv_heads, padded_len - seq_len, head_dim), dtype=dtype
            )
            k_padded = mx.concatenate([k_history, padding], axis=2)
            v_padded = mx.concatenate([v_history, padding], axis=2)
        else:
            k_padded = k_history
            v_padded = v_history

        k_reshaped = k_padded.reshape(
            batch_size, num_kv_heads, num_blocks_per_req, block_size, head_dim
        )
        v_reshaped = v_padded.reshape(
            batch_size, num_kv_heads, num_blocks_per_req, block_size, head_dim
        )

        k_ready = k_reshaped.transpose(0, 2, 1, 3, 4).reshape(
            total_blocks, num_kv_heads, block_size, head_dim
        )
        v_ready = v_reshaped.transpose(0, 2, 1, 3, 4).reshape(
            total_blocks, num_kv_heads, block_size, head_dim
        )

        key_cache = k_ready[None, ...]
        value_cache = v_ready[None, ...]
        mx.eval(key_cache, value_cache)

        # Run Kernel
        out = paged_attention(
            q,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            block_size,
            scale,
            num_kv_heads,
            0,
        )
        mx.eval(out)

        # Run Reference
        ref_out = ref_attention_large(
            q.astype(mx.float32), k_history.astype(mx.float32), v_history.astype(mx.float32), scale
        )
        kernel_out = out.astype(mx.float32).squeeze(2)

        # Check Correctness
        # Relax tolerance slightly for large scale FP16 accumulations
        diff = mx.abs(ref_out - kernel_out)
        max_diff = mx.max(diff).item()

        assert (
            max_diff < 1e-2
        ), f"Large scale test failed for {params['desc']}, max_diff: {max_diff}"

    def test_benchmark_paged_vs_native(self):
        """
        Benchmark PagedAttention vs Native MLX SDPA.
        """
        # Config
        batch_size = 8
        num_heads = 32
        num_kv_heads = 32
        head_dim = 128
        seq_len = 200  # Length
        block_size = 1024
        dtype = mx.float16
        scale = 1.0 / math.sqrt(head_dim)

        # Data
        q = mx.random.uniform(shape=(batch_size, num_heads, 1, head_dim)).astype(dtype)
        k_cont = mx.random.uniform(shape=(batch_size, num_kv_heads, seq_len, head_dim)).astype(
            dtype
        )
        v_cont = mx.random.uniform(shape=(batch_size, num_kv_heads, seq_len, head_dim)).astype(
            dtype
        )

        # Setup Paged
        num_blocks = (seq_len + block_size - 1) // block_size
        total_blocks = num_blocks * batch_size
        # Quick setup without manager for pure kernel benchmark
        key_cache = mx.zeros((1, total_blocks, num_kv_heads, block_size, head_dim), dtype=dtype)
        value_cache = mx.zeros((1, total_blocks, num_kv_heads, block_size, head_dim), dtype=dtype)
        all_blocks = np.arange(total_blocks, dtype=np.int32).reshape(batch_size, num_blocks)
        block_tables = mx.array(all_blocks)
        context_lengths = mx.array([seq_len] * batch_size, dtype=mx.int32)
        # Force materialize paged cache
        mx.eval(key_cache, value_cache)

        # Benchmark Native (using mx.fast.scaled_dot_product_attention if available or manual)
        # For fair comparison, we assume native implementation is fast.
        # Manual implementation is slow in python.
        # Let's check if mx.fast.scaled_dot_product_attention exists
        has_sdpa = hasattr(mx.fast, "scaled_dot_product_attention")

        print(f"\n[Benchmark BS={batch_size}, Len={seq_len}]")

        if has_sdpa:
            # K needs to be (B, H, L, D) for SDPA?
            # SDPA docs: queries, keys, values.
            # Usually expects (B, H, L, D) or (B, L, H, D) depending on layout.
            # MLX SDPA: (q, k, v, ...).
            # Let's assume (B, H, L, D) for simplicity or transpose.
            # Native SDPA is highly optimized.
            start = time.perf_counter()
            for _ in range(100):
                # Transpose to match SDPA expectations if needed
                _ = mx.fast.scaled_dot_product_attention(q, k_cont, v_cont, scale=scale)
                mx.eval(_)
            end = time.perf_counter()
            native_time = (end - start) / 100 * 1000
            print(f"Native SDPA:     {native_time:.3f} ms")
        else:
            print("Native SDPA:     N/A (Not available in this MLX version)")

        # Benchmark Paged
        # Warmup
        for _ in range(5):
            _ = paged_attention(
                q,
                key_cache,
                value_cache,
                block_tables,
                context_lengths,
                block_size,
                scale,
                num_kv_heads,
                0,
            )
            mx.eval(_)

        start = time.perf_counter()
        for _ in range(100):
            out = paged_attention(
                q,
                key_cache,
                value_cache,
                block_tables,
                context_lengths,
                block_size,
                scale,
                num_kv_heads,
                0,
            )
            mx.eval(out)
        end = time.perf_counter()
        paged_time = (end - start) / 100 * 1000
        print(f"Paged Attention: {paged_time:.3f} ms")

    @pytest.mark.parametrize(
        "params",
        [
            {"bs": 1, "len": 128, "heads": 16, "kv_heads": 2, "dim": 128, "desc": "Small GQA"},
            {"bs": 8, "len": 512, "heads": 32, "kv_heads": 8, "dim": 128, "desc": "Medium GQA"},
            {"bs": 4, "len": 2048, "heads": 32, "kv_heads": 8, "dim": 128, "desc": "Large GQA"},
            {"bs": 2, "len": 4096, "heads": 16, "kv_heads": 2, "dim": 128, "desc": "Very Long GQA"},
        ],
    )
    def test_benchmark_sdpa_vs_metal(self, params):
        """
        Benchmark SDPA-based paged_attention vs Metal-based metal_paged_attention.
        Compares correctness and performance.
        """
        from parallax.metal.paged_attention.kernel import metal_paged_attention

        batch_size = params["bs"]
        seq_len = params["len"]
        num_heads = params["heads"]
        num_kv_heads = params["kv_heads"]
        head_dim = params["dim"]
        block_size = 32
        dtype = mx.bfloat16
        scale = 1.0 / math.sqrt(head_dim)
        num_iters = 50

        # Setup cache
        num_blocks_per_req = (seq_len + block_size - 1) // block_size
        total_blocks = 1024  # Fixed large value for benchmarking

        key_cache = mx.random.normal((1, total_blocks, num_kv_heads, block_size, head_dim)).astype(
            dtype
        )
        value_cache = mx.random.normal(
            (1, total_blocks, num_kv_heads, block_size, head_dim)
        ).astype(dtype)

        # Use scattered block indices from the large pool to simulate real usage
        all_blocks = np.random.choice(total_blocks, size=(batch_size, num_blocks_per_req), replace=False).astype(np.int32)
        block_tables = mx.array(all_blocks)
        context_lengths = mx.array([seq_len] * batch_size, dtype=mx.int32)

        q = mx.random.normal((batch_size, num_heads, 1, head_dim)).astype(dtype)

        mx.eval(key_cache, value_cache, q, block_tables, context_lengths)

        # Verify correctness first
        out_sdpa = paged_attention(
            q,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            block_size,
            scale,
            num_kv_heads,
            0,
        )
        out_metal = metal_paged_attention(
            q,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            block_size,
            scale,
            num_kv_heads,
            0,
        )
        mx.eval(out_sdpa, out_metal)

        diff = mx.abs(out_sdpa.astype(mx.float32) - out_metal.astype(mx.float32))
        max_diff = mx.max(diff).item()
        mean_diff = mx.mean(diff).item()

        # Warmup SDPA
        for _ in range(5):
            _ = paged_attention(
                q,
                key_cache,
                value_cache,
                block_tables,
                context_lengths,
                block_size,
                scale,
                num_kv_heads,
                0,
            )
            mx.eval(_)

        # Benchmark SDPA
        start = time.perf_counter()
        for _ in range(num_iters):
            out = paged_attention(
                q,
                key_cache,
                value_cache,
                block_tables,
                context_lengths,
                block_size,
                scale,
                num_kv_heads,
                0,
            )
            mx.eval(out)
        end = time.perf_counter()
        sdpa_time = (end - start) / num_iters * 1000

        # Warmup Metal
        for _ in range(5):
            _ = metal_paged_attention(
                q,
                key_cache,
                value_cache,
                block_tables,
                context_lengths,
                block_size,
                scale,
                num_kv_heads,
                0,
            )
            mx.eval(_)

        # Benchmark Metal
        start = time.perf_counter()
        for _ in range(num_iters):
            out = metal_paged_attention(
                q,
                key_cache,
                value_cache,
                block_tables,
                context_lengths,
                block_size,
                scale,
                num_kv_heads,
                0,
            )
            mx.eval(out)
        end = time.perf_counter()
        metal_time = (end - start) / num_iters * 1000

        speedup = metal_time / sdpa_time if sdpa_time > 0 else 0

        print(f"\n[{params['desc']}] BS={batch_size}, Len={seq_len}, H={num_heads}, KV_H={num_kv_heads}")
        print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        print(f"  SDPA Paged:  {sdpa_time:.3f} ms")
        print(f"  Metal Paged: {metal_time:.3f} ms")
        print(f"  Speedup (SDPA vs Metal): {speedup:.2f}x")

        # Assert correctness (allow small diff due to float32 vs native precision)
        assert max_diff < 0.01, f"Results differ too much: max_diff={max_diff}"

