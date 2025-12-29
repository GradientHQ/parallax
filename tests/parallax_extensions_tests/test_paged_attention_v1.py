import math
import time

import mlx.core as mx
import numpy as np
import pytest

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

    def test_benchmark_paged_vs_native(self):
        mx.random.seed(42)
        bs = 8
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

        print(f"\n[Benchmark BS={bs}, Len={seq_len}, Float16]")

        if hasattr(mx.fast, "scaled_dot_product_attention"):
            q_native = q[:, :, None, :]
            for _ in range(10):
                _ = mx.fast.scaled_dot_product_attention(q_native, k_cont, v_cont, scale=scale)
            mx.eval(_)
            mx.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                out = mx.fast.scaled_dot_product_attention(q_native, k_cont, v_cont, scale=scale)
                mx.eval(out)
            mx.synchronize()
            end = time.perf_counter()
            print(f"Native SDPA:   {(end-start)*10:.3f} ms")

        # Paged Attention V1
        for _ in range(10):
            _ = paged_attention_v1(
                q, key_cache, value_cache, block_tables, seq_lens, block_size, scale, n_kv_heads
            )
        mx.eval(_)
        mx.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            out = paged_attention_v1(
                q, key_cache, value_cache, block_tables, seq_lens, block_size, scale, n_kv_heads
            )
            mx.eval(out)
        mx.synchronize()
        end = time.perf_counter()
        print(f"Paged Attn V1: {(end-start)*10:.3f} ms")
