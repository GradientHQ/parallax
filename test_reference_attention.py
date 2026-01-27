"""Compare both kernels against a reference implementation."""

import math

import mlx.core as mx
import numpy as np

# Old kernel
from parallax.metal.paged_attention.kernel import paged_attention as old_paged_attention
from parallax.metal.paged_attention.kernel import (
    reshape_and_cache as old_reshape_and_cache,
)

# New kernel
from parallax_extensions.ops import paged_attention_v1 as new_paged_attention
from parallax_extensions.ops import reshape_and_cache as new_reshape_and_cache


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
    """Reference paged attention implementation."""
    batch_size, num_heads, _, head_dim = q.shape
    _, _, seq_len, _ = k.shape

    # For MHA/GQA, repeat KV heads to match Q heads
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

    # Compute attention scores
    # q: (batch, num_heads, 1, head_dim)
    # k: (batch, num_heads, seq_len, head_dim)
    scores = (q_f32 @ k_f32.transpose(0, 1, 3, 2)) * scale

    # Mask padding tokens beyond context length.
    positions = mx.arange(seq_len)[None, None, None, :]
    ctx = context_lengths[:, None, None, None]
    scores = mx.where(positions >= ctx, -float("inf"), scores)

    # Apply sliding window mask: keep [context_len - 1 - window_size, context_len - 1].
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

    # Apply to values
    output = attn_weights @ v_f32  # (batch, num_heads, 1, head_dim)
    if output.ndim == 3:
        output = output[:, :, None, :]

    return output


def test_against_reference():
    """Compare both kernels against reference."""
    mx.random.seed(42)
    np.random.seed(42)

    # Test configuration
    batch_size = 1
    num_heads = 2
    num_kv_heads = 2
    head_dim = 64
    block_size = 16
    seq_len = 32
    dtype = mx.float16
    scale = 1.0 / math.sqrt(head_dim)

    print("=" * 80)
    print(
        f"Testing: BS={batch_size}, Heads={num_heads}, KV_Heads={num_kv_heads}, D={head_dim}, SeqLen={seq_len}"
    )
    print("=" * 80)

    # Generate test data
    q = mx.random.uniform(shape=(batch_size, num_heads, 1, head_dim)).astype(dtype)

    # Generate KV sequence
    k_seq = mx.random.uniform(shape=(batch_size, num_kv_heads, seq_len, head_dim)).astype(dtype)
    v_seq = mx.random.uniform(shape=(batch_size, num_kv_heads, seq_len, head_dim)).astype(dtype)

    context_lengths = mx.array([seq_len], dtype=mx.int32)
    num_blocks = (seq_len + block_size - 1) // block_size
    block_tables = mx.arange(num_blocks, dtype=mx.int32)[None, :]

    verify_cache = False

    # ===== OLD kernel cache =====

    old_key_cache = mx.zeros((1, num_blocks, num_kv_heads, block_size, head_dim), dtype=dtype)
    old_value_cache = mx.zeros((1, num_blocks, num_kv_heads, block_size, head_dim), dtype=dtype)

    # Write all tokens to cache
    for t in range(seq_len):
        k_t = k_seq[:, :, t : t + 1, :]
        v_t = v_seq[:, :, t : t + 1, :]
        old_reshape_and_cache(
            k_t,
            v_t,
            old_key_cache,
            old_value_cache,
            block_tables,
            mx.array([t + 1], dtype=mx.int32),
            block_size,
        )

    mx.eval(old_key_cache, old_value_cache)

    x = 8 if dtype == mx.float16 else 4
    new_key_cache = mx.zeros((num_blocks, num_kv_heads, head_dim // x, block_size, x), dtype=dtype)
    new_value_cache = mx.zeros((num_blocks, num_kv_heads, head_dim, block_size), dtype=dtype)

    # Write all tokens to cache
    for t in range(seq_len):
        k_t = k_seq[:, :, t : t + 1, :]
        v_t = v_seq[:, :, t : t + 1, :]
        new_reshape_and_cache(
            k_t,
            v_t,
            new_key_cache,
            new_value_cache,
            block_tables,
            mx.array([t + 1], dtype=mx.int32),
            block_size,
        )

    mx.eval(new_key_cache, new_value_cache)

    if verify_cache:
        print("\nVerifying NEW cache contents...")
        key_cache_np = np.array(new_key_cache)
        value_cache_np = np.array(new_value_cache)
        block_tables_np = np.array(block_tables)
        k_seq_np = np.array(k_seq)
        v_seq_np = np.array(v_seq)

        recon_k = np.zeros((num_kv_heads, seq_len, head_dim), dtype=np.float32)
        recon_v = np.zeros((num_kv_heads, seq_len, head_dim), dtype=np.float32)
        for t in range(seq_len):
            block_idx_in_table = t // block_size
            block_offset = t % block_size
            physical_block = int(block_tables_np[0, block_idx_in_table])
            for h in range(num_kv_heads):
                k_block = key_cache_np[physical_block, h, :, block_offset, :]
                recon_k[h, t, :] = k_block.reshape(head_dim).astype(np.float32)
                recon_v[h, t, :] = value_cache_np[physical_block, h, :, block_offset].astype(
                    np.float32
                )

        k_ref = k_seq_np[0].astype(np.float32)
        v_ref = v_seq_np[0].astype(np.float32)
        k_diff = np.abs(recon_k - k_ref)
        v_diff = np.abs(recon_v - v_ref)
        k_head_max = np.max(k_diff, axis=(1, 2))
        v_head_max = np.max(v_diff, axis=(1, 2))
        print(f"  K per-head max diff: {k_head_max}")
        print(f"  V per-head max diff: {v_head_max}")

        # Linear-index verification using the exact cache indexing math.
        k_flat = key_cache_np.reshape(-1)
        v_flat = value_cache_np.reshape(-1)
        k_head_max_lin = np.zeros(num_kv_heads, dtype=np.float32)
        v_head_max_lin = np.zeros(num_kv_heads, dtype=np.float32)
        for h in range(num_kv_heads):
            max_k = 0.0
            max_v = 0.0
            for t in range(seq_len):
                block_idx_in_table = t // block_size
                block_offset = t % block_size
                physical_block = int(block_tables_np[0, block_idx_in_table])
                for d in range(head_dim):
                    x_idx = d // x
                    x_offset = d % x
                    k_linear = (
                        ((physical_block * num_kv_heads + h) * (head_dim // x) + x_idx)
                        * block_size
                        * x
                        + block_offset * x
                        + x_offset
                    )
                    v_linear = (
                        (physical_block * num_kv_heads + h) * head_dim + d
                    ) * block_size + block_offset
                    max_k = max(max_k, abs(float(k_flat[k_linear]) - float(k_ref[h, t, d])))
                    max_v = max(max_v, abs(float(v_flat[v_linear]) - float(v_ref[h, t, d])))
            k_head_max_lin[h] = max_k
            v_head_max_lin[h] = max_v
        print(f"  K per-head max diff (linear): {k_head_max_lin}")
        print(f"  V per-head max diff (linear): {v_head_max_lin}")

    def compare_to_reference(label, ref_out, out, atol=1e-3):
        ref_f32 = ref_out.astype(mx.float32)
        out_f32 = out.astype(mx.float32)
        diff = mx.abs(ref_f32 - out_f32)
        max_diff = mx.max(diff).item()
        mean_diff = mx.mean(diff).item()
        all_close = mx.allclose(ref_f32, out_f32, atol=atol).item()
        print(f"\n{label} vs Reference:")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        print(f"  All close (atol={atol:.0e}): {all_close}")
        if not all_close:
            diff_np = np.array(diff)
            ref_np = np.array(ref_f32)
            out_np = np.array(out_f32)
            if ref_np.shape != diff_np.shape:
                ref_np = np.broadcast_to(ref_np, diff_np.shape)
            if out_np.shape != diff_np.shape:
                out_np = np.broadcast_to(out_np, diff_np.shape)
            idx = np.unravel_index(np.argmax(diff_np), diff_np.shape)
            ref_val = float(ref_np[idx])
            out_val = float(out_np[idx])
            print(f"  max diff at {idx}: ref={ref_val:.6f} out={out_val:.6f}")
            per_head_max = np.max(diff_np, axis=(0, 2, 3))
            per_head_mean = np.mean(diff_np, axis=(0, 2, 3))
            print(f"  per-head max diff: {per_head_max}")
            print(f"  per-head mean diff: {per_head_mean}")
        return all_close

    sinks = mx.array(np.random.uniform(-0.5, 0.5, size=(num_heads,)), dtype=mx.float32)
    window_sizes = [block_size // 2, block_size]
    cases = [
        {"name": "baseline", "window_size": 0, "sinks": None, "use_old": True},
    ]
    for ws in window_sizes:
        cases.append({"name": f"window-{ws}", "window_size": ws, "sinks": None})
    cases.append({"name": "sink-only", "window_size": 0, "sinks": sinks})
    for ws in window_sizes:
        cases.append({"name": f"window-{ws}+sink", "window_size": ws, "sinks": sinks})

    all_pass = True
    for case in cases:
        case_name = case["name"]
        window_size = case["window_size"]
        sinks_case = case["sinks"]
        use_old = case.get("use_old", False)
        sink_desc = "off" if sinks_case is None else f"on {np.array(sinks_case)}"

        print("\n" + "-" * 80)
        print(f"Case: {case_name} (window_size={window_size}, sink={sink_desc})")
        print("-" * 80)

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
        print(f"Reference output sample: {ref_output[0, 0, 0, :5]}")

        case_ok = True
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
            print(f"Old output sample: {old_output[0, 0, 0, :5]}")
            case_ok &= compare_to_reference("Old kernel", ref_output, old_output, atol=1e-3)

        new_output = new_paged_attention(
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
        print(f"New output sample: {new_output[0, 0, 0, :5]}")
        case_ok &= compare_to_reference("New kernel", ref_output, new_output, atol=atol)

        if case_ok:
            print("✅ Case matches reference!")
        else:
            print("❌ Case mismatch detected!")
        all_pass &= case_ok

    if all_pass:
        print("\n✅ All cases match reference!")
    else:
        print("\n❌ Some cases failed. See details above.")
    return all_pass


if __name__ == "__main__":
    success = test_against_reference()
    exit(0 if success else 1)
