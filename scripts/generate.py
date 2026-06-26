"""
Simple offline inference script

Example command:

single node:
    python scripts/generate.py

tensor parallel:
    https://ml-explore.github.io/mlx/build/html/usage/distributed.html#enabling-rdma

    mlx.distributed_config --verbose \
    --hosts macmini1,macmini2 \
    --over thunderbolt --backend jaccl \
    --auto-setup --output hosts.json

    mlx.launch \
    --backend jaccl \
    --env MLX_METAL_FAST_SYNCH=1 \
    --hostfile hosts.json \
    scripts/generate.py
"""

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm.server import convert_chat, process_message_content

from parallax.server.cache_manager import CacheManager
from parallax.server.request import InitialRequest
from parallax.server.sampling.sampler import SamplingBatchInfo
from parallax.server.sampling.sampling_params import SamplingParams
from parallax.server.scheduler import _normalize_token_ids
from parallax.server.shard_loader import MLXModelLoader
from parallax.utils.utils import create_causal_mask, get_layer_types

tp_size = 1
tp_rank = 0


def print_rank(message):
    if tp_size == 1:
        print(message)
    else:
        print(f"[Rank {tp_rank}] {message}")


def get_eos_token_ids(config, tokenizer):
    eos_token_id = config.get("eos_token_id")
    tokenizer_eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer_eos_token_id

    eos_token_ids = _normalize_token_ids(eos_token_id)
    eos_token_ids.update(_normalize_token_ids(tokenizer_eos_token_id))
    return eos_token_ids


def build_prompt(messages, tokenizer):
    if tokenizer.chat_template:
        process_message_content(messages)
        prompt_tokens = tokenizer.apply_chat_template(
            messages,
            None,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        full_prompt = tokenizer.apply_chat_template(
            messages,
            None,
            tokenize=False,
            add_generation_prompt=True,
            return_dict=False,
        )
    else:
        full_prompt = convert_chat(messages, None)
        prompt_tokens = tokenizer.encode(full_prompt)
    return full_prompt, prompt_tokens


def percentile(values, percentile_value):
    if not values:
        return 0

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * percentile_value / 100
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def main():
    parser = argparse.ArgumentParser(description="Simple offline inference script")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-32B-MLX-4bit", help="Model path or HF repo"
    )
    parser.add_argument("--prompt", type=str, default="Hi", help="Prompt for inference")
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate"
    )
    parser.add_argument("--topk", type=int, default=1, help="Top-k sampling parameter")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature for sampling")
    args = parser.parse_args()

    # TP Initialization
    global tp_size, tp_rank
    group = mx.distributed.init()
    tp_rank = group.rank()
    tp_size = group.size()

    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    # 1. Load Model
    print_rank(f"Loading model from {args.model}...")

    loader = MLXModelLoader(
        args.model,
    )
    model, config, tokenizer = loader.load()

    # 2. Initialize CacheManager
    num_layers = config.get("num_hidden_layers")
    num_kv_heads = config.get("num_key_value_heads")
    if num_kv_heads is None:
        num_kv_heads = config.get("num_attention_groups")
    head_dim = config.get("head_dim") or config.get("hidden_size") // config.get(
        "num_attention_heads"
    )

    # Check for DeepSeek style head dims
    qk_nope_head_dim = config.get("qk_nope_head_dim")
    qk_rope_head_dim = config.get("qk_rope_head_dim")
    if qk_nope_head_dim is not None and qk_rope_head_dim is not None:
        head_dim = qk_nope_head_dim + qk_rope_head_dim

    v_head_dim = config.get("v_head_dim")
    layer_types = get_layer_types(config, 0, num_layers)
    linear_key_head_dim = config.get("linear_key_head_dim")
    linear_value_head_dim = config.get("linear_value_head_dim")
    linear_conv_kernel_dim = config.get("linear_conv_kernel_dim")
    linear_num_key_heads = config.get("linear_num_key_heads")
    linear_num_value_heads = config.get("linear_num_value_heads")
    key_dim, value_dim, conv_dim = None, None, None
    if linear_key_head_dim is not None and linear_num_key_heads is not None:
        key_dim = linear_key_head_dim * linear_num_key_heads
    if linear_value_head_dim is not None and linear_num_value_heads is not None:
        value_dim = linear_value_head_dim * linear_num_value_heads
    if key_dim is not None and value_dim is not None:
        conv_dim = key_dim * 2 + value_dim

    cache_manager = CacheManager(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads // tp_size,  # Shard heads
        head_dim=head_dim,
        dtype=model.dtype,
        block_size=1,
        cache_memory_fraction=0.1,
        max_num_seqs=1,
        head_dim_v=v_head_dim,
        layer_types=layer_types,
        conv_dim=conv_dim,
        conv_kernel_size=linear_conv_kernel_dim,
        linear_k_dim=linear_key_head_dim,
        linear_v_dim=linear_value_head_dim,
        linear_num_k_heads=linear_num_key_heads,
        linear_num_v_heads=linear_num_value_heads,
    )

    # 3. Tokenize and Create Request
    messages = [{"role": "user", "content": args.prompt}]
    full_prompt, prompt_tokens = build_prompt(messages, tokenizer)
    sampling_params = SamplingParams(temperature=args.temp, top_k=args.topk)
    request = InitialRequest(
        prompt=full_prompt,
        input_ids=prompt_tokens,
        sampling_params=sampling_params,
        max_new_tokens=args.max_tokens,
    )

    eos_token_ids = get_eos_token_ids(config, tokenizer)
    if not eos_token_ids:
        raise ValueError("EOS token ID must be set for generation.")

    # 4. Prefill
    print_rank(f"Full prompt:\n {full_prompt}")

    if tp_size > 1:
        mx.eval(mx.distributed.all_sum(mx.ones(1)))
        print_rank("Forced sync before prefill")

    success, _ = cache_manager.allocate_request(request.request_id, request.prompt_len)
    if not success:
        print_rank("Failed to allocate cache")
        return

    input_ids = mx.array([request.input_ids])
    block_table = mx.array([cache_manager.get_block_table(request.request_id)], dtype=mx.int32)
    context_lengths = mx.array([request.prompt_len], dtype=mx.int32)
    state_slot_mapping = None
    if cache_manager.needs_slots:
        state_slot_mapping = mx.array([cache_manager.get_slot(request.request_id)], dtype=mx.int32)

    block_size = cache_manager.block_size
    slot_mapping = []
    for i in range(request.prompt_len):
        block_idx = i // block_size
        block_offset = i % block_size
        physical_block = cache_manager.get_block_table(request.request_id)[block_idx]
        slot_mapping.append(physical_block * block_size + block_offset)
    slot_mapping = mx.array(slot_mapping, dtype=mx.int64)

    mask = create_causal_mask(request.prompt_len, request.prompt_len, model.dtype)

    prefill_start = time.perf_counter()

    logits = model(
        input_ids,
        cache=cache_manager.get_caches(),
        mask=mask,
        block_tables=block_table,
        context_lengths=context_lengths,
        slot_mapping=slot_mapping,
        state_slot_mapping=state_slot_mapping,
    )

    sampling_info = SamplingBatchInfo.from_reqs([request])
    token_sampling_info = None if sampling_info.is_all_greedy else sampling_info

    next_token_id = model.logits_to_tokens(logits, context_lengths, token_sampling_info)

    token_id = int(next_token_id[0])
    is_finished = token_id in eos_token_ids
    if not is_finished:
        request.commit_new_token(token_id)

    prefill_time = time.perf_counter() - prefill_start
    print_rank(f"Token 1 (Prefill) time: {prefill_time * 1000:.2f} ms")

    # 5. Decode Loop
    decode_step_times = []
    for i in range(args.max_tokens - 1):
        if is_finished:
            break

        decode_step_start = time.perf_counter()

        success = cache_manager.append_slot(request.request_id)
        if not success:
            print_rank("\nOOM during decoding")
            break

        block_table_list = cache_manager.get_block_table(request.request_id)
        context_length = cache_manager.get_context_length(request.request_id)
        block_table = mx.array([block_table_list], dtype=mx.int32)
        context_lengths = mx.array([context_length], dtype=mx.int32)
        token_pos = context_length - 1
        block_idx = token_pos // cache_manager.block_size
        block_offset = token_pos % cache_manager.block_size
        physical_block = block_table_list[block_idx]
        slot_mapping = mx.array(
            [physical_block * cache_manager.block_size + block_offset],
            dtype=mx.int64,
        )
        logits = model(
            mx.expand_dims(next_token_id, axis=0),
            cache=cache_manager.get_caches(),
            mask=None,
            block_tables=block_table,
            context_lengths=context_lengths,
            slot_mapping=slot_mapping,
            state_slot_mapping=state_slot_mapping,
        )

        next_token_id = model.logits_to_tokens(logits, sampling_info=token_sampling_info)

        token_id = int(next_token_id[0])
        is_finished = token_id in eos_token_ids
        if is_finished:
            break
        request.commit_new_token(token_id)

        decode_step_time = time.perf_counter() - decode_step_start
        decode_step_times.append(decode_step_time)
        print_rank(f"Token {i + 2} time: {decode_step_time * 1000:.2f} ms")

    print_rank("\nGenerated Content:")
    print_rank(tokenizer.decode(request.output_ids))

    # Summary Statistics
    total_decode_time = sum(decode_step_times)
    prompt_tps = request.prompt_len / prefill_time
    generation_tps = len(request.output_ids) / total_decode_time if total_decode_time > 0 else 0
    peak_mem = mx.get_peak_memory() / 1024**3

    print_rank("-" * 20)
    print_rank(f"Prompt: {request.prompt_len} tokens, {prompt_tps:.3f} tokens-per-sec")
    print_rank(f"Generation: {len(request.output_ids)} tokens, {generation_tps:.3f} tokens-per-sec")
    if decode_step_times:
        decode_step_times_ms = [step_time * 1000 for step_time in decode_step_times]
        decode_step_time_mean = sum(decode_step_times_ms) / len(decode_step_times_ms)
        print_rank(
            "Decode step time (ms): "
            f"min={min(decode_step_times_ms):.2f}, "
            f"median={percentile(decode_step_times_ms, 50):.2f}, "
            f"max={max(decode_step_times_ms):.2f}, "
            f"mean={decode_step_time_mean:.2f}, "
            f"std={statistics.pstdev(decode_step_times_ms):.2f}, "
            f"p90={percentile(decode_step_times_ms, 90):.2f}, "
            f"p95={percentile(decode_step_times_ms, 95):.2f}, "
            f"p99={percentile(decode_step_times_ms, 99):.2f}"
        )
    else:
        print_rank("Decode step time (ms): n/a")
    print_rank(f"Peak memory: {peak_mem:.3f} GB")
    cache_manager.free_request(request.request_id)


if __name__ == "__main__":
    main()
