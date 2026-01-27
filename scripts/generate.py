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
import time

import mlx.core as mx

from parallax.server.cache_manager import CacheManager
from parallax.server.request import InitialRequest
from parallax.server.sampling.sampler import SamplingBatchInfo
from parallax.server.sampling.sampling_params import SamplingParams
from parallax.server.shard_loader import MLXModelLoader
from parallax.utils.utils import create_causal_mask, get_layer_types

tp_size = 1
tp_rank = 0


def print_rank(message):
    if tp_size == 1:
        print(message)
    else:
        print(f"[Rank {tp_rank}] {message}")


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

    mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])

    # 1. Load Model
    print_rank(f"Loading model from {args.model}...")

    loader = MLXModelLoader(
        args.model,
    )
    model, config, tokenizer = loader.load()

    # 2. Initialize CacheManager
    num_layers = config.get("num_hidden_layers")
    num_kv_heads = config.get("num_key_value_heads")
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

    cache_manager = CacheManager(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads // tp_size,  # Shard heads
        head_dim=head_dim,
        dtype=model.dtype,
        block_size=32,
        cache_memory_fraction=0.1,
        head_dim_v=v_head_dim,
        layer_types=layer_types,
    )

    # 3. Tokenize and Create Request
    messages = [{"role": "user", "content": args.prompt}]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        full_prompt = args.prompt

    prompt_tokens = tokenizer.encode(full_prompt)
    sampling_params = SamplingParams(temperature=args.temp, top_k=args.topk)
    request = InitialRequest(
        prompt=full_prompt,
        input_ids=prompt_tokens,
        sampling_params=sampling_params,
        max_new_tokens=args.max_tokens,
    )

    eos_token_ids = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_token_ids.extend(tokenizer.eos_token_id)
        else:
            eos_token_ids.append(tokenizer.eos_token_id)
    config_eos = config.get("eos_token_id")
    if config_eos is not None:
        if isinstance(config_eos, list):
            for e in config_eos:
                if e not in eos_token_ids:
                    eos_token_ids.append(e)
        elif config_eos not in eos_token_ids:
            eos_token_ids.append(config_eos)

    eos_token_ids = set(eos_token_ids)

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
    )

    sampling_info = SamplingBatchInfo.from_reqs([request])

    next_token_id = model.logits_to_tokens(logits, context_lengths, sampling_info)

    token_id = int(next_token_id[0])
    request.commit_new_token(token_id)

    prefill_time = time.perf_counter() - prefill_start
    print_rank(f"Token 1 (Prefill) time: {prefill_time * 1000:.2f} ms")

    # 5. Decode Loop
    total_decode_time = 0
    for i in range(args.max_tokens - 1):
        decode_step_start = time.perf_counter()

        success = cache_manager.append_slot(request.request_id)
        if not success:
            print_rank("\nOOM during decoding")
            break

        block_table = mx.array([cache_manager.get_block_table(request.request_id)], dtype=mx.int32)
        context_lengths = mx.array(
            [cache_manager.get_context_length(request.request_id)], dtype=mx.int32
        )
        logits = model(
            mx.expand_dims(next_token_id, axis=0),
            cache=cache_manager.get_caches(),
            mask=None,
            block_tables=block_table,
            context_lengths=context_lengths,
        )

        next_token_id = model.logits_to_tokens(logits, mx.array([1]), sampling_info)

        token_id = int(next_token_id[0])
        if token_id in eos_token_ids:
            break
        request.commit_new_token(token_id)

        decode_step_time = time.perf_counter() - decode_step_start
        total_decode_time += decode_step_time
        print_rank(f"Token {i + 2} time: {decode_step_time * 1000:.2f} ms")

    print_rank("\nGenerated Content:")
    print_rank(tokenizer.decode(request.output_ids))

    # Summary Statistics
    prompt_tps = request.prompt_len / prefill_time
    generation_tps = len(request.output_ids) / total_decode_time if total_decode_time > 0 else 0
    peak_mem = mx.get_peak_memory() / 1024**3

    print_rank("-" * 20)
    print_rank(f"Prompt: {request.prompt_len} tokens, {prompt_tps:.3f} tokens-per-sec")
    print_rank(f"Generation: {len(request.output_ids)} tokens, {generation_tps:.3f} tokens-per-sec")
    print_rank(f"Peak memory: {peak_mem:.3f} GB")
    cache_manager.free_request(request.request_id)


if __name__ == "__main__":
    main()
