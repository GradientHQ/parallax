"""
Unit tests for the Executor class, using Qwen3-0.6B-bf16.
For ubuntu-GPU, test 3 pipelines
  - cuda -> cuda -> cuda
  - cuda -> mlx(cpu) -> cuda
  - mlx(cpu) -> cuda -> mlx(cpu)
For MAC, test 1 pipeline
  - mlx -> mlx -> mlx
"""

import pytest
from mlx_lm.generate import generate
from mlx_lm.utils import get_model_path, load_model

from parallax.p2p.message_util import proto_to_request, request_to_proto
from parallax.server.request import InitialRequest
from parallax.utils.tokenizer_utils import load_tokenizer
from parallax.utils.utils import get_current_device

MLX_MODEL_REPO = "mlx-community/Qwen3-0.6B-bf16"
CUDA_MODEL_REPO = "Qwen/Qwen3-0.6B"

model_path = get_model_path(MLX_MODEL_REPO)[0]
ref_model, ref_config = load_model(model_path)
ref_tokenizer = load_tokenizer(model_path, eos_token_ids=ref_config.get("eos_token_id", None))


def create_executor(start_layer, end_layer, device):
    """Create a pipeline sharded executor"""
    if device == "mlx":
        model_repo = MLX_MODEL_REPO
        from parallax.server.executor.mlx_executor import MLXExecutor

        executor = MLXExecutor(
            model_repo=model_repo,
            start_layer=start_layer,
            end_layer=end_layer,
            kv_cache_memory_fraction=0.3,
            dtype="bfloat16",
            device=device,
        )
    else:
        model_repo = CUDA_MODEL_REPO
        from parallax.server.executor.sglang_executor import SGLExecutor

        executor = SGLExecutor(
            model_repo=model_repo,
            start_layer=start_layer,
            end_layer=end_layer,
            kv_cache_memory_fraction=0.3,
            dtype="bfloat16",
            device=device,
        )
    return executor


def run_executor_pipeline_stage(executor, requests, batch_type, is_last_peer):
    """Run executor pipeline stage. Input and output should be requests"""
    executor.handle_input_requests(requests)
    executor.scheduler.admit_requests()
    input_batch = executor.scheduler.form_batch()
    prepared_batch = executor.prepare_batch_inputs(input_batch)
    assert prepared_batch is not None, "Failed to prepare batch inputs"
    batch_data = prepared_batch[batch_type]
    hidden_states = executor.process_batch(batch_data, return_decoded_tokens=is_last_peer)
    output_reqs = executor.prepare_next_batch_requests(
        requests=batch_data["requests"],
        hidden_states=hidden_states,
        lengths=batch_data["lengths"],
    )
    return output_reqs, hidden_states


@pytest.mark.parametrize(
    "pipeline_devices",
    [
        ("cuda", "cuda", "cuda"),
        ("cuda", "mlx", "cuda"),
        ("mlx", "cuda", "mlx"),
        ("mlx", "mlx", "mlx"),
    ],
)
@pytest.mark.parametrize("pp_end_layers", [(10, 18, 28)])
@pytest.mark.parametrize("num_decode_steps", [8])
def test_decode_pipeline_multiple_steps(pipeline_devices, pp_end_layers, num_decode_steps):
    """Tests a multi-step decode pipeline with batched requests."""
    device = get_current_device()
    if device == "mlx" and "cuda" in pipeline_devices:
        return

    # 1. Setup executors
    assert pp_end_layers[2] == ref_config.get("num_hidden_layers")
    executor_peer1 = create_executor(
        start_layer=0,
        end_layer=pp_end_layers[0],
        device=pipeline_devices[0],
    )
    executor_peer2 = create_executor(
        start_layer=pp_end_layers[0],
        end_layer=pp_end_layers[1],
        device=pipeline_devices[1],
    )
    executor_peer3 = create_executor(
        start_layer=pp_end_layers[1],
        end_layer=pp_end_layers[2],
        device=pipeline_devices[2],
    )

    # 2. Setup initial requests for multiple prompts
    prompts = [
        "The capital of France is",
        "Qwen is a large language model developed by",
    ]
    initial_requests = [
        InitialRequest(request_id=f"req{i}", input_ids=executor_peer1.tokenizer.encode(p))
        for i, p in enumerate(prompts)
    ]

    # 3. Prefill
    prefill_reqs_out1, _ = run_executor_pipeline_stage(
        executor_peer1, initial_requests, "prefill_batch", False
    )
    prefill_proto_p1 = request_to_proto(prefill_reqs_out1, device=pipeline_devices[0])

    prefill_reqs_in2 = proto_to_request(prefill_proto_p1, device=pipeline_devices[1])
    prefill_reqs_out2, _ = run_executor_pipeline_stage(
        executor_peer2, prefill_reqs_in2, "prefill_batch", False
    )
    prefill_proto_p2 = request_to_proto(prefill_reqs_out2, device=pipeline_devices[1])

    prefill_reqs_in3 = proto_to_request(prefill_proto_p2, device=pipeline_devices[2])
    prefill_reqs_out3, gen_tokens = run_executor_pipeline_stage(
        executor_peer3, prefill_reqs_in3, "prefill_batch", True
    )
    prefill_proto_p3 = request_to_proto(prefill_reqs_out3, device=pipeline_devices[2])

    generated_tokens_pipeline = [gen_tokens]
    print(f"Prefill done: generated_tokens_pipeline: {generated_tokens_pipeline}")
    first_rank_proto = prefill_proto_p3

    # 4. Decode
    for _ in range(num_decode_steps):
        decode_reqs_in1 = proto_to_request(first_rank_proto, device=pipeline_devices[0])
        decode_reqs_out1, _ = run_executor_pipeline_stage(
            executor_peer1, decode_reqs_in1, "decode_batch", False
        )
        decode_proto_p1 = request_to_proto(decode_reqs_out1, device=pipeline_devices[0])

        decode_reqs_in2 = proto_to_request(decode_proto_p1, device=pipeline_devices[1])
        decode_reqs_out2, _ = run_executor_pipeline_stage(
            executor_peer2, decode_reqs_in2, "decode_batch", False
        )
        decode_proto_p2 = request_to_proto(decode_reqs_out2, device=pipeline_devices[1])

        decode_reqs_in3 = proto_to_request(decode_proto_p2, device=pipeline_devices[2])
        decode_reqs_out3, next_gen_tokens = run_executor_pipeline_stage(
            executor_peer3, decode_reqs_in3, "decode_batch", True
        )
        decode_proto_p3 = request_to_proto(decode_reqs_out3, device=pipeline_devices[2])

        first_rank_proto = decode_proto_p3
        generated_tokens_pipeline.append(next_gen_tokens)

    # 5. Compare with reference
    total_tokens_to_generate = 1 + num_decode_steps
    for i, prompt in enumerate(prompts):
        # Generate reference tokens using mlx-lm's standard generation
        ref_output_text = generate(
            ref_model,
            ref_tokenizer,
            prompt,
            max_tokens=total_tokens_to_generate,
            verbose=False,
        )
        print(f"prompt: {prompt}")
        print(f"mlx-lm reference generation: {ref_output_text}")
        output_tokens_for_prompt = [
            gen_step_tokens[i].item() for gen_step_tokens in generated_tokens_pipeline
        ]

        # Decode the token IDs into a string
        output_text = executor_peer1.tokenizer.decode(output_tokens_for_prompt)
        print(f"parallax test generation: {output_text}")

        # Trim the first whitespace in our output
        assert ref_output_text[:5] == output_text[1:6]

    # 6. Release resources for next tests
    executor_peer1.shutdown()
    executor_peer2.shutdown()
    executor_peer3.shutdown()
    del executor_peer1
    del executor_peer2
    del executor_peer3
