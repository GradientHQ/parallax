"""
Unit tests for the Executor class, using Qwen3-0.6B-bf16.
"""

import pytest
from mlx_lm.generate import generate
from mlx_lm.utils import get_model_path, load_model

from parallax.server.executor import Executor
from parallax.server.request import InitialRequest
from parallax.utils.tokenizer_utils import load_tokenizer

MODEL_REPO = "mlx-community/Qwen3-0.6B-bf16"

model_path = get_model_path(MODEL_REPO)[0]
ref_model, ref_config = load_model(model_path)
ref_tokenizer = load_tokenizer(model_path, eos_token_ids=ref_config.get("eos_token_id", None))


@pytest.mark.parametrize("start_layer, end_layer", [(0, 12)])
@pytest.mark.parametrize("num_decode_steps", [8])
def test_decode_pipeline_multiple_steps(start_layer, end_layer, num_decode_steps):
    """Tests a multi-step decode pipeline with batched requests."""
    # 1. Setup executors
    executor_peer1 = Executor(
        model_repo=MODEL_REPO,
        start_layer=start_layer,
        end_layer=end_layer,
        kv_cache_memory_fraction=0.1,
        dtype="bfloat16",
    )
    executor_peer2 = Executor(
        model_repo=MODEL_REPO,
        start_layer=end_layer,
        end_layer=ref_config.get("num_hidden_layers"),
        kv_cache_memory_fraction=0.1,
        dtype="bfloat16",
    )

    # 2. Setup initial requests for multiple prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]
    initial_requests = [
        InitialRequest(request_id=f"req{i}", input_ids=executor_peer1.tokenizer.encode(p))
        for i, p in enumerate(prompts)
    ]

    executor_peer1._handle_input_requests(initial_requests)
    executor_peer1.scheduler.admit_requests()
    prefill_batch_p1 = executor_peer1.scheduler.form_batch()
    prefill_inputs_p1 = executor_peer1._prepare_batch_inputs(prefill_batch_p1)
    assert prefill_inputs_p1 is not None, "Failed to prepare batch inputs"
    prefill_batch_data = prefill_inputs_p1["prefill_batch"]
    hidden_states_p1 = executor_peer1.process_batch(prefill_batch_data, return_decoded_tokens=False)
    prefill_reqs_p2 = executor_peer1._prepare_next_batch_requests(
        requests=prefill_batch_data["requests"],
        hidden_states=hidden_states_p1,
        lengths=prefill_batch_data["lengths"],
    )

    # send to next peer

    executor_peer2._handle_input_requests(prefill_reqs_p2)
    executor_peer2.scheduler.admit_requests()
    prefill_batch_p2 = executor_peer2.scheduler.form_batch()
    prefill_inputs_p2 = executor_peer2._prepare_batch_inputs(prefill_batch_p2)
    assert prefill_inputs_p2 is not None, "Failed to prepare batch inputs"
    prefill_batch_data = prefill_inputs_p2["prefill_batch"]
    gen_tokens_mx = executor_peer2.process_batch(prefill_batch_data, return_decoded_tokens=True)
    generated_tokens_pipeline = [gen_tokens_mx]
    print(f"Prefill done: generated_tokens_pipeline: {generated_tokens_pipeline}")

    for _ in range(num_decode_steps):
        # 1. Simulate feedback from Peer 2 to Peer 1
        feedback_reqs = executor_peer2._prepare_next_batch_requests(
            requests=prefill_batch_data["requests"],
            hidden_states=generated_tokens_pipeline[-1],
            lengths=prefill_batch_data["lengths"],  # Not used by last peer
        )

        # 2. Peer 1: process feedback, commit token, re-enqueue
        executor_peer1._handle_input_requests(feedback_reqs)

        # 3. Peer 1: form and process decode batch
        executor_peer1.scheduler.admit_requests()
        decode_batch_p1 = executor_peer1.scheduler.form_batch()
        decode_inputs_p1 = executor_peer1._prepare_batch_inputs(decode_batch_p1)
        assert decode_inputs_p1 is not None, "Failed to prepare batch inputs"
        decode_batch_data = decode_inputs_p1["decode_batch"]
        decode_hidden_states_p1 = executor_peer1.process_batch(
            decode_batch_data, return_decoded_tokens=False
        )

        # 4. Peer 1 -> Peer 2: Prepare next decode batch
        decode_reqs_p2 = executor_peer1._prepare_next_batch_requests(
            requests=decode_batch_data["requests"],
            hidden_states=decode_hidden_states_p1,
            lengths=decode_batch_data["lengths"],
        )

        # 5. Peer 2: process decode batch to get next tokens
        executor_peer2._handle_input_requests(decode_reqs_p2)
        executor_peer2.scheduler.admit_requests()
        decode_batch_p2 = executor_peer2.scheduler.form_batch()
        decode_inputs_p2 = executor_peer2._prepare_batch_inputs(decode_batch_p2)
        assert decode_inputs_p2 is not None, "Failed to prepare batch inputs"
        decode_batch_data = decode_inputs_p2["decode_batch"]
        next_gen_tokens_mx = executor_peer2.process_batch(
            decode_batch_data, return_decoded_tokens=True
        )
        generated_tokens_pipeline.append(next_gen_tokens_mx)

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
        assert ref_output_text == output_text[1:]
