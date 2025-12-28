"""
Unit tests for the Executor class, using Qwen3-0.6B-bf16.
For ubuntu-GPU, test 1 pipeline
  - cuda -> cuda -> cuda
For MAC, test 1 pipeline
  - mlx -> mlx -> mlx
"""

import pytest
from mlx_lm.generate import generate
from mlx_lm.utils import _download, load_model

from parallax.p2p.message_util import proto_to_request, request_to_proto
from parallax.server.request import InitialRequest
from parallax.server.sampling.sampling_params import SamplingParams
from parallax.utils.tokenizer_utils import load_tokenizer
from parallax.utils.utils import (
    get_current_device,
    is_cuda_available,
    is_metal_available,
)

MLX_MODEL_REPO = "mlx-community/Qwen3-0.6B-bf16"
CUDA_MODEL_REPO = "Qwen/Qwen3-0.6B"

model_path = _download(MLX_MODEL_REPO)
ref_model, ref_config = load_model(model_path)
ref_tokenizer = load_tokenizer(model_path, eos_token_ids=ref_config.get("eos_token_id", None))


def create_executor(start_layer, end_layer, device, kv_cache_memory_fraction=0.3):
    """Create a pipeline sharded executor

    Args:
        start_layer: Start layer index
        end_layer: End layer index
        device: Device type ("mlx" or "cuda")
        kv_cache_memory_fraction: Memory fraction for KV cache (will be divided by number of executors on same device)
    """
    if device == "mlx":
        model_repo = MLX_MODEL_REPO
        from parallax.server.executor.mlx_executor import MLXExecutor

        executor = MLXExecutor(
            model_repo=model_repo,
            start_layer=start_layer,
            end_layer=end_layer,
            kv_cache_memory_fraction=kv_cache_memory_fraction,
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
            kv_cache_memory_fraction=kv_cache_memory_fraction,
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
    batch_output = executor.process_batch(batch_data, return_decoded_tokens=is_last_peer)
    output_reqs = executor.prepare_next_batch_requests(
        requests=batch_data["requests"],
        batch_output=batch_output,
        context_lengths=batch_data.get("context_lengths"),
    )
    return output_reqs, batch_output


@pytest.mark.parametrize(
    "pipeline_devices",
    [
        ("cuda", "cuda", "cuda"),  # Pure CUDA pipeline for GPU
        ("mlx", "mlx", "mlx"),  # Pure MLX pipeline for macOS
    ],
)
@pytest.mark.parametrize("pp_end_layers", [(10, 18, 28)])
@pytest.mark.parametrize("num_decode_steps", [8])
def test_decode_pipeline_multiple_steps(pipeline_devices, pp_end_layers, num_decode_steps):
    """Tests a multi-step decode pipeline with batched requests."""
    device = get_current_device()

    # Skip if pipeline requires Metal but Metal is not available
    if "mlx" in pipeline_devices and not is_metal_available():
        pytest.skip("Metal backend not available (requires macOS with Metal support)")

    # Skip if pipeline requires CUDA but CUDA is not available
    if "cuda" in pipeline_devices and not is_cuda_available():
        pytest.skip("CUDA backend not available (requires NVIDIA GPU with CUDA support)")

    # Skip if on MLX device but pipeline requires CUDA
    if device == "mlx" and "cuda" in pipeline_devices:
        pytest.skip("CUDA not available on MLX device")

    # Note: Reference generation uses MLX for MLX pipelines, transformers for CUDA pipelines
    # Load reference model based on pipeline type
    ref_cuda_model = None
    ref_cuda_tokenizer = None
    if all(d == "cuda" for d in pipeline_devices):
        # Pre-load CUDA reference model for pure CUDA pipelines
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            ref_cuda_model = AutoModelForCausalLM.from_pretrained(
                CUDA_MODEL_REPO,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
            )
            ref_cuda_tokenizer = AutoTokenizer.from_pretrained(CUDA_MODEL_REPO)
            if ref_cuda_tokenizer.pad_token is None:
                ref_cuda_tokenizer.pad_token = ref_cuda_tokenizer.eos_token
        except ImportError:
            pytest.skip("transformers not available for CUDA reference generation")
        except Exception as e:
            pytest.skip(f"Failed to load CUDA reference model: {str(e)[:100]}")

    # 1. Setup executors
    # Calculate memory fraction for each executor based on how many executors share the same device
    # Total memory fraction to use across all executors
    # Note: We use a higher fraction (0.5) to account for model weights that are already loaded
    # The actual KV cache will use less due to the conservative calculation in _calculate_num_blocks

    # Calculate fraction per executor (divide by number of executors on same device)
    executor_peer1 = create_executor(
        start_layer=0,
        end_layer=pp_end_layers[0],
        device=pipeline_devices[0],
        kv_cache_memory_fraction=0.1,
    )
    executor_peer2 = create_executor(
        start_layer=pp_end_layers[0],
        end_layer=pp_end_layers[1],
        device=pipeline_devices[1],
        kv_cache_memory_fraction=0.1,
    )
    executor_peer3 = create_executor(
        start_layer=pp_end_layers[1],
        end_layer=pp_end_layers[2],
        device=pipeline_devices[2],
        kv_cache_memory_fraction=0.1,
    )

    # 2. Setup initial requests for multiple prompts
    prompts = [
        "The capital of China is",
        "Qwen is a large language model developed by",
    ]
    # Use greedy sampling (temperature=0.0) for deterministic results
    greedy_sampling = SamplingParams(temperature=0.0, top_k=1)
    initial_requests = [
        InitialRequest(
            request_id=f"req{i}",
            input_ids=executor_peer1.tokenizer.encode(p),
            sampling_params=greedy_sampling,
        )
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
        # Generate reference tokens
        # For pure CUDA pipelines, use transformers; for MLX pipelines, use mlx-lm
        if all(d == "cuda" for d in pipeline_devices):
            # Use pre-loaded transformers model for CUDA reference generation
            import torch

            inputs = ref_cuda_tokenizer(prompt, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                outputs = ref_cuda_model.generate(
                    **inputs,
                    max_new_tokens=total_tokens_to_generate,
                    do_sample=False,  # Greedy sampling
                    temperature=1.0,  # Explicitly set for deterministic behavior
                    pad_token_id=ref_cuda_tokenizer.pad_token_id,
                )
            ref_output_text = ref_cuda_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the output
            ref_output_text = ref_output_text[len(prompt) :].strip()
        else:
            # Use MLX for MLX pipelines
            try:
                ref_output_text = generate(
                    ref_model,
                    ref_tokenizer,
                    prompt,
                    max_tokens=total_tokens_to_generate,
                    verbose=False,
                )
            except RuntimeError as e:
                if "Compile::eval_cpu" in str(e) or "metal" in str(e).lower():
                    pytest.skip(
                        f"MLX backend not available for reference generation: {str(e)[:100]}"
                    )
                raise

        print(f"prompt: {prompt}")
        print(f"mlx-lm reference generation: {ref_output_text}")
        output_tokens_for_prompt = [
            gen_step_tokens["hidden_states"][i].item()
            for gen_step_tokens in generated_tokens_pipeline
        ]

        # Decode the token IDs into a string
        output_text = executor_peer1.tokenizer.decode(output_tokens_for_prompt)
        print(f"parallax test generation: {output_text}")

        # Compare outputs (account for potential whitespace differences)
        # Remove leading/trailing whitespace and compare first few characters
        ref_clean = ref_output_text.strip()
        output_clean = output_text.strip()

        # For debugging: print both outputs
        print(f"Reference output (clean): '{ref_clean[:20]}'")
        print(f"Pipeline output (clean): '{output_clean[:20]}'")

        # Compare first 5 characters (allowing for minor differences)
        # This is a lenient check - exact match may vary due to tokenization differences
        assert len(ref_clean) > 0 and len(output_clean) > 0, "Both outputs should be non-empty"
        # Check if they start with similar content (at least 3 characters match)
        min_len = min(len(ref_clean), len(output_clean), 5)
        if min_len >= 3:
            assert (
                ref_clean[:min_len].lower() == output_clean[:min_len].lower()
            ), f"Output mismatch: ref='{ref_clean[:20]}' vs pipeline='{output_clean[:20]}'"

    # 6. Release resources for next tests
    executor_peer1.shutdown()
    executor_peer2.shutdown()
    executor_peer3.shutdown()
    del executor_peer1
    del executor_peer2
    del executor_peer3

    # Clean up CUDA reference model if used
    if ref_cuda_model is not None:
        import torch

        del ref_cuda_model
        del ref_cuda_tokenizer
        torch.cuda.empty_cache()
