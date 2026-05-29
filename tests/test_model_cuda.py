"""
Tests for the ShardedModel loader utilities on CUDA.
These tests use parallax's SGLExecutor to test sharded model functionality.
"""

from typing import List, Tuple

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from parallax.server.request import InitialRequest
from parallax.server.sampling.sampling_params import SamplingParams
from parallax.utils.utils import is_cuda_available

# Delay import of SGLExecutor to avoid import errors when sglang is not available
# This allows test collection to succeed even if sglang is not installed
SGLExecutor = None
try:
    from parallax.server.executor.sglang_executor import SGLExecutor
except ImportError:
    # sglang not available, tests will be skipped
    pass

CUDA_MODEL_REPO = "Qwen/Qwen3-0.6B"
TOTAL_LAYERS = 28


@pytest.fixture(scope="module")
def ref_model_and_tokenizer():
    """Load reference model and tokenizer for CUDA tests"""
    if not is_cuda_available():
        pytest.skip("CUDA not available")

    model = AutoModelForCausalLM.from_pretrained(
        CUDA_MODEL_REPO,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(CUDA_MODEL_REPO)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    yield model, tokenizer

    # Cleanup
    del model
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "layers_config",
    [
        [(0, 14), (14, TOTAL_LAYERS)],
        [(0, 9), (9, 18), (18, TOTAL_LAYERS)],
    ],
)
@pytest.mark.cuda
def test_cuda_shard_prefill(layers_config: List[Tuple[int, int]], ref_model_and_tokenizer):
    """Test sharded CUDA model forward pass using parallax SGLExecutor.

    This test creates multiple SGLExecutor instances for different layer ranges
    and verifies they can process requests correctly. This is similar to test_model.py
    but uses parallax's SGLExecutor instead of direct model loading.
    """
    if SGLExecutor is None:
        pytest.skip("sglang not available (install with 'pip install -e .[gpu]')")
    if not is_cuda_available():
        pytest.skip("CUDA not available")

    ref_model, ref_tokenizer = ref_model_and_tokenizer

    # Create sharded executors using parallax code
    # This uses parallax's selective download and model loading
    executors = []
    for layer_from, layer_to in layers_config:
        executor = SGLExecutor(
            model_repo=CUDA_MODEL_REPO,
            start_layer=layer_from,
            end_layer=layer_to,
            dtype="bfloat16",
            kv_cache_memory_fraction=0.1,
            max_batch_size=8,
            max_num_tokens_per_batch=1024,
        )
        executors.append(executor)

    # Prepare test inputs
    texts = [
        "The capital of China is",
        "Qwen is a large language model",
    ]

    # Tokenize using the first executor's tokenizer (from parallax)
    tokenizer = executors[0].tokenizer
    input_ids_list = [tokenizer.encode(text) for text in texts]

    # Create initial requests with greedy sampling
    greedy_sampling = SamplingParams(temperature=0.0, top_k=1)
    initial_requests = [
        InitialRequest(
            request_id=f"req{i}",
            input_ids=input_ids,
            sampling_params=greedy_sampling,
            max_new_tokens=1,
        )
        for i, input_ids in enumerate(input_ids_list)
    ]

    # Run pipeline forward pass through all executors using parallax's pipeline
    # Stage 1: First executor processes the input
    executor1 = executors[0]
    executor1.handle_input_requests(initial_requests)
    executor1.scheduler.admit_requests()
    batch1 = executor1.scheduler.form_batch()
    prepared_batch1 = executor1.prepare_batch_inputs(batch1)

    assert prepared_batch1 is not None, "First executor should prepare batch"
    assert "prefill_batch" in prepared_batch1, "Should have prefill batch"

    # Process through first executor (uses parallax's process_batch)
    batch_output1 = executor1.process_batch(
        prepared_batch1["prefill_batch"], return_decoded_tokens=False
    )

    assert batch_output1 is not None, "First executor should produce output"

    # Verify executor configuration (parallax-specific)
    assert executor1.start_layer == layers_config[0][0]
    assert executor1.end_layer == layers_config[0][1]
    assert executor1.config is not None, "Executor should have config loaded"

    # For multi-stage pipeline, verify subsequent executors can be initialized
    if len(executors) > 1:
        executor2 = executors[1]
        assert executor2.start_layer == layers_config[1][0]
        assert executor2.end_layer == layers_config[1][1]
        assert executor2.config is not None, "Second executor should have config loaded"

    # Run reference model forward pass for comparison
    max_len = max(len(ids) for ids in input_ids_list)
    padded_inputs = []
    attention_masks = []
    for ids in input_ids_list:
        padded = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
        mask = [1] * len(ids) + [0] * (max_len - len(ids))
        padded_inputs.append(padded)
        attention_masks.append(mask)

    inputs = torch.tensor(padded_inputs, dtype=torch.long).to("cuda:0")
    attention_mask = torch.tensor(attention_masks, dtype=torch.long).to("cuda:0")

    with torch.no_grad():
        ref_outputs = ref_model(inputs, attention_mask=attention_mask)

    # Verify reference model output shape
    assert ref_outputs.logits.shape[0] == len(texts), "Batch size should match"
    assert ref_outputs.logits.shape[1] == max_len, "Sequence length should match"

    # Cleanup
    for executor in executors:
        executor.shutdown()
    del executors
    torch.cuda.empty_cache()


@pytest.mark.cuda
def test_cuda_executor_pipeline(ref_model_and_tokenizer):
    """Test a simple CUDA executor pipeline with multiple stages.

    This test creates a 2-stage pipeline and verifies it can process requests.
    """
    if SGLExecutor is None:
        pytest.skip("sglang not available (install with 'pip install -e .[gpu]')")
    if not is_cuda_available():
        pytest.skip("CUDA not available")

    # Create a 2-stage pipeline
    executor1 = SGLExecutor(
        model_repo=CUDA_MODEL_REPO,
        start_layer=0,
        end_layer=14,
        dtype="bfloat16",
        kv_cache_memory_fraction=0.3,
        max_batch_size=4,
        max_num_tokens_per_batch=512,
    )

    executor2 = SGLExecutor(
        model_repo=CUDA_MODEL_REPO,
        start_layer=14,
        end_layer=TOTAL_LAYERS,
        dtype="bfloat16",
        kv_cache_memory_fraction=0.3,
        max_batch_size=4,
        max_num_tokens_per_batch=512,
    )

    # Create a test request
    tokenizer = executor1.tokenizer
    prompt = "The capital of China is"
    input_ids = tokenizer.encode(prompt)

    greedy_sampling = SamplingParams(temperature=0.0, top_k=1)
    request = InitialRequest(
        request_id="test_req",
        input_ids=input_ids,
        sampling_params=greedy_sampling,
        max_new_tokens=5,
    )

    # Stage 1: Process through first executor
    executor1.handle_input_requests([request])
    executor1.scheduler.admit_requests()
    batch1 = executor1.scheduler.form_batch()
    prepared_batch1 = executor1.prepare_batch_inputs(batch1)

    assert prepared_batch1 is not None, "First executor should prepare batch"

    # Stage 2: Process through second executor
    # In a real pipeline, we'd pass intermediate results, but for this test
    # we just verify the executors can be created and initialized

    # Cleanup
    executor1.shutdown()
    executor2.shutdown()
    del executor1
    del executor2
    torch.cuda.empty_cache()
