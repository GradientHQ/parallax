"""
Store information about a vLLM batch.

This module provides batch formation utilities for vLLM v1 backend integration.
It transforms Parallax requests into vLLM-compatible structures for both prefill
and decode stages.

Key differences from SGLang:
- vLLM uses SchedulerOutput (flat) vs SGLang's ScheduleBatch (hierarchical)
- KV Cache is managed independently via KVCache object
- Sampling is integrated in execute_model() call
"""

from __future__ import annotations

from typing import Any, Dict, List

from parallax.server.request import Request
from parallax.server.sampling.sampling_params import (
    SamplingParams as ParallaxSamplingParams,
)
from vllm.v1.request import Request as VLLMRequest
from vllm.sampling_params import (
    SamplingParams as VLLMSamplingParams,
    StructuredOutputsParams,
)
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


def transform_sampling_params_to_vllm(old_params: ParallaxSamplingParams) -> VLLMSamplingParams:
    """Transforms Parallax SamplingParams to vLLM SamplingParams format.

    Args:
        old_params: Parallax sampling parameters

    Returns:
        vLLM SamplingParams object
    """
    # Map Parallax json_schema -> vLLM structured_outputs
    structured = (
        StructuredOutputsParams(json=old_params.json_schema)
        if getattr(old_params, "json_schema", None) is not None
        else None
    )

    # vLLM uses max_tokens/min_tokens naming
    params = VLLMSamplingParams(
        max_tokens=old_params.max_new_tokens,
        min_tokens=old_params.min_new_tokens,
        temperature=old_params.temperature,
        top_p=old_params.top_p,
        min_p=old_params.min_p,
        top_k=old_params.top_k,
        stop_token_ids=(
            list(old_params.stop_token_ids)
            if getattr(old_params, "stop_token_ids", None) is not None
            else None
        ),
        ignore_eos=old_params.ignore_eos,
        stop=old_params.stop_strs,
        repetition_penalty=old_params.repetition_penalty,
        presence_penalty=old_params.presence_penalty,
        frequency_penalty=old_params.frequency_penalty,
        structured_outputs=structured,
    )
    return params


def transform_requests_to_vllm(batched_requests: List[Request]) -> List[VLLMRequest]:
    """Transforms Parallax Request to vLLM Request format.

    Note: Only used if we later choose to feed vLLM Engine directly.
    Currently we bypass the Engine and use GPUModelRunner directly.

    Args:
        batched_requests: List of Parallax requests

    Returns:
        List of vLLM Request objects
    """
    vllm_reqs = []
    for old_req in batched_requests:
        sampling_params = transform_sampling_params_to_vllm(old_req.sampling_params)
        vllm_req = VLLMRequest(
            request_id=old_req.request_id,
            prompt_token_ids=old_req.input_ids,
            sampling_params=sampling_params,
            eos_token_id=getattr(old_req, "eos_token_id", None),
            client_index=getattr(old_req, "client_index", 0),
        )
        vllm_reqs.append(vllm_req)

    return vllm_reqs


def form_vllm_batch_prefill(
    batched_requests: List[Request],
    model_runner: Any = None,
) -> Dict[str, Any]:
    """Prepare a vLLM prefill batch.

    Constructs a SchedulerOutput for vLLM v1 GPUModelRunner that contains:
    - NewRequestData for each request (new prefill requests)
    - KV cache block allocations
    - Token scheduling information

    Args:
        batched_requests: List of Parallax requests to prefill
        model_runner: vLLM GPUModelRunner instance

    Returns:
        Dict containing:
            - scheduler_output: SchedulerOutput for vLLM
            - requests: Original Parallax requests
        Returns None if batched_requests is empty
    """
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

    if not batched_requests:
        return None

    # Initialize KV cache manager if not already done
    # This is a lightweight wrapper around vLLM's KV cache
    if not hasattr(model_runner, "_parallax_kv_cache"):
        from parallax.vllm.model_runner import VLLMKVCacheManager

        model_runner._parallax_kv_cache = VLLMKVCacheManager(
            model_runner, model_runner.kv_cache_config.block_size
        )

    kv_cache = model_runner._parallax_kv_cache

    # Build NewRequestData for each request
    new_request_data_list = []
    for req in batched_requests:
        sampling_params = transform_sampling_params_to_vllm(req.sampling_params)

        # Allocate KV cache blocks for this request
        block_ids = kv_cache.allocate(req.request_id, len(req.input_ids))

        new_req_data = NewRequestData(
            req_id=req.request_id,
            prompt_token_ids=req.input_ids,
            mm_features=[],  # Multimodal features (empty for text-only)
            sampling_params=sampling_params,
            pooling_params=None,  # For embedding models
            block_ids=block_ids,
            num_computed_tokens=0,  # Prefill starts from scratch
            lora_request=None,  # LoRA adapter
            prompt_embeds=None,  # Soft prompts
        )
        new_request_data_list.append(new_req_data)

    # Build SchedulerOutput
    # This is the main data structure that vLLM's model runner expects
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=new_request_data_list,
        scheduled_cached_reqs=CachedRequestData.make_empty(),  # No cached reqs in prefill
        num_scheduled_tokens={req.request_id: len(req.input_ids) for req in batched_requests},
        total_num_scheduled_tokens=sum(len(req.input_ids) for req in batched_requests),
        scheduled_spec_decode_tokens={},  # Speculative decoding tokens
        scheduled_encoder_inputs={},  # For encoder-decoder models
        num_common_prefix_blocks=[],  # Prefix caching
        finished_req_ids=set(),  # No finished requests in prefill
        free_encoder_mm_hashes=[],  # Encoder multimodal hash cleanup
        structured_output_request_ids=[],  # Requests using structured output
        grammar_bitmask=None,  # Grammar constraints
        kv_connector_metadata=None,  # KV connector for disaggregation
    )

    return scheduler_output, batched_requests


def form_vllm_batch_decode(
    batched_requests: List[Request],
    model_runner: Any = None,
) -> Dict[str, Any]:
    """Prepare a vLLM decode batch.

    Constructs a SchedulerOutput for vLLM v1 GPUModelRunner for decode stage.
    Key differences from prefill:
    - Uses CachedRequestData (not NewRequestData)
    - Each request processes exactly 1 token
    - KV cache blocks are already allocated

    Args:
        batched_requests: List of Parallax requests in decode phase
        model_runner: vLLM GPUModelRunner instance

    Returns:
        Dict containing:
            - scheduler_output: SchedulerOutput for vLLM
            - requests: Original Parallax requests
        Returns None if batched_requests is empty
    """
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    if not batched_requests:
        return None

    # Get KV cache manager (should already be initialized in prefill)
    kv_cache = model_runner._parallax_kv_cache

    req_ids = [req.request_id for req in batched_requests]

    # Build CachedRequestData for decode
    # These are requests that already have KV cache allocated
    cached_req_data = CachedRequestData(
        req_ids=req_ids,
        resumed_from_preemption=[False] * len(req_ids),  # Not resuming from preemption
        new_token_ids=[[] for _ in req_ids],  # Empty for non-pipeline-parallel
        resumed_req_token_ids=[None for _ in req_ids],  # Not resumed
        new_block_ids=[None for _ in req_ids],  # No new blocks needed for decode
        num_computed_tokens=[req.current_position for req in batched_requests],
        num_output_tokens=[
            len(req.output_ids) if hasattr(req, "output_ids") else 0 for req in batched_requests
        ],
    )

    # Build SchedulerOutput for decode
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],  # No new requests in decode
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={req_id: 1 for req_id in req_ids},  # 1 token per request in decode
        total_num_scheduled_tokens=len(req_ids),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        structured_output_request_ids=[],
        grammar_bitmask=None,
        kv_connector_metadata=None,
    )

    return scheduler_output


def release_vllm_request(model_runner: Any, request_id: str):
    """Release KV Cache and other resources for finished/aborted requests.

    Similar to SGLang's release_cuda_request but for vLLM backend.

    Args:
        model_runner: vLLM GPUModelRunner instance
        request_id: ID of the request to release
    """
    if not hasattr(model_runner, "_parallax_kv_cache"):
        logger.warning(f"KV cache manager not found when releasing request {request_id}")
        return

    kv_cache = model_runner._parallax_kv_cache
    kv_cache.free(request_id)
