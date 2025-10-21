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
    - KV cache block allocations via vLLM's native KVCacheManager
    - Token scheduling information

    Args:
        batched_requests: List of Parallax requests to prefill
        model_runner: ParallaxVLLMModelRunner instance with initialized kv_cache_manager

    Returns:
        Dict containing:
            - scheduler_output: SchedulerOutput for vLLM
            - requests: Original Parallax requests
        Returns None if batched_requests is empty
    """
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
    from vllm.v1.request import Request as VLLMRequest

    if not batched_requests:
        return None

    # Get vLLM's KVCacheManager from model_runner
    if not hasattr(model_runner, "kv_cache_manager"):
        raise RuntimeError(
            "model_runner must have kv_cache_manager initialized. "
            "Call model_runner.initialize_kv_cache_manager() first."
        )

    kv_cache_manager = model_runner.kv_cache_manager

    # Build NewRequestData for each request
    new_request_data_list = []
    vllm_requests = []

    for req in batched_requests:
        sampling_params = transform_sampling_params_to_vllm(req.sampling_params)

        # Create vLLM Request object for KV cache management
        vllm_req = VLLMRequest(
            request_id=req.request_id,
            prompt_token_ids=req.input_ids,
            sampling_params=sampling_params,
            eos_token_id=getattr(req, "eos_token_id", None),
            arrival_time=getattr(req, "arrival_time", 0.0),
        )
        vllm_requests.append(vllm_req)

        # Check for prefix cache hits
        computed_blocks, num_computed_tokens = kv_cache_manager.get_computed_blocks(vllm_req)

        # Allocate KV cache blocks for the remaining tokens
        num_new_tokens = len(req.input_ids) - num_computed_tokens
        if num_new_tokens > 0:
            new_blocks = kv_cache_manager.allocate_slots(
                request=vllm_req,
                num_new_tokens=num_new_tokens,
                num_new_computed_tokens=num_computed_tokens,
                new_computed_blocks=computed_blocks if num_computed_tokens > 0 else None,
            )

            if new_blocks is None:
                # Cannot allocate blocks (OOM)
                logger.warning(f"Cannot allocate KV cache for request {req.request_id}")
                # Free any allocated blocks for previous requests in this batch
                for prev_req in vllm_requests[:-1]:
                    kv_cache_manager.free(prev_req)
                return None

            # Combine computed blocks and new blocks
            all_blocks = computed_blocks + new_blocks if num_computed_tokens > 0 else new_blocks
        else:
            all_blocks = computed_blocks

        # Get block IDs for the request
        block_ids = all_blocks.get_block_ids()

        new_req_data = NewRequestData(
            req_id=req.request_id,
            prompt_token_ids=req.input_ids,
            mm_features=[],  # Multimodal features (empty for text-only)
            sampling_params=sampling_params,
            pooling_params=None,  # For embedding models
            block_ids=block_ids,
            num_computed_tokens=num_computed_tokens,
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
    - KV cache blocks already allocated, may need to extend

    Args:
        batched_requests: List of Parallax requests in decode phase
        model_runner: ParallaxVLLMModelRunner instance with initialized kv_cache_manager

    Returns:
        Dict containing:
            - scheduler_output: SchedulerOutput for vLLM
            - requests: Original Parallax requests
        Returns None if batched_requests is empty
    """
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
    from vllm.v1.request import Request as VLLMRequest

    if not batched_requests:
        return None

    # Get vLLM's KVCacheManager
    if not hasattr(model_runner, "kv_cache_manager"):
        raise RuntimeError(
            "model_runner must have kv_cache_manager initialized. "
            "Call model_runner.initialize_kv_cache_manager() first."
        )

    kv_cache_manager = model_runner.kv_cache_manager

    req_ids = []
    vllm_requests = []

    for req in batched_requests:
        req_ids.append(req.request_id)

        # Create or retrieve vLLM Request object
        # In decode phase, request should already exist
        sampling_params = transform_sampling_params_to_vllm(req.sampling_params)
        vllm_req = VLLMRequest(
            request_id=req.request_id,
            prompt_token_ids=req.input_ids,
            sampling_params=sampling_params,
            eos_token_id=getattr(req, "eos_token_id", None),
            arrival_time=getattr(req, "arrival_time", 0.0),
        )
        vllm_req.num_computed_tokens = req.current_position - 1  # Update computed tokens
        vllm_requests.append(vllm_req)

        # Allocate slot for 1 new decode token
        # This may require allocating a new block if current block is full
        new_blocks = kv_cache_manager.allocate_slots(
            request=vllm_req,
            num_new_tokens=1,  # Decode generates 1 token at a time
            num_new_computed_tokens=0,
        )

        if new_blocks is None:
            # Cannot allocate (OOM), need to preempt or wait
            logger.warning(f"Cannot allocate KV cache for decode request {req.request_id}")
            return None

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

    Uses vLLM's native KVCacheManager to properly free allocated blocks
    and update prefix cache if enabled.

    Args:
        model_runner: ParallaxVLLMModelRunner instance with kv_cache_manager
        request_id: ID of the request to release
    """
    from vllm.v1.request import Request as VLLMRequest

    if not hasattr(model_runner, "kv_cache_manager"):
        logger.warning(f"KV cache manager not found when releasing request {request_id}")
        return

    kv_cache_manager = model_runner.kv_cache_manager

    # Create a minimal vLLM Request object for the free operation
    # Note: We need the request object, not just the ID
    # In a real scenario, we should maintain a mapping of request_id -> vLLMRequest
    # For now, we'll use the KVCacheManager's coordinator directly
    try:
        # The coordinator can free by request_id directly
        kv_cache_manager.coordinator.free(request_id)
        logger.debug(f"Released KV cache for request {request_id}")
    except Exception as e:
        logger.warning(f"Error releasing KV cache for request {request_id}: {e}")
