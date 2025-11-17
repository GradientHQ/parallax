from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from vllm.sampling_params import SamplingParams as VLLMSamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.request import Request as VLLMRequest

from parallax.server.request import Request
from parallax.server.sampling.sampling_params import (
    SamplingParams as ParallaxSamplingParams,
)
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_expected_intermediate_tokens(scheduler_output: Any, model_runner: Any) -> Optional[int]:
    """
    Estimate the padded token count expected by vLLM for this batch.

    This function computes the total number of tokens including padding that vLLM
    expects for data parallel processing.

    Args:
        scheduler_output: SchedulerOutput from vLLM scheduler
        model_runner: The vLLM model runner instance

    Returns:
        Expected total token count including padding, or None if unable to compute
    """
    if scheduler_output is None:
        return None

    total_tokens = getattr(scheduler_output, "total_num_scheduled_tokens", None)
    if total_tokens is None:
        return None

    try:
        total_tokens = int(total_tokens)
    except (TypeError, ValueError):
        return None

    if model_runner is None:
        return None

    get_num_input_tokens = getattr(model_runner, "_get_num_input_tokens", None)
    get_dp_padding = getattr(model_runner, "get_dp_padding", None)
    if get_num_input_tokens is None or get_dp_padding is None:
        return None

    num_input_tokens = get_num_input_tokens(total_tokens)
    num_pad, _ = get_dp_padding(num_input_tokens)
    return num_input_tokens + num_pad


def pad_or_trim_tensor(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Pad or trim a tensor to the target length along dimension 0.

    Args:
        tensor: Input tensor to pad/trim
        target_len: Target length for dimension 0. If negative, returns unchanged.

    Returns:
        Tensor with dimension 0 adjusted to target_len
    """
    if target_len < 0:
        return tensor
    current_len = tensor.shape[0]
    if current_len == target_len:
        return tensor
    if current_len > target_len:
        return tensor[:target_len]
    pad_shape = (target_len - current_len,) + tensor.shape[1:]
    pad = tensor.new_zeros(pad_shape)
    return torch.cat((tensor, pad), dim=0)


def resize_intermediate_tensors(
    intermediate_tensors: IntermediateTensors, target_len: Optional[int]
) -> IntermediateTensors:
    """
    Resize all tensors in IntermediateTensors to match the target length.

    This is needed for vLLM pipeline parallelism when the actual token count
    doesn't match the expected padded count for data parallel processing.

    Args:
        intermediate_tensors: vLLM IntermediateTensors containing hidden states
        target_len: Target token count. If None or negative, returns unchanged.

    Returns:
        IntermediateTensors with all tensors resized to target_len
    """
    if intermediate_tensors is None or target_len is None:
        return intermediate_tensors
    if target_len < 0:
        return intermediate_tensors

    # Create a list to avoid "dictionary changed size during iteration".
    for key, tensor in list(intermediate_tensors.items()):
        intermediate_tensors[key] = pad_or_trim_tensor(tensor, target_len)
    return intermediate_tensors


def transform_sampling_params_to_vllm(old_params: ParallaxSamplingParams) -> VLLMSamplingParams:
    structured = (
        StructuredOutputsParams(json=old_params.json_schema)
        if getattr(old_params, "json_schema", None) is not None
        else None
    )
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


def _build_vllm_request(
    req: Request,
    sampling_params: VLLMSamplingParams,
    model_runner: Any,
    *,
    include_outputs: bool,
) -> VLLMRequest:
    block_hasher = getattr(model_runner, "request_block_hasher", None)
    vllm_req = VLLMRequest(
        request_id=req.request_id,
        prompt_token_ids=getattr(req, "input_ids", None),
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=getattr(req, "eos_token_id", None),
        arrival_time=getattr(req, "arrival_time", 0.0),
        block_hasher=block_hasher,
    )
    if include_outputs:
        output_ids = getattr(req, "output_ids", None) or []
        if output_ids:
            vllm_req.append_output_token_ids(output_ids)
    return vllm_req


def form_vllm_batch_prefill(
    batched_requests: List[Request],
    model_runner: Any = None,
) -> Optional[SchedulerOutput]:
    if not batched_requests:
        return None

    if not hasattr(model_runner, "kv_cache_manager"):
        raise RuntimeError(
            "model_runner must have kv_cache_manager initialized. "
            "Call model_runner.initialize_kv_cache_manager() first."
        )

    kv_cache_manager = model_runner.kv_cache_manager

    num_common_prefix_blocks = [0] * len(model_runner.kv_cache_config.kv_cache_groups)

    created_vllm_requests: List[VLLMRequest] = []

    new_request_data_list = []
    num_scheduled_tokens: Dict[str, int] = {}
    total_tokens = 0

    for req in batched_requests:
        sampling_params = transform_sampling_params_to_vllm(req.sampling_params)

        vllm_req = _build_vllm_request(req, sampling_params, model_runner, include_outputs=False)
        created_vllm_requests.append(vllm_req)

        computed_blocks, num_computed_tokens = kv_cache_manager.get_computed_blocks(vllm_req)

        prompt_token_ids = getattr(req, "input_ids", None) or []
        num_new_tokens = max(len(prompt_token_ids) - num_computed_tokens, 0)
        if num_new_tokens > 0:
            new_blocks = kv_cache_manager.allocate_slots(
                request=vllm_req,
                num_new_tokens=num_new_tokens,
                num_new_computed_tokens=num_computed_tokens,
                new_computed_blocks=computed_blocks if num_computed_tokens > 0 else None,
            )

            if new_blocks is None:
                logger.warning(f"Cannot allocate KV cache for request {req.request_id}")
                for prev_req in created_vllm_requests[:-1]:
                    kv_cache_manager.free(prev_req)
                return None

            all_blocks = computed_blocks + new_blocks if num_computed_tokens > 0 else new_blocks
        else:
            all_blocks = computed_blocks

        block_ids = all_blocks.get_block_ids()

        new_req_data = NewRequestData(
            req_id=req.request_id,
            prompt_token_ids=req.input_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=block_ids,
            num_computed_tokens=num_computed_tokens,
            lora_request=None,
            prompt_embeds=None,
        )
        new_request_data_list.append(new_req_data)

        scheduled_tokens = len(prompt_token_ids)
        num_scheduled_tokens[req.request_id] = scheduled_tokens
        total_tokens += scheduled_tokens

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=new_request_data_list,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=num_common_prefix_blocks,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        structured_output_request_ids=[],
        grammar_bitmask=None,
        kv_connector_metadata=None,
    )

    return scheduler_output


def form_vllm_batch_decode(
    batched_requests: List[Request],
    model_runner: Any = None,
    scheduler: Any = None,
    **kwargs,
) -> Optional[SchedulerOutput]:
    if not batched_requests:
        return None

    if not hasattr(model_runner, "kv_cache_manager"):
        raise RuntimeError(
            "model_runner must have kv_cache_manager initialized. "
            "Call model_runner.initialize_kv_cache_manager() first."
        )

    kv_cache_manager = model_runner.kv_cache_manager

    req_ids: List[str] = []
    resumed_from_preemption: List[bool] = []
    new_token_ids: List[List[int]] = []
    resumed_req_token_ids: List[List[int] | None] = []
    new_block_ids: List[tuple[List[int], ...] | None] = []
    num_computed_tokens: List[int] = []
    num_output_tokens: List[int] = []
    num_scheduled_tokens: Dict[str, int] = {}

    for req in batched_requests:
        req_ids.append(req.request_id)
        resumed_from_preemption.append(False)

        # For GPU workers (non-first peer), IntermediateRequest doesn't have output_ids
        # We need to get it from vLLM's CachedRequestState in model_runner
        output_ids = getattr(req, "output_ids", None) or []

        # If this request doesn't have output_ids (IntermediateRequest case),
        # try to get it from model_runner's cached request state (vLLM internal state)
        if not output_ids and hasattr(model_runner, "requests"):
            cached_req_state = model_runner.requests.get(req.request_id)
            if cached_req_state is not None:
                output_ids = getattr(cached_req_state, "output_token_ids", [])
                logger.debug(
                    f"[Decode] Retrieved output_token_ids from vLLM CachedRequestState for "
                    f"{req.request_id}: len={len(output_ids)}"
                )

        # Fallback: try scheduler if available
        if not output_ids and scheduler is not None:
            running_req = scheduler.get_running_request(req.request_id)
            if running_req is not None:
                output_ids = getattr(running_req, "output_ids", None) or []
                logger.debug(
                    f"[Decode] Retrieved output_ids from scheduler for {req.request_id}: "
                    f"len={len(output_ids)}"
                )

        if output_ids:
            last_token = output_ids[-1]
            new_token_ids.append([last_token])
        else:
            new_token_ids.append([])

        resumed_req_token_ids.append([])

        sampling_params = transform_sampling_params_to_vllm(req.sampling_params)
        vllm_req = _build_vllm_request(req, sampling_params, model_runner, include_outputs=True)

        prompt_ids = getattr(req, "input_ids", None) or []
        # For decode stage, computed_token_count should be the total number of tokens
        # that have been processed (including all output tokens).
        # In pipeline parallelism, this must match what GPU worker expects.
        if output_ids:
            # All tokens (prompt + all generated outputs) have been computed
            computed_token_count = len(prompt_ids) + len(output_ids) - 1
        else:
            # First decode step: only prompt has been computed
            computed_token_count = len(prompt_ids)
        vllm_req.num_computed_tokens = computed_token_count

        # Debug logging to track state synchronization
        logger.debug(
            f"[Decode] req_id={req.request_id}, prompt_len={len(prompt_ids)}, "
            f"output_len={len(output_ids)}, computed_tokens={computed_token_count}"
        )

        new_blocks = kv_cache_manager.allocate_slots(
            request=vllm_req,
            num_new_tokens=1,
            num_new_computed_tokens=0,
        )

        if new_blocks is None:
            logger.warning(f"Cannot allocate KV cache for decode request {req.request_id}")
            return None

        new_block_ids.append(new_blocks.get_block_ids(allow_none=True))
        num_computed_tokens.append(computed_token_count)
        num_output_tokens.append(len(output_ids))
        num_scheduled_tokens[req.request_id] = 1

    cached_req_data = CachedRequestData(
        req_ids=req_ids,
        resumed_from_preemption=resumed_from_preemption,
        new_token_ids=new_token_ids,
        new_block_ids=new_block_ids,
        num_computed_tokens=num_computed_tokens,
    )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0] * getattr(kv_cache_manager, "num_kv_cache_groups", 1),
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        structured_output_request_ids=[],
        grammar_bitmask=None,
        kv_connector_metadata=None,
    )

    return scheduler_output


def release_vllm_request(model_runner: Any, request_id: str):
    if not hasattr(model_runner, "kv_cache_manager"):
        logger.warning(f"KV cache manager not found when releasing request {request_id}")
        return

    kv_cache_manager = model_runner.kv_cache_manager

    try:
        kv_cache_manager.coordinator.free(request_id)
        logger.debug(f"Released KV cache for request {request_id}")
    except Exception as e:
        logger.warning(f"Error releasing KV cache for request {request_id}: {e}")
