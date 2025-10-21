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
    """Transforms Parallax SamplingParams to vLLM SamplingParams format."""
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


def form_vllm_batch_prefill(batched_requests: List[Request], pad_token_id: int) -> Dict[str, Any] | None:
    """Builds the vLLM prefill batch inputs for the first peer.

    Returns a dict with:
      - input_ids: List[List[int]] padded to max prompt length with pad_token_id.
    """
    batch_size = len(batched_requests)
    if batch_size == 0:
        return None

    # Collect prompts and compute max length
    seqs: List[List[int]] = []
    max_len = 0
    for req in batched_requests:
        assert req.is_prefill, f"Request {req.request_id} is not a prefill request."
        assert req.input_ids is not None and len(req.input_ids) > 0, (
            f"Request {req.request_id} has empty input_ids for prefill"
        )
        seqs.append(req.input_ids)
        if len(req.input_ids) > max_len:
            max_len = len(req.input_ids)

    # Right-pad to max_len with pad_token_id
    padded: List[List[int]] = [seq + [pad_token_id] * (max_len - len(seq)) for seq in seqs]

    return {"input_ids": padded}


def form_vllm_batch_decode(batched_requests: List[Request], is_first_peer: bool) -> Dict[str, Any] | None:
    """Builds the vLLM decode batch inputs for the first peer.

    For decode, the first peer feeds the last generated token per request.
    Other peers return None (they use pp_proxy_tensors path).
    """
    if not is_first_peer:
        return None

    # For first peer, gather the next-step input token ids (last output token)
    tokens: List[int] = []
    for req in batched_requests:
        assert req.is_decoding, f"Request {req.request_id} is not a decode request."
        assert req.output_ids is not None and len(req.output_ids) > 0, (
            f"Decode step requires at least one output token for {req.request_id}"
        )
        tokens.append(req.output_ids[-1])

    # Use shape [batch, 1] for consistency
    return {"input_ids": [[tok] for tok in tokens]}
