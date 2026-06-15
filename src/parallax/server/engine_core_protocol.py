"""vLLM engine-core wire protocol helpers.

The Rust frontend talks to engines over the vLLM engine-core ZMQ boundary:

* frontend input socket: ROUTER
* engine input socket: DEALER
* frontend output socket: PULL
* engine output socket: PUSH

Payloads are MessagePack-encoded DTOs. The top-level request/output structs are
array-like; nested sampling params are map-like.
"""

from __future__ import annotations

import time
from enum import IntEnum
from typing import Any, Dict, Iterable, List, Mapping, Optional

import msgpack

from parallax.server.request import InitialRequest
from parallax.server.sampling.sampling_params import SamplingParams

ENGINE_IDENTITY = (0).to_bytes(2, "little")

REQUEST_TYPE_ADD = b"\x00"
REQUEST_TYPE_ABORT = b"\x01"
REQUEST_TYPE_START_DP_WAVE = b"\x02"
REQUEST_TYPE_UTILITY = b"\x03"


class EngineCoreFinishReason(IntEnum):
    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4


class UnsupportedEngineCoreField(ValueError):
    """Raised when a vLLM engine-core request uses unsupported features."""


ENGINE_CORE_REQUEST_FIELDS = [
    "request_id",
    "prompt_token_ids",
    "mm_features",
    "sampling_params",
    "pooling_params",
    "arrival_time",
    "lora_request",
    "cache_salt",
    "data_parallel_rank",
    "prompt_embeds",
    "prompt_is_token_ids",
    "client_index",
    "current_wave",
    "priority",
    "trace_headers",
    "resumable",
    "external_req_id",
    "reasoning_ended",
    "reasoning_parser_kwargs",
    "abort_immediately",
]

ENGINE_CORE_REQUEST_DEFAULTS = {
    "prompt_embeds": None,
    "prompt_is_token_ids": None,
    "client_index": 0,
    "current_wave": 0,
    "priority": 0,
    "trace_headers": None,
    "resumable": False,
    "external_req_id": None,
    "reasoning_ended": None,
    "reasoning_parser_kwargs": None,
    "abort_immediately": False,
}

ENGINE_CORE_OUTPUT_FIELD_COUNT = 13
ENGINE_CORE_OUTPUTS_FIELD_COUNT = 8
PARALLAX_ROUTING_TABLE_EXTRA_ARG = "parallax_routing_table"
PARALLAX_SCHEDULER_REQUEST_ID_EXTRA_ARG = "parallax_scheduler_request_id"
PARALLAX_ENGINE_CORE_VERSION = "parallax"


def _unpack_msgpack(payload: bytes) -> Any:
    return msgpack.unpackb(payload, raw=False, strict_map_key=False)


def _pack_msgpack(value: Any) -> bytes:
    return msgpack.packb(value, use_bin_type=True)


def _normalize_mapping(value: Mapping[Any, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        result[str(key)] = item
    return result


def _array_to_dict(value: Any, fields: List[str], defaults: Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        result = _normalize_mapping(value)
    elif isinstance(value, (list, tuple)):
        if len(value) > len(fields):
            raise ValueError(f"Expected at most {len(fields)} fields, got {len(value)}")
        result = {field: value[index] for index, field in enumerate(fields[: len(value)])}
    else:
        raise TypeError(f"Expected MessagePack array or map, got {type(value)!r}")

    for field in fields:
        result.setdefault(field, defaults.get(field))
    return result


def decode_engine_core_request(payload: bytes | Any) -> Dict[str, Any]:
    """Decode an EngineCoreRequest payload into a field dictionary."""
    value = _unpack_msgpack(payload) if isinstance(payload, bytes) else payload
    return _array_to_dict(value, ENGINE_CORE_REQUEST_FIELDS, ENGINE_CORE_REQUEST_DEFAULTS)


def decode_engine_core_abort(payload: bytes | Any) -> List[str]:
    """Decode an Abort payload into request IDs."""
    value = _unpack_msgpack(payload) if isinstance(payload, bytes) else payload
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Expected abort request id list, got {type(value)!r}")
    return [str(request_id) for request_id in value]


def decode_engine_core_frame(request_type: bytes, payload: bytes) -> tuple[str, Any]:
    """Decode one request frame received from the Rust frontend."""
    if request_type == REQUEST_TYPE_ADD:
        return "add", decode_engine_core_request(payload)
    if request_type == REQUEST_TYPE_ABORT:
        return "abort", decode_engine_core_abort(payload)
    if request_type == REQUEST_TYPE_UTILITY:
        raise UnsupportedEngineCoreField("EngineCoreRequest Utility calls are not supported")
    if request_type == REQUEST_TYPE_START_DP_WAVE:
        raise UnsupportedEngineCoreField("EngineCoreRequest StartDpWave is not supported")
    raise ValueError(f"Unknown engine-core request type frame: {request_type!r}")


def encode_engine_core_request(request: Mapping[str, Any] | Iterable[Any]) -> bytes:
    """Encode an EngineCoreRequest-shaped value. Intended for tests."""
    return _pack_msgpack(request)


def encode_engine_core_abort(request_ids: Iterable[str]) -> bytes:
    """Encode an Abort payload. Intended for tests."""
    return _pack_msgpack(list(request_ids))


def engine_core_ready_payload(
    *,
    max_model_len: int,
    dtype: Optional[str],
    num_gpu_blocks: int = 0,
    dp_stats_address: Optional[str] = None,
) -> bytes:
    """Build the registration payload sent by the engine DEALER socket."""
    return _pack_msgpack(
        {
            "max_model_len": int(max_model_len),
            "num_gpu_blocks": int(num_gpu_blocks),
            "dp_stats_address": dp_stats_address,
            "dtype": dtype,
            "vllm_version": PARALLAX_ENGINE_CORE_VERSION,
        }
    )


def _sampling_param_value(params: Mapping[str, Any], name: str, default: Any) -> Any:
    if name in params:
        return params[name]
    encoded_name = name.encode("utf-8")
    if encoded_name in params:
        return params[encoded_name]
    return default


def _validate_supported_sampling_params(params: Mapping[str, Any]) -> None:
    unsupported_if_present = [
        "logprobs",
        "prompt_logprobs",
        "logit_bias",
        "allowed_token_ids",
        "_bad_words_token_ids",
        "bad_words_token_ids",
        "structured_outputs",
        "logprob_token_ids",
        "skip_reading_prefix_cache",
    ]
    for field in unsupported_if_present:
        value = _sampling_param_value(params, field, None)
        if value is not None:
            raise UnsupportedEngineCoreField(f"EngineCoreSamplingParams.{field} is not supported")

    extra_args = _sampling_param_value(params, "extra_args", None)
    if extra_args is None:
        return
    if not isinstance(extra_args, Mapping):
        raise UnsupportedEngineCoreField("EngineCoreSamplingParams.extra_args must be a map")

    allowed_extra_args = {
        PARALLAX_ROUTING_TABLE_EXTRA_ARG,
        PARALLAX_SCHEDULER_REQUEST_ID_EXTRA_ARG,
    }
    normalized_extra_args = _normalize_mapping(extra_args)
    unsupported_extra_args = sorted(set(normalized_extra_args) - allowed_extra_args)
    if unsupported_extra_args:
        raise UnsupportedEngineCoreField(
            "EngineCoreSamplingParams.extra_args contains unsupported fields: "
            + ", ".join(unsupported_extra_args)
        )


def routing_table_from_engine_core_sampling_params(
    params: Optional[Mapping[str, Any]],
) -> List[str]:
    """Extract Parallax routing metadata forwarded through vllm_xargs."""
    if params is None:
        return []

    params = _normalize_mapping(params)
    extra_args = _sampling_param_value(params, "extra_args", None)
    if extra_args is None:
        return []
    if not isinstance(extra_args, Mapping):
        raise UnsupportedEngineCoreField("EngineCoreSamplingParams.extra_args must be a map")

    extra_args = _normalize_mapping(extra_args)
    routing_table = extra_args.get(PARALLAX_ROUTING_TABLE_EXTRA_ARG)
    if routing_table is None:
        return []
    if not isinstance(routing_table, (list, tuple)):
        raise UnsupportedEngineCoreField(
            f"EngineCoreSamplingParams.extra_args.{PARALLAX_ROUTING_TABLE_EXTRA_ARG} "
            "must be a list"
        )

    result = []
    for node_id in routing_table:
        if isinstance(node_id, bytes):
            node_id = node_id.decode("utf-8")
        if not isinstance(node_id, str) or not node_id:
            raise UnsupportedEngineCoreField(
                f"EngineCoreSamplingParams.extra_args.{PARALLAX_ROUTING_TABLE_EXTRA_ARG} "
                "must contain non-empty strings"
            )
        result.append(node_id)
    return result


def sampling_params_from_engine_core(params: Optional[Mapping[str, Any]]) -> SamplingParams:
    """Map vLLM EngineCoreSamplingParams into Parallax SamplingParams."""
    if params is None:
        return SamplingParams()

    params = _normalize_mapping(params)
    _validate_supported_sampling_params(params)

    eos_token_id = _sampling_param_value(params, "_eos_token_id", None)
    stop_token_ids = _sampling_param_value(params, "stop_token_ids", None) or []

    return SamplingParams(
        max_new_tokens=int(_sampling_param_value(params, "max_tokens", 128)),
        min_new_tokens=int(_sampling_param_value(params, "min_tokens", 0)),
        temperature=float(_sampling_param_value(params, "temperature", 1.0)),
        top_p=float(_sampling_param_value(params, "top_p", 1.0)),
        min_p=float(_sampling_param_value(params, "min_p", 0.0)),
        top_k=int(_sampling_param_value(params, "top_k", 0)),
        stop_token_ids=[int(token_id) for token_id in stop_token_ids],
        ignore_eos=eos_token_id is None,
        repetition_penalty=float(_sampling_param_value(params, "repetition_penalty", 1.0)),
        presence_penalty=float(_sampling_param_value(params, "presence_penalty", 0.0)),
        frequency_penalty=float(_sampling_param_value(params, "frequency_penalty", 0.0)),
    )


def _validate_supported_request(request: Mapping[str, Any]) -> None:
    unsupported_fields = [
        "mm_features",
        "pooling_params",
        "lora_request",
        "prompt_embeds",
        "reasoning_parser_kwargs",
    ]
    for field in unsupported_fields:
        if request.get(field) is not None:
            raise UnsupportedEngineCoreField(f"EngineCoreRequest.{field} is not supported")

    prompt_mask = request.get("prompt_is_token_ids")
    if prompt_mask is not None and not all(bool(item) for item in prompt_mask):
        raise UnsupportedEngineCoreField("EngineCoreRequest.prompt_embeds is not supported")

    data_parallel_rank = request.get("data_parallel_rank")
    if data_parallel_rank not in (None, 0):
        raise UnsupportedEngineCoreField("EngineCoreRequest.data_parallel_rank is not supported")

    if request.get("abort_immediately"):
        raise UnsupportedEngineCoreField("EngineCoreRequest.abort_immediately is not supported")


def engine_core_request_to_initial_request(
    request: Mapping[str, Any] | bytes,
    *,
    max_sequence_length: Optional[int] = None,
) -> InitialRequest:
    """Convert vLLM EngineCoreRequest into Parallax InitialRequest."""
    decoded = decode_engine_core_request(request)
    _validate_supported_request(decoded)

    prompt_token_ids = decoded.get("prompt_token_ids")
    if prompt_token_ids is None:
        raise UnsupportedEngineCoreField("EngineCoreRequest.prompt_token_ids is required")

    input_ids = [int(token_id) for token_id in prompt_token_ids]
    raw_sampling_params = decoded.get("sampling_params")
    routing_table = routing_table_from_engine_core_sampling_params(raw_sampling_params)
    sampling_params = sampling_params_from_engine_core(raw_sampling_params)
    max_new_tokens = max(1, int(sampling_params.max_new_tokens))
    max_total_length = len(input_ids) + max_new_tokens
    if max_sequence_length is not None:
        max_total_length = min(max_total_length, int(max_sequence_length))

    return InitialRequest(
        request_id=str(decoded["request_id"]),
        input_ids=input_ids,
        sampling_params=sampling_params,
        max_new_tokens=max_new_tokens,
        max_total_length=max_total_length,
        routing_table=routing_table,
        return_probs=False,
    )


def _finish_reason_to_wire(finish_reason: EngineCoreFinishReason | int | None) -> Optional[int]:
    if finish_reason is None:
        return None
    return int(finish_reason)


def make_engine_core_output(
    *,
    request_id: str,
    new_token_ids: Optional[List[int]] = None,
    finish_reason: EngineCoreFinishReason | int | None = None,
    stop_reason: int | str | None = None,
) -> List[Any]:
    """Build one EngineCoreOutput as a full array-like MessagePack value."""
    return [
        request_id,
        [int(token_id) for token_id in (new_token_ids or [])],
        None,
        None,
        None,
        _finish_reason_to_wire(finish_reason),
        stop_reason,
        None,
        None,
        None,
        None,
        None,
        0,
    ]


def make_engine_core_outputs(
    outputs: List[List[Any]],
    *,
    engine_index: int = 0,
    finished_requests: Optional[Iterable[str]] = None,
    timestamp: Optional[float] = None,
) -> List[Any]:
    """Build one EngineCoreOutputs as a full array-like MessagePack value."""
    return [
        int(engine_index),
        outputs,
        None,
        time.monotonic() if timestamp is None else float(timestamp),
        None,
        list(finished_requests) if finished_requests is not None else None,
        None,
        None,
    ]


def encode_engine_core_outputs(
    outputs: List[List[Any]],
    *,
    engine_index: int = 0,
    finished_requests: Optional[Iterable[str]] = None,
    timestamp: Optional[float] = None,
) -> bytes:
    """Encode an EngineCoreOutputs payload."""
    return _pack_msgpack(
        make_engine_core_outputs(
            outputs,
            engine_index=engine_index,
            finished_requests=finished_requests,
            timestamp=timestamp,
        )
    )
