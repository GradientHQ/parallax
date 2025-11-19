"""
Utility functions for message serialization and deserialization.

This module contains utility functions for serializing and deserializing messages
between the P2P server and the executor.
"""

import io
from typing import Any, List, Optional

import mlx.core as mx

from parallax.p2p.proto import forward_pb2
from parallax.server.request import IntermediateRequest, Request, RequestStatus
from parallax.server.sampling.sampling_params import SamplingParams


def request_to_proto(
    requests: List[IntermediateRequest],
    device: Optional[str] = "mlx",
) -> forward_pb2.ForwardRequest:
    """
    Convert a list of IntermediateRequest objects to a ForwardRequest protobuf message.
    IntermediateRequest contains request_id, current_position, status, and hidden_states.
    """
    forward_request = forward_pb2.ForwardRequest()
    assert len(requests) > 0, "No requests to convert"
    assert all(
        request.status == requests[0].status for request in requests
    ), "All requests must have the same status"
    if requests[0].status == RequestStatus.PREFILLING:
        forward_request.forward_mode = forward_pb2.ForwardMode.EXTEND
    elif requests[0].status == RequestStatus.DECODING:
        forward_request.forward_mode = forward_pb2.ForwardMode.DECODE
    else:
        raise ValueError(f"Invalid status: {requests[0].status}")

    for request in requests:
        proto_req = forward_pb2.Req()
        proto_req.rid = request.request_id
        proto_req.output_length = request.current_position - len(request.input_ids)
        proto_req.input_ids.extend(request.input_ids)
        proto_req.routing_table.extend(request.routing_table)
        proto_req.sampling_params.CopyFrom(sampling_params_to_proto(request.sampling_params))

        if request.hidden_states is not None:
            proto_req.hidden_states = tensor_to_bytes(request.hidden_states, device=device)

        if request.next_token_id is not None:
            proto_req.next_token_id = request.next_token_id

        forward_request.reqs.append(proto_req)

    return forward_request


def proto_to_request(
    proto_request: forward_pb2.ForwardRequest,
    device: Optional[str] = "mlx",
) -> List[IntermediateRequest]:
    """
    Convert a ForwardRequest protobuf message to a IntermediateRequest object.
    """

    requests = []

    for proto_req in proto_request.reqs:
        current_position = len(proto_req.input_ids) + proto_req.output_length

        next_token_id = proto_req.next_token_id

        hidden_states = None
        if proto_req.hidden_states:
            hidden_states = bytes_to_tensor(proto_req.hidden_states, device)

        status = None
        if hidden_states is None:
            status = RequestStatus.FINISHED_EOS
        elif proto_request.forward_mode == forward_pb2.ForwardMode.EXTEND:
            status = RequestStatus.PREFILLING
        elif proto_request.forward_mode == forward_pb2.ForwardMode.DECODE:
            status = RequestStatus.DECODING
        else:
            raise ValueError(f"Invalid forward mode: {proto_request.forward_mode}")

        sampling_params = proto_to_sampling_params(proto_req.sampling_params)

        request = IntermediateRequest(
            request_id=proto_req.rid,
            current_position=current_position,
            status=status,
            input_ids=list(proto_req.input_ids),
            hidden_states=hidden_states,
            routing_table=list(proto_req.routing_table),
            next_token_id=next_token_id,
            sampling_params=sampling_params,
        )

        requests.append(request)

    return requests


def abort_request_to_proto(reqs: List[Request]) -> forward_pb2.AbortRequest:
    """Converts aborted/finished requests to a AbortRequest"""
    proto = forward_pb2.AbortRequest()
    for req in reqs:
        req_proto = forward_pb2.Req()
        req_proto.rid = req.request_id
        if req.routing_table is not None:
            req_proto.routing_table.extend(req.routing_table)
        proto.reqs.append(req_proto)
    return proto


def proto_to_abort_request(proto_request: forward_pb2.AbortRequest) -> List[IntermediateRequest]:
    """
    Converts a AbortRequest a list of IntermediateRequest objects.
    Only request_id and routing table are useful information.
    """
    status = RequestStatus.FINISHED_EOS
    requests = []
    for proto_req in proto_request.reqs:
        request = IntermediateRequest(
            request_id=proto_req.rid,
            current_position=0,
            status=status,
            routing_table=proto_req.routing_table,
        )

        requests.append(request)

    return requests


def proto_to_sampling_params(proto: forward_pb2.SamplingParams) -> SamplingParams:
    """Convert protobuf message to SamplingParams."""
    if proto is None:
        return SamplingParams()
    sampling_params = SamplingParams(
        max_new_tokens=proto.max_new_tokens,
        min_new_tokens=proto.min_new_tokens,
        temperature=proto.temperature,
        top_p=proto.top_p,
        min_p=proto.min_p,
        top_k=proto.top_k,
        stop_strs=list(proto.stop_strs),
        stop_token_ids=list(proto.stop_token_ids),
        ignore_eos=proto.ignore_eos,
        repetition_penalty=proto.repetition_penalty,
        presence_penalty=proto.presence_penalty,
        frequency_penalty=proto.frequency_penalty,
        json_schema=proto.json_schema,
    )
    return sampling_params


def sampling_params_to_proto(params: SamplingParams) -> forward_pb2.SamplingParams:
    """Convert SamplingParams to protobuf message."""
    proto = forward_pb2.SamplingParams()

    proto.max_new_tokens = params.max_new_tokens
    proto.min_new_tokens = params.min_new_tokens
    proto.temperature = params.temperature
    proto.top_p = params.top_p
    proto.min_p = params.min_p
    proto.top_k = params.top_k
    if params.stop_strs is not None:
        proto.stop_strs.extend(params.stop_strs)
    if params.stop_token_ids is not None:
        proto.stop_token_ids.extend(params.stop_token_ids)
    proto.ignore_eos = params.ignore_eos
    proto.repetition_penalty = params.repetition_penalty
    proto.presence_penalty = params.presence_penalty
    proto.frequency_penalty = params.frequency_penalty
    if params.json_schema is not None:
        proto.json_schema = params.json_schema
    return proto


def tensor_to_bytes(tensor: Any, device: Optional[str] = "mlx") -> bytes:
    """Convert tensor to protobuf Tensor using safetensor serialization."""
    if device == "cuda":
        from safetensors.torch import save

        # Convert tensor to CPU
        if tensor.device.type != "cpu":
            cpu_tensor = tensor.cpu()
        else:
            cpu_tensor = tensor
        # Store buffer using safetensor (dtype and size are automatically preserved)
        serialized_data = save({"tensor": cpu_tensor.contiguous()})
        return serialized_data
    else:
        assert tensor.size > 0, "Tensor must have size > 0"
        buffer = io.BytesIO()
        mx.save_safetensors(buffer, {"tensor": tensor})
        return buffer.getvalue()


def bytes_to_tensor(
    tensor: bytes,
    device: Optional[str] = "mlx",
) -> Any:
    """Convert bytes (safetensor format) to tensor."""
    if device == "cuda":
        from safetensors.torch import load

        tensor_dict = load(tensor)
        tensor = tensor_dict["tensor"].to(device)
    else:
        buffer = io.BytesIO(tensor)
        tensors_dict = mx.load(buffer, format="safetensors")
        tensor = tensors_dict["tensor"]
    return tensor
