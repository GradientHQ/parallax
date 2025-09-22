"""
Test the message utility functions.
"""

import mlx.core as mx
import numpy as np
import pytest

from parallax.p2p.message_util import (
    abort_request_to_proto,
    bytes_to_tensor,
    proto_to_abort_request,
    proto_to_request,
    proto_to_sampling_params,
    request_to_proto,
    sampling_params_to_proto,
    tensor_to_bytes,
)
from parallax.p2p.proto import forward_pb2
from parallax.server.request import IntermediateRequest, Request, RequestStatus
from parallax.server.sampling.sampling_params import SamplingParams


class TestMessageUtil:
    """
    Test the message utility functions.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.request_id = "test_request_123"
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=100,
            stop_strs=["\n"],
        )

    def test_request_to_proto_prefilling(self):
        """Test converting a PREFILLING IntermediateRequest to protobuf."""
        hidden_states = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32)
        request = IntermediateRequest(
            request_id=self.request_id,
            input_ids=[1, 2, 3],
            current_position=3,
            status=RequestStatus.PREFILLING,
            hidden_states=hidden_states,
            sampling_params=self.sampling_params,
            routing_table=["layer1", "layer2"],
        )

        forward_request = request_to_proto([request])

        assert forward_request.forward_mode == forward_pb2.ForwardMode.EXTEND
        assert len(forward_request.reqs) == 1
        proto_req = forward_request.reqs[0]
        assert proto_req.rid == self.request_id
        assert list(proto_req.input_ids) == [1, 2, 3]
        assert proto_req.output_length == 0
        assert proto_req.sampling_params.temperature == pytest.approx(0.7)
        assert list(proto_req.routing_table) == ["layer1", "layer2"]

        # Verify hidden_states
        deserialized_hs = bytes_to_tensor(proto_req.hidden_states)
        np.testing.assert_array_equal(
            np.array(deserialized_hs.tolist()), np.array(hidden_states.tolist())
        )

    def test_request_to_proto_decoding(self):
        """Test converting a DECODING IntermediateRequest to protobuf."""
        # The IntermediateRequest requires hidden_states for initialization.
        # For DECODING status, it should be serialized.
        hidden_states = mx.array([[0.0]], dtype=mx.float32)
        request = IntermediateRequest(
            request_id=self.request_id,
            input_ids=[],
            current_position=10,
            status=RequestStatus.DECODING,
            hidden_states=hidden_states,
            next_token_id=42,
            sampling_params=self.sampling_params,
        )

        forward_request = request_to_proto([request])

        assert forward_request.forward_mode == forward_pb2.ForwardMode.DECODE
        assert len(forward_request.reqs) == 1
        proto_req = forward_request.reqs[0]
        assert proto_req.rid == self.request_id
        assert proto_req.next_token_id == 42
        assert proto_req.hidden_states  # hidden_states should be serialized

    def test_proto_to_request_conversion(self):
        """Test the round-trip conversion from request to proto and back."""
        hidden_states = mx.array([[1.0, 2.0]], dtype=mx.bfloat16)
        original_request = IntermediateRequest(
            request_id=self.request_id,
            input_ids=[10, 20],
            current_position=12,
            status=RequestStatus.PREFILLING,
            hidden_states=hidden_states,
            next_token_id=50,
            sampling_params=self.sampling_params,
            routing_table=["nodeA"],
        )

        proto_request = request_to_proto([original_request])
        converted_requests = proto_to_request(proto_request)

        assert len(converted_requests) == 1
        converted_request = converted_requests[0]

        assert converted_request.request_id == original_request.request_id
        assert converted_request.status == original_request.status
        assert converted_request.input_ids == original_request.input_ids
        # current_position = output_length + len(input_ids)
        assert converted_request.current_position == original_request.current_position
        assert converted_request.next_token_id == original_request.next_token_id

        # Compare sampling params field by field since __eq__ is not implemented
        assert converted_request.sampling_params.temperature == pytest.approx(
            original_request.sampling_params.temperature
        )
        assert converted_request.sampling_params.top_p == pytest.approx(
            original_request.sampling_params.top_p
        )
        assert (
            converted_request.sampling_params.max_new_tokens
            == original_request.sampling_params.max_new_tokens
        )
        assert (
            converted_request.sampling_params.stop_strs
            == original_request.sampling_params.stop_strs
        )

        assert converted_request.routing_table == original_request.routing_table

        np.testing.assert_array_equal(
            np.array(converted_request.hidden_states.tolist()),
            np.array(original_request.hidden_states.tolist()),
        )

    def test_multiple_requests(self):
        """Test conversion of multiple requests."""
        req1 = IntermediateRequest(
            request_id="req1",
            input_ids=[1],
            current_position=1,
            status=RequestStatus.PREFILLING,
            hidden_states=mx.array([[1.0]], dtype=mx.float32),
        )
        req2 = IntermediateRequest(
            request_id="req2",
            input_ids=[2],
            current_position=1,
            status=RequestStatus.PREFILLING,
            hidden_states=mx.array([[2.0]], dtype=mx.float32),
        )

        forward_request = request_to_proto([req1, req2])
        assert len(forward_request.reqs) == 2

        # Test that hidden_states are not concatenated
        deserialized_reqs = proto_to_request(forward_request)
        assert len(deserialized_reqs) == 2
        np.testing.assert_array_equal(
            np.array(deserialized_reqs[0].hidden_states.tolist()), np.array([[1.0]])
        )
        np.testing.assert_array_equal(
            np.array(deserialized_reqs[1].hidden_states.tolist()), np.array([[2.0]])
        )

    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (mx.float32, (2, 2)),
            (mx.bfloat16, (1, 8)),
            (mx.int32, (5,)),
        ],
    )
    def test_tensor_serialization(self, dtype, shape):
        """Test tensor serialization and deserialization."""
        original_tensor = mx.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

        serialized_bytes = tensor_to_bytes(original_tensor)
        deserialized_tensor = bytes_to_tensor(serialized_bytes)

        assert deserialized_tensor.shape == original_tensor.shape
        assert deserialized_tensor.dtype == original_tensor.dtype
        np.testing.assert_array_equal(
            np.array(deserialized_tensor.tolist()),
            np.array(original_tensor.tolist()),
        )

    def test_sampling_params_conversion(self):
        """Test SamplingParams conversion to and from proto."""
        params = SamplingParams(
            max_new_tokens=200,
            min_new_tokens=10,
            temperature=0.5,
            top_p=0.8,
            min_p=0.1,
            top_k=40,
            stop_strs=["\n", "stop"],
            stop_token_ids=[0, 1],
            ignore_eos=True,
            repetition_penalty=1.2,
            presence_penalty=0.2,
            frequency_penalty=0.3,
            json_schema='{"type": "string"}',
        )

        proto = sampling_params_to_proto(params)
        converted_params = proto_to_sampling_params(proto)

        assert converted_params.max_new_tokens == params.max_new_tokens
        assert converted_params.min_new_tokens == params.min_new_tokens
        assert pytest.approx(converted_params.temperature) == params.temperature
        assert pytest.approx(converted_params.top_p) == params.top_p
        assert pytest.approx(converted_params.min_p) == params.min_p
        assert converted_params.top_k == params.top_k
        assert converted_params.stop_strs == params.stop_strs
        assert converted_params.stop_token_ids == params.stop_token_ids
        assert converted_params.ignore_eos == params.ignore_eos
        assert pytest.approx(converted_params.repetition_penalty) == params.repetition_penalty
        assert pytest.approx(converted_params.presence_penalty) == params.presence_penalty
        assert pytest.approx(converted_params.frequency_penalty) == params.frequency_penalty
        assert converted_params.json_schema == params.json_schema

    def test_abort_request(self):
        """Test abort request conversion."""
        req1 = Request(request_id="abort1", routing_table=["nodeA", "nodeB"])
        req2 = Request(request_id="abort2", routing_table=["nodeC"])

        abort_proto = abort_request_to_proto([req1, req2])
        assert len(abort_proto.reqs) == 2
        assert abort_proto.reqs[0].rid == "abort1"
        assert list(abort_proto.reqs[0].routing_table) == ["nodeA", "nodeB"]

        intermediate_reqs = proto_to_abort_request(abort_proto)
        assert len(intermediate_reqs) == 2
        assert intermediate_reqs[0].request_id == "abort1"
        assert intermediate_reqs[0].status == RequestStatus.FINISHED_EOS
        assert intermediate_reqs[0].routing_table == ["nodeA", "nodeB"]
        assert intermediate_reqs[1].request_id == "abort2"
