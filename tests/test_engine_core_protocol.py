import msgpack
import pytest

from parallax.server.engine_core_protocol import (
    PARALLAX_ENGINE_CORE_VERSION,
    REQUEST_TYPE_ABORT,
    REQUEST_TYPE_ADD,
    EngineCoreFinishReason,
    UnsupportedEngineCoreField,
    decode_engine_core_frame,
    encode_engine_core_abort,
    encode_engine_core_outputs,
    encode_engine_core_request,
    engine_core_ready_payload,
    engine_core_request_to_initial_request,
    make_engine_core_output,
)


def _sampling_params(**overrides):
    params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 42,
        "seed": None,
        "max_tokens": 12,
        "min_tokens": 2,
        "logprobs": None,
        "prompt_logprobs": None,
        "min_p": 0.05,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
        "repetition_penalty": 1.1,
        "stop_token_ids": [7, 8],
        "_eos_token_id": 9,
        "_all_stop_token_ids": [7, 8, 9],
        "logit_bias": None,
        "allowed_token_ids": None,
        "_bad_words_token_ids": None,
        "structured_outputs": None,
        "logprob_token_ids": None,
        "skip_reading_prefix_cache": None,
        "extra_args": None,
    }
    params.update(overrides)
    return params


def _engine_request(**overrides):
    request = [
        "req-1",
        [101, 102, 103],
        None,
        _sampling_params(),
        None,
        123.4,
        None,
        None,
        None,
        None,
        None,
        0,
        0,
        0,
        None,
        False,
        None,
        None,
        None,
        False,
    ]
    for index, value in overrides.items():
        request[index] = value
    return request


def test_vllm_add_frame_decodes_engine_core_request():
    payload = encode_engine_core_request(_engine_request())

    message_type, request = decode_engine_core_frame(REQUEST_TYPE_ADD, payload)

    assert message_type == "add"
    assert request["request_id"] == "req-1"
    assert request["prompt_token_ids"] == [101, 102, 103]
    assert request["sampling_params"]["max_tokens"] == 12


def test_vllm_abort_frame_decodes_request_id_list():
    payload = encode_engine_core_abort(["req-1", "req-2"])

    message_type, request_ids = decode_engine_core_frame(REQUEST_TYPE_ABORT, payload)

    assert message_type == "abort"
    assert request_ids == ["req-1", "req-2"]


def test_engine_core_ready_payload_matches_m3_release_schema():
    payload = engine_core_ready_payload(
        max_model_len=32768,
        block_size=64,
        num_gpu_blocks=0,
        dp_stats_address=None,
        dtype="bfloat16",
    )

    decoded = msgpack.unpackb(payload, raw=False)

    assert decoded == {
        "max_model_len": 32768,
        "num_gpu_blocks": 0,
        "block_size": 64,
        "dp_stats_address": None,
        "dtype": "bfloat16",
        "vllm_version": PARALLAX_ENGINE_CORE_VERSION,
        "world_size": 1,
        "data_parallel_size": 1,
    }


def test_engine_core_request_maps_to_initial_request():
    req = engine_core_request_to_initial_request(encode_engine_core_request(_engine_request()))

    assert req.request_id == "req-1"
    assert req.input_ids == [101, 102, 103]
    assert req.max_new_tokens == 12
    assert req.max_total_length == 15
    assert req.sampling_params.temperature == 0.7
    assert req.sampling_params.top_p == 0.9
    assert req.sampling_params.top_k == 42
    assert req.sampling_params.min_p == 0.05
    assert req.sampling_params.min_new_tokens == 2
    assert req.sampling_params.stop_token_ids == {7, 8}
    assert req.sampling_params.ignore_eos is False
    assert req.sampling_params.frequency_penalty == 0.1
    assert req.sampling_params.presence_penalty == 0.2
    assert req.sampling_params.repetition_penalty == 1.1


def test_engine_core_request_extracts_parallax_routing_table():
    request = _engine_request()
    request[3] = _sampling_params(
        extra_args={
            "parallax_routing_table": ["node-a", "node-b"],
            "parallax_scheduler_request_id": "scheduler-req",
        }
    )

    req = engine_core_request_to_initial_request(encode_engine_core_request(request))

    assert req.routing_table == ["node-a", "node-b"]


def test_engine_core_request_rejects_unknown_extra_args():
    request = _engine_request()
    request[3] = _sampling_params(extra_args={"unknown_extension": True})

    with pytest.raises(UnsupportedEngineCoreField, match="unknown_extension"):
        engine_core_request_to_initial_request(encode_engine_core_request(request))


def test_engine_core_request_rejects_unsupported_fields():
    request = _engine_request()
    request[2] = [{"type": "image"}]

    with pytest.raises(UnsupportedEngineCoreField, match="mm_features"):
        engine_core_request_to_initial_request(encode_engine_core_request(request))


def test_engine_core_request_rejects_unsupported_logprobs():
    request = _engine_request()
    request[3] = _sampling_params(logprobs=5)

    with pytest.raises(UnsupportedEngineCoreField, match="logprobs"):
        engine_core_request_to_initial_request(encode_engine_core_request(request))


def test_token_output_encodes_engine_core_outputs():
    output = make_engine_core_output(request_id="req-1", new_token_ids=[42])
    payload = encode_engine_core_outputs([output], timestamp=1.25)
    decoded = msgpack.unpackb(payload, raw=False)

    assert decoded[0] == 0
    assert decoded[1][0][0] == "req-1"
    assert decoded[1][0][1] == [42]
    assert decoded[1][0][5] is None
    assert decoded[3] == 1.25
    assert decoded[5] is None


@pytest.mark.parametrize(
    ("finish_reason", "wire_value"),
    [
        (EngineCoreFinishReason.STOP, 0),
        (EngineCoreFinishReason.LENGTH, 1),
        (EngineCoreFinishReason.ABORT, 2),
    ],
)
def test_terminal_output_finish_reason_mapping(finish_reason, wire_value):
    output = make_engine_core_output(
        request_id="req-1",
        new_token_ids=[42] if finish_reason != EngineCoreFinishReason.ABORT else [],
        finish_reason=finish_reason,
        stop_reason=42 if finish_reason == EngineCoreFinishReason.STOP else None,
    )
    payload = encode_engine_core_outputs(
        [output],
        finished_requests=["req-1"],
        timestamp=2.5,
    )
    decoded = msgpack.unpackb(payload, raw=False)

    assert decoded[1][0][5] == wire_value
    assert decoded[5] == ["req-1"]
    if finish_reason == EngineCoreFinishReason.STOP:
        assert decoded[1][0][6] == 42
