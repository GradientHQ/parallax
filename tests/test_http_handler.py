import asyncio
import json
from http import HTTPStatus

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch might be unavailable in CI
    import importlib.machinery
    import sys
    import types

    torch_stub = types.ModuleType("torch")
    torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)

    class _DeviceStatus:
        @staticmethod
        def is_available():
            return False

    torch_stub.cuda = _DeviceStatus()
    torch_stub.mps = _DeviceStatus()
    torch_stub.float16 = "float16"
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.float32 = "float32"
    sys.modules.setdefault("torch", torch_stub)

from parallax.server.http_server import HTTPHandler, HTTPRequestInfo


def _decode_sse_json(chunk: bytes):
    line = chunk.decode().strip()
    assert line.startswith("data: ")
    return json.loads(line[len("data: ") :])


def test_qwen3_6_stream_first_chunk_includes_think_marker():
    handler = HTTPHandler.__new__(HTTPHandler)
    handler.model_path_str = "mlx-community/Qwen3.6-27B-mxfp4"
    handler.processing_requests = {}

    rid = "req-qwen36-stream"
    handler.processing_requests[rid] = HTTPRequestInfo(
        id=rid,
        stream=True,
        model="test-model",
    )

    payload = _decode_sse_json(handler._generate_stream_chunk(rid, None, is_first=True))

    assert payload["choices"][0]["delta"]["role"] == "assistant"
    assert payload["choices"][0]["delta"]["content"] == "<think>"


def test_qwen3_6_stream_first_chunk_respects_disable_thinking():
    handler = HTTPHandler.__new__(HTTPHandler)
    handler.model_path_str = "mlx-community/Qwen3.6-27B-mxfp4"
    handler.processing_requests = {}

    rid = "req-qwen36-no-thinking-stream"
    handler.processing_requests[rid] = HTTPRequestInfo(
        id=rid,
        stream=True,
        model="test-model",
        enable_thinking=False,
    )

    payload = _decode_sse_json(handler._generate_stream_chunk(rid, None, is_first=True))

    assert payload["choices"][0]["delta"]["role"] == "assistant"
    assert payload["choices"][0]["delta"]["content"] == ""


def test_http_handler_thinking_enabled_uses_extra_body_chat_template_kwargs():
    assert (
        HTTPHandler._is_thinking_enabled(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": True},
                "sampling_params": {"top_k": 3},
            }
        )
        is True
    )
    assert (
        HTTPHandler._is_thinking_enabled(
            {
                "chat_template_kwargs": {"enable_thinking": True},
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
        )
        is False
    )


def test_qwen3_6_non_stream_response_includes_think_marker_when_enabled():
    handler = HTTPHandler.__new__(HTTPHandler)
    handler.model_path_str = "mlx-community/Qwen3.6-27B-mxfp4"
    handler.processing_requests = {}

    rid = "req-qwen36-non-stream"
    request_info = HTTPRequestInfo(
        id=rid,
        stream=False,
        model="test-model",
        enable_thinking=True,
    )
    request_info.text = "reasoning"
    handler.processing_requests[rid] = request_info

    payload = handler.generate_non_stream_response(rid)

    assert payload["choices"][0]["message"]["content"] == "<think>reasoning"


def test_qwen3_6_non_stream_response_respects_disable_thinking():
    handler = HTTPHandler.__new__(HTTPHandler)
    handler.model_path_str = "mlx-community/Qwen3.6-27B-mxfp4"
    handler.processing_requests = {}

    rid = "req-qwen36-no-thinking-non-stream"
    request_info = HTTPRequestInfo(
        id=rid,
        stream=False,
        model="test-model",
        enable_thinking=False,
    )
    request_info.text = "answer"
    handler.processing_requests[rid] = request_info

    payload = handler.generate_non_stream_response(rid)

    assert payload["choices"][0]["message"]["content"] == "answer"


def test_http_handler_marks_non_stream_error():
    async def scenario():
        handler = HTTPHandler.__new__(HTTPHandler)
        handler.processing_requests = {}

        rid = "req-non-stream"
        request_info = HTTPRequestInfo(id=rid, stream=False)
        handler.processing_requests[rid] = request_info

        await handler._handle_executor_error(
            rid,
            {
                "error": "Invalid template",
                "error_type": "TemplateError",
                "status_code": HTTPStatus.BAD_REQUEST.value,
            },
        )
        return request_info

    request_info = asyncio.run(scenario())

    assert request_info.is_finish is True
    assert request_info.finish_reason == "error"
    assert request_info.error_message == "Invalid template"
    assert request_info.error_type == "TemplateError"
    assert request_info.error_status == HTTPStatus.BAD_REQUEST


def test_http_handler_stream_error_pushes_queue_event():
    async def scenario():
        handler = HTTPHandler.__new__(HTTPHandler)
        handler.processing_requests = {}

        rid = "req-stream"
        request_info = HTTPRequestInfo(id=rid, stream=True)
        request_info.token_queue = asyncio.Queue()
        handler.processing_requests[rid] = request_info

        await handler._handle_executor_error(
            rid,
            {
                "error": "Executor failure",
                "error_type": "InternalServerError",
                "status_code": HTTPStatus.INTERNAL_SERVER_ERROR.value,
            },
        )

        error_chunk = await request_info.token_queue.get()
        sentinel = await request_info.token_queue.get()
        return request_info, error_chunk, sentinel

    request_info, error_chunk, sentinel = asyncio.run(scenario())

    assert error_chunk["type"] == "error"
    assert error_chunk["payload"]["message"] == "Executor failure"
    assert error_chunk["payload"]["type"] == "InternalServerError"
    assert error_chunk["payload"]["code"] == HTTPStatus.INTERNAL_SERVER_ERROR.value
    assert sentinel is None
