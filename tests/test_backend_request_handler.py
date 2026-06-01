import asyncio
import json

from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.server.constants import NODE_STATUS_AVAILABLE, NODE_STATUS_WAITING
from backend.server.openai_compat import encode_http_response_envelope
from backend.server.request_handler import RequestHandler


class DummySchedulerManage:
    scheduler = None

    def get_model_name(self):
        return "Qwen/Qwen3-0.6B"


class ForwardingSchedulerManage(DummySchedulerManage):
    def __init__(self, status=NODE_STATUS_AVAILABLE, routing_table=None):
        self.status = status
        self.routing_table = ["node-a"] if routing_table is None else routing_table

    def get_schedule_status(self):
        return self.status

    def get_routing_table(self, request_id, received_ts):
        return self.routing_table


class StaticStub:
    def __init__(self, chunks):
        self.chunks = chunks
        self.request = None

    def chat_completion(self, request):
        self.request = request
        return iter(self.chunks)


def test_prepare_backend_request_uses_vllm_xargs_for_parallax_metadata():
    handler = RequestHandler()
    handler.set_scheduler_manage(DummySchedulerManage())
    request_data = {
        "model": "client-model",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true}'},
        ],
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "response_format": {"type": "json_object"},
        "stream_options": {"include_usage": True},
        "max_completion_tokens": 42,
        "extra_future_field": {"kept": True},
        "rid": "old-rid",
        "routing_table": ["old-node"],
        "vllm_xargs": {
            "user_arg": 1,
            "parallax_routing_table": ["user-node"],
            "parallax_scheduler_request_id": "user-req",
        },
    }

    backend_request = handler._prepare_backend_request(
        request_data,
        "scheduler-req",
        ["node-a", "node-b"],
    )

    assert "rid" not in backend_request
    assert "routing_table" not in backend_request
    assert backend_request["request_id"] == "scheduler-req"
    assert backend_request["model"] == "Qwen/Qwen3-0.6B"
    assert backend_request["messages"] == request_data["messages"]
    assert backend_request["tools"] == request_data["tools"]
    assert backend_request["tool_choice"] == request_data["tool_choice"]
    assert backend_request["response_format"] == request_data["response_format"]
    assert backend_request["stream_options"] == request_data["stream_options"]
    assert backend_request["max_completion_tokens"] == 42
    assert backend_request["extra_future_field"] == {"kept": True}
    assert backend_request["vllm_xargs"] == {
        "user_arg": 1,
        "parallax_routing_table": ["node-a", "node-b"],
        "parallax_scheduler_request_id": "scheduler-req",
    }
    assert request_data["routing_table"] == ["old-node"]
    assert request_data["vllm_xargs"]["parallax_routing_table"] == ["user-node"]


def test_forward_request_returns_openai_error_when_scheduler_not_ready():
    handler = RequestHandler()
    handler.set_scheduler_manage(ForwardingSchedulerManage(status=NODE_STATUS_WAITING))

    response = asyncio.run(
        handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}]},
            "scheduler-req",
            1.0,
        )
    )

    payload = json.loads(response.body)
    assert response.status_code == 503
    assert payload["error"]["message"] == "Server is not ready"
    assert payload["error"]["type"] == "server_unavailable"
    assert payload["error"]["code"] == "server_not_ready"


def test_forward_request_returns_openai_error_when_pipelines_are_busy():
    handler = RequestHandler()
    handler.MAX_ROUTING_RETRY = 1
    handler.set_scheduler_manage(ForwardingSchedulerManage(routing_table=[]))

    response = asyncio.run(
        handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}]},
            "scheduler-req",
            1.0,
        )
    )

    payload = json.loads(response.body)
    assert response.status_code == 429
    assert payload["error"]["type"] == "rate_limit_error"
    assert payload["error"]["code"] == "rate_limit_exceeded"


def test_forward_request_preserves_non_stream_downstream_status_and_content_type():
    handler = RequestHandler()
    handler.set_scheduler_manage(ForwardingSchedulerManage())
    body = (
        b'{"error":{"message":"bad request","type":"invalid_request_error",'
        b'"param":null,"code":"bad_request"}}'
    )
    handler.stubs["node-a"] = StaticStub(
        [
            encode_http_response_envelope(
                status_code=400,
                content_type="application/json; charset=utf-8",
                body=body,
            )
        ]
    )

    response = asyncio.run(
        handler.v1_chat_completions(
            {"messages": [{"role": "user", "content": "hello"}]},
            "scheduler-req",
            1.0,
        )
    )

    assert response.status_code == 400
    assert response.body == body
    assert response.headers["content-type"] == "application/json; charset=utf-8"


def test_openai_models_returns_empty_list_without_scheduler(monkeypatch):
    monkeypatch.setattr(backend_main, "scheduler_manage", None)

    response = TestClient(backend_main.app).get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}


def test_openai_models_returns_scheduler_model(monkeypatch):
    monkeypatch.setattr(backend_main, "scheduler_manage", DummySchedulerManage())

    response = TestClient(backend_main.app).get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "Qwen/Qwen3-0.6B",
                "object": "model",
                "created": 0,
                "owned_by": "parallax",
            }
        ],
    }


def test_openai_chat_rejects_non_object_body():
    response = TestClient(backend_main.app).post("/v1/chat/completions", json=["not-object"])

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["message"] == "Request body must be a JSON object"
    assert payload["error"]["type"] == "invalid_request_error"
