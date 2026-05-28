from backend.server.request_handler import RequestHandler


class DummySchedulerManage:
    scheduler = None

    def get_model_name(self):
        return "Qwen/Qwen3-0.6B"


def test_prepare_backend_request_uses_vllm_xargs_for_parallax_metadata():
    handler = RequestHandler()
    handler.set_scheduler_manage(DummySchedulerManage())
    request_data = {
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "rid": "old-rid",
        "routing_table": ["old-node"],
        "vllm_xargs": {"user_arg": 1},
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
    assert backend_request["vllm_xargs"] == {
        "user_arg": 1,
        "parallax_routing_table": ["node-a", "node-b"],
        "parallax_scheduler_request_id": "scheduler-req",
    }
    assert request_data["routing_table"] == ["old-node"]
