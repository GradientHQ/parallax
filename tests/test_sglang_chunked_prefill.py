import pytest

pytest.importorskip("sglang")

import torch

from parallax.server.executor.sglang_executor import SGLExecutor


class FakeForwardMode:
    def is_extend(self):
        return True


class FakeSGLReq:
    def __init__(self, rid):
        self.rid = rid


class FakeScheduleBatch:
    def __init__(self, reqs, chunked_req=None):
        self.reqs = list(reqs)
        self.chunked_req = chunked_req
        self.tree_cache = object()
        self.forward_mode = FakeForwardMode()
        self.filtered_with = None

    def is_empty(self):
        return len(self.reqs) == 0

    def filter_batch(self, chunked_req_to_exclude=None):
        self.filtered_with = chunked_req_to_exclude
        self.reqs = [req for req in self.reqs if req is not chunked_req_to_exclude]


class FakeRunningBatch:
    def __init__(self):
        self.reqs = []
        self.merged = []

    def is_empty(self):
        return len(self.reqs) == 0

    def merge_batch(self, batch):
        self.merged.append(batch)
        self.reqs.extend(batch.reqs)


class FakeModelRunner:
    def forward(self, forward_batch, pp_proxy_tensors=None):
        logits_output = type(
            "LogitsOutput",
            (),
            {
                "tensors": {
                    "hidden_states": torch.zeros((1, 1)),
                    "residual": torch.zeros((1, 1)),
                }
            },
        )()
        return type("ForwardOutput", (), {"logits_output": logits_output})()


def make_executor(cur_batch, running_batch):
    executor = object.__new__(SGLExecutor)
    executor.cur_batch = cur_batch
    executor.running_batch = running_batch
    executor.sgl_chunked_req = None
    executor.model_runner = FakeModelRunner()
    return executor


def test_middle_chunk_is_stashed_and_not_merged(monkeypatch):
    sgl_req = FakeSGLReq("chunked")
    cur_batch = FakeScheduleBatch([sgl_req], chunked_req=sgl_req)
    running_batch = FakeRunningBatch()
    executor = make_executor(cur_batch, running_batch)
    stashed = []

    def fake_stash(req, tree_cache, chunked=False):
        stashed.append((req, tree_cache, chunked))

    monkeypatch.setattr(
        "parallax.server.executor.sglang_executor.maybe_cache_unfinished_req",
        fake_stash,
    )

    output = SGLExecutor.process_batch(
        executor,
        {"forward_batch": object(), "pp_proxy_tensors": None, "requests": []},
        return_decoded_tokens=False,
    )

    assert output["hidden_states"].shape == (1, 1)
    assert stashed == [(sgl_req, cur_batch.tree_cache, True)]
    assert cur_batch.filtered_with is sgl_req
    assert running_batch.reqs == []
    assert executor.sgl_chunked_req is sgl_req
    assert executor.cur_batch is None


def test_final_prefill_chunk_is_merged_into_running_batch(monkeypatch):
    sgl_req = FakeSGLReq("final")
    cur_batch = FakeScheduleBatch([sgl_req], chunked_req=None)
    running_batch = FakeRunningBatch()
    executor = make_executor(cur_batch, running_batch)

    monkeypatch.setattr(
        "parallax.server.executor.sglang_executor.maybe_cache_unfinished_req",
        lambda *args, **kwargs: pytest.fail("final chunk should not be stashed"),
    )

    SGLExecutor.process_batch(
        executor,
        {"forward_batch": object(), "pp_proxy_tensors": None, "requests": []},
        return_decoded_tokens=False,
    )

    assert executor.running_batch is cur_batch
    assert executor.sgl_chunked_req is None
    assert executor.cur_batch is None
