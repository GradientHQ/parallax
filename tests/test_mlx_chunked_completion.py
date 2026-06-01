import mlx.core as mx
import numpy as np

from parallax.server.executor.mlx_executor import MLXExecutor
from parallax.server.request import InitialRequest, IntermediateRequest, RequestStatus


class FakeCacheManager:
    def __init__(self):
        self.released = []

    def release_request(self, request_id):
        self.released.append(request_id)


class FakeScheduler:
    def __init__(self, request):
        self.request = request
        self.enqueued = []
        self.evicted = []

    def get_running_request(self, request_id):
        if request_id == self.request.request_id:
            return self.request
        return None

    def enque_request(self, request):
        request.ready_for_next_step = True
        self.enqueued.append(request)

    def evict_request(self, request_id):
        self.evicted.append(request_id)


def make_executor(request, *, is_first_peer=True, is_last_peer=False):
    executor = object.__new__(MLXExecutor)
    executor.is_first_peer = is_first_peer
    executor.is_last_peer = is_last_peer
    executor.chunked_req = request
    executor.cache_manager = FakeCacheManager()
    executor.scheduler = FakeScheduler(request)
    return executor


def test_middle_chunk_completion_requeues_locally_and_returns_downstream_payload():
    request = InitialRequest.from_prompt_ids(list(range(8)), 4, 16)
    request.status = RequestStatus.PREFILLING
    request.input_ids = request.origin_input_ids[:4]
    request._effective_total_length = 4
    request.is_chunked = True

    executor = make_executor(request, is_first_peer=True, is_last_peer=False)
    hidden_states = np.zeros((1, 4, 3), dtype=np.float32)

    downstream_reqs = MLXExecutor.prepare_next_batch_requests(
        executor,
        requests=[request],
        batch_output={"hidden_states": hidden_states, "probs": None},
        context_lengths=[4],
    )

    assert len(downstream_reqs) == 1
    assert downstream_reqs[0].request_id == request.request_id
    assert downstream_reqs[0].status == RequestStatus.PREFILLING
    assert downstream_reqs[0].input_ids == request.origin_input_ids
    assert downstream_reqs[0].current_position == 4
    assert downstream_reqs[0].hidden_states.shape == (4, 3)

    assert request.is_chunked is False
    assert executor.cache_manager.released == [request.request_id]
    assert executor.scheduler.enqueued == [request]
    assert executor.scheduler.evicted == []


def test_last_peer_middle_chunk_completion_drops_sampled_token_payload():
    request = IntermediateRequest(
        request_id="chunked-last-peer",
        current_position=4,
        status=RequestStatus.PREFILLING,
        input_ids=list(range(8)),
        hidden_states=np.zeros((4, 3), dtype=np.float32),
    )
    request.input_ids = request.origin_input_ids[:4]
    request._effective_total_length = 4
    request.is_chunked = True

    executor = make_executor(request, is_first_peer=False, is_last_peer=True)
    hidden_states = mx.array([123], dtype=mx.uint32)

    downstream_reqs = MLXExecutor.prepare_next_batch_requests(
        executor,
        requests=[request],
        batch_output={"hidden_states": hidden_states, "probs": [1.0]},
        context_lengths=[4],
    )

    assert downstream_reqs == []
    assert request.is_chunked is False
    assert executor.cache_manager.released == [request.request_id]
    assert executor.scheduler.enqueued == []
    assert executor.scheduler.evicted == [request.request_id]


def test_single_peer_middle_chunk_completion_requeues_and_drops_sampled_token():
    request = InitialRequest.from_prompt_ids(list(range(8)), 4, 16)
    request.status = RequestStatus.PREFILLING
    request.input_ids = request.origin_input_ids[:4]
    request._effective_total_length = 4
    request.is_chunked = True

    executor = make_executor(request, is_first_peer=True, is_last_peer=True)
    hidden_states = mx.array([123], dtype=mx.uint32)

    downstream_reqs = MLXExecutor.prepare_next_batch_requests(
        executor,
        requests=[request],
        batch_output={"hidden_states": hidden_states, "probs": [1.0]},
        context_lengths=[4],
    )

    assert downstream_reqs == []
    assert request.is_chunked is False
    assert executor.cache_manager.released == [request.request_id]
    assert executor.scheduler.enqueued == [request]
    assert executor.scheduler.evicted == []
