from parallax.server.request import InitialRequest, Request, RequestStatus
from parallax.server.scheduler import Scheduler


class FakeKVCacheManager:
    def __init__(self, allow: bool = True):
        self.allow = allow
        self._reqs = set()

    def has_request(self, request_id: str) -> bool:
        return request_id in self._reqs

    def add_request(self, request: Request, num_tokens: int = 0) -> bool:
        if not self.allow:
            return False
        self._reqs.add(request.request_id)
        return True


def make_prefill(rid: str, prompt_len: int) -> InitialRequest:
    return InitialRequest(request_id=rid, input_ids=[0] * prompt_len)


def make_decode(rid: str, ready: bool = True) -> Request:
    r = Request(request_id=rid, status=RequestStatus.DECODING)
    r.ready_for_next_step = ready
    return r


def test_prefill_fifo_and_micro_batch():
    sched = Scheduler(max_batch_size=8, max_num_tokens_per_batch=10_000, micro_batch_ratio=1)
    # micro_batch_size = max_batch_size // ratio = 8
    # Enqueue 3 prefills in order
    r1 = make_prefill("r1", 5)
    r2 = make_prefill("r2", 6)
    r3 = make_prefill("r3", 7)
    sched.enque_request(r1)
    sched.enque_request(r2)
    sched.enque_request(r3)

    batch = sched.form_batch()
    ids = [r.request_id for r in batch]
    assert ids[:3] == ["r1", "r2", "r3"]


def test_decode_ready_order_and_prefill_first():
    # micro_batch_size = 3
    sched = Scheduler(max_batch_size=3, max_num_tokens_per_batch=10_000, micro_batch_ratio=1)

    # Two decodes already running
    d1 = make_decode("d1")
    d2 = make_decode("d2")
    sched._running_requests[d1.request_id] = d1
    sched._running_requests[d2.request_id] = d2

    # One prefill in queue
    p1 = make_prefill("p1", 8)
    sched.enque_request(p1)

    # Mark d1 ready first, then d2
    sched.enque_request(d1)  # sets ready_for_next_step + LRU move_to_end
    sched.enque_request(d2)

    sched.admit_requests()
    batch = sched.form_batch()
    ids = [r.request_id for r in batch]

    # Prefill first, then decodes in the order they became ready
    assert ids == ["p1", "d1", "d2"]


def test_token_budget_prefill_skipped_decode_taken():
    # Token budget too small for prefill, but enough for decodes (cost=1)
    sched = Scheduler(max_batch_size=2, max_num_tokens_per_batch=1, micro_batch_ratio=1)

    # One large prefill
    p_big = make_prefill("p_big", 5)
    sched.enque_request(p_big)

    # One ready decode already running
    d = make_decode("d")
    sched._running_requests[d.request_id] = d
    sched.enque_request(d)

    batch = sched.form_batch()
    ids = [r.request_id for r in batch]
    assert ids == ["d"]
    # ready flag should be reset after batching
    assert getattr(d, "ready_for_next_step", False) is False


def test_kv_cache_admission_guard_blocks_prefill():
    # A KV manager that rejects additions
    kv_mgr = FakeKVCacheManager(allow=False)
    sched = Scheduler(
        max_batch_size=2,
        max_num_tokens_per_batch=100,
        micro_batch_ratio=1,
        kv_cache_manager=kv_mgr,
    )
    p = make_prefill("p", 4)
    sched.enque_request(p)

    # Admission should fail and running set remains empty; batch should be empty
    batch = sched.form_batch()
    assert len(batch) == 0
    assert sched.num_running_requests == 0
