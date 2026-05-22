from typing import Optional

import pytest

from parallax.server.request import InitialRequest, Request, RequestStatus
from parallax.server.scheduler import Scheduler


class FakeCacheManager:
    def __init__(self, allow: bool = True):
        self.allow = allow
        self._reqs = set()

    def has_request(self, request_id: str) -> bool:
        return request_id in self._reqs

    def allocate_request(
        self, request_id: str, prompt_len: int, token_ids: Optional[list[int]] = None
    ) -> tuple[bool, int]:
        """PagedKV interface."""
        if not self.allow:
            return False, 0
        self._reqs.add(request_id)
        return True, 0


class FakeTokenizer:
    def __init__(self, eos_token_id=None):
        self.eos_token_id = eos_token_id

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]


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
    cache_mgr = FakeCacheManager(allow=False)
    sched = Scheduler(
        max_batch_size=2,
        max_num_tokens_per_batch=100,
        micro_batch_ratio=1,
        cache_manager=cache_mgr,
    )
    p = make_prefill("p", 4)
    sched.enque_request(p)

    # Admission should fail and running set remains empty; batch should be empty
    batch = sched.form_batch()
    assert len(batch) == 0
    assert sched.num_running_requests == 0


def test_request_status_uses_tokenizer_eos_when_config_eos_missing():
    sched = Scheduler(
        max_batch_size=2,
        is_first_peer=True,
        tokenizer=FakeTokenizer(eos_token_id=200020),
        eos_token_id=None,
    )
    req = InitialRequest(
        request_id="minimax",
        input_ids=[1],
        output_ids=[200020],
        status=RequestStatus.DECODING,
    )

    assert sched.check_and_update_request_status(req) is True
    assert req.status == RequestStatus.FINISHED_EOS


def test_request_status_accepts_zero_eos_token_id():
    sched = Scheduler(max_batch_size=2, is_first_peer=True, eos_token_id=0)
    req = InitialRequest(
        request_id="zero-eos",
        input_ids=[1],
        output_ids=[0],
        status=RequestStatus.DECODING,
    )

    assert sched.check_and_update_request_status(req) is True
    assert req.status == RequestStatus.FINISHED_EOS


def test_request_status_requires_eos_for_first_peer_status_checks():
    sched = Scheduler(max_batch_size=2, is_first_peer=True, eos_token_id=None)
    req = InitialRequest(
        request_id="no-eos",
        input_ids=[1],
        output_ids=[123],
        status=RequestStatus.DECODING,
        max_new_tokens=5,
        max_total_length=10,
    )

    with pytest.raises(AssertionError, match="EOS token ID must be set"):
        sched.check_and_update_request_status(req)
    assert req.status == RequestStatus.DECODING
