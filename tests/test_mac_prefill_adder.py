from parallax.server.request import InitialRequest
from parallax.utils.mac_prefill_adder import AddReqResult, MACPrefillAdder


class FakePrefixCache:
    def __init__(self):
        self.matched_tokens = 0

    def match_prefix(self, token_ids):
        return [], min(self.matched_tokens, len(token_ids))


class FakeCacheManager:
    def __init__(self, block_size=4):
        self.block_size = block_size
        self.prefix_cache = FakePrefixCache()


def test_mac_prefill_adder_chunks_and_restores_effective_length():
    cache_manager = FakeCacheManager(block_size=4)
    req = InitialRequest(request_id="req", input_ids=list(range(10)))

    adder = MACPrefillAdder(page_size=4, rem_chunk_tokens=4, cache_manager=cache_manager)
    result = adder.add_one_req(req)

    assert result == AddReqResult.OTHER
    assert adder.new_chunked_req is req
    assert req.input_ids == [0, 1, 2, 3]
    assert req.total_length == 4
    assert req.origin_input_ids == list(range(10))

    cache_manager.prefix_cache.matched_tokens = 4
    adder = MACPrefillAdder(page_size=4, rem_chunk_tokens=4, cache_manager=cache_manager)
    still_chunked = adder.add_chunked_req(req)

    assert still_chunked is req
    assert req.input_ids == list(range(8))
    assert req.total_length == 8

    cache_manager.prefix_cache.matched_tokens = 8
    adder = MACPrefillAdder(page_size=4, rem_chunk_tokens=4, cache_manager=cache_manager)
    final_chunk = adder.add_chunked_req(req)

    assert final_chunk is None
    assert req.input_ids == list(range(10))
    assert req.total_length == 10

    req.commit_new_token(99)
    assert req.input_ids == list(range(10))
    assert req.total_length == 11


def test_mac_prefill_adder_full_cache_hit_does_not_extend_past_prompt():
    cache_manager = FakeCacheManager(block_size=4)
    cache_manager.prefix_cache.matched_tokens = 8
    req = InitialRequest(request_id="req", input_ids=list(range(8)))

    adder = MACPrefillAdder(page_size=4, rem_chunk_tokens=4, cache_manager=cache_manager)
    final_chunk = adder.add_chunked_req(req)

    assert final_chunk is None
    assert req.input_ids == list(range(8))
    assert req.total_length == 8
