import mlx.core as mx

from parallax.server.block_radix_cache import BlockRadixCache
from parallax.server.cache.linear_cache import LinearCache
from parallax.server.cache_manager import CacheManager


def _linear_cache(cache_manager: CacheManager) -> LinearCache:
    for cache in cache_manager.get_caches():
        if isinstance(cache, LinearCache):
            return cache
    raise AssertionError("expected a LinearCache")


def _fixed_cache_allocation(monkeypatch, num_blocks: int, num_linear_prefix_slots: int = 0):
    def calculate_cache_allocation(self, cache_memory_fraction, dtype):
        return num_blocks, num_linear_prefix_slots

    monkeypatch.setattr(CacheManager, "_calculate_cache_allocation", calculate_cache_allocation)


def test_linear_aware_radix_requires_linear_slot_for_match():
    cache = BlockRadixCache(block_size=1, has_linear_cache=True)
    node1 = cache.insert_block([1], block_id=10)
    node2 = cache.insert_block([2], block_id=11, parent_path=[node1])

    assert cache.match_prefix([1, 2, 3]) == ([], 0)

    node2.linear_slot = 4

    assert cache.match_prefix([1, 2, 3]) == ([10, 11], 2)


def test_radix_evicts_attached_linear_slot():
    freed_slots = []
    cache = BlockRadixCache(
        block_size=1,
        has_linear_cache=True,
        on_linear_slot_evict=freed_slots.append,
    )
    node1 = cache.insert_block([1], block_id=10)
    node1.linear_slot = 4

    cache.insert_block([2], block_id=11)
    assert node1.linear_slot == 4
    assert freed_slots == []

    assert cache.evict_lru_blocks(1) == 1

    assert node1.linear_slot is None
    assert freed_slots == [4]


def test_cache_manager_sizes_linear_cache_with_prefix_slots(monkeypatch):
    _fixed_cache_allocation(monkeypatch, num_blocks=8, num_linear_prefix_slots=2)

    cache_manager = CacheManager(
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
        dtype=mx.float32,
        block_size=1,
        layer_types=["attention", "linear"],
        max_num_seqs=4,
        conv_dim=2,
        conv_kernel_size=2,
        linear_k_dim=2,
        linear_v_dim=2,
        linear_num_k_heads=1,
        linear_num_v_heads=1,
        enable_prefix_cache=True,
        chunked_prefill_size=4,
    )

    assert cache_manager.num_linear_prefix_slots == 2
    assert _linear_cache(cache_manager).max_num_seqs == 6


def test_cache_manager_evicts_prefix_blocks_when_kv_allocation_is_full(monkeypatch):
    _fixed_cache_allocation(monkeypatch, num_blocks=3)

    cache_manager = CacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        dtype=mx.float32,
        block_size=1,
        layer_types=["attention"],
        enable_prefix_cache=True,
    )

    ok, matched = cache_manager.allocate_request("cached", 3, token_ids=[1, 2, 3])
    assert ok
    assert matched == 0
    cache_manager.insert_full_blocks_to_cache("cached")
    cache_manager.free_request("cached")

    assert cache_manager.allocator.get_num_free_blocks() == 0
    assert cache_manager.prefix_cache.num_cached_blocks == 3

    ok, matched = cache_manager.allocate_request("new", 1, token_ids=[9])
    assert ok
    assert matched == 0

    assert cache_manager.prefix_cache.num_cached_blocks == 2
    assert len(cache_manager.get_block_table("new")) == 1


def test_cache_manager_evicts_prefix_node_when_linear_prefix_slots_are_full(monkeypatch):
    _fixed_cache_allocation(monkeypatch, num_blocks=4, num_linear_prefix_slots=1)

    cache_manager = CacheManager(
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
        dtype=mx.float32,
        block_size=1,
        layer_types=["attention", "linear"],
        max_num_seqs=2,
        conv_dim=2,
        conv_kernel_size=2,
        linear_k_dim=2,
        linear_v_dim=2,
        linear_num_k_heads=1,
        linear_num_v_heads=1,
        enable_prefix_cache=True,
        chunked_prefill_size=4,
    )
    assert cache_manager.num_linear_prefix_slots == 1

    ok, _ = cache_manager.allocate_request("cached1", 1, token_ids=[1])
    assert ok
    cache_manager.insert_full_blocks_to_cache("cached1")
    node1 = cache_manager.prefix_cache.get_node_for_token_ids([1])
    assert node1.linear_slot is not None
    cache_manager.free_request("cached1")

    ok, _ = cache_manager.allocate_request("cached2", 1, token_ids=[2])
    assert ok
    cache_manager.insert_full_blocks_to_cache("cached2")

    assert cache_manager.prefix_cache.get_node_for_token_ids([1]) is None
    node2 = cache_manager.prefix_cache.get_node_for_token_ids([2])
    assert node2.linear_slot is not None


def test_cache_manager_uses_prompt_minus_last_for_linear_prefix_match(monkeypatch):
    _fixed_cache_allocation(monkeypatch, num_blocks=8, num_linear_prefix_slots=1)

    cache_manager = CacheManager(
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
        dtype=mx.float32,
        block_size=1,
        layer_types=["attention", "linear"],
        max_num_seqs=4,
        conv_dim=2,
        conv_kernel_size=2,
        linear_k_dim=2,
        linear_v_dim=2,
        linear_num_k_heads=1,
        linear_num_v_heads=1,
        enable_prefix_cache=True,
    )
    assert cache_manager.num_linear_prefix_slots > 0

    ok, matched = cache_manager.allocate_request("cached", 3, token_ids=[1, 2, 3])
    assert ok
    assert matched == 0

    linear_cache = _linear_cache(cache_manager)
    cached_slot = cache_manager.get_slot("cached")
    linear_cache.conv_state_cache[0, cached_slot] = mx.ones_like(
        linear_cache.conv_state_cache[0, cached_slot]
    )
    linear_cache.linear_state_cache[0, cached_slot] = (
        mx.ones_like(linear_cache.linear_state_cache[0, cached_slot]) * 2
    )
    mx.eval(linear_cache.conv_state_cache, linear_cache.linear_state_cache)

    cache_manager.insert_full_blocks_to_cache("cached")
    cached_node = cache_manager.prefix_cache.get_node_for_token_ids([1, 2, 3])
    assert cached_node.linear_slot is not None
    assert cached_node.linear_slot >= cache_manager.max_num_seqs
    cache_manager.free_request("cached")

    assert cache_manager.get_reusable_prefix_len([1, 2, 3]) == 0
    assert cache_manager.get_reusable_prefix_len([1, 2, 3, 4]) == 3

    ok, matched = cache_manager.allocate_request("extended", 4, token_ids=[1, 2, 3, 4])
    assert ok
    assert matched == 3

    restored_slot = cache_manager.get_slot("extended")
    restored_conv = linear_cache.conv_state_cache[0, restored_slot]
    restored_linear = linear_cache.linear_state_cache[0, restored_slot]

    assert mx.all(restored_conv == 1).item()
    assert mx.all(restored_linear == 2).item()


def test_cache_manager_keeps_last_prompt_token_for_attention_only_match(monkeypatch):
    _fixed_cache_allocation(monkeypatch, num_blocks=8)

    cache_manager = CacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        dtype=mx.float32,
        block_size=1,
        layer_types=["attention"],
        enable_prefix_cache=True,
    )

    ok, matched = cache_manager.allocate_request("cached", 3, token_ids=[1, 2, 3])
    assert ok
    assert matched == 0
    cache_manager.insert_full_blocks_to_cache("cached")
    cache_manager.free_request("cached")

    assert cache_manager.get_reusable_prefix_len([1, 2, 3]) == 2

    ok, matched = cache_manager.allocate_request("same", 3, token_ids=[1, 2, 3])
    assert ok
    assert matched == 2
