import mlx.core as mx

from parallax.server.block_radix_cache import BlockRadixCache
from parallax.server.cache.linear_cache import LinearCache
from parallax.server.cache_manager import CacheManager


def _linear_cache(cache_manager: CacheManager) -> LinearCache:
    for cache in cache_manager.get_caches():
        if isinstance(cache, LinearCache):
            return cache
    raise AssertionError("expected a LinearCache")


def test_linear_aware_radix_requires_snapshot_for_match():
    cache = BlockRadixCache(block_size=1, has_linear_cache=True)
    node1 = cache.insert_block([1], block_id=10)
    node2 = cache.insert_block([2], block_id=11, parent_path=[node1])

    assert cache.match_prefix([1, 2, 3]) == ([], 0)

    node2.linear_snapshot = object()

    assert cache.match_prefix([1, 2, 3]) == ([10, 11], 2)


def test_attach_linear_snapshot_uses_exact_prefix_tokens():
    cache = BlockRadixCache(block_size=1, has_linear_cache=True)
    node1 = cache.insert_block([1], block_id=10)
    node2 = cache.insert_block([2], block_id=11, parent_path=[node1])

    snapshot1 = object()
    snapshot2 = object()

    assert cache.attach_linear_snapshot([1], snapshot1)
    assert cache.attach_linear_snapshot([1, 2], snapshot2)

    assert node1.linear_snapshot is snapshot1
    assert node2.linear_snapshot is snapshot2
    assert not cache.attach_linear_snapshot([1, 2, 3], object())


def test_cache_manager_uses_prompt_minus_last_for_linear_prefix_match():
    cache_manager = CacheManager(
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
        dtype=mx.float32,
        block_size=1,
        num_gpu_blocks=8,
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


def test_cache_manager_keeps_last_prompt_token_for_attention_only_match():
    cache_manager = CacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        dtype=mx.float32,
        block_size=1,
        num_gpu_blocks=8,
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
