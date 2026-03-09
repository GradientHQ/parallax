"""
Tests for the radix tree.
"""

import mlx.core as mx

from parallax.server.radix_cache import RadixCache

if __name__ == "__main__":
    DATA_TYPE = mx.bfloat16
    tree = RadixCache(
        num_kv_heads=1,
        head_dim=4,
        num_layers=10,
        dtype=DATA_TYPE,
        page_size=1,
        max_num_tokens=10000,
    )
    arr_for_test = mx.zeros([tree.num_layers, tree.num_kv_heads, 1, tree.head_dim], dtype=DATA_TYPE)

    tree.insert("Hello", None, arr_for_test, arr_for_test)
    tree.insert("Hello", None, arr_for_test, arr_for_test)
    tree.insert("Hello_L.A.!", None, arr_for_test, arr_for_test)
    tree.insert("Hello_world! Happy", None, arr_for_test, arr_for_test)
    tree.insert("I love you!", None, arr_for_test, arr_for_test)
    tree.pretty_print()

    print(tree.match_prefix("I love you! aha"))

    tree.evict(5)
    tree.evict(10)
    tree.pretty_print()
