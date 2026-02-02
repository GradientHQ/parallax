import unittest

import mlx.core as mx
import numpy as np

from parallax.metal.paged_attention.kernel import reshape_and_cache
from parallax.server.cache_manager import CacheManager
from parallax.utils.utils import is_metal_available


@unittest.skip(
    "Deprecated: old Python/Metal kernel incompatible with KVCachePacked layout. Use parallax_extensions ops instead."
)
class TestPagedKVIntegration(unittest.TestCase):
    def setUp(self):
        # Skip entire test class if Metal is not available
        if not is_metal_available():
            self.skipTest("Metal backend not available (requires macOS with Metal support)")

        self.num_layers = 1
        self.num_kv_heads = 4
        self.head_dim = 16
        self.block_size = 16
        self.dtype = mx.float32

        # Initialize Cache Manager
        # Mocking device info to avoid OOM or device dependency in test env
        # Assuming cache_memory_fraction results in enough blocks
        # We will manually override num_gpu_blocks if needed or rely on default fallback
        self.cache_manager = CacheManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            block_size=self.block_size,
            cache_memory_fraction=0.5,
            sliding_window=16,
        )

        # Ensure we have enough blocks for testing
        if self.cache_manager.num_gpu_blocks < 100:
            # Manually resize if auto-calc gave too few (e.g. on small device)
            pass

    def test_prefill_slot_mapping(self):
        """
        Test that slot_mapping is correctly generated and used in reshape_and_cache
        for a batch of sequences with different lengths (Prefill phase).
        """
        # 1. Setup Requests
        req_ids = ["req1", "req2"]
        seq_lens = [10, 20]  # Both < block_size * 2 for simplicity

        # Allocate blocks
        for rid, slen in zip(req_ids, seq_lens):
            self.cache_manager.allocate_request(rid, slen)

        block_tables = []
        for rid in req_ids:
            block_tables.append(self.cache_manager.get_block_table(rid))

        # Pad block tables
        max_blocks = max(len(bt) for bt in block_tables)
        padded_block_tables = []
        for bt in block_tables:
            padded_block_tables.append(bt + [0] * (max_blocks - len(bt)))
        block_tables_tensor = mx.array(padded_block_tables, dtype=mx.int32)

        context_lengths_tensor = mx.array(seq_lens, dtype=mx.int32)

        # 2. Generate Input Data (Batch, MaxLen, KV_Heads, Head_Dim)
        # We need to flatten this for reshape_and_cache
        max_len = 20
        batch_size = 2

        # Create dummy key/value data
        # We use a pattern to easily verify: key[b, l] = b * 1000 + l
        keys_np = np.zeros(
            (batch_size, max_len, self.num_kv_heads, self.head_dim), dtype=np.float32
        )
        values_np = np.zeros(
            (batch_size, max_len, self.num_kv_heads, self.head_dim), dtype=np.float32
        )

        for b in range(batch_size):
            for l in range(seq_lens[b]):
                keys_np[b, l, :, :] = b * 1000 + l
                values_np[b, l, :, :] = -(b * 1000 + l)

        keys = mx.array(keys_np)
        values = mx.array(values_np)

        # Flatten inputs for kernel: (Batch * MaxLen, ...)
        keys_flat = keys.reshape(-1, self.num_kv_heads, self.head_dim)
        values_flat = values.reshape(-1, self.num_kv_heads, self.head_dim)

        # 3. Generate Slot Mapping (Logic from Executor)
        slot_mapping_flat = []
        for i in range(batch_size):
            block_table = block_tables[i]
            length = seq_lens[i]

            for seq_idx in range(max_len):
                if seq_idx < length:
                    block_idx = seq_idx // self.block_size
                    block_offset = seq_idx % self.block_size
                    physical_block = block_table[block_idx]
                    slot = physical_block * self.block_size + block_offset
                    slot_mapping_flat.append(slot)
                else:
                    slot_mapping_flat.append(-1)

        slot_mapping_tensor = mx.array(slot_mapping_flat, dtype=mx.int64)

        # 4. Run Kernel (get cache for layer 0)
        layer_cache = self.cache_manager.get_caches()[0]
        key_cache, value_cache = layer_cache.get_cache()

        reshape_and_cache(
            keys_flat,
            values_flat,
            key_cache,
            value_cache,
            block_tables_tensor,
            context_lengths_tensor,
            self.block_size,
            slot_mapping=slot_mapping_tensor,
        )

        mx.eval(key_cache, value_cache)

        # 5. Verify Cache Content
        # Check req1 (batch 0, len 10)
        # req1 fits in 1 block (block_size=16)
        block_idx_req1 = block_tables[0][0]
        # Check first token
        # key_cache shape: (1, num_blocks, num_kv_heads, block_size, head_dim)
        cached_k_0 = key_cache[
            0, block_idx_req1, :, 0, :
        ]  # dim0=placeholder, block, heads, offset 0, dim
        expected_k_0 = mx.array(keys_np[0, 0, :, :])
        self.assertTrue(mx.allclose(cached_k_0, expected_k_0).item(), "Req1 Token 0 Key Mismatch")

        # Check last token of req1
        cached_k_9 = key_cache[0, block_idx_req1, :, 9, :]
        expected_k_9 = mx.array(keys_np[0, 9, :, :])
        self.assertTrue(mx.allclose(cached_k_9, expected_k_9).item(), "Req1 Token 9 Key Mismatch")

        # Check req2 (batch 1, len 20)
        # req2 spans 2 blocks. 0-15 in block 0, 16-19 in block 1
        block_idx_req2_0 = block_tables[1][0]
        block_idx_req2_1 = block_tables[1][1]

        # Check token 0 (in first block)
        cached_k_b1_0 = key_cache[0, block_idx_req2_0, :, 0, :]
        expected_k_b1_0 = mx.array(keys_np[1, 0, :, :])
        self.assertTrue(
            mx.allclose(cached_k_b1_0, expected_k_b1_0).item(), "Req2 Token 0 Key Mismatch"
        )

        # Check token 16 (in second block, offset 0)
        cached_k_b1_16 = key_cache[0, block_idx_req2_1, :, 0, :]
        expected_k_b1_16 = mx.array(keys_np[1, 16, :, :])
        self.assertTrue(
            mx.allclose(cached_k_b1_16, expected_k_b1_16).item(),
            "Req2 Token 16 Key Mismatch (Cross Block)",
        )


if __name__ == "__main__":
    unittest.main()
