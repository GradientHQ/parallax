import unittest

import mlx.core as mx
import numpy as np

from parallax.server.cache_manager import CacheManager
from parallax_extensions.ops import reshape_and_cache


class TestPagedKVIntegration(unittest.TestCase):

    def setUp(self):
        self.num_layers = 1
        self.num_kv_heads = 4
        self.head_dim = 16
        self.block_size = 16
        self.dtype = mx.float32

        # Initialize Cache Manager
        # (Key Cache shape [blocks, heads, dim/x, block_size, x])
        self.cache_manager = CacheManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            block_size=self.block_size,
            cache_memory_fraction=0.5,
        )

        # Ensure we have enough blocks for testing
        if self.cache_manager.num_gpu_blocks < 100:
            pass

    def test_prefill_slot_mapping(self):
        """
        Test that slot_mapping is correctly generated and used in reshape_and_cache
        for a batch of sequences with different lengths (Prefill phase).
        """
        # 1. Setup Requests
        req_ids = ["req1", "req2"]
        seq_lens = [10, 20]

        # Allocate blocks
        for rid, slen in zip(req_ids, seq_lens):
            self.cache_manager.allocate_request(rid, slen)

        block_tables = []
        for rid in req_ids:
            block_tables.append(self.cache_manager.get_block_table(rid))

        # Pad block tables (needed for wrapper inputs, though new kernel uses slot_mapping)
        max_blocks = max(len(bt) for bt in block_tables)
        padded_block_tables = []
        for bt in block_tables:
            padded_block_tables.append(bt + [0] * (max_blocks - len(bt)))

        block_tables_tensor = mx.array(padded_block_tables, dtype=mx.int32)
        context_lengths_tensor = mx.array(seq_lens, dtype=mx.int32)

        # 2. Generate Input Data (Batch, MaxLen, KV_Heads, Head_Dim)
        max_len = 20
        batch_size = 2

        # Create dummy key/value data
        # Pattern: key[b, l] = b * 1000 + l
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

        # Flatten inputs for kernel: (Batch * MaxLen, Heads, Dim)
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
                    # Padding token
                    slot_mapping_flat.append(-1)

        slot_mapping_tensor = mx.array(slot_mapping_flat, dtype=mx.int64)

        # 4. Run Kernel (get cache for layer 0)
        layer_cache = self.cache_manager.get_caches()[0]
        key_cache, value_cache = layer_cache.get_cache()

        # Call NEW wrapper
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

        # Force sync to ensure writing is done (kernel is async and has no output handle)
        mx.eval(key_cache, value_cache)
        mx.synchronize()

        # 5. Verify Cache Content

        def unpack_key_block(block_idx):
            # Key Cache Physics: [blocks, heads, dim/x, block_size, x]
            # We access specific block -> [heads, dim/x, block_size, x]
            k_phys = key_cache[block_idx]
            # Transpose to [heads, block_size, dim/x, x]
            k_trans = k_phys.transpose(0, 2, 1, 3)
            # Reshape to [heads, block_size, dim]
            return k_trans.reshape(self.num_kv_heads, self.block_size, self.head_dim)

        def unpack_value_block(block_idx):
            # Value Cache Physics: [blocks, heads, dim, block_size]
            v_phys = value_cache[block_idx]
            # Transpose to [heads, block_size, dim]
            return v_phys.transpose(0, 2, 1)

        # --- Verify Req 1 (batch 0, len 10) ---
        # req1 fits in 1 block
        block_idx_req1 = block_tables[0][0]

        # Get logical view of this block
        k_block_logic = unpack_key_block(block_idx_req1)

        # Check first token (offset 0)
        cached_k_0 = k_block_logic[:, 0, :]
        expected_k_0 = mx.array(keys_np[0, 0, :, :])
        self.assertTrue(mx.allclose(cached_k_0, expected_k_0).item(), "Req1 Token 0 Key Mismatch")

        # Check last token (offset 9)
        cached_k_9 = k_block_logic[:, 9, :]
        expected_k_9 = mx.array(keys_np[0, 9, :, :])
        self.assertTrue(mx.allclose(cached_k_9, expected_k_9).item(), "Req1 Token 9 Key Mismatch")

        # --- Verify Req 2 (batch 1, len 20) ---
        # req2 spans 2 blocks
        block_idx_req2_0 = block_tables[1][0]
        block_idx_req2_1 = block_tables[1][1]

        # Check token 0 (in first block)
        k_block_b1 = unpack_key_block(block_idx_req2_0)
        cached_k_b1_0 = k_block_b1[:, 0, :]
        expected_k_b1_0 = mx.array(keys_np[1, 0, :, :])
        self.assertTrue(
            mx.allclose(cached_k_b1_0, expected_k_b1_0).item(), "Req2 Token 0 Key Mismatch"
        )

        # Check token 16 (in second block, offset 0)
        k_block_b2 = unpack_key_block(block_idx_req2_1)
        cached_k_b1_16 = k_block_b2[:, 0, :]
        expected_k_b1_16 = mx.array(keys_np[1, 16, :, :])
        self.assertTrue(
            mx.allclose(cached_k_b1_16, expected_k_b1_16).item(),
            "Req2 Token 16 Key Mismatch (Cross Block)",
        )
