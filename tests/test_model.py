"""
Tests for the ShardedModel loader utilities.
"""

from typing import List, Tuple

import mlx.core as mx
import pytest
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import _download, load_model

from parallax.server.cache_manager import CacheManager
from parallax.server.shard_loader import MLXModelLoader
from parallax.utils.tokenizer_utils import load_tokenizer
from parallax.utils.utils import is_metal_available, pad_inputs

REPO_ID = "mlx-community/Qwen3-0.6B-bf16"
TOTAL_LAYERS = 28


model_path = _download(REPO_ID)
ref_model, ref_config = load_model(model_path)
ref_tokenizer = load_tokenizer(model_path, eos_token_ids=ref_config.get("eos_token_id", None))


@pytest.mark.parametrize(
    "layers_config",
    [
        [(0, 12), (12, TOTAL_LAYERS)],
        [(0, 8), (8, 16), (16, TOTAL_LAYERS)],
    ],
)
def test_shard_prefill(layers_config: List[Tuple[int, int]]) -> None:
    """Load sharded model based on layers_config and
    compare its forward pass with a full reference model.
    """
    if not is_metal_available():
        pytest.skip("Metal backend not available (requires macOS with Metal support)")

    dtype = mx.bfloat16

    # Load sharded models
    model_shards = []
    for layer_from, layer_to in layers_config:
        loader = MLXModelLoader(
            model_path_or_hf_repo=REPO_ID,
            start_layer=layer_from,
            end_layer=layer_to,
        )
        model_shard_instance, _, _ = loader.load()
        model_shards.append(model_shard_instance)

    # Prepare test inputs
    texts = [
        "This is a test.",
        "This is yet another test.",
        "what color is Mars",
        "what color is Moon",
    ]
    ref_ids = [ref_tokenizer.encode(text) for text in texts]
    ref_pad_token_id = ref_tokenizer.pad_token_id or ref_tokenizer.eos_token_id
    ref_ids, ref_mask = pad_inputs(ref_pad_token_id, ref_ids, dtype=dtype)

    # Run reference model
    def _call_with_mask(self, inputs, cache=None, mask=None):
        h = self.model.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            mask_inner = create_attention_mask(h, cache[0])
        else:
            mask_inner = mask
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask_inner, c)
        h = self.model.norm(h)
        if self.args.tie_word_embeddings:
            h = self.model.embed_tokens.as_linear(h)
        else:
            h = self.lm_head(h)
        return h

    type(ref_model).__call__ = _call_with_mask
    ref_out = ref_model(ref_ids, mask=ref_mask)

    # Prepare PagedKV cache managers for each shard
    batch_size = len(texts)
    max_seq_len = ref_ids.shape[1]
    num_kv_heads = ref_config.get("num_key_value_heads")
    head_dim = ref_config.get("head_dim") or ref_config.get("hidden_size") // ref_config.get(
        "num_attention_heads"
    )

    cache_managers = []
    cache_memory_fraction = 0
    for shard in model_shards:
        num_shard_layers = shard.end_layer - shard.start_layer
        cache_memory_fraction += 0.1
        cache_mgr = CacheManager(
            num_layers=num_shard_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            block_size=64,
            num_gpu_blocks=200,
        )
        cache_managers.append(cache_mgr)

    # Prepare common inputs
    padding_mask = (ref_ids != ref_pad_token_id).astype(dtype)
    actual_seq_lengths = [int((padding_mask[i] > 0).sum()) for i in range(batch_size)]

    # Run sharded models
    x = None
    for shard_idx, shard in enumerate(model_shards):
        cache_manager = cache_managers[shard_idx]

        # Allocate blocks and prepare metadata
        block_tables_list = []
        context_lengths_list = []
        slot_mapping_flat = []

        for i in range(batch_size):
            req_id = f"req_{i}_shard_{shard_idx}"
            seq_len = actual_seq_lengths[i]
            context_lengths_list.append(seq_len)

            success = cache_manager.allocate_request(req_id, seq_len)
            assert success, f"Failed to allocate blocks for request {i} in shard {shard_idx}"

            block_table = cache_manager.get_block_table(req_id)
            block_tables_list.append(block_table)

            # Generate slot mapping
            for seq_idx in range(max_seq_len):
                if seq_idx < seq_len:
                    block_idx = seq_idx // cache_manager.block_size
                    block_offset = seq_idx % cache_manager.block_size
                    physical_block = block_table[block_idx]
                    slot = physical_block * cache_manager.block_size + block_offset
                    slot_mapping_flat.append(slot)
                else:
                    slot_mapping_flat.append(-1)

        # Pad block tables
        max_blocks = max(len(bt) for bt in block_tables_list)
        padded_block_tables = [bt + [0] * (max_blocks - len(bt)) for bt in block_tables_list]

        block_tables = mx.array(padded_block_tables, dtype=mx.int32)
        context_lengths = mx.array(context_lengths_list, dtype=mx.int32)
        slot_mapping = mx.array(slot_mapping_flat, dtype=mx.int64)
        cache = cache_manager.get_caches()

        # Forward pass
        input_data = ref_ids if shard.start_layer == 0 else x
        x = shard(
            input_data,
            cache=cache,
            mask=ref_mask,
            block_tables=block_tables,
            context_lengths=context_lengths,
            slot_mapping=slot_mapping,
        )

    assert mx.allclose(x, ref_out, atol=1e-3, rtol=1e-3)
