"""
Tests for the ShardedModel loader utilities.
"""

from typing import List, Tuple

import mlx.core as mx
import pytest
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import get_model_path, load_model

from parallax.server.server_info import ShardedModelInfo
from parallax.server.shard_loader import MLXModelLoader
from parallax.utils.tokenizer_utils import load_tokenizer
from parallax.utils.utils import pad_inputs

REPO_ID = "mlx-community/Qwen3-0.6B-bf16"
TOTAL_LAYERS = 28


model_path = get_model_path(REPO_ID)[0]
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
    model_shards = []
    tokenizer = None
    for layer_from, layer_to in layers_config:
        loader = MLXModelLoader(
            model_path_or_hf_repo=REPO_ID,
            start_layer=layer_from,
            end_layer=layer_to,
        )
        model_shard_instance, _, _tokenizer = loader.load()
        if layer_from == 0:
            tokenizer = _tokenizer
        model_info = ShardedModelInfo.from_sharded_model(model_shard_instance)
        print(model_info)

        model_shards.append(model_shard_instance)

    assert len(model_shards) == len(layers_config), "Number of loaded shards should match config"

    texts = [
        "This is a test.",
        "This is yet another test.",
        "what color is Mars",
        "what color is Moon",
    ]
    ref_ids = [ref_tokenizer.encode(text) for text in texts]
    ref_pad_token_id = ref_tokenizer.pad_token_id
    if ref_pad_token_id is None:
        ref_pad_token_id = ref_tokenizer.eos_token_id
    ref_ids, ref_mask = pad_inputs(ref_pad_token_id, ref_ids)

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

    mask = None
    for shard in model_shards:
        if shard.start_layer == 0:
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id
            ids = [tokenizer.encode(text) for text in texts]
            ids, mask = pad_inputs(pad_token_id, ids)
            x, _ = shard(ids, cache=None, mask=mask)
        else:
            x, _ = shard(x, cache=None, mask=mask)

    assert mx.allclose(x, ref_out, atol=1e-3, rtol=1e-3)
