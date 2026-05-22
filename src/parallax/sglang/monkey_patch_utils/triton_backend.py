from typing import Optional

import torch
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.model_executor.model_runner import ModelRunner

_original_triton_backend_init = TritonAttnBackend.__init__


def parallax_triton_backend_init(
    self,
    model_runner: ModelRunner,
    skip_prefill: bool = False,
    kv_indptr_buf: Optional[torch.Tensor] = None,
):
    pp_start_layer = getattr(model_runner, "pp_start_layer", 0)
    token_to_kv_pool = model_runner.token_to_kv_pool
    token_to_kv_pool_dict = getattr(token_to_kv_pool, "__dict__", {})
    had_get_value_buffer_override = "get_value_buffer" in token_to_kv_pool_dict
    get_value_buffer_override = token_to_kv_pool_dict.get("get_value_buffer")
    original_get_value_buffer = token_to_kv_pool.get_value_buffer

    def get_value_buffer(layer_id: int, *args, **kwargs):
        if layer_id == 0:
            layer_id = pp_start_layer
        return original_get_value_buffer(layer_id, *args, **kwargs)

    token_to_kv_pool.get_value_buffer = get_value_buffer
    try:
        return _original_triton_backend_init(
            self,
            model_runner,
            skip_prefill=skip_prefill,
            kv_indptr_buf=kv_indptr_buf,
        )
    finally:
        if had_get_value_buffer_override:
            token_to_kv_pool.get_value_buffer = get_value_buffer_override
        else:
            delattr(token_to_kv_pool, "get_value_buffer")


def apply_triton_backend_init_monkey_patch():
    TritonAttnBackend.__init__ = parallax_triton_backend_init
