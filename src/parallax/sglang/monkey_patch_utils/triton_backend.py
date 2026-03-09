from typing import Optional

import torch
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import get_bool_env_var, get_device_core_count, get_int_env_var


def parallax_triton_backend_init(
    self,
    model_runner: ModelRunner,
    skip_prefill: bool = False,
    kv_indptr_buf: Optional[torch.Tensor] = None,
):
    # Lazy import to avoid the initialization of cuda context
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        decode_attention_fwd,
    )
    from sglang.srt.layers.attention.triton_ops.extend_attention import (
        extend_attention_fwd,
    )

    self.decode_attention_fwd = torch.compiler.disable(decode_attention_fwd)
    self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)

    # Parse args
    self.skip_prefill = skip_prefill
    max_bs = model_runner.req_to_token_pool.size
    self.sliding_window_size = model_runner.sliding_window_size
    self.req_to_token = model_runner.req_to_token_pool.req_to_token
    self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
    self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
    self.speculative_num_steps = model_runner.server_args.speculative_num_steps
    self.num_head = model_runner.model_config.num_attention_heads // get_attention_tp_size()
    self.num_kv_head = model_runner.model_config.get_num_kv_heads(get_attention_tp_size())
    # Modifies layer id to support pipeline parallel
    if model_runner.hybrid_gdn_config is not None:
        # For hybrid linear models, layer_id = 0 may not be full attention
        self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
    else:

        ################################################################################
        ## Patch for PP: get pp_start_layer
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(
            model_runner.pp_start_layer
        ).shape[-1]
        ## End of patch
        ################################################################################
    self.max_context_len = model_runner.model_config.context_len
    self.device = model_runner.device
    self.device_core_count = get_device_core_count(model_runner.gpu_id)
    self.static_kv_splits = get_bool_env_var("SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false")
    self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits

    # Decide whether enable deterministic inference with batch-invariant operations
    self.enable_deterministic = model_runner.server_args.enable_deterministic_inference

    # Configure deterministic inference settings
    if self.enable_deterministic:
        # Use fixed split tile size for batch invariance
        self.split_tile_size = get_int_env_var("SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", 256)
        # Set static_kv_splits to False to use deterministic logic instead
        self.static_kv_splits = False
    else:
        self.split_tile_size = model_runner.server_args.triton_attention_split_tile_size

    if self.split_tile_size is not None:
        self.max_kv_splits = (
            self.max_context_len + self.split_tile_size - 1
        ) // self.split_tile_size
    # Check arguments
    assert not (
        model_runner.sliding_window_size is not None
        and model_runner.model_config.is_encoder_decoder
    ), "Sliding window and cross attention are not supported together"

    # Initialize buffers
    # TODO(Jianan Ji): Make sure it behaves as expected when kv_indptr_buf is provided and sliding window is enabled
    if kv_indptr_buf is None:
        self.kv_indptr = torch.zeros((max_bs + 1,), dtype=torch.int32, device=model_runner.device)
    else:
        self.kv_indptr = kv_indptr_buf

    # If sliding window is enabled, we might need two sets of buffers
    # because of interleaved attention types (e.g. for Gemma3)
    self.window_kv_indptr = None
    if self.sliding_window_size is not None and self.sliding_window_size > 0:
        if kv_indptr_buf is None:
            self.window_kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            # When provided a buffer, create a clone for the second buffer
            self.window_kv_indptr = torch.zeros_like(kv_indptr_buf)

    if not self.skip_prefill:
        self.qo_indptr = torch.zeros((max_bs + 1,), dtype=torch.int32, device=model_runner.device)

        self.mask_indptr = torch.zeros((max_bs + 1,), dtype=torch.int64, device=model_runner.device)

    # Initialize forward metadata
    from sglang.srt.layers.attention.triton_backend import ForwardMetadata

    self.forward_metadata: ForwardMetadata = None

    self.cuda_graph_custom_mask = None


def apply_triton_backend_init_monkey_patch():
    TritonAttnBackend.__init__ = parallax_triton_backend_init
