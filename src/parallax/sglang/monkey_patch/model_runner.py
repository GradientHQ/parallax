# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ModelRunner runs the forward passes of the models."""

import logging
import os
from typing import Optional

import torch
from sglang.srt.distributed import get_world_group
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator_ascend import AscendPagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    AscendMLAPagedTokenToKVPool,
    AscendTokenToKVPool,
    DoubleSparseTokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_available_gpu_memory,
    is_hip,
    is_npu,
)

_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()

# Use a small KV cache pool size for tests in CI
SGLANG_CI_SMALL_KV_SIZE = os.getenv("SGLANG_CI_SMALL_KV_SIZE", None)

# Detect stragger ranks in model loading
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 300

logger = logging.getLogger(__name__)


def monkey_patch_profile_max_num_token(self, total_gpu_memory: int):
    available_gpu_memory = get_available_gpu_memory(
        self.device,
        self.gpu_id,
        distributed=get_world_group().world_size > 1,
        cpu_group=get_world_group().cpu_group,
    )
    if self.is_draft_worker:
        num_layers = getattr(
            self.model_config.hf_config,
            "num_nextn_predict_layers",
            self.num_effective_layers,
        )
    elif self.is_hybrid_gdn:
        num_layers = len(self.model_config.hf_config.full_attention_layer_ids)
    else:
        num_layers = self.num_effective_layers
    if self.use_mla_backend:
        cell_size = (
            (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
            * num_layers
            * torch._utils._element_size(self.kv_cache_dtype)
        )
    else:
        cell_size = (
            self.model_config.get_num_kv_heads(get_attention_tp_size())
            * self.model_config.head_dim
            * num_layers
            * 2
            * torch._utils._element_size(self.kv_cache_dtype)
        )
    rest_memory = available_gpu_memory - total_gpu_memory * (1 - self.mem_fraction_static)
    # print(f"Line-1126 rest_memory: {rest_memory}")
    # print(f"Line-1127 max_mamba_cache_size: {self.server_args.max_mamba_cache_size}")
    # print(f"Line-1128 mamba_cache_per_req: {self.model_config.hf_config.mamba_cache_per_req}")

    if self.is_hybrid_gdn:
        all_mamba_layers = self.model_config.hf_config.linear_layer_ids
        start_layer_id, end_layer_id = self.model.model.start_layer, self.model.model.end_layer
        pp_mamba_layers = [i for i in all_mamba_layers if i >= start_layer_id and i < end_layer_id]
        all_mamba_layers_len = len(all_mamba_layers)
        pp_mamba_layers_len = len(pp_mamba_layers)
        all_layers_mamba_cache_per_req = self.model_config.hf_config.mamba_cache_per_req
        pp_layers_mamba_cache_per_req = (
            all_layers_mamba_cache_per_req * pp_mamba_layers_len // all_mamba_layers_len
        )

        # print(f"Line-1231 pp_layers_mamba_cache_per_req: {pp_layers_mamba_cache_per_req}")
        rest_memory -= (
            self.server_args.max_mamba_cache_size * pp_layers_mamba_cache_per_req / (1 << 30)
        )
    # print(f"Line-1233 rest_memory: {rest_memory}, cell_size: {cell_size}")
    max_num_token = int(rest_memory * (1 << 30) // cell_size)
    return max_num_token


def monkey_patch_init_memory_pool(
    self,
    total_gpu_memory: int,
    max_num_reqs: Optional[int] = None,
    max_total_tokens: Optional[int] = None,
):
    # Determine the kv cache dtype
    if self.server_args.kv_cache_dtype == "auto":
        self.kv_cache_dtype = self.dtype
    elif self.server_args.kv_cache_dtype == "fp8_e5m2":
        if _is_hip:  # Using natively supported format
            self.kv_cache_dtype = torch.float8_e5m2fnuz
        else:
            self.kv_cache_dtype = torch.float8_e5m2
    elif self.server_args.kv_cache_dtype == "fp8_e4m3":
        if _is_hip:  # Using natively supported format
            self.kv_cache_dtype = torch.float8_e4m3fnuz
        else:
            self.kv_cache_dtype = torch.float8_e4m3fn
    else:
        raise ValueError(f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}.")

    # print(f"Line-1245 total_gpu_memory: {total_gpu_memory}")
    self.max_total_num_tokens = self.profile_max_num_token(total_gpu_memory)
    # print(f"Line-1249 Profiled max_total_num_tokens: {self.max_total_num_tokens}")

    if SGLANG_CI_SMALL_KV_SIZE:
        self.max_total_num_tokens = int(SGLANG_CI_SMALL_KV_SIZE)

    if max_num_reqs is None:
        max_num_reqs = min(
            max(
                int(self.max_total_num_tokens / self.model_config.context_len * 512),
                2048,
            ),
            4096,
        )
    if self.is_hybrid_gdn:
        max_num_reqs = min(max_num_reqs, self.server_args.max_mamba_cache_size)

    if not self.spec_algorithm.is_none():
        if self.is_draft_worker:
            self.max_total_num_tokens = self.server_args.draft_runner_cache_size
            max_num_reqs = self.server_args.max_num_reqs
        else:
            # We are sharing the `token_to_kv_pool`, and both verify and draft tokens
            # can be concurrently allocated, so we should give a headroom for it.
            self.server_args.draft_runner_cache_size = (
                self.max_total_num_tokens
                # draft
                + max_num_reqs
                * self.server_args.speculative_num_steps
                * self.server_args.speculative_eagle_topk
                # verify
                + max_num_reqs * self.server_args.speculative_num_draft_tokens
                # buffer
                + 100
            )
            # Target worker and draft worker shares the same indices for the
            # token_to_kv_pool, so we should make sure to match max_total_num_tokens.
            self.max_total_num_tokens = self.server_args.draft_runner_cache_size
            self.server_args.max_num_reqs = max_num_reqs

    if max_total_tokens is not None:
        if max_total_tokens > self.max_total_num_tokens:
            logging.warning(
                f"max_total_tokens={max_total_tokens} is larger than the profiled value "
                f"{self.max_total_num_tokens}. "
                f"Use the profiled value instead."
            )
        self.max_total_num_tokens = min(self.max_total_num_tokens, max_total_tokens)

    self.max_total_num_tokens = (
        self.max_total_num_tokens // self.server_args.page_size * self.server_args.page_size
    )
    # different pp rank may have different num of layers, so we need to reduce the max_total_num_tokens
    if self.pp_size > 1:
        tensor = torch.tensor(self.max_total_num_tokens, dtype=torch.int64)
        torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=get_world_group().cpu_group,
        )
        self.max_total_num_tokens = tensor.item()

    # create token size for hybrid cache
    if self.is_hybrid:
        self.set_num_token_hybrid()

    if self.max_total_num_tokens <= 0:
        raise RuntimeError("Not enough memory. Please try to increase --mem-fraction-static.")

    # Initialize req_to_token_pool
    if self.req_to_token_pool is None:
        # FIXME(lsyin): this is the temporary fix for the context length issue when using speculative decoding
        extra_max_context_len = 4
        if self.server_args.speculative_num_draft_tokens is not None:
            extra_max_context_len += self.server_args.speculative_num_draft_tokens

        if self.server_args.disaggregation_mode == "decode":
            from sglang.srt.disaggregation.decode import DecodeReqToTokenPool

            # subscribe memory for pre-allocated requests
            # if max_num_reqs <= 32, we pre-allocate 2x requests
            pre_alloc_size = max_num_reqs * 2 if max_num_reqs <= 32 else 0
            self.req_to_token_pool = DecodeReqToTokenPool(
                size=max_num_reqs,
                max_context_len=self.model_config.context_len + extra_max_context_len,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                pre_alloc_size=pre_alloc_size,
            )
        elif self.is_hybrid_gdn:
            config = self.model_config.hf_config
            (
                conv_state_shape,
                temporal_state_shape,
                conv_dtype,
                ssm_dtype,
                mamba_layers,
            ) = config.hybrid_gdn_params

            start_layer_id, end_layer_id = (
                self.model.model.start_layer,
                self.model.model.end_layer,
            )
            pp_mamba_layers = [i for i in mamba_layers if i >= start_layer_id and i < end_layer_id]

            self.req_to_token_pool = HybridReqToTokenPool(
                size=max_num_reqs,
                max_context_len=self.model_config.context_len + extra_max_context_len,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                conv_state_shape=conv_state_shape,
                temporal_state_shape=temporal_state_shape,
                conv_dtype=conv_dtype,
                ssm_dtype=ssm_dtype,
                mamba_layers=pp_mamba_layers,
                speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
            )
        else:
            self.req_to_token_pool = ReqToTokenPool(
                size=max_num_reqs,
                max_context_len=self.model_config.context_len + extra_max_context_len,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
            )
    else:
        # Draft worker shares req_to_token_pool with the target worker.
        assert self.is_draft_worker

    # Initialize token_to_kv_pool
    if self.server_args.attention_backend == "ascend":
        if self.use_mla_backend:
            self.token_to_kv_pool = AscendMLAPagedTokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
            )
        else:
            self.token_to_kv_pool = AscendTokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
                head_dim=self.model_config.head_dim,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
            )
    elif self.use_mla_backend:
        self.token_to_kv_pool = MLATokenToKVPool(
            self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            layer_num=self.num_effective_layers,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
        )
    elif self.server_args.enable_double_sparsity:
        self.token_to_kv_pool = DoubleSparseTokenToKVPool(
            self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
            head_dim=self.model_config.head_dim,
            layer_num=self.num_effective_layers,
            device=self.device,
            heavy_channel_num=self.server_args.ds_heavy_channel_num,
            enable_memory_saver=self.server_args.enable_memory_saver,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
        )
    else:
        if self.is_hybrid:
            self.token_to_kv_pool = SWAKVPool(
                size=self.full_max_total_num_tokens,
                size_swa=self.swa_max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
                head_dim=self.model_config.head_dim,
                swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                enable_kvcache_transpose=False,
                device=self.device,
            )
        elif self.is_hybrid_gdn:
            self.token_to_kv_pool = HybridLinearKVPool(
                size=self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
                head_dim=self.model_config.head_dim,
                # if draft worker, we only need 1 attention layer's kv pool
                full_attention_layer_ids=(
                    [0]
                    if self.is_draft_worker
                    else self.model_config.hf_config.full_attention_layer_ids
                ),
                enable_kvcache_transpose=False,
                device=self.device,
            )
        else:
            self.token_to_kv_pool = MHATokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
                head_dim=self.model_config.head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
            )

    # Initialize token_to_kv_pool_allocator
    need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
    if self.token_to_kv_pool_allocator is None:
        if self.server_args.attention_backend == "ascend":
            self.token_to_kv_pool_allocator = AscendPagedTokenToKVPoolAllocator(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                device=self.device,
                kvcache=self.token_to_kv_pool,
                need_sort=need_sort,
            )
        else:
            if self.page_size == 1:
                if self.is_hybrid:
                    self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                        self.full_max_total_num_tokens,
                        self.swa_max_total_num_tokens,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
            else:
                assert not self.is_hybrid
                self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=need_sort,
                )
    else:
        assert self.is_draft_worker

    logger.info(
        f"Memory pool end. "
        f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
    )
