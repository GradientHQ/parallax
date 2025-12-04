"""
MLX-LM backend implementation of high level executor
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

from parallax.server.executor.base_executor import BaseExecutor
from parallax.server.paged_kv_cache import PagedKVCacheManager
from parallax.server.request import (
    InitialRequest,
    IntermediateRequest,
    Request,
    RequestStatus,
)
from parallax.server.sampling.sampler import SamplingBatchInfo
from parallax.server.shard_loader import MLXModelLoader
from parallax.utils.utils import (
    combine_padding_and_causal_masks,
    create_causal_mask,
    get_device_dtype,
    pad_inputs,
)
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class MLXExecutor(BaseExecutor):
    def __init__(
        self,
        # Model Configs
        model_repo: str,
        start_layer: int,
        end_layer: int,
        dtype: str = "float16",
        # Device override
        device: Optional[str] = None,
        use_hfcache: bool = False,
        # Scheduler Configs
        max_batch_size: Optional[int] = 8,
        max_sequence_length: Optional[int] = None,
        max_tokens_in_kv_pool: Optional[int] = None,
        # Controlling perfill / decode ratio
        max_num_tokens_per_batch: int = 1024,
        prefill_priority: int = 0,
        micro_batch_ratio: int = 2,
        scheduler_wait_ms: int = 500,
        request_timeout_s: Optional[int] = 600,
        # Metrics Configs
        layer_latency_update_every: int = 4096,
        # KV Cache Configs
        kv_block_size: int = 64,
        kv_cache_memory_fraction: float = 0.8,
        enable_prefix_cache: Optional[bool] = False,
        # Communication Configs
        # P2P Communication Configs
        send_to_peer_addr: Optional[str] = None,
        recv_from_peer_addr: Optional[str] = None,
        # IPC Communication Configs
        executor_input_ipc_addr: Optional[str] = None,
        executor_output_ipc_addr: Optional[str] = None,
        # GPU Specialized Configs
        attention_backend: Optional[str] = "flashinfer",
        moe_runner_backend: Optional[str] = "auto",
        enable_lora: Optional[bool] = False,
        max_lora_rank: Optional[int] = None,
        lora_target_modules: Optional[List[str]] = None,
        lora_paths: Optional[List[str]] = None,
        max_loras_per_batch: Optional[int] = None,
        max_loaded_loras: Optional[int] = None,
        lora_eviction_policy: Optional[str] = "lru",
        lora_backend: Optional[str] = "triton",
        max_lora_chunk_size: Optional[int] = 128,
        # Tensor Parallel Configs
        tp_rank: Optional[int] = 0,
        tp_size: Optional[int] = 1,
        nccl_port: Optional[int] = 4000,
        # Optional shared state for layer reallocation detection (when running in subprocess)
        shared_state: Optional[dict] = None,
    ):
        logger.debug(
            f"Initializing MLX sharded model loader for repo={model_repo}, layers=[{start_layer}, {end_layer})"
        )
        self.shard_loader = MLXModelLoader(
            model_repo,
            start_layer=start_layer,
            end_layer=end_layer,
            use_hfcache=use_hfcache,
        )
        t0 = time.time()
        self.model_shard, self.config, self.tokenizer = self.shard_loader.load()

        adapters = lora_paths[0] if lora_paths else None
        if adapters:
            logger.debug(f"mlx adapters is: {adapters}")
            self.model_shard = self.shard_loader.load_lora(self.model_shard, adapters)

        logger.debug(
            f"MLX sharded model loaded in {(time.time() - t0) * 1000:.1f} ms; num_layers={self.config.get('num_hidden_layers')}"
        )

        # TODO: Duplicate code to BaseExecutor since num_shard_layers and dtype are needed for initializing kv cache
        self.num_shard_layers = end_layer - start_layer
        self.dtype = get_device_dtype(dtype, device)
        logger.debug(
            f"Executor dtype set to {dtype} (resolved={self.dtype}); shard_layers={self.num_shard_layers}"
        )

        # Calculate feature dimensions for kv cache
        num_key_value_heads = self.config.get("num_key_value_heads")
        head_dim = self.config.get("head_dim") or self.config.get("hidden_size") // self.config.get(
            "num_attention_heads"
        )
        qk_nope_head_dim = self.config.get("qk_nope_head_dim", None)
        qk_rope_head_dim = self.config.get("qk_rope_head_dim", None)
        if qk_nope_head_dim is not None and qk_rope_head_dim is not None:
            logger.debug(
                f"qk_nope_head_dim={qk_nope_head_dim}, qk_rope_head_dim={qk_rope_head_dim}"
            )
            head_dim = qk_nope_head_dim + qk_rope_head_dim

        v_head_dim = self.config.get("v_head_dim", None)
        linear_key_head_dim = self.config.get("linear_key_head_dim", None)
        linear_value_head_dim = self.config.get("linear_value_head_dim", None)
        linear_conv_kernel_dim = self.config.get("linear_conv_kernel_dim", None)
        linear_num_key_heads = self.config.get("linear_num_key_heads", None)
        linear_num_value_heads = self.config.get("linear_num_value_heads", None)
        key_dim, value_dim, conv_dim = None, None, None
        if linear_key_head_dim is not None and linear_num_key_heads is not None:
            key_dim = linear_key_head_dim * linear_num_key_heads
        if linear_value_head_dim is not None and linear_num_value_heads is not None:
            value_dim = linear_value_head_dim * linear_num_value_heads
        if key_dim is not None and value_dim is not None:
            conv_dim = key_dim * 2 + value_dim
        self.using_state_cache = linear_conv_kernel_dim is not None and conv_dim is not None

        indexer_key_head_dim = self.config.get("indexer_key_head_dim", None)
        indexer_num_kv_heads = self.config.get("indexer_num_kv_heads", None)

        logger.debug(
            "Initializing PagedKVCacheManager (mlx) with block_size=%d, layers=%d",
            kv_block_size,
            self.num_shard_layers,
        )
        self.kv_cache_manager = PagedKVCacheManager(
            num_layers=self.num_shard_layers,
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
            dtype=self.dtype,
            block_size=kv_block_size,
            cache_memory_fraction=kv_cache_memory_fraction,
            head_dim_v=v_head_dim,
            indexer_key_head_dim=indexer_key_head_dim,
            indexer_num_kv_heads=indexer_num_kv_heads,
        )
        super().__init__(
            start_layer=start_layer,
            end_layer=end_layer,
            dtype=dtype,
            device=device,
            max_batch_size=max_batch_size,
            max_sequence_length=max_sequence_length,
            max_num_tokens_per_batch=max_num_tokens_per_batch,
            prefill_priority=prefill_priority,
            micro_batch_ratio=micro_batch_ratio,
            scheduler_wait_ms=scheduler_wait_ms,
            request_timeout_s=request_timeout_s,
            layer_latency_update_every=layer_latency_update_every,
            send_to_peer_addr=send_to_peer_addr,
            recv_from_peer_addr=recv_from_peer_addr,
            executor_input_ipc_addr=executor_input_ipc_addr,
            executor_output_ipc_addr=executor_output_ipc_addr,
            tp_rank=tp_rank,
            tp_size=tp_size,
            shared_state=shared_state,
        )

        try:
            mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
        except Exception:
            logger.warning(f"Using mlx without metal backend.")

        # Prefix Cache Manager
        self.enable_prefix_cache = enable_prefix_cache
        # self.prefix_cache = RadixCache(
        #     num_kv_heads=num_key_value_heads,
        #     head_dim=head_dim,
        #     head_dim_v=v_head_dim,
        #     num_layers=self.num_shard_layers,
        #     dtype=self.dtype,
        #     page_size=1,
        # )

        logger.debug(
            f"KVCacheManager ready; wired_limit set; prefix_cache={'on' if self.enable_prefix_cache else 'off'}"
        )

    def handle_input_requests(self, requests: List[Request]):
        """Update requests states and status in scheduler and cache manager."""
        if not requests:
            return
        if self.is_first_peer:
            # First peer can receive InitialRequests from the client RPC,
            # or IntermediateRequests from the last peer.
            for req in requests:
                if isinstance(req, InitialRequest):
                    self.scheduler.enque_request(req)
                elif isinstance(req, IntermediateRequest):
                    original_req = self.scheduler.get_running_request(req.request_id)
                    if original_req is None:
                        logger.warning(
                            f"Received IntermediateRequest {req.request_id}. "
                            "But no corresponding request found in scheduler. "
                            "It might have been cancelled or finished."
                        )
                        continue
                    if not self.kv_cache_manager.has_request(req.request_id):
                        logger.warning(
                            f"Received IntermediateRequest {req.request_id}. "
                            "But no corresponding request found in cache manager. "
                            "It might have been cancelled or finished."
                        )
                        continue

                    assert req.next_token_id is not None
                    original_req.commit_new_token(req.next_token_id)
                    if len(req.routing_table) > 0:
                        original_req.routing_table = req.routing_table

                    # Check for termination.
                    if self.scheduler.check_and_update_request_status(original_req):
                        self.kv_cache_manager.release_request(original_req.request_id)
                        logger.debug(
                            f"Released resources for finished request {req.request_id}, "
                            f"memory usage: {mx.get_active_memory() / 1024**3 :.3f} GB"
                        )
                        if not self.is_last_peer:
                            self.finished_batch.append(req)
                    else:
                        self.scheduler.enque_request(original_req)

                    # detokenize and send to http server
                    if self.tp_rank == 0:
                        req_dict = {
                            "prompt_tokens": len(req.input_ids),
                            "next_token_id": req.next_token_id,
                            "rid": req.request_id,
                        }
                        if original_req.status == RequestStatus.FINISHED_EOS:
                            req_dict["eos"] = True
                        if original_req.status == RequestStatus.FINISHED_MAX_LENGTH:
                            req_dict["length"] = True
                        if hasattr(self, "send_to_ipc_socket"):
                            self.send_to_ipc_socket.send_pyobj(req_dict)
                else:
                    raise TypeError(f"First peer received unexpected request type: {type(req)}")

        else:
            # Intermediate and Last peers receive IntermediateRequests from the previous peer.
            for req in requests:
                assert isinstance(
                    req, IntermediateRequest
                ), "Non-first peers must receive IntermediateRequests."
                if req.is_finished or req.hidden_states is None:
                    if self.enable_prefix_cache:
                        keys, values = self.kv_cache_manager.gather_kv_cache(req.request_id)
                        self.prefix_cache.cache_finished_request(req, keys, values)
                        self.prefix_cache.evict_request(req.request_id)

                    self.kv_cache_manager.release_request(req.request_id)
                    logger.debug(
                        f"Released resources for finished request {req.request_id}, "
                        f"kv cache manager has {self.kv_cache_manager.tokens_in_cache} tokens, "
                        f"memory usage: {mx.get_active_memory() / 1024**3 :.3f} GB"
                    )
                    self.scheduler.evict_request(req.request_id)
                    if not self.is_last_peer:
                        self.finished_batch.append(req)
                else:
                    # This is an active request, add it to the scheduler queue to be processed.
                    self.scheduler.enque_request(req)

    def process_batch(self, prepared_inputs: Dict[str, Any], return_decoded_tokens: bool = True):
        """Process a batch of requests in MLX."""
        # Run model and get updated cache
        # Note: Paged Attention writes KV cache in-place within the model (via reshape_and_cache).
        # The returned 'hidden_states' is what we need.
        # The returned cache tuple (_, _) is ignored/unused here.
        hidden_states = self.model_shard(
            h_or_tokens=prepared_inputs["h_or_tokens"],
            cache=prepared_inputs["cache"],
            mask=prepared_inputs.get("mask"),
            block_tables=prepared_inputs.get("block_tables"),
            context_lengths=prepared_inputs.get("context_lengths"),
            slot_mapping=prepared_inputs.get("slot_mapping"),
            indexer_cache=prepared_inputs.get("indexer_cache"),
        )

        logger.debug(
            f"Processing batch with {len(prepared_inputs['requests'])} requests, "
            f"request status: {prepared_inputs['requests'][0].status}, "
            f"hidden_states shape: {hidden_states.shape}"
        )

        lengths = mx.zeros((len(prepared_inputs["requests"]),), dtype=mx.int32)
        requests = prepared_inputs["requests"]
        for i, req in enumerate(requests):
            if req.is_prefill:
                lengths[i] = prepared_inputs.get("context_lengths")[i]
            elif req.is_decoding:
                lengths[i] = 1
            else:
                continue

        # Note: With PagedAttention, we don't need to explicitly update requests with new K/V
        # because they are written in-place to the global cache.
        # self.kv_cache_manager.update_requests(...) is REMOVED.

        # Update prefix cache (TODO: Adapt to PagedKV)
        if self.enable_prefix_cache:
            pass
            # for _, req in enumerate(requests):
            #    if req.is_prefill:
            #        keys, values = self.kv_cache_manager.gather_kv_cache(req.request_id)
            #        self.prefix_cache.cache_unfinished_request(req, keys, values)

        # Process last peer: need additional sampling + detokenization
        if return_decoded_tokens:
            sampling_info = SamplingBatchInfo.from_reqs(requests)
            return mx.array(
                self.model_shard.logits_to_tokens(hidden_states, lengths, sampling_info)
            )

        return hidden_states

    def _release_request(self, rid: str):
        """Release per-request resources in MLX."""
        try:
            if hasattr(self, "kv_cache_manager") and self.kv_cache_manager is not None:
                self.kv_cache_manager.release_request(rid)
        except Exception:
            pass

    def _gen_token_id_from_hidden(self, hidden_states) -> Tuple[int, Any]:
        """
        Inplace modifies hidden_states.
        Returns token_id, hidden_states
        """
        assert hidden_states.dtype == mx.uint32, "Single node must generate an output_id."
        next_token_id = int(hidden_states[0])
        hidden_states = hidden_states.astype(mx.int32)
        return next_token_id, hidden_states

    def _prepare_prefill_batch(self, batched_requests: List[Request]) -> Dict[str, Any]:
        """Prepares inputs for ShardedModel from a batch of prefill requests."""
        batch_size = len(batched_requests)
        if batch_size == 0:
            return None

        h_or_tokens_list = []
        block_tables_list = []
        context_lengths_list = []

        # TODO: Adapt Prefix Cache to PagedKV

        for req in batched_requests:
            assert req.is_prefill, f"Request {req.request_id} is not a prefill request."
            if self.is_first_peer:
                h_or_tokens_list.append(req.input_ids)
            else:
                h_or_tokens_list.append(req.hidden_states)

            # Allocate Paged KV blocks
            # For first peer and intermediate peers, we allocate based on prompt length
            success = self.kv_cache_manager.allocate_request(req.request_id, req.total_length)
            if not success:
                raise RuntimeError(f"OOM during prefill allocation for {req.request_id}")

            block_table = self.kv_cache_manager.get_block_table(req.request_id)
            block_tables_list.append(block_table)
            # For prefill, context length after this step will be total_length
            context_lengths_list.append(req.total_length)

        if self.is_first_peer:
            padded_inputs, padding_mask = pad_inputs(
                self.pad_token_id, h_or_tokens_list, self.dtype
            )
        else:
            padded_inputs, padding_mask = pad_inputs(0, h_or_tokens_list, self.dtype)

        # Generate slot_mapping (Batch * MaxLen) for prefill
        max_len = padded_inputs.shape[1]
        slot_mapping_flat = []

        for i, req in enumerate(batched_requests):
            block_table = block_tables_list[i]
            length = req.total_length

            for seq_idx in range(max_len):
                if seq_idx < length:
                    # Valid token
                    block_idx = seq_idx // self.kv_cache_manager.block_size
                    block_offset = seq_idx % self.kv_cache_manager.block_size
                    physical_block = block_table[block_idx]
                    slot = physical_block * self.kv_cache_manager.block_size + block_offset
                    slot_mapping_flat.append(slot)
                else:
                    # Padding token
                    # Map to -1. The kernel should ignore this.
                    slot_mapping_flat.append(-1)

        slot_mapping_tensor = mx.array(slot_mapping_flat, dtype=mx.int64)

        # Pad block tables
        max_blocks = max(len(bt) for bt in block_tables_list)
        padded_block_tables = []
        for bt in block_tables_list:
            padded_block_tables.append(bt + [0] * (max_blocks - len(bt)))

        block_tables_tensor = mx.array(padded_block_tables, dtype=mx.int32)
        context_lengths_tensor = mx.array(context_lengths_list, dtype=mx.int32)

        # Create mask for standard attention (used during Prefill computation)
        causal_mask = create_causal_mask(padded_inputs.shape[1], padded_inputs.shape[1], self.dtype)
        mask = combine_padding_and_causal_masks(padding_mask, causal_mask, self.dtype)

        ret = {
            "h_or_tokens": padded_inputs,
            "cache": self.kv_cache_manager.get_cache(),
            "indexer_cache": self.kv_cache_manager.get_indexer_cache(),
            "mask": mask,
            "requests": batched_requests,
            "block_tables": block_tables_tensor,
            "context_lengths": context_lengths_tensor,
            "slot_mapping": slot_mapping_tensor,
            "state_cache": None,
        }
        logger.debug(f"Prepared MLX prefill batch (size={batch_size})")
        return ret

    def _prepare_decode_batch(self, batched_requests: List[Request]) -> Optional[Dict[str, Any]]:
        """Prepares inputs for ShardedModel from a batch of decode requests."""
        batch_size = len(batched_requests)
        if batch_size == 0:
            return None

        h_or_tokens_list = []
        block_tables_list = []
        context_lengths_list = []

        for req in batched_requests:
            assert req.is_decoding, f"Request {req.request_id} is not a decode request."

            if self.is_first_peer:
                # First peer input is the last generated token
                h_or_tokens_list.append([req.output_ids[-1]])
            else:
                h_or_tokens_list.append(req.hidden_states)

            # TODO: Prefix cache update

            # Allocate slot for new token
            success = self.kv_cache_manager.append_slot(req.request_id)
            if not success:
                raise RuntimeError(f"OOM during decode for {req.request_id}")

            block_table = self.kv_cache_manager.get_block_table(req.request_id)
            block_tables_list.append(block_table)
            context_lengths_list.append(self.kv_cache_manager.get_context_length(req.request_id))

        if isinstance(h_or_tokens_list[0], list):
            # First peer case: h_or_tokens_list is list of list of ints [[token_id], ...]
            padded_inputs = mx.array(h_or_tokens_list, dtype=mx.int32)  # (Batch, 1)
        else:
            # Intermediate peer case: h_or_tokens_ list is list of mx.arrays (1, D)
            padded_inputs = mx.concatenate(h_or_tokens_list, axis=0)  # (Batch, D)
            padded_inputs = padded_inputs.reshape(batch_size, 1, -1)  # (Batch, 1, D)

        # Pad block tables
        max_blocks = max(len(bt) for bt in block_tables_list)
        padded_block_tables = []
        for bt in block_tables_list:
            padded_block_tables.append(bt + [0] * (max_blocks - len(bt)))

        block_tables_tensor = mx.array(padded_block_tables, dtype=mx.int32)
        context_lengths_tensor = mx.array(context_lengths_list, dtype=mx.int32)

        ret = {
            "h_or_tokens": padded_inputs,
            "cache": self.kv_cache_manager.get_cache(),
            "indexer_cache": self.kv_cache_manager.get_indexer_cache(),
            "mask": None,
            "requests": batched_requests,
            "block_tables": block_tables_tensor,
            "context_lengths": context_lengths_tensor,
            "slot_mapping": None,
            "state_cache": None,
        }
        logger.debug(f"Prepared MLX decode batch (size={batch_size})")
        return ret
