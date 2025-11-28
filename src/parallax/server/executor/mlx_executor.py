"""
MLX-LM backend implementation of high level executor
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

from parallax.server.executor.base_executor import BaseExecutor
from parallax.server.kv_cache import KVCacheManager
from parallax.server.radix_cache import RadixCache
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
    get_infinite_value_by_dtype,
    pad_inputs,
    pad_prefix_caches,
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

        logger.debug(
            "Initializing KVCacheManager (mlx) with block_size=%d, layers=%d",
            kv_block_size,
            self.num_shard_layers,
        )
        self.kv_cache_manager = KVCacheManager(
            block_size=kv_block_size,
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
            num_layers=self.num_shard_layers,
            dtype=self.dtype,
            cache_memory_fraction=kv_cache_memory_fraction,
            conv_dim=conv_dim if conv_dim and conv_dim > 0 else None,
            conv_kernel_size=linear_conv_kernel_dim,
            linear_k_dim=linear_key_head_dim,
            linear_v_dim=linear_value_head_dim,
            linear_num_k_heads=linear_num_key_heads,
            linear_num_v_heads=linear_num_value_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            max_num_tokens=max_tokens_in_kv_pool,
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
        self.prefix_cache = RadixCache(
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
            num_layers=self.num_shard_layers,
            dtype=self.dtype,
            page_size=1,
        )

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
                            f"kv cache manager has {self.kv_cache_manager.tokens_in_cache} tokens, "
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
        if self.using_state_cache:
            hidden_states, (k_caches, v_caches, states0, states1) = self.model_shard(
                h_or_tokens=prepared_inputs["h_or_tokens"],
                cache=prepared_inputs["cache"],
                lengths=prepared_inputs["lengths"],
                mask=prepared_inputs["mask"],
                state_cache=prepared_inputs["state_cache"],
                using_state_cache=self.using_state_cache,
            )
        else:
            hidden_states, (k_caches, v_caches) = self.model_shard(
                h_or_tokens=prepared_inputs["h_or_tokens"],
                cache=prepared_inputs["cache"],
                lengths=prepared_inputs["lengths"],
                mask=prepared_inputs["mask"],
                using_state_cache=self.using_state_cache,
            )
            states0, states1 = [None for _ in range(len(k_caches))], [
                None for _ in range(len(k_caches))
            ]
        # k_caches shape: (num_layers, B, num_kv_heads, L_padded, head_dim)
        logger.debug(
            f"Processing batch with {len(prepared_inputs['requests'])} requests, "
            f"request status: {prepared_inputs['requests'][0].status}, "
            f"hidden_states shape: {hidden_states.shape}, "
            f"k_caches shape: {k_caches.shape}, "
            f"v_caches shape: {v_caches.shape}"
        )

        lengths = mx.zeros((len(prepared_inputs["requests"]),), dtype=mx.int32)
        requests = prepared_inputs["requests"]
        for i, req in enumerate(requests):
            if req.is_prefill:
                lengths[i] = prepared_inputs["lengths"][i]
            elif req.is_decoding:
                lengths[i] = 1
            else:
                continue
        self.kv_cache_manager.update_requests(
            requests, k_caches, v_caches, lengths, states0, states1
        )

        # Update prefix cache.
        if self.enable_prefix_cache:
            for _, req in enumerate(requests):
                if req.is_prefill:
                    keys, values = self.kv_cache_manager.gather_kv_cache(req.request_id)
                    self.prefix_cache.cache_unfinished_request(req, keys, values)

        # Process last peer: need additional sampling + detokenization
        if return_decoded_tokens:
            sampling_info = SamplingBatchInfo.from_reqs(requests)
            return mx.array(
                self.model_shard.logits_to_tokens(hidden_states, lengths, sampling_info)
            )
        logger.debug(
            f"Processed batch (device={self.device}, return_tokens={return_decoded_tokens})"
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

        h = []
        lengths = []
        actual_lengths = []
        k_caches = []
        v_caches = []
        matched_prefix = False
        for req in batched_requests:
            assert req.is_prefill, f"Request {req.request_id} is not a prefill request."
            if self.is_first_peer:
                assert hasattr(
                    req, "input_ids"
                ), f"Request {req.request_id} should has attribute input_ids in FirstPeer."
                h.append(req.input_ids)
            else:
                assert isinstance(
                    req, IntermediateRequest
                ), f"Request {req.request_id} should not be in FirstPeer."
                h.append(req.hidden_states)
            lengths.append(req.total_length)

            if self.enable_prefix_cache:
                self.prefix_cache.update_req_to_token(req.request_id, req.input_ids)
                value, node = self.prefix_cache.match_prefix(req.input_ids[:-1])
                if value:
                    kv = self.prefix_cache.fetch_kv_cache(node)
                    k_caches.append(kv[0])
                    v_caches.append(kv[1])
                    assert len(value) == (
                        kv[0].shape[2]
                    ), f"Mached prefix length{len(value)} mismatches kv cache length {kv[0].shape[2]}."
                    matched_prefix = True
                    self.kv_cache_manager.add_matched_prefix_request(req, kv[0], kv[1], len(value))
                    actual_lengths.append(req.total_length - len(value))
                else:
                    k_caches.append(
                        mx.zeros(
                            [
                                self.prefix_cache.num_layers,
                                self.prefix_cache.num_kv_heads,
                                0,
                                self.prefix_cache.head_dim,
                            ],
                            dtype=self.dtype,
                        )
                    )
                    v_caches.append(
                        mx.zeros(
                            [
                                self.prefix_cache.num_layers,
                                self.prefix_cache.num_kv_heads,
                                0,
                                self.prefix_cache.head_dim,
                            ],
                            dtype=self.dtype,
                        )
                    )
                    actual_lengths.append(req.total_length)

        if self.is_first_peer:
            padded_inputs, padding_mask = pad_inputs(self.pad_token_id, h, self.dtype)
        else:
            padded_inputs, padding_mask = pad_inputs(0, h, self.dtype)

        k_batched = None
        v_batched = None
        if matched_prefix:
            k_batched, k_padding_mask = pad_prefix_caches(k_caches, lengths, self.dtype)
            v_batched, _ = pad_prefix_caches(v_caches, lengths, self.dtype)
            padding_mask = k_padding_mask
            causal_mask = create_causal_mask(padded_inputs.shape[1], max(lengths), self.dtype)
            mask = combine_padding_and_causal_masks(padding_mask, causal_mask, self.dtype)
        else:
            causal_mask = create_causal_mask(
                padded_inputs.shape[1], padded_inputs.shape[1], self.dtype
            )
            mask = combine_padding_and_causal_masks(padding_mask, causal_mask, self.dtype)

        ret = {
            "h_or_tokens": padded_inputs,
            "cache": (k_batched, v_batched) if matched_prefix else None,
            "lengths": mx.array(actual_lengths) if matched_prefix else mx.array(lengths),
            "mask": mask,
            "requests": batched_requests,
            "state_cache": None,
        }
        logger.debug(
            f"Prepared MLX prefill batch (size={batch_size}, matched_prefix={matched_prefix})"
        )
        return ret

    def _prepare_decode_batch(self, batched_requests: List[Request]) -> Optional[Dict[str, Any]]:
        """Prepares inputs for ShardedModel from a batch of decode requests."""
        batch_size = len(batched_requests)
        if batch_size == 0:
            return None

        h_list = []
        cache_lengths = []
        kv_cache_list = []

        for req in batched_requests:
            assert req.is_decoding, f"Request {req.request_id} is not a decode request."

            if self.is_first_peer:
                assert isinstance(req, InitialRequest)
                # First peer input is the last generated token
                h_list.append([req.output_ids[-1]])
            else:
                assert isinstance(req, IntermediateRequest)
                assert req.hidden_states is not None and req.hidden_states.shape[0] == 1
                h_list.append(req.hidden_states)
            if self.enable_prefix_cache:
                self.prefix_cache.update_req_to_token(req.request_id, list([req.next_token_id]))

            num_tokens_in_cache = self.kv_cache_manager.request_length(req.request_id)
            cache_lengths.append(num_tokens_in_cache)
            kv_cache = self.kv_cache_manager.gather_kv_cache(req.request_id)
            kv_cache_list.append(kv_cache)

        padded_inputs, _ = pad_inputs(0, h_list, self.dtype)

        if not kv_cache_list:
            raise ValueError("No KV cache found for request.")

        k_caches = [kv[0] for kv in kv_cache_list]
        v_caches = [kv[1] for kv in kv_cache_list]
        states0 = [kv[2] for kv in kv_cache_list]
        states1 = [kv[3] for kv in kv_cache_list]

        k_batched, k_padding_mask = pad_inputs(0, k_caches, self.dtype)
        v_batched, _ = pad_inputs(0, v_caches, self.dtype)

        # The mask from padding K is for the PAST tokens. It has shape (B, 1, 1, source_len_padded).
        # We need to add a '1' for the CURRENT token so the final mask can be broadcast
        # to the attention weights of shape (B, n_heads, 1, source_len_padded + 1).
        ones_for_current_token = mx.ones((k_padding_mask.shape[0], 1, 1, 1), dtype=self.dtype)
        final_padding_mask = mx.concatenate([k_padding_mask, ones_for_current_token], axis=3)
        inf_value = get_infinite_value_by_dtype(self.dtype)
        attention_mask = (1.0 - final_padding_mask) * -inf_value

        model_lengths = mx.array([kv[0].shape[2] for kv in kv_cache_list])

        if self.using_state_cache:
            states0 = mx.stack(states0, 0)
            states1 = mx.stack(states1, 0)

        ret = {
            "h_or_tokens": padded_inputs,
            "cache": (k_batched, v_batched),
            "lengths": model_lengths,
            "mask": attention_mask,
            "requests": batched_requests,
            "state_cache": (states0, states1),
        }
        logger.debug(f"Prepared MLX decode batch (size={batch_size})")
        return ret
