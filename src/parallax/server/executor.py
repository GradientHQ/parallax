"""
High-level executor for managing model shards, scheduler, and cache pool on each Peer.

Executor handles
1. Loading model shards from the repository;
2. Instantiate scheduler, kv cache manager;
3. Handles tokenization / detokenization if needed;
4. Keep listening to RPC to get requests, feed these to scheduler's request pool;
5. Get batched requests from the scheduler,
    - prepare the MLX tensor input
    - rebuild KV cache
    - feed to model runner;
    For now we process prefill and decode requests separately.
    Later when we have Ragged Paged Flash Attention kernel, we can process both in one batch.
6. Run model forward, our model will returned updated caches,
    kv cache manager will handle updating caches per layer;
7. Get the hidden-states from the model execution.
"""

import argparse
import time
from typing import Any, Dict, List, Optional

import mlx.core as mx
import torch
import zmq
from mlx_lm.server import convert_chat, process_message_content

from parallax.p2p.message_util import (
    abort_request_to_proto,
    proto_to_abort_request,
    proto_to_request,
    request_to_proto,
)
from parallax.p2p.proto import forward_pb2
from parallax.server.kv_cache import KVCacheManager
from parallax.server.metrics import update_metrics
from parallax.server.radix_cache import RadixCache
from parallax.server.request import (
    InitialRequest,
    IntermediateRequest,
    Request,
    RequestStatus,
)
from parallax.server.sampling.sampler import SamplingBatchInfo
from parallax.server.sampling.sampling_params import SamplingParams
from parallax.server.scheduler import Scheduler
from parallax.server.shard_loader import MLXModelLoader
from parallax.utils.utils import (
    combine_padding_and_causal_masks,
    create_causal_mask,
    get_current_device,
    get_device_dtype,
    get_infinite_value_by_dtype,
    get_zmq_socket,
    pad_inputs,
    pad_prefix_caches,
)
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class Executor:
    """High-level executor for managing model shards, scheduler, and cache pool on each Peer."""

    def __init__(
        self,
        # Model Configs
        model_repo: str,
        start_layer: int,
        end_layer: int,
        dtype: str = "float16",
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
        # GPU/SGLang Specialized Configs
        attention_backend: Optional[str] = "torch_native",
        moe_runner_backend: Optional[str] = "auto",
    ):
        # Backend
        self.device = get_current_device()
        logger.debug(f"Executor initializing on device: {self.device}")

        # Sharded Model
        if self.device == "cuda":
            from sglang.srt.managers.schedule_batch import ScheduleBatch

            from parallax.sglang.model_runner import initialize_sgl_model_runner

            logger.debug(
                f"Initializing CUDA model runner for repo={model_repo}, layers=[{start_layer}, {end_layer})"
            )
            self.model_runner, self.config, self.tokenizer = initialize_sgl_model_runner(
                model_repo,
                start_layer,
                end_layer,
                kv_cache_memory_fraction,
                attention_backend,
                kv_block_size,
                moe_runner_backend,
            )
            logger.debug(
                f"CUDA model runner initialized. num_layers={self.config.get('num_hidden_layers')}"
            )
            # SGL KV Cache Manager is already initialized in ScheduleBatch
            # TODO: Replace ScheduleBatch to Parallax inflight batch
            self.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
            self.cur_batch = None
        else:
            logger.debug(
                f"Initializing MLX sharded model loader for repo={model_repo}, layers=[{start_layer}, {end_layer})"
            )
            self.shard_loader = MLXModelLoader(
                model_repo, start_layer=start_layer, end_layer=end_layer
            )
            t0 = time.time()
            self.model_shard, self.config, self.tokenizer = self.shard_loader.load()
            logger.debug(
                f"MLX sharded model loaded in {(time.time() - t0) * 1000:.1f} ms; num_layers={self.config.get('num_hidden_layers')}"
            )

        # for window attention need to calculate causal mask size
        self.finished_batch = []
        self.start_layer = start_layer
        self.end_layer = end_layer

        self.is_first_peer = start_layer == 0
        self.is_last_peer = end_layer == self.config.get("num_hidden_layers")
        self.num_shard_layers = end_layer - start_layer

        # Metrics throttling for per-layer latency updates
        self.layer_latency_update_every = int(max(1, layer_latency_update_every))
        self._decode_steps_since_metric = self.layer_latency_update_every

        self.dtype = get_device_dtype(dtype, self.device)
        logger.debug(
            f"Executor dtype set to {dtype} (resolved={self.dtype}); shard_layers={self.num_shard_layers}"
        )
        self.num_key_value_heads = self.config.get("num_key_value_heads")
        self.head_dim = self.config.get("head_dim") or self.config.get(
            "hidden_size"
        ) // self.config.get("num_attention_heads")
        self.qk_nope_head_dim = self.config.get("qk_nope_head_dim", None)
        self.qk_rope_head_dim = self.config.get("qk_rope_head_dim", None)
        self.enable_prefix_cache = enable_prefix_cache
        self.linear_key_head_dim = self.config.get("linear_key_head_dim", None)
        self.linear_value_head_dim = self.config.get("linear_value_head_dim", None)
        self.linear_conv_kernel_dim = self.config.get("linear_conv_kernel_dim", None)
        self.linear_num_key_heads = self.config.get("linear_num_key_heads", None)
        self.linear_num_value_heads = self.config.get("linear_num_value_heads", None)
        self.key_dim, self.value_dim, self.conv_dim = None, None, None
        if self.linear_key_head_dim is not None and self.linear_num_key_heads is not None:
            self.key_dim = self.linear_key_head_dim * self.linear_num_key_heads
        if self.linear_value_head_dim is not None and self.linear_num_value_heads is not None:
            self.value_dim = self.linear_value_head_dim * self.linear_num_value_heads
        if self.key_dim is not None and self.value_dim is not None:
            self.conv_dim = self.key_dim * 2 + self.value_dim
        self.using_state_cache = (
            self.linear_conv_kernel_dim is not None and self.conv_dim is not None
        )
        # logger.debug(
        #     f"Model config: n_kv_heads={self.num_key_value_heads}, head_dim={self.head_dim}, tokenizer_pad={self.tokenizer.pad_token_id}"
        # )

        if self.tokenizer.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id

        self.eos_token_id = self.tokenizer.eos_token_id
        if isinstance(self.eos_token_id, list):
            self.eos_token_id = self.eos_token_id[0]

        if self.device == "mlx":
            # Other setup for MAC
            logger.debug(
                "Initializing KVCacheManager (mlx) with block_size=%d, layers=%d",
                kv_block_size,
                self.num_shard_layers,
            )
            self.kv_cache_manager = KVCacheManager(
                block_size=kv_block_size,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                num_layers=self.num_shard_layers,
                dtype=self.dtype,
                cache_memory_fraction=kv_cache_memory_fraction,
                conv_dim=self.conv_dim if self.conv_dim and self.conv_dim > 0 else None,
                conv_kernel_size=self.linear_conv_kernel_dim,
                linear_k_dim=self.linear_key_head_dim,
                linear_v_dim=self.linear_value_head_dim,
                linear_num_k_heads=self.linear_num_key_heads,
                linear_num_v_heads=self.linear_num_value_heads,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                max_num_tokens=max_tokens_in_kv_pool,
            )
            mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
            logger.debug(
                f"KVCacheManager ready; wired_limit set; prefix_cache={'on' if self.enable_prefix_cache else 'off'}"
            )
            self.kv_cache_manager.max_num_tokens

        # Scheduler: derive final max_batch_size with KV constraints
        # Remove this for now as it's not working on gpu devices
        # max_batch_size = compute_max_batch_size(
        #     requested_max_batch_size=max_batch_size,
        #     max_sequence_len=max_sequence_length,
        #     device=self.device,
        #     kv_cache_memory_fraction=kv_cache_memory_fraction,
        #     num_shard_layers=self.num_shard_layers,
        #     num_key_value_heads=self.num_key_value_heads,
        #     head_dim=self.head_dim,
        #     dtype=self.dtype,
        # )

        self.scheduler = Scheduler(
            max_batch_size=max_batch_size,
            max_num_tokens_per_batch=max_num_tokens_per_batch,
            prefill_priority=prefill_priority,
            scheduler_wait_ms=scheduler_wait_ms,
            micro_batch_ratio=micro_batch_ratio,
            is_first_peer=self.is_first_peer,
            tokenizer=self.tokenizer,
            kv_cache_manager=self.kv_cache_manager if self.device == "mlx" else None,
            request_timeout_s=request_timeout_s,
        )
        logger.debug(
            f"Scheduler initialized (max_batch_size={max_batch_size}, max_tokens={max_num_tokens_per_batch}, wait_ms={scheduler_wait_ms})"
        )

        # Prefix Cache Manager
        self.prefix_cache = RadixCache(
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            num_layers=self.num_shard_layers,
            dtype=self.dtype,
            page_size=1,
        )

        # Communication Related
        self.zmq_context = zmq.Context()
        if recv_from_peer_addr:
            self.recv_from_peer_socket = get_zmq_socket(
                self.zmq_context, zmq.PULL, recv_from_peer_addr, bind=False
            )
        if send_to_peer_addr:
            self.send_to_peer_socket = get_zmq_socket(
                self.zmq_context, zmq.PUSH, send_to_peer_addr, bind=False
            )
        if executor_input_ipc_addr:
            self.recv_from_ipc_socket = get_zmq_socket(
                self.zmq_context, zmq.PULL, executor_input_ipc_addr, bind=False
            )
        if executor_output_ipc_addr:
            self.send_to_ipc_socket = get_zmq_socket(
                self.zmq_context, zmq.PUSH, executor_output_ipc_addr, bind=False
            )

    @classmethod
    def create_from_args(cls, args: argparse.Namespace):
        """Create executor from command line arguments."""
        return cls(**create_executor_config(args))

    def recv_requests_from_http(self) -> List[Request]:
        """Receives requests from http frontend"""
        recv_reqs = []
        while True:
            try:
                raw_request = self.recv_from_ipc_socket.recv_pyobj(zmq.NOBLOCK)

                # Check if this is an abort request
                if isinstance(raw_request, dict) and raw_request.get("type") == "abort":
                    logger.debug(
                        f"Received abort request from HTTP for request ID: {raw_request.get('rid')}"
                    )
                    self.scheduler.cancel_request(raw_request.get("rid"))
                else:
                    # Normal request processing - do tokenization and form InitialRequest
                    req = self._handle_raw_request(raw_request)
                    recv_reqs.append(req)
            except zmq.ZMQError:
                break
            except Exception as e:
                logger.exception(f"Error receiving http request: {e}")
        if recv_reqs:
            logger.debug(f"Received {len(recv_reqs)} HTTP requests")
        return recv_reqs

    def recv_requests_from_peer(self) -> List[Request]:
        """Receives requests from the RPC server."""
        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_peer_socket.recv_multipart(zmq.NOBLOCK)
                assert len(recv_req) == 2, f"Received invalid request: {recv_req}"
                if recv_req[0] == b"forward":
                    # Create a new ForwardRequest instance and parse from bytes
                    forward_request = forward_pb2.ForwardRequest()
                    forward_request.ParseFromString(recv_req[1])
                    recv_req = proto_to_request(forward_request, self.device)

                    # Convert hidden_states dtype if necessary
                    if recv_req is not None and len(recv_req) > 0:
                        for req in recv_req:
                            if req.hidden_states is not None:
                                if req.hidden_states.dtype != self.dtype:
                                    logger.debug(
                                        f"Converting hidden_states dtype from {req.hidden_states.dtype} to {self.dtype} for request {req.request_id}"
                                    )
                                    if self.device == "cuda":
                                        req.hidden_states = req.hidden_states.to(self.dtype)
                                    elif self.device == "mlx":
                                        req.hidden_states = req.hidden_states.astype(self.dtype)
                                    else:
                                        raise ValueError(f"Unsupported device type: {self.device}")

                    # Move current position for first peer
                    if self.is_first_peer:
                        for req in recv_req:
                            req.current_position += 1
                    recv_reqs.extend(recv_req)
                elif recv_req[0] == b"abort":
                    abort_request = forward_pb2.AbortRequest()
                    abort_request.ParseFromString(recv_req[1])
                    recv_req = proto_to_abort_request(abort_request)
                    recv_reqs.extend(recv_req)
                else:
                    raise ValueError(f"Unknown request type: {recv_req[0]}")
                # First peer is responsible for tokenization
                # if self.is_first_peer and isinstance(recv_req, InitialRequest):
                #     recv_req.input_ids = self.tokenizer.encode(recv_req.prompt)
                #     recv_req.prompt_len = len(recv_req.input_ids)
                #     recv_req.max_total_length = min(
                #         recv_req.max_total_length, recv_req.prompt_len + recv_req.max_new_tokens
                #     )

            except zmq.ZMQError:
                break
            except Exception as e:
                logger.exception(f"Error receiving or deserializing request: {e}")
        if recv_reqs:
            logger.debug(f"Received {len(recv_reqs)} peer requests")
        return recv_reqs

    def _prepare_cuda_prefill_batch(self, batched_requests: List[Request]) -> Dict[str, Any]:
        """
        Prepares inputs for SGLang model runner from a batch of prefill requests.
        Returns: SGLang ScheduleBatch
        """
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        from parallax.sglang.batch_info import form_sgl_batch_prefill

        batch_size = len(batched_requests)
        if batch_size == 0:
            return None
        schedule_batch, forward_batch = form_sgl_batch_prefill(batched_requests, self.model_runner)
        self.cur_batch = schedule_batch

        pp_proxy_tensors = None
        if not self.is_first_peer:
            hidden_states = torch.cat(
                [
                    (
                        req.hidden_states
                        if req.hidden_states.ndim == 2
                        else req.hidden_states.unsqueeze(0)
                    )
                    for req in batched_requests
                ],
                dim=0,
            )
            residual = torch.zeros(
                hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device
            )
            pp_proxy_tensors = PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
            logger.debug(f"PP Proxy: hidden_states shape: {hidden_states.shape}")
        lengths = []
        for req in batched_requests:
            lengths.append(req.total_length)
        ret = {
            "forward_batch": forward_batch,
            "pp_proxy_tensors": pp_proxy_tensors,
            "lengths": torch.tensor(lengths, device=self.device),
            "requests": batched_requests,
        }
        logger.debug(f"Prepared CUDA prefill batch (size={batch_size})")
        return ret

    def _prepare_cuda_decode_batch(self, batched_requests: List[Request]) -> Dict[str, Any]:
        """
        Prepares inputs for SGLang model runner from a batch of decode requests.
        Returns: SGLang ScheduleBatch
        """
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        from parallax.sglang.batch_info import form_sgl_batch_decode

        batch_size = len(batched_requests)
        if batch_size == 0:
            return None

        lengths = []
        for req in batched_requests:
            lengths.append(req.total_length)
        forward_batch = form_sgl_batch_decode(
            batched_requests,
            self.model_runner,
            self.running_batch,
            self.is_first_peer,
        )
        pp_proxy_tensors = None
        if not self.is_first_peer:
            hidden_states = torch.cat(
                [
                    (
                        req.hidden_states
                        if req.hidden_states.ndim == 2
                        else req.hidden_states.unsqueeze(0)
                    )
                    for req in batched_requests
                ],
                dim=0,
            )
            residual = torch.zeros(
                hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device
            )
            pp_proxy_tensors = PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
            logger.debug(f"PP Proxy: hidden_states shape: {hidden_states.shape}")
        ret = {
            "forward_batch": forward_batch,
            "pp_proxy_tensors": pp_proxy_tensors,
            "lengths": torch.tensor(lengths, device=self.device),
            "requests": batched_requests,
        }
        logger.debug(f"Prepared CUDA decode batch (size={batch_size})")
        return ret

    def _prepare_mlx_prefill_batch(self, batched_requests: List[Request]) -> Dict[str, Any]:
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

    def _prepare_mlx_decode_batch(
        self, batched_requests: List[Request]
    ) -> Optional[Dict[str, Any]]:
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

    def _prepare_batch_inputs(self, batched_requests: List[Request]) -> Optional[Dict[str, Any]]:
        """Prepares inputs for ShardedModel from a batch of requests.
        Args:
            batched_requests: A list of requests to prepare inputs for.

        Returns:
            A dictionary containing the prepared inputs for the ShardedModel.
            The dictionary contains "prefill_batch" and "decode_batch",
            with the prepared inputs for the corresponding request type.

            For now we process prefill and decode requests separately.
            Later when we have Ragged Paged Flash Attention kernel,
            we can process both in one batch.
        """
        if len(batched_requests) == 0:
            return None

        prefill_reqs: List[Request] = []
        decode_reqs: List[Request] = []
        for req in batched_requests:
            if req.is_prefill:
                prefill_reqs.append(req)
            elif req.is_decoding:
                decode_reqs.append(req)
        if self.device == "cuda":
            prefill_batch = self._prepare_cuda_prefill_batch(prefill_reqs)
            decode_batch = self._prepare_cuda_decode_batch(decode_reqs)
        else:
            prefill_batch = self._prepare_mlx_prefill_batch(prefill_reqs)
            decode_batch = self._prepare_mlx_decode_batch(decode_reqs)
        if prefill_batch is None and decode_batch is None:
            return None
        if prefill_batch is not None:
            logger.debug(f"Prepared prefill batch with {len(prefill_batch['requests'])} requests.")
        if decode_batch is not None:
            logger.debug(f"Prepared decode batch with {len(decode_batch['requests'])} requests.")
        return {
            "prefill_batch": prefill_batch,
            "decode_batch": decode_batch,
        }

    def _handle_raw_request(self, raw_request: Dict):
        assert "messages" in raw_request, "Request did not contain messages"

        rid = raw_request["rid"]
        if self.tokenizer.chat_template:
            messages = raw_request["messages"]
            process_message_content(messages)
            chat_template_kwargs = raw_request.get("chat_template_kwargs", {})
            # check extra_body for backward compatibility
            if "extra_body" in raw_request and "chat_template_kwargs" in raw_request["extra_body"]:
                chat_template_kwargs.update(raw_request["extra_body"]["chat_template_kwargs"])

            prompt = self.tokenizer.apply_chat_template(
                messages,
                raw_request.get("tools") or None,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        else:
            prompt = convert_chat(raw_request["messages"], raw_request.get("role_mapping"))
            prompt = self.tokenizer.encode(prompt)

        max_new_tokens = raw_request.get("max_tokens")
        if max_new_tokens is None:
            max_new_tokens = 2048
        max_total_length = len(prompt) + max_new_tokens

        raw_sampling_params = raw_request.get("sampling_params")
        if raw_sampling_params is None:
            sampling_params = SamplingParams()
        else:
            # TODO: Support more sampling params
            sampling_params = SamplingParams()
            if "temperature" in raw_sampling_params:
                sampling_params.temperature = raw_sampling_params["temperature"]
            if "top_k" in raw_sampling_params:
                sampling_params.top_k = raw_sampling_params["top_k"]
            if "top_p" in raw_sampling_params:
                sampling_params.top_p = raw_sampling_params["top_p"]

        req = InitialRequest(
            request_id=rid,
            output_ids=None,
            input_ids=prompt,
            sampling_params=sampling_params,
            max_new_tokens=max_new_tokens,
            max_total_length=max_total_length,
        )
        if "routing_table" in raw_request:
            req.routing_table = raw_request["routing_table"]
        return req

    def _handle_cuda_input_requests(self, requests: List[Request]):
        """
        Cuda specialized handle function.
        The main difference is to remove all the kv cache operations.
        """
        from parallax.sglang.batch_info import release_cuda_request

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
                            "But no corresponding request found in scheduler (CUDA). "
                            "It might have been cancelled or finished."
                        )
                        continue

                    assert req.next_token_id is not None
                    original_req.commit_new_token(req.next_token_id)
                    if len(req.routing_table) > 0:
                        original_req.routing_table = req.routing_table

                    # Check for termination.
                    if self.scheduler.check_and_update_request_status(original_req):
                        logger.debug(f"Releasing resources for finished request {req.request_id}")
                        release_cuda_request(self.running_batch, req.request_id)
                        if not self.is_last_peer:
                            self.finished_batch.append(req)
                    else:
                        self.scheduler.enque_request(original_req)

                    # detokenize and send to http server
                    req_dict = {
                        "prompt_tokens": len(req.input_ids),
                        "next_token_id": req.next_token_id,
                        "rid": req.request_id,
                    }
                    if req.next_token_id == self.tokenizer.eos_token_id:
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
                    self.scheduler.evict_request(req.request_id)
                    release_cuda_request(self.running_batch, req.request_id)
                    if not self.is_last_peer:
                        self.finished_batch.append(req)
                else:
                    # This is an active request, add it to the scheduler queue to be processed.
                    self.scheduler.enque_request(req)

    def _handle_input_requests(self, requests: List[Request]):
        """Update requests states and status in scheduler and cache manager."""
        if len(requests) > 0:
            logger.debug(f"Handling {len(requests)} requests.")
        if not requests:
            return

        if self.device == "cuda":
            self._handle_cuda_input_requests(requests)
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
                    req_dict = {
                        "prompt_tokens": len(req.input_ids),
                        "next_token_id": req.next_token_id,
                        "rid": req.request_id,
                    }
                    if req.next_token_id == self.tokenizer.eos_token_id:
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

    def _prepare_next_single_request(self, request: Request, hidden_states: Any) -> Request:
        """Handle request state changes both inter and intra peers.

        This function prepares the request object to be sent to the *next* peer in the
        pipeline, or back to the first peer if this is the last peer.

        Args:
            request: The request that was just processed by this peer.
            hidden_states: The output hidden_states/output_ids from the model for this request.

        Returns:
            A new Request object ready to be sent to the next destination.
        """
        # This peer is the last peer or a single node.
        if self.is_last_peer and self.is_first_peer:
            assert isinstance(
                request, (InitialRequest, IntermediateRequest)
            ), "Invalid request type for decoding."
            if self.device == "cuda":
                assert hidden_states.dtype in (
                    torch.int64,
                    torch.int32,
                ), "Single node must generate an output_id."
                next_token_id = int(hidden_states[0])
            else:
                assert hidden_states.dtype == mx.uint32, "Single node must generate an output_id."
                next_token_id = int(hidden_states[0])
                hidden_states = hidden_states.astype(mx.int32)
            return IntermediateRequest(
                request_id=request.request_id,
                status=RequestStatus.DECODING,
                current_position=request.total_length + 1,
                input_ids=request.input_ids,
                hidden_states=hidden_states,
                next_token_id=next_token_id,
                routing_table=request.routing_table,
            )
        if self.is_last_peer:
            # Last peer decodes a token and sends it back to the first peer.
            # The token is wrapped in an IntermediateRequest.
            assert isinstance(
                request, IntermediateRequest
            ), "Last peer must receive an IntermediateRequest."
            if self.device == "cuda":
                assert hidden_states.dtype in (
                    torch.int64,
                    torch.int32,
                ), "Last peer must generate an output_id."
                next_token_id = int(hidden_states[0])
            else:
                assert hidden_states.dtype == mx.uint32, "Last peer must generate an output_id."
                next_token_id = int(hidden_states[0])
                # Compatible to GPU tensor load format
                hidden_states = hidden_states.astype(mx.int32)
            return IntermediateRequest(
                request_id=request.request_id,
                status=RequestStatus.DECODING,  # Last peer always changes status to DECODING
                current_position=request.total_length,
                input_ids=request.input_ids,
                hidden_states=hidden_states,
                next_token_id=next_token_id,
                routing_table=request.routing_table,
            )
        # This peer is the first or an intermediate peer.
        if self.is_first_peer:
            assert isinstance(request, InitialRequest), "First peer must process an InitialRequest."
            if request.is_finished:
                hidden_states = None
            return IntermediateRequest.from_initial_request(request, hidden_states=hidden_states)
        assert isinstance(
            request, IntermediateRequest
        ), "Intermediate peer must process an IntermediateRequest."
        return IntermediateRequest.from_intermediate_request(request, hidden_states)

    def _prepare_next_batch_requests(
        self, requests: List[Request], hidden_states: Any, lengths: Any
    ) -> List[Request]:
        """Prepares a batch of requests for the next stage of the pipeline."""
        batched_requests = []
        pre_length = 0
        for i, src_request in enumerate(requests):
            if self.is_last_peer:
                # Last peer gets a 1D array of token IDs
                hidden_state_for_req = hidden_states[i : i + 1]
            else:
                # Other peers get a 3D array of hidden states
                if src_request.is_prefill:
                    true_length = int(lengths[i])
                    if hidden_states.ndim == 3:
                        hidden_state_for_req = hidden_states[i, :true_length, :]
                    else:
                        hidden_state_for_req = hidden_states[
                            pre_length : pre_length + true_length, :
                        ]
                    pre_length += true_length
                else:
                    if hidden_states.ndim == 3:
                        hidden_state_for_req = hidden_states[i, :, :]
                    else:
                        hidden_state_for_req = hidden_states[pre_length : pre_length + 1, :]
                    pre_length += 1

            next_req = self._prepare_next_single_request(src_request, hidden_state_for_req)
            batched_requests.append(next_req)

        return batched_requests

    def _process_batch_cuda(
        self, prepared_inputs: Dict[str, Any], return_decoded_tokens: bool = True
    ):
        """
        Process a batch of requests in CUDA.
        """
        assert "forward_batch" in prepared_inputs, "forward_batch should be in cuda prepared inputs"
        assert (
            "pp_proxy_tensors" in prepared_inputs
        ), "pp_proxy_tensors should be in cuda prepared inputs"
        forward_batch = prepared_inputs["forward_batch"]
        pp_proxy_tensors = prepared_inputs["pp_proxy_tensors"]
        logits_output, _ = self.model_runner.forward(
            forward_batch=forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.cur_batch:
            if self.cur_batch.forward_mode.is_extend():
                # Merge the new batch into the running batch
                if not self.cur_batch.is_empty():
                    if self.running_batch.is_empty():
                        self.running_batch = self.cur_batch
                    else:
                        # Merge running_batch with prefill batch
                        self.running_batch.merge_batch(self.cur_batch)
            self.cur_batch = None

        if return_decoded_tokens:
            next_token_ids = self.model_runner.sample(logits_output, forward_batch)
            return next_token_ids
        # Currently hack the result of (hidden_state + residual) here for GPU
        final_hidden_states = (
            logits_output.tensors["hidden_states"] + logits_output.tensors["residual"]
        )
        return final_hidden_states

    def _process_batch_mlx(
        self, prepared_inputs: Dict[str, Any], return_decoded_tokens: bool = True
    ):
        """
        Process a batch of requests in MLX.
        """
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

        return hidden_states

    def process_batch(
        self, prepared_inputs: Dict[str, Any], return_decoded_tokens: bool = True
    ) -> mx.array:
        """
        Process a batch of requests.

        Args:
            prepared_inputs: A dictionary containing the prepared inputs for the ShardedModel.
            return_decoded_tokens: Whether to return decoded tokens.

        Returns:
            A tensor of shape (B, L, D) containing the hidden states for the next peer.
            or (B,) containing the decoded tokens.
        """
        if self.device == "cuda":
            ret = self._process_batch_cuda(prepared_inputs, return_decoded_tokens)
        else:
            ret = self._process_batch_mlx(prepared_inputs, return_decoded_tokens)
        logger.debug(
            f"Processed batch (device={self.device}, return_tokens={return_decoded_tokens})"
        )
        return ret

    def _release_and_evict_request(self, rid: str):
        """Release per-request resources and evict from scheduler. Best-effort, never raises."""
        # Release resources
        if self.device == "cuda":
            from parallax.sglang.batch_info import release_cuda_request

            try:
                release_cuda_request(self.running_batch, rid)
            except Exception:
                pass
        else:
            try:
                if hasattr(self, "kv_cache_manager") and self.kv_cache_manager is not None:
                    self.kv_cache_manager.release_request(rid)
            except Exception:
                pass

        # Evict from scheduler
        try:
            self.scheduler.evict_request(rid)
        except Exception:
            pass

    def run_loop(self):
        """The main loop of the executor."""
        logger.debug(
            f"Executor for layers [{self.start_layer}, {self.end_layer}) starting run loop..."
        )
        while True:
            # 1. Ingest new requests from the http frontend
            if self.is_first_peer:
                http_requests = self.recv_requests_from_http()
                self._handle_input_requests(http_requests)

            # 2. Ingest new requests from the RPC server
            incoming_requests = self.recv_requests_from_peer()
            self._handle_input_requests(incoming_requests)

            # 3. Send finished batch to next peer
            if len(self.finished_batch) > 0 and self.is_first_peer:
                self.send_to_peer_socket.send_multipart(
                    [b"abort", abort_request_to_proto(self.finished_batch).SerializeToString()]
                )
                self.finished_batch = []

            # 4. Admit requests into running set up to capacity, then form batch
            self.scheduler.admit_requests()
            # 4.1 Check for request timeouts and abort timed out requests
            try:
                timed_out_reqs = self.scheduler.get_timed_out_requests()
                if timed_out_reqs:
                    for req in timed_out_reqs:
                        rid = req.request_id
                        logger.warning(
                            f"Request {rid} exceeded timeout ({req.timeout_s}s). Aborting and releasing resources."
                        )
                        self._release_and_evict_request(rid)

                        # Notify downstream peers to abort if this peer is the first peer in a pipeline
                        if self.is_first_peer and not self.is_last_peer:
                            self.finished_batch.append(req)
            except Exception:
                # Non-fatal; continue serving
                pass
            batch_to_process = self.scheduler.form_batch()
            if not batch_to_process:
                continue
            logger.debug(f"Formed batch with {len(batch_to_process)} requests.")

            # 6. Process the batch
            try:
                prepared_inputs_dict = self._prepare_batch_inputs(batch_to_process)

                # We will process prefill and decode batches separately for now
                for batch_type in ["prefill_batch", "decode_batch"]:
                    if prepared_inputs_dict and prepared_inputs_dict.get(batch_type):
                        prepared_inputs = prepared_inputs_dict[batch_type]

                        start_time = time.time()
                        output = self.process_batch(
                            prepared_inputs, return_decoded_tokens=self.is_last_peer
                        )
                        # Update metrics with per-layer latency sample (throttled by decode steps)
                        if batch_type == "decode_batch":
                            try:
                                self._decode_steps_since_metric += len(prepared_inputs["requests"])
                                if (
                                    self._decode_steps_since_metric
                                    >= self.layer_latency_update_every
                                ):
                                    elapsed_ms = (time.time() - start_time) * 1000.0
                                    assert self.num_shard_layers > 0
                                    per_layer_ms = elapsed_ms / float(self.num_shard_layers)
                                    update_metrics(layer_latency_ms_sample=per_layer_ms)
                                    self._decode_steps_since_metric = 0
                            except Exception:
                                pass
                        # 7. Prepare requests for the next stage in the pipeline
                        next_batch = self._prepare_next_batch_requests(
                            requests=prepared_inputs["requests"],
                            hidden_states=output,
                            lengths=prepared_inputs["lengths"],
                        )

                        # 8. Dispatch to the appropriate destination
                        if self.is_last_peer and self.is_first_peer:
                            # Single node: handle locally
                            self._handle_input_requests(next_batch)
                        else:
                            # Send output to next peer
                            self.send_to_peer_socket.send_multipart(
                                [
                                    b"forward",
                                    request_to_proto(next_batch, self.device).SerializeToString(),
                                ]
                            )
                            logger.debug(
                                f"Processed batch of type {batch_type} with {len(next_batch)} requests "
                                f"in {(time.time() - start_time) * 1000:.3f} ms"
                            )

            except Exception as e:
                logger.exception(f"Error processing batch: {e}")
                # Naive error handling: release and evict all requests in the batch
                for req in batch_to_process:
                    self._release_and_evict_request(req.request_id)

    def run_loop_in_background(self):
        """Run the executor loop in the background."""

    def shutdown(self):
        """Shuts down the executor."""
        logger.debug("Executor shutting down...")
        self.recv_from_peer_socket.close()
        self.send_to_peer_socket.close()
        self.recv_from_ipc_socket.close()
        self.send_to_ipc_socket.close()
        self.zmq_context.term()
        logger.debug("Executor shutdown complete.")


def create_executor_config(args: argparse.Namespace):
    """Create executor configuration from command line arguments."""

    config = {
        "model_repo": args.model_path,
        "start_layer": args.start_layer,
        "end_layer": args.end_layer,
        "dtype": args.dtype,
        "max_sequence_length": args.max_sequence_length if "max_sequence_length" in args else None,
        "max_batch_size": args.max_batch_size if "max_batch_size" in args else None,
        "kv_block_size": args.kv_block_size,
        "kv_cache_memory_fraction": args.kv_cache_memory_fraction,
        "enable_prefix_cache": args.enable_prefix_cache,
        "max_num_tokens_per_batch": args.max_num_tokens_per_batch,
        "prefill_priority": args.prefill_priority,
        "micro_batch_ratio": args.micro_batch_ratio,
        "scheduler_wait_ms": args.scheduler_wait_ms,
        "send_to_peer_addr": args.send_to_peer_addr if "send_to_peer_addr" in args else None,
        "recv_from_peer_addr": args.recv_from_peer_addr if "recv_from_peer_addr" in args else None,
        "executor_input_ipc_addr": args.executor_input_ipc,
        "executor_output_ipc_addr": args.executor_output_ipc,
        "attention_backend": args.attention_backend,
        "moe_runner_backend": args.moe_runner_backend,
    }
    return config
