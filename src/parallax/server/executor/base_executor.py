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

import time
from abc import abstractmethod
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple

import zmq
from jinja2 import TemplateError
from mlx_lm.server import convert_chat, process_message_content

from parallax.p2p.message_util import (
    abort_request_to_proto,
    proto_to_abort_request,
    proto_to_request,
    request_to_proto,
)
from parallax.p2p.proto import forward_pb2
from parallax.p2p.server import ServerState
from parallax.server.request import (
    InitialRequest,
    IntermediateRequest,
    Request,
    RequestStatus,
)
from parallax.server.sampling.sampling_params import SamplingParams
from parallax.server.scheduler import Scheduler
from parallax.utils.shared_state import SharedState
from parallax.utils.utils import get_current_device, get_device_dtype, get_zmq_socket
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseExecutor:
    """High-level executor for managing model shards, scheduler, and cache pool on each Peer."""

    def __init__(
        self,
        # Model Configs
        start_layer: int,
        end_layer: int,
        dtype: str = "float16",
        # Device override
        device: Optional[str] = None,
        # Scheduler Configs
        max_batch_size: Optional[int] = 8,
        max_sequence_length: Optional[int] = None,
        # Controlling perfill / decode ratio
        max_num_tokens_per_batch: int = 1024,
        prefill_priority: int = 0,
        micro_batch_ratio: int = 2,
        scheduler_wait_ms: int = 500,
        request_timeout_s: Optional[int] = 600,
        # Metrics Configs
        layer_latency_update_every: int = 4096,
        # Communication Configs
        # P2P Communication Configs
        send_to_peer_addr: Optional[str] = None,
        recv_from_peer_addr: Optional[str] = None,
        # IPC Communication Configs
        executor_input_ipc_addr: Optional[str] = None,
        executor_output_ipc_addr: Optional[str] = None,
        # Tensor Parallel Configs
        tp_rank: Optional[int] = 0,
        tp_size: Optional[int] = 1,
        # Optional shared state for layer reallocation detection (when running in subprocess)
        shared_state: Optional[dict] = None,
    ):
        # Backend
        if device is not None:
            self.device = device
        else:
            self.device = get_current_device()
        logger.debug(f"Executor initializing on device: {self.device}")

        # for window attention need to calculate causal mask size
        self.finished_batch = []
        self.start_layer = start_layer
        self.end_layer = end_layer
        self._should_stop = False  # Flag to gracefully stop the executor
        # Reference to shared state for layer reallocation detection (when in subprocess mode)
        if shared_state is not None:
            self.shared_state = SharedState(shared_state)  # Auto-converts dict to SharedState
        else:
            self.shared_state = None

        self.is_first_peer = start_layer == 0
        self.is_last_peer = end_layer == self.config.get("num_hidden_layers")
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        # Metrics throttling for per-layer latency updates
        self.layer_latency_update_every = int(max(1, layer_latency_update_every))
        self._decode_steps_since_metric = self.layer_latency_update_every

        # TODO: Duplicate code to MLXExecutor.
        self.num_shard_layers = end_layer - start_layer
        self.dtype = get_device_dtype(dtype, self.device)
        logger.debug(
            f"Executor dtype set to {dtype} (resolved={self.dtype}); shard_layers={self.num_shard_layers}"
        )

        if self.tokenizer.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id

        self.eos_token_id = self.config.get("eos_token_id", None)

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
            eos_token_id=self.eos_token_id,
            kv_cache_manager=self.kv_cache_manager if self.device == "mlx" else None,
            request_timeout_s=request_timeout_s,
            shared_state=self.shared_state,
        )
        logger.debug(
            f"Scheduler initialized (max_batch_size={max_batch_size}, max_tokens={max_num_tokens_per_batch}, wait_ms={scheduler_wait_ms})"
        )

        # Communication Related
        if self.tp_rank == 0:
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
        if self.shared_state is not None:
            self.shared_state.set_status(ServerState.READY.value)

    @abstractmethod
    def handle_input_requests(self, requests: List[Request]):
        """Update requests states and status in scheduler and cache manager."""

    @abstractmethod
    def process_batch(self, prepared_inputs: Dict[str, Any], return_decoded_tokens: bool = True):
        """
        Process a batch of requests.

        Args:
            prepared_inputs: A dictionary containing the prepared inputs for the ShardedModel.
            return_decoded_tokens: Whether to return decoded tokens.

        Returns:
            A tensor of shape (B, L, D) containing the hidden states for the next peer.
            or (B,) containing the decoded tokens.
        """

    @abstractmethod
    def _prepare_prefill_batch(self, batched_requests: List[Request]) -> Dict[str, Any]:
        """Prepares inputs for ShardedModel from a batch of prefill requests."""

    @abstractmethod
    def _prepare_decode_batch(self, batched_requests: List[Request]) -> Dict[str, Any]:
        """Prepares inputs for ShardedModel from a batch of decode requests."""

    @abstractmethod
    def _gen_token_id_from_hidden(self, hidden_states) -> Tuple[int, Any]:
        """
        Inplace modifies hidden_states.
        Returns token_id, hidden_states
        """

    @abstractmethod
    def _release_request(self, rid: str):
        """Release request in backend frameworks"""

    def recv_requests_from_http(self) -> List[Request]:
        """Receives requests from http frontend"""
        if self.tp_rank != 0:
            return []

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
                self._notify_http_request_error(raw_request, e)
        if len(recv_reqs) > 0:
            logger.debug(f"Received {len(recv_reqs)} HTTP requests")
        return recv_reqs

    def recv_requests_from_peer(self) -> List[Request]:
        """Receives requests from the RPC server."""
        if self.tp_rank == 0:
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
                                            raise ValueError(
                                                f"Unsupported device type: {self.device}"
                                            )

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
        else:
            recv_reqs = []

        return recv_reqs

    def prepare_batch_inputs(self, batched_requests: List[Request]) -> Optional[Dict[str, Any]]:
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
        prefill_batch = self._prepare_prefill_batch(prefill_reqs)
        decode_batch = self._prepare_decode_batch(decode_reqs)
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

    def prepare_next_batch_requests(
        self, requests: List[Request], hidden_states: Any, lengths: Any
    ) -> List[Request]:
        """Prepares a batch of requests for the next stage of the pipeline."""
        if self.tp_rank == 0:
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
        else:
            batched_requests = None

        return batched_requests

    def release_and_evict_request(self, rid: str):
        """Release per-request resources and evict from scheduler. Best-effort, never raises."""
        # Release resources
        self._release_request(rid)

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
        self._should_stop = False
        while not self._should_stop:
            received_requests = []

            # Receive requests from http frontend
            if self.is_first_peer:
                received_requests = self.recv_requests_from_http()

            # Receive requests from peer
            received_requests.extend(self.recv_requests_from_peer())

            self.handle_input_requests(received_requests)

            # Send finished batch to next peer
            if len(self.finished_batch) > 0 and self.is_first_peer and self.tp_rank == 0:
                self.send_to_peer_socket.send_multipart(
                    [b"abort", abort_request_to_proto(self.finished_batch).SerializeToString()]
                )
                self.finished_batch = []

            # Check for layer reallocation signal (before batch processing)
            layer_changed = False
            if self.shared_state is not None:
                layer_changed = self.shared_state.get_layer_allocation_changed()

            if layer_changed:
                logger.info(
                    "Layer reallocation detected. Stopping executor to reload with new layers."
                )
                self._should_stop = True
                break

            # 5. Admit requests into running set up to capacity, then form batch
            self.scheduler.admit_requests()
            # 5.1 Check for request timeouts and abort timed out requests
            try:
                timed_out_reqs = self.scheduler.get_timed_out_requests()
                if timed_out_reqs:
                    for req in timed_out_reqs:
                        rid = req.request_id
                        logger.warning(
                            f"Request {rid} exceeded timeout ({req.timeout_s}s). Aborting and releasing resources."
                        )
                        self.release_and_evict_request(rid)

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
                prepared_inputs_dict = self.prepare_batch_inputs(batch_to_process)

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
                                    if self.shared_state is not None:
                                        self.shared_state.update_metrics(
                                            layer_latency_ms_sample=per_layer_ms
                                        )
                                    self._decode_steps_since_metric = 0
                            except Exception:
                                pass
                        # 7. Prepare requests for the next stage in the pipeline
                        next_batch = self.prepare_next_batch_requests(
                            requests=prepared_inputs["requests"],
                            hidden_states=output,
                            lengths=prepared_inputs["lengths"],
                        )

                        # 8. Dispatch to the appropriate destination
                        if self.tp_rank == 0:
                            if self.is_last_peer and self.is_first_peer:
                                # Single node: handle locally
                                self.handle_input_requests(next_batch)
                            else:
                                # Send output to next peer
                                self.send_to_peer_socket.send_multipart(
                                    [
                                        b"forward",
                                        request_to_proto(
                                            next_batch, self.device
                                        ).SerializeToString(),
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
                    self.release_and_evict_request(req.request_id)

    def run_loop_in_background(self):
        """Run the executor loop in the background."""

    def shutdown(self):
        """Shuts down the executor."""
        logger.debug("Executor shutting down...")
        self._should_stop = True
        import time

        time.sleep(0.1)  # Give run_loop a moment to exit gracefully

        try:
            all_requests = [req for _, _, _, req in self.scheduler._request_queue] + list(
                self.scheduler._running_requests.values()
            )
            for req in all_requests:
                try:
                    self.scheduler.evict_request(req.request_id, RequestStatus.CANCELLED)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if self.tp_rank == 0:
                self.recv_from_peer_socket.close()
                self.send_to_peer_socket.close()
                self.recv_from_ipc_socket.close()
                self.send_to_ipc_socket.close()
                self.zmq_context.term()
        except Exception as e:
            logger.debug(f"Error closing sockets (may already be closed): {e}")

        logger.debug("Executor shutdown complete.")

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
                tokenize=True,
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
            if "ignore_eos" in raw_sampling_params:
                sampling_params.ignore_eos = raw_sampling_params["ignore_eos"]

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

    def _notify_http_request_error(self, raw_request: Optional[Dict], error: Exception):
        """Best-effort notification to HTTP server when request parsing fails."""
        if not hasattr(self, "send_to_ipc_socket") or self.send_to_ipc_socket is None:
            return
        if not isinstance(raw_request, dict):
            return
        rid = raw_request.get("rid")
        if rid is None:
            return

        is_template_error = isinstance(error, TemplateError)
        status = (
            HTTPStatus.BAD_REQUEST
            if isinstance(error, ValueError) or is_template_error
            else HTTPStatus.INTERNAL_SERVER_ERROR
        )
        payload = {
            "type": "error",
            "rid": rid,
            "error": str(error),
            "error_type": error.__class__.__name__,
            "status_code": status.value,
        }
        try:
            self.send_to_ipc_socket.send_pyobj(payload)
        except Exception:  # pragma: no cover - best effort notification
            logger.debug("Failed to send error notification to HTTP handler", exc_info=True)

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

            next_token_id, hidden_states = self._gen_token_id_from_hidden(hidden_states)
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

            next_token_id, hidden_states = self._gen_token_id_from_hidden(hidden_states)
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
