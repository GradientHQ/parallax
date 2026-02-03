"""
vLLM backend implementation of high level executor
"""

import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from vllm.sequence import IntermediateTensors

from parallax.server.executor.base_executor import BaseExecutor
from parallax.server.request import (
    InitialRequest,
    IntermediateRequest,
    Request,
    RequestStatus,
)
from parallax.vllm.batch_info import (
    compute_expected_intermediate_tokens,
    form_vllm_batch_decode,
    form_vllm_batch_prefill,
    release_vllm_request,
    resize_intermediate_tensors,
)
from parallax.vllm.model_runner import initialize_vllm_model_runner, refit_vllm_model
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class VLLMExecutor(BaseExecutor):
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
        max_num_tokens_per_batch: int = 16384,
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
        lora_paths: Optional[List[str]] = None,
        max_loras_per_batch: Optional[int] = None,
        max_loaded_loras: Optional[int] = None,
        fully_sharded_loras: bool = False,
        # Tensor Parallel Configs
        tp_rank: Optional[int] = 0,
        tp_size: Optional[int] = 1,
        nccl_port: Optional[int] = 4000,
        # Data Parallel Configs (not used in vLLM, but accepted for compatibility)
        enable_dp_attention: Optional[bool] = False,
        dp_rank: Optional[int] = 0,
        dp_size: Optional[int] = 1,
        # Optional shared state for layer reallocation detection (when running in subprocess)
        shared_state: Optional[dict] = None,
        # Weight Refit
        enable_weight_refit: Optional[bool] = False,
        weight_refit_mode: Optional[str] = "disk",
        # Routed experts
        enable_return_routed_experts: bool = False,
        # Pipe communication
        conn: Optional[List[Any]] = [],
    ):
        self.enable_return_routed_experts = enable_return_routed_experts
        self.routed_experts_reader = None
        self.routed_experts_instance_id = None

        self.enable_lora = True if lora_paths is not None else enable_lora
        self.lora_paths = lora_paths
        self.max_lora_rank = max_lora_rank
        self.max_loras_per_batch = 1 if max_loras_per_batch is None else max_loras_per_batch
        self.max_loaded_loras = max_loaded_loras

        if self.lora_paths is not None and len(self.lora_paths) > 0:
            self.check_lora_server_args()

        # output lora paths
        if self.lora_paths is not None:
            logger.info(f"LoRA paths provided: {[str(lora_path) for lora_path in self.lora_paths]}")
        # force routed experts for RL
        if enable_weight_refit and self.lora_paths is not None:
            self.enable_return_routed_experts = True

        if self.enable_return_routed_experts:
            self.routed_experts_instance_id = f"parallax_{os.getpid()}_{uuid.uuid4().hex}"

        model_runner_params = {
            "model_repo": model_repo,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "kv_cache_memory_fraction": kv_cache_memory_fraction,
            "attention_backend": attention_backend,
            "kv_block_size": kv_block_size,
            "max_batch_size": max_batch_size,
            "max_sequence_length": max_sequence_length,
            "max_num_tokens_per_batch": max_num_tokens_per_batch,
            "dtype": dtype,
            "moe_runner_backend": moe_runner_backend,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "nccl_port": nccl_port,
            "using_hfcache": use_hfcache,
            "enable_lora": self.enable_lora,
            "max_lora_rank": self.max_lora_rank,
            "lora_path": self.lora_paths[0] if self.lora_paths else None,
            "max_loras_per_batch": self.max_loras_per_batch,
            "max_loaded_loras": self.max_loaded_loras,
            "fully_sharded_loras": fully_sharded_loras,
            "enable_return_routed_experts": self.enable_return_routed_experts,
            "instance_id": self.routed_experts_instance_id,
        }
        logger.debug(
            f"Initializing vLLM model runner for repo={model_repo}, layers=[{start_layer}, {end_layer})"
        )
        self.model_runner, self.config, self.tokenizer = initialize_vllm_model_runner(
            **model_runner_params
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
            enable_weight_refit=enable_weight_refit,
            weight_refit_mode=weight_refit_mode,
            conn=conn,
        )
        if self.enable_return_routed_experts:
            self._init_routed_experts_reader()

    def check_and_refit_weight(self, refit_weight_path: str):
        if self.tp_size > 1:
            weight_path = self._tensor_parallel_broadcast_pyobj(refit_weight_path)
        else:
            weight_path = refit_weight_path

        if weight_path == "":
            return

        if self.weight_refit_mode == "cpu":
            conn = self.conn[0]
            tensors = conn.recv()
            refit_vllm_model(self.model_runner, tensors=tensors)
        elif self.weight_refit_mode == "disk":
            refit_vllm_model(self.model_runner, refit_weight_path=weight_path)
        else:
            logger.warning(f"Unrecognized weight refit mode={self.weight_refit_mode}")

    def check_lora_server_args(self):
        assert self.max_loras_per_batch > 0, "max_loras_per_batch must be positive"

        # Enable LoRA if any LoRA paths are provided for backward compatibility.
        if self.lora_paths:
            if self.enable_lora is None:
                self.enable_lora = True
                logger.warning("--enable-lora is set to True because --lora-paths is provided.")
            elif self.enable_lora is False:
                logger.warning(
                    "--enable-lora is set to False, any provided lora_paths will be ignored."
                )

    def handle_input_requests(self, requests: List[Request]):
        """Update requests states and status in scheduler and cache manager."""
        if self.tp_size > 1:
            requests = self._tensor_parallel_broadcast_pyobj(requests)
            for req in requests:
                if hasattr(req, "hidden_states") and req.hidden_states is not None:
                    if hasattr(req.hidden_states, "to"):  # PyTorch tensor
                        req.hidden_states = req.hidden_states.to(self.device)
        if len(requests) > 0:
            logger.debug(f"Handling {len(requests)} requests.")

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

                    # If it's an abort signal (e.g. from OOM), next_token_id might be None or dummy
                    if not req.abort and req.next_token_id is not None:
                        original_req.commit_new_token(req.next_token_id)
                        logger.debug(
                            f"[FirstPeer-CUDA] Committed token {req.next_token_id} for {req.request_id}, "
                            f"output_ids now has {len(original_req.output_ids)} tokens"
                        )

                    if len(req.routing_table) > 0:
                        original_req.routing_table = req.routing_table

                    # Check for termination.
                    if req.abort:
                        original_req.abort = True

                    routed_experts = None
                    if self.scheduler.check_and_update_request_status(original_req):
                        routed_experts = self._get_routed_experts_for_request(original_req)
                        logger.debug(f"Releasing resources for finished request {req.request_id}")
                        self.release_and_evict_request(req.request_id)
                        if not self.is_last_peer:
                            self.finished_batch.append(req)
                    else:
                        self.scheduler.enque_request(original_req)

                    # detokenize and send to http server
                    if self.tp_rank == 0:
                        # Only send token if it's valid
                        token_to_send = req.next_token_id if req.next_token_id is not None else -1
                        req_dict = {
                            "prompt_tokens": len(req.input_ids),
                            "next_token_id": token_to_send,
                            "rid": req.request_id,
                        }
                        if original_req.status == RequestStatus.FINISHED_EOS:
                            req_dict["eos"] = True
                        if original_req.status == RequestStatus.FINISHED_MAX_LENGTH:
                            req_dict["length"] = True
                        if original_req.status == RequestStatus.FINISHED_ABORT:
                            req_dict["abort"] = True
                        if self.enable_weight_refit:
                            req_dict["weight_version"] = self.weight_version
                        if original_req.return_probs and req.token_prob is not None:
                            req_dict["probs"] = req.token_prob
                        if routed_experts is not None:
                            req_dict["routed_experts"] = routed_experts
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
                    self.release_and_evict_request(req.request_id)
                    if not self.is_last_peer:
                        self.finished_batch.append(req)
                else:
                    # This is an active request, add it to the scheduler queue to be processed.
                    self.scheduler.enque_request(req)

    def process_batch(self, prepared_inputs: Dict[str, Any], return_decoded_tokens: bool = True):
        """Process a batch of requests in vLLM."""
        assert (
            "scheduler_output" in prepared_inputs
        ), "scheduler_output should be provided for vLLM backend"
        assert (
            "pp_proxy_tensors" in prepared_inputs
        ), "pp_proxy_tensors should be in cuda prepared inputs"
        scheduler_output = prepared_inputs["scheduler_output"]
        pp_proxy_tensors = prepared_inputs["pp_proxy_tensors"]
        # For vLLM, pp_proxy_tensors is already an IntermediateTensors object
        intermediate_tensors = pp_proxy_tensors if pp_proxy_tensors is not None else None
        if intermediate_tensors is not None:
            logger.debug(f"vLLM: Using intermediate_tensors for PP (non-first peer)")

        requests = prepared_inputs.get("requests", [])

        # Execute model with vLLM
        execute_model_state, sampled_token_ids, sampled_token_ids_cpu, sampler_output, logits = (
            self.model_runner.execute_model(
                scheduler_output=scheduler_output,
                intermediate_tensors=intermediate_tensors,
                return_decoded_tokens=return_decoded_tokens,
            )
        )

        # Return appropriate output based on peer position
        if return_decoded_tokens:
            needs_probs = any(
                (isinstance(req, InitialRequest) and req.return_probs)
                or (isinstance(req, IntermediateRequest) and req.return_probs)
                for req in requests
            )

            token_probs = None
            if needs_probs and logits is not None and isinstance(logits, torch.Tensor):

                if logits.ndim == 3:
                    logits = logits[:, -1, :]  # [batch, seq, vocab_size]
                elif logits.ndim != 2:
                    logger.warning(f"Unexpected logits shape: {logits.shape}")
                    logits = None

                if logits is not None:
                    probs = F.log_softmax(logits, dim=-1)
                    if isinstance(sampled_token_ids, torch.Tensor):
                        sampled_ids = sampled_token_ids
                    else:
                        sampled_ids = torch.tensor(
                            sampled_token_ids, device=logits.device, dtype=torch.long
                        )
                    probs = torch.gather(probs, 1, sampled_ids)
                    token_probs = probs.cpu().float().tolist()

            # Align outputs to request order if vLLM reorders the batch internally.
            input_batch = getattr(self.model_runner, "input_batch", None)
            req_id_to_index = getattr(input_batch, "req_id_to_index", None)
            if req_id_to_index:
                request_ids = [req.request_id for req in requests]
                if all(rid in req_id_to_index for rid in request_ids):
                    order = [req_id_to_index[rid] for rid in request_ids]
                    if isinstance(sampled_token_ids_cpu, torch.Tensor):
                        sampled_token_ids_cpu = sampled_token_ids_cpu[order]
                    elif isinstance(sampled_token_ids_cpu, list):
                        sampled_token_ids_cpu = [sampled_token_ids_cpu[i] for i in order]
                    if token_probs is not None:
                        token_probs = [token_probs[i] for i in order]

            return {"hidden_states": sampled_token_ids_cpu, "probs": token_probs}
        else:
            # Intermediate peer: return hidden states for next peer
            return {"hidden_states": execute_model_state.hidden_states, "probs": None}

    def _release_request(self, rid: str):
        """Release per-request resources in vLLM."""
        try:
            release_vllm_request(self.model_runner, rid)
        except Exception:
            pass

    def _abort_requests_due_to_kv_cache(self, batched_requests: List[Request], reason: str):
        """
        Abort requests due to KV cache shortage and notify relevant parties.
        """
        logger.warning(f"Aborting {len(batched_requests)} requests due to: {reason}")

        for req in batched_requests:
            req.update_status(RequestStatus.FINISHED_ABORT)

            # Notify HTTP Server to return partial results
            if self.is_first_peer and self.tp_rank == 0:
                req_dict = {
                    "prompt_tokens": req.prompt_len,
                    "next_token_id": (
                        req.output_ids[-1] if hasattr(req, "output_ids") and req.output_ids else -1
                    ),
                    "rid": req.request_id,
                    "abort": True,
                }
                if hasattr(self, "send_to_ipc_socket"):
                    self.send_to_ipc_socket.send_pyobj(req_dict)
                    time.sleep(0.05)  # Give ZMQ time to flush

            # Add to finished_batch to trigger abort notification to other peers
            self.finished_batch.append(req)
            self.scheduler.evict_request(req.request_id)

    def _gen_token_id_from_hidden(self, hidden_states) -> Tuple[int, Any]:
        """
        Inplace modifies hidden_states.
        Returns token_id, hidden_states
        """
        assert hidden_states.dtype in (
            torch.int64,
            torch.int32,
        ), "Single node must generate an output_id."
        next_token_id = int(hidden_states[0])
        return next_token_id, hidden_states

    def _tensor_parallel_broadcast_pyobj(self, broadcast_obj):
        """Wrapper for broadcast pyobject in TP group"""
        if self.tp_rank == 0:
            for i in range(1, self.tp_size):
                conn = self.conn[i]
                conn.send(broadcast_obj)
            broadcast_result = broadcast_obj
        else:
            conn = self.conn[0]
            broadcast_result = conn.recv()

        return broadcast_result

    def _prepare_prefill_batch(self, batched_requests: List[Request]) -> Dict[str, Any]:
        """
        Prepares inputs for CUDA backends from a batch of prefill requests.
        """
        batch_size = len(batched_requests)
        if batch_size == 0:
            return None

        # Prepare PP proxy tensors (common for both backends when not first peer)
        pp_proxy_tensors = None
        if not self.is_first_peer:
            # Concatenate hidden states from all requests
            # For vLLM, we need to flatten to (total_tokens, hidden_size)
            hidden_states_list = []
            for req in batched_requests:
                hs = req.hidden_states
                if hs.ndim == 2:
                    # Already (seq_len, hidden_size) or (1, hidden_size)
                    hidden_states_list.append(hs)
                elif hs.ndim == 3:
                    # (1, seq_len, hidden_size) -> (seq_len, hidden_size)
                    hidden_states_list.append(hs.squeeze(0))
                else:
                    # (hidden_size,) -> (1, hidden_size)
                    hidden_states_list.append(hs.unsqueeze(0))

            # Concatenate along sequence dimension to get (total_tokens, hidden_size)
            hidden_states = torch.cat(hidden_states_list, dim=0)

            # Create residual tensor with same shape
            residual = torch.zeros(
                hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device
            )

            pp_proxy_tensors = IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
            logger.debug(f"PP Proxy: hidden_states shape: {hidden_states.shape}")

        # Prepare lengths (common for both backends)
        lengths = []
        for req in batched_requests:
            lengths.append(req.total_length)
        lengths_tensor = torch.tensor(lengths, device=self.device)

        schedule_outputs_prefill = form_vllm_batch_prefill(batched_requests, self.model_runner)

        # Check if KV cache allocation failed
        if schedule_outputs_prefill is None:
            self._abort_requests_due_to_kv_cache(
                batched_requests, "KV cache insufficient for prefill"
            )
            return None

        if not self.is_first_peer and pp_proxy_tensors is not None:
            target_tokens = compute_expected_intermediate_tokens(
                schedule_outputs_prefill, self.model_runner
            )
            pp_proxy_tensors = resize_intermediate_tensors(pp_proxy_tensors, target_tokens)

        ret = {
            "scheduler_output": schedule_outputs_prefill,
            "pp_proxy_tensors": pp_proxy_tensors,
            "context_lengths": lengths_tensor,
            "requests": batched_requests,
        }
        logger.debug(f"Prepared CUDA prefill batch (vllm, size={batch_size})")
        return ret

    def _init_routed_experts_reader(self) -> None:
        if not self.enable_return_routed_experts:
            return
        num_hidden_layers = None
        if isinstance(self.config, dict):
            num_hidden_layers = self.config.get("num_hidden_layers")
        else:
            num_hidden_layers = getattr(self.config, "num_hidden_layers", None)

        if num_hidden_layers is None:
            logger.warning("Unable to determine num_hidden_layers; routed experts disabled.")
            self.enable_return_routed_experts = False
            return

        if not (self.start_layer == 0 and self.end_layer == num_hidden_layers):
            logger.warning(
                "Routed experts is only supported for single peer. "
                "Disabling routed experts for layers [%s, %s).",
                self.start_layer,
                self.end_layer,
            )
            self.enable_return_routed_experts = False
            return

        try:
            from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
                RoutedExpertsReader,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Failed to import RoutedExpertsReader: %s", exc)
            self.enable_return_routed_experts = False
            return

        kv_cache_config = getattr(self.model_runner, "kv_cache_config", None)
        if kv_cache_config is None or not getattr(kv_cache_config, "kv_cache_groups", None):
            logger.warning("KV cache config unavailable; routed experts disabled.")
            self.enable_return_routed_experts = False
            return

        block_size = self.model_runner.cache_config.block_size
        num_groups = len(kv_cache_config.kv_cache_groups)
        max_num_kv_tokens = (kv_cache_config.num_blocks // num_groups + 1) * block_size

        instance_id = self.model_runner.vllm_config.instance_id
        reader = RoutedExpertsReader.create()
        try:
            reader.attach_buffer(
                max_num_kv_tokens=max_num_kv_tokens,
                model_config=self.model_runner.model_config,
                instance_id=instance_id,
            )
        except Exception as exc:
            logger.warning("Failed to attach routed experts buffer: %s", exc)
            self.enable_return_routed_experts = False
            return
        self.routed_experts_reader = reader

    def _get_routed_experts_for_request(self, request: Request) -> Optional[List]:
        if not self.enable_return_routed_experts or self.routed_experts_reader is None:
            return None

        kv_cache_manager = getattr(self.model_runner, "kv_cache_manager", None)
        if kv_cache_manager is None:
            return None

        try:
            kv_blocks = kv_cache_manager.get_blocks(request.request_id)
            block_ids = kv_blocks.get_block_ids()
        except Exception as exc:
            logger.debug("Failed to get KV blocks for routed experts: %s", exc)
            return None

        if (
            isinstance(block_ids, (list, tuple))
            and block_ids
            and isinstance(block_ids[0], (list, tuple))
        ):
            block_ids = block_ids[0]

        if not block_ids:
            return None

        num_tokens = getattr(request, "total_length", None)
        if num_tokens is None or num_tokens <= 0:
            return None

        block_size = self.model_runner.cache_config.block_size
        block_ids_array = np.array(block_ids, dtype=np.int32)
        block_offsets = np.arange(0, block_size, dtype=np.int32)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_array.reshape((len(block_ids_array), 1)) * block_size
        ).flatten()

        slot_mapping = slot_mapping[: min(num_tokens, slot_mapping.size)]
        if slot_mapping.size == 0:
            return None

        try:
            routed_experts = self.routed_experts_reader.get_routed_experts(indices=slot_mapping)
        except Exception as exc:
            logger.debug("Failed to read routed experts: %s", exc)
            return None

        return routed_experts.tolist()

    def _prepare_decode_batch(self, batched_requests: List[Request]) -> Dict[str, Any]:
        """
        Prepares inputs for CUDA backends from a batch of decode requests.
        """

        batch_size = len(batched_requests)
        if batch_size == 0:
            return None

        # Prepare PP proxy tensors (common for both backends when not first peer)
        pp_proxy_tensors = None
        if not self.is_first_peer:
            # Concatenate hidden states from all requests
            # For vLLM, we need to flatten to (total_tokens, hidden_size)
            hidden_states_list = []
            for req in batched_requests:
                hs = req.hidden_states
                if hs.ndim == 2:
                    # Already (seq_len, hidden_size) or (1, hidden_size)
                    hidden_states_list.append(hs)
                elif hs.ndim == 3:
                    # (1, seq_len, hidden_size) -> (seq_len, hidden_size)
                    hidden_states_list.append(hs.squeeze(0))
                else:
                    # (hidden_size,) -> (1, hidden_size)
                    hidden_states_list.append(hs.unsqueeze(0))

            # Concatenate along sequence dimension to get (total_tokens, hidden_size)
            hidden_states = torch.cat(hidden_states_list, dim=0)

            # Create residual tensor with same shape
            residual = torch.zeros(
                hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device
            )

            pp_proxy_tensors = IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
            logger.debug(f"PP Proxy: hidden_states shape: {hidden_states.shape}")

        # Prepare lengths (common for both backends)
        lengths = []
        for req in batched_requests:
            lengths.append(req.total_length)
        lengths_tensor = torch.tensor(lengths, device=self.device)

        scheduler_outputs_decode = form_vllm_batch_decode(batched_requests, self.model_runner)

        # Check if KV cache allocation failed
        if scheduler_outputs_decode is None:
            self._abort_requests_due_to_kv_cache(
                batched_requests, "KV cache insufficient for decode"
            )
            return None

        ret = {
            "scheduler_output": scheduler_outputs_decode,
            "pp_proxy_tensors": pp_proxy_tensors,
            "context_lengths": lengths_tensor,
            "requests": batched_requests,
        }
        logger.debug(f"Prepared CUDA decode batch (vllm, size={batch_size})")
        return ret
