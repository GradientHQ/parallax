"""
vLLM backend implementation of high level executor
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
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
from parallax.vllm.model_runner import initialize_vllm_model_runner
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
        model_runner_params = {
            "model_repo": model_repo,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "kv_cache_memory_fraction": kv_cache_memory_fraction,
            "attention_backend": attention_backend,
            "kv_block_size": kv_block_size,
            "max_num_tokens_per_batch": max_num_tokens_per_batch,
            "dtype": dtype,
            "moe_runner_backend": moe_runner_backend,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "nccl_port": nccl_port,
            "using_hfcache": use_hfcache,
            "enable_lora": enable_lora,
            "max_lora_rank": max_lora_rank,
            "lora_target_modules": lora_target_modules,
            "lora_paths": lora_paths,
            "max_loras_per_batch": max_loras_per_batch,
            "max_loaded_loras": max_loaded_loras,
            "lora_eviction_policy": lora_eviction_policy,
            "lora_backend": lora_backend,
            "max_lora_chunk_size": max_lora_chunk_size,
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
                            "But no corresponding request found in scheduler (CUDA). "
                            "It might have been cancelled or finished."
                        )
                        continue

                    assert req.next_token_id is not None
                    original_req.commit_new_token(req.next_token_id)
                    logger.debug(
                        f"[FirstPeer-CUDA] Committed token {req.next_token_id} for {req.request_id}, "
                        f"output_ids now has {len(original_req.output_ids)} tokens"
                    )
                    if len(req.routing_table) > 0:
                        original_req.routing_table = req.routing_table

                    # Check for termination.
                    if self.scheduler.check_and_update_request_status(original_req):
                        logger.debug(f"Releasing resources for finished request {req.request_id}")
                        self.release_and_evict_request(req.request_id)
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

        # Import IntermediateTensors for type checking

        # Execute model with vLLM
        output = self.model_runner.execute_model(
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate_tensors,
        )

        # Return appropriate output based on peer position
        if return_decoded_tokens:
            sampled_token_ids = output.sampled_token_ids
            if isinstance(sampled_token_ids, list) and len(sampled_token_ids) > 0:
                # Convert to tensor: pad sequences to same length
                max_len = max(len(seq) for seq in sampled_token_ids)
                padded_tokens = []
                for seq in sampled_token_ids:
                    padded_seq = seq + [-1] * (max_len - len(seq))  # Pad with -1
                    padded_tokens.append(padded_seq)
                return torch.tensor(padded_tokens, dtype=torch.int64)
            else:
                return torch.tensor(sampled_token_ids, dtype=torch.int64)
        else:
            # Intermediate peer: return hidden states for next peer
            final_hidden_states = output.tensors["hidden_states"] + output.tensors["residual"]
            return final_hidden_states

    def _release_request(self, rid: str):
        """Release per-request resources in vLLM."""
        try:
            release_vllm_request(self.model_runner, rid)
        except Exception:
            pass

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

        if not self.is_first_peer and pp_proxy_tensors is not None:
            target_tokens = compute_expected_intermediate_tokens(
                schedule_outputs_prefill, self.model_runner
            )
            pp_proxy_tensors = resize_intermediate_tensors(pp_proxy_tensors, target_tokens)

        ret = {
            "scheduler_output": schedule_outputs_prefill,
            "pp_proxy_tensors": pp_proxy_tensors,
            "lengths": lengths_tensor,
            "requests": batched_requests,
        }
        logger.debug(f"Prepared CUDA prefill batch (vllm, size={batch_size})")
        return ret

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
        ret = {
            "scheduler_output": scheduler_outputs_decode,
            "pp_proxy_tensors": pp_proxy_tensors,
            "lengths": lengths_tensor,
            "requests": batched_requests,
        }
        logger.debug(f"Prepared CUDA decode batch (vllm, size={batch_size})")
        return ret
