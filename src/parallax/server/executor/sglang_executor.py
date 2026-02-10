"""
SGLang backend implementation of high level executor
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache as PageRadixCache
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.utils.common import SUPPORTED_LORA_TARGET_MODULES

from parallax.server.executor.base_executor import BaseExecutor
from parallax.server.request import (
    InitialRequest,
    IntermediateRequest,
    Request,
    RequestStatus,
)
from parallax.sglang.batch_info import (
    form_sgl_batch_decode,
    form_sgl_batch_prefill,
    release_sglang_request,
)
from parallax.sglang.model_runner import initialize_sgl_model_runner, refit_sgl_model
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class SGLExecutor(BaseExecutor):
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
        enable_dp_attention: bool = False,
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
        dp_rank: Optional[int] = 0,
        dp_size: Optional[int] = 1,
        nccl_port: Optional[int] = 4000,
        # Optional shared state for layer reallocation detection (when running in subprocess)
        shared_state: Optional[dict] = None,
        # Weight Refit
        enable_weight_refit: Optional[bool] = False,
        weight_refit_mode: Optional[str] = "disk",
        # Pipe communication
        conn: Optional[List[Any]] = [],
    ):

        self.enable_lora = True if lora_paths is not None else enable_lora
        self.lora_paths = lora_paths
        self.max_lora_rank = max_lora_rank
        self.lora_target_modules = lora_target_modules
        self.max_loras_per_batch = 1 if max_loras_per_batch is None else max_loras_per_batch
        self.max_loaded_loras = max_loaded_loras
        self.lora_eviction_policy = lora_eviction_policy
        self.lora_backend = lora_backend
        self.max_lora_chunk_size = max_lora_chunk_size

        if self.lora_paths is not None and len(self.lora_paths) > 0:
            self.check_lora_server_args()

        # output lora paths
        if self.lora_paths is not None:
            logger.info(f"LoRA paths provided: {[str(lora_path) for lora_path in self.lora_paths]}")

        model_runner_params = {
            "model_repo": model_repo,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "kv_cache_memory_fraction": kv_cache_memory_fraction,
            "attention_backend": attention_backend,
            "enable_dp_attention": enable_dp_attention,
            "kv_block_size": kv_block_size,
            "max_num_tokens_per_batch": max_num_tokens_per_batch,
            "dtype": dtype,
            "moe_runner_backend": moe_runner_backend,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "dp_rank": dp_rank,
            "dp_size": dp_size,
            "nccl_port": nccl_port,
            "using_hfcache": use_hfcache,
            "enable_lora": self.enable_lora,
            "max_lora_rank": self.max_lora_rank,
            "lora_target_modules": self.lora_target_modules,
            "lora_paths": self.lora_paths,
            "max_loras_per_batch": self.max_loras_per_batch,
            "max_loaded_loras": self.max_loaded_loras,
            "lora_eviction_policy": self.lora_eviction_policy,
            "lora_backend": self.lora_backend,
            "max_lora_chunk_size": self.max_lora_chunk_size,
        }
        logger.debug(
            f"Initializing SGLang model runner for repo={model_repo}, layers=[{start_layer}, {end_layer})"
        )
        self.model_runner, self.config, self.tokenizer = initialize_sgl_model_runner(
            **model_runner_params
        )
        logger.debug(
            f"SGLang model runner initialized. num_layers={self.config.get('num_hidden_layers')}"
        )

        # Set device to specific CUDA device based on tp_rank
        # This ensures tensors are moved to the correct GPU
        if device is None or device == "cuda":
            device = f"cuda:{tp_rank}"

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
            dp_rank=dp_rank,
            dp_size=dp_size,
            shared_state=shared_state,
            enable_weight_refit=enable_weight_refit,
            weight_refit_mode=weight_refit_mode,
            conn=conn,
        )
        self.cur_batch = None
        self.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        self.tp_group = self.model_runner.tp_group
        self.tp_cpu_group = self.tp_group.cpu_group

        # create a page tree cache for sglang prefill
        if enable_prefix_cache:
            cache_params = CacheInitParams(
                disable=False,
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
                page_size=self.model_runner.page_size,
            )
            self.page_tree_cache = PageRadixCache(cache_params)
            logger.info(
                f"Sglang Page tree cache created with page size {self.model_runner.page_size}"
            )
        else:
            self.page_tree_cache = None

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
            refit_sgl_model(self.model_runner, tensors=tensors)
        elif self.weight_refit_mode == "disk":
            refit_sgl_model(self.model_runner, refit_weight_path=weight_path)
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

        if self.enable_lora:
            # Parse lora_paths
            if isinstance(self.lora_paths, list):
                lora_paths = self.lora_paths
                self.lora_paths = []
                for lora_path in lora_paths:
                    if isinstance(lora_path, str):
                        if "=" in lora_path:
                            name, path = lora_path.split("=", 1)
                            lora_ref = LoRARef(lora_name=name, lora_path=path, pinned=False)
                        else:
                            lora_ref = LoRARef(
                                lora_name=lora_path, lora_path=lora_path, pinned=False
                            )
                    elif isinstance(lora_path, dict):
                        assert (
                            "lora_name" in lora_path and "lora_path" in lora_path
                        ), f"When providing LoRA paths as a list of dict, each dict should contain 'lora_name' and 'lora_path' keys. Got: {lora_path}"
                        lora_ref = LoRARef(
                            lora_name=lora_path["lora_name"],
                            lora_path=lora_path["lora_path"],
                            pinned=lora_path.get("pinned", False),
                        )
                    else:
                        raise ValueError(
                            f"Invalid type for item in --lora-paths list: {type(lora_path)}. "
                            "Expected a string or a dictionary."
                        )
                    self.lora_paths.append(lora_ref)
            elif isinstance(self.lora_paths, dict):
                self.lora_paths = [
                    LoRARef(lora_name=k, lora_path=v, pinned=False)
                    for k, v in self.lora_paths.items()
                ]
            elif self.lora_paths is None:
                self.lora_paths = []
            else:
                raise ValueError(
                    f"Invalid type for --lora-paths: {type(self.lora_paths)}. "
                    "Expected a list or a dictionary."
                )

            # Expand target modules
            if self.lora_target_modules:
                self.lora_target_modules = set(self.lora_target_modules)
                if "all" in self.lora_target_modules:
                    assert (
                        len(self.lora_target_modules) == 1
                    ), "If 'all' is specified in --lora-target-modules, it should be the only module specified."
                    self.lora_target_modules = set(SUPPORTED_LORA_TARGET_MODULES)

            # Ensure sufficient information is provided for LoRA initialization.
            assert self.lora_paths or (
                self.max_lora_rank and self.lora_target_modules
            ), "When no initial --lora-paths is provided, you need to specify both --max-lora-rank and --lora-target-modules for LoRA initialization."

            # Validate max_loaded_loras
            if self.max_loaded_loras is not None:
                assert self.max_loaded_loras >= self.max_loras_per_batch, (
                    "max_loaded_loras should be greater than or equal to max_loras_per_batch. "
                    f"max_loaded_loras={self.max_loaded_loras}, max_loras_per_batch={self.max_loras_per_batch}"
                )
                assert len(self.lora_paths) <= self.max_loaded_loras, (
                    "The number of LoRA paths should not exceed max_loaded_loras. "
                    f"max_loaded_loras={self.max_loaded_loras}, lora_paths={len(self.lora_paths)}"
                )

            if self.max_lora_chunk_size is not None:
                assert (
                    16 <= self.max_lora_chunk_size <= 128
                    and (self.max_lora_chunk_size & (self.max_lora_chunk_size - 1)) == 0
                ), "--max-lora-chunk-size must be a power of 2 between 16 and 128."

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

                    if self.scheduler.check_and_update_request_status(original_req):
                        logger.debug(f"Releasing resources for finished request {req.request_id}")
                        self.release_and_evict_request(req.request_id)
                        if not self.is_last_peer and not req.abort:
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

                        # Add prob value for the sampled token (if requested and available)
                        if original_req.return_probs and req.token_prob is not None:
                            req_dict["probs"] = req.token_prob
                        if self.enable_weight_refit:
                            req_dict["weight_version"] = self.weight_version
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
                    if not self.is_last_peer and not req.abort:
                        self.finished_batch.append(req)
                else:
                    # This is an active request, add it to the scheduler queue to be processed.
                    self.scheduler.enque_request(req)

    def process_batch(self, prepared_inputs: Dict[str, Any], return_decoded_tokens: bool = True):
        """Process a batch of requests in SGLang."""
        assert "forward_batch" in prepared_inputs, "forward_batch should be in cuda prepared inputs"
        assert (
            "pp_proxy_tensors" in prepared_inputs
        ), "pp_proxy_tensors should be in cuda prepared inputs"

        forward_batch = prepared_inputs["forward_batch"]
        pp_proxy_tensors = prepared_inputs["pp_proxy_tensors"]
        requests = prepared_inputs.get("requests", [])

        # Execute model with SGLang
        out = self.model_runner.forward(
            forward_batch=forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        logits_output = out.logits_output

        # Merge prefill batch into running batch
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

        # Return appropriate output based on peer position
        if return_decoded_tokens:
            # Last peer: sample and return token IDs
            next_token_ids = self.model_runner.sample(logits_output, forward_batch)

            # Only compute probs if any request in the batch needs it
            # Check if any InitialRequest has return_probs=True
            needs_probs = any(
                (isinstance(req, InitialRequest) and req.return_probs)
                or (isinstance(req, IntermediateRequest) and req.return_probs)
                for req in requests
            )

            token_probs = None
            # Extract probs for the sampled tokens only if needed
            if needs_probs and hasattr(logits_output, "next_token_logits"):
                # Get probs for sampled tokens (next_token_logits contains probabilities)
                real_probs = torch.gather(logits_output.next_token_logits, 1, next_token_ids)
                token_probs = real_probs.cpu().float().tolist()

            # Return dict with token_ids and optional probs
            return {"hidden_states": next_token_ids, "probs": token_probs}
        else:
            # Intermediate peer: return hidden states for next peer
            # Note: SGLang stores hidden_states + residual separately
            final_hidden_states = (
                logits_output.tensors["hidden_states"] + logits_output.tensors["residual"]
            )
            return {"hidden_states": final_hidden_states, "probs": None}

    def _release_request(self, rid: str):
        """Release per-request resources in SGLang."""
        try:
            release_sglang_request(self.running_batch, rid)
        except Exception:
            pass

    def _check_kv_cache_available(self, num_tokens: int) -> bool:
        """
        Check if there is enough KV cache space for the requested tokens.

        Returns True if there is enough space, False otherwise.
        """
        try:
            allocator = self.model_runner.token_to_kv_pool_allocator
            available = allocator.available_size()

            if available < num_tokens:
                logger.warning(
                    f"KV cache space insufficient: need {num_tokens} tokens, "
                    f"but only {available} available"
                )
                return False
            return True
        except Exception as e:
            logger.warning(f"Failed to check KV cache availability: {e}")
            # If we can't check, allow the operation to proceed
            return True

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

        # Pre-check: Verify KV cache has enough space for prefill
        total_tokens_needed = sum(req.total_length for req in batched_requests)
        if not self._check_kv_cache_available(total_tokens_needed):
            self._abort_requests_due_to_kv_cache(
                batched_requests,
                f"KV cache insufficient for prefill ({total_tokens_needed} tokens needed)",
            )
            return None

        # Prepare PP proxy tensors
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

            # Create residual tensor with same shape
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

        # Prepare lengths (common for both backends)
        lengths = []
        for req in batched_requests:
            if req.lora_path is not None and self.lora_paths is not None:
                for lora_ref in self.lora_paths:
                    if lora_ref.lora_path == req.lora_path:
                        req.lora_id = lora_ref.lora_id
                        break
            else:
                req.lora_id = (
                    self.lora_paths[0].lora_id
                    if self.lora_paths and len(self.lora_paths) > 0
                    else None
                )
            lengths.append(req.total_length)
        lengths_tensor = torch.tensor(lengths, device=self.device)

        schedule_batch, forward_batch = form_sgl_batch_prefill(
            batched_requests,
            self.model_runner,
            self.page_tree_cache,
        )
        self.cur_batch = schedule_batch

        ret = {
            "forward_batch": forward_batch,
            "pp_proxy_tensors": pp_proxy_tensors,
            "context_lengths": lengths_tensor,
            "requests": batched_requests,
        }
        logger.debug(f"Prepared CUDA prefill batch (sglang, size={batch_size})")
        return ret

    def _prepare_decode_batch(self, batched_requests: List[Request]) -> Optional[Dict[str, Any]]:
        """
        Prepares inputs for CUDA backends from a batch of decode requests.
        """

        batch_size = len(batched_requests)
        if batch_size == 0:
            return None

        # Pre-check: Verify KV cache has enough space for decode (1 token per request)
        tokens_needed = batch_size
        if not self._check_kv_cache_available(tokens_needed):
            self._abort_requests_due_to_kv_cache(
                batched_requests,
                f"KV cache insufficient for decode ({tokens_needed} tokens needed)",
            )
            return None

        # Prepare PP proxy tensors
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

            # Create residual tensor with same shape
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

        # Prepare lengths (common for both backends)
        lengths = []
        for req in batched_requests:
            if req.lora_path is not None and self.lora_paths is not None:
                for lora_ref in self.lora_paths:
                    if lora_ref.lora_path == req.lora_path:
                        req.lora_id = lora_ref.lora_id
                        break
            else:
                req.lora_id = (
                    self.lora_paths[0].lora_id
                    if self.lora_paths and len(self.lora_paths) > 0
                    else None
                )
            lengths.append(req.total_length)
        lengths_tensor = torch.tensor(lengths, device=self.device)

        forward_batch = form_sgl_batch_decode(
            batched_requests,
            self.model_runner,
            self.running_batch,
            self.is_first_peer,
        )

        ret = {
            "forward_batch": forward_batch,
            "pp_proxy_tensors": pp_proxy_tensors,
            "context_lengths": lengths_tensor,
            "requests": batched_requests,
        }
        logger.debug(f"Prepared CUDA decode batch (sglang, size={batch_size})")
        return ret
