"""
Continuous Batching Scheduler.

State managed by the scheduler:
    1. Prefill Wait Queue (FIFO): incoming prefill requests waiting for admission;
    2. Running Requests: inflight requests with KV-cache residency;
Main `form_batch` function will return the concrete batch chosen for the next model forward.

We use an explicit 2-Phase approach:
    * Phase 1 (Admission): wait queue -> running requests
        Implemented by `admit_requests`. We admit requests when capacity
        allows (e.g., max concurrent requests, memory availability). Admitted
        requests get KV-cache residency and become inflight.
    * Phase 2 (Batching): running requests -> active batch for actual forward
        Implemented by `form_batch`. We prioritize PREFILL requests
        first within `max_num_tokens_per_batch` and `micro_batch_size`,
        then include DECODE requests that are marked ready for the next decode step.

Our scheduler also handles tokenization and pre-processing for the First Peer's requests.
"""

import time
from collections import OrderedDict
from typing import Dict, List, Optional

from parallax.server.kv_cache import KVCacheManager
from parallax.server.request import InitialRequest, Request, RequestStatus
from parallax.utils.shared_state import SharedState
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class Scheduler:
    """
    2-Phase approach:
        * Phase 1: wait queue -> running requests (all inflight requests)
        * Phase 2: running requests -> active batch (actual model forward)
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        max_num_tokens_per_batch: int = 4096,
        scheduler_wait_ms: int = 200,
        micro_batch_ratio: int = 2,
        is_first_peer: bool = False,
        kv_cache_manager: Optional[KVCacheManager] = None,
        request_timeout_s: Optional[int] = 600,
        shared_state: Optional[SharedState] = None,
        **kwargs,
    ):
        """
        Args:
            max_batch_size: Maximum number of running / inflight requests;
            max_num_tokens_per_batch: Maxmimum number of prefill + decode tokens in a single batch;
            scheduler_wait_ms: The minimum time to wait before dispatching a batch;
            micro_batch_ratio: micro_batch_size = max_batch_size // micro_batch_ratio;
            tokenizer: The tokenizer to use for the model;
            kv_cache_manager: The KV cache manager to use for the scheduler.
            request_timeout_s: timeout for each inflight request (default 10mins).
        """
        self.max_batch_size = max_batch_size
        self.max_num_tokens_per_batch = max_num_tokens_per_batch
        self.micro_batch_size = max(1, max_batch_size // micro_batch_ratio)
        self.scheduler_wait_ms = scheduler_wait_ms
        self.is_first_peer = is_first_peer
        if is_first_peer:
            # Load configs for building InitialRequest
            self.tokenizer = kwargs.get("tokenizer", None)
            self.eos_token_id = kwargs.get("eos_token_id", None)
            self.max_new_tokens = kwargs.get("max_new_tokens", 512)
            self.max_total_length = kwargs.get("max_total_length", 1024)

        # Prefill wait queue (FIFO) for admission
        self._wait_queue: List[Request] = []
        # Keeps track of all in-flight requests
        self._running_requests: Dict[str, Request] = OrderedDict()

        self.kv_cache_manager = kv_cache_manager
        self.shared_state = shared_state
        # Default timeout for requests if not set on request object
        self.request_timeout_s = request_timeout_s

        self._last_dispatch_ts = time.time()
        # Track last reported running requests to avoid redundant metric updates
        self._last_reported_running_requests: int = 0
        logger.debug(
            f"Scheduler initialized: max_batch_size={self.max_batch_size}, "
            f"max_num_tokens_per_batch={self.max_num_tokens_per_batch}"
        )

    @property
    def num_queued_requests(self) -> int:
        """Get the number of requests in the scheduler."""
        return len(self._wait_queue)

    @property
    def num_running_requests(self) -> int:
        """Get the number of requests currently being processed."""
        return len(self._running_requests)

    def get_running_request(self, request_id: str) -> Optional[Request]:
        """Gets a request that is currently in the running state."""
        return self._running_requests.get(request_id)

    def _prompt_string_to_request(self, request_str: str) -> InitialRequest:
        """Convert the prompt string to InitialRequest."""
        assert self.is_first_peer, "Only first peer can enqueue InitialRequest."
        input_ids = self.tokenizer.encode(request_str)
        return InitialRequest.from_prompt_ids(
            input_ids, self.eos_token_id, self.max_new_tokens, self.max_total_length
        )

    def enque_request(self, request: Request | str):
        """Enque a request to the scheduler's wait queue."""
        if isinstance(request, str):
            request = self._prompt_string_to_request(request)

        if request.is_finished:
            logger.warning(
                f"Request {request.request_id} is already "
                f"{request.status}. Not adding to the scheduler."
            )
            return

        request.ready_for_next_step = True
        request.last_updated_time = time.time()
        # TODO: Handle chunked prefill.
        if request.is_decoding:
            rid = request.request_id
            if rid not in self._running_requests:
                raise ValueError(
                    f"Decode request {rid} must already be admitted (in running requests)."
                )
            # Merge incoming decode readiness/state into the existing running request
            self._running_requests[rid] = request
            # Update recency ordering so earlier-ready decodes are encountered first during batching
            self._running_requests.move_to_end(rid)
            logger.debug(f"Decode request {rid} marked ready for next decode.")
            return

        self._wait_queue.append(request)
        logger.debug(
            f"Prefill request {request.request_id} added to the prefill wait queue (size={len(self._wait_queue)})."
        )

    def evict_request(self, request_id: str):
        """Removes a request from the scheduler's running queue."""
        if request_id in self._running_requests:
            self._running_requests.pop(request_id)
            logger.debug(f"Evicted request {request_id} from scheduler.")
            # Update metrics only if running count changed since last report
            try:
                if self.shared_state is not None:
                    curr = self.num_running_requests
                    self.shared_state.update_metrics(current_requests=curr)
            except Exception:
                pass
        else:
            raise ValueError(f"Attempted to evict non-existent request {request_id}.")

    def cancel_request(self, request_id: str):
        """Cancels a request from the scheduler."""
        if request_id in self._running_requests:
            req = self._running_requests[request_id]
            req.abort = True
            logger.debug(f"Cancelled request {request_id} from scheduler.")
        else:
            raise ValueError(f"Attempted to cancel non-existent request {request_id}.")

    def check_and_update_request_status(self, request: InitialRequest) -> bool:
        """Checks if a request has met any finishing conditions and updates its status."""
        assert self.is_first_peer, "Only first peer can check and update request status."
        assert (
            self.eos_token_id is not None
        ), "EOS token ID must be set for request status checking."
        if request.is_finished:
            return True

        finished = False
        last_token_id = request.output_ids[-1] if request.output_ids else None
        if request.abort:
            finished = True
        if not request.sampling_params.ignore_eos and (
            self.eos_token_id
            and (
                last_token_id == self.eos_token_id
                or (isinstance(self.eos_token_id, list) and last_token_id in self.eos_token_id)
            )
        ):
            request.update_status(RequestStatus.FINISHED_EOS)
            finished = True
        elif not request.sampling_params.ignore_eos and (
            self.tokenizer
            and self.tokenizer.eos_token_id
            and last_token_id == self.tokenizer.eos_token_id
        ):
            request.update_status(RequestStatus.FINISHED_EOS)
            finished = True
        elif request.output_length >= request.max_new_tokens:
            request.update_status(RequestStatus.FINISHED_MAX_LENGTH)
            finished = True
        elif request.total_length >= request.max_total_length:
            request.update_status(RequestStatus.FINISHED_MAX_LENGTH)
            finished = True

        if finished:
            logger.debug(f"Request {request.request_id} finished with status {request.status}.")
            # Remove from running requests. The executor will handle KV cache release.
            self.evict_request(request.request_id)

        return finished

    def admit_requests(self):
        """Move requests from wait queue into running (inflight) set, up to capacity.

        Pushes admitted requests directly into the running set.
        """
        while self._wait_queue and len(self._running_requests) < self.max_batch_size:
            req = self._wait_queue.pop(0)
            rid = req.request_id
            if rid in self._running_requests:
                # Already inflight; chunked-prefill, skip
                continue
            # Check kv cache pool
            if self.kv_cache_manager is not None:
                if not self.kv_cache_manager.has_request(req.request_id):
                    if not self.kv_cache_manager.add_request(req, req.total_length):
                        logger.warning(
                            f"Request {rid} can't be admit to running batch due to KV cache size."
                        )
                        continue
            self._running_requests[rid] = req
            # Initialize timing for timeout enforcement
            req.last_updated_time = time.time()
            logger.debug(
                f"Admitted to running: rid={rid}, status={req.status}, running_size={len(self._running_requests)}, ready={req.ready_for_next_step}"
            )

        # Reflect current running requests metric after admission
        try:
            if self.shared_state is not None:
                curr = self.num_running_requests
                if curr != self._last_reported_running_requests:
                    self.shared_state.update_metrics(current_requests=curr)
                    self._last_reported_running_requests = curr
        except Exception:
            pass

        return

    def get_timed_out_requests(self) -> List[Request]:
        """Return running requests that exceeded their timeout and mark them aborted.

        This does not evict or release resources; callers must handle cleanup.
        """
        timed_out: List[Request] = []
        now = time.time()
        for req in list(self._running_requests.values()):
            try:
                if req.last_updated_time is None:
                    raise ValueError("Requests should have last updated time set.")
                if now - req.last_updated_time > self.timeout_s:
                    req.abort = True
                    timed_out.append(req)
            except Exception:
                continue
        return timed_out

    def form_batch(self) -> List[Request]:
        """Form the active batch for the next forward pass.

        - Select prefills first (FIFO by admission), then decodes that are ready
          following the OrderedDict iteration order where ready decodes are
          moved-to-end upon readiness, while respecting micro_batch_size and
          max_num_tokens_per_batch.
        """
        self.admit_requests()
        if not self._running_requests:
            return []

        inflight_tokens = 0
        batch: List[Request] = []

        # Prefill candidates: preserve admission order via OrderedDict iteration
        prefill_candidates = []
        decode_candidates = []
        for req in self._running_requests.values():
            if req.ready_for_next_step:
                if req.is_prefill:
                    prefill_candidates.append(req)
                elif req.is_decoding:
                    decode_candidates.append(req)

        # 1) Fill with prefills first
        for req in prefill_candidates:
            if len(batch) >= self.micro_batch_size:
                break
            cost = req.prompt_len
            if cost + inflight_tokens > self.max_num_tokens_per_batch:
                continue
            batch.append(req)
            inflight_tokens += cost

        # 2) Fill remaining with ready decodes
        for req in decode_candidates:
            if len(batch) >= self.micro_batch_size:
                break
            cost = 1
            if cost + inflight_tokens > self.max_num_tokens_per_batch:
                continue
            batch.append(req)
            inflight_tokens += cost

        # Clear ready flags for decodes included in this batch
        for r in batch:
            r.ready_for_next_step = False
            r.last_updated_time = time.time()

        if batch:
            logger.debug(
                "Form batch selected=%s inflight_tokens=%d",
                [f"{r.request_id}:{r.status}, ready:{r.ready_for_next_step}" for r in batch],
                inflight_tokens,
            )
        return batch
