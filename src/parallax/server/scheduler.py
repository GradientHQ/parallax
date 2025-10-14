"""
Scheduling requests to form batches.sche

A scheduler will maintain a Priority Queue for request waiting pool.
We support continuous batching, and similar to TensorRT-LLM,
    we favors prefill requests over decode requests.
"""

import heapq
import time
from typing import Dict, List, Literal, Optional, Tuple

from parallax.server.metrics import update_metrics
from parallax.server.request import InitialRequest, Request, RequestStatus
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class Scheduler:
    """
    A simple scheduler to manage requests and form them into batches.
    This scheduler is designed to handle requests in a FIFO manner.
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        max_num_tokens_per_batch: int = 4096,
        prefill_priority: Literal[0, 1] = 0,
        scheduler_wait_ms: int = 200,
        micro_batch_ratio: int = 2,
        is_first_peer: bool = False,
        **kwargs,
    ):
        """
        Args:
            max_batch_size: Maximum number of running requests;
            max_num_tokens_per_batch: Maxmimum number of prefill + decode tokens in a single batch;
            prefill_priority: Priority for prefill requests,
                default 0 for prefill, 1 for decode, 0 for higher priority;
            scheduler_wait_ms: The minimum time to wait before dispatching a batch;
            micro_batch_ratio: micro_batch_size = max_batch_size // micro_batch_ratio
            tokenizer: The tokenizer to use for the model.
        """
        self.max_batch_size = max_batch_size
        self.max_num_tokens_per_batch = max_num_tokens_per_batch
        self.micro_batch_size = max(1, max_batch_size // micro_batch_ratio)
        self.scheduler_wait_ms = scheduler_wait_ms
        self.is_first_peer = is_first_peer
        if is_first_peer:
            # Load configs for building InitialRequest
            self.tokenizer = kwargs.get("tokenizer")
            self.eos_token_id = self.tokenizer.eos_token_id
            self.max_new_tokens = kwargs.get("max_new_tokens", 512)
            self.max_total_length = kwargs.get("max_total_length", 1024)

        # Priority queue: (priority, arrival_time, request_id, request_object)
        self._request_queue: List[Tuple[int, float, str, Request]] = []
        # Keeps track of all in-flight requests
        self._running_requests: Dict[str, Request] = {}

        self.priority_map = {
            RequestStatus.PREFILLING: prefill_priority,
            RequestStatus.DECODING: 1 - prefill_priority,
        }
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
        return len(self._request_queue)

    @property
    def num_running_requests(self) -> int:
        """Get the number of requests currently being processed."""
        return len(self._running_requests)

    @property
    def has_pending_requests(self) -> bool:
        """Check if there are any pending requests in the scheduler."""
        return len(self._request_queue) > 0

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
        """Add a request to the scheduler."""
        if isinstance(request, str):
            request = self._prompt_string_to_request(request)

        if request.is_finished:
            logger.warning(
                f"Request {request.request_id} is already "
                f"{request.status}. Not adding to the scheduler."
            )
            return
        arrival_time = time.time()
        priority = self.priority_map.get(request.status, 1)
        heapq.heappush(self._request_queue, (priority, arrival_time, request.request_id, request))
        logger.debug(f"Request {request.request_id} added to the scheduler.")
        logger.debug(f"Scheduler queue size: {len(self._request_queue)}")
        # Running count does not change on enqueue; do not update metrics here

    def evict_request(self, request_id: str, status: Optional[RequestStatus] = None):
        """Removes a request from the scheduler's running queue."""
        _ = status  # status is used by the first peer's logic but not here.
        if request_id in self._running_requests:
            self._running_requests.pop(request_id)
            logger.debug(f"Evicted request {request_id} from scheduler.")
            # Update metrics only if running count changed since last report
            try:
                curr = self.num_running_requests
                # if curr != self._last_reported_running_requests:
                update_metrics(current_requests=curr)
                # self._last_reported_running_requests = curr
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
        if last_token_id == self.eos_token_id:
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

    def should_dispatch(self) -> bool:
        """Helper check if the scheduler should dispatch a batch."""
        waited = (time.time() - self._last_dispatch_ts) * 1000 >= self.scheduler_wait_ms
        queued = self.num_queued_requests >= self.micro_batch_size
        return waited or queued

    def form_batch(self) -> List[Request]:
        """Get the next batch of requests.

        At-most `micro_batch_size` requests will be returned.
        """
        if not self.has_pending_requests:
            return []

        inflight_tokens = 0

        batch = []
        save_index = []
        for index, request in enumerate(self._request_queue):
            if len(batch) >= self.micro_batch_size:
                save_index.append(index)
                continue
            _, _, rid, req = request

            cost = req.prompt_len if req.is_prefill else 1
            if cost + inflight_tokens > self.max_num_tokens_per_batch:
                save_index.append(index)
                continue

            if rid not in self._running_requests:
                if len(self._running_requests) >= self.max_batch_size:
                    save_index.append(index)
                    continue

            batch.append(req)
            self._running_requests[rid] = req

            inflight_tokens += cost

        self._request_queue = [self._request_queue[i] for i in save_index]

        # Reflect current running requests metric after forming the batch
        try:
            curr = self.num_running_requests
            if curr != self._last_reported_running_requests:
                update_metrics(current_requests=curr)
                self._last_reported_running_requests = curr
        except Exception:
            pass

        return batch
