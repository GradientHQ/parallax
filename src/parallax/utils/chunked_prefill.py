"""Backend-neutral helpers for Parallax chunked prefill request state."""

from collections.abc import Callable
from typing import Optional

from parallax.server.request import Request, RequestStatus
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


def set_request_prefill_chunk(req: Request, chunk_end: int, is_chunked: bool) -> None:
    """Expose the prompt prefix visible to the current prefill chunk."""
    if req.origin_input_ids is not None:
        req.input_ids = req.origin_input_ids[:chunk_end]
    req._effective_total_length = chunk_end
    req.is_chunked = is_chunked


def complete_local_middle_chunk(
    executor,
    request_id: str,
    release_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """Finish local bookkeeping after a non-final prefill chunk."""
    chunked_req = getattr(executor, "chunked_req", None)
    if chunked_req is None or chunked_req.rid != request_id:
        return False

    chunked_req.is_chunked = False
    if release_callback is not None:
        release_callback(request_id)

    if executor.is_first_peer:
        original_req = executor.scheduler.get_running_request(request_id)
        if original_req is None:
            logger.warning(
                "Completed local chunk for %s, but no running request was found.",
                request_id,
            )
            return True
        original_req.status = RequestStatus.PREFILLING
        executor.scheduler.enque_request(original_req)
    else:
        executor.scheduler.evict_request(request_id)

    return True


def filter_middle_chunk_next_batch(
    executor,
    requests: list[Request],
    next_batch: list[Request],
    release_callback: Optional[Callable[[str], None]] = None,
) -> list[Request]:
    """Drop user-visible output for middle chunks and requeue local work."""
    chunked_req = getattr(executor, "chunked_req", None)
    if (
        chunked_req is None
        or not chunked_req.is_chunked
        or chunked_req.rid not in [req.request_id for req in requests]
    ):
        return next_batch

    chunked_rid = chunked_req.rid
    filtered_next_batch = []
    for req in next_batch:
        if req.request_id == chunked_rid:
            req.status = RequestStatus.PREFILLING
            if executor.is_last_peer:
                continue
        filtered_next_batch.append(req)

    complete_local_middle_chunk(executor, chunked_rid, release_callback)
    return filtered_next_batch
