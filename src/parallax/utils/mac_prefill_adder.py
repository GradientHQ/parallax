from enum import Enum, auto
from typing import Optional

from parallax.server.cache_manager import CacheManager
from parallax.server.request import Request


class AddReqResult(Enum):
    CONTINUE = auto()
    NO_TOKEN = auto()
    OTHER = auto()


class MACPrefillAdder:
    """Selects the prompt span to run for MLX chunked prefill."""

    def __init__(
        self,
        page_size: int,
        rem_chunk_tokens: Optional[int],
        cache_manager: CacheManager,
    ):
        self.page_size = page_size
        self.rem_chunk_tokens = rem_chunk_tokens
        self.can_run_list: list[Request] = []
        self.new_chunked_req: Optional[Request] = None
        self.cache_manager = cache_manager

    def _matched_tokens(self, req: Request) -> int:
        token_ids = req.origin_input_ids
        if token_ids is None or self.cache_manager.prefix_cache is None:
            return 0
        _, matched_tokens = self.cache_manager.prefix_cache.match_prefix(token_ids)
        return matched_tokens

    def add_chunked_req(self, chunked_req: Request) -> Optional[Request]:
        if chunked_req is None or chunked_req.origin_input_ids is None:
            return None

        matched_tokens = self._matched_tokens(chunked_req)
        total_len = len(chunked_req.origin_input_ids)
        remaining_tokens = max(0, total_len - matched_tokens)
        budget_tokens = max(1, remaining_tokens)

        if self.rem_chunk_tokens is None:
            truncated = False
            chunked_req_offset = total_len
        else:
            truncated = remaining_tokens > self.rem_chunk_tokens
            if remaining_tokens == 0:
                chunked_req_offset = total_len
            else:
                chunked_req_offset = min(self.rem_chunk_tokens, remaining_tokens) + matched_tokens

        chunked_req.input_ids = chunked_req.origin_input_ids[:chunked_req_offset]
        chunked_req._effective_total_length = chunked_req_offset
        self.can_run_list.append(chunked_req)

        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= min(self.rem_chunk_tokens, budget_tokens)

        return chunked_req if truncated else None

    def add_one_req(self, req: Request) -> AddReqResult:
        if req.origin_input_ids is None:
            self.can_run_list.append(req)
            return AddReqResult.CONTINUE

        matched_tokens = self._matched_tokens(req)
        total_len = len(req.origin_input_ids)
        remaining_tokens = max(0, total_len - matched_tokens)
        budget_tokens = max(1, remaining_tokens)
        extend_input_len = (
            (remaining_tokens + self.page_size - 1) // self.page_size * self.page_size
            if remaining_tokens > 0
            else budget_tokens
        )

        if self.rem_chunk_tokens is None or extend_input_len <= self.rem_chunk_tokens:
            self.can_run_list.append(req)
            if self.rem_chunk_tokens is not None:
                self.rem_chunk_tokens -= extend_input_len
        else:
            trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
            if trunc_len <= 0:
                return AddReqResult.NO_TOKEN
            chunked_req_offset = trunc_len + matched_tokens
            req.input_ids = req.origin_input_ids[:chunked_req_offset]
            req._effective_total_length = chunked_req_offset
            self.can_run_list.append(req)
            self.new_chunked_req = req
            self.rem_chunk_tokens -= trunc_len

        if self.rem_chunk_tokens is None or self.rem_chunk_tokens <= 0:
            return AddReqResult.OTHER
        return AddReqResult.CONTINUE
