from enum import Enum, auto

from mpmath import extend
from parallax.server.request import Request


class AddReqResult(Enum):
    CONTINUE = auto()  # Continue to add requests
    NO_TOKEN = auto()  # No token left
    OTHER = auto()  # Other reasons to stop adding requests

class MACPrefillAdder:
    """
    MACPrefillAdder is a class that adds prefill requests to the MAC prefill batch.
    """
    def __init__(
        self,
        page_size: int,
        rem_chunk_tokens: int,
        chunked_req_offset: int
    ):
        self.page_size = page_size
        self.rem_chunk_tokens = rem_chunk_tokens
        self.can_run_list = []
        self.new_chunked_req = None
        self.chunked_req_offset = chunked_req_offset
    def add_chunked_req(self, chunked_req: Request) -> Request:
        if chunked_req is None:
            return None
        extend_input_len = len(chunked_req.origin_input_ids) - self.chunked_req_offset
        truncated = extend_input_len > self.rem_chunk_tokens
        self.chunked_req_offset += min(self.rem_chunk_tokens, extend_input_len)
        self.chunked_req_offset = min(self.chunked_req_offset, len(chunked_req.origin_input_ids))
        chunked_req.input_ids = chunked_req.origin_input_ids[: self.chunked_req_offset]
        chunked_req.total_len = self.chunked_req_offset
        self.can_run_list.append(chunked_req)
        self.rem_chunk_tokens -= min(self.rem_chunk_tokens, extend_input_len)
        self.chunked_req_offset = 0 if not truncated else self.chunked_req_offset
        return chunked_req if truncated else None
    
    def add_one_req(self, req: Request) -> AddReqResult:
        extend_input_len = len(req.origin_input_ids) - self.chunked_req_offset
        # align to page size
        extend_input_len = (extend_input_len + self.page_size - 1) // self.page_size * self.page_size
        if self.rem_chunk_tokens is None or extend_input_len <= self.rem_chunk_tokens:
            self.chunked_req_offset += extend_input_len
            self.chunked_req_offset = min(self.chunked_req_offset, len(req.origin_input_ids))
            req.input_ids = req.origin_input_ids[: self.chunked_req_offset]
            req.total_len = self.chunked_req_offset
            self.can_run_list.append(req)
            self.chunked_req_offset = 0
            self.rem_chunk_tokens -= extend_input_len
        else:
            # make sure at least one page is available
            trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
            if trunc_len <= 0:
                return AddReqResult.OTHER
            extend_input_len = trunc_len
            self.chunked_req_offset = extend_input_len
            self.chunked_req_offset = min(self.chunked_req_offset, len(req.origin_input_ids))
            req.input_ids = req.origin_input_ids[: self.chunked_req_offset]
            req.total_len = self.chunked_req_offset
            self.can_run_list.append(req)
            self.new_chunked_req = req
            self.rem_chunk_tokens -= extend_input_len
        if self.rem_chunk_tokens is None or self.rem_chunk_tokens <= 0:
            return AddReqResult.OTHER
        return AddReqResult.CONTINUE