"""
A minimum implementation of Request objects for managing inference requests.

Each time a client initialize a generation request by sending prompt texts to the First Peer,
    it will initialize an `InitialRequest` instance.
Each Intermediate Peer will only hold `IntermediateRequest`,
    which contains part of the info of `InitialRequest`.
        * request id,
        * current position.
        * relevant hidden_states (prompt_len, hidden_size) or (1, hidden_size)
    We will pack `IntermediateRequest` and send to the following peers.

In our current design, the Last Peer won't hold `InitialRequest` for several reasons:
    * in case of weight tying, it doens't need to load and compute lm_head GEMM;
    * easier management for Request state updates (only the First Peer will handle it);
    * no need to hold tokenizer so that we can decokenize first and then stream to Client;
    * we need to announce to the First Peer to evict the request from its running batch anyway.
    - slight delay for client to recieve the streaming token.

Forming requests to batches (see scheduler.py for details):
    Our scheduler can manage both `InitialRequest` and `IntermediateRequest`.

A complete workflow:
Prefill:
    First Peer:
        * Initialize `InitialRequest`;
        (* Put the request into the schduler's pool;)
        * Feed `input_ids` into executor;
        * Store each layer's hidden states to paged KV Cache;
        * Generates the last layer (that it holds)'s hidden states;
        * Build `IntermediateRequest` with the request id,
            current position, and hidden states (prompt_len, hidden_size);
        * Sends `IntermediateRequest` to the next Peer.
    Intermediate Peers:
        * Accepts `IntermediateRequest`;
        * Feed `hidden_states` (prompt_len, hidden_size) to executor, updates caches
        * Update `IntermediateRequest` with the hidden states and send it to the next peer;
    Last Peer:
        * Accepts `IntermediateRequest`;
        * Generates and sends the first `output_id` (without weight tying) to the first Peer.

    Now, the first peer will
        1. Detokenize the request and stream it to the client;
        2. Switch request status from Prefill to Decode
        3. Add output token id to the ouputs and increase the processed length.
        4. Check if the request should be switch to Finished state, evict if necessary.

Decode:
    First Peer: Feed singleton `output_id` to executor,
        generates and sends `hidden_states` (1, hidden_size);
    Intermediate Peers: takes in `hidden_states`,
        generates and sends `hidden_states` (1, hidden_size);
    Last Peer: takes in `hidden_states`,
        generates and sends `output_id` to the first Peer.

TODO:
    1. Add support for multiple output_ids in a single step (e.g. beam width, top-k sampling, etc.);
    2. Accepts more generation configs like repetition penalties.
"""

import uuid
from enum import Enum
from typing import Any, List, Optional

from parallax.server.sampling.sampling_params import SamplingParams
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class RequestStatus(Enum):
    """Enumeration of possible request statuses for the First Peer."""

    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    FINISHED_EOS = "FINISHED_EOS"
    FINISHED_MAX_LENGTH = "FINISHED_MAX_LENGTH"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class Request:
    """
    Base class for requests in the Parallax server.
    This is a placeholder and can be extended for specific request types.
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        status: RequestStatus = RequestStatus.PREFILLING,
        prompt_len: int = 0,
        input_ids: Optional[List[int]] = None,
        output_ids: Optional[List[int]] = None,
        routing_table: Optional[List[str]] = [],
        sampling_params: Optional[SamplingParams] = None,
    ):
        self.request_id = request_id or str(uuid.uuid4())
        self.status = status
        self.prompt_len = prompt_len
        self.output_ids = output_ids or []
        self.input_ids = input_ids or []
        self.routing_table = routing_table
        self.sampling_params = sampling_params or SamplingParams()
        self.abort = False
        self.ready_for_next_step = False
        self.last_updated_time: Optional[float] = None

    @property
    def is_finished(self) -> bool:
        """Checks if the request has finished processing."""
        return self.status in [
            RequestStatus.FINISHED_EOS,
            RequestStatus.FINISHED_MAX_LENGTH,
            RequestStatus.ERROR,
            RequestStatus.CANCELLED,
        ]

    @property
    def is_prefill(self) -> bool:
        """Checks if the request is in the prefill stage."""
        return self.status == RequestStatus.PREFILLING

    @property
    def is_decoding(self) -> bool:
        """Checks if the request is in the decoding stage."""
        return self.status == RequestStatus.DECODING

    def update_status(self, new_status: RequestStatus = RequestStatus.DECODING):
        """
        Update the status of the request.
        """
        if self.is_finished:
            logger.warning(
                f"Request {self.request_id}: Attempted to update status of a finished request."
            )
            return
        self.status = new_status
        logger.debug(f"Request {self.request_id} status updated to {self.status}.")


class InitialRequest(Request):
    """
    Represents the full state of a user's generation request, managed by the First Peer.
    """

    def __init__(
        self,
        prompt: Optional[str] = None,
        request_id: Optional[str] = None,
        output_ids: Optional[List[int]] = None,
        input_ids: Optional[List[int]] = None,
        sampling_params: Optional[SamplingParams] = None,
        max_new_tokens: int = 512,
        max_total_length: int = 1024,
        status: RequestStatus = RequestStatus.PREFILLING,
    ):
        if not prompt and not input_ids:
            raise ValueError("prompt or input_ids cannot be empty.")
        super().__init__(
            request_id=request_id,
            status=status,
            prompt_len=len(input_ids) if input_ids else 0,
            input_ids=input_ids,
            sampling_params=sampling_params,
        )
        self.prompt = prompt

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")
        self.max_new_tokens = max_new_tokens
        self.max_total_length = max_total_length
        self.output_ids = output_ids or []
        self.hidden_states = None

        if len(self.output_ids) > 0 and self.status == RequestStatus.PREFILLING:
            raise ValueError(f"Cannot initialize with output_ids given {self.status}.")

    @property
    def input_length(self) -> int:
        """Length of the input sequence (input_ids)."""
        if self.input_ids is None:
            raise ValueError("Cannot get input length before tokenization.")
        return len(self.input_ids)

    @property
    def output_length(self) -> int:
        """Length of the generated output (output_ids)."""
        return len(self.output_ids)

    @property
    def total_length(self) -> int:
        """Total length of the sequence (input + output)."""
        return self.prompt_len + self.output_length

    def get_model_input_for_first_peer(self) -> List[int]:
        """
        Returns the token IDs the First Peer's model should process for the current step.
        """
        if self.input_ids is None:
            raise ValueError("Cannot get model input before tokenization.")

        if not self.output_ids:
            return self.input_ids
        return [self.output_ids[-1]]

    def commit_new_token(self, token_id: int):
        """
        Called by the First Peer when a new token is received from the Last Peer.
        Appends the token, updates length, and checks finishing conditions.
        """
        if self.is_finished:
            logger.warning(
                f"Request {self.request_id}: Attempted to commit token to a finished request."
            )
            return

        self.output_ids.append(token_id)

        # Finishing condition checks are now handled by the Scheduler.
        if self.status == RequestStatus.PREFILLING:
            self.status = RequestStatus.DECODING

    @classmethod
    def from_prompt_ids(
        cls,
        prompt_ids: List[int],
        max_new_tokens: int,
        max_total_length: int,
    ) -> "InitialRequest":
        """
        Convert a prompt string to an InitialRequest.
        """
        return cls(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            max_total_length=max_total_length,
        )


class IntermediateRequest(Request):
    """
    Lightweight data packet sent between intermediate peers in the pipeline.
    This is what gets packed and sent over the network.

    # TODO: add attention_mask, logits...
    """

    def __init__(
        self,
        request_id: str,
        current_position: int,
        status: RequestStatus = RequestStatus.PREFILLING,
        input_ids: Optional[List[int]] = None,
        hidden_states: Optional[Any] = None,
        next_token_id: Optional[int] = None,
        routing_table: Optional[List[str]] = [],
        sampling_params: Optional[SamplingParams] = None,
    ):
        super().__init__(
            request_id=request_id,
            status=status,
            routing_table=routing_table,
            input_ids=input_ids,
            sampling_params=sampling_params,
        )
        # Hidden states from the previous peer's computation.
        # Shape:
        #   prefill: (prompt_len, hidden_dim)
        #   decode: (1, hidden_dim)
        # For data sent from Last Peer to First Peer, this can also be a single token_id
        # wrapped in a numpy array, e.g., np.array([token_id]).
        if not self.is_finished and hidden_states is None:
            raise ValueError(f"hidden_states cannot be None for unfinished request {request_id}.")

        self.current_position = current_position
        self.hidden_states = hidden_states
        self.next_token_id = next_token_id

    @property
    def input_length(self) -> int:
        """Length of the input sequence (hidden_states)."""
        assert self.is_prefill
        return self.current_position

    @property
    def total_length(self) -> int:
        """Total length of the sequence (input + output)."""
        return self.current_position

    @classmethod
    def from_initial_request(
        cls, initial_request: InitialRequest, hidden_states: Optional[Any] = None
    ) -> "IntermediateRequest":
        """Convert an InitialRequest to an IntermediateRequest.

        Pack hidden states and set current position.
        This is typically used by the First Peer to start the pipeline.

        Args:
            initial_request: The initial request to convert.
            hidden_states: The hidden states from the previous peer.
                If None, it indicates that the request is finished.

        Returns:
            An IntermediateRequest instance with the request ID,
            status, current position, and hidden states from InitialRequest.
        """
        if hidden_states is None:
            assert initial_request.is_finished, "Hidden states can't be None for unfinished request"
        if initial_request.output_ids is None or len(initial_request.output_ids) == 0:
            next_token_id = None
        else:
            next_token_id = initial_request.output_ids[-1]

        return IntermediateRequest(
            request_id=initial_request.request_id,
            status=initial_request.status,
            input_ids=initial_request.input_ids,
            next_token_id=next_token_id,
            current_position=initial_request.total_length,
            hidden_states=hidden_states,
            sampling_params=initial_request.sampling_params,
            routing_table=initial_request.routing_table,
        )

    @classmethod
    def from_intermediate_request(
        cls,
        old_request: "IntermediateRequest",
        new_hidden_states: Any,
    ) -> "IntermediateRequest":
        """
        Creates a new IntermediateRequest from an old one, with updated hidden states.
        This is used by intermediate peers to pass the request along the pipeline.
        """
        return IntermediateRequest(
            request_id=old_request.request_id,
            status=old_request.status,
            current_position=old_request.total_length,
            input_ids=old_request.input_ids,
            next_token_id=old_request.next_token_id,
            hidden_states=new_hidden_states,
            routing_table=old_request.routing_table,
            sampling_params=old_request.sampling_params,
        )

    def __repr__(self):
        fields = [
            f"request_id={self.request_id}",
            f"status={self.status}",
            f"current_position={self.current_position}",
            f"input_ids={self.input_ids}",
            f"hidden_states={self.hidden_states}",
            f"routing_table={self.routing_table}",
        ]

        if self.hidden_states is not None:
            fields.append(f"hidden_states_shape={self.hidden_states.shape}")

        fields.append(f"next_token_id={self.next_token_id}")

        field_str = ",\n    ".join(fields)
        return f"IntermediateRequest(\n    {field_str}\n)"
