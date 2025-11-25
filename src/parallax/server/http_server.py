"""
This module contains the http frontend server for Parallax.
Two classes that handles a post request from the frontend service:

  -- ParallaxHttpServer:
    The uvicorn server that communicates with the frontend and posts responses
    to users.
    This module launches a subprocess.

  -- HTTPHandler:
    1.Gets requests from ParallaxHttpServer and maintains status of these requests.
    2.Send raw requests by ipc to parallax executor.
    3.Waits for ipc response from the executor and stores the results.
"""

import asyncio
import json
import multiprocessing as mp
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Dict, Optional

import fastapi
import uvicorn
import zmq
import zmq.asyncio
from fastapi.responses import ORJSONResponse, StreamingResponse
from mlx_lm.tokenizer_utils import StreamingDetokenizer
from mlx_lm.utils import load_config
from pydantic import BaseModel
from starlette.datastructures import State

from parallax.utils.selective_download import download_metadata_only
from parallax.utils.tokenizer_utils import load_detokenizer, load_tokenizer
from parallax.utils.utils import get_zmq_socket
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


def get_exception_traceback():
    """Traceback function to handle asyncio function errors"""
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        error_trace = get_exception_traceback()
        logger.error(f"TokenizerManager hit an exception: {error_trace}")
        sys.exit(1)


@dataclass
class HTTPRequestInfo:
    """HTTP Request information"""

    id: str
    text: str = ""
    stream: bool = False
    finish_reason: str = None
    object: str = "chat.completion"
    model: str = "default"
    create_time: float = 0.0
    update_time: float = 0.0
    logprobs: float = None
    matched_stop: int = None
    # usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # helper
    is_finish: bool = False
    # Queue for streaming tokens one by one
    token_queue: Optional[asyncio.Queue] = field(default=None, repr=False)
    detokenizer: StreamingDetokenizer = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    error_status: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR


class HTTPHandler:
    """
    A global handler that maintains raw requests. It has 2 main functions:
    1.Interprocess communicate with the model executor.
    2.Maintains the request -> prompts dict for ParallaxHTTPServer.
    """

    def __init__(
        self,
        executor_input_ipc_name,
        executor_output_ipc_name,
        model_path_str,
    ):
        self.asyncio_tasks = set()
        # Init inter-process communication
        context = zmq.asyncio.Context(2)
        self.send_to_executor = get_zmq_socket(context, zmq.PUSH, executor_input_ipc_name, True)
        self.recv_from_executor = get_zmq_socket(context, zmq.PULL, executor_output_ipc_name, True)
        self.processing_requests: Dict[str, HTTPRequestInfo] = {}

        # Load tokenizer for separate detokenizers.
        # Important: avoid triggering full weight downloads here.
        # Only download metadata/config/tokenizer files.
        from pathlib import Path

        if Path(model_path_str).exists():
            model_path = Path(model_path_str)
        else:
            model_path = download_metadata_only(model_path_str)
        config = load_config(model_path)
        self.model_path_str = model_path_str
        self.tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id", None))
        self.detokenizer_class, self.tokenmap = load_detokenizer(model_path, self.tokenizer)

    def create_request(self, request: Dict):
        """Creates a new request information"""
        rid = request["rid"]
        stream = request.get("stream", False)
        model = request.get("model", "default")
        chat_object = "chat.completion.chunk" if stream else "chat.completion"
        detokenizer = self.detokenizer_class(self.tokenizer, self.tokenmap)
        create_time = time.time()
        update_time = create_time
        request_info = HTTPRequestInfo(
            id=rid,
            stream=stream,
            model=model,
            object=chat_object,
            create_time=create_time,
            update_time=update_time,
            detokenizer=detokenizer,
        )
        if stream:
            request_info.token_queue = asyncio.Queue()
        self.processing_requests[rid] = request_info

    def release_request(self, rid: str):
        """Releases the request resources"""
        del self.processing_requests[rid]

    def send_request(self, request: Dict):
        """Sends the request to model executor using IPC."""
        self.send_to_executor.send_pyobj(request)

    def abort_request(self, request_id: str):
        """Sends abort request to executor for a specific request ID."""
        logger.info(f"Sending abort request for request ID: {request_id}")
        self.send_to_executor.send_pyobj({"type": "abort", "rid": request_id})

    async def stream_response_wrapper(self, rid):
        """Wraps the generator to handle client disconnects using a finally block."""
        generator = self.generate_stream_response(rid)
        try:
            async for chunk in generator:
                yield chunk
        finally:
            # This block executes when the client disconnects or the stream finishes.
            req_info = self.processing_requests.get(rid)
            # If the request is still in processing and not marked as finished, it means
            # the client disconnected midway.
            if req_info and not req_info.is_finish:
                logger.warning(f"Client disconnected for streaming request {rid}.")
                self.abort_request(rid)
                self.release_request(rid)
            elif req_info:
                self.release_request(rid)

    def _generate_stream_chunk(self, rid, token, is_first=False, is_last=False):
        """Generates a SSE chunk for a single token."""
        request_info = self.processing_requests[rid]

        if is_first:
            role = "assistant"
            content = ""
            if "minimax-m2" in self.model_path_str.lower():
                content = "<think>"
        elif is_last:
            role = None
            content = None
        else:
            role = None
            content = token

        response = {
            "id": rid,
            "object": "chat.completion.chunk",
            "model": request_info.model,
            "created": request_info.create_time,
            "choices": [
                {
                    "index": 0,
                    "logprobs": request_info.logprobs,
                    "finish_reason": request_info.finish_reason if is_last else None,
                    "matched_stop": request_info.matched_stop,
                },
            ],
            "usage": {
                "prompt_tokens": request_info.prompt_tokens,
                "total_tokens": request_info.prompt_tokens + request_info.completion_tokens,
                "completion_tokens": request_info.completion_tokens,
            },
        }
        choice = response["choices"][0]
        choice["delta"] = {"role": role, "content": content}
        response_json = json.dumps(response, separators=(",", ":"))
        return f"data: {response_json}\n\n".encode()

    def _generate_error_stream_chunk(self, rid, error_payload: Dict[str, str]):
        """Generates a SSE chunk representing an error."""
        request_info = self.processing_requests[rid]
        response = {
            "id": rid,
            "object": request_info.object,
            "model": request_info.model,
            "created": request_info.create_time,
            "error": error_payload,
        }
        response_json = json.dumps(response, separators=(",", ":"))
        return f"data: {response_json}\n\n".encode()

    async def generate_stream_response(self, rid):
        """Generates a streaming response by consuming from a token queue."""
        # Send first chunk with role
        yield self._generate_stream_chunk(rid, None, is_first=True)

        request_info = self.processing_requests.get(rid)
        if not request_info or not request_info.stream:
            return

        while True:
            token = await request_info.token_queue.get()
            if token is None:  # End of stream sentinel
                break
            if isinstance(token, dict) and token.get("type") == "error":
                yield self._generate_error_stream_chunk(rid, token.get("payload", {}))
                continue
            yield self._generate_stream_chunk(rid, token)

        # Send final chunk with finish reason
        yield self._generate_stream_chunk(rid, None, is_last=True)
        yield b"data: [DONE]\n\n"

    def generate_non_stream_response(self, rid):
        """Generates a non-streaming response"""
        request_info = self.processing_requests[rid]
        response = {
            "id": rid,
            "object": request_info.object,
            "model": request_info.model,
            "created": request_info.create_time,
            "choices": [
                {
                    "index": 0,
                    "logprobs": request_info.logprobs,
                    "finish_reason": request_info.finish_reason,
                    "matched_stop": request_info.matched_stop,
                },
            ],
            "usage": {
                "prompt_tokens": request_info.prompt_tokens,
                "total_tokens": request_info.prompt_tokens + request_info.completion_tokens,
                "completion_tokens": request_info.completion_tokens,
            },
        }
        choice = response["choices"][0]
        choice["message"] = {
            "role": "assistant",
            "content": request_info.text,
            "reasoning_content": None,
            "tool_calls": None,
        }
        return response

    async def _handle_executor_error(self, rid: str, recv_dict: Dict):
        """Handles error notifications sent from the executor process."""
        request_info = self.processing_requests.get(rid)
        if request_info is None:
            return

        message = recv_dict.get("error", "Unknown error")
        err_type = recv_dict.get("error_type", "InternalServerError")
        status_code = recv_dict.get("status_code", HTTPStatus.BAD_REQUEST.value)
        try:
            status = HTTPStatus(status_code)
        except ValueError:
            status = HTTPStatus.BAD_REQUEST

        request_info.error_message = message
        request_info.error_type = err_type
        request_info.error_status = status
        request_info.finish_reason = "error"
        request_info.is_finish = True

        if request_info.stream and request_info.token_queue is not None:
            payload = {
                "message": message,
                "type": err_type,
                "code": status.value,
            }
            await request_info.token_queue.put({"type": "error", "payload": payload})
            await request_info.token_queue.put(None)

    async def _handle_loop(self):
        """The event loop that handles returned requests"""
        while True:
            recv_dict = await self.recv_from_executor.recv_pyobj()
            rid = recv_dict["rid"]
            if rid not in self.processing_requests:
                continue

            if recv_dict.get("type") == "error":
                await self._handle_executor_error(rid, recv_dict)
                continue

            request_info = self.processing_requests[rid]
            request_info.update_time = time.time()
            request_info.prompt_tokens = recv_dict["prompt_tokens"]
            next_token_id = recv_dict["next_token_id"]
            request_info.completion_tokens += 1
            request_info.detokenizer.add_token(next_token_id)
            output = request_info.detokenizer.last_segment

            is_finished = recv_dict.get("eos", False) or recv_dict.get("length", False)

            # Only process and send non-EOS tokens
            if not is_finished and len(output) > 0:
                # Accumulate full text for non-streaming and potentially for logging
                request_info.text += output

                # For streaming, put the individual token into the queue.
                if request_info.stream:
                    await request_info.token_queue.put(output)

            # If it is the end of the stream, update status and send sentinel
            if is_finished:
                if recv_dict.get("length", False):
                    logger.debug(f"Request {rid} finished with length")
                    request_info.finish_reason = "length"
                elif recv_dict.get("eos", False):
                    logger.debug(f"Request {rid} finished with eos")
                    request_info.finish_reason = "eos"
                    request_info.matched_stop = next_token_id
                else:
                    logger.debug(f"Request {rid} finished with unknown reason")
                    request_info.finish_reason = "unknown"

                request_info.is_finish = True
                if request_info.stream:
                    await request_info.token_queue.put(None)  # Sentinel for stream end

    async def create_handle_loop(self):
        """Create asyncio event loop task function"""
        task_loop = asyncio.create_task(print_exception_wrapper(self._handle_loop))
        await task_loop


class ErrorResponse(BaseModel):
    """An Error data structure."""

    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


def create_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
):
    """Creates a json error response for the frontend."""
    error = ErrorResponse(message=message, type=err_type, code=status_code.value)
    return ORJSONResponse(content=error.model_dump(), status_code=error.code)


# Fast API
app = fastapi.FastAPI(
    openapi_url="/openapi.json",
)


async def init_app_states(
    state: State, executor_input_ipc: str, executor_output_ipc: str, model_path: str
):
    """Init FastAPI app states, including http handler, etc."""
    state.http_handler = HTTPHandler(
        executor_input_ipc,
        executor_output_ipc,
        model_path,
    )


async def v1_chat_completions(raw_request: fastapi.Request):
    """
    Handles the v1/chat/completions requests asynchronously.
    It gets the prompts from HTTPHandler and returns to the frontend.
    """
    try:
        request_json = await raw_request.json()
    except Exception as e:
        return create_error_response("Invalid request body, error: ", str(e))

    # Check if request_json has "rid", otherwise generate new one
    request_id = request_json.get("rid")
    if request_id is None:
        request_id = str(uuid.uuid4())
        request_json["rid"] = request_id

    app.state.http_handler.create_request(request_json)
    app.state.http_handler.send_request(request_json)
    req = app.state.http_handler.processing_requests.get(request_id)
    if req is None:
        return create_error_response("Request not found", "RequestNotFoundError")
    is_stream = req.stream

    if is_stream:
        return StreamingResponse(
            app.state.http_handler.stream_response_wrapper(request_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        try:
            # For non-streaming requests, we poll for completion. This is simpler
            # than event-based synchronization and acceptable for requests that
            # are expected to be relatively short.
            while True:
                # Check if client is still connected
                if await raw_request.is_disconnected():
                    logger.warning(f"Client disconnected for non-streaming request {request_id}")
                    if request_id in app.state.http_handler.processing_requests:
                        app.state.http_handler.abort_request(request_id)
                        app.state.http_handler.release_request(request_id)
                    return create_error_response("Client disconnected", "ClientDisconnectedError")

                await asyncio.sleep(0.01)
                req = app.state.http_handler.processing_requests.get(request_id)
                if req is None:  # Request might have been cleaned up due to error
                    return create_error_response("Request not found", "RequestNotFoundError")
                is_finish = req.is_finish
                if is_finish:
                    break
            if req.error_message:
                response = create_error_response(
                    req.error_message,
                    req.error_type or "InternalServerError",
                    status_code=req.error_status,
                )
                app.state.http_handler.release_request(request_id)
                return response

            response = app.state.http_handler.generate_non_stream_response(request_id)
            app.state.http_handler.release_request(request_id)
            return ORJSONResponse(status_code=200, content=response)
        except Exception as e:
            # Handle any unexpected errors during processing
            logger.error(f"Error processing non-streaming request {request_id}: {e}")
            if request_id in app.state.http_handler.processing_requests:
                logger.info(f"Sending abort request due to error: {request_id}")
                app.state.http_handler.abort_request(request_id)
                app.state.http_handler.release_request(request_id)
            return create_error_response("Internal server error", "InternalServerError")


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: fastapi.Request):
    """OpenAI v1/chat/complete post function"""
    return await v1_chat_completions(raw_request)


class ParallaxHttpServer:
    """
    The frontend http server drived by asyncio.
    Since uvicorn API runs an asyncio task, we need a new uvicorn
    wrapper to run other async tasks.
    """

    def __init__(self, args):
        self.host = args.host
        self.port = args.port
        self.executor_input_ipc_name = args.executor_input_ipc
        self.executor_output_ipc_name = args.executor_output_ipc
        self.model_path = args.model_path

    async def run_uvicorn(self):
        """
        Since uvicorn.run() uses asyncio.run, we need another wrapper
        to create a uvicorn asyncio task to run multiple tasks.
        """
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            timeout_keep_alive=5,
            loop="uvloop",
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def run_tasks(self):
        """Gather results of all asyncio tasks"""
        await asyncio.gather(self.run_uvicorn(), app.state.http_handler.create_handle_loop())

    def run(self):
        """
        Launch A FastAPI server that routes requests to the executor.

        Note:
        1. The HTTP server and executor both run in the main process.
        2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
        """
        asyncio.run(
            init_app_states(
                app.state,
                self.executor_input_ipc_name,
                self.executor_output_ipc_name,
                self.model_path,
            )
        )
        asyncio.run(self.run_tasks())


def launch_http_server(args):
    """
    Launch function of frontend server.
    It creates a sub-process for the http server.
    """
    http_server = ParallaxHttpServer(args)
    process = mp.Process(target=http_server.run)
    process.start()
    return process


def stop_http_server(http_server_process):
    """
    Stop HTTP server process if it exists.
    """
    if http_server_process is not None:
        logger.info("Stopping HTTP server process...")
        try:
            http_server_process.kill()
            http_server_process.join()
        except Exception as e:
            logger.error(f"Failed to terminate HTTP server process: {e}")
        return None
    return http_server_process


def restart_http_server(args, http_server_process):
    """
    Restart HTTP server with new args.
    Stops the old server if it exists and starts a new one.
    """
    http_server_process = stop_http_server(http_server_process)
    logger.info("Restarting HTTP server...")
    return launch_http_server(args)
