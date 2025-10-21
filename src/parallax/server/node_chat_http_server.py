import asyncio
import json
import multiprocessing as mp
import time
from typing import Dict

import fastapi
import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from lattica import Lattica
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import State

from backend.server.rpc_connection_handler import RPCConnectionHandler
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

import uuid

# Fast API
app = fastapi.FastAPI(
    openapi_url="/openapi.json",
)


async def init_app_states(state: State, node_chat_http_server):
    """Init FastAPI app states, including http handler, etc."""
    state.http_server = node_chat_http_server


async def v1_chat_completions(request_data: Dict, request_id: str, received_ts: int):
    """
    Handles the v1/chat/completions requests asynchronously.
    It gets the prompts from HTTPHandler and returns to the frontend.
    """
    return app.state.http_server.chat_completion(request_data, request_id, received_ts)


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: Request):
    """OpenAI v1/chat/complete post function"""
    request_data = await raw_request.json()
    request_id = uuid.uuid4()
    received_ts = time.time()
    return await v1_chat_completions(request_data, request_id, received_ts)


class NodeChatHttpServer:

    def __init__(self, args):
        self.host = args.host
        self.port = args.node_chat_port
        self.tcp_port = args.tcp_port
        self.udp_port = args.udp_port
        self.scheduler_addr = args.scheduler_addr
        self.relay_servers = args.relay_servers
        self.announce_maddrs = args.announce_maddrs
        self.initial_peers = args.initial_peers
        self.host_maddrs = (
            [f"/ip4/0.0.0.0/tcp/{self.tcp_port}", f"/ip4/0.0.0.0/udp/{self.udp_port}/quic-v1"],
        )
        self.scheduler_peer_id = None
        self.scheduler_stub = None
        self.lattica = None

    def build_lattica(self):
        self.lattica = Lattica.builder().with_listen_addrs(self.host_maddrs)

        if self.scheduler_addr is not None and self.scheduler_addr != "auto":
            if self.scheduler_addr.startswith("/"):
                logger.info(f"Using scheduler addr: {self.scheduler_addr}")
                self.lattica.with_bootstraps([self.scheduler_addr])
            self.scheduler_peer_id = self.scheduler_addr.split("/")[-1]

        if len(self.relay_servers) > 0:
            logger.info(f"Using relay servers: {self.relay_servers}")
            self.lattica.with_relay_servers(self.relay_servers).with_dcutr(True)
            if self.scheduler_peer_id is not None:
                logger.info(f"Using protocol: /{self.scheduler_peer_id}")
                self.lattica.with_protocol("/" + self.scheduler_peer_id)

        if len(self.announce_maddrs) > 0:
            logger.info(f"Using announce maddrs: {self.announce_maddrs}")
            self.lattica.with_external_addrs(self.announce_maddrs)

        if len(self.initial_peers) > 0:
            logger.info(f"Using initial peers: {self.initial_peers}")
            self.lattica.with_bootstraps(self.initial_peers)

        self.lattica.build()

        if self.scheduler_addr == "auto":
            self.scheduler_peer_id = None
            for _ in range(20):
                try:
                    time.sleep(3)
                    self.scheduler_peer_id = self.lattica.get("scheduler_peer_id")
                    if self.scheduler_peer_id is not None:
                        self.scheduler_peer_id = self.scheduler_peer_id.value
                        logger.info(f"Found scheduler peer id: {self.scheduler_peer_id}")
                        break
                    logger.info(
                        f"Discovering scheduler peer id, {_ + 1} times, you can specify scheduler peer id by -s"
                    )
                except Exception as e:
                    logger.warning(f"Failed to get scheduler addr: {e}, waiting for 3 seconds.")
            if self.scheduler_peer_id is None:
                logger.error("Failed to get scheduler peer id")
                return False

        return True

    def chat_completion(self, request_data, request_id: str, received_ts: int):
        if self.scheduler_addr is not None:  # central scheduler mode
            try:
                self.scheduler_stub = RPCConnectionHandler(self.lattica, None).get_stub(
                    self.scheduler_peer_id
                )
                stub = self.scheduler_stub
                is_stream = request_data.get("stream", False)
                try:
                    if is_stream:

                        async def stream_generator():
                            response = stub.chat_completion(request_data)
                            try:
                                iterator = iterate_in_threadpool(response)
                                async for chunk in iterator:
                                    yield chunk
                            finally:
                                logger.debug(f"client disconnected for {request_id}")
                                response.cancel()

                        resp = StreamingResponse(
                            stream_generator(),
                            media_type="text/event-stream",
                            headers={
                                "X-Content-Type-Options": "nosniff",
                                "Cache-Control": "no-cache",
                            },
                        )
                        logger.debug(f"Streaming response initiated for {request_id}")
                        return resp
                    else:
                        response = stub.chat_completion(request_data)
                        response = next(response).decode()
                        logger.debug(f"Non-stream response completed for {request_id}")
                        # response is a JSON string; parse to Python object before returning
                        return JSONResponse(content=json.loads(response))
                except Exception as e:
                    logger.exception(f"Error in _forward_request: {e}")
                    return JSONResponse(
                        content={"error": "Internal server error"},
                        status_code=500,
                    )

            except Exception as e:
                logger.exception(f"Error in chat completion: {e}")
                return JSONResponse(
                    content={"error": "Internal server error"},
                    status_code=500,
                )
        else:
            logger.error("No scheduler address specified")
            return JSONResponse(
                content={"error": "No scheduler address specified"},
                status_code=500,
            )

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

    def run(self):
        """
        Launch A FastAPI server that routes requests to the executor.

        Note:
        1. The HTTP server and executor both run in the main process.
        2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
        """
        if not self.build_lattica():
            logger.error("Failed to build lattica")
            exit(1)

        asyncio.run(
            init_app_states(
                app.state,
                self,
            )
        )
        asyncio.run(self.run_uvicorn())


def launch_node_chat_http_server(args):
    """
    Launch function of node chat http server.
    It creates a sub-process for the http server.
    """
    node_chat_http_server = NodeChatHttpServer(args)
    process = mp.Process(target=node_chat_http_server.run)
    process.start()
    return process
