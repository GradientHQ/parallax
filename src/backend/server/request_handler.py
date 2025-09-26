from typing import Dict

import aiohttp
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from backend.server.constants import NODE_STATUS_AVAILABLE
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)


class RequestHandler:
    """HTTP request forwarder with scheduler-aware routing and retry logic.

    Behavior for routing resolution:
    - routing_table is None: scheduler has not decided yet -> treat as error for this attempt
    - routing_table is []: all pipelines are full now -> retry up to max attempts
    - routing_table is non-empty: forward to first hop
    """

    MAX_ROUTING_RETRY = 20
    RETRY_DELAY_SEC = 5

    def __init__(self):
        self.scheduler_manage = None

    def set_scheduler_manage(self, scheduler_manage):
        self.scheduler_manage = scheduler_manage

    async def _forward_request(
        self, endpoint: str, request_data: Dict, request_id: str, received_ts: int
    ):
        logger.info(
            f"Forwarding request {request_id} to endpoint {endpoint}; stream={request_data.get('stream', False)}"
        )
        if (
            self.scheduler_manage is None
            or not self.scheduler_manage.get_schedule_status() == NODE_STATUS_AVAILABLE
        ):
            return JSONResponse(
                content={"error": "Server is not ready"},
                status_code=500,
            )

        # Try to resolve routing; retry if table is an empty list (capacity full)
        attempts = 0
        routing_table = None
        while attempts < self.MAX_ROUTING_RETRY:
            try:
                routing_table = self.scheduler_manage.get_routing_table(request_id, received_ts)
                logger.info(
                    f"get_routing_table for request {request_id} return: {routing_table} (attempt {attempts+1})"
                )
            except Exception as e:
                logger.exception(f"get_routing_table error: {e}")
                return JSONResponse(
                    content={"error": "Routing table not found"},
                    status_code=500,
                )

            # None -> scheduler has not set yet; treat as hard error (no waiting here)
            if routing_table is None:
                return JSONResponse(
                    content={"error": "Routing not ready"},
                    status_code=503,
                )

            # Non-empty -> proceed
            if len(routing_table) > 0:
                break

            # Empty list -> capacity full now, retry after short delay
            attempts += 1
            if attempts < self.MAX_ROUTING_RETRY:
                # small async delay before re-forwarding
                import asyncio

                await asyncio.sleep(self.RETRY_DELAY_SEC)

        # If still empty after retries, return 429 Too Many Requests
        if routing_table is not None and len(routing_table) == 0:
            return JSONResponse(
                content={"error": "All pipelines are busy. Please retry later."},
                status_code=429,
            )

        request_data["routing_table"] = routing_table
        call_url = self.scheduler_manage.get_call_url_by_node_id(routing_table[0])
        logger.info(
            f"Resolved call_url for request {request_id}: node={routing_table[0]} -> {call_url}"
        )

        if not call_url:
            return JSONResponse(
                content={"error": "Call url not found of peer id: " + routing_table[0]},
                status_code=500,
            )

        url = call_url + endpoint
        is_stream = request_data.get("stream", False)
        logger.info(f"POST upstream: url={url}, stream={is_stream}")

        async def _process_upstream_response(response: aiohttp.ClientResponse):
            logger.info(f"post: {request_id}, code: {response.status}, params: {request_data}")
            if response.status != 200:
                error_text = await response.text()
                error_msg = (
                    f"Upstream service returned status {response.status}, response: {error_text}"
                )
                logger.error(f"completions error: {error_msg}, request_id: {request_id}")
                raise HTTPException(status_code=response.status, detail=error_msg)

        if is_stream:

            async def stream_generator():
                async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
                    async with session.post(url, json=request_data) as response:
                        await _process_upstream_response(response)

                        async for chunk in response.content:
                            if chunk:
                                yield chunk

            resp = StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "X-Content-Type-Options": "nosniff",
                    "Cache-Control": "no-cache",
                },
            )
            logger.info(f"Streaming response initiated for {request_id}")
            return resp
        else:
            async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
                async with session.post(url, json=request_data) as response:
                    await _process_upstream_response(response)
                    result = await response.json()
                    logger.info(f"Non-stream response completed for {request_id}")
                    return JSONResponse(content=result)

    async def v1_completions(self, request_data: Dict, request_id: str, received_ts: int):
        return await self._forward_request("/v1/completions", request_data, request_id, received_ts)

    async def v1_chat_completions(self, request_data: Dict, request_id: str, received_ts: int):
        return await self._forward_request(
            "/v1/chat/completions", request_data, request_id, received_ts
        )
