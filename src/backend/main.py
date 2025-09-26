import time
import uuid
import asyncio
import json

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backend.server.request_handler import RequestHandler
from backend.server.scheduler_manage import SchedulerManage
from backend.server.server_args import parse_args
from parallax_utils.logging_config import get_logger
from backend.server.static_config import get_model_list, get_node_join_command

app = FastAPI()

logger = get_logger(__name__)

scheduler_manage = None
request_handler = RequestHandler()


@app.get("/")
async def get():
    return {"message": "Hello, World!"}


@app.get("/hello")
async def hello():
    return {"message": "Hello, World!"}


@app.get("/model/list")
async def model_list():
    return JSONResponse(
        content={
            "type": "model_list",
            "data": get_model_list(),
        },
        status_code=200,
    )


@app.post("/scheduler/init")
async def scheduler_init(raw_request: Request):
    request_data = await raw_request.json()
    model_name = request_data.get("model_name")
    init_nodes_num = request_data.get("init_nodes_num")
    is_local_network = request_data.get("is_local_network")
    if scheduler_manage.is_running():
        # todo reinit
        pass
    else:
        scheduler_manage.run(model_name, init_nodes_num, is_local_network)
    return JSONResponse(
        content={
            "type": "scheduler_init",
            "data": None,
        },
        status_code=200,
    )


@app.get("/node/join/command")
async def node_join_command():
    model_name = scheduler_manage.get_model_name()
    is_local_network = scheduler_manage.get_is_local_network()

    return JSONResponse(
        content={
            "type": "node_join_command",
            "data": get_node_join_command(model_name, "${scheduler_addr}", is_local_network),
        },
        status_code=200,
    )


@app.get("/cluster/status")
async def cluster_status():
    async def stream_cluster_status():
        while True:
            yield json.dumps(scheduler_manage.get_cluster_status(), ensure_ascii=False) + "\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        stream_cluster_status(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/v1/completions")
async def openai_v1_completions(raw_request: Request):
    request_data = await raw_request.json()
    request_id = uuid.uuid4()
    received_ts = time.time()
    return await request_handler.v1_completions(request_data, request_id, received_ts)


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: Request):
    request_data = await raw_request.json()
    request_id = uuid.uuid4()
    received_ts = time.time()
    return await request_handler.v1_chat_completions(request_data, request_id, received_ts)


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"args: {args}")

    host_maddrs = args.host_maddrs
    dht_port = args.dht_port
    if args.dht_port is not None:
        assert host_maddrs is None, "You can't use --dht-port and --host-maddrs at the same time"
    else:
        dht_port = 0
    if host_maddrs is None:
        host_maddrs = [f"/ip4/0.0.0.0/tcp/{dht_port}", f"/ip6/::/tcp/{dht_port}"]

    scheduler_manage = SchedulerManage(
        initial_peers=args.initial_peers,
        relay_servers=args.relay_servers,
        dht_prefix=args.dht_prefix,
        host_maddrs=host_maddrs,
        announce_maddrs=args.announce_maddrs,
    )

    request_handler.set_scheduler_manage(scheduler_manage)

    model_name = args.model_name
    init_nodes_num = args.init_nodes_num
    if model_name is not None and init_nodes_num is not None:
        scheduler_manage.run(model_name, init_nodes_num)

    port = args.port

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", loop="uvloop")
