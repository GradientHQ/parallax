import time
import uuid

import uvicorn
from fastapi import FastAPI, Request

from backend.server.request_handler import RequestHandler
from backend.server.scheduler_manage import SchedulerManage
from backend.server.server_args import parse_args
from parallax_utils.logging_config import get_logger

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
    else:
        logger.error("model_name and init_nodes_num are not set")
        exit(1)

    port = args.port

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", loop="uvloop")
