"""
P2P server for Parallax.

This module contains the P2P server for Parallax.

It is used to handle the communication between the peers, and communicate with the executor by zmq.

"""

import dataclasses
import enum
import logging
import threading
import time
from typing import List, Optional

import dijkstar
import httpx
import zmq
from lattica import ConnectionHandler, Lattica, rpc_method, rpc_stream

from backend.server.rpc_connection_handler import RPCConnectionHandler
from parallax.p2p.proto import forward_pb2
from parallax.p2p.utils import AsyncWorker
from parallax.server.metrics import get_metrics, set_metrics_publisher
from parallax.server.server_info import detect_node_hardware
from parallax.utils.utils import get_zmq_socket

logger = logging.getLogger(__name__)

# Global HTTP client for reuse
_http_client = None


async def get_http_client():
    """Get or create a shared HTTP client"""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(3),  # 3 second timeout
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=100),
        )
    return _http_client


class ServerState(enum.Enum):
    """Server state enum."""

    JOINING = "joining"
    INITIALIZING = "initializing"
    READY = "ready"
    OFFLINE = "offline"
    ERROR = "error"


@dataclasses.dataclass
class ServerInfo:
    """Server info data class."""

    state: ServerState
    throughput: Optional[float] = None
    max_batch_size: Optional[int] = None
    max_sequence_len: Optional[int] = None
    error_message: Optional[str] = None


def send_notify(notify_url, block_start_index, block_end_index, request, status):
    payload = [
        {
            "session_id": req.rid,
            "step_id": req.output_length + (block_start_index == 0 and status == "started"),
            "block_idx": block_start_index,
            "total_blocks": block_end_index - block_start_index,
            "status": status,
        }
        for req in request.reqs
    ]

    logger.info(f"Send {status} notification, batch size: {len(payload)}")

    if notify_url is not None:

        async def send_async(notify_url, payload):
            try:
                client = await get_http_client()
                await client.post(notify_url, json=payload)
            except Exception as e:
                logger.exception(f"Error in send_async: {e}")

        if not hasattr(send_notify, "async_worker"):
            send_notify.async_worker = AsyncWorker()
        send_notify.async_worker.run_coroutine(send_async(notify_url, payload), return_future=True)


class TransformerConnectionHandler(ConnectionHandler):
    """
    Handles RPC requests from clients, forwarding them to the appropriate TransformerBackend.
    Inherits from hivemind's ConnectionHandler.
    """

    def __init__(
        self,
        lattica: Lattica,
        recv_from_peer_addr: str,
        send_to_peer_addr: str,
        block_start_index: int,
        block_end_index: int,
        notify_url: Optional[str] = None,
    ):
        # Initialize the base class
        super().__init__(lattica)
        self.recv_from_peer_addr = recv_from_peer_addr
        self.send_to_peer_addr = send_to_peer_addr
        self.block_start_index = block_start_index
        self.block_end_index = block_end_index
        self.notify_url = notify_url
        self._recv_from_peer = None

    @property
    def recv_from_peer(self):
        if self._recv_from_peer is None:
            self._recv_from_peer = get_zmq_socket(
                zmq.Context(2), zmq.PUSH, self.recv_from_peer_addr, True
            )
        return self._recv_from_peer

    @rpc_stream
    def rpc_pp_forward(
        self,
        request: forward_pb2.ForwardRequest,
    ) -> forward_pb2.ForwardResponse:
        """Handle forward pass request with explicit proxy tensors support"""
        try:
            send_notify(
                self.notify_url, self.block_start_index, self.block_end_index, request, "started"
            )
            self.recv_from_peer.send_multipart([b"forward", request.SerializeToString()])
        except Exception as e:
            logger.exception(f"Error in rpc_pp_forward: {e}")
        return forward_pb2.ForwardResponse()

    @rpc_method
    def rpc_abort(
        self,
        request: forward_pb2.AbortRequest,
    ) -> forward_pb2.AbortResponse:
        try:
            self.recv_from_peer.send_multipart([b"abort", request.SerializeToString()])
        except Exception as e:
            logger.exception(f"Error in rpc_abort: {e}")
        return forward_pb2.AbortResponse()


class GradientServer:
    """
    Main server class for Parallax.

    This class handles communication between peers and communicates with the executor by zmq.
    """

    def __init__(
        self,
        recv_from_peer_addr: str,
        send_to_peer_addr: str,
        initial_peers: List[str] = [],
        scheduler_addr: Optional[str] = None,
        relay_servers: List[str] = [],
        block_start_index: int = 0,
        block_end_index: int = 1,
        hidden_layers: int = 128,
        dht_prefix: str = "gradient",
        host_maddrs: List[str] = [],
        public_ip: Optional[str] = None,
        http_port: int = 3000,
        announce_maddrs: List[str] = [],
        notify_url: str = None,
        model_name: Optional[str] = None,
    ):
        assert not (
            scheduler_addr is not None and len(initial_peers) > 0
        ), "scheduler_addr and initial_peers are not allowed at the same time"
        self.recv_from_peer_addr = recv_from_peer_addr
        self.send_to_peer_addr = send_to_peer_addr
        self.initial_peers = initial_peers
        self.scheduler_addr = scheduler_addr
        self.relay_servers = relay_servers
        self.block_start_index = block_start_index
        self.block_end_index = block_end_index
        self.hidden_layers = hidden_layers
        self.dht_prefix = dht_prefix
        self.host_maddrs = host_maddrs
        self.announce_maddrs = announce_maddrs
        self.public_ip = public_ip
        self.http_port = http_port
        self.notify_url = notify_url
        self.model_name = model_name

        self.node_id = f"{dht_prefix}_announce"
        self.lattica = None
        self.routing_table = None
        self.routing_table_update_interval = 10
        self.server_info = ServerInfo(state=ServerState.JOINING)
        self.stubs = {}

    def run(self):
        self.lattica = Lattica.builder().with_listen_addrs(self.host_maddrs)

        if len(self.relay_servers) > 0:
            logger.info(f"Using relay servers: {self.relay_servers}")
            self.lattica.with_relay_servers(self.relay_servers).with_dcutr(True)

        if len(self.announce_maddrs) > 0:
            logger.info(f"Using announce maddrs: {self.announce_maddrs}")
            self.lattica.with_external_addrs(self.announce_maddrs)

        if len(self.initial_peers) > 0:
            logger.info(f"Using initial peers: {self.initial_peers}")
            self.lattica.with_bootstraps(self.initial_peers)

        if self.scheduler_addr is not None:
            logger.info(f"Using scheduler addr: {self.scheduler_addr}")
            self.lattica.with_bootstraps([self.scheduler_addr])
            self.scheduler_peer_id = self.scheduler_addr.split("/")[-1]
        else:
            self.scheduler_peer_id = None

        self.lattica.build()

        if self.scheduler_addr is not None:  # central scheduler mode
            try:
                self.scheduler_stub = RPCConnectionHandler(self.lattica, None).get_stub(
                    self.scheduler_peer_id
                )
                logger.info(f"Send node_join request to scheduler")
                response = self.scheduler_stub.node_join(self.get_node_info())
                response = response.result()
                if response == {}:
                    logger.error("Failed to join scheduler")
                    exit(1)

                logger.info(f"Join scheduler response: {response}")

                self.block_start_index = response.get("start_layer")
                self.block_end_index = response.get("end_layer")

                # Publish executor metrics to backend on each update
                def _publish_metrics(_snapshot):
                    try:
                        self.scheduler_stub.node_update(self.get_node_info(is_update=True))
                    except Exception:
                        pass

                set_metrics_publisher(_publish_metrics)

            except Exception as e:
                logger.exception(f"Error in join scheduler: {e}")
                exit(1)
        else:  # decentralized mode
            self.start_routing_table_updater()  # thread

        self.connection_handler = TransformerConnectionHandler(
            lattica=self.lattica,
            recv_from_peer_addr=self.recv_from_peer_addr,
            send_to_peer_addr=self.send_to_peer_addr,
            block_start_index=self.block_start_index,
            block_end_index=self.block_end_index,
            notify_url=self.notify_url,
        )  # thread

        self.start_node_announcer()  # thread
        self.start_node_sender()  # main loop

    def find_servers(self):
        """Find available servers in the DHT network"""
        # Find all announced blocks
        server_blocks = []
        block_announced_key = f"{self.dht_prefix}_announce"
        block_servers = self.lattica.get(block_announced_key)
        if block_servers is None:
            return []
        for peer_id, value in block_servers.value.items():
            server_blocks.append(
                {
                    "peer_id": peer_id,
                    "block_start_index": value.value["block_start_index"],
                    "block_end_index": value.value["block_end_index"],
                }
            )

        return server_blocks

    def get_stub(self, peer_id):
        if peer_id not in self.stubs:
            self.stubs[peer_id] = self.connection_handler.get_stub(peer_id)
        return self.stubs[peer_id]

    def start_routing_table_updater(self):
        def _updater_thread():
            while True:
                try:
                    graph = dijkstar.Graph()
                    servers = self.find_servers()
                    for server in servers:
                        start_index = server["block_start_index"]
                        end_index = server["block_end_index"]
                        peer_id = server["peer_id"]
                        graph.add_edge(start_index, end_index, (1, peer_id))
                    try:
                        path = dijkstar.find_path(
                            graph,
                            self.block_end_index,
                            self.hidden_layers,
                            cost_func=lambda u, v, e, prev_path: e[0],
                        )
                        routing_table = [self.lattica.peer_id()] + [edge[1] for edge in path.edges]
                        if self.routing_table != routing_table:
                            self.routing_table = routing_table
                            logger.info(f"Set routing table: {routing_table}")
                    except dijkstar.NoPathError:
                        self.routing_table = None
                        logger.warning(
                            f"No path found from 0 to {self.hidden_layers}, find servers {servers}"
                        )
                except Exception as e:
                    logger.exception(f"Error in routing table updater: {e}")

                time.sleep(self.routing_table_update_interval)

        if self.block_start_index == 0:
            self.updater = threading.Thread(target=_updater_thread, daemon=True)
            self.updater.start()

    def start_node_sender(self):
        send_to_peer = get_zmq_socket(zmq.Context(2), zmq.PULL, self.send_to_peer_addr, True)

        while True:
            try:
                if (
                    self.scheduler_addr is None
                    and self.block_start_index == 0
                    and self.routing_table is None
                ):
                    logger.info("Routing table is not ready in head rank, waiting for it to be set")
                    time.sleep(self.routing_table_update_interval)
                    continue

                message_type, message_body = send_to_peer.recv_multipart()[:2]

                if message_type == b"forward":
                    forward_request = forward_pb2.ForwardRequest()
                    forward_request.ParseFromString(message_body)
                    if len(forward_request.reqs) == 0:
                        raise RuntimeError("No requests in the forward request")

                    # suppose all requests have same routing table
                    routing_table = None
                    for req in forward_request.reqs:
                        if len(req.routing_table) == 0:
                            if self.scheduler_addr is not None:
                                logger.error("Routing table is not set for scheduler mode")
                            assert (
                                self.block_start_index == 0
                            ), "Request routing table is not set for non-head rank"

                            req.routing_table.extend(self.routing_table)
                            logger.info(
                                f"Set routing table {self.routing_table} for request {req.rid}"
                            )
                        # Assume all requests have same routing table
                        routing_table = list(req.routing_table)

                    try:
                        self_index = routing_table.index(self.lattica.peer_id())
                    except ValueError as exc:
                        raise RuntimeError("Can not find self in the routing table") from exc

                    next_peer_id = routing_table[(self_index + 1) % len(routing_table)]
                    stub = self.get_stub(next_peer_id)
                    start = time.time()
                    logger.info(f"Start forwarding data to {next_peer_id}")
                    response = stub.rpc_pp_forward(forward_request)
                    response.result()
                    send_notify(
                        self.notify_url,
                        self.block_start_index,
                        self.block_end_index,
                        forward_request,
                        "completed",
                    )

                    logger.info(
                        f"Forwarding data to {next_peer_id}, "
                        f"total size: {len(message_body) / (1024 * 1024):.3f} MB, "
                        f"cost time: {(time.time() - start) * 1000:.3f} ms, "
                        f"speed: {len(message_body) / (time.time() - start) / (1024 * 1024):.3f} MB/s"
                    )

                elif message_type == b"abort":
                    abort_request = forward_pb2.AbortRequest()
                    abort_request.ParseFromString(message_body)
                    if len(abort_request.reqs) == 0:
                        raise RuntimeError("No requests in the abort request")

                    # suppose all requests have same routing table
                    routing_table = None
                    for req in forward_request.reqs:
                        if len(req.routing_table) == 0:
                            assert (
                                self.block_start_index == 0
                            ), "Request routing table is not set for non-head rank"
                            req.routing_table.extend(self.routing_table)
                        # Assume all requests have same routing table
                        routing_table = list(req.routing_table)

                    try:
                        self_index = routing_table.index(self.lattica.peer_id())
                    except ValueError as exc:
                        raise RuntimeError("Can not find self in the routing table") from exc

                    # broadcast to all other nodes
                    for index, peer_id in enumerate(routing_table):
                        if index != self_index:
                            stub = self.get_stub(peer_id)
                            logger.info(
                                f"Send abort request: {[r.rid for r in abort_request.reqs]} to: {peer_id}"
                            )
                            stub.rpc_abort(abort_request)
                else:
                    logger.error(f"Unknown message type: {message_type}")

            except Exception as e:
                logger.exception(f"Error in handle_request: {e}")
                time.sleep(1)

    def start_node_announcer(self):
        """Start a thread that regularly announces this module's presence on DHT"""

        def _announcer_thread():
            try:
                while not self.stop_event.is_set():
                    # Announce the range ID
                    try:
                        if self.scheduler_addr is not None:
                            self.scheduler_stub.node_update(self.get_node_info(is_update=True))
                        else:
                            self.lattica.store(
                                key=self.node_id,
                                subkey=self.lattica.peer_id(),
                                value={
                                    "block_start_index": self.block_start_index,
                                    "block_end_index": self.block_end_index,
                                },
                                expiration_time=time.time() + 60,  # Valid for 60 seconds
                            )
                    except Exception as e:
                        logger.warning(f"Failed to announce {self.node_id}: {e}")

                    time.sleep(10)
            except Exception as e:
                logger.exception(f"Module announcer thread error: {e}")

        # Start announcer thread
        self.stop_event = threading.Event()
        self.announcer = threading.Thread(target=_announcer_thread, daemon=True)
        self.announcer.start()

    def get_node_info(self, is_update: bool = False):
        peer_id = self.lattica.peer_id() if self.lattica is not None else None
        info = {
            "call_url": f"http://{self.public_ip}:{self.http_port}",
            "node_id": peer_id,
            "hardware": detect_node_hardware(peer_id),
            "model_name": self.model_name,
            "kv_cache_ratio": 0.35,
            "param_hosting_ratio": 0.5,
            "max_concurrent_requests": 16,
            "max_sequence_length": 1024,
        }
        if is_update:
            metrics = get_metrics()
            info["current_requests"] = metrics.get("current_requests", 0)
            if metrics.get("layer_latency_ms") is not None:
                info["layer_latency_ms"] = metrics.get("layer_latency_ms")
            info["rtt_to_nodes"] = {}
            info["start_layer"] = self.block_start_index
            info["end_layer"] = self.block_end_index
        return info


def launch_p2p_server(
    initial_peers: List[str],
    scheduler_addr: Optional[str],
    relay_servers: List[str],
    pp_start_layer: int,
    pp_end_layer: int,
    hidden_layers: int,
    dht_port: Optional[int],
    dht_prefix: str,
    host_maddrs: Optional[List[str]],
    announce_maddrs: List[str],
    public_ip: Optional[str],
    http_port: int,
    notify_url: str,
    recv_from_peer_addr: str,
    send_to_peer_addr: str,
    model_name: Optional[str],
):
    if dht_port is not None:
        assert host_maddrs is None, "You can't use --dht-port and --host-maddrs at the same time"
    else:
        dht_port = 0
    if host_maddrs is None:
        host_maddrs = [f"/ip4/0.0.0.0/tcp/{dht_port}", f"/ip6/::/tcp/{dht_port}"]

    # Run the server in a separate thread to keep the main thread free for event loop
    server = GradientServer(
        recv_from_peer_addr=recv_from_peer_addr,
        send_to_peer_addr=send_to_peer_addr,
        initial_peers=initial_peers,
        scheduler_addr=scheduler_addr,
        relay_servers=relay_servers,
        block_start_index=pp_start_layer,
        block_end_index=pp_end_layer,
        hidden_layers=hidden_layers,
        dht_prefix=dht_prefix,
        host_maddrs=host_maddrs,
        announce_maddrs=announce_maddrs,
        public_ip=public_ip if public_ip is not None else "127.0.0.1",
        http_port=http_port,
        notify_url=notify_url,
        model_name=model_name,
    )
    # Start the server
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    while server.block_start_index is None:
        time.sleep(1)

    return server.block_start_index, server.block_end_index
