"""
P2P server for Parallax.

This module contains the P2P server for Parallax.

It is used to handle the communication between the peers, and communicate with the executor by zmq.

"""

import dataclasses
import enum
import json
import logging
import threading
import time
from typing import List, Optional

import dijkstar
import httpx
import zmq
from lattica import ConnectionHandler, Lattica, rpc_method, rpc_stream, rpc_stream_iter

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
        http_port: Optional[int] = None,
        notify_url: Optional[str] = None,
    ):
        # Initialize the base class
        super().__init__(lattica)
        self.recv_from_peer_addr = recv_from_peer_addr
        self.send_to_peer_addr = send_to_peer_addr
        self.block_start_index = block_start_index
        self.block_end_index = block_end_index
        self.http_port = http_port
        self.notify_url = notify_url
        self._recv_from_peer = None
        self._recv_from_peer_lock = threading.Lock()

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
            with self._recv_from_peer_lock:
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
            with self._recv_from_peer_lock:
                self.recv_from_peer.send_multipart([b"abort", request.SerializeToString()])
        except Exception as e:
            logger.exception(f"Error in rpc_abort: {e}")
        return forward_pb2.AbortResponse()

    @rpc_stream_iter
    def chat_completion(
        self,
        request,
    ):
        """Handle chat completion request"""
        logger.debug(f"Chat completion request: {request}, type: {type(request)}")
        try:
            with httpx.Client(timeout=10 * 60, proxy=None, trust_env=False) as client:
                if request.get("stream", False):
                    with client.stream(
                        "POST",
                        f"http://localhost:{self.http_port}/v1/chat/completions",
                        json=request,
                    ) as response:
                        for chunk in response.iter_bytes():
                            if chunk:
                                yield chunk
                else:
                    response = client.post(
                        f"http://localhost:{self.http_port}/v1/chat/completions", json=request
                    ).json()
                    yield json.dumps(response).encode()
        except Exception as e:
            logger.exception(f"Error in chat completion: {e}")
            yield b"internal server error"


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
        http_port: Optional[int] = None,
        announce_maddrs: List[str] = [],
        notify_url: str = None,
        model_name: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
    ):
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
        self.http_port = http_port
        self.notify_url = notify_url
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.prefix_id = f"{dht_prefix}_announce"
        self.lattica = None
        self.routing_table = None
        self.routing_table_update_interval = 10
        self.server_info = ServerInfo(state=ServerState.JOINING)
        self.stubs = {}
        self.rtts = {}
        self.rtt_last_update = 0
        self.rtt_update_interval = 60
        self.status = ServerState.JOINING

        self.scheduler_stub = None
        self.scheduler_peer_id = None
        self.routing_table_updater = None
        self.announcer = None
        self.connection_handler = None
        self.stop_event = threading.Event()

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
            self.lattica.with_bootstraps(self.initial_peers).with_mdns(False)

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

    def run(self):
        if self.build_lattica():
            logger.info("Lattica built successfully")
        else:
            logger.error("Failed to build lattica")
            exit(1)

        if self.scheduler_addr is not None:  # central scheduler mode
            try:
                self.scheduler_stub = RPCConnectionHandler(self.lattica, None, None).get_stub(
                    self.scheduler_peer_id
                )
                node_info = self.get_node_info()
                if node_info == {}:
                    logger.error("Failed to get node info, try again after 10 seconds")
                    self.lattica.close()
                    self.lattica = None
                    time.sleep(10)
                    return self.run()
                response = self.scheduler_stub.node_join(node_info)
                response = response.result(timeout=300)
                if response == {}:
                    logger.error("Failed to join scheduler")
                    exit(1)

                logger.info(f"Join scheduler response: {response}")

                self.block_start_index = response.get("start_layer")
                self.block_end_index = response.get("end_layer")
                self.model_name = response.get("model_name")

                # Publish executor metrics to backend on each update
                def _publish_metrics(_snapshot):
                    try:
                        # Fire-and-forget: send metrics update without blocking for response
                        # The announcer thread will periodically check for restart signals
                        # This avoids blocking during inference when network is congested
                        self.scheduler_stub.node_update(self.get_node_info(is_update=True))
                        # Note: We don't wait for response here to avoid timeout during busy inference
                        # Restart signals will be caught by the announcer thread
                    except Exception as e:
                        # Log but don't crash on metrics publish errors
                        logger.debug(f"Metrics publish error (non-critical): {e}")

                set_metrics_publisher(_publish_metrics)

            except Exception as e:
                logger.exception(f"Error in join scheduler: {e}")
                exit(1)
        else:  # no scheduler mode
            self.start_routing_table_updater()  # thread

        self.connection_handler = TransformerConnectionHandler(
            lattica=self.lattica,
            recv_from_peer_addr=self.recv_from_peer_addr,
            send_to_peer_addr=self.send_to_peer_addr,
            block_start_index=self.block_start_index,
            block_end_index=self.block_end_index,
            http_port=self.http_port,
            notify_url=self.notify_url,
        )  # thread

        self.start_node_announcer()  # thread
        self.start_node_sender()  # main loop

    def find_servers(self):
        """Find available servers in the DHT network"""
        # Find all announced blocks
        server_blocks = []
        block_servers = self.lattica.get(self.prefix_id)
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
            while True and not self.stop_event.is_set():
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
            self.routing_table_updater = threading.Thread(target=_updater_thread, daemon=True)
            self.routing_table_updater.start()

    def start_node_sender(self):
        send_to_peer = get_zmq_socket(zmq.Context(2), zmq.PULL, self.send_to_peer_addr, True)

        def group_requests_by_next_peer(requests: List[forward_pb2.Req]):
            grouped_requests = {}
            for req in requests:
                assert len(req.routing_table) > 0, "Request routing table is not set"
                try:
                    self_index = list(req.routing_table).index(self.lattica.peer_id())
                except ValueError as exc:
                    raise RuntimeError("Can not find self in the routing table") from exc

                next_peer_id = req.routing_table[(self_index + 1) % len(req.routing_table)]
                if next_peer_id not in grouped_requests:
                    grouped_requests[next_peer_id] = []
                grouped_requests[next_peer_id].append(req)
            if len(grouped_requests) > 1:
                logger.warning(
                    f"Grouped requests by next peer: {len(grouped_requests)}, {grouped_requests.keys()}"
                )
            return grouped_requests

        while True and not self.stop_event.is_set():
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

                    requests = []
                    for req in forward_request.reqs:
                        # set routing table if not scheduler mode
                        if len(req.routing_table) == 0 and self.scheduler_addr is None:
                            assert (
                                self.block_start_index == 0
                            ), "Request routing table is not set for non-head rank"

                            req.routing_table.extend(self.routing_table)
                            logger.info(
                                f"Set routing table {self.routing_table} for request {req.rid}"
                            )

                        if len(req.routing_table) > 0:
                            requests.append(req)
                        else:
                            logger.error(f"Request {req.rid} has no routing table, drop it")

                    grouped_requests = group_requests_by_next_peer(requests)

                    for next_peer_id, requests in grouped_requests.items():
                        stub = self.get_stub(next_peer_id)
                        start = time.time()
                        logger.info(f"Start forwarding data to {next_peer_id}")
                        new_forward_request = forward_pb2.ForwardRequest()
                        new_forward_request.forward_mode = forward_request.forward_mode
                        new_forward_request.reqs.extend(requests)
                        response = stub.rpc_pp_forward(new_forward_request)
                        response.result()
                        send_notify(
                            self.notify_url,
                            self.block_start_index,
                            self.block_end_index,
                            new_forward_request,
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

                    grouped_requests = {}
                    for req in abort_request.reqs:
                        # set routing table if not scheduler mode
                        if len(req.routing_table) == 0 and self.scheduler_addr is None:
                            assert (
                                self.block_start_index == 0
                            ), "Request routing table is not set for non-head rank"

                            req.routing_table.extend(self.routing_table)
                            logger.info(
                                f"Set routing table {self.routing_table} for request {req.rid}"
                            )

                        if len(req.routing_table) > 0:
                            # broadcast to all other nodes
                            for peer_id in req.routing_table:
                                if peer_id not in grouped_requests:
                                    grouped_requests[peer_id] = []
                                grouped_requests[peer_id].append(req)
                        else:
                            logger.error(f"Abort Request {req.rid} has no routing table, drop it")

                    for peer_id, requests in grouped_requests.items():
                        if peer_id != self.lattica.peer_id():
                            stub = self.get_stub(peer_id)
                            logger.info(
                                f"Send abort request: {[r.rid for r in requests]} to: {peer_id}"
                            )
                            new_abort_request = forward_pb2.AbortRequest()
                            new_abort_request.reqs.extend(requests)
                            stub.rpc_abort(new_abort_request)
                else:
                    logger.error(f"Unknown message type: {message_type}")

            except Exception as e:
                logger.exception(f"Error in handle_request: {e}")
                time.sleep(1)

    def start_node_announcer(self):
        """Start a thread that regularly announces this module's presence on DHT"""

        def _announcer_thread():
            try:
                logger.debug("Node announcer thread started")
                while not self.stop_event.is_set():
                    # Announce the range ID
                    try:
                        if self.scheduler_peer_id is not None:
                            logger.debug("Calling node_update from announcer thread")
                            node_info = self.get_node_info(is_update=True)
                            logger.debug(f"Got node_info: {node_info}")
                            response_future = self.scheduler_stub.node_update(node_info)
                            logger.debug(f"Announcer received future: {response_future}")
                            # Wait for async response with longer timeout
                            # This is the only place we check for restart signals, so it's critical
                            response = response_future.result(timeout=30) if hasattr(response_future, 'result') else response_future
                            logger.debug(f"Announcer resolved response: {response}")
                            # Check if rebalance restart is needed
                            if response and isinstance(response, dict):
                                new_start = response.get("start_layer")
                                new_end = response.get("end_layer")
                                needs_restart = response.get("needs_restart", False)
                                rebalance_reason = response.get("rebalance_reason", "")
                                
                                logger.debug(
                                    f"Announcer check: needs_restart={needs_restart}, "
                                    f"new_layers=[{new_start}, {new_end}), "
                                    f"current_layers=[{self.block_start_index}, {self.block_end_index})"
                                )
                                
                                # If explicit restart signal or layer allocation changed
                                if needs_restart or (new_start is not None and new_end is not None and 
                                                     (new_start != self.block_start_index or new_end != self.block_end_index)):
                                    logger.critical(
                                        f"🔄 RESTART SIGNAL: {rebalance_reason or 'Layer allocation changed'}\n"
                                        f"   Current: ({self.block_start_index}-{self.block_end_index})\n"
                                        f"   New: ({new_start}-{new_end})\n"
                                        f"   Node will exit to restart..."
                                    )
                                    self.shutdown(restart_for_rebalance=True)
                                    return
                        else:
                            self.lattica.store(
                                key=self.prefix_id,
                                subkey=self.lattica.peer_id(),
                                value={
                                    "block_start_index": self.block_start_index,
                                    "block_end_index": self.block_end_index,
                                },
                                expiration_time=time.time() + 60,  # Valid for 60 seconds
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to announce {self.prefix_id}_{self.lattica.peer_id()}: {e}"
                        )

                    time.sleep(10)
            except Exception as e:
                logger.exception(f"Module announcer thread error: {e}")

        # Start announcer thread
        self.announcer = threading.Thread(target=_announcer_thread, daemon=True)
        self.announcer.start()

    def get_node_info(self, is_update: bool = False, is_rebalance_leave: bool = False):
        # update rtt to nodes
        if time.time() - self.rtt_last_update > self.rtt_update_interval:
            self.rtts = {}
            all_peers = []
            for _ in range(1 if is_update else 10):
                all_peers = self.lattica.get_all_peers()
                if len(all_peers) > 0 and self.scheduler_peer_id in all_peers:
                    break
                logger.warning(
                    "No peers found or scheduler peer id not found, waiting for 1 second."
                )
                time.sleep(1)

            if len(all_peers) == 0 or self.scheduler_peer_id not in all_peers:
                logger.warning(
                    "No peers found or scheduler peer id not found, return empty node info."
                )
                return {}

            for peer_id in all_peers:
                rtt = None
                for _ in range(1 if is_update else 30):
                    try:
                        rtt = self.lattica.get_peer_rtt(peer_id) * 1000
                    except Exception as e:
                        logger.warning(f"Failed to get rtt to {peer_id}: {e}")
                    if rtt is not None:
                        break
                    logger.warning(f"Failed to get rtt to {peer_id}, waiting for 1 second.")
                    time.sleep(1)

                self.rtts[peer_id] = rtt if rtt is not None else 100
            self.rtt_last_update = time.time()

        info = {
            "node_id": self.lattica.peer_id(),
            "hardware": detect_node_hardware(self.lattica.peer_id()),
            "kv_cache_ratio": 0.25,
            "param_hosting_ratio": 0.06,
            "max_concurrent_requests": self.max_batch_size,
            "max_sequence_length": (
                1024 if self.max_sequence_length is None else self.max_sequence_length
            ),
            "rtt_to_nodes": self.rtts,
            "status": self.status.value,
            "is_active": self.status == ServerState.READY,
        }

        if is_update:
            metrics = get_metrics()
            info["current_requests"] = metrics.get("current_requests", 0)
            if metrics.get("layer_latency_ms") is not None:
                info["layer_latency_ms"] = metrics.get("layer_latency_ms")
            info["start_layer"] = self.block_start_index
            info["end_layer"] = self.block_end_index
        
        # Mark if this is a rebalance-triggered leave (to avoid cascading rebalances)
        if is_rebalance_leave:
            info["is_rebalance_leave"] = True

        return info

    def needs_rebalance_restart(self):
        """Check if node needs to restart for rebalancing."""
        return getattr(self, '_restart_for_rebalance', False)
    
    def set_executor(self, executor):
        """Set executor reference for shutdown coordination."""
        self._executor = executor
    
    def shutdown(self, restart_for_rebalance=False):
        import threading
        
        current_thread_name = threading.current_thread().name
        logger.debug(f"shutdown() called by thread={current_thread_name}, restart_for_rebalance={restart_for_rebalance}")
        
        # Prevent duplicate shutdown calls
        if hasattr(self, '_shutdown_called') and self._shutdown_called:
            logger.debug(f"Shutdown already called, skipping duplicate call from thread={current_thread_name}")
            return
        self._shutdown_called = True
        
        # Store restart flag for later checking by main thread
        if restart_for_rebalance:
            self._restart_for_rebalance = True
            logger.info(f"🔄 Rebalance restart requested (thread={current_thread_name}). Stopping executor...")
            # Stop executor to interrupt main loop
            if hasattr(self, '_executor') and self._executor is not None:
                try:
                    self._executor.shutdown()
                    logger.info("Executor shutdown initiated")
                except Exception as e:
                    logger.warning(f"Error shutting down executor: {e}")

        # Send node_leave RPC BEFORE setting offline status and closing connection
        # This ensures the RPC completes successfully
        if self.scheduler_addr is not None:
            logger.info(f"Leave scheduler: {self.lattica.peer_id()}, is_rebalance_leave={restart_for_rebalance}")
            try:
                # Mark if this leave is due to rebalance restart (to avoid cascading rebalances)
                node_info = self.get_node_info(is_update=True, is_rebalance_leave=restart_for_rebalance)
                
                # If get_node_info returns empty (peers disconnected), create minimal info
                if not node_info or not node_info.get('node_id'):
                    logger.warning("get_node_info returned empty, creating minimal leave info")
                    node_info = {
                        "node_id": self.lattica.peer_id(),
                        "hardware": {
                            "node_id": self.lattica.peer_id(),
                            "tflops_fp16": 7.1,
                            "gpu_name": "Unknown",
                            "memory_gb": 16.0,
                            "memory_bandwidth_gbps": 100.0
                        },
                        "kv_cache_ratio": 0.25,
                        "param_hosting_ratio": 0.06,
                        "max_concurrent_requests": self.max_batch_size,
                        "max_sequence_length": 1024,
                        "rtt_to_nodes": {},
                        "status": self.status.value,
                        "is_active": False,
                        "start_layer": self.block_start_index,
                        "end_layer": self.block_end_index,
                    }
                    # Critical: ensure is_rebalance_leave is set
                    if restart_for_rebalance:
                        node_info["is_rebalance_leave"] = True
                
                logger.debug(f"Sending node_leave with is_rebalance_leave={node_info.get('is_rebalance_leave', False)}")
                
                # CRITICAL: Wait for RPC to complete before closing connection
                # Otherwise the RPC call will fail with "channel closed" error
                future = self.scheduler_stub.node_leave(node_info)
                try:
                    # Wait up to 3 seconds for the RPC to complete (timeout must be int)
                    result = future.result(timeout=3)
                    logger.info(f"node_leave RPC completed successfully: {result}")
                except Exception as rpc_error:
                    logger.error(f"node_leave RPC failed: {rpc_error}")
            except Exception as e:
                logger.warning(f"Failed to notify scheduler of node leave: {e}")
        
        # NOW set stop_event and offline status after RPC is sent
        self.stop_event.set()
        self.status = ServerState.OFFLINE

        # Avoid joining the current thread (deadlock)
        current_thread = threading.current_thread()
        if self.announcer is not None and self.announcer != current_thread:
            self.announcer.join(timeout=2)
        if self.routing_table_updater is not None and self.routing_table_updater != current_thread:
            self.routing_table_updater.join(timeout=2)
        
        # Close connection AFTER waiting for RPC to complete
        if self.lattica is not None:
            logger.debug("Closing lattica connection...")
            self.lattica.close()
            logger.debug("Lattica connection closed")


def launch_p2p_server(
    initial_peers: List[str],
    scheduler_addr: Optional[str],
    relay_servers: List[str],
    pp_start_layer: int,
    pp_end_layer: int,
    hidden_layers: int,
    tcp_port: int,
    udp_port: int,
    dht_prefix: str,
    announce_maddrs: List[str],
    http_port: Optional[int],
    notify_url: str,
    recv_from_peer_addr: str,
    send_to_peer_addr: str,
    model_name: Optional[str],
    max_batch_size: Optional[int] = None,
    max_sequence_length: Optional[int] = None,
):
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
        host_maddrs=[f"/ip4/0.0.0.0/tcp/{tcp_port}", f"/ip4/0.0.0.0/udp/{udp_port}/quic-v1"],
        announce_maddrs=announce_maddrs,
        http_port=http_port,
        notify_url=notify_url,
        model_name=model_name,
        max_batch_size=max_batch_size,
        max_sequence_length=max_sequence_length,
    )
    # Start the server
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    while server.block_start_index is None:
        time.sleep(1)

    return server
