import threading
import time
from typing import List

from lattica import Lattica

from backend.server.constants import NODE_STATUS_AVAILABLE, NODE_STATUS_WAITING
from backend.server.rpc_connection_handler import RPCConnectionHandler
from backend.server.static_config import get_model_info, get_node_join_command
from parallax_utils.logging_config import get_logger
from scheduling.node import RequestSignal
from scheduling.scheduler import Scheduler

logger = get_logger(__name__)


class SchedulerManage:
    """
    Coordinates the in-process scheduler and the P2P RPC layer.

    This manager owns the `Scheduler` instance and the Lattica P2P node,
    wiring RPC calls from workers to scheduler events.
    """

    def __init__(
        self,
        initial_peers: List[str] = [],
        relay_servers: List[str] = [],
        dht_prefix: str = "gradient",
        host_maddrs: List[str] = [],
        announce_maddrs: List[str] = [],
    ):
        """Initialize the manager with networking bootstrap parameters."""
        self.initial_peers = initial_peers
        self.relay_servers = relay_servers
        self.dht_prefix = dht_prefix
        self.host_maddrs = host_maddrs
        self.announce_maddrs = announce_maddrs

        self.model_name = None
        self.init_nodes_num = None
        self.scheduler = None
        self.node_id = f"{dht_prefix}_announce"
        self.lattica = None
        self.stubs = {}
        self.is_local_network = False

    def run(self, model_name, init_nodes_num, is_local_network=False):
        """
        Start the scheduler and the P2P service for RPC handling.
        """
        logger.info(
            f"SchedulerManage starting: model_name={model_name}, init_nodes_num={init_nodes_num}"
        )
        self.is_local_network = is_local_network
        self._start_scheduler(model_name, init_nodes_num)
        self._start_lattica()

    def is_running(self):
        """
        Returns True if the scheduler is running, False otherwise.
        """
        return self.scheduler is not None

    def get_model_name(self):
        return self.model_name

    def get_init_nodes_num(self):
        return self.init_nodes_num

    def get_is_local_network(self):
        return self.is_local_network

    def get_cluster_status(self):
        return {
            "type": "cluster_status",
            "data": {
                "status": self.get_schedule_status(),
                "model_name": self.model_name,
                "init_nodes_num": self.init_nodes_num,
                "node_join_command": get_node_join_command(
                    self.model_name, "${scheduler-addr}", self.is_local_network
                ),
                "node_list": self.get_node_list(),
            },
        }

    def get_node_list(self):
        if self.scheduler is None:
            return []

        return [self.build_node_info(node) for node in self.scheduler.nodes]

    def build_node_info(self, node):
        return {
            "node_id": node.node_id,
            "status": NODE_STATUS_AVAILABLE if node.is_active else NODE_STATUS_WAITING,
            "gpu_name": node.hardware.gpu_name,
            "gpu_memory": node.hardware.memory_gb,
        }

    def _start_scheduler(self, model_name, init_nodes_num):
        """
        Create the scheduler and start its background run loop if needed.
        """
        if self.scheduler is not None:
            logger.info("Scheduler already started; skipping re-initialization")
            return

        self.model_name = model_name
        self.init_nodes_num = init_nodes_num

        model_info = get_model_info(model_name)
        self.scheduler = Scheduler(model_info, [], min_nodes_bootstrapping=init_nodes_num)

        # Run the scheduler's event/dispatch loops in background so the process
        # can continue to serve RPCs and HTTP traffic.
        threading.Thread(
            target=self.scheduler.run,
            kwargs={"poll_interval": 0.05},
            name="SchedulerMain",
            daemon=True,
        ).start()
        logger.info("Scheduler background thread started (poll_interval=0.05)")

    def _start_lattica(self):
        """
        Initialize and start the Lattica P2P node used for RPCs.
        """
        logger.info(
            f"Starting Lattica with host_maddrs={self.host_maddrs}, mdns=False, dht_prefix={self.dht_prefix}"
        )
        self.lattica = (
            Lattica.builder()
            .with_listen_addrs(self.host_maddrs)
            .with_mdns(False)
            .with_key_path(".")
        )

        if len(self.relay_servers) > 0:
            print(f"Using relay servers: {self.relay_servers}")
            self.lattica.with_relay_servers(self.relay_servers).with_dcutr(True)

        if len(self.announce_maddrs) > 0:
            print(f"Using announce maddrs: {self.announce_maddrs}")
            self.lattica.with_external_addrs(self.announce_maddrs)

        if len(self.initial_peers) > 0:
            print(f"Using initial peers: {self.initial_peers}")
            self.lattica.with_bootstraps(self.initial_peers)

        self.lattica.build()
        logger.info("Lattica node built")

        self.connection_handler = RPCConnectionHandler(
            lattica=self.lattica,
            scheduler=self.scheduler,
        )
        logger.info("RPCConnectionHandler initialized")

    def get_routing_table(self, request_id, received_ts):
        """Block briefly until the scheduler assigns a routing path for the request.

        Distinguish three states via `RequestSignal.routing_table`:
        - None: not yet decided, keep waiting up to timeout
        - []: decided but no capacity (pipelines full), return immediately
        - [..]: valid routing path, return immediately
        """
        logger.info(f"Routing table requested for request_id={request_id}")
        request = RequestSignal(request_id, received_ts)
        self.scheduler.receive_request(request)

        # Wait up to 5 seconds, but return immediately if the routing table is set (including an empty list)
        start_time = time.time()
        while request.routing_table is None and (time.time() - start_time) < 5.0:
            time.sleep(0.05)

        # Return the routing_table
        if request.routing_table is None:
            logger.info(
                f"Routing table not ready after {(time.time() - start_time):.2f}s for request_id={request_id}"
            )
        else:
            logger.info(
                f"Routing table resolved for request_id={request_id}: {request.routing_table}"
            )
        return request.routing_table

    def get_schedule_status(self):
        """
        Return whether a full pipeline has been allocated across joined nodes.
        """
        if self.scheduler is None:
            logger.info("SchedulerManage status queried: waiting (scheduler not initialized)")
            return NODE_STATUS_WAITING

        # todo rebalance status
        status = (
            NODE_STATUS_AVAILABLE
            if self.scheduler.layer_allocator.has_full_active_pipeline()
            else NODE_STATUS_WAITING
        )
        logger.info(f"SchedulerManage status queried: {status}")
        return status

    def get_call_url_by_node_id(self, node_id):
        """
        Lookup the HTTP endpoint for a given node id managed by the RPC layer.
        """
        url = self.connection_handler.get_call_url_by_node_id(node_id)
        logger.info(f"Lookup call_url for node_id={node_id} -> {url}")
        return url
