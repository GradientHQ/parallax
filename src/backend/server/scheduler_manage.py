import threading
import time
from typing import List

from lattica import Lattica

from backend.server.rpc_connection_handler import RPCConnectionHandler
from backend.server.static_config import get_model_info
from scheduling.node import RequestSignal
from scheduling.scheduler import Scheduler


class SchedulerManage:
    """Coordinates the in-process scheduler and the P2P RPC layer.

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

        self.scheduler = None
        self.node_id = f"{dht_prefix}_announce"
        self.lattica = None
        self.stubs = {}

    def run(self, model_name, init_nodes_num):
        """Start the scheduler and the P2P service for RPC handling."""
        self._start_scheduler(model_name, init_nodes_num)
        self._start_lattica()

    def _start_scheduler(self, model_name, init_nodes_num):
        """Create the scheduler and start its background run loop if needed."""
        if self.scheduler is not None:
            return

        mode_info = get_model_info(model_name)
        # 初始化 scheduler
        self.scheduler = Scheduler(mode_info, [], min_nodes_bootstrapping=init_nodes_num)

        # Run the scheduler's event/dispatch loops in background so the process
        # can continue to serve RPCs and HTTP traffic.
        threading.Thread(
            target=self.scheduler.run,
            kwargs={"poll_interval": 0.05},
            name="SchedulerMain",
            daemon=True,
        ).start()

    def _start_lattica(self):
        """Initialize and start the Lattica P2P node used for RPCs."""
        self.lattica = Lattica.builder().with_listen_addrs(self.host_maddrs).with_mdns(False)

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

        self.connection_handler = RPCConnectionHandler(
            lattica=self.lattica,
            scheduler=self.scheduler,
        )

    def get_routing_table(self, request_id, received_ts):
        """Block briefly until the scheduler assigns a routing path for the request."""
        request = RequestSignal(request_id, received_ts)
        self.scheduler.receive_request(request)

        # 等待最长 5s
        start_time = time.time()
        while len(request.routing_table) == 0 and (time.time() - start_time) < 5.0:
            time.sleep(0.05)

        # 返回routing_table
        return request.routing_table

    def get_schedule_status(self):
        """Return whether a full pipeline has been allocated across joined nodes."""
        if self.scheduler is None:
            return "waiting"

        if self.scheduler.layer_allocator.has_full_pipeline():
            return "success"
        else:
            return "waiting"

    def get_call_url_by_node_id(self, node_id):
        """Lookup the HTTP endpoint for a given node id managed by the RPC layer."""
        return self.connection_handler.get_call_url_by_node_id(node_id)
