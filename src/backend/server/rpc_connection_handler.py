import time

from lattica import ConnectionHandler, Lattica, rpc_method, rpc_stream

from backend.server.static_config import get_model_info
from parallax_utils.logging_config import get_logger
from scheduling.node import Node, NodeHardwareInfo
from scheduling.scheduler import Scheduler

logger = get_logger(__name__)


class RPCConnectionHandler(ConnectionHandler):
    """
    Handles RPC requests from clients, forwarding them to the appropriate TransformerBackend.
    Inherits from hivemind's ConnectionHandler.
    """

    def __init__(
        self,
        lattica: Lattica,
        scheduler: Scheduler,
    ):
        # Initialize the base class
        super().__init__(lattica)
        self.scheduler = scheduler
        self.call_url_map = {}

    @rpc_stream
    def node_join(self, message):
        # node = {
        #     "call_url": "http://127.0.0.1:8000",
        #     "node_id": "lattica peer id",
        #     "hardware": {
        #         "node_id": "lattica peer id",
        #         "tflops_fp16": 100,
        #         "memory_gb": 100,
        #         "memory_bandwidth_gbps": 100,
        #     },
        #     "model_name": "",
        #     "kv_cache_ratio": 0.3,
        #     "param_hosting_ratio": 0.5,
        #     "max_concurrent_requests": 16,
        #     "max_sequence_length": 1024,
        # }
        logger.info(f"receive node_join request: {message}")
        try:
            node = self.build_node(message)
            self.call_url_map[node.node_id] = message.get("call_url")
            self.scheduler.enqueue_join(node)

            response = self.wait_layer_allocation(node.node_id, wait_seconds=300)
            logger.info(f"node_join response: {response}")
            return response
        except Exception as e:
            logger.exception(f"node_join error: {e}")
            return {}

    @rpc_method
    def node_leave(self, message):
        logger.info(f"receive node_leave request: {message}")
        try:
            node = self.build_node(message)
            self.scheduler.enqueue_leave(node.node_id)
            self.call_url_map.pop(node.node_id)
            return {}
        except Exception as e:
            logger.exception(f"node_leave error: {e}")
            return {}

    @rpc_method
    def node_update(self, message):
        logger.debug(f"receive node_update request: {message}")
        try:
            node = self.build_node(message)
            self.scheduler.enqueue_node_update(
                node.node_id,
                current_requests=node.current_requests,
                layer_latency_ms=node.layer_latency_ms,
                new_rtt_to_nodes=node.rtt_to_nodes,
            )
            return {}
        except Exception as e:
            logger.exception(f"node_update error: {e}")
            return {}

    def wait_layer_allocation(self, current_node_id, wait_seconds):
        start_time = time.time()
        while True:
            layer_allocation = self.get_layer_allocation(current_node_id)
            if layer_allocation:
                return layer_allocation
            if time.time() - start_time > wait_seconds:
                return {}
            time.sleep(0.5)

    def get_layer_allocation(self, current_node_id):
        list_node_allocations = self.scheduler.list_node_allocations()
        for node_id, start_layer, end_layer in list_node_allocations:
            if current_node_id == node_id:
                return {"node_id": node_id, "start_layer": start_layer, "end_layer": end_layer}
        return {}

    def build_node(self, node_json: dict):
        node = Node(
            node_id=node_json.get("node_id"),
            hardware=self.build_hardware(node_json.get("hardware")),
            model_info=get_model_info(node_json.get("model_name")),
            kv_cache_ratio=node_json.get("kv_cache_ratio"),
            param_hosting_ratio=node_json.get("param_hosting_ratio"),
            max_concurrent_requests=node_json.get("max_concurrent_requests"),
            max_sequence_length=node_json.get("max_sequence_length"),
        )
        if node_json.get("start_layer", None) is not None:
            node.start_layer = node_json.get("start_layer")
        if node_json.get("end_layer", None) is not None:
            node.end_layer = node_json.get("end_layer")
        if node_json.get("current_requests", None) is not None:
            node.current_requests = node_json.get("current_requests")
        if node_json.get("layer_latency_ms", None) is not None:
            node.avg_layer_latency_ms = node_json.get("layer_latency_ms")
        if node_json.get("rtt_to_nodes", None) is not None:
            node.rtt_to_nodes = node_json.get("rtt_to_nodes")
        return node

    def build_hardware(self, hardware_json):
        node_id = hardware_json.get("node_id")
        tflops_fp16 = hardware_json.get("tflops_fp16")
        gpu_name = hardware_json.get("gpu_name")
        memory_gb = hardware_json.get("memory_gb")
        memory_bandwidth_gbps = hardware_json.get("memory_bandwidth_gbps")
        return NodeHardwareInfo(
            node_id=node_id,
            tflops_fp16=tflops_fp16,
            gpu_name=gpu_name,
            memory_gb=memory_gb,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
        )

    def get_call_url_by_node_id(self, node_id):
        return self.call_url_map.get(node_id, None)
