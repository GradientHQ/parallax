"""
Scheduling primitives for distributed LLM inference.

- `NodeHardwareInfo`: static hardware properties
- `RequestSignal`: minimal request envelope (id, received timestamp)
- `RooflinePerformanceModel`: compute/IO roofline estimator with configurable
  sequence/batch shape
- `Node`: worker serving state; manages layer allocation, capacity helpers,
  latency tracking, and RTT cache for network-aware request routing
"""

import time
from dataclasses import dataclass, field
from math import floor
from typing import Callable, Dict, Optional

from scheduling.model_info import ModelInfo


@dataclass
class NodeHardwareInfo:
    """
    Hardware-only description of a node.

    Contains static properties that do not depend on a specific model, and
    optionally cached RTTs to other nodes for network-aware decisions.
    """

    node_id: str
    tflops_fp16: float
    memory_gb: float
    memory_bandwidth_gbps: float


@dataclass
class RequestSignal:
    """
    Minimal request signal container for scheduling.

    - request_id: Unique identifier (hash) for the request
    - received_ts: UNIX timestamp (seconds) when the request was received
    """

    request_id: str
    received_ts: float = field(default_factory=time.time)


class RooflinePerformanceModel:
    """
    Lightweight roofline-based performance estimator.

    Encapsulates compute- and IO-bound latency estimations for a given
    `(hardware, model_info)` pair. Sequence/batch shape can be updated to
    reflect current request context.
    """

    def __init__(
        self,
        hardware: NodeHardwareInfo,
        model_info: ModelInfo,
        quantization_speedup: float = 1.0,
        *,
        batch_size: int = 1,
        target_seq_len: int = 1,
        source_seq_len: int = 256,
    ) -> None:
        self.tflops = hardware.tflops_fp16
        self.io_bandwidth = hardware.memory_bandwidth_gbps
        self.model_info = model_info
        self.quantization_speedup = quantization_speedup
        self.batch_size = batch_size
        self.target_seq_len = target_seq_len
        self.source_seq_len = source_seq_len

    def get_compute_roofline_latency_ms(self, flops: int) -> float:
        """Compute-bound latency in milliseconds for the given floating-point ops."""
        return flops / (self.quantization_speedup * self.tflops * 1e9)

    def get_io_roofline_latency_ms(self, io_bytes: int) -> float:
        """Memory/IO-bound latency in milliseconds for the given data transfer size."""
        return io_bytes / (self.io_bandwidth * 1e6)

    def set_sequence_shape(
        self,
        *,
        batch_size: Optional[int] = None,
        target_seq_len: Optional[int] = None,
        source_seq_len: Optional[int] = None,
    ) -> None:
        """Convenience setter to update any of batch/target/source sequence sizes."""
        if batch_size is not None:
            self.batch_size = batch_size
        if target_seq_len is not None:
            self.target_seq_len = target_seq_len
        if source_seq_len is not None:
            self.source_seq_len = source_seq_len

    def roofline_layer_latency_ms(
        self,
        include_input_embed: bool = False,
        include_lm_head: bool = False,
        num_decoder_layers: int = 1,
    ) -> float:
        """Estimate latency to execute the specified layer set on this node.

        Args:
            include_input_embed: Whether to include input embedding I/O
            include_lm_head: Whether to include LM head compute and I/O
            num_decoder_layers: Number of decoder layers included

        Returns:
            Total latency (ms) combining decoder layers and optional endpoints.
        """
        decoder_layer_compute_latency = self.get_compute_roofline_latency_ms(
            self.model_info.decoder_layer_flops(
                batch_size=self.batch_size,
                target_seq_len=self.target_seq_len,
                source_seq_len=self.source_seq_len,
            )
        )
        decoder_layer_io_latency = self.get_io_roofline_latency_ms(
            self.model_info.decoder_layer_io_bytes(
                roofline=True,
                batch_size=self.batch_size,
                target_seq_len=self.target_seq_len,
                source_seq_len=self.source_seq_len,
            )
        )

        # For first / last layers
        flops, io_bytes = 0, 0
        if include_input_embed:
            # Embedding lookup is I/O-dominant
            io_bytes += self.model_info.embedding_io_bytes

        if include_lm_head:
            flops += self.model_info.lm_head_flops(self.target_seq_len)
            io_bytes += self.model_info.embedding_io_bytes

        compute_time_ms = self.get_compute_roofline_latency_ms(flops)
        io_time_ms = self.get_io_roofline_latency_ms(io_bytes)
        return num_decoder_layers * max(
            decoder_layer_compute_latency, decoder_layer_io_latency
        ) + max(compute_time_ms, io_time_ms)


@dataclass
class Node:
    """
    Dynamic worker node's serving state and network-aware routing hooks.

    - Tracks layer allocation and request load;
    - Capacity helpers for layer allocation;
    - Latency tracking and estimation if not available from node broadcasting;
    - Networking: optional RTT cache and getter for on-demand RTT measurement.

    """

    node_id: str
    hardware: NodeHardwareInfo
    model_info: ModelInfo

    kv_cache_ratio: float = 0.3
    param_hosting_ratio: float = 0.5

    max_concurrent_requests: int = 16
    max_sequence_length: int = 1024

    start_layer: Optional[int] = None  # inclusive
    end_layer: Optional[int] = None  # exclusive
    current_requests: int = 0

    is_active: bool = True
    last_heartbeat: float = 0.0
    # Will be updated by node broadcasting
    # otherwise, use roofline performance model to estimate
    avg_layer_latency_ms: Optional[float] = None

    rtt_to_nodes: Optional[Dict[str, float]] = None
    rtt_getter: Optional[Callable[["Node", "Node"], float]] = None

    def __post_init__(self):
        if self.last_heartbeat == 0.0:
            self.last_heartbeat = time.time()
        if self.rtt_to_nodes is None:
            self.rtt_to_nodes = {}

    @property
    def max_requests(self) -> int:
        """Number of requests given by max num KV tokens, assuming all requests have max sequence length."""
        if self.start_layer is None or self.end_layer is None:
            return self.max_concurrent_requests

        kv_size = self.model_info.per_token_per_layer_kv_size * self.num_decoder_layers
        num_kv_tokens_can_host = floor(
            self.hardware.memory_gb * 1024 * 1024 * 1024 * self.kv_cache_ratio / kv_size
        )
        return min(self.max_requests, floor(num_kv_tokens_can_host / self.max_sequence_length))

    @property
    def num_current_layers(self) -> int:
        """Number of currently allocated layers."""
        if self.start_layer is None or self.end_layer is None:
            return 0
        return self.end_layer - self.start_layer

    @property
    def num_decoder_layers(self) -> int:
        """Number of decoder layers."""
        if self.start_layer is None or self.end_layer is None:
            return 0
        start_layer = self.start_layer if self.has_embedding else self.start_layer + 1
        end_layer = self.end_layer if self.has_lm_head else self.end_layer - 1
        return end_layer - start_layer

    @property
    def has_embedding(self) -> bool:
        """Check if this node hosts the embedding layer (layer 0)."""
        if self.start_layer is None:
            return False
        return self.start_layer == 0

    @property
    def has_lm_head(self) -> bool:
        """Check if this node hosts the LM head layer (last layer)."""
        if self.end_layer is None:
            return False
        return self.end_layer == self.model_info.num_layers

    @property
    def is_overloaded(self) -> bool:
        """Check if node is at capacity for requests."""
        return self.current_requests >= self.max_requests

    def get_decoder_layer_capacity(
        self, include_input_embed: bool = False, include_lm_head: bool = False
    ) -> int:
        """Return how many decoder layers this node can store for parameters.

        Capacity is measured using the parameter memory budget on the device.
        """
        available_memory_bytes = floor(
            self.hardware.memory_gb * 1024 * 1024 * 1024 * self.param_hosting_ratio
        )
        if include_input_embed:
            available_memory_bytes -= self.model_info.embedding_io_bytes
        if include_lm_head:
            if not (include_input_embed and self.model_info.tie_embedding):
                available_memory_bytes -= self.model_info.embedding_io_bytes

        return floor(
            available_memory_bytes / self.model_info.decoder_layer_io_bytes(roofline=False)
        )

    @property
    def per_decoder_layer_kv_cache_memory(self) -> Optional[int]:
        """Return the available memory for kv cache per layer."""
        if self.num_decoder_layers == 0:
            return None
        return floor(
            (self.hardware.memory_gb * 1024 * 1024 * 1024 * self.kv_cache_ratio)
            / self.num_decoder_layers
        )

    @property
    def per_decoder_layer_flops(self) -> Optional[int]:
        """Return the FLOPs per decoder layer for this node."""
        if self.num_decoder_layers == 0:
            return None
        flops = self.hardware.tflops_fp16
        if self.has_lm_head:
            flops -= self.model_info.lm_head_flops()

        return floor(flops / self.num_decoder_layers)

    def set_layer_allocation(self, start_layer: int, end_layer: int) -> None:
        """Set the layer range allocated to this node."""
        self.start_layer = start_layer
        self.end_layer = end_layer

    def clear_layer_allocation(self) -> None:
        """Clear the layer allocation for this node."""
        self.start_layer = None
        self.end_layer = None

    def set_layer_latency_ms(self, latency_ms: float) -> None:
        """Update the layer latency for this node."""
        self.avg_layer_latency_ms = latency_ms

    def roofline_layer_latency_ms(self) -> float:
        """Get the roofline layer latency for this node."""
        perf_model = RooflinePerformanceModel(
            hardware=self.hardware,
            model_info=self.model_info,
            quantization_speedup=self.quantization_speedup,
            batch_size=self.current_requests,
            target_seq_len=1,
            source_seq_len=self.max_sequence_length,
        )
        return perf_model.roofline_layer_latency_ms(
            include_input_embed=self.has_embedding,
            include_lm_head=self.has_lm_head,
            num_decoder_layers=self.num_decoder_layers,
        )

    @property
    def layer_latency_ms(self) -> float:
        """Get effective layer latency considering both roofline and load."""
        if self.is_overloaded:
            return float("inf")
        if self.avg_layer_latency_ms is None:
            return self.roofline_layer_latency_ms()
        return self.avg_layer_latency_ms

    def update_rtt(self, target_node_id: str, rtt_ms: float):
        """Update RTT measurement to another node."""
        self.rtt_to_nodes[target_node_id] = rtt_ms

    def get_rtt_to(self, other: "Node") -> float:
        """Get RTT to another node, measuring via `rtt_getter` if needed.

        Falls back to 0.0 if no getter is provided and no cached RTT exists.
        """
        if self == other:
            return 0.0
        if other.node_id in self.rtt_to_nodes:
            return self.rtt_to_nodes[other.node_id]
        if self.rtt_getter is None:
            return 0.0
        rtt_ms = float(self.rtt_getter(self, other))
        self.update_rtt(other.node_id, rtt_ms)
        return rtt_ms

    def hosts_layer(self, layer_id: int) -> bool:
        """Return True if this node hosts the given layer id.

        Interprets `current_layers` as a half-open interval [start, end).
        """
        if self.current_layers is None:
            return False
        start_layer, end_layer = self.current_layers
        return start_layer <= layer_id < end_layer

    def add_request(self):
        """Add a request to this node."""
        self.current_requests += 1

    def remove_request(self):
        """Remove a request from this node."""
        self.current_requests -= 1
