"""
Reusable test helpers for model/node builders and RTT utilities.
"""

from __future__ import annotations

from math import sqrt
from typing import Dict, Iterable, List, Tuple

from scheduling.model_info import ModelInfo
from scheduling.node import Node, NodeHardwareInfo

A100_80G = NodeHardwareInfo(
    node_id="a100-80g",
    num_gpus=1,
    tflops_fp16=312.0,
    gpu_name="",
    memory_gb=80.0,
    memory_bandwidth_gbps=2039,
    device="cuda",
)
A100_40G = NodeHardwareInfo(
    node_id="a100-40g",
    num_gpus=1,
    tflops_fp16=312.0,
    gpu_name="",
    memory_gb=40.0,
    memory_bandwidth_gbps=1935,
    device="cuda",
)
RTX5090 = NodeHardwareInfo(
    node_id="rtx5090",
    num_gpus=1,
    tflops_fp16=104.8,
    gpu_name="",
    memory_gb=32.0,
    memory_bandwidth_gbps=1792,
    device="cuda",
)
RTX4090 = NodeHardwareInfo(
    node_id="rtx4090",
    num_gpus=1,
    tflops_fp16=82.6,
    gpu_name="",
    memory_gb=24.0,
    memory_bandwidth_gbps=1008,
    device="cuda",
)


def build_model_info(num_layers: int) -> ModelInfo:
    """Build a model config used across tests (matches allocation tests)."""
    return ModelInfo(
        model_name=f"GPUOss-{num_layers}L",
        mlx_model_name=f"MLXOss-{num_layers}L",
        head_size=64,
        hidden_dim=2880,
        intermediate_dim=2880,
        num_attention_heads=64,
        num_kv_heads=8,
        vocab_size=201088,
        num_layers=num_layers,
        ffn_num_projections=3,
        num_local_experts=128,
        num_experts_per_tok=4,
        param_bytes_per_element=1,
        mlx_param_bytes_per_element=1,
        cache_bytes_per_element=2,
        embedding_bytes_per_element=2,
    )


def build_node(
    node_id: str,
    model: ModelInfo,
    tflops: float = 200.0,
    mem_gb: float = 80.0,
    x: float = 0.0,
    y: float = 0.0,
    mem_bandwidth_gbps: float = 100.0,
) -> Node:
    """Create a `Node` with hardware info and attach test-only coordinates/bandwidth."""
    hw = NodeHardwareInfo(
        node_id=node_id,
        num_gpus=1,
        tflops_fp16=tflops,
        gpu_name="",
        memory_gb=mem_gb,
        memory_bandwidth_gbps=mem_bandwidth_gbps,
        device="cuda",
    )
    n = Node(node_id=node_id, hardware=hw, model_info=model, _force_max_concurrent_requests=True)
    # Attach coordinates for RTT synthesis in tests
    setattr(n, "_x", float(x))
    setattr(n, "_y", float(y))
    # Ensure roofline uses a defined speedup
    setattr(n, "quantization_speedup", 1.0)
    return n


def compute_rtts_from_coords(nodes: Iterable[Node]) -> Dict[Tuple[str, str], float]:
    """Map Euclidean distances between nodes' (x, y) to RTTs in [10, 200] ms."""
    node_list = list(nodes)
    if not node_list:
        return {}
    coords: Dict[str, Tuple[float, float]] = {
        n.node_id: (
            float(getattr(n, "_x", 0.0)),
            float(getattr(n, "_y", 0.0)),
        )
        for n in node_list
    }
    ids = [n.node_id for n in node_list]

    max_dist = 0.0
    for i, aid in enumerate(ids):
        ax, ay = coords[aid]
        for bid in ids[i + 1 :]:
            bx, by = coords[bid]
            d = sqrt((ax - bx) ** 2 + (ay - by) ** 2)
            max_dist = max(max_dist, d)

    def to_latency(d: float) -> float:
        return 10.0 if max_dist <= 0 else 10.0 + 190.0 * (d / max_dist)

    rtts: Dict[Tuple[str, str], float] = {(nid, nid): 10.0 for nid in ids}
    for i, aid in enumerate(ids):
        ax, ay = coords[aid]
        for bid in ids[i + 1 :]:
            bx, by = coords[bid]
            d = sqrt((ax - bx) ** 2 + (ay - by) ** 2)
            lat = to_latency(d)
            rtts[(aid, bid)] = lat
            rtts[(bid, aid)] = lat
    return rtts


def set_rtt_from_coords(nodes: List[Node]) -> None:
    """Populate `rtt_to_nodes` on each node based on their coordinates."""
    all_rtts = compute_rtts_from_coords(nodes)
    node_map = {n.node_id: n for n in nodes}
    ids = list(node_map.keys())

    for aid in ids:
        node_a = node_map[aid]
        if node_a.rtt_to_nodes is None:
            node_a.rtt_to_nodes = {}
        for bid in ids:
            if aid == bid:
                continue
            rtt = all_rtts.get((aid, bid))
            if rtt is not None:
                node_a.rtt_to_nodes[bid] = rtt


def geo_rtt_provider(positions: Dict[str, Tuple[float, float]]):
    """Create an RTT provider mapping Euclidean distance to [10, 200] ms.

    Scales by the maximum pairwise distance among provided positions.
    """
    ids = list(positions.keys())
    # Compute max pairwise distance for scaling
    max_dist = 0.0
    for i, aid in enumerate(ids):
        ax, ay = positions[aid]
        for bid in ids[i + 1 :]:
            bx, by = positions[bid]
            max_dist = max(max_dist, ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)

    def to_latency(d: float) -> float:
        return 10.0 if max_dist <= 0 else 10.0 + 190.0 * (d / max_dist)

    def provider(src: Node, dst: Node) -> float:
        if src.node_id == dst.node_id:
            return 0.0
        sx, sy = positions.get(src.node_id, (0.0, 0.0))
        dx, dy = positions.get(dst.node_id, (0.0, 0.0))
        dist = ((sx - dx) ** 2 + (sy - dy) ** 2) ** 0.5
        return to_latency(dist)

    return provider
