"""
Tests for Phase 1: layer allocation and rebalancing.

Covers:
- Capacity sanity for `Node`
- Water-filling rebalancing within a pipeline (stage splits across heterogeneous nodes)
- Gap-patch dynamic rebalancing (join/leave behavior)
- Greedy and DP allocators producing contiguous [start, end) layer ranges
"""

from collections import Counter
from typing import Literal

import pytest

from scheduling.layer_allocation import (
    BaseLayerAllocator,
    DynamicProgrammingLayerAllocator,
    GreedyLayerAllocator,
)
from scheduling.model_info import ModelInfo
from scheduling.node import Node, NodeHardwareInfo

from .test_utils import build_model_info


def _build_node(gpu_type: str, model: ModelInfo, id_suffix: str = "") -> Node:
    hw_map = {
        "a100-80g": NodeHardwareInfo("a100-80g" + id_suffix, 1, 312.0, "", 80.0, 2039.0, "cuda"),
        "a100-40g": NodeHardwareInfo("a100-40g" + id_suffix, 1, 312.0, "", 40.0, 1935.0, "cuda"),
        "rtx5090": NodeHardwareInfo("rtx5090" + id_suffix, 1, 165, "", 32.0, 1792.0, "cuda"),
        "rtx4090": NodeHardwareInfo("rtx4090" + id_suffix, 1, 82.6, "", 24.0, 1008.0, "cuda"),
    }
    hw = hw_map[gpu_type]
    return Node(node_id=hw.node_id, hardware=hw, model_info=model)


def test_capacity_sanity_check():
    """Sanity check for capacity calculations."""
    # 36 layers for actual GPT-OSS 120B
    model = build_model_info(36)
    print(f"decoder layer flops: {model.decoder_layer_flops}")
    print(f"lm head flops: {model.lm_head_flops}")
    print(f"decoder layer io in GB: {model.decoder_layer_io_bytes(roofline=False) / (1024 ** 3)}")
    print(f"embedding table in GB: {model.embedding_io_bytes / (1024 ** 3)}")

    for gpu_type in ["a100-80g", "a100-40g", "rtx5090", "rtx4090"]:
        # (capacity, with embed) -> (13, 13), (6, 6), (5, 5), (4, 3)
        node = _build_node(gpu_type, model)
        capacity = node.get_decoder_layer_capacity()
        capacity_with_embed = node.get_decoder_layer_capacity(include_input_embed=True)
        assert capacity_with_embed <= capacity


@pytest.mark.parametrize(
    "num_layers,gpu_types,expected_layers",
    [
        (21, ["a100-80g", "rtx5090", "rtx4090"], [13, 5, 3]),
        (15, ["a100-80g", "rtx5090"], [11, 4]),
        # (20 * 312 : 20 * 165 : 20 * 82.6) / 559.6 = 11.1 : 5.8 : 2.9 -> 12 : 5 : 3
        (20, ["a100-80g", "rtx5090", "rtx4090"], [12, 5, 3]),
        (25, ["a100-80g", "rtx5090", "rtx4090", "rtx4090"], [13, 5, 4, 3]),
        (29, ["rtx4090", "a100-80g", "rtx5090", "rtx5090", "rtx4090"], [3, 13, 5, 5, 3]),
        (8, ["rtx5090", "rtx5090"], [4, 4]),
        (7, ["a100-40g", "rtx5090"], [5, 2]),
    ],
)
def test_water_filling_rebalance(num_layers: int, gpu_types: list[str], expected_layers: list[int]):
    """Test water-filling rebalancer with various GPU configurations."""
    model = build_model_info(num_layers)

    nodes = [_build_node(g, model, id_suffix=f"-{i}") for i, g in enumerate(gpu_types)]

    allocator = GreedyLayerAllocator(model, nodes)
    allocator.adjust_pipeline_layers(nodes, assume_sorted=False)

    actual_layers = []
    for node in nodes:
        assert node.start_layer is not None and node.end_layer is not None
        actual_layers.append(node.end_layer - node.start_layer)

    assert sum(actual_layers) == num_layers
    # Order-insensitive: only the multiset of stage sizes must match
    assert Counter(actual_layers) == Counter(expected_layers)

    for i, node in enumerate(nodes):
        cap = node.get_decoder_layer_capacity(
            include_input_embed=(i == 0), include_lm_head=(i == len(nodes) - 1)
        )
        assert actual_layers[i] <= cap


def _test_gap_patch_rebalance(allocator: BaseLayerAllocator):
    """Sanity checks for gap-patch dynamic rebalancing using allocator state."""
    assert allocator.layer_loads_heap, "Layer loads heap should not be empty"
    assert allocator.node_id_to_node, "Allocator should have node mappings"
    model_info = allocator.model_info

    # Heap top should include a host with minimal per-layer KV memory
    per_node_mem = {
        nid: (node.per_decoder_layer_kv_cache_memory or 0)
        for nid, node in allocator.node_id_to_node.items()
        if nid in allocator.node_allocation
    }
    min_mem = min(per_node_mem.values()) if per_node_mem else 0

    top_load = allocator.layer_loads_heap[0]
    assert top_load.hosting_nodes
    top_hosts_min = min(per_node_mem.get(h, float("inf")) for h in top_load.hosting_nodes)
    assert top_hosts_min == min_mem

    # Join a new small GPU and verify hosting set updates
    before_layer_id = top_load.layer_id
    before_mem = allocator.layer_to_load[before_layer_id].current_kv_size

    new_node = _build_node("rtx4090", model_info, id_suffix="-gap")
    allocator.join(new_node)

    after_mem = allocator.layer_to_load[before_layer_id].current_kv_size
    assert after_mem >= before_mem
    assert new_node.node_id in allocator.layer_to_load[before_layer_id].hosting_nodes

    allocator.leave(new_node.node_id)
    restored_mem = allocator.layer_to_load[before_layer_id].current_kv_size
    assert new_node.node_id not in allocator.layer_to_load[before_layer_id].hosting_nodes
    assert restored_mem == before_mem


@pytest.mark.parametrize(
    "num_layers,counts,expected_ranges,strategy",
    [
        # Six A100-80g: expect two pipelines, 12 each per stage in creation order
        (36, (6, 0, 0, 0), [(0, 12), (12, 24), (24, 36), (0, 12), (12, 24), (24, 36)], "greedy"),
        (36, (6, 0, 0, 0), [(0, 12), (12, 24), (24, 36), (0, 12), (12, 24), (24, 36)], "dp"),
        # 22 Layers, capacity (13, 13, 6, 6, 3, 3) -> greedy assigns (11, 11)
        (
            22,
            (2, 2, 0, 2),
            [
                (0, 11),
                (11, 22),
            ],
            "greedy",
        ),
        # For DP, we expect two pipelines, 13 each per stage in creation order
        (
            22,
            (2, 2, 0, 2),
            [
                (0, 13),
                (13, 19),
                (19, 22),
                (0, 13),
                (13, 19),
                (19, 22),
            ],
            "dp",
        ),
        # 14 Layers, capacity (13, 5, 5, 3, 3) -> greedy assigns (10, 4)
        (
            14,
            (1, 0, 2, 2),
            [
                (0, 10),
                (10, 14),
            ],
            "greedy",
        ),
        # 7 Layers, capacity (6, 5, 5, 3, 3) -> greedy assigns (5, 2, 4, 3)
        (
            7,
            (0, 1, 2, 2),
            [
                (0, 5),
                (5, 7),
                (0, 4),
                (4, 7),
            ],
            "greedy",
        ),
    ],
)
def test_allocator(
    num_layers: int,
    counts: tuple[int, int, int, int],
    expected_ranges: list[tuple[int, int]],
    strategy: Literal["greedy", "dp"],
):
    """Test allocator with greedy and dp strategies."""
    model = build_model_info(num_layers)

    n_a100_80g, n_a100_40g, n_5090, n_4090 = counts

    nodes: list[Node] = []
    for i in range(n_a100_80g):
        nodes.append(_build_node("a100-80g", model, id_suffix=f"-{i}"))
    for i in range(n_a100_40g):
        nodes.append(_build_node("a100-40g", model, id_suffix=f"-{i}"))
    for i in range(n_5090):
        nodes.append(_build_node("rtx5090", model, id_suffix=f"-{i}"))
    for i in range(n_4090):
        nodes.append(_build_node("rtx4090", model, id_suffix=f"-{i}"))

    allocator = (
        GreedyLayerAllocator(model, nodes, assign_left_over_nodes=False)
        if strategy == "greedy"
        else DynamicProgrammingLayerAllocator(model, nodes, assign_left_over_nodes=False)
    )
    allocator.global_allocation()
    _test_gap_patch_rebalance(allocator)

    # Collect (start,end) per node in creation order
    actual_ranges: list[tuple[int, int]] = []
    for node in nodes:
        if node.start_layer is None or node.end_layer is None:
            actual_ranges.append((0, 0))
        else:
            actual_ranges.append((node.start_layer, node.end_layer))

    # Trim trailing (0,0) if no assignment expected
    actual_trimmed = [r for r in actual_ranges if r != (0, 0)]
    expected_total = sum(e - s for (s, e) in expected_ranges)
    assert sum(e - s for (s, e) in actual_trimmed) == expected_total
    # Order-insensitive comparison: ranges represent stages; allow pipeline reordering
    assert Counter(actual_trimmed) == Counter(
        expected_ranges
    ), f"Stage ranges mismatch (order-insensitive):\nactual={actual_trimmed}\nexpected={expected_ranges}"


@pytest.mark.parametrize("strategy", ["greedy", "dp"])
def test_single_node_can_host_all_layers_greedy(strategy: Literal["greedy", "dp"]):
    """Small model fully hosted by a single strong node (A100-80g)."""
    model = build_model_info(5)
    node = _build_node("a100-80g", model)
    alloc = (
        GreedyLayerAllocator(model, [node])
        if strategy == "greedy"
        else DynamicProgrammingLayerAllocator(model, [node])
    )
    initialized = alloc.global_allocation()
    assert initialized is True
    assert alloc.has_full_pipeline() is True
    assert node.start_layer == 0 and node.end_layer == model.num_layers


@pytest.mark.parametrize("strategy", ["greedy", "dp"])
def test_mixed_pool_single_host_available(strategy: Literal["greedy", "dp"]):
    """Intermediate model: one A100 can host alone; others (4090) remain unused."""
    model = build_model_info(6)
    a100 = _build_node("a100-80g", model, id_suffix="-a")
    r1 = _build_node("rtx4090", model, id_suffix="-1")
    r2 = _build_node("rtx4090", model, id_suffix="-2")
    alloc = (
        GreedyLayerAllocator(model, [a100, r1, r2])
        if strategy == "greedy"
        else DynamicProgrammingLayerAllocator(model, [a100, r1, r2])
    )
    initialized = alloc.global_allocation()
    assert initialized is True
    # A100 should cover entire model
    assert a100.start_layer == 0 and a100.end_layer == model.num_layers
    assert r1.start_layer == 0 and r1.end_layer == 3
    assert r2.start_layer == 3 and r2.end_layer == model.num_layers


@pytest.mark.parametrize("strategy", ["greedy", "dp"])
def test_pipeline_required_with_midrange_only(strategy: Literal["greedy", "dp"]):
    """Model requires pipeline across multiple mid-range GPUs (RTX4090)."""
    model = build_model_info(7)
    nodes = [_build_node("rtx4090", model, id_suffix=f"-{i}") for i in range(3)]
    alloc = (
        GreedyLayerAllocator(model, nodes)
        if strategy == "greedy"
        else DynamicProgrammingLayerAllocator(model, nodes)
    )
    ok = alloc.global_allocation()
    assert ok is True
    # At least two nodes should be assigned to cover 7 layers
    assigned = [(n.node_id, n.start_layer, n.end_layer) for n in nodes if n.start_layer is not None]
    assert len(assigned) >= 2
    total = sum(e - s for _, s, e in assigned)
    assert total == model.num_layers


@pytest.mark.parametrize("strategy", ["greedy", "dp"])
def test_allocator_does_not_duplicate_leftover_nodes(strategy: Literal["greedy", "dp"]):
    """Both allocators should not duplicate self.nodes when left over after allocation.

    Greedy: builds multiple pipelines, leaves nodes that can't form another pipeline
    DP: optimizes for best pipeline configuration, may leave suboptimal nodes unallocated
    """
    model = build_model_info(12)

    if strategy == "greedy":
        # Two strong nodes form two complete pipelines (greedy maximizes pipelines)
        # One weak node left unallocated
        a100_1 = _build_node("a100-80g", model, id_suffix="-a1")
        a100_2 = _build_node("a100-80g", model, id_suffix="-a2")
        r1 = _build_node("rtx4090", model, id_suffix="-1")
        nodes = [a100_1, a100_2, r1]
        expected_node_count = 3
    else:
        # One strong node handles all layers (DP finds this optimal)
        # One weak node left unallocated
        a100 = _build_node("a100-80g", model, id_suffix="-a")
        r1 = _build_node("rtx4090", model, id_suffix="-1")
        nodes = [a100, r1]
        expected_node_count = 2

    alloc = (
        GreedyLayerAllocator(model, nodes)
        if strategy == "greedy"
        else DynamicProgrammingLayerAllocator(model, nodes)
    )
    ok = alloc.global_allocation()
    assert ok is True
    assert len(alloc.nodes) == expected_node_count, "Should not duplicate nodes during allocation"
