from __future__ import annotations

import pytest

from scheduling.node_management import NodeManager, NodeState
from tests.scheduler_tests.test_utils import build_model_info, build_node


def test_num_full_pipelines_counts_paths_over_active_allocations():
    # total_layers = 4
    model = build_model_info(4)

    # Build 4 nodes:
    # - a: [0,4) direct
    # - b: [0,2)
    # - c1: [2,4)
    # - c2: [2,4) (alternative tail)
    a = build_node("a", model, mem_gb=80.0)
    b = build_node("b", model, mem_gb=80.0)
    c1 = build_node("c1", model, mem_gb=80.0)
    c2 = build_node("c2", model, mem_gb=80.0)

    a.set_layer_allocation(0, 4)
    b.set_layer_allocation(0, 2)
    c1.set_layer_allocation(2, 4)
    c2.set_layer_allocation(2, 4)

    reg = NodeManager(initial_nodes=[a, b, c1, c2])
    # By default, initial nodes are STANDBY. Activate all four.
    reg.activate([a.node_id, b.node_id, c1.node_id, c2.node_id])

    # Pipelines:
    # - [a]
    # - [b, c1]
    # - [b, c2]
    assert reg.num_full_pipelines(4) == 3


def test_list_full_pipelines_respects_max_pipelines_cap():
    model = build_model_info(4)
    b = build_node("b", model, mem_gb=80.0)
    c1 = build_node("c1", model, mem_gb=80.0)
    c2 = build_node("c2", model, mem_gb=80.0)
    b.set_layer_allocation(0, 2)
    c1.set_layer_allocation(2, 4)
    c2.set_layer_allocation(2, 4)
    reg = NodeManager(initial_nodes=[b, c1, c2])
    reg.activate([b.node_id, c1.node_id, c2.node_id])

    assert reg.num_full_pipelines(4) == 2


def test_num_full_pipelines_respects_active_state():
    model = build_model_info(4)
    b = build_node("b", model, mem_gb=80.0)
    c = build_node("c", model, mem_gb=80.0)
    b.set_layer_allocation(0, 2)
    c.set_layer_allocation(2, 4)

    reg = NodeManager(initial_nodes=[b, c])

    # Only activate the head; tail remains STANDBY -> no full pipeline.
    reg.activate([b.node_id])
    assert reg.num_full_pipelines(4) == 0

    reg.activate([c.node_id])
    assert reg.num_full_pipelines(4) == 1


def test_num_full_pipelines_raises_on_invalid_ranges_but_ignores_unallocated_ranges():
    model = build_model_info(4)
    n1 = build_node("n1", model, mem_gb=80.0)
    n2 = build_node("n2", model, mem_gb=80.0)
    n3 = build_node("n3", model, mem_gb=80.0)

    # n1 allocated properly
    n1.set_layer_allocation(0, 4)
    # n2 unallocated (None, None) -> ignored
    # n3 invalid (end > total_layers) -> raises
    n3.set_layer_allocation(0, 999)

    reg = NodeManager(initial_nodes=[n1, n2, n3])
    reg.activate([n1.node_id, n2.node_id, n3.node_id])

    with pytest.raises(ValueError):
        _ = reg.num_full_pipelines(4)


def test_pipeline_min_load_and_total_capacity_computes_bottleneck_remaining_capacity():
    model = build_model_info(4)
    a = build_node("a", model, mem_gb=80.0)
    b = build_node("b", model, mem_gb=80.0)
    c = build_node("c", model, mem_gb=80.0)
    d = build_node("d", model, mem_gb=80.0)
    reg = NodeManager(initial_nodes=[a, b, c, d])

    # Pipelines: [a,b] and [c,d]
    reg.register_pipelines([[a.node_id, b.node_id], [c.node_id, d.node_id]])

    # All nodes have max_requests=16 in tests (_force_max_concurrent_requests=True)
    a.current_requests = 3  # remaining 13
    b.current_requests = 10  # remaining 6 -> pipeline0 bottleneck = 6
    c.current_requests = 0  # remaining 16
    d.current_requests = 15  # remaining 1 -> pipeline1 bottleneck = 1

    per, total, cur = reg.report_pipeline_capacity()
    assert per == {0: (16, 6), 1: (16, 1)}
    assert total == 32
    assert cur == 7


def test_remove_detaches_pipeline_and_clears_remaining_members():
    model = build_model_info(4)
    a = build_node("a", model, mem_gb=80.0)
    b = build_node("b", model, mem_gb=80.0)
    a.set_layer_allocation(0, 2)
    b.set_layer_allocation(2, 4)

    reg = NodeManager(initial_nodes=[a, b])
    reg.activate([a.node_id, b.node_id])
    reg.register_pipelines([[a.node_id, b.node_id]])

    removed = reg.remove(a.node_id)
    assert removed is a
    assert reg.get_registered_pipelines() == {}
    assert reg.pipeline_id_of_node(b.node_id) is None
    assert reg.state_of(b.node_id) == NodeState.STANDBY
    assert b.start_layer is None and b.end_layer is None
