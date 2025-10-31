"""
Minimal tests for the Scheduler orchestrator.
"""

from __future__ import annotations

from scheduling.model_info import ModelInfo
from scheduling.node import Node, NodeHardwareInfo, RequestSignal
from scheduling.scheduler import Scheduler

from .test_utils import build_model_info


def _build_node(node_id: str, model: ModelInfo, *, tflops: float, mem_gb: float) -> Node:
    hw = NodeHardwareInfo(
        node_id=node_id,
        tflops_fp16=tflops,
        gpu_name="",
        memory_gb=mem_gb,
        memory_bandwidth_gbps=1000.0,
        device="cuda",
    )
    n = Node(node_id=node_id, hardware=hw, model_info=model)
    # Ensure latency estimation uses a defined speedup
    setattr(n, "quantization_speedup", 1.0)
    return n


def test_scheduler_initialize_and_dispatch():
    """Allocate, then enqueue one request and dispatch it."""
    model = build_model_info(12)
    n1 = _build_node("a100-0", model, tflops=312.0, mem_gb=80.0)
    n2 = _build_node("a100-1", model, tflops=312.0, mem_gb=80.0)

    sched = Scheduler(model, [n1, n2], strategy="greedy", min_nodes_bootstrapping=1)
    sched.layer_allocator.global_allocation()
    allocs = sched.list_node_allocations()
    assert allocs, "Allocator should assign at least one pipeline"
    # Check coverage equals total layers
    total = sum(e - s for _, s, e in allocs)
    assert total >= model.num_layers

    # Push a request and dispatch
    req = RequestSignal(request_id="req-1")
    sched.receive_request(req)
    assignment = sched.dispatch_next_request()
    assert assignment is not None
    req_id, path, latency = assignment
    assert req_id == req.request_id
    assert path, "Path should be non-empty"
    assert latency >= 0.0


def test_scheduler_join_and_leave():
    """New node can join and be assigned; leave removes it and may rebalance."""
    model = build_model_info(12)
    n1 = _build_node("a100-0", model, tflops=312.0, mem_gb=80.0)
    n2 = _build_node("a100-1", model, tflops=312.0, mem_gb=80.0)
    sched = Scheduler(model, [n1, n2], strategy="greedy", min_nodes_bootstrapping=1)

    # Join a new node
    n3 = _build_node("rtx4090-x", model, tflops=82.6, mem_gb=24.0)
    sched.join(n3)
    assert n3.start_layer is not None and n3.end_layer is not None

    # Leave
    sched.leave(n3.node_id)
    assert n3 not in sched.nodes


def test_scheduler_bootstrap_wait_and_dynamic_events():
    """Scheduler waits for min nodes, bootstraps, then handles join/leave events."""
    model = build_model_info(12)
    # Start with no nodes assigned yet; bootstrap needs 2
    n1 = _build_node("a100-0", model, tflops=312.0, mem_gb=80.0)
    sched = Scheduler(model, [], strategy="dp", min_nodes_bootstrapping=2)

    # Enqueue one join; should not bootstrap yet (insufficient nodes)
    sched.enqueue_join(n1)
    # Process events once (simulate part of event loop)
    sched._process_joins()  # type: ignore[attr-defined]
    assert len(sched.nodes) == 1
    assert not sched.layer_allocator.has_full_pipeline()

    # Add second node and process join; now bootstrap should succeed
    n2 = _build_node("5090-1", model, tflops=165.0, mem_gb=32.0)
    sched.enqueue_join(n2)
    sched._process_joins()  # type: ignore[attr-defined]
    ok = sched.bootstrap()
    assert ok
    assert sched.layer_allocator.has_full_pipeline()

    # Dynamic join after bootstrap should assign immediately
    n3 = _build_node("rtx4090-x", model, tflops=82.6, mem_gb=24.0)
    sched.enqueue_join(n3)
    sched._process_joins()  # type: ignore[attr-defined]
    assert n3.start_layer is not None and n3.end_layer is not None
    print(sched.layer_allocator.list_node_allocations())

    # Leave a non-critical node; if still full pipeline, no global rebalance forced
    remaining_before = sched.layer_allocator.has_full_pipeline()
    sched.leave(n3.node_id)
    assert sched.layer_allocator.has_full_pipeline() == remaining_before

    print(sched.layer_allocator.list_node_allocations())

    for node in list(sched.nodes):
        if node.start_layer is not None and node.end_layer is not None:
            sched.layer_allocator.deallocate(node)
    # Re-allocate only first node to make pipeline incomplete
    sched.layer_allocator.allocate(sched.nodes[0], 0, model.num_layers - 1)
    # Now leave that node to break coverage and trigger global rebalance path
    core_id = sched.nodes[0].node_id
    sched.leave(core_id)


def test_scheduler_single_node_leave_then_rejoin_reassigns_layers():
    """With one node, after leave then re-join, layers should be re-assigned.

    Reproduction of observed issue: when `min_nodes_bootstrapping=1`, after killing the
    only node (leave) and re-joining it, the scheduler fails to re-assign layers.
    This test encodes the expected behavior (should re-assign), so it currently fails.
    """
    model = build_model_info(12)

    # Start with a single capable node and bootstrap successfully
    n1 = _build_node("solo-0", model, tflops=312.0, mem_gb=80.0)
    sched = Scheduler(model, [n1], strategy="dp", min_nodes_bootstrapping=1)
    ok = sched.bootstrap()
    assert ok
    assert n1.start_layer is not None and n1.end_layer is not None

    # Simulate node leave (e.g., the process was killed)
    sched.leave(n1.node_id)
    assert n1 not in sched.nodes
    assert not sched.layer_allocator.has_full_pipeline()

    # Re-join the (same) node id; scheduler should re-assign layers
    n1_rejoin = _build_node("solo-0", model, tflops=312.0, mem_gb=80.0)
    sched.enqueue_join(n1_rejoin)
    sched._process_joins()  # type: ignore[attr-defined]

    # Expected behavior: after re-join with min_nodes_bootstrapping=1, layers are assigned again
    assert (
        n1_rejoin.start_layer is not None and n1_rejoin.end_layer is not None
    ), "After re-join, single node should be assigned a full layer range"
