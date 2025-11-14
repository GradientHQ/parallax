"""
Minimal tests for the Scheduler orchestrator.
"""

from __future__ import annotations

from scheduling.node import RequestSignal
from scheduling.scheduler import Scheduler

from .test_utils import build_model_info, build_node, set_rtt_from_coords


def test_scheduler_initialize_and_dispatch():
    """Allocate, then enqueue one request and dispatch it."""
    model = build_model_info(12)
    n1 = build_node("a100-0", model, tflops=312.0, mem_gb=80.0, x=0, y=0)
    n2 = build_node("a100-1", model, tflops=312.0, mem_gb=80.0, x=1, y=0)
    set_rtt_from_coords([n1, n2])

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
    n1 = build_node("a100-0", model, tflops=312.0, mem_gb=80.0, x=0, y=0)
    n2 = build_node("a100-1", model, tflops=312.0, mem_gb=80.0, x=1, y=0)
    set_rtt_from_coords([n1, n2])
    sched = Scheduler(model, [n1, n2], strategy="greedy", min_nodes_bootstrapping=1)

    # Join a new node
    n3 = build_node("rtx4090-x", model, tflops=82.6, mem_gb=24.0, x=0, y=1)
    sched.join(n3)
    assert n3.start_layer is not None and n3.end_layer is not None

    # Leave
    sched.leave(n3.node_id)
    assert n3 not in sched.nodes


def test_scheduler_bootstrap_wait_and_dynamic_events():
    """Scheduler waits for min nodes, bootstraps, then handles join/leave events."""
    model = build_model_info(12)
    # Start with no nodes assigned yet; bootstrap needs 2
    n1 = build_node("a100-0", model, tflops=312.0, mem_gb=80.0, x=0, y=0)
    sched = Scheduler(model, [], strategy="dp", min_nodes_bootstrapping=2)

    # Enqueue one join; should not bootstrap yet (insufficient nodes)
    sched.enqueue_join(n1)
    # Process events once (simulate part of event loop)
    sched._process_joins()  # type: ignore[attr-defined]
    assert len(sched.nodes) == 1
    assert not sched.layer_allocator.has_full_pipeline()

    # Add second node and process join; now bootstrap should succeed
    n2 = build_node("5090-1", model, tflops=165.0, mem_gb=32.0, x=1, y=0)
    sched.enqueue_join(n2)
    sched._process_joins()  # type: ignore[attr-defined]
    # RTTs are needed for DP routing strategy
    set_rtt_from_coords(sched.nodes)
    ok = sched.bootstrap()
    assert ok
    assert sched.layer_allocator.has_full_pipeline()

    # Dynamic join after bootstrap should assign immediately
    n3 = build_node("rtx4090-x", model, tflops=82.6, mem_gb=24.0, x=0, y=1)
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
    n1 = build_node("solo-0", model, tflops=312.0, mem_gb=80.0, x=0, y=0)
    set_rtt_from_coords([n1])
    sched = Scheduler(model, [n1], strategy="dp", min_nodes_bootstrapping=1)
    ok = sched.bootstrap()
    assert ok
    assert n1.start_layer is not None and n1.end_layer is not None

    # Simulate node leave (e.g., the process was killed)
    sched.leave(n1.node_id)
    assert n1 not in sched.nodes
    assert not sched.layer_allocator.has_full_pipeline()

    # Re-join the (same) node id; scheduler should re-assign layers
    n1_rejoin = build_node("solo-0", model, tflops=312.0, mem_gb=80.0, x=0, y=0)
    sched.enqueue_join(n1_rejoin)
    sched._process_joins()  # type: ignore[attr-defined]

    # Expected behavior: after re-join with min_nodes_bootstrapping=1, layers are assigned again
    assert (
        n1_rejoin.start_layer is not None and n1_rejoin.end_layer is not None
    ), "After re-join, single node should be assigned a full layer range"


def test_scheduler_three_nodes_sequential_join_leave_rejoin():
    """Test scheduler with 28-layer model, 3 nodes each capable of 22 layers.

    Scenario:
    - 28-layer model
    - n1, n2, n3 all can host 22 layers
    - min_nodes_bootstrapping=2
    - n1, n2, n3 join sequentially
    - n1 leaves and rejoins
    - n2 leaves and rejoins
    - n3 leaves and rejoins
    """
    model = build_model_info(28)

    # Create nodes that can each host 22 layers
    # Calculation: 100GB can host 16 layers, so 22 layers need ~137.5GB
    # Using 150GB to ensure capacity for 22 layers with some margin
    n1 = build_node("n1", model, tflops=312.0, mem_gb=138.0, x=0, y=0)
    n2 = build_node("n2", model, tflops=312.0, mem_gb=138.0, x=1, y=0)
    n3 = build_node("n3", model, tflops=312.0, mem_gb=138.0, x=2, y=0)

    # Verify nodes can host 22 layers
    assert n1.get_decoder_layer_capacity() >= 22, "n1 should be able to host 22 layers"
    assert n2.get_decoder_layer_capacity() >= 22, "n2 should be able to host 22 layers"
    assert n3.get_decoder_layer_capacity() >= 22, "n3 should be able to host 22 layers"

    # Initialize scheduler with min_nodes_bootstrapping=2, no nodes initially
    sched = Scheduler(model, [], strategy="dp", min_nodes_bootstrapping=2)

    # Step 1: n1 joins (not enough nodes yet)
    sched.enqueue_join(n1)
    sched._process_joins()  # type: ignore[attr-defined]
    assert len(sched.nodes) == 1
    assert not sched.layer_allocator.has_full_pipeline()

    # Step 2: n2 joins (now we have 2 nodes, should bootstrap)
    sched.enqueue_join(n2)
    sched._process_joins()  # type: ignore[attr-defined]
    set_rtt_from_coords(sched.nodes)
    ok = sched.bootstrap()
    assert ok, "Bootstrap should succeed with 2 nodes"
    assert sched.layer_allocator.has_full_pipeline()

    # Step 3: n3 joins (dynamic join after bootstrap)
    sched.enqueue_join(n3)
    sched._process_joins()  # type: ignore[attr-defined]
    set_rtt_from_coords(sched.nodes)
    assert n3.start_layer is not None and n3.end_layer is not None
    assert len(sched.nodes) == 3

    # Step 4: n1 leaves and rejoins
    n1_id = n1.node_id
    sched.leave(n1_id)
    assert n1 not in sched.nodes
    assert len(sched.nodes) == 2
    assert sched.layer_allocator.has_full_pipeline()

    # Rejoin n1
    n1_rejoin = build_node("n1", model, tflops=312.0, mem_gb=138.0, x=0, y=0)
    sched.enqueue_join(n1_rejoin)
    sched._process_joins()  # type: ignore[attr-defined]
    set_rtt_from_coords(sched.nodes)
    assert n1_rejoin.start_layer is not None and n1_rejoin.end_layer is not None
    assert len(sched.nodes) == 3
    assert sched.layer_allocator.has_full_pipeline()

    # Step 5: n2 leaves and rejoins
    n2_id = n2.node_id
    sched.leave(n2_id)
    assert n2 not in sched.nodes
    assert len(sched.nodes) == 2
    assert sched.layer_allocator.has_full_pipeline()

    # Rejoin n2
    n2_rejoin = build_node("n2", model, tflops=312.0, mem_gb=138.0, x=1, y=0)
    sched.enqueue_join(n2_rejoin)
    sched._process_joins()  # type: ignore[attr-defined]
    set_rtt_from_coords(sched.nodes)
    assert n2_rejoin.start_layer is not None and n2_rejoin.end_layer is not None
    assert len(sched.nodes) == 3
    assert sched.layer_allocator.has_full_pipeline()

    # Step 6: n3 leaves and rejoins
    n3_id = n3.node_id
    sched.leave(n3_id)
    assert n3 not in sched.nodes
    assert len(sched.nodes) == 2
    assert sched.layer_allocator.has_full_pipeline()

    # Rejoin n3
    n3_rejoin = build_node("n3", model, tflops=312.0, mem_gb=138.0, x=2, y=0)
    sched.enqueue_join(n3_rejoin)
    sched._process_joins()  # type: ignore[attr-defined]
    set_rtt_from_coords(sched.nodes)
    assert n3_rejoin.start_layer is not None and n3_rejoin.end_layer is not None
    assert len(sched.nodes) == 3
    assert sched.layer_allocator.has_full_pipeline()

    # Final verification: all nodes should have layer assignments
    allocations = sched.list_node_allocations()
    assert len(allocations) == 3, "All 3 nodes should have layer assignments"
    # Verify full pipeline coverage
    total_covered = sum(e - s for _, s, e in allocations)
    assert total_covered >= model.num_layers, "All layers should be covered"
