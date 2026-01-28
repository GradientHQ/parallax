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

    # Use scheduler bootstrap so RR pipelines are registered in NodeManager.
    sched = Scheduler(
        model, [n1, n2], strategy="greedy", routing_strategy="rr", min_nodes_bootstrapping=1
    )
    ok = sched.bootstrap()
    assert ok
    allocs = sched.node_manager.list_node_allocations(model.num_layers)
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
    sched = Scheduler(
        model, [n1, n2], strategy="greedy", routing_strategy="dp", min_nodes_bootstrapping=1
    )

    # Join a new node
    n3 = build_node("rtx4090-x", model, tflops=82.6, mem_gb=24.0, x=0, y=1)
    sched.enqueue_join(n3)
    sched._process_joins()
    assert n3.start_layer is not None and n3.end_layer is not None

    # Leave
    sched.enqueue_leave(n3.node_id)
    sched._process_leaves()
    assert n3 not in sched.node_manager.active_nodes
    assert n3 not in sched.node_manager.standby_nodes


def test_scheduler_bootstrap_wait_and_dynamic_events():
    """Scheduler waits for min nodes, bootstraps, then handles join/leave events."""
    model = build_model_info(12)
    # Start with no nodes assigned yet; bootstrap needs 2
    n1 = build_node("a100-0", model, tflops=312.0, mem_gb=80.0, x=0, y=0)
    sched = Scheduler(model, [], strategy="dp", routing_strategy="dp", min_nodes_bootstrapping=2)

    # Enqueue one join; should not bootstrap yet (insufficient nodes)
    sched.enqueue_join(n1)
    # Process events once (simulate part of event loop)
    sched._process_joins()  # type: ignore[attr-defined]
    assert sched.node_manager.num_nodes == 1
    assert not sched.node_manager.has_full_pipeline(model.num_layers)

    # Add second node and process join; now bootstrap should succeed
    n2 = build_node("5090-1", model, tflops=165.0, mem_gb=32.0, x=1, y=0)
    sched.enqueue_join(n2)
    sched._process_joins()  # type: ignore[attr-defined]
    # RTTs are needed for DP routing strategy
    set_rtt_from_coords(sched.node_manager.nodes)
    ok = sched.bootstrap()
    assert ok
    assert sched.node_manager.has_full_pipeline(model.num_layers)

    # Dynamic join after bootstrap should assign immediately
    n3 = build_node("rtx4090-x", model, tflops=82.6, mem_gb=24.0, x=0, y=1)
    sched.enqueue_join(n3)
    sched._process_joins()  # type: ignore[attr-defined]
    assert n3.start_layer is not None and n3.end_layer is not None
    print(sched.node_manager.list_node_allocations(model.num_layers))

    # Leave a non-critical node; if still full pipeline, no global rebalance forced
    remaining_before = sched.node_manager.has_full_pipeline(model.num_layers)
    sched.enqueue_leave(n3.node_id)
    sched._process_leaves()  # type: ignore[attr-defined]
    assert sched.node_manager.has_full_pipeline(model.num_layers) == remaining_before

    print(sched.node_manager.list_node_allocations(model.num_layers))

    for node in list(sched.node_manager.nodes):
        if node.start_layer is not None and node.end_layer is not None:
            sched.layer_allocator.deallocate(node)  # type: ignore[attr-defined]
    # Re-allocate only first node to make pipeline incomplete
    sched.layer_allocator.allocate(sched.node_manager.nodes[0], 0, model.num_layers - 1)  # type: ignore[attr-defined]
    # Now leave that node to break coverage and trigger global rebalance path
    core_id = sched.node_manager.nodes[0].node_id
    sched.enqueue_leave(core_id)
    sched._process_leaves()  # type: ignore[attr-defined]


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
    sched.enqueue_leave(n1.node_id)
    sched._process_leaves()  # type: ignore[attr-defined]

    assert n1 not in sched.node_manager.nodes
    assert not sched.node_manager.has_full_pipeline(model.num_layers)

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
    sched = Scheduler(model, [], strategy="dp", routing_strategy="dp", min_nodes_bootstrapping=2)

    # Step 1: n1 joins (not enough nodes yet)
    sched.enqueue_join(n1)
    sched._process_joins()  # type: ignore[attr-defined]
    assert sched.node_manager.num_nodes == 1
    assert sched.node_manager.num_standby_nodes == 1
    assert not sched.node_manager.has_full_pipeline(model.num_layers)

    # Step 2: n2 joins (now we have 2 nodes, should bootstrap)
    sched.enqueue_join(n2)
    sched._process_joins()  # type: ignore[attr-defined]
    set_rtt_from_coords(sched.node_manager.nodes)
    ok = sched.bootstrap()
    assert ok, "Bootstrap should succeed with 2 nodes"
    assert sched.node_manager.has_full_pipeline(model.num_layers)
    assert sched.node_manager.num_active_nodes == 2
    assert sched.node_manager.num_standby_nodes == 0

    # Step 3: n3 joins (dynamic join after bootstrap)
    sched.enqueue_join(n3)
    sched._process_joins()  # type: ignore[attr-defined]
    set_rtt_from_coords(sched.node_manager.nodes)
    assert n3.start_layer is not None and n3.end_layer is not None
    assert sched.node_manager.num_nodes == 3
    assert sched.node_manager.num_active_nodes == 3
    assert sched.node_manager.num_standby_nodes == 0
    print(sched.node_manager.list_node_allocations(model.num_layers))

    # Step 4: n1 leaves and rejoins
    n1_id = n1.node_id
    sched.enqueue_leave(n1_id)
    sched._process_leaves()  # type: ignore[attr-defined]
    assert n1 not in sched.node_manager.nodes
    assert sched.node_manager.num_nodes == 2
    print(sched.node_manager.list_node_allocations(model.num_layers))
    assert sched.node_manager.has_full_pipeline(model.num_layers)

    # Rejoin n1
    n1_rejoin = build_node("n1", model, tflops=312.0, mem_gb=138.0, x=0, y=0)
    sched.enqueue_join(n1_rejoin)
    sched._process_joins()  # type: ignore[attr-defined]
    set_rtt_from_coords(sched.node_manager.nodes)
    assert n1_rejoin.start_layer is not None and n1_rejoin.end_layer is not None
    assert sched.node_manager.num_nodes == 3
    assert sched.node_manager.has_full_pipeline(model.num_layers)

    # Step 5: n2 leaves and rejoins
    n2_id = n2.node_id
    sched.enqueue_leave(n2_id)
    sched._process_leaves()  # type: ignore[attr-defined]
    assert n2 not in sched.node_manager.nodes
    assert sched.node_manager.num_nodes == 2
    assert sched.node_manager.has_full_pipeline(model.num_layers)

    # Rejoin n2
    n2_rejoin = build_node("n2", model, tflops=312.0, mem_gb=138.0, x=1, y=0)
    sched.enqueue_join(n2_rejoin)
    sched._process_joins()  # type: ignore[attr-defined]
    set_rtt_from_coords(sched.node_manager.nodes)
    assert n2_rejoin.start_layer is not None and n2_rejoin.end_layer is not None
    assert sched.node_manager.num_nodes == 3
    assert sched.node_manager.has_full_pipeline(model.num_layers)

    # Step 6: n3 leaves and rejoins
    n3_id = n3.node_id
    sched.enqueue_leave(n3_id)
    sched._process_leaves()
    assert n3 not in sched.node_manager.nodes
    assert sched.node_manager.num_nodes == 2
    assert sched.node_manager.has_full_pipeline(model.num_layers)

    # Rejoin n3
    n3_rejoin = build_node("n3", model, tflops=312.0, mem_gb=138.0, x=2, y=0)
    sched.enqueue_join(n3_rejoin)
    sched._process_joins()  # type: ignore[attr-defined]
    set_rtt_from_coords(sched.node_manager.nodes)
    assert n3_rejoin.start_layer is not None and n3_rejoin.end_layer is not None
    assert sched.node_manager.num_nodes == 3
    assert sched.node_manager.has_full_pipeline(model.num_layers)

    # Final verification: all nodes should have layer assignments
    allocations = sched.node_manager.list_node_allocations(model.num_layers)
    assert len(allocations) == 3, "All 3 nodes should have layer assignments"
    # Verify full pipeline coverage
    total_covered = sum(e - s for _, s, e in allocations)
    assert total_covered >= model.num_layers, "All layers should be covered"


def test_rr_expand_pipelines_from_newly_joined_standby_nodes():
    """In RR mode, joining new STANDBY nodes after bootstrap should expand pipelines."""
    model = build_model_info(12)

    # Make single nodes capable of hosting full [0, L) so each can be its own pipeline.
    # Use large memory to ensure capacity >= num_layers.
    n1 = build_node("p0", model, tflops=312.0, mem_gb=400.0, x=0, y=0)
    n2 = build_node("p1", model, tflops=312.0, mem_gb=400.0, x=1, y=0)
    set_rtt_from_coords([n1, n2])

    sched = Scheduler(
        model, [n1], strategy="greedy", routing_strategy="rr", min_nodes_bootstrapping=1
    )
    ok = sched.bootstrap()
    assert ok

    registered = sched.node_manager.get_registered_pipelines()
    assert len(registered) == 1

    # Join another node; scheduler should keep it STANDBY, then expand RR pipelines
    # by allocating from STANDBY and extending registered pipelines.
    sched.enqueue_join(n2)
    sched._process_joins()  # type: ignore[attr-defined]

    # Throttle logic uses time; force allow another expansion attempt if needed.
    sched._rr_last_expand_ts = 0.0  # type: ignore[attr-defined]
    sched._maybe_expand_rr_pipelines()  # type: ignore[attr-defined]

    registered2 = sched.node_manager.get_registered_pipelines()
    assert len(registered2) == 2
    # Both nodes should now be ACTIVE
    assert sched.node_manager.num_active_nodes == 2


def test_complicated_rr():
    """In RR mode, joining new STANDBY nodes after bootstrap should expand pipelines."""
    model = build_model_info(44)

    # Make single nodes capable of hosting full [0, L) so each can be its own pipeline.
    # Use large memory to ensure capacity >= num_layers.
    n1 = build_node("p0", model, tflops=312.0, mem_gb=138.0, x=0, y=0)
    n2 = build_node("p1", model, tflops=312.0, mem_gb=138.0, x=1, y=0)
    n3 = build_node("p2", model, tflops=312.0, mem_gb=138.0, x=2, y=0)
    n4 = build_node("p3", model, tflops=312.0, mem_gb=138.0, x=1, y=0)
    set_rtt_from_coords([n1, n2, n3, n4])

    sched = Scheduler(
        model, [n1, n2], strategy="greedy", routing_strategy="rr", min_nodes_bootstrapping=2
    )
    ok = sched.bootstrap()
    assert ok

    registered = sched.node_manager.get_registered_pipelines()
    assert len(registered) == 1
    print(sched.node_manager.list_node_allocations(model.num_layers))

    # Join another node; scheduler should keep it STANDBY, then expand RR pipelines
    # by allocating from STANDBY and extending registered pipelines.
    sched.enqueue_join(n3)
    sched._process_joins()  # type: ignore[attr-defined]
    registered = sched.node_manager.get_registered_pipelines()
    assert len(registered) == 1
    print(sched.node_manager.list_node_allocations(model.num_layers))
    assert sched.node_manager.num_active_nodes == 2
    assert sched.node_manager.num_standby_nodes == 1

    sched.enqueue_join(n4)
    sched._process_joins()  # type: ignore[attr-defined]
    sched._maybe_expand_rr_pipelines()  # type: ignore[attr-defined]

    registered2 = sched.node_manager.get_registered_pipelines()
    print(sched.node_manager.list_node_allocations(model.num_layers))
    assert len(registered2) == 2
    assert sched.node_manager.num_active_nodes == 4
    assert sched.node_manager.num_standby_nodes == 0

    sched.enqueue_leave(n3.node_id)
    sched._process_leaves()  # type: ignore[attr-defined]
    assert n3 not in sched.node_manager.nodes
    assert sched.node_manager.num_nodes == 3
    assert sched.node_manager.num_active_nodes == 2
    assert sched.node_manager.num_standby_nodes == 1
    # Leaving any member should invalidate its entire registered pipeline.
    registered_after_leave = sched.node_manager.get_registered_pipelines()
    assert len(registered_after_leave) == 1
    assert all(n3.node_id not in p and n4.node_id not in p for p in registered_after_leave.values())

    sched.enqueue_join(n3)
    sched._process_joins()  # type: ignore[attr-defined]
    assert n3 in sched.node_manager.nodes
    assert sched.node_manager.num_nodes == 4
    assert sched.node_manager.num_active_nodes == 4
    assert sched.node_manager.num_standby_nodes == 0

    sched.enqueue_leave(n1.node_id)
    sched.enqueue_leave(n4.node_id)
    sched._process_leaves()  # type: ignore[attr-defined]
    assert n1 not in sched.node_manager.nodes
    assert n4 not in sched.node_manager.nodes
    assert sched.node_manager.num_nodes == 2
    print(sched.node_manager.list_node_allocations(model.num_layers))
    # REBOOT
    assert sched.node_manager.num_active_nodes == 2
    assert sched.node_manager.num_standby_nodes == 0
