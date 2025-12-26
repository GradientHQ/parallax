"""
Unit tests for request routing strategies.

Covers:
- Shard-level DP path using `Node` APIs (latency + RTT)
- Turning point detection via layer-level DP
- Parametrized scenarios with different splits/overlaps
- Fixed-pipeline round-robin routing and overload skipping
- Dynamic pipeline discovery enumeration
"""

import pytest

from scheduling.node import Node
from scheduling.request_routing import (
    DynamicProgrammingRouting,
    RandomizedOverDynamicPipelinesRouting,
    RoundRobinOverFixedPipelinesRouting,
    find_turning_points,
)

from .test_utils import build_model_info as build_model
from .test_utils import build_node, build_node_management, set_rtt_from_coords


def test_optimal_path_simple_chain():
    """Two-node chain [0, k) -> [k, L): verify path and latency sum."""
    num_layers = 12
    split = 7
    model = build_model(num_layers)
    n1 = build_node("n1", model, tflops=200.0, x=0.0, y=0.0)
    n2 = build_node("n2", model, tflops=200.0, x=1.0, y=0.0)
    n1.set_layer_allocation(0, split)
    n2.set_layer_allocation(split, num_layers)
    set_rtt_from_coords([n1, n2])

    router = DynamicProgrammingRouting()
    node_ids, latency = router.find_optimal_path([n1, n2], num_layers)

    assert node_ids == ["n1", "n2"]
    expected = float(n1.layer_latency_ms) + float(n1.get_rtt_to(n2)) + float(n2.layer_latency_ms)
    assert latency == pytest.approx(expected, rel=1e-6)


def test_optimal_path_single_node():
    """Single-node [0, L) path should have no hops and sum only node latency."""
    num_layers = 10
    model = build_model(num_layers)
    n = build_node("solo", model, tflops=250.0, x=0.0, y=0.0)
    n.set_layer_allocation(0, num_layers)

    router = DynamicProgrammingRouting()
    node_ids, latency = router.find_optimal_path([n], num_layers)

    assert node_ids == ["solo"]
    assert latency == pytest.approx(float(n.layer_latency_ms), rel=1e-6)


def test_optimal_path_missing_rtt():
    """If RTT is missing between two nodes in a path, it should be invalid."""
    num_layers = 12
    model = build_model(num_layers)
    n1 = build_node("n1", model, tflops=200.0, x=0.0, y=0.0)
    n2 = build_node("n2", model, tflops=200.0, x=1.0, y=0.0)
    n1.set_layer_allocation(0, 6)
    n2.set_layer_allocation(6, 12)
    nodes = [n1, n2]
    set_rtt_from_coords(nodes)

    # Manually remove the RTT info in both directions
    if n2.node_id in n1.rtt_to_nodes:
        del n1.rtt_to_nodes[n2.node_id]
    if n1.node_id in n2.rtt_to_nodes:
        del n2.rtt_to_nodes[n1.node_id]

    router = DynamicProgrammingRouting()
    node_ids, latency = router.find_optimal_path(nodes, num_layers)

    assert node_ids == []
    assert latency == float("inf")


@pytest.mark.parametrize(
    "num_layers,segments,expected_path",
    [
        (
            15,
            [("a", 0, 5, 200.0, 0.0), ("b", 5, 10, 200.0, 1.0), ("c", 10, 15, 200.0, 2.0)],
            ["a", "b", "c"],
        ),
        (
            12,
            [("full", 0, 12, 150.0, 0.0), ("tail", 6, 12, 300.0, 1.0)],
            ["full"],
        ),
    ],
)
def test_optimal_path_parametrized(
    num_layers: int,
    segments: list[tuple[str, int, int, float, float]],
    expected_path: list[str],
):
    """Parameterized shard-level routing across contiguous and overlap cases."""
    model = build_model(num_layers)
    nodes: list[Node] = []
    for node_id, start, end, tflops, x in segments:
        n = build_node(node_id, model, tflops=tflops, x=x, y=0.0)
        n.set_layer_allocation(start, end)
        nodes.append(n)

    router = DynamicProgrammingRouting()
    set_rtt_from_coords(nodes)
    node_ids, latency = router.find_optimal_path(nodes, num_layers)
    assert node_ids == expected_path
    # Sanity: latency equals sum of node latencies along path plus RTTs
    total = 0.0
    for i, nid in enumerate(node_ids):
        n = next(n for n in nodes if n.node_id == nid)
        total += float(n.layer_latency_ms)
        if i > 0:
            prev = next(n for n in nodes if n.node_id == node_ids[i - 1])
            total += float(prev.get_rtt_to(n))
    assert latency == pytest.approx(total, rel=1e-6)


@pytest.mark.parametrize(
    "num_layers,segments,expected_turns",
    [
        # tail has dramatically larger I/O and close-enough y so small RTT
        # 'head' should 'turn' to the tail ASAP (tail truncation)
        (10, [("head", 0, 6, 1.0, 0.0), ("tail", 4, 10, 3000.0, 0.01)], [("head", 4, "tail")]),
        # front truncation: path first uses node 'mid' at layer 3, so drop [0,3) on 'mid'
        (
            12,
            [("head", 0, 6, 500.0, 0.0), ("mid", 2, 8, 1.0, 0.5), ("end", 7, 12, 500.0, 0.6)],
            [("mid", 7, "tail"), ("mid", 6, "head")],
        ),
        (8, [("solo", 0, 8, 250.0, 0.0)], []),
    ],
)
def test_turning_points(
    num_layers: int,
    segments: list[tuple[str, int, int, float, float]],
    expected_turns: list[tuple[str, int, str]],
):
    """Parameterized turning-point detection across contiguous and overlap cases."""
    model = build_model(num_layers)
    nodes: list[Node] = []
    for node_id, start, end, io, x in segments:
        n = build_node(node_id, model, mem_bandwidth_gbps=io, x=x, y=0.0)
        n.set_layer_allocation(start, end)
        nodes.append(n)

    DynamicProgrammingRouting()
    set_rtt_from_coords(nodes)
    turns = find_turning_points(nodes, num_layers)
    # Order-insensitive comparison: we only require the same set of truncation points
    assert set(turns) == set(expected_turns)


def test_round_robin_pipelines_cycle_between_two_complete_paths():
    """Two registered fixed pipelines -> round-robin alternates between them deterministically."""
    num_layers = 12
    model = build_model(num_layers)
    # Pipelines: [a(0,6), b(6,12)] and [c(0,4), d(4,12)]
    a = build_node("a", model, tflops=200.0, x=0.0, y=0.0)
    b = build_node("b", model, tflops=200.0, x=1.0, y=0.0)
    c = build_node("c", model, tflops=220.0, x=0.0, y=1.0)
    d = build_node("d", model, tflops=220.0, x=1.0, y=1.0)
    a.set_layer_allocation(0, 6)
    b.set_layer_allocation(6, 12)
    c.set_layer_allocation(0, 4)
    d.set_layer_allocation(4, 12)
    nodes = [a, b, c, d]
    set_rtt_from_coords(nodes)

    node_manager = build_node_management(nodes)
    rr = RoundRobinOverFixedPipelinesRouting(node_manager)
    registered = rr.register_pipelines(nodes, num_layers)
    assert len(registered) == 2
    paths = []
    for _ in range(4):
        node_ids, latency = rr.find_optimal_path(nodes, num_layers)
        assert node_ids in (["a", "b"], ["c", "d"])
        # Latency equals sum of node latencies plus one RTT
        n0 = next(n for n in nodes if n.node_id == node_ids[0])
        n1 = next(n for n in nodes if n.node_id == node_ids[1])
        expected = (
            float(n0.layer_latency_ms) + float(n0.get_rtt_to(n1)) + float(n1.layer_latency_ms)
        )
        assert latency == pytest.approx(expected, rel=1e-6)
        paths.append(tuple(node_ids))

    # Expect strict alternation between the two pipelines
    assert paths[0] != paths[1]
    assert paths[0] == paths[2]
    assert paths[1] == paths[3]


def test_round_robin_skips_overloaded_pipeline():
    """If any node in a registered fixed pipeline is overloaded, that pipeline is skipped."""
    num_layers = 10
    model = build_model(num_layers)
    # Two pipelines: [p1a, p1b] and [p2a, p2b]
    p1a = build_node("p1a", model, tflops=200.0, x=0.0, y=0.0)
    p1b = build_node("p1b", model, tflops=200.0, x=1.0, y=0.0)
    p2a = build_node("p2a", model, tflops=200.0, x=0.0, y=1.0)
    p2b = build_node("p2b", model, tflops=200.0, x=1.0, y=1.0)
    p1a.set_layer_allocation(0, 4)
    p1b.set_layer_allocation(4, 10)
    p2a.set_layer_allocation(0, 3)
    p2b.set_layer_allocation(3, 10)
    nodes = [p1a, p1b, p2a, p2b]
    set_rtt_from_coords(nodes)

    # Overload p1b to invalidate pipeline 1
    p1b.current_requests = p1b.max_requests

    node_manager = build_node_management(nodes)
    rr = RoundRobinOverFixedPipelinesRouting(node_manager)
    # Multiple calls should always pick the viable pipeline [p2a, p2b]
    for _ in range(3):
        node_ids, latency = rr.find_optimal_path(nodes, num_layers)
        assert node_ids == ["p2a", "p2b"]
        n0, n1 = p2a, p2b
        expected = (
            float(n0.layer_latency_ms) + float(n0.get_rtt_to(n1)) + float(n1.layer_latency_ms)
        )
        assert latency == pytest.approx(expected, rel=1e-6)

    # Now overload p2a as well -> no viable pipelines
    p2a.current_requests = p2a.max_requests
    node_ids, latency = rr.find_optimal_path(nodes, num_layers)
    assert node_ids == []
    assert latency == float("inf")


def test_round_robin_pipeline_discovery_overlapping_heads_and_tails():
    """Ensure pipeline_discovery finds multiple pipelines with overlapping heads/tails.

    Scenario layers [0, 64): nodes with ranges
        [0, 22), [0, 23), [22, 43), [23, 42), [23, 47), [43, 64), [47, 64)
    Expected pipelines:
        - 0 -> 22 -> 43 -> 47 -> 64
        - 0 -> 23 -> 47 -> 64
    """
    num_layers = 64
    model = build_model(num_layers)

    def N(nid: str, s: int, e: int) -> Node:
        n = build_node(nid, model, tflops=200.0, x=0.0, y=0.0)
        n.set_layer_allocation(s, e)
        return n

    nodes = [
        N("h1", 0, 22),
        N("h2", 0, 23),
        N("m1", 22, 43),
        N("m2", 23, 42),
        N("m3", 23, 47),
        N("t1", 43, 64),
        N("t2", 47, 64),
    ]

    pipelines = RandomizedOverDynamicPipelinesRouting.pipeline_discovery(nodes, num_layers)

    # Pipelines should be sequences of node ids
    assert len(pipelines) >= 2
    # Convert to layer ranges to validate coverage and specific two expected pipelines
    id_to_node = {n.node_id: n for n in nodes}
    ranges = [
        [(id_to_node[nid].start_layer, id_to_node[nid].end_layer) for nid in p] for p in pipelines
    ]

    expected1 = [
        (0, 22),
        (22, 43),
        (43, 64),
    ]  # path via h1 -> m1 -> t1/t2 (choose minimal ends greedily)
    # Given the greedy rule, 43->47->64 is also a possible chain; we accept either tail
    expected1_alt = [(0, 22), (22, 43), (47, 64)]
    expected2 = [(0, 23), (23, 47), (47, 64)]

    def matches_path(path, expected):
        return len(path) == len(expected) and all(a == b for a, b in zip(path, expected))

    has_expected1 = any(
        matches_path(r, expected1) or matches_path(r, expected1_alt) for r in ranges
    )
    has_expected2 = any(matches_path(r, expected2) for r in ranges)
    assert has_expected1 and has_expected2


def test_round_robin_pipeline_diversity():
    """
    Scenario:
    Head nodes: H1, H2
    Mid nodes: M1 (fast), M2 (slightly slower)
    Tail nodes: T1
    Pipelines from H1: [H1, M1, T1] (cost 10), [H1, M2, T1] (cost 12)
    Pipelines from H2: [H2, M1, T1] (cost 10), [H2, M2, T1] (cost 12)
    """
    num_layers = 3
    model = build_model(num_layers)

    # Construct nodes with explicit latencies to control cost
    # Costs: H=1, M1=1, M2=3, T=1. RTT=0 for simplicity.
    # Path 1: 1+1+1 = 3
    # Path 2: 1+3+1 = 5

    h1 = build_node("h1", model)
    h1.set_layer_allocation(0, 1)
    h1.set_layer_latency_ms(1.2)
    h2 = build_node("h2", model)
    h2.set_layer_allocation(0, 1)
    h2.set_layer_latency_ms(1.0)

    m1 = build_node("m1", model)
    m1.set_layer_allocation(1, 2)
    m1.set_layer_latency_ms(1.0)
    m2 = build_node("m2", model)
    m2.set_layer_allocation(1, 2)
    m2.set_layer_latency_ms(3.0)

    t1 = build_node("t1", model)
    t1.set_layer_allocation(2, 3)
    t1.set_layer_latency_ms(1.0)

    nodes = [h1, h2, m1, m2, t1]

    # Mock RTT to be 0
    for n in nodes:
        n.rtt_to_nodes = {other.node_id: 0.0 for other in nodes}

    node_manager = build_node_management(nodes)
    rr = RoundRobinOverFixedPipelinesRouting(node_manager)
    pipelines = rr.register_pipelines(nodes, num_layers)

    assert len(pipelines) == 1
    assert pipelines[0] == ["h2", "m1", "t1"]

    t2 = build_node("t2", model)
    t2.set_layer_allocation(2, 3)
    t2.set_layer_latency_ms(1.0)
    nodes.append(t2)
    node_manager.upsert(t2)
    for n in nodes:
        n.rtt_to_nodes = {other.node_id: 0.0 for other in nodes}
    # TODO: apparently t2 shouldn't be paired with m2, but with current RR
    # we don't consider cross-chain sum of RTTs. Leaving this to DP case.
    t2.rtt_to_nodes[m2.node_id] = 4.0
    m2.rtt_to_nodes[t2.node_id] = 4.0

    rr.clear_registered_pipelines()
    h1.set_layer_allocation(0, 1)
    h2.set_layer_allocation(0, 1)
    m1.set_layer_allocation(1, 2)
    m2.set_layer_allocation(1, 2)
    t1.set_layer_allocation(2, 3)
    t2.set_layer_allocation(2, 3)

    pp = rr.register_pipelines(nodes, num_layers)
    assert len(pp) == 2
    assert pp[1] == ["h2", "m2", "t1"]


def test_rr_24_node_topology_utilization():
    """Verify that with the 24-node topology, we utilize almost all nodes.
    Topology:
    - 6x [0, 9)
    - 5x [9, 18)
    - 1x [9, 19)
    - 5x [18, 27)
    - 1x [19, 28)
    - 5x [27, 36)
    - 1x [28, 36)
    Total 24 nodes.
    We expect 6 pipelines.
    Ideally all 24 nodes should be used.
    """
    num_layers = 36
    model = build_model(num_layers)
    nodes = []

    def add_nodes(count, start, end):
        len(nodes)
        for i in range(count):
            node_id = f"n_{start}_{end}_{i}"
            n = build_node(node_id, model)
            n.set_layer_allocation(start, end)
            # Set uniform latency/RTT so cost doesn't skew selection away from diversity
            n.set_layer_latency_ms(10.0)
            nodes.append(n)

    # 6x [0, 9)
    add_nodes(6, 0, 9)
    # 5x [9, 18)
    add_nodes(5, 9, 18)
    # 1x [9, 19)
    add_nodes(1, 9, 19)
    # 5x [18, 27)
    add_nodes(5, 18, 27)
    # 1x [19, 28)
    add_nodes(1, 19, 28)
    # 5x [27, 36)
    add_nodes(5, 27, 36)
    # 1x [28, 36)
    add_nodes(1, 28, 36)

    # Mock RTT to be 0
    for n in nodes:
        n.rtt_to_nodes = {other.node_id: 0.0 for other in nodes}

    randomized = RandomizedOverDynamicPipelinesRouting()
    pipelines = randomized.pipeline_discovery(nodes, num_layers)

    assert len(pipelines) == 756

    node_manager = build_node_management(nodes)
    rr = RoundRobinOverFixedPipelinesRouting(node_manager)
    pipelines = rr.register_pipelines(nodes, num_layers)
    assert len(pipelines) == 6

    unique_nodes_used = set()
    for p in pipelines.values():
        for nid in p:
            unique_nodes_used.add(nid)

    print(f"Used {len(unique_nodes_used)} unique nodes out of {len(nodes)}")
    # We expect high utilization.
    # 6 pipelines * 4 stages = 24 slots.
    # We have 24 nodes available.
    # Ideally 24. Allow slight slack if topology logic is tricky, but should be > 20.
    assert len(unique_nodes_used) >= 24


def test_rr_select_best_pipelines_no_node_overlap_establishes_three_pipelines():
    """40-layer model, 6 identical nodes each holding 20 layers => 3 disjoint pipelines.

    Topology:
      - 3x heads covering [0, 20)
      - 3x tails covering [20, 40)
    Total candidate pipelines = 3*3 = 9, but we must register only node-disjoint ones => 3.
    """
    num_layers = 40
    model = build_model(num_layers)

    nodes: list[Node] = []
    # 3 heads [0, 20)
    for i in range(3):
        n = build_node(f"h{i}", model)
        n.set_layer_allocation(0, 20)
        n.set_layer_latency_ms(1.0)
        nodes.append(n)
    # 3 tails [20, 40)
    for i in range(3):
        n = build_node(f"t{i}", model)
        n.set_layer_allocation(20, 40)
        n.set_layer_latency_ms(1.0)
        nodes.append(n)

    # Mock RTT to be 0 so cost is purely node latency.
    for n in nodes:
        n.rtt_to_nodes = {other.node_id: 0.0 for other in nodes}

    node_manager = build_node_management(nodes)
    rr = RoundRobinOverFixedPipelinesRouting(node_manager)
    registered = rr.register_pipelines(nodes, num_layers)

    assert len(registered) == 3

    # Must use all 6 nodes exactly once across pipelines (no overlap).
    flat = [nid for p in registered.values() for nid in p]
    assert len(flat) == 6
    assert len(set(flat)) == 6

    id_to_node = {n.node_id: n for n in nodes}
    for p in registered.values():
        assert len(p) == 2
        h, t = p
        assert (id_to_node[h].start_layer, id_to_node[h].end_layer) == (0, 20)
        assert (id_to_node[t].start_layer, id_to_node[t].end_layer) == (20, 40)
