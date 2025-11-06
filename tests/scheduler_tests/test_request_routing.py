"""
Unit tests for request routing strategies.

Covers:
- Shard-level DP path using `Node` APIs (latency + RTT)
- Turning point detection via layer-level DP
- Parametrized scenarios with different splits/overlaps
- Round-robin baseline pipeline routing and overload skipping
"""

import pytest

from scheduling.node import Node
from scheduling.request_routing import (
    DynamicProgrammingRouting,
    RoundRobinPipelineRouting,
)

from .test_utils import build_model_info as build_model
from .test_utils import build_node, set_rtt_from_coords


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

    # Manually remove the RTT info between n1 and n2
    if n2.node_id in n1.rtt_to_nodes:
        del n1.rtt_to_nodes[n2.node_id]

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

    router = DynamicProgrammingRouting()
    set_rtt_from_coords(nodes)
    turns = router.find_turning_points(nodes, num_layers)
    # Order-insensitive comparison: we only require the same set of truncation points
    assert set(turns) == set(expected_turns)


def test_round_robin_pipelines_cycle_between_two_complete_paths():
    """Two complete pipelines -> round-robin alternates between them deterministically."""
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

    rr = RoundRobinPipelineRouting()
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
    """If any node in a pipeline is overloaded, that entire pipeline is skipped."""
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

    rr = RoundRobinPipelineRouting()
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

    rr = RoundRobinPipelineRouting()
    pipelines = rr.pipeline_discovery(nodes, num_layers)

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
