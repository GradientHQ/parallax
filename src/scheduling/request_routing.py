"""
Phase 2: Request routing.

Provides:
- A base strategy interface.
- A dynamic-programming router that minimizes end-to-end latency across nodes.
- A round-robin router that uses round-robin over complete pipelines.

Routing is at node granularity: once a request enters a node, it runs all layers
hosted by that node. We can optionally compute layer-level turning points for a
warm-up phase (to inform rebalancing), then perform shard-level DP to produce the
final node path and total latency.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from parallax_utils.logging_config import get_logger
from scheduling.node import Node

logger = get_logger(__name__)


class RequestRoutingStrategy(ABC):
    """Base abstract class for request routing strategies."""

    @abstractmethod
    def find_turning_points(self, nodes: List[Node], num_layers: int) -> List[Tuple[str, int, str]]:
        """Find truncation points.

        Turning points mark where shards can be trimmed based on optimal routing.
        Returns a list of (node_id, layer_index, kind), where kind in {"head", "tail"}:
        - (node, l, "tail"): the route switches away at layer l although node still
          hosts l, so drop [l, end) on that node.
        - (node, l, "head"): the route first uses this node at layer l (> start),
          so drop [start, l) on that node.
        """

    @abstractmethod
    def find_optimal_path(self, nodes: List[Node], num_layers: int) -> Tuple[List[str], float]:
        """Shard-level DP path across nodes. Returns (node_ids, latency)."""


class DynamicProgrammingRouting(RequestRoutingStrategy):
    """
    Dynamic-programming router.

    - Warm-up: run a layer-level DP to identify turning points (where the optimal
      path switches nodes even if the current node still hosts the next layer).
    - Routing: run a shard-level DP over node assignments (contiguous layer ranges),
      using per-node execution latency and RTT via `Node.get_rtt_to`, to obtain a
      minimum-latency node sequence and total latency.
    """

    def find_turning_points(self, nodes: List[Node], num_layers: int) -> List[Tuple[str, int, str]]:
        """Find shard truncation points via layer-level DP.

        DP state is (layer l, node i that hosts l). Node cost uses the node's
        per-layer latency proxy; edge cost uses RTT between nodes.
        """
        if num_layers <= 0 or not nodes:
            return []

        # Build host lists per layer using start/end layer ranges
        layer_hosts: List[List[int]] = []
        for l in range(num_layers):
            hosts = [i for i, n in enumerate(nodes) if n.hosts_layer(l)]
            layer_hosts.append(hosts)

        # If any layer lacks a host, return empty
        if any(len(h) == 0 for h in layer_hosts):
            return []

        # layer_id: node_id -> cost
        dp: List[Dict[int, float]] = [{i: float("inf") for i in layer_hosts[0]}]
        back: List[Dict[int, Optional[int]]] = [{i: None for i in layer_hosts[0]}]

        # Init layer 0
        for i in layer_hosts[0]:
            dp[0][i] = nodes[i].layer_latency_ms

        # Recurrrence: dp[l+1][g] = min_g' (dp[l][g] + rtt(g,g') + latency(g'))
        for l in range(1, num_layers):
            curr: Dict[int, float] = {i: float("inf") for i in layer_hosts[l]}
            prev_back: Dict[int, Optional[int]] = {i: None for i in layer_hosts[l]}
            for i in layer_hosts[l]:
                node_i = nodes[i]
                best_cost = float("inf")
                best_j: Optional[int] = None
                for j, prev_cost in dp[l - 1].items():
                    if prev_cost == float("inf"):
                        continue
                    node_j = nodes[j]
                    trans = 0.0 if i == j else node_j.get_rtt_to(node_i)
                    total = prev_cost + trans + node_i.layer_latency_ms
                    if total < best_cost:
                        best_cost = total
                        best_j = j
                curr[i] = best_cost
                prev_back[i] = best_j
            dp.append(curr)
            back.append(prev_back)

        # Backtrack optimal node index per layer
        last = dp[-1]
        end_i = min(last, key=lambda k: last[k])
        path_idx: List[int] = [end_i]
        for l in range(num_layers - 1, 0, -1):
            prev_i = back[l][path_idx[-1]]
            if prev_i is None:
                break
            path_idx.append(prev_i)
        path_idx.reverse()

        # Identify turning points: tail truncations when switching away
        turning: List[Tuple[str, int, str]] = []
        for l in range(1, len(path_idx)):
            prev_i = path_idx[l - 1]
            cur_i = path_idx[l]
            if prev_i == cur_i:
                continue
            prev_node = nodes[prev_i]
            if prev_node.hosts_layer(l):
                turning.append((nodes[prev_i].node_id, l, "tail"))
        # Identify front truncations: for each node on the path, if the first
        # layer used is greater than its hosted start, we can drop the prefix
        # [start, first_used_layer)
        first_used: Dict[int, int] = {}
        for l, idx in enumerate(path_idx):
            if idx not in first_used:
                first_used[idx] = l
        for idx, l0 in first_used.items():
            n = nodes[idx]
            if n.start_layer is None:
                continue
            if l0 > n.start_layer:
                turning.append((n.node_id, l0, "head"))
        return turning

    def find_optimal_path(self, nodes: List[Node], num_layers: int) -> Tuple[List[str], float]:
        """Shard-level DP path across node ranges using `Node` APIs."""
        if num_layers <= 0 or not nodes:
            return [], 0.0

        # Collect vertices from nodes with valid layer ranges
        starts: Dict[int, List[int]] = {}
        ends: Dict[int, List[int]] = {}
        for idx, n in enumerate(nodes):
            if n.start_layer is None or n.end_layer is None:
                continue
            starts.setdefault(n.start_layer, []).append(idx)
            ends.setdefault(n.end_layer, []).append(idx)

        # DP over vertices sorted by (start, end)
        order = [
            i
            for i, n in sorted(
                [
                    (i, n)
                    for i, n in enumerate(nodes)
                    if n.start_layer is not None and n.end_layer is not None
                ],
                key=lambda p: (p[1].start_layer, p[1].end_layer),
            )
        ]

        dp: Dict[int, float] = {i: float("inf") for i in order}
        parent: Dict[int, Optional[int]] = {i: None for i in order}

        # Initialize with nodes starting at layer 0
        for i in starts.get(0, []):
            dp[i] = float(nodes[i].layer_latency_ms)
            parent[i] = None

        # Transitions: j -> i if end(j) == start(i)
        for i in order:
            if dp[i] == float("inf"):
                # Not reachable yet; still try to relax successors using INF + ... won't help
                pass
            n_i = nodes[i]
            if n_i.start_layer is None:
                continue
            for j in ends.get(n_i.start_layer, []):
                if dp[j] == float("inf"):
                    continue
                n_j = nodes[j]
                trans = 0.0 if n_j.node_id == n_i.node_id else float(n_j.get_rtt_to(n_i))
                cand = dp[j] + trans + float(n_i.layer_latency_ms)
                if cand < dp[i]:
                    dp[i] = cand
                    parent[i] = j

        # Pick best terminal node that ends at num_layers
        terminals = ends.get(num_layers, [])
        if not terminals:
            return [], float("inf")
        end_idx = min(terminals, key=lambda k: dp.get(k, float("inf")))
        if dp.get(end_idx, float("inf")) == float("inf"):
            return [], float("inf")

        # Reconstruct path
        path_indices: List[int] = []
        cur: Optional[int] = end_idx
        while cur is not None:
            path_indices.append(cur)
            cur = parent[cur]
        path_indices.reverse()
        return [nodes[i].node_id for i in path_indices], dp[end_idx]


class RoundRobinPipelineRouting(RequestRoutingStrategy):
    """
    Baseline routing strategy using round-robin over complete pipelines.

    A complete pipeline is a sequence of nodes with contiguous layer ranges that
    exactly covers [0, num_layers). We enumerate all such pipelines from current
    node allocations, skip any pipeline that contains an overloaded node, and
    dispatch requests by rotating among the remaining pipelines.

    This implementation discovers all complete pipelines once, caches them as
    node-id sequences sorted by their estimated end-to-end latency (including
    RTT), and then round-robins over that cached list. Use `reset_pipelines()`
    to force rediscovery if allocations change.
    """

    def __init__(self) -> None:
        self._rr_cursor: int = 0
        self._pipelines: Optional[List[List[str]]] = None

    def pipeline_discovery(self, nodes: List[Node], num_layers: int) -> List[List[str]]:
        """Discover and return all complete pipelines greedily.

        A naive greedy procedure:
        - Build a mapping from start_layer to nodes starting there.
        - For each head node with start_layer == 0, repeatedly choose the next
          node that starts at the current end and has the smallest end_layer
          (strictly greater than current), until we reach num_layers.
        - If the chain reaches exactly num_layers, record the node-id sequence
          as a pipeline.

        Returns a list of pipelines as node-id sequences. Does not cache.
        """
        if not nodes or num_layers <= 0:
            return []

        # Index nodes by start layer
        start_to_nodes: Dict[int, List[Node]] = {}
        for n in nodes:
            if n.start_layer is None or n.end_layer is None:
                continue
            start_to_nodes.setdefault(n.start_layer, []).append(n)

        heads = start_to_nodes.get(0, [])
        pipelines: List[List[str]] = []

        seen = set()
        for head in heads:
            path_ids: List[str] = [head.node_id]
            current_end = head.end_layer
            ok = True
            while current_end < num_layers:
                candidates = [
                    n
                    for n in start_to_nodes.get(current_end, [])
                    if n.end_layer and n.end_layer > current_end and n.node_id not in seen
                ]
                if not candidates:
                    ok = False
                    break
                # Greedy pick: choose the node that ends earliest among candidates
                nxt = min(candidates, key=lambda n: n.end_layer)  # type: ignore[arg-type]
                path_ids.append(nxt.node_id)
                current_end = int(nxt.end_layer)  # type: ignore[arg-type]
            if ok and current_end == num_layers:
                for nid in path_ids:
                    seen.add(nid)
                pipelines.append(path_ids)

        logger.info(f"Discovered {len(pipelines)} pipelines")
        logger.info(f"Pipelines: {pipelines}")
        return pipelines

    def find_turning_points(self, nodes: List[Node], num_layers: int) -> List[Tuple[str, int, str]]:
        """No warm-up/truncation in the baseline; return no turning points."""
        return []

    def _ensure_pipelines(self, nodes: List[Node], num_layers: int) -> None:
        """Ensure cached pipelines exist; discover and cache if missing."""
        if self._pipelines is None:
            self._pipelines = self.pipeline_discovery(nodes, num_layers)

    def find_optimal_path(self, nodes: List[Node], num_layers: int) -> Tuple[List[str], float]:
        """Round-robin among cached pipelines, skipping overloaded ones.

        Selection procedure:
        - On first use, greedily discover and cache all full pipelines.
        - Pick the pipeline at `rr_cursor % len(pipelines)`.
        - If any node in that pipeline is overloaded or missing, advance cursor
          and try the next one, up to the number of pipelines.
        - Return the first viable pipeline and its latency estimate using
          current per-node stats and RTTs. If none are viable, return empty.
        """
        if not nodes or num_layers <= 0:
            return [], float("inf")

        self._ensure_pipelines(nodes, num_layers)
        if not self._pipelines:
            return [], float("inf")

        id_to_node: Dict[str, Node] = {n.node_id: n for n in nodes}

        attempts = 0
        total_pipelines = len(self._pipelines)
        self._rr_cursor %= total_pipelines
        while attempts < total_pipelines:
            idx = self._rr_cursor % total_pipelines
            candidate_ids = self._pipelines[idx]
            # Check overloaded / presence
            viable = True
            prev: Optional[Node] = None
            total_latency = 0.0
            for nid in candidate_ids:
                node = id_to_node.get(nid)
                if node is None or node.is_overloaded:
                    viable = False
                    break
                total_latency += float(node.layer_latency_ms)
                if prev is not None:
                    total_latency += (
                        0.0 if prev.node_id == node.node_id else float(prev.get_rtt_to(node))
                    )
                prev = node
            self._rr_cursor += 1
            attempts += 1
            if viable:
                return candidate_ids, total_latency

        return [], float("inf")
