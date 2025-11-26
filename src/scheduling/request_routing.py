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

import random
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

    @staticmethod
    def find_turning_points(nodes: List[Node], num_layers: int) -> List[Tuple[str, int, str]]:
        """Find shard truncation points via layer-level DP.

        DP state is (layer l, node i that hosts l). Node cost uses the node's
        per-layer latency proxy; edge cost uses RTT between nodes.

        This is a static method that can be called directly without creating an instance:
        DynamicProgrammingRouting.find_turning_points(nodes, num_layers)

        It can also be called via an instance, which will work due to Python's method resolution.
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
            if n.start_layer is None or n.end_layer is None or n.is_active is False:
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
        self._pipelines: Optional[Dict[str, List[str]]] = None

    @property
    def pipelines(self) -> Optional[Dict[str, List[str]]]:
        """Return the cached pipelines map."""
        return self._pipelines

    def pipeline_discovery(self, nodes: List[Node], num_layers: int) -> Dict[str, List[str]]:
        """Discover and return all complete pipelines via DFS backtracking.

        Robust enumeration procedure:
        - Build a mapping from start_layer to nodes starting there.
        - For each head node with start_layer == 0, perform a depth-first search
          from its end_layer, trying all candidate next nodes whose start equals
          the current end and whose end strictly increases.
        - Record any chain that reaches exactly `num_layers` as a complete pipeline.

        This approach backtracks when a candidate cannot lead to completion,
        avoiding the brittleness of a single greedy choice and ensuring that
        overlapping heads/tails yield all valid pipelines.

        Returns:
            A dict of pipelines as {pipeline_id: node-id sequence}. Does not cache.
        """
        if not nodes or num_layers <= 0:
            return {}

        # Index nodes by start layer
        start_to_nodes: Dict[int, List[Node]] = {}
        for n in nodes:
            if n.start_layer is None or n.end_layer is None:
                continue
            start_to_nodes.setdefault(n.start_layer, []).append(n)

        heads = start_to_nodes.get(0, [])
        pipelines_list: List[List[str]] = []

        def dfs(current_end: Optional[int], path_ids: List[str]) -> None:
            if current_end is None:
                return
            if current_end == num_layers:
                pipelines_list.append(list(path_ids))
                return
            candidates = [
                n
                for n in start_to_nodes.get(int(current_end), [])
                if n.end_layer is not None and n.end_layer > current_end
            ]
            # Deterministic order: try shorter segments first
            candidates.sort(key=lambda n: n.end_layer)  # type: ignore[arg-type]
            for nxt in candidates:
                path_ids.append(nxt.node_id)
                dfs(int(nxt.end_layer), path_ids)  # type: ignore[arg-type]
                path_ids.pop()

        for head in heads:
            if head.end_layer is None:
                continue
            path_ids: List[str] = [head.node_id]
            dfs(int(head.end_layer), path_ids)  # type: ignore[arg-type]

        # Assign unique pipeline IDs to nodes in discovered pipelines
        # This optimization helps the scheduler identify and repair specific pipelines
        id_to_node = {n.node_id: n for n in nodes}
        pipelines_map = {}
        for i, pipeline in enumerate(pipelines_list):
            pid = f"pipe-{i}"
            pipelines_map[pid] = pipeline
            for node_id in pipeline:
                if node_id in id_to_node:
                    id_to_node[node_id].pipeline_id = pid

        logger.debug(f"Discovered {len(pipelines_map)} pipelines")
        logger.debug(f"Pipelines: {pipelines_map}")
        return pipelines_map

    def find_turning_points(self, nodes: List[Node], num_layers: int) -> List[Tuple[str, int, str]]:
        """No warm-up/truncation in the baseline; return no turning points."""
        return []

    def _ensure_pipelines(self, nodes: List[Node], num_layers: int) -> None:
        """Ensure cached pipelines exist; discover and cache if missing."""
        if self._pipelines is None:
            self._pipelines = self.pipeline_discovery(nodes, num_layers)

    def _build_start_index(self, nodes: List[Node]) -> Dict[int, List[Node]]:
        """Build an index of nodes by their `start_layer` for fast lookups.

        Only nodes with both `start_layer` and `end_layer` set are included.
        """
        index: Dict[int, List[Node]] = {}
        for n in nodes:
            if n.start_layer is None or n.end_layer is None:
                continue
            index.setdefault(n.start_layer, []).append(n)
        return index

    def _attempt_repair_pipeline(
        self, candidate_ids: List[str], nodes: List[Node], num_layers: int
    ) -> Optional[List[str]]:
        """Best-effort repair of an overloaded pipeline by backtracking from the tail.

        Starting from the end of the proposed pipeline, keep the longest viable
        prefix (no missing/overloaded nodes) and search for an alternative suffix
        that completes coverage to `num_layers`. The search explores all nodes that
        start at the split layer and are not overloaded, performing DFS until a
        complete chain is found or possibilities are exhausted.

        Returns:
            A repaired pipeline (list of node_ids) or None if not found.
        """
        id_to_node: Dict[str, Node] = {n.node_id: n for n in nodes}
        start_to_nodes = self._build_start_index(nodes)

        # Identify which positions in the original pipeline are viable
        def is_viable_node_id(nid: str) -> bool:
            node = id_to_node.get(nid)
            return node is not None and not node.is_overloaded

        # Try backtracking from the tail to earlier split points
        for split_idx in range(len(candidate_ids) - 1, -1, -1):
            # Check that prefix [0, split_idx) remains viable
            prefix_ok = True
            for i in range(split_idx):
                if not is_viable_node_id(candidate_ids[i]):
                    prefix_ok = False
                    break
            if not prefix_ok:
                continue

            # Determine split layer where we start reconstructing the suffix
            if split_idx == 0:
                split_layer = 0
            else:
                prev_node = id_to_node.get(candidate_ids[split_idx - 1])
                if prev_node is None or prev_node.end_layer is None:
                    continue
                split_layer = int(prev_node.end_layer)

            # Depth-first search to build a non-overloaded suffix covering [split_layer, L)
            repaired_suffix: Optional[List[str]] = None

            def dfs(layer: int, acc: List[str]) -> bool:
                nonlocal repaired_suffix
                if layer == num_layers:
                    repaired_suffix = list(acc)
                    return True
                candidates = [
                    n
                    for n in start_to_nodes.get(layer, [])
                    if n.end_layer is not None and n.end_layer > layer and not n.is_overloaded
                ]
                # Prefer shorter segments first for responsiveness
                candidates.sort(key=lambda n: n.end_layer)  # type: ignore[arg-type]
                for nxt in candidates:
                    acc.append(nxt.node_id)
                    if dfs(int(nxt.end_layer), acc):  # type: ignore[arg-type]
                        return True
                    acc.pop()
                return False

            if dfs(split_layer, []):
                new_pipeline = candidate_ids[:split_idx] + (repaired_suffix or [])
                # Sanity check: ensure coverage starts from 0 and ends at L
                # (prefix guarantees contiguous coverage up to split_layer)
                return new_pipeline if new_pipeline else None

        return None

    def mark_pipeline_broken(
        self, pipeline_id: str, node_id_to_node: Dict[str, Node]
    ) -> Optional[List[str]]:
        """Mark a pipeline as broken: remove from cache and mark nodes as left-over.

        Args:
            pipeline_id: The ID of the broken pipeline.
            node_id_to_node: Map to look up node objects to set left_over=True.

        Returns:
            The list of node IDs that were in the broken pipeline, or None if not found.
        """
        if self._pipelines is None or pipeline_id not in self._pipelines:
            logger.warning("Pipeline %s not found in cache", pipeline_id)
            return None

        logger.debug("Marking pipeline %s as broken and nodes as left-over", pipeline_id)
        pipeline_nodes = self._pipelines.pop(pipeline_id)
        for nid in pipeline_nodes:
            if nid in node_id_to_node:
                node_id_to_node[nid].left_over = True

        return pipeline_nodes

    def register_pipeline(self, pipeline_id: Optional[str], nodes: List[Node]) -> str:
        """Register a new pipeline with the router.

        1. If no pipeline_id provided, generates a new one.
        2. Marks all nodes as NOT left-over.
        3. Sorts nodes by hosting layer (start_layer).
        4. Updates node.pipeline_id for all nodes.
        5. Adds to cached pipelines map.

        Args:
            pipeline_id: Optional existing ID. If None, generates 'pipe-N'.
            nodes: List of nodes forming the complete pipeline.

        Returns:
            The pipeline_id used for registration.
        """
        if not self._pipelines:
            self._pipelines = {}

        # 1. Generate ID if needed
        if not pipeline_id:
            # Simple generation strategy: find max pipe-N and increment
            idx = 0
            while f"pipe-{idx}" in self._pipelines:
                idx += 1
            pipeline_id = f"pipe-{idx}"

        # 3. Sort nodes by layer order
        sorted_nodes = sorted(
            [n for n in nodes if n.start_layer is not None], key=lambda n: n.start_layer or 0
        )
        node_ids = [n.node_id for n in sorted_nodes]

        # 2. & 4. Mark left-over False and set pipeline_id
        for node in sorted_nodes:
            node.left_over = False
            node.pipeline_id = pipeline_id

        # 5. Add to cache
        self._pipelines[pipeline_id] = node_ids
        logger.debug("Registered pipeline %s: %s", pipeline_id, node_ids)

        return pipeline_id

    def find_optimal_path(self, nodes: List[Node], num_layers: int) -> Tuple[List[str], float]:
        """Round-robin among cached pipelines, skipping overloaded ones.

        Selection procedure:
        - On first use, greedily discover and cache all full pipelines.
        - Pick the pipeline at `rr_cursor % len(pipelines)`.
        - If any node in that pipeline is overloaded or missing, advance cursor
          and try the next one, up to the number of pipelines. Additionally, if
          the selected pipeline contains overloaded nodes, attempt a best-effort
          repair by backtracking from the tail to find an alternative suffix that
          completes coverage without overloaded nodes.
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
        pipelines_list = list(self._pipelines.values())
        total_pipelines = len(pipelines_list)
        self._rr_cursor %= total_pipelines
        while attempts < total_pipelines:
            idx = self._rr_cursor % total_pipelines
            candidate_ids = pipelines_list[idx]
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
            if viable and total_latency != float("inf"):
                return candidate_ids, total_latency
            # Attempt a one-shot repair if the selected pipeline is not viable
            repaired = self._attempt_repair_pipeline(candidate_ids, nodes, num_layers)
            if repaired:
                # Compute latency for the repaired path
                total_latency = 0.0
                prev = None
                for nid in repaired:
                    node = id_to_node.get(nid)
                    # If any node is missing/overloaded, skip this repair
                    if node is None or node.is_overloaded:
                        total_latency = float("inf")
                        break
                    total_latency += float(node.layer_latency_ms)
                    if prev is not None:
                        total_latency += (
                            0.0 if prev.node_id == node.node_id else float(prev.get_rtt_to(node))
                        )
                    prev = node
                if total_latency != float("inf"):
                    return repaired, total_latency

        return [], float("inf")


class RandomPipelineRouting(RoundRobinPipelineRouting):
    """
    Random routing strategy.

    Simply dispatches request randomly to any of the pipeline (as our experiment baseline).
    """

    def find_optimal_path(self, nodes: List[Node], num_layers: int) -> Tuple[List[str], float]:
        """Randomly select a pipeline."""
        if not nodes or num_layers <= 0:
            return [], float("inf")

        self._ensure_pipelines(nodes, num_layers)
        if not self._pipelines:
            return [], float("inf")

        candidate_ids = random.choice(list(self._pipelines.values()))

        # Calculate latency
        id_to_node: Dict[str, Node] = {n.node_id: n for n in nodes}
        total_latency = 0.0
        prev: Optional[Node] = None

        for nid in candidate_ids:
            node = id_to_node.get(nid)
            if node is None:
                return [], float("inf")

            total_latency += float(node.layer_latency_ms)
            if prev is not None:
                total_latency += (
                    0.0 if prev.node_id == node.node_id else float(prev.get_rtt_to(node))
                )
            prev = node

        return candidate_ids, total_latency
