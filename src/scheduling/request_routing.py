"""
Phase 2 - Request routing.

This module provides several routing strategies that turn the current layer
allocation into an end-to-end execution path for a single request.

Fixed (static) vs dynamic routing:
- **Dynamic** routing recomputes a route from the *current* cluster snapshot
  (allocations, RTT, and per-node load). It naturally handles node join/leave and
  shifting workload, because every decision is based on live state.
- **Fixed (static)** routing first *registers* a small set of pipelines at
  bootstrap time, then only routes over that fixed set. Node join/leave is
  handled by re-registering some of the pipelines;

e.g. 24 Nodes, GPT-OSS 120B, 36 layers, each node roughly handles 9 layers
    Layer Partitions:
        6 x [0-9],
        5 x [9 - 18] + 1 x [9 - 19],
        5 x [18 - 27] + 1 x [19 - 28],
        5 x [27 - 36] + 1 x [28 - 36]
    In total we have
        Path A: [0-9] - [9-18] - [18-27] - [27-36]
        # 6 x 5 x 5 x 5 = 750
        Path B: [0-9] - [9-19] - [19-28] - [28-36]
        # 6 x 1 x 1 x 1 = 6
    Total 756 possible pipelines, which can be served dynamically (dp, and randomized);
    Or 6 fixed pipelines, with 5 Path A + 1 Path B (used in round-robin).

Provided strategies:
- **`DynamicProgrammingRouting`** (**dynamic**): computes a minimum-latency route
  via DP over the currently allocated node ranges. It does *not* use fixed
  pipelines.
- **`RandomizedOverDynamicPipelinesRouting`** (**fixed**): enumerates *all*
  complete pipelines implied by current allocations, so a single node may appear
  in many candidate pipelines. Each request randomly selects a currently-viable
  pipeline, which tends to balance load when the number of pipelines is large.
- **`RoundRobinOverFixedPipelinesRouting`** (**fixed/static**): at bootstrap,
  uses **search → score → register** to keep a small set of high-quality pipelines,
  where one node is registered to at most one pipeline, no overlap, and no gap.
  During serving, round-robins only across registered pipelines,keeping behavior stable and predictable.

Routing is at node granularity: once a request enters a node, it runs all layers
hosted by that node. We can optionally compute layer-level turning points for a
warm-up phase (to inform rebalancing), then perform shard-level DP to produce the
final node path and total latency.
"""

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from parallax_utils.logging_config import get_logger
from scheduling.layer_allocation import BaseLayerAllocator
from scheduling.node import Node
from scheduling.node_management import NodeManager

logger = get_logger(__name__)


def estimate_pipeline_latency(
    pipeline_node_ids: List[str], *, id_to_node: Dict[str, Node]
) -> float:
    """Estimate end-to-end latency for a node-id pipeline.

    Returns `inf` if any node is missing, overloaded, or if any required RTT is missing.
    """
    total = 0.0
    prev: Optional[Node] = None
    for nid in pipeline_node_ids:
        n = id_to_node.get(nid)
        if n is None or n.is_overloaded:
            return float("inf")
        node_lat = float(n.layer_latency_ms)
        if node_lat == float("inf"):
            return float("inf")
        total += node_lat
        if prev is not None:
            hop = 0.0 if prev.node_id == n.node_id else float(prev.get_rtt_to(n))
            if hop == float("inf"):
                return float("inf")
            total += hop
        prev = n
    return total


def find_turning_points(nodes: List[Node], num_layers: int) -> List[Tuple[str, int, str]]:
    """Find truncation points (warm-up helper).

    Turning points mark where shards can be trimmed based on optimal routing.
    This is implemented using layer-level DP, with state (layer l, node i that hosts l).
    Node cost uses the node's per-layer latency proxy; edge cost uses RTT between nodes.

    Returns:
        A list of (node_id, layer_index, kind), where kind in {"head", "tail"}:
            - (node, l, "tail"): the route switches away at layer l although node still
            hosts l, so drop [l, end) on that node.
            - (node, l, "head"): the route first uses this node at layer l (> start),
            so drop [start, l) on that node.
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


class RequestRoutingStrategy(ABC):
    """Interface for request routing strategies."""

    def __init__(self, node_manager: NodeManager, total_layers: int = 0) -> None:
        self.node_manager: NodeManager = node_manager
        self.total_layers: int = total_layers

    @abstractmethod
    def find_optimal_path(
        self,
        last_refit_time: Optional[float] = None,
    ) -> Tuple[List[str], float]:
        """Return the chosen node-id path and its estimated latency.

        Args:
            last_refit_time: Last refit time for weight refit

        Returns:
            (node_ids, latency_ms). If no valid route exists, returns ([], inf).
        """
        raise NotImplementedError

    def bootstrap(self) -> None:
        """Optional bootstrap for best-effort initialization."""
        return None

    def routing_ready(self) -> bool:
        """Return True iff the router can dispatch requests right now."""
        return True

    def expand_pipelines(self) -> None:
        """Opportunistically expand pipelines for fixed-pipeline routers.

        - DP: no-op
        - RandomizedOverDynamicPipelinesRouting: raises NotImplementedError (not a fixed registry)
        - RR: allocate from STANDBY + discover + extend registered fixed pipelines
        """
        return None

    def scheduler_format_snapshot(self) -> str:
        """Return a human-readable snapshot string for logs/observability."""
        assignments = self.node_manager.list_node_allocations(self.total_layers)
        header = f"Current allocations ({len(assignments)} nodes)"
        sep = "-" * len(header)
        lines: List[str] = [header, sep]
        for node_id, start_layer, end_layer in assignments:
            node = self.node_manager.get(node_id)
            if node is None:
                raise ValueError(f"Node {node_id} not found in node manager")
            capacity = node.max_requests
            current = node.current_requests
            latency = node.layer_latency_ms
            latency_str = "inf" if latency == float("inf") else f"{latency:.2f}"
            n_hosted_requests = int(self.node_manager.node_assigned_request_count.get(node_id, 0))
            lines.append(
                "  %-16s layers [%3d, %3d) | load %3d/%-3d | latency %7s ms | assigned request count %3d | active %s"
                % (
                    node_id,
                    start_layer,
                    end_layer,
                    current,
                    capacity,
                    latency_str,
                    n_hosted_requests,
                    node.is_active,
                )
            )
        if len(lines) == 2:
            lines.append("  (none)")
        return "\n".join(lines)


class DynamicProgrammingRouting(RequestRoutingStrategy):
    """
    Dynamic-programming router.

    This is a **dynamic** strategy: it recomputes a route from the current node
    snapshot, and it does not rely on any fixed pipeline set.

    - Warm-up (`find_turning_points`): run a layer-level DP to identify turning points (where the optimal
      path switches nodes even if the current node still hosts the next layer).
    - Routing (`find_optimal_path`): run a shard-level DP over node assignments (contiguous layer ranges),
      using per-node execution latency and RTT via `Node.get_rtt_to`, to obtain a
      minimum-latency node sequence and total latency.
    """

    def find_optimal_path(
        self,
        last_refit_time: Optional[float] = None,
    ) -> Tuple[List[str], float]:
        """Compute a minimum-latency node-id path using shard-level DP.

        The DP treats each node's allocated range `[start_layer, end_layer)` as a
        vertex. A transition is allowed only when the previous range ends exactly
        where the next starts (contiguous cover). Vertex cost is `layer_latency_ms`;
        edge cost is RTT via `get_rtt_to`.

        Returns ([], inf) if no full cover `[0, num_layers)` exists or if any
        required RTT is missing.
        """
        nodes = self.node_manager.active_nodes
        num_layers = self.total_layers

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

    def scheduler_format_snapshot(self) -> str:
        assignments = self.node_manager.list_node_allocations(self.total_layers)
        header = f"Current allocations ({len(assignments)} nodes)"
        sep = "-" * len(header)
        lines: List[str] = [header, sep]
        for node_id, start_layer, end_layer in assignments:
            node = self.node_manager.get(node_id)
            if node is None:
                raise ValueError(f"Node {node_id} not found in node manager")
            # Snapshot values to avoid recomputing/logging side-effects twice
            capacity = node.max_requests
            current = node.current_requests
            latency = node.layer_latency_ms
            latency_str = "inf" if latency == float("inf") else f"{latency:.2f}"
            n_hosted_requests = 0
            if node_id in self.node_manager.node_assigned_request_count:
                n_hosted_requests = self.node_manager.node_assigned_request_count[node_id]
            lines.append(
                "  %-16s layers [%3d, %3d) | load %3d/%-3d | latency %7s ms | assigned request count %3d | active %s"
                % (
                    node_id,
                    start_layer,
                    end_layer,
                    current,
                    capacity,
                    latency_str,
                    n_hosted_requests,
                    node.is_active,
                )
            )
        if len(lines) == 2:
            lines.append("  (none)")
        return "\n".join(lines)

    def routing_ready(self) -> bool:
        """Return True iff DP routing can find a finite-latency path right now."""
        node_ids, lat = self.find_optimal_path()
        return bool(node_ids) and lat != float("inf")

    def expand_pipelines(self) -> None:
        return None


class RandomizedOverDynamicPipelinesRouting(RequestRoutingStrategy):
    """
    Routing strategy using random selection over all complete pipelines.
    This strategy will randomly select a pipeline from the list of all possible pipelines.

    If you want to route over a small, stable set of static pipelines (e.g., 6),
    use `RoundRobinOverFixedPipelinesRouting` instead.
    """

    def __init__(self, node_manager: NodeManager, total_layers: int = 0) -> None:
        super().__init__(node_manager=node_manager, total_layers=total_layers)
        self._pipelines: Optional[List[List[str]]] = None
        self._rng = random.Random()

    @staticmethod
    def pipeline_discovery(nodes: List[Node], num_layers: int) -> List[List[str]]:
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
            A list of pipelines as node-id sequences. Does not cache.
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

        def dfs(current_end: Optional[int], path_ids: List[str]) -> None:
            if current_end is None:
                return
            if current_end == num_layers:
                pipelines.append(list(path_ids))
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

        logger.debug(f"Discovered {len(pipelines)} pipelines")
        logger.debug(f"Pipelines: {pipelines}")
        return pipelines

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

    def find_optimal_path(
        self,
        nodes: Optional[List[Node]] = None,
        num_layers: Optional[int] = None,
        last_refit_time: Optional[float] = None,
    ) -> Tuple[List[str], float]:
        """Randomly choose among cached complete pipelines, skipping overloaded ones.

        Selection procedure:
        - On first use, discover and cache all complete pipelines.
        - Filter to those that are viable under current load and RTT availability.
        - Randomly choose one viable pipeline and return its latency estimate.
        """
        # Backwards compatible: allow callers to omit nodes/num_layers.
        nodes = self.node_manager.active_nodes if nodes is None else nodes
        num_layers = self.total_layers if num_layers is None else num_layers
        if not nodes or num_layers <= 0:
            return [], float("inf")

        self._ensure_pipelines(nodes, num_layers)
        if not self._pipelines:
            return [], float("inf")

        id_to_node: Dict[str, Node] = {n.node_id: n for n in nodes}
        viable: List[Tuple[List[str], float]] = []
        for p in self._pipelines:
            lat = estimate_pipeline_latency(p, id_to_node=id_to_node)
            if lat != float("inf"):
                viable.append((p, lat))

        if not viable:
            return [], float("inf")

        chosen, latency = self._rng.choice(viable)
        return list(chosen), float(latency)

    def expand_pipelines(self) -> None:
        raise NotImplementedError(
            "RandomizedOverDynamicPipelinesRouting does not support pipeline expansion."
        )


class RoundRobinOverFixedPipelinesRouting(RequestRoutingStrategy):
    """
    Fixed-pipeline routing using round-robin over *registered* pipelines.

    This strategy is meant for a "bootstrap then serve" mode:

    - **Search**: enumerate all complete pipelines implied by the current allocations.
    - **Score**: estimate end-to-end latency for each pipeline (node latency + RTT).
    - **Register**: pick a small set of *high-quality* pipelines, preferably node-disjoint,
      and keep them fixed until the next (re)bootstrap.

    During serving, requests are dispatched in round-robin over the registered set,
    skipping any pipeline that is currently invalid (overloaded node or missing RTT).

    Back to our 24 nodes, 36 layers examples, we only have 6 pipelines:
    5 x path A, and 1 x path B.
    """

    def __init__(
        self,
        node_manager: NodeManager,
        total_layers: int,
        layer_allocator: Optional[BaseLayerAllocator] = None,
    ) -> None:
        super().__init__(node_manager=node_manager, total_layers=total_layers)
        self._rr_cursor: int = 0
        self.layer_allocator = layer_allocator

    def _select_best_pipelines(
        self, all_pipelines: List[List[str]], nodes: List[Node]
    ) -> List[List[str]]:
        """Helper: Select best node-disjoint pipelines minimizing latency.

        Note: currently we don't consider cross-chain sum of RTTs.
        Leaving this to the DP case.
        """
        if not all_pipelines:
            return []

        id_to_node = {n.node_id: n for n in nodes}

        # Group by head
        by_head: Dict[str, List[Tuple[List[str], float]]] = {}
        for p in all_pipelines:
            if not p:
                continue
            # Strict: no node can appear twice within a single pipeline.
            if len(set(p)) != len(p):
                continue
            head = p[0]
            cost = estimate_pipeline_latency(p, id_to_node=id_to_node)

            if cost != float("inf"):
                by_head.setdefault(head, []).append((p, cost))

        # Sort heads by their best possible cost (fastest heads first)
        sorted_heads = sorted(by_head.keys(), key=lambda h: min(c for _, c in by_head[h]))

        selected = []
        used_nodes: set[str] = set()

        for head in sorted_heads:
            candidates = by_head[head]
            best_p = None
            best_cost = float("inf")

            for p, cost in candidates:
                # Strict: no node overlap across selected pipelines.
                if any(nid in used_nodes for nid in p):
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best_p = p

            if best_p:
                selected.append(best_p)
                for nid in best_p:
                    used_nodes.add(nid)

        # Hard safety: ensure node-disjointness (both within and across pipelines).
        flat = [nid for p in selected for nid in p]
        if len(flat) != len(set(flat)):
            raise ValueError(f"Selected pipelines have node overlap: {selected}")

        logger.debug(
            f"Pipeline selection: selected {len(selected)} pipelines (1 per head). "
            f"Node-disjoint: {len(flat) == len(set(flat))}"
        )
        return selected

    def bootstrap(self) -> Dict[int, List[str]]:
        """Search → score → register a fixed set of pipelines.

        Args:
            nodes: Current nodes (with allocations and RTTs available).
            num_layers: Total decoder layers in the model.
            max_pipelines: Optional cap on the number of pipelines to register.
            require_node_disjoint: If True, register pipelines greedily such that
                no node id appears in more than one registered pipeline.
            skip_overloaded_at_register: If True, exclude pipelines that are
                already invalid due to overloaded nodes at registration time.
        """
        existing = self.node_manager.get_registered_pipeline_node_ids()
        if existing:
            logger.warning("Pipelines already registered in node manager, re-registering")
            self.node_manager.clear_registered_pipelines()
            self._rr_cursor = 0

        nodes = self.node_manager.active_nodes
        num_layers = self.total_layers
        if not nodes or num_layers <= 0:
            return {}

        # Search: enumerate all complete pipelines.
        all_pipelines = RandomizedOverDynamicPipelinesRouting.pipeline_discovery(nodes, num_layers)
        if not all_pipelines:
            return {}
        # Score: based on estimated latency
        selected_pipelines = self._select_best_pipelines(all_pipelines, nodes)
        return self.node_manager.register_pipelines(selected_pipelines)

    def clear_registered_pipelines(self) -> None:
        """Clear currently registered fixed pipelines."""
        self.node_manager.clear_registered_pipelines()

    def get_registered_pipelines(self) -> Dict[int, List[str]]:
        """Return currently registered fixed pipelines (proxy to NodeManager).

        This is primarily used by the scheduler for logging/observability.
        """
        return self.node_manager.get_registered_pipeline_node_ids()

    def routing_ready(self) -> bool:
        """Return True if there's at least one `ready` pipleline among active nodes.

        so that RR routing can actually find a finite-latency path.
        """
        pipelines = self.node_manager.get_registered_pipelines()
        if not pipelines:
            return False
        return any(p.is_ready for p in pipelines.values())

    def expand_pipelines(self) -> None:
        """RR-only: allocate from STANDBY and extend registered pipelines best-effort."""
        if self.layer_allocator is None:
            raise NotImplementedError(
                "RoundRobinOverFixedPipelinesRouting.expand_pipelines requires a layer allocator"
            )
        standby_nodes = self.node_manager.standby_nodes
        if not standby_nodes:
            return

        before_active_ids = {n.node_id for n in self.node_manager.active_nodes}
        ok = self.layer_allocator.allocate_from_standby()
        if not ok:
            return

        after_active_ids = {n.node_id for n in self.node_manager.active_nodes}
        newly_active_ids = after_active_ids - before_active_ids
        if not newly_active_ids:
            return

        newly_active_nodes = self.node_manager.ids_to_nodes(list(newly_active_ids))
        new_pipelines = RandomizedOverDynamicPipelinesRouting.pipeline_discovery(
            newly_active_nodes, self.total_layers
        )
        if not new_pipelines:
            logger.warning("[RR] No new pipelines found for extended RR")
            return

        filtered: List[List[str]] = []
        for p in new_pipelines:
            if all(self.node_manager.pipeline_id_of_node(nid) is None for nid in p):
                filtered.append(p)
        if not filtered:
            return

        try:
            self.node_manager.extend_registered_pipelines(filtered)
            logger.info(
                "[RR] Added %d new pipeline(s) from %d newly-active node(s)",
                len(filtered),
                len(newly_active_nodes),
            )
        except Exception as exc:
            logger.warning(f"[RR] Failed to extend registered pipelines (best-effort): {exc}")

    def scheduler_format_snapshot(self) -> str:
        pipelines = self.node_manager.get_registered_pipelines()
        p_header = f"Registered pipelines ({len(pipelines)})"
        p_sep = "-" * len(p_header)
        lines: List[str] = [p_header, p_sep]

        per_pipeline_min, total_capacity, cur_capacity = self.node_manager.report_pipeline_capacity(
            ready=True
        )
        if per_pipeline_min is None:
            lines.append("Capacity: (no registered pipelines)")
        else:
            lines.append(
                f"Capacity: total={total_capacity} cur={cur_capacity} per_pipeline={per_pipeline_min}"
            )

        if not pipelines:
            lines.append("  (none)")
            return "\n".join(lines)

        for pid in sorted(pipelines.keys()):
            p = pipelines[pid]
            p.recompute_capacity()
            lines.append(
                "  pipeline %-3d | stages=%d | ready=%s | cap=%d cur=%d"
                % (pid, p.num_stages, p.is_ready, p.min_node_capacity, p.min_remaining_capacity)
            )
            for idx, n in enumerate(p.nodes):
                s = -1 if n.start_layer is None else int(n.start_layer)
                e = -1 if n.end_layer is None else int(n.end_layer)
                lat = n.layer_latency_ms
                lat_str = "inf" if lat == float("inf") else f"{lat:.2f}"
                lines.append(
                    "    [%02d] %-16s layers [%3d, %3d) | load %3d/%-3d | latency %7s ms | active %s"
                    % (
                        idx,
                        n.node_id,
                        s,
                        e,
                        n.current_requests,
                        n.max_requests,
                        lat_str,
                        n.is_active,
                    )
                )
        return "\n".join(lines)

    def find_optimal_path(
        self,
        nodes: Optional[List[Node]] = None,
        num_layers: Optional[int] = None,
        last_refit_time: Optional[float] = None,
    ) -> Tuple[List[str], float]:
        """Return the next viable *registered* pipeline in round-robin order.

        Returns ([], inf) if nothing is registered or if all registered pipelines
        are currently invalid due to overload/missing RTT/missing nodes.
        """
        pipelines = self.node_manager.get_registered_pipelines()
        if not pipelines:
            _ = self.bootstrap()
            pipelines = self.node_manager.get_registered_pipelines()

        # Build lookup from the latest node snapshot.
        id_to_node: Dict[str, Node] = {n.node_id: n for n in self.node_manager.nodes}

        attempts = 0
        pipelines_list = [pipelines[k] for k in sorted(pipelines.keys())]
        total_pipelines = len(pipelines_list)
        while attempts < total_pipelines:
            pid = self._rr_cursor % total_pipelines
            candidate_pipeline = pipelines_list[pid]
            candidate = list(candidate_pipeline.node_ids)
            self._rr_cursor += 1
            attempts += 1

            # If any stage is not ready, skip quickly.
            if not candidate_pipeline.is_ready:
                logger.warning(f"Pipeline {candidate} is not ready, skipping")
                continue

            latency = estimate_pipeline_latency(candidate, id_to_node=id_to_node)
            for nid in candidate:
                if nid not in id_to_node:
                    raise ValueError(
                        f"To be dispatched node {nid} in pipeline {candidate} not found in node manager!"
                    )
                if not id_to_node[nid].is_active:
                    # If node is not active, skip the pipeline
                    logger.warning(f"Pipeline {candidate} is not active, skipping")
                    latency = float("inf")
                if (
                    last_refit_time is not None
                    and id_to_node[nid].last_refit_time < last_refit_time
                ):
                    # If node holds an older version of weight, skip the pipeline
                    logger.warning(f"Pipeline {candidate} holds an old version of weight, skipping")
                    latency = float("inf")

            if latency != float("inf"):
                return list(candidate), float(latency)

        return [], float("inf")
