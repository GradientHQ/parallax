"""
Node registry and lifecycle management.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Dict, List, Optional, Tuple

from parallax_utils.logging_config import get_logger
from scheduling.node import Node

logger = get_logger(__name__)


class NodeState(str, Enum):
    """Lifecycle state of a joined node."""

    ACTIVE = "active"
    STANDBY = "standby"


class NodeManager:
    """Thread-safe node membership + lifecycle management.

    Responsibilities:
    - store node membership by node_id
    - track lifecycle state (active vs standby)
    - provide thread-safe snapshots for routing/allocation decisions
    """

    def __init__(self, *, initial_nodes: Optional[List[Node]] = None) -> None:
        self._lock = threading.RLock()
        self._nodes: Dict[str, Node] = {}
        self._state: Dict[str, NodeState] = {}
        self._registered_pipelines: Dict[int, List[str]] = {}
        self._node_to_pipeline: Dict[str, int] = {}
        self.node_assigned_request_count: Dict[str, int] = {}

        if initial_nodes:
            for n in initial_nodes:
                self.upsert(n, state=NodeState.STANDBY)

    def upsert(self, node: Node, *, state: Optional[NodeState] = None) -> None:
        """Add or replace a node by node_id."""
        with self._lock:
            self._nodes[node.node_id] = node
            if state is None:
                self._state.setdefault(node.node_id, NodeState.STANDBY)
            else:
                self._state[node.node_id] = state

    def _standby_locked(
        self,
        node_ids: List[str],
        *,
        allow_missing: bool = False,
    ) -> List[Node]:
        """Transition nodes to STANDBY under the registry lock.

        Returns the corresponding Node objects so callers can do any potentially
        slower per-node work (e.g. clearing allocations) outside the lock.
        """
        nodes_to_clear: List[Node] = []
        for nid in node_ids:
            node = self._nodes.get(nid)
            if node is None:
                if allow_missing:
                    continue
                raise ValueError(f"Node {nid} not found in registry")

            prev_state = self._state.get(nid)
            if prev_state is not None and prev_state != NodeState.ACTIVE:
                raise ValueError(f"Node {nid} is not ACTIVE, current state: {prev_state}")

            self._state[nid] = NodeState.STANDBY
            nodes_to_clear.append(node)
            self.node_assigned_request_count.pop(nid, None)
        return nodes_to_clear

    def remove(self, node_id: str) -> Optional[Node]:
        """Remove a node; returns removed node if present."""
        nodes_to_clear: List[Node] = []
        with self._lock:
            self._state.pop(node_id, None)
            removed = self._nodes.pop(node_id, None)
            self.node_assigned_request_count.pop(node_id, None)
            pipeline_id = self._node_to_pipeline.pop(node_id, None)
            if pipeline_id is not None:
                pipeline_nodes = self._registered_pipelines.pop(pipeline_id, [])
                logger.warning(
                    "Node %s left; removing pipeline_id=%s from registered pipelines and detaching %d member(s): %s",
                    node_id,
                    pipeline_id,
                    len(pipeline_nodes),
                    pipeline_nodes,
                )

                # This pipeline is no longer valid if any member leaves; detach all members.
                for nid in pipeline_nodes:
                    self._node_to_pipeline.pop(nid, None)

                # Transition remaining members to STANDBY while holding the lock,
                # but clear per-node allocations outside the lock.
                remaining = [nid for nid in pipeline_nodes if nid != node_id]
                nodes_to_clear = self._standby_locked(remaining, allow_missing=True)

        for node in nodes_to_clear:
            node.clear_layer_allocation()

        # In case it got rejoined in the system without clearing status
        removed.clear_layer_allocation()
        return removed

    def get(self, node_id: str) -> Optional[Node]:
        with self._lock:
            return self._nodes.get(node_id)

    def state_of(self, node_id: str) -> Optional[NodeState]:
        with self._lock:
            return self._state.get(node_id)

    def activate(self, node_ids: List[str]) -> None:
        """Mark nodes as ACTIVE (actively serving as part of a pipeline)."""
        with self._lock:
            for nid in node_ids:
                if nid not in self._nodes:
                    raise ValueError(f"Node {nid} not found in registry")
                if self._state.get(nid) != NodeState.STANDBY:
                    raise ValueError(f"Node {nid} is not STANDBY")
                self._state[nid] = NodeState.ACTIVE

    def standby(self, node_ids: List[str]) -> None:
        """Mark nodes as STANDBY (joined but not actively serving)."""
        nodes_to_clear: List[Node] = []
        with self._lock:
            nodes_to_clear = self._standby_locked(node_ids)

        # Do per-node work outside the registry lock to reduce contention.
        for node in nodes_to_clear:
            node.clear_serving_state()

    def ids_to_nodes(self, node_ids: List[str]) -> List[Node]:
        """Return a copy of nodes, optionally filtered by node_ids."""
        with self._lock:
            return [self._nodes[nid] for nid in node_ids]

    @property
    def nodes(self) -> List[Node]:
        with self._lock:
            return list(self._nodes.values())

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_active_nodes(self) -> int:
        return len(self.active_nodes)

    @property
    def num_standby_nodes(self) -> int:
        return len(self.standby_nodes)

    @property
    def active_nodes(self) -> List[Node]:
        with self._lock:
            return [
                n for n in self._nodes.values() if self._state.get(n.node_id) == NodeState.ACTIVE
            ]

    @property
    def standby_nodes(self) -> List[Node]:
        with self._lock:
            return [
                n for n in self._nodes.values() if self._state.get(n.node_id) == NodeState.STANDBY
            ]

    def list_node_allocations(self, total_layers: int) -> List[Tuple[str, int, int]]:
        """Snapshot ACTIVE segments as (node_id, start, end) under the registry lock."""
        if total_layers <= 0:
            return []
        segments: List[Tuple[str, int, int]] = []
        with self._lock:
            for nid, node in self._nodes.items():
                if self._state.get(nid) != NodeState.ACTIVE:
                    continue
                s, e = node.start_layer, node.end_layer
                if s is None or e is None:
                    continue
                if s < 0 or e <= s or e > total_layers:
                    raise ValueError(f"Invalid layer range: {s}, {e} for node {nid}")
                segments.append((nid, int(s), int(e)))
        return segments

    def num_full_pipelines(self, total_layers: int) -> int:
        """Count how many complete pipelines exist among ACTIVE nodes.

        A "pipeline" is a sequence of ACTIVE nodes whose allocated layer ranges form
        a contiguous cover from 0 up to `total_layers` (exclusive), i.e.:
            [0, a) -> [a, b) -> ... -> [y, total_layers)

        Counting is done as number of distinct node-sequences (paths) in the DAG
        induced by edges (start_layer -> end_layer) for each ACTIVE node allocation.

        e.g. we have 4 layers and 3 nodes:
        - A: [0, 4) direct; B: [0, 2) C1: [2, 4) C2: [2, 4)
        - we have 2 pipelines:- [A, B, C1], [A, B, C2]
        """

        if total_layers <= 0:
            return 0

        segments = self.list_node_allocations(total_layers)
        if not segments:
            return 0

        # DP over layer boundaries: ways[pos] = number of ways to reach boundary `pos`.
        # Initialize at boundary 0.
        ways: Dict[int, int] = {0: 1}
        # Sort ensures deterministic behavior and allows single-pass forward DP.
        ranges: List[Tuple[int, int]] = [(s, e) for _, s, e in segments]
        ranges.sort(key=lambda p: (p[0], p[1]))

        for s, e in ranges:
            w = ways.get(s, 0)
            if w:
                ways[e] = ways.get(e, 0) + w

        return int(ways.get(total_layers, 0))

    def has_full_pipeline(self, num_total_layers: int) -> bool:
        """Check if there is a full pipeline among ACTIVE nodes."""
        return self.num_full_pipelines(num_total_layers) > 0

    def add_request(self, node_id: str) -> None:
        """Add a request to a node."""
        with self._lock:
            if self._nodes.get(node_id) is None:
                raise ValueError(f"Node {node_id} not found in registry")
            if self._state.get(node_id) != NodeState.ACTIVE:
                raise ValueError(f"Node {node_id} is not ACTIVE")
            self.node_assigned_request_count[node_id] = (
                self.node_assigned_request_count.get(node_id, 0) + 1
            )
            self._nodes[node_id].add_request()

    def register_pipelines(self, pipelines: List[List[str]]) -> Dict[int, List[str]]:
        """Fixed-pipeline registry (for round-robin routing)

        Args:
            pipelines: A list of lists of node ids, each representing a pipeline.

        Returns:
            A dictionary of pipeline ids to lists of node ids.
        """
        with self._lock:
            self._registered_pipelines = {}
            self._node_to_pipeline = {}

            for pid, p in enumerate(pipelines):
                self._registered_pipelines[pid] = list(p)
                for nid in p:
                    # Strict enforcement: a node must belong to exactly one pipeline.
                    if nid in self._node_to_pipeline:
                        raise ValueError(
                            f"Node {nid} is already registered to pipeline {self._node_to_pipeline[nid]}"
                        )
                    if nid not in self._nodes:
                        logger.warning(f"Node {nid} not found in registry")
                    if self._state.get(nid) != NodeState.ACTIVE:
                        logger.warning(f"Node {nid} is not ACTIVE.")
                        self._state[nid] = NodeState.ACTIVE
                    self._node_to_pipeline[nid] = pid

            return {pid: list(p) for pid, p in self._registered_pipelines.items()}

    def extend_registered_pipelines(self, pipelines: List[List[str]]) -> Dict[int, List[str]]:
        """Append additional pipelines to the registry (thread-safe).

        This is used when we allocate *new* full pipelines from STANDBY nodes and want
        to add them without re-registering everything.
        """
        with self._lock:
            next_pid = (
                (max(self._registered_pipelines.keys()) + 1) if self._registered_pipelines else 0
            )
            for p in pipelines:
                if not p:
                    continue
                pid = next_pid
                next_pid += 1
                self._registered_pipelines[pid] = list(p)
                for nid in p:
                    if nid in self._node_to_pipeline:
                        raise ValueError(
                            f"Node {nid} is already registered to pipeline {self._node_to_pipeline[nid]}"
                        )
                    self._node_to_pipeline[nid] = pid
            return {pid: list(p) for pid, p in self._registered_pipelines.items()}

    def clear_registered_pipelines(self) -> None:
        """Clear any fixed pipeline registrations and detach member nodes."""
        self.standby(list(self._node_to_pipeline.keys()))
        with self._lock:
            self._registered_pipelines = {}
            self._node_to_pipeline = {}

    def get_registered_pipelines(self) -> Dict[int, List[str]]:
        """Return a copy of the currently registered fixed pipelines."""
        with self._lock:
            return {pid: list(p) for pid, p in self._registered_pipelines.items()}

    def pipeline_id_of_node(self, node_id: str) -> Optional[int]:
        """Return the pipeline id a node is registered to (if any)."""
        with self._lock:
            return self._node_to_pipeline.get(node_id)

    def report_pipeline_capacity(
        self,
    ) -> Tuple[Optional[Dict[int, Tuple[int, int]]], int, int]:
        """Return per-pipeline bottleneck load + total request capacity across pipelines.

        Definitions:
        - Per-node remaining request capacity = max(0, node.max_requests - node.current_requests)
        - Per-pipeline capacity = min(remaining capacity of each worker in that pipeline)
        - Total capacity = sum(per-pipeline capacity) across all registered pipelines

        Returns:
            per_pipeline_min: Dict of pipeline id -> (min_node_capacity, min_remaining_capacity).
            total_capacity: The total capacity of all registered pipelines.
            cur_capacity: The current capacity (counting existing request load) of all registered pipelines.
        """
        with self._lock:
            if not self._registered_pipelines:
                return None, 0, 0

            per_pipeline_min: Dict[int, Tuple[int, int]] = {}
            cur_capacity = 0
            total_capacity = 0

            # Iterate deterministically for stable display/tests.
            for pid in sorted(self._registered_pipelines.keys()):
                node_ids = self._registered_pipelines.get(pid, [])
                if not node_ids:
                    raise ValueError(f"Pipeline {pid} is empty")

                min_cur_capacity = None
                min_node_capacity = None
                for nid in node_ids:
                    node = self._nodes.get(nid)
                    if node is None:
                        raise ValueError(f"Node {nid} not found in registry, but in pipeline {pid}")

                    node_capacity = node.max_requests
                    node_cur_capacity = max(0, node_capacity - node.current_requests)
                    min_cur_capacity = (
                        node_cur_capacity
                        if min_cur_capacity is None
                        else min(min_cur_capacity, node_cur_capacity)
                    )
                    min_node_capacity = (
                        node_capacity
                        if min_node_capacity is None
                        else min(min_node_capacity, node_capacity)
                    )

                per_pipeline_min[pid] = (int(min_node_capacity), int(min_cur_capacity))
                total_capacity += int(min_node_capacity)
                cur_capacity += int(min_cur_capacity)

            return per_pipeline_min, int(total_capacity), int(cur_capacity)
