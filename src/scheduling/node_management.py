"""
Node registry and lifecycle management.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from parallax_utils.logging_config import get_logger
from scheduling.node import Node

logger = get_logger(__name__)


class NodeState(str, Enum):
    """Lifecycle state of a joined node."""

    ACTIVE = "active"
    STANDBY = "standby"


@dataclass
class Pipeline:
    """A fixed pipeline definition for **RR (round-robin) routing only**.

    This is a *static/registered* pipeline: an ordered list of participating node ids
    representing the stages of a full end-to-end execution path.
    """

    nodes: List[Node]

    min_node_capacity: int = field(init=False, default=0)
    min_remaining_capacity: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("Pipeline is empty")

        self.nodes.sort(
            key=lambda n: (
                float("inf") if n.start_layer is None else int(n.start_layer),
                float("inf") if n.end_layer is None else int(n.end_layer),
            )
        )

        # No gap, no overlap.
        model_num_layers = int(self.nodes[0].model_info.num_layers)
        for n in self.nodes:
            if int(n.model_info.num_layers) != model_num_layers:
                raise ValueError(
                    f"Pipeline nodes disagree on model num_layers: expected {model_num_layers}, got {n.model_info.num_layers} on node {n.node_id}"
                )
            if n.start_layer is None or n.end_layer is None:
                raise ValueError(
                    f"Pipeline node {n.node_id} missing layer allocation: start_layer={n.start_layer}, end_layer={n.end_layer}"
                )
            s, e = int(n.start_layer), int(n.end_layer)
            if s < 0 or e <= s or e > model_num_layers:
                raise ValueError(
                    f"Pipeline node {n.node_id} has invalid layer range: [{s}, {e}) for total_layers={model_num_layers}"
                )

        first = self.nodes[0]
        if int(first.start_layer) != 0:  # type: ignore[arg-type]
            raise ValueError(
                f"Pipeline must start at layer 0; got start_layer={first.start_layer} on node {first.node_id}"
            )

        for prev, cur in zip(self.nodes, self.nodes[1:]):
            prev_end = int(prev.end_layer)  # type: ignore[arg-type]
            cur_start = int(cur.start_layer)  # type: ignore[arg-type]
            if cur_start != prev_end:
                raise ValueError(
                    f"Pipeline is not contiguous between {prev.node_id} and {cur.node_id}: "
                    f"prev_end={prev_end}, cur_start={cur_start}"
                )

        last = self.nodes[-1]
        if int(last.end_layer) != model_num_layers:  # type: ignore[arg-type]
            raise ValueError(
                f"Pipeline must end at total_layers={model_num_layers}; got end_layer={last.end_layer} on node {last.node_id}"
            )

        # Initialize capacity fields.
        self.recompute_capacity()

    @property
    def node_ids(self) -> Tuple[str, ...]:
        """Stage-ordered node ids (derived; not stored)."""
        return tuple(n.node_id for n in self.nodes)

    @classmethod
    def from_node_ids(
        cls, node_ids: Sequence[str], *, nodes_by_id: Mapping[str, Node]
    ) -> "Pipeline":
        """Build a Pipeline from node ids using the current node registry."""
        nodes: List[Node] = []
        for nid in node_ids:
            n = nodes_by_id.get(nid)
            if n is None:
                raise ValueError(f"Node {nid} not found in registry")
            nodes.append(n)
        return cls(nodes=list(nodes))

    @property
    def num_stages(self) -> int:
        """Number of stages in this pipeline (one stage per node)."""
        return len(self.nodes)

    @property
    def is_ready(self) -> bool:
        """True iff all member nodes are currently active/ready for RR serving.

        RR fixed pipelines are registered at bootstrap, but node liveness/load can
        change over time. This property checks the *live* `Node.is_active` flag for
        every stage.
        """
        return all(bool(getattr(n, "is_active", False)) for n in self.nodes)

    def capacity_report(self) -> Tuple[int, int]:
        """Backward-compatible helper returning (min_node_capacity, min_remaining_capacity).

        - Per-node capacity is `node.max_requests`
        - Remaining capacity is `max(0, node.max_requests - node.current_requests)`
        - Pipeline capacity is the bottleneck (min) across stages.
        """
        self.recompute_capacity()
        return int(self.min_node_capacity), int(self.min_remaining_capacity)

    def recompute_capacity(self) -> None:
        """Recompute and store capacity fields for this RR fixed pipeline.

        Fields updated:
        - `min_node_capacity`: min(node.max_requests) across stages
        - `min_remaining_capacity`: min(max(0, max_requests - current_requests)) across stages
        """
        min_node_capacity: Optional[int] = None
        min_remaining_capacity: Optional[int] = None

        for node in self.nodes:
            node_capacity = int(node.max_requests)
            node_remaining = int(max(0, node_capacity - int(node.current_requests)))

            min_node_capacity = (
                node_capacity
                if min_node_capacity is None
                else min(min_node_capacity, node_capacity)
            )
            min_remaining_capacity = (
                node_remaining
                if min_remaining_capacity is None
                else min(min_remaining_capacity, node_remaining)
            )

        self.min_node_capacity = int(min_node_capacity or 0)
        self.min_remaining_capacity = int(min_remaining_capacity or 0)

    def detach_on_member_leave(self, leaving_node_id: str, *, node_manager: "NodeManager") -> None:
        """RR-only: if any member leaves, the whole fixed pipeline becomes invalid.

        We:
        - remove all node->pipeline membership for members
        - transition remaining members to STANDBY
        - clear their layer allocations and reset their in-flight request count
        """
        pipeline_node_ids = list(self.node_ids)
        remaining = [nid for nid in pipeline_node_ids if nid != leaving_node_id]

        nodes_to_clear: List[Node] = []
        with node_manager._lock:
            for nid in pipeline_node_ids:
                node_manager._node_to_pipeline.pop(nid, None)
            if remaining:
                nodes_to_clear = node_manager._standby_locked(remaining, allow_missing=True)

        for node in nodes_to_clear:
            node.clear_serving_state()
            node.current_requests = 0


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
        # Only used for routing_strategy == "rr" (fixed/registered pipelines).
        self._registered_pipelines: Dict[int, Pipeline] = {}
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
        pipeline_to_detach: Optional[Pipeline] = None
        with self._lock:
            self._state.pop(node_id, None)
            removed = self._nodes.pop(node_id, None)
            if removed is None:
                return None
            self.node_assigned_request_count.pop(node_id, None)
            pipeline_id = self._node_to_pipeline.pop(node_id, None)
            if pipeline_id is not None:
                pipeline = self._registered_pipelines.pop(pipeline_id, None)
                pipeline_nodes = list(pipeline.node_ids) if pipeline is not None else []
                logger.warning(
                    "Node %s left; removing pipeline_id=%s from registered pipelines and detaching %d member(s): %s",
                    node_id,
                    pipeline_id,
                    len(pipeline_nodes),
                    pipeline_nodes,
                )
                if pipeline is not None:
                    pipeline.detach_on_member_leave(node_id, node_manager=self)
        removed.clear_serving_state()
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
            # TODO: Remove Runtime KV Cache
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

    def list_node_allocations(
        self, total_layers: int, ready_only: bool = False
    ) -> List[Tuple[str, int, int]]:
        """Snapshot ACTIVE segments as (node_id, start, end) under the registry lock."""
        if total_layers <= 0:
            return []
        segments: List[Tuple[str, int, int]] = []
        with self._lock:
            for nid, node in self._nodes.items():
                if self._state.get(nid) != NodeState.ACTIVE or (ready_only and not node.is_active):
                    continue
                s, e = node.start_layer, node.end_layer
                if s is None or e is None:
                    continue
                if s < 0 or e <= s or e > total_layers:
                    raise ValueError(f"Invalid layer range: {s}, {e} for node {nid}")
                segments.append((nid, int(s), int(e)))
        return segments

    def num_full_pipelines(self, total_layers: int, ready_only: bool = False) -> int:
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

        segments = self.list_node_allocations(total_layers, ready_only)
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

    def has_full_pipeline(self, num_total_layers: int, ready_only: bool = False) -> bool:
        """Check if there is a full pipeline among ACTIVE nodes."""
        return self.num_full_pipelines(num_total_layers, ready_only) > 0

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
                self._registered_pipelines[pid] = Pipeline.from_node_ids(p, nodes_by_id=self._nodes)
                for nid in p:
                    # Strict enforcement: a node must belong to exactly one pipeline.
                    if nid in self._node_to_pipeline:
                        raise ValueError(
                            f"Node {nid} is already registered to pipeline {self._node_to_pipeline[nid]}"
                        )
                    if self._state.get(nid) != NodeState.ACTIVE:
                        logger.warning(f"Node {nid} is not ACTIVE.")
                        self._state[nid] = NodeState.ACTIVE
                    self._node_to_pipeline[nid] = pid

            return {pid: list(p.node_ids) for pid, p in self._registered_pipelines.items()}

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
                self._registered_pipelines[pid] = Pipeline.from_node_ids(p, nodes_by_id=self._nodes)
                for nid in p:
                    if nid in self._node_to_pipeline:
                        raise ValueError(
                            f"Node {nid} is already registered to pipeline {self._node_to_pipeline[nid]}"
                        )
                    self._node_to_pipeline[nid] = pid
            return {pid: list(p.node_ids) for pid, p in self._registered_pipelines.items()}

    def clear_registered_pipelines(self) -> None:
        """Clear any fixed pipeline registrations and detach member nodes."""
        self.standby(list(self._node_to_pipeline.keys()))
        with self._lock:
            self._registered_pipelines = {}
            self._node_to_pipeline = {}

    def get_registered_pipelines(self) -> Dict[int, Pipeline]:
        """Return the currently registered RR fixed pipelines as `Pipeline` objects.

        This is intended for observability/routing helpers that want richer per-pipeline
        metadata (stages, readiness, capacity fields). Callers should treat returned
        objects as read-only.
        """
        with self._lock:
            return dict(self._registered_pipelines)

    def get_registered_pipeline_node_ids(self) -> Dict[int, List[str]]:
        """Return a copy of the currently registered RR fixed pipelines as node-id lists.

        This preserves the original `{pipeline_id: [node_id, ...]}` view for callers that
        only need ids (e.g. UI formatting or external APIs).
        """
        with self._lock:
            return {pid: list(p.node_ids) for pid, p in self._registered_pipelines.items()}

    def pipeline_id_of_node(self, node_id: str) -> Optional[int]:
        """Return the pipeline id a node is registered to (if any)."""
        with self._lock:
            return self._node_to_pipeline.get(node_id)

    def report_pipeline_capacity(
        self,
        ready_only: bool = True,
    ) -> Tuple[Optional[Dict[int, Tuple[int, int]]], int, int]:
        """Return per-pipeline bottleneck load + total request capacity across pipelines.

        Args:
            ready_only: If True, requires all nodes in the pipeline to be is_active (ready to serve);

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
                pipeline = self._registered_pipelines[pid]
                pipeline.recompute_capacity()
                remaining_capacity = int(pipeline.min_remaining_capacity)
                if ready_only and not pipeline.is_ready:
                    remaining_capacity = 0
                per_pipeline_min[pid] = (int(pipeline.min_node_capacity), remaining_capacity)
                total_capacity += int(pipeline.min_node_capacity)
                cur_capacity += remaining_capacity

            return per_pipeline_min, int(total_capacity), int(cur_capacity)
