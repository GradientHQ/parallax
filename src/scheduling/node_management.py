"""
Node registry and lifecycle management.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Dict, List, Optional, Tuple, Iterator, DefaultDict
from collections import defaultdict

from scheduling.node import Node


class NodeState(str, Enum):
    """Lifecycle state of a joined node."""

    ACTIVE = "active"
    STANDBY = "standby"


class NodeManagement:
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
        self._registered_pipelines: List[List[str]] = []

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

    def remove(self, node_id: str) -> Optional[Node]:
        """Remove a node; returns removed node if present."""
        with self._lock:
            self._state.pop(node_id, None)
            removed = self._nodes.pop(node_id, None)
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
        with self._lock:
            for nid in node_ids:
                if nid not in self._nodes:
                    raise ValueError(f"Node {nid} not found in registry")
                prev_state = self._state.get(nid)
                if prev_state is not None and prev_state != NodeState.ACTIVE:
                    raise ValueError(f"Node {nid} is not ACTIVE, current state: {prev_state}")
                self._nodes[nid].clear_layer_allocation()
                self._state[nid] = NodeState.STANDBY

    def snapshot(self, *, state: Optional[NodeState] = None) -> List[Node]:
        """Return a copy of nodes, optionally filtered by state."""
        with self._lock:
            if state is None:
                return list(self._nodes.values())
            return [n for nid, n in self._nodes.items() if self._state.get(nid) == state]

    @property
    def num_nodes(self) -> int:
        with self._lock:
            return len(self._nodes)

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
