"""
Node registry and lifecycle management.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Dict, List, Optional

from scheduling.node import Node


class NodeState(str, Enum):
    """Lifecycle state of a joined node."""

    ACTIVE = "active"
    STANDBY = "standby"


class NodeRegistry:
    """Thread-safe registry for node membership and lifecycle.

    Responsibilities:
    - store node membership by node_id
    - track lifecycle state (active vs standby)
    - provide thread-safe snapshots for routing/allocation decisions
    """

    def __init__(self, *, initial_nodes: Optional[List[Node]] = None) -> None:
        self._lock = threading.RLock()
        self._nodes: Dict[str, Node] = {}
        self._state: Dict[str, NodeState] = {}

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
            return self._nodes.pop(node_id, None)

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

    def move_to_standby(self, node_ids: List[str]) -> None:
        """Mark nodes as STANDBY (joined but not actively serving)."""
        with self._lock:
            for nid in node_ids:
                if nid not in self._nodes:
                    raise ValueError(f"Node {nid} not found in registry")
                if self._state.get(nid) != NodeState.ACTIVE:
                    raise ValueError(f"Node {nid} is not ACTIVE")
                self._state[nid] = NodeState.STANDBY

    def snapshot(self, *, state: Optional[NodeState] = None) -> List[Node]:
        """Return a copy of nodes, optionally filtered by state."""
        with self._lock:
            if state is None:
                return list(self._nodes.values())
            return [n for nid, n in self._nodes.items() if self._state.get(nid) == state]
