"""
Scheduler for Layer Allocation and Request Routing.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Literal, Optional, Tuple

from parallax_utils.logging_config import get_logger
from scheduling.layer_allocation import (
    DynamicProgrammingLayerAllocator,
    GreedyLayerAllocator,
)
from scheduling.model_info import ModelInfo
from scheduling.node import Node, RequestSignal
from scheduling.request_routing import (
    DynamicProgrammingRouting,
    RoundRobinPipelineRouting,
)

logger = get_logger(__name__)


class Scheduler:
    """Coordinates allocation, node materialization, and request routing."""

    def __init__(
        self,
        model_info: ModelInfo,
        nodes: List[Node],
        min_nodes_bootstrapping: int = 1,
        strategy: Literal["greedy", "dp"] = "dp",
        routing_strategy: Literal["rr", "dp"] = "rr",
        *,
        request_arrival_horizon_sec: float = 600.0,
        rebalance_threshold: float = float("inf"),
        water_filling_max_iterations: int = 40,
        request_warm_up_for_reshard: int = 0,
        heartbeat_timeout: float = 60.0,
    ) -> None:
        """Initialize the scheduler.

        Args:
            model_info: Model architecture information used by allocators and routers.
            nodes: Initial list of candidate nodes.
            min_nodes_bootstrapping: Minimum nodes required to attempt initial allocation.
            strategy: Layer allocation strategy ("dp" or "greedy").
            routing_strategy: Request routing strategy ("dp" for dynamic programming, or
                "greedy" for round-robin over complete pipelines skipping overloaded ones).
            request_arrival_horizon_sec: Sliding window horizon for arrival-rate tracking.
            rebalance_threshold: Threshold for triggering rebalancing in allocation.
            water_filling_max_iterations: Max iterations for water-filling allocation.
            request_warm_up_for_reshard: Number of warm-up requests to detect truncation.
            heartbeat_timeout: Time in seconds to consider node heartbeat stale.
        """
        self.model_info = model_info
        self.num_layers = model_info.num_layers

        allocator_class = (
            GreedyLayerAllocator if strategy == "greedy" else DynamicProgrammingLayerAllocator
        )
        self.layer_allocator = allocator_class(
            model_info,
            nodes,
            rebalance_threshold=rebalance_threshold,
            water_filling_max_iterations=water_filling_max_iterations,
        )
        # Ensure Scheduler and allocator share the same node list to avoid divergence.
        self.nodes = self.layer_allocator.nodes
        self.node_id_to_node: Dict[str, Node] = self.layer_allocator.node_id_to_node
        self.min_nodes_bootstrapping = min_nodes_bootstrapping

        self.request_router = (
            DynamicProgrammingRouting() if routing_strategy == "dp" else RoundRobinPipelineRouting()
        )
        self.request_warm_up_for_reshard = request_warm_up_for_reshard

        self._request_queue: "queue.Queue[RequestSignal]" = queue.Queue()
        self.request_arrival_horizon_sec = request_arrival_horizon_sec
        self.heartbeat_timeout = heartbeat_timeout
        self._arrival_ts: Deque[float] = deque()

        # Event queues for main loop orchestration (thread-safe)
        self._pending_joins: "queue.Queue[Node]" = queue.Queue()
        self._pending_leaves: "queue.Queue[str]" = queue.Queue()
        self._pending_node_updates: "queue.Queue[Tuple[str, Optional[int], Optional[float], Optional[Dict[str, float]], Optional[bool]]]" = (queue.Queue())

        # Concurrency controls
        self._stop_event: threading.Event = threading.Event()
        self._wake_event: threading.Event = threading.Event()
        self._node_count_cv: threading.Condition = threading.Condition()
        self._event_thread: Optional[threading.Thread] = None
        self._dispatch_thread: Optional[threading.Thread] = None
        self._alloc_log_thread: Optional[threading.Thread] = None
        # Thread-safe bootstrap state
        self._bootstrapped: bool = False
        self._bootstrapped_event: threading.Event = threading.Event()
        logger.debug(
            f"Scheduler initialized, min_nodes_bootstrapping {self.min_nodes_bootstrapping}, "
            f"strategy {strategy}, rebalance threshold {rebalance_threshold}"
        )
        self._node_assigned_request_count: Dict[str, int] = {}

        # Eager bootstrap for initial allocation if enough nodes are present
        try:
            if len(self.nodes) >= self.min_nodes_bootstrapping:
                logger.debug(
                    f"Eager allocation attempt with {len(self.nodes)} nodes (min required: {self.min_nodes_bootstrapping})"
                )
                self.layer_allocator.global_allocation()
        except Exception:  # best-effort eager allocation
            pass

    # Orchestration helpers
    def bootstrap(self) -> bool:
        """Bootstrapping: first-time layer allocation and optional warm-up.

        Returns True if a full pipeline was established; False otherwise.
        """
        if len(self.nodes) < self.min_nodes_bootstrapping:
            logger.debug(
                f"Bootstrapping deferred: have {len(self.nodes)} nodes; need >= {self.min_nodes_bootstrapping}"
            )
            return False
        logger.debug("Bootstrapping layer allocator")
        success = self.layer_allocator.global_allocation()
        if not success:
            logger.warning("Bootstrapping failed to produce a full pipeline")
            return False
        assignments = self.list_node_allocations()
        logger.debug(f"Layer allocator assignments: {assignments}")
        # Optional warm-up to find turning points and truncate node ranges
        if self.request_warm_up_for_reshard > 0:
            self._run_warmup_and_truncate()
            assignments = self.list_node_allocations()
            logger.debug(f"Layer allocator assignments after turn-point warm-up: {assignments}")

        if not self.layer_allocator.has_full_pipeline():
            logger.warning("Bootstrapping failed to produce a full pipeline")
            return False
        self._bootstrapped = True
        self._bootstrapped_event.set()
        logger.debug("Bootstrapping completed successfully; full pipeline established")
        return True

    def list_node_allocations(self) -> List[Tuple[str, int, int]]:
        """List the allocations of all nodes."""
        return self.layer_allocator.list_node_allocations()

    # Warm-up and re-shard
    def _run_warmup_and_truncate(self) -> None:
        """Run a brief warm-up to detect truncation points and shrink shards.

        Uses layer-level DP turning points (node_id, layer_idx, kind):
        - kind == "tail": drop [layer_idx, end) on that node
        - kind == "head": drop [start, layer_idx) on that node
        """
        nodes_list = list(self.nodes)
        if not nodes_list:
            return
        num_layers = self.model_info.num_layers
        # The number of warm-up requests can be used to repeat detection, but a
        # single pass is sufficient with our DP model; we repeat to smooth noise.
        agg_turns: Dict[Tuple[str, int, str], int] = {}
        for _ in range(self.request_warm_up_for_reshard):
            turns = self.request_router.find_turning_points(nodes_list, num_layers)
            for t in turns:
                agg_turns[t] = agg_turns.get(t, 0) + 1
        # Apply truncation for consistently observed turning points
        for node_id, layer_idx, kind in agg_turns:
            node = next((n for n in self.nodes if n.node_id == node_id), None)
            if node is None or node.start_layer is None or node.end_layer is None:
                continue
            start, end = node.start_layer, node.end_layer
            if kind == "tail":
                if layer_idx < end:
                    node.set_layer_allocation(start, layer_idx)
            elif kind == "head":
                if layer_idx > start:
                    node.set_layer_allocation(layer_idx, end)

    def update_node_info(
        self,
        node: Node,
        *,
        current_requests: Optional[int] = None,
        layer_latency_ms: Optional[float] = None,
        new_rtt_to_nodes: Optional[Dict[str, float]] = None,
        is_active: Optional[bool] = None,
    ) -> None:
        """Update the info of a node."""
        if current_requests is not None:
            node.current_requests = current_requests
        if layer_latency_ms is not None:
            node.set_layer_latency_ms(layer_latency_ms)
        if new_rtt_to_nodes is not None:
            node.rtt_to_nodes.update(new_rtt_to_nodes)
        if is_active is not None:
            node.is_active = is_active
        node.last_heartbeat = time.time()
        # logger.debug(
        #     "Node updated: %s (requests=%s, latency_ms=%s, rtt_updates=%s)",
        #     node.node_id,
        #     current_requests if current_requests is not None else node.current_requests,
        #     layer_latency_ms if layer_latency_ms is not None else node.avg_layer_latency_ms,
        #     0 if new_rtt_to_nodes is None else len(new_rtt_to_nodes),
        # )

    # Async-style event enqueuers for main loop
    def enqueue_join(self, node: Node) -> None:
        """Enqueue a join event."""
        logger.debug(f"Enqueueing join event for node {node.node_id}")
        self._pending_joins.put(node)
        self._wake_event.set()

    def enqueue_leave(self, node_id: str) -> None:
        """Enqueue a leave event."""
        self._pending_leaves.put(node_id)
        self._wake_event.set()

    def enqueue_node_update(
        self,
        node_id: str,
        *,
        current_requests: Optional[int] = None,
        layer_latency_ms: Optional[float] = None,
        new_rtt_to_nodes: Optional[Dict[str, float]] = None,
        is_active: Optional[bool] = None,
    ) -> None:
        """Enqueue a node update event."""
        self._pending_node_updates.put(
            (node_id, current_requests, layer_latency_ms, new_rtt_to_nodes, is_active)
        )
        self._wake_event.set()

    def checking_node_heartbeat(self) -> None:
        """Check the heartbeat of all nodes."""
        for node in self.nodes:
            if not node.is_active:
                continue
            if time.time() - node.last_heartbeat > self.heartbeat_timeout:
                logger.debug(f"Node {node.node_id} heartbeat timeout")
                self.leave(node.node_id)

    # Dynamic node management
    def join(self, node: Node, bootstrap: bool = False) -> None:
        """Add a node to allocation and refresh plan and materialized nodes."""
        logger.debug(
            "Joining node %s (kv_ratio=%.2f, param_ratio=%.2f)",
            node.node_id,
            node.kv_cache_ratio,
            node.param_hosting_ratio,
        )
        self.layer_allocator.declare(node)
        if not bootstrap:
            self.layer_allocator.join(node)
        # Notify waiters that node count changed
        with self._node_count_cv:
            self._node_count_cv.notify_all()

    def leave(self, node_id: str) -> None:
        """Remove a node from allocation and refresh plan and materialized nodes."""
        if node_id not in self.layer_allocator.node_id_to_node:
            raise ValueError(f"Node {node_id} not found in nodes")
        node = self.node_id_to_node[node_id]
        logger.debug(
            "Leaving node %s (start=%s, end=%s)", node_id, node.start_layer, node.end_layer
        )
        self.layer_allocator.leave(node_id)
        if self.layer_allocator.should_global_rebalance():
            logger.debug("Global rebalance triggered due to node leave")
            # TODO: send a signal to the nodes to stop running requests
            #       and re-assign start/end layers so nodes can re-shard
            self._bootstrapped = False
            self._bootstrapped_event.clear()
            for n in self.nodes:
                if n.start_layer is not None and n.end_layer is not None:
                    self.layer_allocator.deallocate(n)
            self.layer_allocator.global_allocation()
        with self._node_count_cv:
            self._node_count_cv.notify_all()

    def receive_request(self, request: RequestSignal) -> None:
        """Add a request to the wait pool."""
        self._request_queue.put(request)
        self._wake_event.set()
        now = time.time()
        self._arrival_ts.append(now)
        logger.debug(
            "Received request %s (queue_size=%d)", request.request_id, self._request_queue.qsize()
        )
        # Trim old timestamps to keep arrival-rate window bounded
        horizon = self.request_arrival_horizon_sec
        while self._arrival_ts and now - self._arrival_ts[0] > horizon:
            self._arrival_ts.popleft()

    def dispatch_next_request(self) -> Optional[Tuple[str, List[str], float]]:
        """Route the next request in the wait pool; returns (request_id, path, latency)."""
        try:
            req = self._request_queue.get_nowait()
        except queue.Empty:
            req = None
        if req is None:
            return None
        path, latency = self.request_router.find_optimal_path(self.nodes, self.num_layers)
        req.routing_table = path
        # Update simple load counters
        for node_id in path:
            n = self.node_id_to_node[node_id]
            if n is not None:
                self._node_assigned_request_count[node_id] = (
                    self._node_assigned_request_count.get(node_id, 0) + 1
                )
                n.add_request()
        logger.debug(
            "Dispatched request %s via path %s (est_lat=%.2fms)", req.request_id, path, latency
        )
        return req.request_id, path, latency

    def run(self, *, poll_interval: float = 0.05, allocation_log_interval: float = 5.0) -> None:
        """Run the scheduler concurrently until `stop()` is called.

        Starts background threads for event processing (joins/leaves/updates/heartbeats)
        and request dispatching. At startup, waits until at least
        `min_nodes_bootstrapping` nodes are present, then runs `bootstrap()`.
        """
        logger.debug("Running scheduler")
        self._stop_event.clear()

        # Start event thread first so joins can be processed while we wait to bootstrap
        self._event_thread = threading.Thread(
            target=self._event_loop, args=(poll_interval,), name="SchedulerEventLoop", daemon=True
        )
        self._event_thread.start()

        # Bootstrap gating
        if not self._wait_for_bootstrap(poll_interval):
            return

        # Start dispatcher only after successful bootstrap
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            args=(poll_interval,),
            name="SchedulerDispatcher",
            daemon=True,
        )
        self._dispatch_thread.start()

        # Start periodic allocation logger thread
        def _alloc_log_loop() -> None:
            """Periodically log current layer allocations."""
            while not self._stop_event.is_set():
                try:
                    assignments = self.list_node_allocations()
                    header = f"Current allocations ({len(assignments)} nodes)"
                    sep = "-" * len(header)
                    logger.debug("%s\n%s", header, sep)
                    for node_id, start_layer, end_layer in assignments:
                        node = self.node_id_to_node[node_id]
                        # Snapshot values to avoid recomputing/logging side-effects twice
                        capacity = node.max_requests
                        current = node.current_requests
                        latency = node.layer_latency_ms
                        latency_str = "inf" if latency == float("inf") else f"{latency:.2f}"
                        n_hosted_requests = 0
                        if node_id in self._node_assigned_request_count:
                            n_hosted_requests = self._node_assigned_request_count[node_id]
                        logger.debug(
                            "  %-16s layers [%3d, %3d) | load %3d/%-3d | latency %7s ms | assigned request count %3d",
                            node_id,
                            start_layer,
                            end_layer,
                            current,
                            capacity,
                            latency_str,
                            n_hosted_requests,
                        )
                except Exception as exc:
                    logger.warning(f"Allocation logger error: {exc}")
                time.sleep(max(1.0, allocation_log_interval))

        self._alloc_log_thread = threading.Thread(
            target=_alloc_log_loop, name="SchedulerAllocLogger", daemon=True
        )
        self._alloc_log_thread.start()

        # Block until stop is requested
        try:
            while not self._stop_event.is_set():
                time.sleep(max(0.5, poll_interval))
        finally:
            if self._event_thread is not None:
                self._event_thread.join(timeout=2.0)
            if self._dispatch_thread is not None:
                self._dispatch_thread.join(timeout=2.0)
            if self._alloc_log_thread is not None:
                self._alloc_log_thread.join(timeout=2.0)

    # === Modularized worker loops ===
    def _event_loop(self, poll_interval: float) -> None:
        """Process joins/leaves/updates and perform heartbeat checks."""
        last_hb_check = 0.0
        while not self._stop_event.is_set():
            self._process_node_updates()
            self._process_joins()
            self._process_leaves()
            now = time.time()
            if now - last_hb_check >= max(0.5, poll_interval):
                self.checking_node_heartbeat()
                last_hb_check = now
            self._wake_event.wait(timeout=poll_interval)
            self._wake_event.clear()

    def _dispatch_loop(self, poll_interval: float) -> None:
        """Continuously dispatch incoming requests while running."""
        while not self._stop_event.is_set():
            try:
                req = self._request_queue.get(timeout=poll_interval)
                if req is None:
                    continue
                path, _ = self.request_router.find_optimal_path(self.nodes, self.num_layers)
                req.routing_table = path
                for node_id in path:
                    n = self.node_id_to_node[node_id]
                    if n is not None:
                        self._node_assigned_request_count[node_id] = (
                            self._node_assigned_request_count.get(node_id, 0) + 1
                        )
                        n.add_request()
                logger.debug(
                    "Dispatched request %s via path %s", getattr(req, "request_id", "?"), path
                )
            except queue.Empty:
                continue

    def _wait_for_bootstrap(self, poll_interval: float) -> bool:
        """Wait until enough nodes then run bootstrap. Returns False if stopped."""
        logger.debug("Waiting for bootstrap")
        while not self._stop_event.is_set() and not self._bootstrapped_event.is_set():
            with self._node_count_cv:
                if len(self.nodes) < self.min_nodes_bootstrapping:
                    self._node_count_cv.wait(timeout=max(0.5, poll_interval))
                    continue
            boot_ok = self.bootstrap()
            if boot_ok:
                break
            with self._node_count_cv:
                self._node_count_cv.wait(timeout=max(1.0, poll_interval))
        return not self._stop_event.is_set()

    def _process_node_updates(self) -> None:
        """Apply pending node stats updates from the queue."""
        while True:
            try:
                node_id, cur, lat, rtts, is_active = self._pending_node_updates.get_nowait()
            except queue.Empty:
                break
            self.update_node_info(
                self.node_id_to_node[node_id],
                current_requests=cur,
                layer_latency_ms=lat,
                new_rtt_to_nodes=rtts,
                is_active=is_active,
            )

    def _process_joins(self) -> None:
        """Handle pending join events, honoring bootstrap state for assignment."""
        joined_any = False
        while True:
            try:
                node = self._pending_joins.get_nowait()
            except queue.Empty:
                break
            # During bootstrap (no full pipeline yet), only declare nodes; no dynamic assignment.
            # After bootstrap, allow dynamic light-weight joins.
            self.join(node, bootstrap=not self._bootstrapped_event.is_set())
            joined_any = True

        # If we are not bootstrapped (e.g., after a leave-triggered rebalance) and
        # new nodes just joined, attempt a greedy bootstrap immediately when we have
        # enough nodes. If it doesn't produce a full pipeline, we'll try again on
        # subsequent joins.
        if joined_any and not self._bootstrapped_event.is_set():
            if len(self.nodes) >= self.min_nodes_bootstrapping:
                try:
                    ok = self.bootstrap()
                    if not ok:
                        logger.debug(
                            "Bootstrap attempt after join did not produce a full pipeline; will retry on future joins"
                        )
                except Exception as exc:
                    logger.debug(
                        f"Bootstrap attempt after join failed: {exc}; will retry on future joins"
                    )
            else:
                logger.debug(
                    "Deferring bootstrap: have %d nodes; need >= %d",
                    len(self.nodes),
                    self.min_nodes_bootstrapping,
                )

    def _process_leaves(self) -> None:
        """Handle pending leave events safely."""
        while True:
            try:
                node_id = self._pending_leaves.get_nowait()
            except queue.Empty:
                break
            try:
                self.leave(node_id)
            except Exception as exc:
                logger.warning(f"Leave failed for {node_id}: {exc}")

    def stop(self) -> None:
        """Signal background threads to stop and wake any waiters."""
        self._stop_event.set()
        self._wake_event.set()
        with self._node_count_cv:
            self._node_count_cv.notify_all()
