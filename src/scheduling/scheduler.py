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
from scheduling.node_management import NodeManager
from scheduling.request_routing import (
    DynamicProgrammingRouting,
    RandomizedOverDynamicPipelinesRouting,
    RoundRobinOverFixedPipelinesRouting,
)

logger = get_logger(__name__)


class Scheduler:
    """Coordinates allocation, node materialization, and request routing."""

    def __init__(
        self,
        model_info: ModelInfo,
        nodes: List[Node],
        min_nodes_bootstrapping: int = 1,
        enable_weight_refit: bool = False,
        weight_refit_mode: str = "disk",
        strategy: Literal["greedy", "dp"] = "dp",
        routing_strategy: Literal["rr", "dp"] = "rr",
        *,
        request_arrival_horizon_sec: float = 600.0,
        rebalance_threshold: float = float("inf"),
        water_filling_max_iterations: int = 40,
        heartbeat_timeout: float = 60.0,
        trim_layers_on_turning_points: bool = True,
    ) -> None:
        """Initialize the scheduler.

        Args:
            model_info: Model architecture information used by allocators and routers.
            nodes: Initial list of candidate nodes.
            min_nodes_bootstrapping: Minimum nodes required to attempt initial allocation.
            strategy: Layer allocation strategy ("dp" or "greedy").
            routing_strategy: Request routing strategy:
                - "dp": dynamic-programming routing over current allocations (minimum latency).
                - "rr": round-robin selection over fixed, small set of pipelines.
            request_arrival_horizon_sec: Sliding window horizon for arrival-rate tracking.
            rebalance_threshold: Threshold for triggering rebalancing in allocation.
            water_filling_max_iterations: Max iterations for water-filling allocation.
            heartbeat_timeout: Time in seconds to consider node heartbeat stale.
            trim_layers_on_turning_points: Whether to trim layers on turning points.
        """
        self.model_info = model_info
        self.num_layers = model_info.num_layers
        self.routing_strategy: Literal["rr", "dp"] = routing_strategy
        self.enable_weight_refit = enable_weight_refit
        self.weight_refit_mode = weight_refit_mode
        self.refit_request = {}
        self.node_manager = NodeManager(initial_nodes=nodes)

        allocator_class = (
            GreedyLayerAllocator if strategy == "greedy" else DynamicProgrammingLayerAllocator
        )
        self.dynamic_pipelines_router = routing_strategy == "dp"
        # TODO: expose DP's alpha
        self.layer_allocator = allocator_class(
            model_info=model_info,
            node_management=self.node_manager,
            dynamic_pipelines_router=self.dynamic_pipelines_router,
            rebalance_threshold=rebalance_threshold,
            water_filling_max_iterations=water_filling_max_iterations,
            trim_layers_on_turning_points=trim_layers_on_turning_points,
        )
        # Ensure Scheduler and allocator share the same node list to avoid divergence.
        self.min_nodes_bootstrapping = min_nodes_bootstrapping

        self.request_router = (
            DynamicProgrammingRouting()
            if routing_strategy == "dp"
            else RoundRobinOverFixedPipelinesRouting(self.node_manager)
        )

        self._request_queue: "queue.Queue[RequestSignal]" = queue.Queue()
        self.request_arrival_horizon_sec = request_arrival_horizon_sec
        self.heartbeat_timeout = heartbeat_timeout
        self._arrival_ts: Deque[float] = deque()

        # Event queues for main loop orchestration (thread-safe)
        self._pending_joins: "queue.Queue[Node]" = queue.Queue()
        self._pending_leaves: "queue.Queue[str]" = queue.Queue()
        self._pending_node_updates: "queue.Queue[Tuple[str, Optional[int], Optional[float], Optional[Dict[str, float]], Optional[bool], Optional[bool]]]" = (queue.Queue())

        # Concurrency controls
        self._stop_event: threading.Event = threading.Event()
        self._wake_event: threading.Event = threading.Event()
        self._bootstrapped_event: threading.Event = threading.Event()
        self._node_count_cv: threading.Condition = threading.Condition()
        self._event_thread: Optional[threading.Thread] = None
        self._dispatch_thread: Optional[threading.Thread] = None
        self._alloc_log_thread: Optional[threading.Thread] = None
        # Latest formatted allocation snapshot (string) for status/inspection.
        # This is updated by `emit_alloc_log_snapshot()`.
        self.alloc_log_snapshot: str = ""
        # Avoid spamming: only emit the "all nodes active" INFO log on transitions.
        self._all_nodes_active_logged: bool = False
        logger.info(
            f"Scheduler initialized, min_nodes_bootstrapping {self.min_nodes_bootstrapping}, "
            f"Layer allocations trategy {strategy}, Request routing strategy {routing_strategy}."
        )

        # Weight refit
        self.refit_request = {}
        self.refit_set = set()
        self.last_refit_time = 0.0

    def list_node_allocations(
        self, total_layers: Optional[int] = None
    ) -> List[Tuple[str, int, int]]:
        """Return current (node_id, start_layer, end_layer) allocations.

        This is a small convenience wrapper around `NodeManager.list_node_allocations` and is
        relied upon by some callers (e.g. backend RPC handler, docs).
        """
        return self.node_manager.list_node_allocations(total_layers or self.num_layers)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Fetch a node by id (compat helper for callers that shouldn't reach into NodeManager)."""
        return self.node_manager.get(node_id)

    def has_full_pipeline(self) -> bool:
        """Check if there is a full pipeline among ACTIVE nodes."""
        return self.node_manager.has_full_pipeline(self.num_layers)

    def _maybe_expand_rr_pipelines(self) -> None:
        """RR-only: try to allocate/register additional pipelines from STANDBY nodes.

        This is a best-effort opportunistic expansion. It never touches ACTIVE nodes because
        allocator `allocate_from_standby()` only draws from the STANDBY pool.
        """
        if self.routing_strategy != "rr":
            return
        if not self._bootstrapped_event.is_set():
            return

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
            newly_active_nodes, self.num_layers
        )
        if not new_pipelines:
            logger.warning("[RR] No new pipelines found for extended RR")
            return

        # Only extend with pipelines that do not overlap existing registered nodes.
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

    def bootstrap(self, reboot: bool = False) -> bool:
        """Initial Node Allocation Assignment."""
        logger.info("[Scheduler] Starting Bootstrap")
        overide_min_node_check = False
        if reboot:
            # Clear any fixed pipeline registrations; they are no longer valid.
            # This also detaches member nodes and clears their layer allocations.
            logger.info("[Scheduler] Rebooting, moving every node to standby")
            self.node_manager.clear_registered_pipelines()
            self._bootstrapped_event.clear()
            overide_min_node_check = True
        else:
            # If we already bootstrapped, return True
            if self._bootstrapped_event.is_set():
                logger.info("[Scheduler] Already bootstrapped, returning Success")
                return True
        # Check if we have enough nodes for bootstraping
        if (
            self.node_manager.num_nodes < self.min_nodes_bootstrapping
            and not overide_min_node_check
        ):
            logger.info(
                f"[Scheduler] Bootstrap deferred: have {self.node_manager.num_nodes} nodes; need >= {self.min_nodes_bootstrapping}"
            )
            return False

        # Perform global allocation
        success = self.layer_allocator.allocate_from_standby()
        if not success:
            logger.warning("Global allocation failed to produce a full pipeline")
            # Stay un-bootstrapped so future joins can retry bootstrap.
            self._bootstrapped_event.clear()
            return False

        assignments = self.node_manager.list_node_allocations(self.num_layers)
        logger.info(f"[Scheduler] Post Bootstrap Layer Assignments: {assignments}")

        # For fixed (RR) routing: register a pipeline set immediately after bootstrap.
        if self.routing_strategy == "rr" and isinstance(
            self.request_router, RoundRobinOverFixedPipelinesRouting
        ):
            try:
                self.request_router.register_pipelines(
                    self.node_manager.active_nodes, self.num_layers
                )
                logger.info(
                    f"[FixedRouter] register_pipelines with bootstrap success, number of pipelines: {len(self.request_router.get_registered_pipelines())}"
                )

            except Exception as exc:
                logger.warning(
                    f"[FixedRouter] register_pipelines after bootstrap failed (best-effort): {exc}"
                )

        self._bootstrapped_event.set()
        # Snapshot at INFO after bootstrap since allocations/pipelines may have materially changed.
        self.emit_alloc_log_snapshot(reason="Post Bootstrap")
        return True

    def update_last_refit_time(self):
        min_refit_time = None
        for node in self.node_manager.nodes:
            cur_node_refit_time = node.last_refit_time
            if cur_node_refit_time < self.last_refit_time:
                continue
            if min_refit_time is None:
                min_refit_time = cur_node_refit_time
            else:
                min_refit_time = min(min_refit_time, cur_node_refit_time)
        if min_refit_time is not None:
            self.last_refit_time = min_refit_time
        return self.last_refit_time

    def update_node_info(
        self,
        node: Node,
        *,
        current_requests: Optional[int] = None,
        layer_latency_ms: Optional[float] = None,
        new_rtt_to_nodes: Optional[Dict[str, float]] = None,
        is_active: Optional[bool] = None,
        last_refit_time: Optional[float] = 0.0,
    ) -> None:
        """Update the info of a node."""
        if current_requests is not None:
            node.current_requests = current_requests
        if layer_latency_ms is not None:
            node.set_layer_latency_ms(layer_latency_ms)
        if new_rtt_to_nodes is not None:
            node.rtt_to_nodes = new_rtt_to_nodes
        if is_active is not None:
            node.is_active = is_active
        if last_refit_time > 0.0:
            node.last_refit_time = last_refit_time
        node.last_heartbeat = time.time()

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
        last_refit_time: Optional[float] = 0.0,
    ) -> None:
        """Enqueue a node update event."""
        self._pending_node_updates.put(
            (
                node_id,
                current_requests,
                layer_latency_ms,
                new_rtt_to_nodes,
                is_active,
                last_refit_time,
            )
        )
        self._wake_event.set()

    def checking_node_heartbeat(self) -> None:
        """Check the heartbeat of all nodes."""
        for node in self.node_manager.active_nodes:
            if time.time() - node.last_heartbeat > self.heartbeat_timeout:
                logger.debug(f"Node {node.node_id} heartbeat timeout")
                # Route leave through the event loop so global rebalance/reboot is serialized.
                self.enqueue_leave(node.node_id)

    # Dynamic node management
    def join(self, node: Node) -> None:
        """Add a node to allocation and refresh plan and materialized nodes."""
        bootstrapped = self._bootstrapped_event.is_set()
        logger.info(
            "Joining node %s (kv_ratio=%.2f, param_ratio=%.2f, manual_assignment=%s, bootstrapped=%s)",
            node.node_id,
            node.kvcache_mem_ratio,
            node.param_mem_ratio,
            node.manual_layer_assignment,
            bootstrapped,
        )
        if self.node_manager.get(node.node_id) is None:
            self.node_manager.upsert(node)
            if bootstrapped:
                if self.dynamic_pipelines_router:
                    # for dynamic pipelines router, join the node to the lightest layer
                    self.layer_allocator.dynamic_join(node)
                if self.routing_strategy == "rr":
                    # for RR mode, try allocate from standby and expand the pipelines
                    self._maybe_expand_rr_pipelines()

        # Manual layer assignment bypasses bootstrap waiting
        if node.manual_layer_assignment:
            # Manual layer assignment: use the layers specified by the node
            if node.start_layer is None or node.end_layer is None:
                raise ValueError(
                    f"Node {node.node_id} has manual_layer_assignment=True "
                    f"but start_layer ({node.start_layer}) or end_layer ({node.end_layer}) is None"
                )
            logger.info(
                f"Manual layer assignment for node {node.node_id}: "
                f"layers [{node.start_layer}, {node.end_layer})"
            )
            # Directly allocate the specified layers without automatic assignment
            self.layer_allocator.allocate(node, node.start_layer, node.end_layer)

            # Check if manual allocations now cover the full pipeline
            if self.has_full_pipeline():
                if not self._bootstrapped_event.is_set():
                    logger.info(
                        "[Scheduler] Manual layer assignments have established a full pipeline; "
                        "marking scheduler as bootstrapped"
                    )
                    self._bootstrapped_event.set()
        elif bootstrapped:
            self.node_manager.upsert(node)

        # Notify waiters that node count changed
        # Snapshot at INFO after join since allocations/pipelines may have changed.
        self.emit_alloc_log_snapshot(reason=f"after join {node.node_id}")
        with self._node_count_cv:
            self._node_count_cv.notify_all()

    def leave(self, node_id: str) -> None:
        """Remove a node from the node manager.

        If using fixed pipeliens:
        - Nullify the pipeline the node is in;
        - Move all remaining nodes in the pipeline to STANDBY;

        Note: Global rebalance/reboot is handled by the event loop (`_process_leaves`) to
        ensure we don't concurrently reboot when multiple leave events arrive.
        """
        node = self.node_manager.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in nodes")
        logger.info("Leaving node %s (start=%s, end=%s)", node_id, node.start_layer, node.end_layer)
        self.node_manager.remove(node_id)

        # Snapshot at INFO after leave since allocations/pipelines may have changed.
        self.emit_alloc_log_snapshot(reason=f"after leave {node_id}")

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
        path, latency = self.request_router.find_optimal_path(
            self.node_manager.active_nodes, self.num_layers, self.last_refit_time
        )
        req.routing_table = path
        # Update simple load counters
        for node_id in path:
            n = self.node_manager.get(node_id)
            if n is not None:
                self.node_manager.add_request(node_id)
        logger.debug(
            "Dispatched request %s via path %s (est_lat=%.2fms)", req.request_id, path, latency
        )
        return req.request_id, path, latency

    def _format_current_allocations_snapshot(self) -> str:
        assignments = self.node_manager.list_node_allocations(self.num_layers)
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

    def _format_rr_registered_pipelines_snapshot(self) -> str:
        if self.routing_strategy != "rr":
            return ""
        pipelines = self.node_manager.get_registered_pipelines()
        p_header = f"Registered pipelines ({len(pipelines)})"
        p_sep = "-" * len(p_header)
        lines: List[str] = [p_header, p_sep]
        # Include capacity summary in the RR snapshot message.
        per_pipeline_min, total_capacity, cur_capacity = self.report_pipeline_capacity()
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
            node_ids = pipelines.get(pid, [])
            lines.append("  pipeline %-3d | stages=%d" % (pid, len(node_ids)))
            for idx, nid in enumerate(node_ids):
                n = self.node_manager.get(nid)
                if n is None:
                    lines.append("    [%02d] %-16s (missing)" % (idx, nid))
                    continue
                s = -1 if n.start_layer is None else int(n.start_layer)
                e = -1 if n.end_layer is None else int(n.end_layer)
                lat = n.layer_latency_ms
                lat_str = "inf" if lat == float("inf") else f"{lat:.2f}"
                lines.append(
                    "    [%02d] %-16s layers [%3d, %3d) | load %3d/%-3d | latency %7s ms | active %s"
                    % (
                        idx,
                        nid,
                        s,
                        e,
                        n.current_requests,
                        n.max_requests,
                        lat_str,
                        n.is_active,
                    )
                )
        return "\n".join(lines)

    def emit_alloc_log_snapshot(self, *, reason: Optional[str] = None) -> str:
        """Update `self.alloc_log_snapshot` and emit it.

        - Periodic/heartbeat snapshots (no reason) are logged at DEBUG.
        - Mutating events (join/leave/bootstrap) provide a reason and are logged at INFO.
        """
        try:
            if self.routing_strategy == "rr":
                snapshot = self._format_rr_registered_pipelines_snapshot()
            else:
                snapshot = self._format_current_allocations_snapshot()
        except Exception as exc:
            snapshot = f"(failed to build allocation snapshot: {exc})"
            logger.warning("Allocation snapshot build error: %s", exc)

        self.alloc_log_snapshot = snapshot

        if reason:
            logger.info("Allocation snapshot (%s)\n%s", reason, snapshot)
        else:
            logger.debug("Allocation snapshot\n%s", snapshot)
        return snapshot

    def report_pipeline_capacity(
        self,
    ) -> Tuple[Optional[Dict[int, Tuple[int, int]]], int, int]:
        """Helper to report the current pipeline capacity.

        Returns:
            per_pipeline_min: Dict of pipeline id -> (min_node_capacity, min_remaining_capacity).
            total_capacity: The total capacity of all registered pipelines.
            cur_capacity: The current capacity (counting existing request load) of all registered pipelines.
        """
        return self.node_manager.report_pipeline_capacity()

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
                    self.emit_alloc_log_snapshot()
                except Exception as exc:
                    logger.warning(f"Allocation logger error: {exc}")

                # After bootstrap, periodically check if *all* nodes report active and log once.
                if self._bootstrapped_event.is_set():
                    nodes = self.node_manager.nodes
                    if nodes:
                        all_active = all(n.is_active for n in nodes)
                        if all_active and not self._all_nodes_active_logged:
                            logger.info("All %d nodes are active", len(nodes))
                            self._all_nodes_active_logged = True
                        elif not all_active:
                            self._all_nodes_active_logged = False
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
                path, path_rtt = self.request_router.find_optimal_path(
                    self.node_manager.active_nodes, self.num_layers
                )
                logger.debug(f"Path RTT: {path_rtt}")
                req.routing_table = path
                for node_id in path:
                    self.node_manager.add_request(node_id)
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
                self._node_count_cv.wait(timeout=max(0.5, poll_interval))
        return not self._stop_event.is_set()

    def _process_node_updates(self) -> None:
        """Apply pending node stats updates from the queue."""
        while True:
            try:
                node_id, cur, lat, rtts, is_active, last_refit_time = (
                    self._pending_node_updates.get_nowait()
                )
            except queue.Empty:
                break
            node = self.node_manager.get(node_id)
            if node is None:
                logger.warning(f"Node {node_id} not found in node manager, ignore the update")
                continue
            self.update_node_info(
                node,
                current_requests=cur,
                layer_latency_ms=lat,
                new_rtt_to_nodes=rtts,
                is_active=is_active,
                last_refit_time=last_refit_time,
            )

    def _process_joins(self) -> None:
        """Handle pending join events, honoring bootstrap state for assignment."""
        joined_any = False
        had_manual_assignment = False
        while True:
            try:
                node = self._pending_joins.get_nowait()
            except queue.Empty:
                break
            # During bootstrap (no full pipeline yet), only declare nodes; no dynamic assignment.
            # After bootstrap, allow dynamic light-weight joins.
            # Exception: manual layer assignments are processed immediately regardless of bootstrap state.
            self.join(node)
            joined_any = True
            if node.manual_layer_assignment:
                had_manual_assignment = True

        # If we are not bootstrapped (e.g., after a leave-triggered rebalance) and
        # new nodes just joined, attempt a greedy bootstrap immediately when we have
        # enough nodes. If it doesn't produce a full pipeline, we'll try again on
        # subsequent joins.
        # Skip bootstrap if manual assignments were used (they handle bootstrapping internally).
        if joined_any and not self._bootstrapped_event.is_set() and not had_manual_assignment:
            if self.node_manager.num_standby_nodes >= self.min_nodes_bootstrapping:
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
                    self.node_manager.num_standby_nodes,
                    self.min_nodes_bootstrapping,
                )

    def _process_leaves(self) -> None:
        """Handle pending leave events safely.

        Important: This is the only place we trigger global rebalance/reboot so leave events
        are serialized by the single event-loop thread.
        """
        removed_any = False
        while True:
            try:
                node_id = self._pending_leaves.get_nowait()
            except queue.Empty:
                break
            try:
                self.leave(node_id)
                removed_any = True
            except Exception as exc:
                logger.warning(f"Leave failed for {node_id}: {exc}")

        # After draining all leaves, decide whether to do a single global rebalance.
        if not removed_any:
            return

        if not self.layer_allocator.should_global_rebalance():
            return

        nodes = self.node_manager.nodes
        logger.warning("Global rebalance triggered due to node leave")

        # Count manual vs automatic nodes
        manual_count = sum(1 for n in nodes if n.manual_layer_assignment)
        total_count = len(nodes)
        logger.debug(f"Node count: {manual_count} manual, {total_count - manual_count} automatic")
        if total_count == 0:
            logger.debug("No nodes left after leave(s); skipping global rebalance")
            return
        if manual_count == total_count:
            logger.debug("All nodes are manual assignment, skipping global rebalance")
            return
        if manual_count > 0:
            logger.error(
                f"Mixed assignment detected ({manual_count} manual, {total_count - manual_count} automatic); skipping rebalance"
            )
            return

        # Move active nodes to standby and re-bootstrap (reboot) once.
        self.node_manager.standby([n.node_id for n in self.node_manager.active_nodes])
        assert (
            self.node_manager.num_standby_nodes == self.node_manager.num_nodes
        ), "All active nodes should be moved to standby"
        assert self.node_manager.num_active_nodes == 0, "No active nodes before re-bootstrap"
        logger.warning("Re-bootstrapping for global rebalance")
        try:
            self.bootstrap(reboot=True)
        finally:
            # Ensure snapshot reflects post-rebalance state even if bootstrap fails.
            self.emit_alloc_log_snapshot(reason="after global rebalance")

    def stop(self) -> None:
        """Signal background threads to stop and wake any waiters."""
        self._stop_event.set()
        self._wake_event.set()
        with self._node_count_cv:
            self._node_count_cv.notify_all()

    def need_more_nodes(self):
        return (
            not self._bootstrapped_event.is_set()
            and self.node_manager.num_standby_nodes >= self.min_nodes_bootstrapping
        )
