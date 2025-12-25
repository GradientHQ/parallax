"""
Layer allocation and dynamic rebalancing primitives (Phase 1 of scheduling).

Key components:
- LayerLoad: per-layer hosting power (combined KV memory + FLOPs) tracked in a min-heap;
- BaseLayerAllocator: shared utilities for static allocate_from_standby(), dynamic join/leave, and
  in-place pipeline rebalancing using a water-filling algorithm;
- GreedyLayerAllocator: builds pipelines greedily to minimize stages and maximize the
  number of pipelines, then rebalances each pipeline in-place;
- DynamicProgrammingLayerAllocator: explores pipeline construction via DP to balance
  concurrency (pipelines) and latency (stages per pipeline), then rebalances each pipeline
  in-place.

All allocators assign contiguous layer ranges directly on `Node` instances and maintain
per-layer load state. Water-filling allocates decoder layers proportional to node compute
power (TFLOPS or memory bandwidth), respecting per-node parameter capacity and reserving
embedding/LM head endpoints on the first/last nodes of a pipeline.
"""

import heapq
from dataclasses import dataclass, field
from functools import lru_cache
from math import floor
from typing import Dict, List, Literal, Optional, Set, Tuple

from parallax_utils.logging_config import get_logger
from scheduling.model_info import ModelInfo
from scheduling.node import Node
from scheduling.node_management import NodeManager

logger = get_logger(__name__)


@dataclass
class LayerLoad:
    """Tracks the load and hosting power for a specific layer."""

    layer_id: int
    current_kv_size: int
    hosting_nodes: Set[str] = field(default_factory=set)

    def add_node(self, node: Node) -> None:
        """Add a node's contribution to this layer's load."""
        self.hosting_nodes.add(node.node_id)
        if node.per_decoder_layer_kv_cache_memory is None:
            raise ValueError("Node must have per_decoder_layer_kv_cache_memory")
        self.current_kv_size += node.per_decoder_layer_kv_cache_memory

    def remove_node(self, node: Node):
        """Remove a node from the layer load."""
        if node.node_id in self.hosting_nodes:
            self.hosting_nodes.remove(node.node_id)
            if node.per_decoder_layer_kv_cache_memory is None:
                raise ValueError("Node must have per_decoder_layer_kv_cache_memory")

            self.current_kv_size -= node.per_decoder_layer_kv_cache_memory

    def __lt__(self, other):
        """For heap ordering: prioritize layers with lower hosting power.

        Primary: less memory (less hosting power)
        Secondary: lower layer ID (less layers hosted)
        """
        return (self.current_kv_size, self.layer_id) < (
            other.current_kv_size,
            other.layer_id,
        )


class BaseLayerAllocator:
    """Base class for layer allocation and rebalancing.

    There are two types of allocations:
    1. Static: initialization, global rebalancing.
       - happens globally, not per-node;
       - more optimal since it considers the entire cluster;
    2. Dynamic: nodes are joining / leaving dynamically
       - assignment done per-node based on layer load heap;
       - less optimal compared to static allocation (global knowledge);
       - necessary because we don't want to
         - re-start the whole server to
         - each worker node gives up running requests
         - each worker node re-load a different shard.

    Global rebalancing, i.e. re-`allocate_from_standby` is needed when:
     - When we find some layers are not hosted by any node,
     - Loads are too imbalanced.

    Pipeline: nodes hosting the first layer to the last layer (inclusive).
    In-place adjustment:
        After each pipeline is formed, the layers within
        it are immediately adjusted to be proportional to the computing power
        (TFLOPS or bandwidth) of the nodes forming the pipeline. Capped by capacity.
        This ensures that the stages are balanced and system throughput is maximized.
        We use water-filling algorithm.
    """

    def __init__(
        self,
        model_info: ModelInfo,
        node_management: NodeManager,
        *,
        dynamic_pipelines_router: bool = False,
        rebalance_threshold: float = 0.25,
        water_filling_max_iterations: int = 40,
        trim_layers_on_turning_points: bool = True,
    ) -> None:
        self.model_info = model_info
        self.num_total_layers = model_info.num_layers
        self.node_management = node_management
        self.layer_to_load: Dict[int, LayerLoad] = {}

        # True if we should trim layers on turning points
        # e.g. 4 layers, 2 nodes, [0, 3), [2,4) -> [0, 3), [3, 4);
        # where we trim the second node.
        self.trim_layers_on_turning_points = trim_layers_on_turning_points

        # Threshold for layer hosting power imbalance to trigger global rebalance
        self.rebalance_threshold = rebalance_threshold
        # Maximum number of iterations to run the water-filling algorithm
        self.water_filling_max_iterations = water_filling_max_iterations

        # True if using DP request router, False for fixed pipelines
        self.dynamic_pipelines_router = dynamic_pipelines_router

        # Heapify Layer Loads
        self.layer_loads_heap: List[LayerLoad] = []
        for layer_id in range(self.num_total_layers):
            layer_load = LayerLoad(layer_id=layer_id, current_kv_size=0)
            self.layer_to_load[layer_id] = layer_load
        self._update_layer_loads_heap()

    def _validate_allocation(self, start_layer: int, end_layer: int):
        """Validate the allocation."""
        if start_layer < 0 or end_layer > self.num_total_layers:
            return False
        if start_layer >= end_layer:
            return False
        return True

    def allocate_from_standby(self) -> bool:
        """Static assignment on STANDBY nodes.

        Returns:
            True if at least one full pipeline (covering [0, num_total_layers)) was allocated.
        """
        return False

    def allocate(self, node: Node, start_layer: int, end_layer: int) -> None:
        """Allocate a node to a specific layer range."""
        if end_layer <= start_layer:
            raise ValueError(
                f"Invalid allocation: start_layer {start_layer} >= end_layer {end_layer}"
            )
        node.set_layer_allocation(start_layer, end_layer)
        self.node_management.activate([node.node_id])
        logger.debug(
            "[LayerAllocator] Allocated node %s to layers [%d, %d)",
            node.node_id,
            start_layer,
            end_layer,
        )
        for layer_id in range(start_layer, end_layer):
            if layer_id not in self.layer_to_load:
                raise ValueError(f"Layer {layer_id} not found in layer_to_load")
            self.layer_to_load[layer_id].add_node(node)
        self._update_layer_loads_heap()

    def deallocate(self, node: Node) -> None:
        """Deallocate a node from its assigned layers."""
        if node.start_layer is None or node.end_layer is None:
            logger.info("[LayerAllocator] Node must have start_layer and end_layer")
            return
        start_layer, end_layer = node.start_layer, node.end_layer
        logger.debug(
            "[LayerAllocator] Deallocating node %s from layers [%d, %d)",
            node.node_id,
            start_layer,
            end_layer,
        )
        for layer_id in range(start_layer, end_layer):
            if layer_id in self.layer_to_load:
                self.layer_to_load[layer_id].remove_node(node)
        self.node_management.standby([node.node_id])
        self._update_layer_loads_heap()

    def reallocate(self, node: Node, start_layer: int, end_layer: int) -> None:
        """Reallocate a node to a specific layer range."""
        self.deallocate(node)
        self.allocate(node, start_layer, end_layer)

    def dynamic_join(self, node: Node) -> None:
        """In case of using Dynamic Programming request router, a node is joined dynamically to the lightest layers."""
        lightest_layer = self.get_lightest_layer()
        logger.info(
            "[LayerAllocator] Joining node %s with the lightest layer %d",
            node.node_id,
            lightest_layer.layer_id,
        )
        if lightest_layer is None:
            raise ValueError("No layers to assign")

        # Assign consecutive layers starting from the lightest layer
        start_layer = lightest_layer.layer_id
        # Greedily assign layers that the node can host
        end_layer = self._adjust_end_layer_for_tail(node, start_layer)
        logger.info(
            "[LayerAllocator] Dynamic assignment candidate for %s: start=%d end=%d",
            node.node_id,
            start_layer,
            end_layer,
        )
        self.allocate(node, start_layer, end_layer)

    def allocate_standby_nodes(self) -> bool:
        """In case of enabling dynamic pipelines, allocate left-over nodes to the lightest layers using dynamic join."""
        if self.dynamic_pipelines_router:
            left_over_nodes = self.node_management.standby_nodes
            for node in left_over_nodes:
                self.dynamic_join(node)
            return True
        else:
            return self.allocate_from_standby()

    def should_global_rebalance(self) -> bool:
        """Trigger global rebalance, i.e. re-run `initialize`  if load imbalance is too high.

        The method calculates a combined, normalized load for each layer based
        on its memory and FLOPs usage relative to the total cluster capacity.
        It then computes the coefficient of variation (std_dev / mean) of these
        loads. If this value exceeds a configurable threshold, it indicates
        significant imbalance and returns True.
        """

        # If we don't currently have a full pipeline covering [0, L), force rebalance
        if not self.node_management.has_full_pipeline(self.num_total_layers):
            return True

        # TODO: add more imbalance checks

        available_nodes = self.node_management.nodes

        layer_heap = self.layer_loads_heap
        if len(layer_heap) < 2:
            return False

        total_cluster_memory = sum(
            (node.hardware.num_gpus * node.hardware.memory_gb) for node in available_nodes
        )

        if total_cluster_memory == 0:
            raise ValueError("Total cluster memory is zero")

        loads = [layer.current_kv_size / total_cluster_memory for layer in layer_heap]

        if not loads:
            return False

        avg_load = sum(loads) / len(loads)
        if avg_load == 0:
            return False

        variance = sum((load - avg_load) ** 2 for load in loads) / len(loads)
        std_dev = variance**0.5

        coefficient_of_variation = std_dev / avg_load

        decision = coefficient_of_variation > self.rebalance_threshold
        logger.debug(
            "[LayerAllocator] Global rebalance check: cv=%.4f threshold=%.4f -> %s",
            coefficient_of_variation,
            self.rebalance_threshold,
            decision,
        )
        return decision

    def adjust_pipeline_layers(
        self,
        pipeline_nodes: List[Node],
        assume_sorted: bool = False,
        power_type: Literal["flops", "bandwidth"] = "flops",
    ) -> None:
        """Rebalance a single pipeline in-place using water-filling and set allocations on nodes.

        Adjusts `start_layer`/`end_layer` on the given `pipeline_nodes` so that:
        - Decoder layers are split proportional to node compute (TFLOPS or bandwidth),
          capped by each node's parameter capacity;
        - The first node reserves input embedding capacity; the last reserves LM head;
        - Assigned stages are contiguous from layer 0 to the final layer.

        Args:
            pipeline_nodes: Nodes that form a full pipeline from embedding to LM head.
            assume_sorted: If True, nodes are assumed sorted by decoder-layer capacity (desc).
            power_type: Type of compute power to use for rebalancing.

        Returns:
            None. Nodes are updated in-place via `set_layer_allocation`.
        """

        total_layers = self.num_total_layers
        if not pipeline_nodes or total_layers <= 0:
            raise ValueError("No nodes or total layers is non-positive")

        nodes = (
            pipeline_nodes
            if assume_sorted
            else sorted(pipeline_nodes, key=lambda n: n.get_decoder_layer_capacity(), reverse=True)
        )
        n = len(nodes)

        # Clear previous allocations for participating nodes (avoid double counting loads)
        for node in nodes:
            if node.start_layer is not None and node.end_layer is not None:
                self.deallocate(node)

        caps: List[int] = []
        compute_powers: List[float] = []
        for i, node in enumerate(nodes):
            if i == 0:
                cap = node.get_decoder_layer_capacity(include_input_embed=True)
            elif i == n - 1:
                cap = node.get_decoder_layer_capacity(include_lm_head=True)
            else:
                cap = node.get_decoder_layer_capacity()
            if cap <= 0:
                raise ValueError(f"Node {node.node_id} has non-positive capacity: {cap}")
            caps.append(cap)
            compute_powers.append(
                node.hardware.tflops_fp16
                if power_type == "flops"
                else node.hardware.memory_bandwidth_gbps
            )

        if sum(caps) < total_layers:
            raise ValueError(f"Total capacity {sum(caps)} is less than total layers {total_layers}")

        # Water-filling: find lambda s.t. sum_i min(c_i, λ F_i) == L
        def total_at(lmbd: float) -> float:
            return sum(min(caps[i], lmbd * compute_powers[i]) for i in range(n))

        lo, hi = 0.0, max((caps[i] / compute_powers[i]) for i in range(n))
        for _ in range(self.water_filling_max_iterations):
            mid = 0.5 * (lo + hi)
            if total_at(mid) >= total_layers:
                hi = mid
            else:
                lo = mid
        lam = hi

        target = [min(caps[i], lam * compute_powers[i]) for i in range(n)]

        # Integerization: floor + largest remainders (respect caps)
        stage_layer_counts = [min(caps[i], int(floor(target[i]))) for i in range(n)]
        assigned = sum(stage_layer_counts)
        remaining = total_layers - assigned
        if remaining > 0:
            frac = [(target[i] - stage_layer_counts[i], -i) for i in range(n)]
            for _, negi in sorted(frac, reverse=True):
                i = -negi
                if remaining == 0:
                    break
                room = caps[i] - stage_layer_counts[i]
                if room > 0:
                    stage_layer_counts[i] += 1
                    remaining -= 1
        elif remaining < 0:
            raise ValueError(f"Remaining {remaining} is negative")

        # Final clamp (safety) and residual distribute, if any
        extra = 0
        for i in range(n):
            if stage_layer_counts[i] > caps[i]:
                extra += stage_layer_counts[i] - caps[i]
                stage_layer_counts[i] = caps[i]
        if extra > 0:
            for i in range(n):
                if extra == 0:
                    break
                room = caps[i] - stage_layer_counts[i]
                take = min(room, extra)
                stage_layer_counts[i] += take
                extra -= take

        # Apply contiguous assignments in stage order directly to nodes
        start_layer = 0
        for idx, node in enumerate(nodes):
            layers = stage_layer_counts[idx]
            if layers <= 0:
                # TODO(chris-t): should we deallocate the node?
                continue
            end_layer = start_layer + layers
            self.allocate(node, start_layer, end_layer)
            start_layer = end_layer

        # Sanity check: ensure coverage from 0..num_total_layers
        if start_layer != total_layers:
            raise ValueError(
                f"Assignment did not cover all layers: assigned {start_layer} of {total_layers}"
            )

    def adjust_pipeline_layers_greedy(self, pipeline_nodes: List[Node]) -> None:
        """Greedily assign contiguous layers to `pipeline_nodes` from 0 to L.

        This simpler alternative to `adjust_pipeline_layers` walks nodes in the given
        order and assigns as many layers as each node can host based on its capacity.
        The first node includes input embedding allowance; the final closing node
        includes LM head allowance. Previous allocations for participating nodes are
        cleared to avoid double counting load.

        Args:
            pipeline_nodes: Nodes that form a full pipeline in stage order.

        Raises:
            ValueError: If total capacity is insufficient to cover all layers.
        """

        total_layers = self.num_total_layers
        if not pipeline_nodes or total_layers <= 0:
            raise ValueError("No nodes or total layers is non-positive")

        # Clear previous allocations for participating nodes (avoid double counting loads)
        for node in pipeline_nodes:
            if node.start_layer is not None and node.end_layer is not None:
                self.deallocate(node)

        start_layer = 0
        remaining_layers = total_layers

        for idx, node in enumerate(pipeline_nodes):
            include_input_embed = start_layer == 0

            # Base capacity without LM head
            base_cap = node.get_decoder_layer_capacity(include_input_embed=include_input_embed)

            # If this node will be the tail that closes the pipeline, allow LM head
            if base_cap >= remaining_layers:
                tail_cap = node.get_decoder_layer_capacity(
                    include_input_embed=include_input_embed, include_lm_head=True
                )
                assign_layers = min(tail_cap, remaining_layers)
            else:
                assign_layers = min(base_cap, remaining_layers)

            if assign_layers <= 0:
                continue

            end_layer = start_layer + assign_layers
            self.allocate(node, start_layer, end_layer)
            start_layer = end_layer
            remaining_layers -= assign_layers

            if remaining_layers == 0:
                break

        if remaining_layers != 0:
            raise ValueError(
                f"Greedy assignment did not cover all layers: remaining {remaining_layers}"
            )

    def adjust_for_turning_points(self, num_layers: int) -> List[Tuple[str, int, str]]:
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
        nodes = self.node_management.active_nodes
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
                self.reallocate(nodes[prev_i], prev_node.start_layer, l)
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
                self.reallocate(n, l0, n.end_layer)
        return turning

    def get_lightest_layer(self) -> Optional[LayerLoad]:
        """Return the current lightest-hosted layer from the heap, if any."""
        if not self.layer_loads_heap:
            return None
        return self.layer_loads_heap[0]

    def _update_layer_loads_heap(self):
        """Rebuild the layer loads heap."""
        self.layer_loads_heap = list(self.layer_to_load.values())
        heapq.heapify(self.layer_loads_heap)

    def _adjust_end_layer_for_tail(self, node: Node, proposed_start_layer: int) -> int:
        """Adjust the number of layers to host for tail nodes."""
        include_input_embed = proposed_start_layer == 0
        node_capacity = node.get_decoder_layer_capacity(include_input_embed=include_input_embed)
        end_layer = min(proposed_start_layer + node_capacity, self.num_total_layers)
        if end_layer == self.num_total_layers:
            adjusted_capacity = node.get_decoder_layer_capacity(
                include_lm_head=True, include_input_embed=include_input_embed
            )
            end_layer = min(proposed_start_layer + adjusted_capacity, self.num_total_layers)

        return end_layer


class GreedyLayerAllocator(BaseLayerAllocator):
    """
    Greedy layer allocator that assigns layers to nodes trying to
    1. Minimize number of stages in each pipeline;
    2. Maximize number of pipelines.

    The algorithm proceeds as follows:
    1.  Initialization: Nodes are sorted by their layer capacity in descending
        order, so nodes with more memory are considered first.

    2.  Iterative Pipeline Construction: The allocator attempts to build one
        pipeline at a time. It iterates through the sorted nodes, adding them
        to the current pipeline until the total number of layers required by
        the model is met. Greedy in the sense that it always assigns the max
        layers allowed by capacity of the current node to the pipeline.
        So to minimize number of stages.

    3.  Look-Ahead Optimization: Before selecting a node, the algorithm "looks ahead"
        to see if the remaining nodes have enough capacity to form at least one more
        full pipeline.
        - If so, it chooses the *smallest* node that can complete the current
          pipeline. This preserves the larger nodes for future pipelines.
        - If not, it falls back to the default greedy behavior and picks the
          *largest* available node to complete the pipeline as efficiently as
          possible.
        So to maximize number of pipelines.

    4.  In-Place Rebalancing: After each pipeline is formed, the layers within
        it are immediately rebalanced to be proportional to the computing power
        (TFLOPS) of the nodes. Capped by capacity. This ensures that the stages
        are balanced and system throughput is maximized.
        We use water-filling algorithm.


    This process repeats until no more complete pipelines can be formed from the
    remaining nodes.
    """

    def __init__(
        self,
        *,
        look_ahead_enable: bool = True,
        pipeline_rebalance_strategy: Literal["greedy", "water_filling"] = "water_filling",
        **kwargs,
    ) -> None:
        """Initialize Greedy allocator runtime knobs.

        Args:
            look_ahead_enable: Toggle the look-ahead optimization when selecting
                the tail node to close a pipeline.
            pipeline_rebalance_strategy: Strategy for in-place per-pipeline
                rebalancing after a pipeline is formed. "greedy" assigns
                contiguous layers in capacity order; "water_filling" uses
                proportional water-filling based on compute power.

        Returns:
            None. Sets internal flags `_look_ahead_enable` and
            `_pipeline_rebalance_strategy`.
        """
        super().__init__(**kwargs)
        self._look_ahead_enable = look_ahead_enable
        self._pipeline_rebalance_strategy = pipeline_rebalance_strategy

    def allocate_from_standby(self) -> bool:
        """
        Allocate layers to nodes greedily to maximize the number of pipelines.

        Builds pipelines from the sorted nodes and uses `adjust_pipeline_layers`
        to assign contiguous layer ranges on each pipeline.
        """
        num_total_layers = self.model_info.num_layers

        available_nodes = self.node_management.standby_nodes
        logger.info(
            "[Greedy LayerAllocator] Starting allocate_from_standby with %d nodes for %d layers",
            len(available_nodes),
            num_total_layers,
        )

        for n in available_nodes:
            logger.info(
                f"[Greedy LayerAllocator] Node {n.node_id} has capacity {n.get_decoder_layer_capacity()}"
            )
        any_assigned = False

        # Read runtime knobs with sensible defaults if `init` wasn't called
        look_ahead_enabled = getattr(self, "_look_ahead_enable", True)
        rebalance_strategy = getattr(self, "_pipeline_rebalance_strategy", "water_filling")

        while available_nodes:
            total_remaining_capacity = sum(
                node.get_decoder_layer_capacity() for node in available_nodes
            )
            if total_remaining_capacity < num_total_layers:
                logger.debug(
                    "[Greedy] Remaining capacity %d < total layers %d; stop",
                    total_remaining_capacity,
                    num_total_layers,
                )
                break

            pipeline_nodes: List[Node] = []
            remaining_layers = num_total_layers
            current_pipeline_total_capacity = total_remaining_capacity

            while remaining_layers > 0 and available_nodes:
                is_start = len(pipeline_nodes) == 0
                # Look-ahead optimization (only for picking the last node to finish a pipeline)
                look_ahead_possible = (
                    look_ahead_enabled
                    and is_start
                    and current_pipeline_total_capacity - remaining_layers >= num_total_layers + 1
                )
                best_fit_idx = -1
                if look_ahead_possible:
                    # Find smallest node that can complete the pipeline while leaving enough for another full pipeline
                    for i, node in enumerate(available_nodes):
                        node_i_capacity = node.get_decoder_layer_capacity(include_lm_head=True)
                        if node_i_capacity >= remaining_layers:
                            remaining_nodes_capacity = (
                                current_pipeline_total_capacity - node_i_capacity
                            )
                            if remaining_nodes_capacity >= num_total_layers:
                                best_fit_idx = (
                                    i  # choose the last matching (smallest due to sorting)
                                )

                node_to_add = (
                    available_nodes.pop(best_fit_idx)
                    if best_fit_idx != -1
                    else available_nodes.pop(0)
                )

                pipeline_nodes.append(node_to_add)
                # Update running totals with appropriate capacity at this position
                node_capacity = node_to_add.get_decoder_layer_capacity(include_input_embed=is_start)
                remaining_layers -= node_capacity
                if remaining_layers <= 0:
                    # Tail node can include LM head allowance
                    remaining_layers += node_capacity
                    node_capacity = node_to_add.get_decoder_layer_capacity(
                        include_input_embed=is_start, include_lm_head=True
                    )
                    remaining_layers -= node_capacity

                current_pipeline_total_capacity -= node_capacity

            if remaining_layers <= 0 and pipeline_nodes:
                # Assign layers within this pipeline in-place
                logger.debug(
                    "[Greedy] Built pipeline with %d nodes; adjusting layers",
                    len(pipeline_nodes),
                )
                if rebalance_strategy == "greedy":
                    self.adjust_pipeline_layers_greedy(pipeline_nodes)
                else:
                    self.adjust_pipeline_layers(pipeline_nodes, assume_sorted=False)
                any_assigned = True
            else:
                # Cannot form a complete pipeline with remaining nodes
                logger.debug("[Greedy] Unable to form complete pipeline; stopping")
                break

        if not any_assigned or not self.node_management.has_full_pipeline(self.num_total_layers):
            logger.warning("[Greedy] allocate_from_standby produced no full pipeline")
            return False
        if self.trim_layers_on_turning_points:
            turning_points = self.adjust_for_turning_points(self.num_total_layers)
            logger.debug(f"Turning points: {turning_points}")
        if self.dynamic_pipelines_router:
            logger.info("[Greedy] Allocating standby nodes using Dynamic Join (lightest layers)")
            self.allocate_standby_nodes()
        logger.info("[Greedy] allocate_from_standby completed successfully")
        return True


class DynamicProgrammingLayerAllocator(BaseLayerAllocator):
    """
    Dynamic programming based allocator that balances two objectives:
     - Concurrency (number of pipelines)
     - Latency (number of stages per pipeline).

    Why DP (vs greedy): interleaving constructions of multiple pipelines
    so cases like capacities (40, 40, 20, 20, 10, 10) with total 70 layers
    yields (40, 20, 10) + (40, 20, 10) instead of a single 2-stage pipe line (40 + 30).

    We want to maximize a simple scalar objective function:
        Z(k) = k / (s*(k) / k) = k^2 / s*(k)
    where:
        k: number of pipelines
        s*(k): minimum total number of stages realizing k pipelines

    DP State:
        dp(i, open_residuals, finshed_pipes) := min total stages needed using GPUs with index >= i;
        where:
            i: GPU index, in [0, N]
            open_residuals: sorted tuple of remaining layers for all open pipelines (values in 1..L-1)
            finished_pipes: number of already fully closed pipelines
        Transitions (for node i with capacity c_i):
            1. Skip node: dp(i + 1, open_residuals, finished_pipes)
            2. Assign to an existing open pipeline j:
               r' = r_j - c_i. If r' <= 0, try closing with LM head; if still <= 0 -> close (remove j, finished+1),
               else keep open with updated residual r'.
            3. Start a new pipeline (if finished + len(open_residuals) < k_target):
               r = L - c_i (with input embedding). If r <= 0, it closes immediately (finished+1), else append r.

    Finally:
        Compute objective Z(k) = (k**alpha) / (T_comp + (total_stages/k)*r_RTT)
        Choose the best k and backtrack to recover assignments
    """

    def __init__(
        self,
        *,
        alpha: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Sort GPUs by layer capacity descending for stronger pruning
        self.alpha = alpha
        self._path: Dict[Tuple[int, Tuple[int, ...], int], Tuple] = {}

    def allocate_from_standby(self) -> bool:
        """
        Allocate nodes in STANDBY pool using dynamic programming.
        """
        num_layers = self.model_info.num_layers

        available_nodes = self.node_management.standby_nodes
        logger.info(
            "[DPLayerAllocator] Starting allocate_from_standby with %d nodes for %d layers",
            len(available_nodes),
            num_layers,
        )
        num_nodes = len(available_nodes)
        total_cap = sum(node.get_decoder_layer_capacity() for node in available_nodes)

        if num_layers <= 0 or num_nodes == 0 or total_cap < num_layers:
            logger.warning(
                "[DPLayerAllocator] Insufficient resources: nodes=%d, layers=%d, total_cap=%d",
                num_nodes,
                num_layers,
                total_cap,
            )
            return False
        else:
            logger.info(
                "[DPLayerAllocator] Sufficient resources: nodes=%d, layers=%d, total_cap=%d",
                num_nodes,
                num_layers,
                total_cap,
            )
        # used for pruning
        suffix_sum = [0] * (num_nodes + 1)
        for i in range(num_nodes - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + available_nodes[i].get_decoder_layer_capacity()

        max_num_pipes = min(num_nodes, total_cap // num_layers)
        best_num_pipes = 0
        best_score: float = float("-inf")

        best_path: Dict[Tuple[int, Tuple[int, ...], int], Tuple] = {}
        for k_target in range(1, max_num_pipes + 1):
            path: Dict[Tuple[int, Tuple[int, ...], int], Tuple] = {}

            @lru_cache(maxsize=None)
            def dp(i: int, open_residuals: Tuple[int, ...], finished_pipes: int) -> int:
                # Completed target with no open pipelines
                if finished_pipes == k_target and len(open_residuals) == 0:
                    path[(i, open_residuals, finished_pipes)] = ("done",)
                    return 0
                if i == num_nodes:
                    return float("inf")

                new_needed = k_target - finished_pipes - len(open_residuals)
                need_open = sum(open_residuals)
                remaining_cap = suffix_sum[i]

                # Pruning
                # 1. already have more (finished + open) than target;
                # 2. remaining capacity is not enough to close ongoing pipelines
                #    and unfulfilled new pipelines
                # 3. remaining nodes are not enough to fulfill new pipelines
                if (
                    new_needed < 0
                    or remaining_cap < need_open + max(0, new_needed) * num_layers
                    or finished_pipes + len(open_residuals) + (num_nodes - i) < k_target
                ):
                    return float("inf")

                # Option 1: Skip this node
                best_cost = dp(i + 1, open_residuals, finished_pipes)
                best_action: Tuple = ("skip",)

                # Option 2: Assign to existing open pipeline
                for j, rj in enumerate(open_residuals):
                    c_norm = available_nodes[i].get_decoder_layer_capacity()
                    r_after = rj - c_norm
                    if r_after <= 0:
                        # try closing with LM head allowance
                        c_close = available_nodes[i].get_decoder_layer_capacity(
                            include_lm_head=True
                        )
                        r_after_close = rj - c_close
                        if r_after_close <= 0:
                            new_open = list(open_residuals)
                            new_open.pop(j)
                            cost = 1 + dp(i + 1, tuple(new_open), finished_pipes + 1)
                            if cost < best_cost:
                                best_cost = cost
                                best_action = ("assign", j, True)
                        else:
                            new_open = list(open_residuals)
                            new_open[j] = r_after_close
                            new_open.sort()
                            cost = 1 + dp(i + 1, tuple(new_open), finished_pipes)
                            if cost < best_cost:
                                best_cost = cost
                                best_action = ("assign", j, False)
                    else:
                        new_open = list(open_residuals)
                        new_open[j] = r_after
                        new_open.sort()
                        cost = 1 + dp(i + 1, tuple(new_open), finished_pipes)
                        if cost < best_cost:
                            best_cost = cost
                            best_action = ("assign", j, False)

                # Option 3: start a new pipeline (if we still need more)
                if new_needed > 0:
                    c_start = available_nodes[i].get_decoder_layer_capacity(
                        include_input_embed=True
                    )
                    r_new = num_layers - c_start
                    if r_new <= 0:
                        cost = 1 + dp(i + 1, open_residuals, finished_pipes + 1)
                        if cost < best_cost:
                            best_cost = cost
                            best_action = ("start", 0, True)
                    else:
                        new_open = list(open_residuals) + [r_new]
                        new_open.sort()
                        cost = 1 + dp(i + 1, tuple(new_open), finished_pipes)
                        if cost < best_cost:
                            best_cost = cost
                            best_action = ("start", r_new, False)

                path[(i, open_residuals, finished_pipes)] = best_action
                return best_cost

            s_star = dp(0, tuple(), 0)
            if s_star < float("inf"):
                score = (k_target * k_target) / s_star  # Z(k) = k^2 / s*(k)
                if score > best_score:
                    best_score, best_num_pipes = score, k_target
                    best_path = dict(path)

        if best_num_pipes is None or best_num_pipes == 0:
            logger.debug("[DPLayerAllocator] Could not find a feasible number of pipelines")
            return False
        self._path = best_path
        pipelines = self._backtrack(best_num_pipes, available_nodes)

        # Assign layers for each pipeline via in-place rebalancing
        for pl_nodes in pipelines:
            if not pl_nodes:
                continue
            logger.debug("[DPLayerAllocator] Adjusting pipeline with %d nodes", len(pl_nodes))
            self.adjust_pipeline_layers(pl_nodes, assume_sorted=False)
        if not self.node_management.has_full_pipeline(self.num_total_layers):
            logger.warning("[DPLayerAllocator] Allocation did not produce a full pipeline")
            return False
        if self.trim_layers_on_turning_points:
            turning_points = self.adjust_for_turning_points(self.num_total_layers)
            logger.debug(f"[DPLayerAllocator] Turning points: {turning_points}")
        if self.dynamic_pipelines_router:
            logger.info(
                "[DPLayerAllocator] Allocating standby nodes using Dynamic Join (lightest layers)"
            )
            self.allocate_standby_nodes()
        logger.info("[DPLayerAllocator] allocate_from_standby completed successfully")
        return True

    def _backtrack(self, best_num_pipes: int, available_nodes: List[Node]) -> List[List[Node]]:
        # Reconstruct pipelines
        logger.debug("[DPLayerAllocator] Backtracking to construct %d pipelines", best_num_pipes)
        pipelines: List[List[Node]] = [[] for _ in range(best_num_pipes)]
        # (residual, nodes list)
        open_list: List[Tuple[int, List[Node]]] = []
        i = 0
        finished = 0
        num_nodes = len(available_nodes)
        while i < num_nodes and finished < best_num_pipes:
            open_tuple = tuple(sorted(r for r, _ in open_list))
            action = self._path.get((i, open_tuple, finished))
            if action is None:
                break
            kind = action[0]
            if kind in ("skip", "done"):
                i += 1
                continue
            node = available_nodes[i]
            if kind == "assign":
                j, closed = action[1], action[2]
                # ensure open_list sorted like open_tuple
                open_list.sort(key=lambda x: x[0])
                rj, nodes_seq = open_list[j]
                c_norm = node.get_decoder_layer_capacity()
                r_after = rj - c_norm
                if r_after <= 0:
                    c_close = node.get_decoder_layer_capacity(include_lm_head=True)
                    r_after = rj - c_close
                nodes_seq.append(node)
                if r_after <= 0 or closed:
                    # pipeline closes here
                    pipelines[finished].extend(nodes_seq)
                    open_list.pop(j)
                    finished += 1
                else:
                    open_list[j] = (r_after, nodes_seq)
                i += 1
            elif kind == "start":
                r_new, closed = action[1], action[2]
                if closed:
                    pipelines[finished].append(node)
                    finished += 1
                else:
                    open_list.append((r_new, [node]))
                i += 1
            else:
                # Unknown action; advance to avoid infinite loop
                i += 1

        return pipelines
