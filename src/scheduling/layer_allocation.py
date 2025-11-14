"""
Layer allocation and dynamic rebalancing primitives (Phase 1 of scheduling).

Key components:
- LayerLoad: per-layer hosting power (combined KV memory + FLOPs) tracked in a min-heap;
- BaseLayerAllocator: shared utilities for static global_allocation(), dynamic join/leave, and
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

    Global rebalancing, i.e. re-`global_allocation` is needed when:
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
        nodes: List[Node],
        *,
        rebalance_threshold: float = 0.25,
        water_filling_max_iterations: int = 40,
        assign_left_over_nodes: bool = True,
    ) -> None:
        self.model_info = model_info
        self.num_total_layers = model_info.num_layers
        # Use the caller-provided list object to keep a single authoritative list.
        # Sort in-place to preserve shared reference with scheduler.
        self.nodes = nodes
        self.nodes.sort(key=lambda node: node.get_decoder_layer_capacity(), reverse=True)

        self.layer_to_load: Dict[int, LayerLoad] = {}
        self.node_id_to_node: Dict[str, Node] = {}
        # Sync dict with initial nodes; prevents declare() from adding duplicates
        # when allocate_left_over_nodes() processes unallocated nodes
        for node in self.nodes:
            self.node_id_to_node[node.node_id] = node

        # Pipeline endpoints for routing
        self.embedding_node_ids: List[str] = []
        self.lm_head_node_ids: List[str] = []
        # How much we value memory vs. FLOPs for hosting power (sum to 1)
        # Threshold for layer hosting power imbalance to trigger global rebalance
        self.rebalance_threshold = rebalance_threshold
        # Maximum number of iterations to run the water-filling algorithm
        self.water_filling_max_iterations = water_filling_max_iterations
        # Whether to assign left-over nodes using dynamic policy for
        # static allocation's leftover nodes
        self.assign_left_over_nodes = assign_left_over_nodes

        # Node allocation
        self.node_allocation: Dict[str, Tuple[int, int]] = {}

        # Heapify Layer Loads
        self.layer_loads_heap: List[LayerLoad] = []
        for layer_id in range(self.num_total_layers):
            layer_load = LayerLoad(layer_id=layer_id, current_kv_size=0)
            self.layer_to_load[layer_id] = layer_load
        self._update_layer_loads_heap()
        logger.debug(
            "Initialized LayerAllocator with %d nodes for %d total layers",
            len(self.nodes),
            self.num_total_layers,
        )

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the allocator."""
        return len(self.nodes)

    def validate_allocation(self, start_layer: int, end_layer: int):
        """Validate the allocation."""
        if start_layer < 0 or end_layer > self.num_total_layers:
            return False
        if start_layer >= end_layer:
            return False
        return True

    def global_allocation(self) -> bool:
        """Static assignment based on existing nodes. For cold-start and global rebalancing.

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
        self.node_id_to_node[node.node_id] = node
        node.set_layer_allocation(start_layer, end_layer)
        self.node_allocation[node.node_id] = (start_layer, end_layer)
        logger.debug("Allocated node %s to layers [%d, %d)", node.node_id, start_layer, end_layer)
        if start_layer == 0:
            self.embedding_node_ids.append(node.node_id)
        if end_layer == self.num_total_layers:
            self.lm_head_node_ids.append(node.node_id)
        for layer_id in range(start_layer, end_layer):
            if layer_id not in self.layer_to_load:
                raise ValueError(f"Layer {layer_id} not found in layer_to_load")
            self.layer_to_load[layer_id].add_node(node)
        self._update_layer_loads_heap()

    def deallocate(self, node: Node) -> None:
        """Deallocate a node from its assigned layers."""
        if node.start_layer is None or node.end_layer is None:
            logger.info("Node must have start_layer and end_layer")
            return
        start_layer, end_layer = node.start_layer, node.end_layer
        logger.debug(
            "Deallocating node %s from layers [%d, %d)", node.node_id, start_layer, end_layer
        )
        if node.node_id in self.node_allocation:
            del self.node_allocation[node.node_id]
        if node.node_id in self.embedding_node_ids:
            self.embedding_node_ids.remove(node.node_id)
        if node.node_id in self.lm_head_node_ids:
            self.lm_head_node_ids.remove(node.node_id)
        for layer_id in range(start_layer, end_layer):
            if layer_id in self.layer_to_load:
                self.layer_to_load[layer_id].remove_node(node)
        node.clear_layer_allocation()
        node.is_active = False
        self._update_layer_loads_heap()

    def reallocate(self, node: Node, start_layer: int, end_layer: int) -> None:
        """Reallocate a node to a specific layer range."""
        self.deallocate(node)
        self.allocate(node, start_layer, end_layer)

    def declare(self, node: Node) -> None:
        """Declare a node to the allocator."""
        if node.node_id not in self.node_id_to_node:
            self.nodes.append(node)
            self.node_id_to_node[node.node_id] = node
        # Keep order deterministic without rebinding the list reference
        self.nodes.sort(key=lambda node: node.get_decoder_layer_capacity(), reverse=True)
        logger.debug("Declared node %s (total declared: %d)", node.node_id, len(self.nodes))

    def join(self, node: Node) -> None:
        """Dynamically assign a new node based on lightest layers."""
        logger.debug("Joining node dynamically: %s", node.node_id)
        self.declare(node)
        lightest_layer = self.get_lightest_layer()
        logger.debug("Lightest layer: %s", lightest_layer)
        if lightest_layer is None:
            raise ValueError("No layers to assign")

        # Assign consecutive layers starting from the lightest layer
        start_layer = lightest_layer.layer_id
        # Greedily assign layers that the node can host
        end_layer = self._adjust_end_layer_for_tail(node, start_layer)
        logger.debug(
            "Dynamic assignment candidate for %s: start=%d end=%d",
            node.node_id,
            start_layer,
            end_layer,
        )
        self.allocate(node, start_layer, end_layer)

    def leave(self, node_id: str) -> None:
        """Dynamically remove a node, update layer loads and pipeline endpoints."""
        node = self.node_id_to_node.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in allocation")
        logger.debug("Node leaving allocator: %s", node_id)
        self.deallocate(node)
        del self.node_id_to_node[node_id]
        # Ensure the shared nodes list is updated
        for node in self.nodes:
            if node.node_id == node_id:
                self.nodes.remove(node)
                break

    def allocate_left_over_nodes(self) -> None:
        """Assign any nodes without allocations by treating them as dynamic joins.

        During bootstrapping or after a global allocation, some nodes may remain
        unassigned because they cannot contribute to a full pipeline. This method
        assigns such nodes one-by-one using the same policy as dynamic `join`:
        repeatedly host the lightest layers to improve replication and balance.
        """
        logger.debug("Allocating left-over nodes (unassigned after global allocation)")
        # Iterate in capacity order for determinism and better packing
        for node in sorted(self.nodes, key=lambda n: n.get_decoder_layer_capacity(), reverse=True):
            if node.node_id not in self.node_allocation:
                try:
                    logger.debug("Attempting left-over allocation for %s", node.node_id)
                    self.join(node)
                except Exception:
                    # Best-effort: if no layers can be assigned, skip
                    logger.debug(
                        "Left-over allocation skipped for %s (no assignable layers)", node.node_id
                    )
                    continue

    def should_global_rebalance(self) -> bool:
        """Trigger global rebalance, i.e. re-run `initialize`  if load imbalance is too high.

        The method calculates a combined, normalized load for each layer based
        on its memory and FLOPs usage relative to the total cluster capacity.
        It then computes the coefficient of variation (std_dev / mean) of these
        loads. If this value exceeds a configurable threshold, it indicates
        significant imbalance and returns True.
        """

        # If we don't currently have a full pipeline covering [0, L), force rebalance
        if not self.has_full_pipeline():
            return True

        layer_heap = self.layer_loads_heap
        if len(layer_heap) < 2:
            return False

        total_cluster_memory = sum(
            (node.hardware.num_gpus * node.hardware.memory_gb) for node in self.nodes
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
            "Global rebalance check: cv=%.4f threshold=%.4f -> %s",
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

    def list_node_allocations(self) -> List[Tuple[str, int, int]]:
        """List current per-node layer allocations as (node_id, start_layer, end_layer).

        Nodes without an allocation are omitted. Results are sorted by start_layer.
        """
        items = [(node_id, se[0], se[1]) for node_id, se in self.node_allocation.items()]
        items.sort(key=lambda x: (x[1], x[2], x[0]))
        return items

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

    def has_full_pipeline(self, active_only: bool = False) -> bool:
        """Return True if there exists at least one pipeline covering [0, num_total_layers).

        Checks whether we can chain contiguous node allocations starting at 0 to reach L.
        This requires that there exists at least one node starting at layer 0 and a chain
        of contiguous node ranges that reaches num_total_layers.
        """
        total_layers = self.num_total_layers

        # Build index of nodes by start_layer
        start_to_nodes: Dict[int, List[Node]] = {}
        for node_id, (s, e) in self.node_allocation.items():
            if s is None or e is None:
                continue
            node = self.node_id_to_node.get(node_id)
            if node is None or (active_only and not node.is_active):
                continue
            start_to_nodes.setdefault(s, []).append(node)

        # Must have at least one node starting at layer 0
        if not start_to_nodes.get(0):
            return False

        # DFS to check if we can reach total_layers from any head node
        def can_reach_target(current_end: int) -> bool:
            if current_end >= total_layers:
                return current_end == total_layers

            for nxt in start_to_nodes.get(current_end, []):
                if nxt.end_layer and nxt.end_layer > current_end:
                    if can_reach_target(nxt.end_layer):
                        return True
            return False

        return any(
            head.end_layer and can_reach_target(head.end_layer)
            for head in start_to_nodes.get(0, [])
        )

    def layer_replication_stats(self) -> Tuple[int, int, float]:
        """Return (min, max, avg) number of nodes hosting each layer.

        Counts the number of hosting nodes per layer from `layer_to_load` and
        aggregates basic statistics. If there are no layers, returns (0, 0, 0.0).
        """
        if not self.layer_to_load:
            return 0, 0, 0.0
        counts = [len(layer.hosting_nodes) for layer in self.layer_to_load.values()]
        if not counts:
            return 0, 0, 0.0
        min_hosts = min(counts)
        max_hosts = max(counts)
        avg_hosts = float(sum(counts)) / float(len(counts))
        return min_hosts, max_hosts, avg_hosts


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

    def init(
        self,
        *,
        look_ahead_enable: bool = True,
        pipeline_rebalance_strategy: Literal["greedy", "water_filling"] = "water_filling",
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
        self._look_ahead_enable = look_ahead_enable
        self._pipeline_rebalance_strategy = pipeline_rebalance_strategy

    def global_allocation(self) -> bool:
        """
        Allocate layers to nodes greedily to maximize the number of pipelines.

        Builds pipelines from the sorted nodes and uses `adjust_pipeline_layers`
        to assign contiguous layer ranges on each pipeline.
        """
        logger.debug(
            "[Greedy] Starting global_allocation with %d nodes for %d layers",
            len(self.nodes),
            self.model_info.num_layers,
        )
        num_total_layers = self.model_info.num_layers

        available_nodes = self.nodes.copy()
        for n in available_nodes:
            logger.warning(f"Node {n.node_id} has capacity {n.get_decoder_layer_capacity()}")
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

        if not any_assigned or not self.has_full_pipeline():
            logger.debug("[Greedy] global_allocation produced no full pipeline")
            return False
        # Assign any nodes that were left unallocated using dynamic policy
        if self.assign_left_over_nodes:
            logger.debug("[Greedy] Assigning left-over nodes")
            self.allocate_left_over_nodes()
        logger.debug("[Greedy] global_allocation completed successfully")
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
        model_info: ModelInfo,
        nodes: List[Node],
        alpha: float = 2.0,
        *,
        assign_left_over_nodes: bool = True,
        rebalance_threshold: float = 0.25,
        water_filling_max_iterations: int = 40,
    ) -> None:
        super().__init__(
            model_info,
            nodes,
            rebalance_threshold=rebalance_threshold,
            water_filling_max_iterations=water_filling_max_iterations,
        )
        # Sort GPUs by layer capacity descending for stronger pruning
        self.alpha = alpha
        self._path: Dict[Tuple[int, Tuple[int, ...], int], Tuple] = {}

    def global_allocation(self) -> bool:
        logger.debug(
            "[DP] Starting global_allocation with %d nodes for %d layers",
            len(self.nodes),
            self.model_info.num_layers,
        )
        num_nodes = len(self.nodes)
        num_layers = int(self.model_info.num_layers)
        total_cap = sum(node.get_decoder_layer_capacity() for node in self.nodes)

        if num_layers <= 0 or num_nodes == 0 or total_cap < num_layers:
            logger.warning(
                "[DP] Insufficient resources: nodes=%d, layers=%d, total_cap=%d",
                num_nodes,
                num_layers,
                total_cap,
            )
            return False
        else:
            logger.debug(
                "[DP] Sufficient resources: nodes=%d, layers=%d, total_cap=%d",
                num_nodes,
                num_layers,
                total_cap,
            )
        # used for pruning
        suffix_sum = [0] * (num_nodes + 1)
        for i in range(num_nodes - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + self.nodes[i].get_decoder_layer_capacity()

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
                    c_norm = self.nodes[i].get_decoder_layer_capacity()
                    r_after = rj - c_norm
                    if r_after <= 0:
                        # try closing with LM head allowance
                        c_close = self.nodes[i].get_decoder_layer_capacity(include_lm_head=True)
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
                    c_start = self.nodes[i].get_decoder_layer_capacity(include_input_embed=True)
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
            logger.debug("[DP] Could not find a feasible number of pipelines")
            return False
        self._path = best_path
        pipelines = self._backtrack(best_num_pipes, num_nodes)

        # Assign layers for each pipeline via in-place rebalancing
        for pl_nodes in pipelines:
            if not pl_nodes:
                continue
            logger.debug("[DP] Adjusting pipeline with %d nodes", len(pl_nodes))
            self.adjust_pipeline_layers(pl_nodes, assume_sorted=False)
        # Assign any nodes that were left unallocated using dynamic policy
        if self.assign_left_over_nodes:
            logger.debug("[DP] Assigning left-over nodes")
            self.allocate_left_over_nodes()
        if not self.has_full_pipeline():
            logger.debug("[DP] Allocation did not produce a full pipeline")
            return False
        logger.debug("[DP] global_allocation completed successfully")
        return True

    def _backtrack(self, best_num_pipes: int, num_nodes: int) -> List[List[Node]]:
        # Reconstruct pipelines
        logger.debug("[DP] Backtracking to construct %d pipelines", best_num_pipes)
        pipelines: List[List[Node]] = [[] for _ in range(best_num_pipes)]
        # (residual, nodes list)
        open_list: List[Tuple[int, List[Node]]] = []
        i = 0
        finished = 0
        while i < num_nodes and finished < best_num_pipes:
            open_tuple = tuple(sorted(r for r, _ in open_list))
            action = self._path.get((i, open_tuple, finished))
            if action is None:
                break
            kind = action[0]
            if kind in ("skip", "done"):
                i += 1
                continue
            node = self.nodes[i]
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
