## Scheduling subsystem

This directory implements a two-phase scheduler for distributed LLM inference:

### Phase 1 — Layer allocation

Assign contiguous decoder layer ranges to nodes and rebalance in place, as illustrated below:

<img width="1874" height="852" alt="parallax_1" src="https://github.com/user-attachments/assets/c57cde77-0cda-48fc-b1ad-6d4aa1b1787b" />

### Phase 2 — Request routing

Compute an end-to-end, minimum-latency path across the assigned node ranges, as illustrated below:

<img width="1828" height="705" alt="parallax_2" src="https://github.com/user-attachments/assets/8a6b4d8f-8d97-402b-ba84-3ce61e4ee313" />

The main entrypoint is `scheduling.scheduler.Scheduler`, which orchestrates allocation, dynamic joins/leaves, health checks, and routing.

### Key concepts
- **`ModelInfo`**: model-level shapes and per-layer cost/IO; used by capacity and latency estimators.
- **`NodeHardwareInfo`**: static hardware facts (TFLOPS, memory size/bandwidth).
- **`Node`**: live worker state. Tracks allocated `[start_layer, end_layer)` range, load (`current_requests`), RTTs to peers, and exposes helpers:
  - `get_decoder_layer_capacity(...)`: parameter-memory-bounded layer capacity.
  - `layer_latency_ms`: effective per-node latency (overload-aware, roofline fallback).
  - `hosts_layer(layer_id)`; allocators provide `has_full_pipeline()` across nodes.
- **Pipeline**: a chain of nodes whose ranges cover `[0, L)` without gaps (L = `ModelInfo.num_layers`).

## Phase 1: Layer allocation

Implemented in `scheduling.layer_allocation`.

- **`BaseLayerAllocator`**: shared utilities and state
  - Tracks per-layer hosting power via `LayerLoad` (min-heap of KV cache hosting power).
  - Assign/deassign helpers: `allocate(...)`, `deallocate(...)`, `declare(...)`.
  - Dynamic events: `join(node)`, `leave(node_id)`; best-effort placement on lightest layers.
  - Global health: `has_full_pipeline()`, `layer_replication_stats()` and `should_global_rebalance()`.
  - In-place rebalancing: `adjust_pipeline_layers(pipeline_nodes, power_type)` uses a water-filling algorithm to split decoder layers proportional to compute/bandwidth while respecting per-node capacity. Endpoints (embedding/LM head) are reserved on the first/last nodes.
  - Leftovers: `allocate_left_over_nodes()` uses the dynamic policy to replicate lightest layers when full pipelines are already formed.

- **`GreedyLayerAllocator`**
  - Goal: maximize number of pipelines while minimizing stages per pipeline.
  - Strategy: iteratively build pipelines using capacity-sorted nodes with a simple look-ahead; after each pipeline, call `adjust_pipeline_layers`.
  - API: `global_allocation() -> bool` returns whether at least one full pipeline exists; unassigned nodes are handled by `allocate_left_over_nodes()`.

- **`DynamicProgrammingLayerAllocator`**
  - Goal: jointly optimize concurrency (pipelines) and latency (stages) via DP.
  - Scoring: chooses `k` to maximize `k^2 / s*(k)`, where `s*(k)` = minimal total stages to realize `k` pipelines.
  - Produces disjoint pipelines and then rebalances each via `adjust_pipeline_layers`.

### Important knobs
- **`rebalance_threshold`**: coefficient-of-variation threshold of layer loads to trigger global rebalance.
- **`water_filling_max_iterations`**: iterations for the water-filling search.

## Phase 2: Request routing

Implemented in `scheduling.request_routing`.

- **`RequestRoutingStrategy`**: interface with
  - `find_turning_points(nodes, num_layers)` for optional warm-up truncations.
  - `find_optimal_path(nodes, num_layers)` to return `(node_ids, latency)`.

- **`DynamicProgrammingRouting`**
  - Warm-up: layer-level DP over hosts of each layer to detect turning points:
    - `(node_id, l, "tail")`: node still hosts `l` but optimal path switches away → drop `[l, end)` on that node.
    - `(node_id, l, "head")`: path first uses node at `l > start` → drop `[start, l)`.
  - Routing: shard-level DP over the assigned contiguous ranges; edge cost is RTT via `Node.get_rtt_to`, vertex cost is `Node.layer_latency_ms`.

## Orchestration: `Scheduler`

Implemented in `scheduling.scheduler`.

- Coordinates layer allocation, node join/leave, periodic heartbeat checks, and request dispatch.
- Supports either `GreedyLayerAllocator` or `DynamicProgrammingLayerAllocator` via `strategy`.
- Maintains thread-safe queues for events and a background dispatcher for requests.
- Bootstrapping:
  - Waits for `min_nodes_bootstrapping` nodes, runs `global_allocation()`, and optional warm-up truncation via `request_warm_up_for_reshard` and `find_turning_points`.
- Dynamic events (non-blocking enqueuers):
  - `enqueue_join(node)`, `enqueue_leave(node_id)`, `enqueue_node_update(...)`.
- Heartbeats: `checking_node_heartbeat()` evicts nodes inactive for `heartbeat_timeout` seconds and can trigger a global rebalance.
- Dispatching: `dispatch_next_request()` or background `_dispatch_loop` compute routes via `RequestRoutingStrategy` and increment per-node load counters.

### Scheduler configuration
Constructor signature (selected arguments):

```python
Scheduler(
  model_info: ModelInfo,
  nodes: List[Node],
  min_nodes_bootstrapping: int = 1,
  strategy: Literal["greedy", "dp"] = "dp",
  request_arrival_horizon_sec: float = 600.0,
  rebalance_threshold: float = float("inf"),
  water_filling_max_iterations: int = 40,
  request_warm_up_for_reshard: int = 0,
  heartbeat_timeout: float = 60.0,
)
```

Notes:
- Setting `rebalance_threshold` low enables auto global rebalancing based on layer load imbalance; `float("inf")` disables it.
- `request_warm_up_for_reshard > 0` runs warm-up routing passes and applies consistent turning points to shrink shards before serving.

## Usage example

A minimal construction using the DP allocator and router:

```python
from scheduling.model_info import ModelInfo
from scheduling.node import Node, NodeHardwareInfo
from scheduling.scheduler import Scheduler

# Define model and two nodes
model = ModelInfo(  # instantiate with your model's parameters
    num_layers=40,
    # ... other fields ...
)

n0 = Node(
    node_id="node-0",
    hardware=NodeHardwareInfo(node_id="node-0", tflops_fp16=180.0, num_gpus=1, gpu_name="", memory_gb=80.0, memory_bandwidth_gbps=2039.0),
    model_info=model,
)

n1 = Node(
    node_id="node-1",
    hardware=NodeHardwareInfo(node_id="node-1", tflops_fp16=180.0, num_gpus=1, gpu_name="", memory_gb=80.0, memory_bandwidth_gbps=2039.0),
    model_info=model,
)

sched = Scheduler(model_info=model, nodes=[n0, n1], strategy="dp")

# Bootstrapping
sched.bootstrap()  # or sched.run() to start background threads

# List assigned ranges
print(sched.list_node_allocations())

# Enqueue a request signal (example)
from scheduling.node import RequestSignal
sched.receive_request(RequestSignal(request_id="req-1"))
print(sched.dispatch_next_request())  # (request_id, [node_id, ...], latency_ms)
```

## Extensibility
- Add a new allocator: subclass `BaseLayerAllocator` and implement `global_allocation()`; reuse `adjust_pipeline_layers()` if applicable.
- Add a new router: implement `RequestRoutingStrategy` with the two abstract methods and plug it into `Scheduler`.
- Custom RTT measurement: set `Node.rtt_getter: Callable[[Node, Node], float]` to lazily populate `rtt_to_nodes`.

## Guarantees and assumptions
- Allocations are contiguous `[start, end)` per node; endpoints are reserved at pipeline edges.
- A request runs all layers hosted on a node before transitioning to the next node in the path.
- Bootstrapping success implies at least one full pipeline covering `[0, L)`.

## Tests
Relevant tests live under `tests/scheduler_tests/` and cover allocation, routing, and scheduling:
- `test_layer_allocation.py`
- `test_request_routing.py`
- `test_scheduler.py`
- `test_utils.py`

Run your project’s test runner to validate changes.
