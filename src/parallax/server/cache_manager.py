from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from parallax.server.block_radix_cache import BlockRadixCache
from parallax.server.cache.allocator import BlockAllocator, SlotAllocator
from parallax.server.cache.base import BaseCache
from parallax.server.cache.dsa_cache import DeepSeekSparseCache
from parallax.server.cache.kv_cache import KVCachePacked
from parallax.server.cache.linear_cache import LinearCache
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    Manages the Layer Caches (KV and Linear) and their memory allocation for requests.
    Supports hybrid models with mix of Attention and Linear layers.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype,
        block_size: int = 16,
        cache_memory_fraction: float = 0.8,
        num_gpu_blocks: Optional[int] = None,
        max_num_seqs: int = 256,  # Max concurrent requests hint
        head_dim_v: Optional[int] = None,
        index_head_dim: Optional[int] = None,
        index_n_heads: Optional[int] = None,
        # Hybrid Config: List of 'attention' or 'linear' or None (default 'attention')
        layer_types: Optional[List[str]] = None,
        # Linear Model / State Cache Params
        conv_dim: Optional[int] = None,
        conv_kernel_size: Optional[int] = None,
        linear_k_dim: Optional[int] = None,
        linear_v_dim: Optional[int] = None,
        linear_num_k_heads: Optional[int] = None,
        linear_num_v_heads: Optional[int] = None,
        # Prefix Cache Config
        enable_prefix_cache: bool = False,
        max_cached_blocks: int = 1000,
        sliding_window: Optional[int] = None,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.dtype = dtype
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.sliding_window = sliding_window

        # Linear cache params (store for memory calculation)
        self.conv_dim = conv_dim
        self.conv_kernel_size = conv_kernel_size
        self.linear_k_dim = linear_k_dim
        self.linear_v_dim = linear_v_dim
        self.linear_num_k_heads = linear_num_k_heads
        self.linear_num_v_heads = linear_num_v_heads
        self.cache_memory_fraction = cache_memory_fraction

        # Determine layer types
        if layer_types is None:
            self.layer_types = ["attention"] * num_layers
        else:
            assert len(layer_types) == num_layers, "layer_types length must match num_layers"
            self.layer_types = layer_types

        # Check if we need blocks (any attention layer) and slots (any linear layer)
        self.needs_blocks = any(t == "attention" for t in self.layer_types)
        self.needs_slots = any(t == "linear" for t in self.layer_types)

        if num_gpu_blocks is None and self.needs_blocks:
            num_gpu_blocks = self._calculate_num_blocks(self.cache_memory_fraction, self.dtype)
        elif not self.needs_blocks:
            num_gpu_blocks = 0

        self.num_gpu_blocks = num_gpu_blocks

        # 1. Initialize Allocators
        self.allocator = (
            BlockAllocator(self.num_gpu_blocks, self.block_size) if self.needs_blocks else None
        )
        self.slot_allocator = SlotAllocator(self.max_num_seqs) if self.needs_slots else None

        # 2. Initialize Layer Caches
        self.caches: List[BaseCache] = []

        for layer_type in self.layer_types:
            self.caches.append(self._create_cache(layer_type))

        if self.needs_blocks:
            logger.info(
                f"Allocated Paged KV Cache for {self.layer_types.count('attention')} layers: "
                f"{self.num_gpu_blocks} blocks, {self.block_size} block_size, max_tokens: {self.num_gpu_blocks * self.block_size}"
            )
        if self.needs_slots:
            logger.info(
                f"Allocated Linear State Cache for {self.layer_types.count('linear')} layers: "
                f"{self.max_num_seqs} max slots"
            )

        # 3. Request State Management
        # Mapping: request_id -> List of physical block indices
        self.block_tables: Dict[str, List[int]] = {}
        # Mapping: request_id -> current context length (number of tokens)
        self.context_lengths: Dict[str, int] = {}
        # Mapping: request_id -> state slot index
        self.request_slots: Dict[str, int] = {}

        # 4. Prefix Cache (Optional)
        self.enable_prefix_cache = enable_prefix_cache
        self.prefix_cache = None
        if enable_prefix_cache and self.needs_blocks:
            self.prefix_cache = BlockRadixCache(
                block_size=block_size,
                max_cached_blocks=max_cached_blocks,
                on_block_evict=self._on_prefix_block_evict,
            )
            logger.info(f"Prefix cache enabled with max_cached_blocks={max_cached_blocks}")

        # Mapping: request_id -> token_ids (for prefix matching)
        self.request_token_ids: Dict[str, List[int]] = {}
        # Mapping: request_id -> matched_tokens (for prefix cache hit tracking)
        self.matched_tokens_cache: Dict[str, int] = {}

    def _on_prefix_block_evict(self, block_id: int):
        """Callback when a block is evicted from prefix cache."""
        if self.needs_blocks:
            self.allocator.free([block_id])
            logger.debug(f"Freed evicted prefix cache block: {block_id}")

    def _create_cache(self, layer_type: str) -> BaseCache:
        if layer_type == "attention":
            if self.index_head_dim is not None and self.index_n_heads is not None:
                return DeepSeekSparseCache(
                    num_blocks=self.num_gpu_blocks,
                    block_size=self.block_size,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    head_dim_v=self.head_dim_v,
                    dtype=self.dtype,
                    index_head_dim=self.index_head_dim,
                    index_n_heads=self.index_n_heads,
                )
            else:
                return KVCachePacked(
                    num_blocks=self.num_gpu_blocks,
                    block_size=self.block_size,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    head_dim_v=self.head_dim_v,
                    dtype=self.dtype,
                )

        elif layer_type == "linear":
            # We assume uniform linear config for all linear layers for now
            return LinearCache(
                max_num_seqs=self.max_num_seqs,
                conv_dim=self.conv_dim,
                conv_kernel_size=self.conv_kernel_size,
                linear_k_dim=self.linear_k_dim,
                linear_v_dim=self.linear_v_dim,
                linear_num_k_heads=self.linear_num_k_heads,
                linear_num_v_heads=self.linear_num_v_heads,
                dtype=self.dtype,
            )
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    def _calculate_linear_cache_bytes(self, dtype_size: int) -> int:
        """Calculate total memory needed for linear cache across all linear layers."""
        num_linear_layers = self.layer_types.count("linear")
        if num_linear_layers == 0:
            return 0

        one_layer_bytes = 0

        # conv_state: (1, max_num_seqs, conv_kernel_size - 1, conv_dim)
        if self.conv_dim is not None and self.conv_kernel_size is not None:
            conv_state_len = self.conv_kernel_size - 1
            one_layer_bytes += self.max_num_seqs * conv_state_len * self.conv_dim * dtype_size

        # linear_state: (1, max_num_seqs, linear_num_v_heads, linear_v_dim, linear_k_dim)
        if (
            self.linear_k_dim is not None
            and self.linear_v_dim is not None
            and self.linear_num_v_heads is not None
        ):
            one_layer_bytes += (
                self.max_num_seqs
                * self.linear_num_v_heads
                * self.linear_v_dim
                * self.linear_k_dim
                * dtype_size
            )

        total_bytes = one_layer_bytes * num_linear_layers

        if total_bytes > 0:
            logger.info(
                f"Linear cache will use {total_bytes / 1024**3:.2f} GB "
                f"for {num_linear_layers} layers"
            )

        return total_bytes

    def _calculate_num_blocks(self, cache_memory_fraction: float, dtype: mx.Dtype) -> int:
        device_info = mx.metal.device_info()
        total_mem = device_info["max_recommended_working_set_size"]
        current_mem = mx.get_active_memory()
        free_mem = total_mem - current_mem
        available_for_cache = free_mem * cache_memory_fraction

        dtype_size = 2 if dtype in [mx.float16, mx.bfloat16] else 4

        # First, calculate linear cache memory (fixed size, allocated upfront)
        linear_cache_bytes = self._calculate_linear_cache_bytes(dtype_size)

        # Remaining memory for KV cache
        available_for_kv = available_for_cache - linear_cache_bytes
        if available_for_kv <= 0:
            logger.warning("Linear cache uses all available memory. No room for KV cache blocks.")
            return 0

        # Calculate bytes per block for ONE attention layer
        one_layer_block_bytes = (
            self.num_kv_heads * self.block_size * (self.head_dim + self.head_dim_v) * dtype_size
        )
        if self.index_head_dim is not None and self.index_n_heads is not None:
            one_layer_block_bytes += (
                self.index_n_heads * self.block_size * self.index_head_dim * dtype_size
            )

        # Total bytes per block = Sum over all attention layers
        num_attention_layers = self.layer_types.count("attention")
        total_block_bytes = one_layer_block_bytes * num_attention_layers

        if total_block_bytes == 0:
            return 0

        num_gpu_blocks = int(available_for_kv // total_block_bytes)

        if num_gpu_blocks <= 0:
            logger.warning("Not enough memory for KV cache. Defaulting to 16 blocks.")
            num_gpu_blocks = 16

        logger.info(
            f"KV cache will use {num_gpu_blocks * total_block_bytes / 1024**3:.2f} GB "
            f"for {num_attention_layers} layers ({num_gpu_blocks} blocks)"
        )

        return num_gpu_blocks

    def can_allocate(self, num_tokens: int) -> bool:
        if not self.needs_blocks:
            return (
                self.slot_allocator.get_num_free_slots() > 0 if self.needs_slots else True
            )  # Should check slots

        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        blocks_ok = self.allocator.get_num_free_blocks() >= num_blocks

        slots_ok = True
        if self.needs_slots:
            slots_ok = self.slot_allocator.get_num_free_slots() > 0

        return blocks_ok and slots_ok

    def allocate_request(
        self, request_id: str, prompt_len: int, token_ids: Optional[List[int]] = None
    ) -> Tuple[bool, int]:
        """Allocate KV cache blocks for a request.

        Returns:
            Tuple[bool, int]: (success, matched_tokens)
                - success: Whether allocation succeeded
                - matched_tokens: Number of tokens matched from prefix cache (0 if no match)
        """
        if request_id in self.block_tables:
            # Already allocated, return cached matched_tokens if available
            cached_matched = self.matched_tokens_cache.get(request_id, 0)
            return True, cached_matched

        # 1. Try to match prefix from cache first
        matched_blocks = []
        matched_tokens = 0
        if self.prefix_cache and token_ids is not None and self.needs_blocks:
            # Always save token_ids for later insertion to prefix cache
            self.request_token_ids[request_id] = token_ids

            # Debug: Log token_ids to understand prefix cache behavior
            logger.debug(
                f"Request {request_id}: input_ids length={len(token_ids)}, "
                f"first 20 tokens={token_ids[:20]}, "
                f"last 20 tokens={token_ids[-20:]}"
            )

            matched_blocks, matched_tokens = self.prefix_cache.match_prefix(token_ids)
            if matched_tokens > 0:
                matched_nodes = []
                temp_node = self.prefix_cache.root
                for i, block_id in enumerate(matched_blocks):
                    block_start = i * self.block_size
                    block_end = block_start + self.block_size
                    block_tokens = token_ids[block_start:block_end]
                    first_token = block_tokens[0]
                    temp_node = temp_node.children[first_token]
                    matched_nodes.append(temp_node)

                self.prefix_cache.register_request(request_id, matched_nodes)

                logger.info(
                    f"Request {request_id}: Prefix cache hit! Reused {len(matched_blocks)} blocks "
                    f"({matched_tokens}/{prompt_len} tokens)"
                )

        # 2. Allocate Slot (if needed)
        slot = -1
        if self.needs_slots:
            slot = self.slot_allocator.allocate()
            if slot == -1:
                return False

        # 3. Allocate Blocks (if needed)
        blocks = matched_blocks.copy()
        if self.needs_blocks:
            num_blocks = (prompt_len + self.block_size - 1) // self.block_size
            num_new_blocks = num_blocks - len(matched_blocks)

            if num_new_blocks > 0:
                new_blocks = self.allocator.allocate(num_new_blocks)
                if len(new_blocks) < num_new_blocks:
                    if new_blocks:
                        self.allocator.free(new_blocks)
                    if slot != -1:
                        self.slot_allocator.free(slot)
                    if self.prefix_cache and request_id in self.prefix_cache.request_to_nodes:
                        self.prefix_cache.release_request(request_id)
                    return False, 0
                blocks.extend(new_blocks)

        # 4. Commit
        if self.needs_blocks:
            self.block_tables[request_id] = blocks
            self.context_lengths[request_id] = prompt_len
            # Cache matched_tokens for later retrieval
            self.matched_tokens_cache[request_id] = matched_tokens

        if self.needs_slots:
            self.request_slots[request_id] = slot
            # Zero out state caches for this slot
            for cache in self.caches:
                if isinstance(cache, LinearCache):
                    # Zero out conv and linear states
                    if cache.conv_state_cache is not None:
                        cache.conv_state_cache[..., slot, :, :] = 0
                    if cache.linear_state_cache is not None:
                        cache.linear_state_cache[..., slot, :, :, :] = 0

        return True, matched_tokens

    def free_request(self, request_id: str):
        # Get blocks that are managed by prefix cache (should not be freed)
        cached_blocks = set()
        if self.prefix_cache and request_id in self.prefix_cache.request_to_nodes:
            nodes = self.prefix_cache.request_to_nodes[request_id]
            cached_blocks = {node.block_id for node in nodes}

        if self.prefix_cache:
            self.prefix_cache.release_request(request_id)

        if self.needs_blocks and request_id in self.block_tables:
            blocks = self.block_tables[request_id]
            # Only free blocks that are NOT in prefix cache
            blocks_to_free = [b for b in blocks if b not in cached_blocks]
            if blocks_to_free:
                self.allocator.free(blocks_to_free)
            del self.block_tables[request_id]
            if request_id in self.context_lengths:
                del self.context_lengths[request_id]
            if request_id in self.matched_tokens_cache:
                del self.matched_tokens_cache[request_id]

        if self.needs_slots and request_id in self.request_slots:
            slot = self.request_slots[request_id]
            self.slot_allocator.free(slot)
            del self.request_slots[request_id]

        if request_id in self.request_token_ids:
            del self.request_token_ids[request_id]

    def release_request(self, request_id: str):
        self.free_request(request_id)

    def has_request(self, request_id: str) -> bool:
        if self.needs_blocks:
            return request_id in self.block_tables
        if self.needs_slots:
            return request_id in self.request_slots
        return False

    def append_slot(self, request_id: str) -> bool:
        """Decode step allocation."""
        if not self.needs_blocks:
            return True

        if request_id not in self.block_tables:
            raise ValueError(f"Request {request_id} not found")

        current_len = self.context_lengths[request_id]

        # if current_len > 0 and current_len % self.block_size == 0:
        #     self.insert_full_blocks_to_cache(request_id)

        if current_len % self.block_size == 0:
            new_blocks = self.allocator.allocate(1)
            if not new_blocks:
                return False
            self.block_tables[request_id].extend(new_blocks)

        self.context_lengths[request_id] += 1
        return True

    def get_block_table(self, request_id: str) -> List[int]:
        return self.block_tables.get(request_id, [])

    def get_context_length(self, request_id: str) -> int:
        return self.context_lengths.get(request_id, 0)

    def get_slot(self, request_id: str) -> int:
        return self.request_slots.get(request_id, -1)

    def get_matched_tokens(self, request_id: str) -> int:
        """Get the number of matched tokens from prefix cache for a request."""
        return self.matched_tokens_cache.get(request_id, 0)

    def get_caches(self) -> List[BaseCache]:
        """Returns the list of layer caches."""
        return self.caches

    def match_and_reuse_prefix(self, request_id: str, token_ids: List[int]) -> int:
        """
        Match prefix before prefill and reuse existing blocks.

        Args:
            request_id: Request ID
            token_ids: Complete token sequence

        Returns:
            matched_tokens: Number of matched tokens
        """
        if not self.prefix_cache or not self.needs_blocks:
            return 0

        self.request_token_ids[request_id] = token_ids

        matched_blocks, matched_tokens = self.prefix_cache.match_prefix(token_ids)

        if matched_tokens == 0:
            return 0

        matched_nodes = []
        temp_node = self.prefix_cache.root
        for block_id in matched_blocks:
            block_start = len(matched_nodes) * self.block_size
            block_end = block_start + self.block_size
            block_tokens = token_ids[block_start:block_end]
            first_token = block_tokens[0]
            temp_node = temp_node.children[first_token]
            matched_nodes.append(temp_node)

        self.prefix_cache.register_request(request_id, matched_nodes)

        if request_id in self.block_tables:
            existing_blocks = self.block_tables[request_id]
            self.block_tables[request_id] = matched_blocks + existing_blocks[len(matched_blocks) :]
        else:
            self.block_tables[request_id] = matched_blocks

        logger.info(
            f"Request {request_id}: Reused {len(matched_blocks)} blocks "
            f"({matched_tokens} tokens) from prefix cache"
        )

        return matched_tokens

    def insert_full_blocks_to_cache(self, request_id: str):
        """
        Insert full blocks from request to prefix cache.
        Called after prefill or when a block is filled.

        Args:
            request_id: Request ID
        """
        if not self.prefix_cache or not self.needs_blocks:
            return

        if request_id not in self.request_token_ids:
            return

        if request_id not in self.block_tables:
            return

        token_ids = self.request_token_ids[request_id]
        block_table = self.block_tables[request_id]
        context_len = self.context_lengths.get(request_id, 0)

        num_full_blocks = context_len // self.block_size

        if num_full_blocks == 0:
            return

        registered_nodes = self.prefix_cache.request_to_nodes.get(request_id, [])
        num_registered = len(registered_nodes)

        parent_path = registered_nodes.copy() if registered_nodes else []

        for block_idx in range(num_registered, num_full_blocks):
            block_start = block_idx * self.block_size
            block_end = block_start + self.block_size

            if block_end > len(token_ids):
                break

            block_tokens = token_ids[block_start:block_end]
            block_id = block_table[block_idx]

            new_node = self.prefix_cache.insert_block(
                token_ids=block_tokens, block_id=block_id, parent_path=parent_path
            )

            parent_path.append(new_node)
            registered_nodes.append(new_node)

            logger.debug(
                f"Request {request_id}: Inserted block {block_idx} "
                f"(block_id={block_id}) to prefix cache"
            )

        if registered_nodes:
            if request_id in self.prefix_cache.request_to_nodes:
                old_nodes = self.prefix_cache.request_to_nodes[request_id]
                self.prefix_cache.decrease_lock_ref(old_nodes)

            self.prefix_cache.request_to_nodes[request_id] = registered_nodes
            self.prefix_cache.increase_lock_ref(registered_nodes)

    def update_request_tokens(self, request_id: str, new_token_ids: List[int]):
        """
        Update request token sequence (for decode phase).

        Args:
            request_id: Request ID
            new_token_ids: New token IDs to append
        """
        if request_id in self.request_token_ids:
            self.request_token_ids[request_id].extend(new_token_ids)
        else:
            self.request_token_ids[request_id] = new_token_ids
