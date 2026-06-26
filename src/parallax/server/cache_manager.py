from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from parallax.server.block_radix_cache import BlockRadixCache
from parallax.server.cache.allocator import BlockAllocator, SlotAllocator
from parallax.server.cache.base import BaseCache
from parallax.server.cache.dsa_cache import DeepSeekSparseCache
from parallax.server.cache.kv_cache import KVCachePacked
from parallax.server.cache.linear_cache import LinearCache
from parallax.server.cache.msa_cache import MSACache
from parallax.utils.layer_types import (
    ATTENTION,
    ATTENTION_LAYER_TYPES,
    DSA_ATTENTION,
    LINEAR,
    MLA_ATTENTION,
    MSA_ATTENTION,
)
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
        max_num_seqs: int = 256,  # Max concurrent requests hint
        head_dim_v: Optional[int] = None,
        index_head_dim: Optional[int] = None,
        index_n_heads: Optional[int] = None,
        index_key_heads: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        # Hybrid Config: List of cache layer types or None (default 'attention')
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
        sliding_window: Optional[int] = None,
        chunked_prefill_size: Optional[int] = None,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_key_heads = index_key_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.dtype = dtype
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.sliding_window = sliding_window
        self.enable_prefix_cache = enable_prefix_cache
        self.chunked_prefill_size = (
            chunked_prefill_size
            if chunked_prefill_size is not None and chunked_prefill_size > 0
            else None
        )

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
            self.layer_types = [ATTENTION] * num_layers
        else:
            assert len(layer_types) == num_layers, "layer_types length must match num_layers"
            self.layer_types = layer_types

        # Check if we need blocks (any attention layer) and slots (any linear layer)
        self.needs_blocks = any(t in ATTENTION_LAYER_TYPES for t in self.layer_types)
        self.needs_slots = any(t == LINEAR for t in self.layer_types)

        self.num_gpu_blocks, self.num_linear_prefix_slots = self._calculate_cache_allocation(
            self.cache_memory_fraction, self.dtype
        )
        self.max_linear_slots = self.max_num_seqs + self.num_linear_prefix_slots

        # 1. Initialize Allocators
        self.allocator = (
            BlockAllocator(self.num_gpu_blocks, self.block_size) if self.needs_blocks else None
        )
        self.slot_allocator = SlotAllocator(self.max_num_seqs) if self.needs_slots else None
        self.prefix_slot_allocator = (
            SlotAllocator(self.num_linear_prefix_slots, start_idx=self.max_num_seqs)
            if self._needs_prefix_linear_slots() and self.num_linear_prefix_slots > 0
            else None
        )

        # 2. Initialize Layer Caches
        self.caches: List[BaseCache] = []

        for layer_type in self.layer_types:
            self.caches.append(self._create_cache(layer_type))

        if self.needs_blocks:
            logger.info(
                f"Allocated Paged KV Cache for {self._num_attention_layers()} layers: "
                f"{self.num_gpu_blocks} blocks, {self.block_size} block_size, max_tokens: {self.num_gpu_blocks * self.block_size}"
            )
        if self.needs_slots:
            logger.info(
                f"Allocated Linear State Cache for {self._num_linear_layers()} layers: "
                f"{self.max_num_seqs} active slots, "
                f"{self.num_linear_prefix_slots} prefix slots"
            )

        # 3. Request State Management
        # Mapping: request_id -> List of physical block indices
        self.block_tables: Dict[str, List[int]] = {}
        # Mapping: request_id -> current context length (number of tokens)
        self.context_lengths: Dict[str, int] = {}
        # Mapping: request_id -> state slot index
        self.request_slots: Dict[str, int] = {}

        # 4. Prefix Cache (Optional)
        self.prefix_cache = None
        if enable_prefix_cache and self.needs_blocks:
            self.prefix_cache = BlockRadixCache(
                block_size=block_size,
                on_block_evict=self._on_prefix_block_evict,
                on_linear_slot_evict=self._on_prefix_linear_slot_evict,
                has_linear_cache=self.needs_slots,
            )
            logger.info("Prefix cache enabled")

        # Mapping: request_id -> token_ids (for prefix matching)
        self.request_token_ids: Dict[str, List[int]] = {}
        # Mapping: request_id -> matched_tokens (for prefix cache hit tracking)
        self.matched_tokens_cache: Dict[str, int] = {}

    def _on_prefix_block_evict(self, block_id: int):
        """Callback when a block is evicted from prefix cache."""
        if self.needs_blocks:
            self.allocator.free([block_id])
            logger.debug(f"Freed evicted prefix cache block: {block_id}")

    def _on_prefix_linear_slot_evict(self, slot: int):
        """Callback when a prefix-cache linear slot is evicted."""
        if self.prefix_slot_allocator is not None:
            self._zero_linear_slot(slot)
            self.prefix_slot_allocator.free(slot)
            logger.debug(f"Freed evicted prefix linear slot: {slot}")

    def _num_attention_layers(self) -> int:
        return sum(1 for t in self.layer_types if t in ATTENTION_LAYER_TYPES)

    def _num_linear_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == LINEAR)

    def _validate_mla_cache_params(self, layer_type: str):
        if self.kv_lora_rank is None or self.qk_rope_head_dim is None:
            raise ValueError(
                f"{layer_type} requires kv_lora_rank and qk_rope_head_dim "
                "for compressed MLA storage."
            )

    def _validate_index_cache_params(self, layer_type: str):
        if self.index_head_dim is None or self.index_n_heads is None:
            raise ValueError(f"{layer_type} requires index_head_dim and index_n_heads.")

    def _create_cache(self, layer_type: str) -> BaseCache:
        if layer_type == ATTENTION:
            return KVCachePacked(
                num_blocks=self.num_gpu_blocks,
                block_size=self.block_size,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                head_dim_v=self.head_dim_v,
                dtype=self.dtype,
            )

        if layer_type == MLA_ATTENTION:
            self._validate_mla_cache_params(layer_type)
            return DeepSeekSparseCache(
                num_blocks=self.num_gpu_blocks,
                block_size=self.block_size,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                head_dim_v=self.head_dim_v,
                dtype=self.dtype,
                index_head_dim=None,
                index_n_heads=None,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                index_key_heads=None,
            )

        if layer_type == DSA_ATTENTION:
            self._validate_mla_cache_params(layer_type)
            self._validate_index_cache_params(layer_type)
            return DeepSeekSparseCache(
                num_blocks=self.num_gpu_blocks,
                block_size=self.block_size,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                head_dim_v=self.head_dim_v,
                dtype=self.dtype,
                index_head_dim=self.index_head_dim,
                index_n_heads=self.index_n_heads,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                index_key_heads=self.index_key_heads or 1,
            )

        if layer_type == MSA_ATTENTION:
            self._validate_index_cache_params(layer_type)
            return MSACache(
                num_blocks=self.num_gpu_blocks,
                block_size=self.block_size,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                head_dim_v=self.head_dim_v,
                dtype=self.dtype,
                index_head_dim=self.index_head_dim,
                index_n_heads=self.index_n_heads,
                index_key_heads=self.index_key_heads or 1,
            )

        if layer_type == LINEAR:
            # We assume uniform linear config for all linear layers for now
            return LinearCache(
                max_num_seqs=self.max_linear_slots,
                conv_dim=self.conv_dim,
                conv_kernel_size=self.conv_kernel_size,
                linear_k_dim=self.linear_k_dim,
                linear_v_dim=self.linear_v_dim,
                linear_num_k_heads=self.linear_num_k_heads,
                linear_num_v_heads=self.linear_num_v_heads,
                dtype=self.dtype,
            )

        raise ValueError(f"Unknown layer type: {layer_type}")

    def _needs_prefix_linear_slots(self) -> bool:
        return self.enable_prefix_cache and self.needs_blocks and self.needs_slots

    def _dtype_size(self, dtype: mx.Dtype) -> int:
        return 2 if dtype in [mx.float16, mx.bfloat16] else 4

    def _calculate_linear_slot_bytes(self, dtype_size: int) -> int:
        """Calculate memory needed for one linear slot across all linear layers."""
        num_linear_layers = self._num_linear_layers()
        if num_linear_layers == 0:
            return 0

        one_layer_bytes = 0

        # conv_state per slot: (conv_kernel_size - 1, conv_dim)
        if self.conv_dim is not None and self.conv_kernel_size is not None:
            conv_state_len = self.conv_kernel_size - 1
            one_layer_bytes += conv_state_len * self.conv_dim * dtype_size

        # linear_state per slot: (linear_num_v_heads, linear_v_dim, linear_k_dim)
        if (
            self.linear_k_dim is not None
            and self.linear_v_dim is not None
            and self.linear_num_v_heads is not None
        ):
            one_layer_bytes += (
                self.linear_num_v_heads * self.linear_v_dim * self.linear_k_dim * dtype_size
            )

        return one_layer_bytes * num_linear_layers

    def _calculate_linear_cache_bytes(self, dtype_size: int, num_slots: int) -> int:
        """Calculate total linear cache bytes for the given slot count."""
        return self._calculate_linear_slot_bytes(dtype_size) * num_slots

    def _calculate_kv_block_bytes(self, dtype_size: int) -> int:
        """Calculate memory needed for one KV block across all attention layers."""
        total_bytes = 0
        standard_block_bytes = (
            self.num_kv_heads * self.block_size * (self.head_dim + self.head_dim_v) * dtype_size
        )
        for layer_type in self.layer_types:
            if layer_type == ATTENTION:
                total_bytes += standard_block_bytes
            elif layer_type == MLA_ATTENTION:
                self._validate_mla_cache_params(layer_type)
                total_bytes += (
                    self.block_size * (self.kv_lora_rank + self.qk_rope_head_dim) * dtype_size
                )
            elif layer_type == DSA_ATTENTION:
                self._validate_mla_cache_params(layer_type)
                self._validate_index_cache_params(layer_type)
                total_bytes += (
                    self.block_size * (self.kv_lora_rank + self.qk_rope_head_dim) * dtype_size
                )
                total_bytes += (
                    (self.index_key_heads or 1) * self.block_size * self.index_head_dim * dtype_size
                )
            elif layer_type == MSA_ATTENTION:
                self._validate_index_cache_params(layer_type)
                total_bytes += standard_block_bytes
                total_bytes += (
                    (self.index_key_heads or 1) * self.block_size * self.index_head_dim * dtype_size
                )
            elif layer_type == LINEAR:
                continue
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        return total_bytes

    def _calculate_prefix_linear_bytes(
        self,
        available_for_kv: float,
        total_block_bytes: int,
        linear_slot_bytes: int,
    ) -> float:
        """Split existing KV cache budget and reserve part for prefix linear slots."""
        if (
            not self._needs_prefix_linear_slots()
            or available_for_kv <= 0
            or total_block_bytes <= 0
            or linear_slot_bytes <= 0
        ):
            return 0.0

        if self.chunked_prefill_size is None:
            return available_for_kv * 0.10

        kv_token_bytes = total_block_bytes / self.block_size
        kv_chunk_bytes = self.chunked_prefill_size * kv_token_bytes
        return available_for_kv * linear_slot_bytes / (kv_chunk_bytes + linear_slot_bytes)

    def _calculate_cache_allocation(
        self,
        cache_memory_fraction: float,
        dtype: mx.Dtype,
    ) -> Tuple[int, int]:
        total_mem = mx.device_info()["max_recommended_working_set_size"]
        current_mem = mx.get_active_memory()
        free_mem = total_mem - current_mem
        available_for_cache = free_mem * cache_memory_fraction

        dtype_size = self._dtype_size(dtype)

        # First, calculate linear cache memory (fixed size, allocated upfront)
        active_linear_cache_bytes = self._calculate_linear_cache_bytes(
            dtype_size, self.max_num_seqs
        )

        # Total bytes per block = Sum over all attention layers
        num_attention_layers = self._num_attention_layers()
        total_block_bytes = self._calculate_kv_block_bytes(dtype_size)

        if total_block_bytes == 0:
            if active_linear_cache_bytes > 0:
                logger.info(
                    f"Linear cache will use {active_linear_cache_bytes / 1024**3:.2f} GB "
                    f"for {self._num_linear_layers()} layers "
                    f"({self.max_num_seqs} active slots, 0 prefix slots)"
                )
            return 0, 0

        # Remaining memory for KV cache
        available_for_kv = available_for_cache - active_linear_cache_bytes
        if available_for_kv <= 0:
            logger.warning("Linear cache uses all available memory. No room for KV cache blocks.")
            return 0, 0

        linear_slot_bytes = self._calculate_linear_slot_bytes(dtype_size)
        prefix_linear_bytes = self._calculate_prefix_linear_bytes(
            available_for_kv, total_block_bytes, linear_slot_bytes
        )
        num_linear_prefix_slots = 0
        if prefix_linear_bytes > 0 and linear_slot_bytes > 0:
            num_linear_prefix_slots = max(1, int(prefix_linear_bytes // linear_slot_bytes))
            available_for_kv -= num_linear_prefix_slots * linear_slot_bytes

        num_gpu_blocks = int(available_for_kv // total_block_bytes)

        if num_gpu_blocks <= 0:
            logger.warning("Not enough memory for KV cache. Defaulting to 16 blocks.")
            num_gpu_blocks = 16

        logger.info(
            f"KV cache will use {num_gpu_blocks * total_block_bytes / 1024**3:.2f} GB "
            f"for {num_attention_layers} layers ({num_gpu_blocks} blocks)"
        )
        if active_linear_cache_bytes > 0 or num_linear_prefix_slots > 0:
            total_linear_cache_bytes = (
                active_linear_cache_bytes + num_linear_prefix_slots * linear_slot_bytes
            )
            logger.info(
                f"Linear cache will use {total_linear_cache_bytes / 1024**3:.2f} GB "
                f"for {self._num_linear_layers()} layers "
                f"({self.max_num_seqs} active slots, "
                f"{num_linear_prefix_slots} prefix slots)"
            )

        return num_gpu_blocks, num_linear_prefix_slots

    def _zero_linear_slot(self, slot: int):
        for cache in self.caches:
            if isinstance(cache, LinearCache):
                cache.zero_slot(slot)

    def _copy_linear_slot(self, dst_slot: int, src_slot: int):
        for cache in self.caches:
            if isinstance(cache, LinearCache):
                cache.copy_slot(dst_slot, src_slot)

    def _evict_prefix_blocks(self, num_blocks: int) -> int:
        if self.prefix_cache is None or num_blocks <= 0:
            return 0
        return self.prefix_cache.evict_lru_blocks(num_blocks)

    def _allocate_prefix_linear_slot(self) -> int:
        if self.prefix_slot_allocator is None:
            return -1

        slot = self.prefix_slot_allocator.allocate()
        while slot == -1:
            evicted = self._evict_prefix_blocks(1)
            if evicted <= 0:
                break
            slot = self.prefix_slot_allocator.allocate()
        return slot

    def _match_token_ids(self, token_ids: List[int]) -> List[int]:
        if token_ids:
            return token_ids[:-1]
        return token_ids

    def get_reusable_prefix_len(self, token_ids: List[int]) -> int:
        """Return the prefix length that can be safely skipped for this model."""
        if not self.prefix_cache or not self.needs_blocks:
            return 0

        _, matched_tokens = self.prefix_cache.match_prefix(self._match_token_ids(token_ids))
        return matched_tokens

    def allocate_request(
        self,
        request_id: str,
        prompt_len: int,
        token_ids: Optional[List[int]] = None,
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
        matched_nodes = []
        matched_linear_slot = None
        matched_tokens = 0
        if self.prefix_cache and token_ids is not None and self.needs_blocks:
            # Always save token_ids for later insertion to prefix cache
            self.request_token_ids[request_id] = list(token_ids)

            # Debug: Log token_ids to understand prefix cache behavior
            logger.debug(
                f"Request {request_id}: input_ids length={len(token_ids)}, "
                f"first 20 tokens={token_ids[:20]}, "
                f"last 20 tokens={token_ids[-20:]}"
            )

            match_token_ids = self._match_token_ids(token_ids)
            matched_blocks, matched_tokens = self.prefix_cache.match_prefix(match_token_ids)
            if matched_tokens > 0:
                matched_nodes = self.prefix_cache.get_path(match_token_ids[:matched_tokens])
                if len(matched_nodes) != len(matched_blocks):
                    matched_blocks = []
                    matched_nodes = []
                    matched_tokens = 0
                elif self.needs_slots:
                    matched_linear_slot = matched_nodes[-1].linear_slot
                    if matched_linear_slot is None:
                        matched_blocks = []
                        matched_nodes = []
                        matched_tokens = 0

            if matched_tokens > 0:
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
                if self.prefix_cache and request_id in self.prefix_cache.request_to_nodes:
                    self.prefix_cache.release_request(request_id)
                return False, 0

        # 3. Allocate Blocks (if needed)
        blocks = matched_blocks.copy()
        if self.needs_blocks:
            num_blocks = (prompt_len + self.block_size - 1) // self.block_size
            num_new_blocks = num_blocks - len(matched_blocks)

            if num_new_blocks > 0:
                free_blocks = self.allocator.get_num_free_blocks()
                if free_blocks < num_new_blocks:
                    self._evict_prefix_blocks(num_new_blocks - free_blocks)

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
            if matched_linear_slot is not None:
                self._copy_linear_slot(slot, matched_linear_slot)
            else:
                self._zero_linear_slot(slot)

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
                self._evict_prefix_blocks(1)
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

    def materialize_linear_caches(self):
        arrays = []
        for cache in self.caches:
            if isinstance(cache, LinearCache):
                arrays.extend(cache.get_state_cache_arrays())
        if arrays:
            mx.eval(*arrays)

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

        match_token_ids = self._match_token_ids(token_ids)
        matched_blocks, matched_tokens = self.prefix_cache.match_prefix(match_token_ids)

        if matched_tokens == 0:
            return 0

        matched_nodes = self.prefix_cache.get_path(match_token_ids[:matched_tokens])
        if len(matched_nodes) != len(matched_blocks):
            return 0
        if self.needs_slots:
            matched_linear_slot = matched_nodes[-1].linear_slot
            if matched_linear_slot is None:
                return 0
            request_slot = self.request_slots.get(request_id)
            if request_slot is not None:
                self._copy_linear_slot(request_slot, matched_linear_slot)

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

        if registered_nodes:
            if request_id in self.prefix_cache.request_to_nodes:
                old_nodes = self.prefix_cache.request_to_nodes[request_id]
                self.prefix_cache.decrease_lock_ref(old_nodes)

            self.prefix_cache.request_to_nodes[request_id] = registered_nodes
            self.prefix_cache.increase_lock_ref(registered_nodes)

        if (
            self.needs_slots
            and context_len % self.block_size == 0
            and request_id in self.request_slots
            and num_full_blocks > 0
            and len(registered_nodes) >= num_full_blocks
        ):
            request_slot = self.request_slots[request_id]
            prefix_tokens = token_ids[:context_len]
            prefix_node = self.prefix_cache.get_node_for_token_ids(prefix_tokens)
            if prefix_node is None:
                return

            prefix_slot = prefix_node.linear_slot
            if prefix_slot is None:
                prefix_slot = self._allocate_prefix_linear_slot()
                if prefix_slot == -1:
                    logger.warning(
                        "No prefix linear slots available for request %s at prefix_len=%d",
                        request_id,
                        context_len,
                    )
                    return

            self._copy_linear_slot(prefix_slot, request_slot)
            prefix_node.linear_slot = prefix_slot
            logger.debug(
                "Attached linear slot %d for request %s at prefix_len=%d",
                prefix_slot,
                request_id,
                context_len,
            )

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
