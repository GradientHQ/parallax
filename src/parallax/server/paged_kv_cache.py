from typing import Dict, List, Optional, Set, Tuple

import mlx.core as mx

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class BlockAllocator:
    """Manages allocation of physical block indices."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Initialize free blocks stack
        # Using a list as a stack is efficient for LIFO allocation
        self.free_blocks: List[int] = list(range(num_blocks))
        # Keep track of used blocks for safety/debugging
        self.used_blocks: Set[int] = set()

    def allocate(self, num_blocks_needed: int) -> List[int]:
        """Allocates `num_blocks_needed` physical blocks."""
        if len(self.free_blocks) < num_blocks_needed:
            # Out of memory
            return []

        # Pop blocks from the stack
        split_idx = len(self.free_blocks) - num_blocks_needed
        allocated = self.free_blocks[split_idx:]
        self.free_blocks = self.free_blocks[:split_idx]

        for b in allocated:
            self.used_blocks.add(b)

        return allocated

    def free(self, blocks: List[int]):
        """Frees the given physical blocks."""
        for b in blocks:
            if b in self.used_blocks:
                self.used_blocks.remove(b)
                self.free_blocks.append(b)
            else:
                logger.warning(f"Double free detected for block {b}")

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def get_num_used_blocks(self) -> int:
        return len(self.used_blocks)


class PagedKVCacheManager:
    """
    Manages the Paged KV Cache tensors and block tables for requests.
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
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.dtype = dtype
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs

        if num_gpu_blocks is None:
            num_gpu_blocks = self._calculate_num_blocks(cache_memory_fraction, dtype)

        self.num_gpu_blocks = num_gpu_blocks

        # 1. Initialize Allocator
        self.allocator = BlockAllocator(num_gpu_blocks, block_size)

        # 2. Allocate Global Cache Tensors
        # Shape: (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
        logger.info(
            f"Allocating Paged KV Cache: {num_gpu_blocks} blocks, {block_size} block_size, "
            f"k_head_dim={self.head_dim}, v_head_dim={self.head_dim_v}"
        )

        self.key_cache = mx.zeros(
            (num_layers, num_gpu_blocks, num_kv_heads, block_size, self.head_dim), dtype=dtype
        )
        self.value_cache = mx.zeros(
            (num_layers, num_gpu_blocks, num_kv_heads, block_size, self.head_dim_v), dtype=dtype
        )

        # Ensure memory is materialized
        mx.eval(self.key_cache, self.value_cache)

        # 3. Request State Management
        # Mapping: request_id -> List of physical block indices
        self.block_tables: Dict[str, List[int]] = {}
        # Mapping: request_id -> current context length (number of tokens)
        self.context_lengths: Dict[str, int] = {}

    def _calculate_num_blocks(self, cache_memory_fraction: float, dtype: mx.Dtype) -> int:

        device_info = mx.metal.device_info()
        total_mem = device_info["max_recommended_working_set_size"]
        current_mem = mx.metal.get_active_memory()
        free_mem = total_mem - current_mem

        # We use a fraction of FREE memory, but for safety in multi-process/multi-model
        # scenarios, we might want to base it on TOTAL memory fraction if we know
        # what we are doing (as in Executor logic).
        # However, to be safe and consistent with previous logic that tried to avoid OOM:
        # Let's stick to the logic that available_for_kv is based on free memory
        # OR total_memory * fraction if we trust the fraction to be per-process adjusted.

        # If fraction is small (e.g. < 0.2), it likely means it's per-process adjusted.
        # But here we stick to "use what is available" to be safe.
        available_for_kv = free_mem * cache_memory_fraction

        dtype_size = 2 if dtype in [mx.float16, mx.bfloat16] else 4

        # Calculate bytes per block considering potentially different K and V head dimensions
        key_block_bytes = (
            self.num_layers * self.num_kv_heads * self.block_size * self.head_dim * dtype_size
        )
        value_block_bytes = (
            self.num_layers * self.num_kv_heads * self.block_size * self.head_dim_v * dtype_size
        )
        block_bytes = key_block_bytes + value_block_bytes

        num_gpu_blocks = int(available_for_kv // block_bytes)

        if num_gpu_blocks <= 0:
            logger.warning(
                f"Not enough memory for KV cache. Total: {total_mem / 1024**3:.2f} GB, "
                f"Used: {current_mem / 1024**3:.2f} GB, Free: {free_mem / 1024**3:.2f} GB. "
                f"Defaulting to minimal 16 blocks."
            )
            num_gpu_blocks = 16

        logger.info(
            f"PagedKVCache: Calculated num_gpu_blocks={num_gpu_blocks} based on "
            f"fraction={cache_memory_fraction:.2f}, "
            f"total_mem={total_mem/1024**3:.2f} GB, "
            f"used_mem={current_mem/1024**3:.2f} GB, "
            f"free_mem={free_mem/1024**3:.2f} GB, "
            f"available_for_kv={available_for_kv/1024**3:.2f} GB"
        )
        return num_gpu_blocks

    def get_num_free_blocks(self) -> int:
        return self.allocator.get_num_free_blocks()

    def can_allocate(self, num_tokens: int) -> bool:
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        return self.allocator.get_num_free_blocks() >= num_blocks

    def allocate_request(self, request_id: str, prompt_len: int) -> bool:
        """
        Allocates initial blocks for a new request (Prefill).
        Returns True if successful, False if OOM.
        """
        if request_id in self.block_tables:
            return True

        num_blocks = (prompt_len + self.block_size - 1) // self.block_size
        blocks = self.allocator.allocate(num_blocks)

        if len(blocks) < num_blocks:
            # Allocation failed
            if blocks:
                self.allocator.free(blocks)
            return False

        self.block_tables[request_id] = blocks
        self.context_lengths[request_id] = prompt_len
        return True

    def has_request(self, request_id: str) -> bool:
        return request_id in self.block_tables

    def free_request(self, request_id: str):
        """Frees all blocks associated with a request."""
        if request_id in self.block_tables:
            blocks = self.block_tables[request_id]
            self.allocator.free(blocks)
            del self.block_tables[request_id]
            del self.context_lengths[request_id]

    def release_request(self, request_id: str):
        """Alias for free_request to match Executor expectation."""
        self.free_request(request_id)

    def append_slot(self, request_id: str) -> bool:
        """
        Allocates a new slot for the next token generation (Decode).
        If the last block is full, allocates a new block.
        """
        if request_id not in self.block_tables:
            raise ValueError(f"Request {request_id} not found")

        current_len = self.context_lengths[request_id]

        if current_len % self.block_size == 0:
            new_blocks = self.allocator.allocate(1)
            if not new_blocks:
                return False  # OOM
            self.block_tables[request_id].extend(new_blocks)

        self.context_lengths[request_id] += 1
        return True

    def get_block_table(self, request_id: str) -> List[int]:
        return self.block_tables.get(request_id, [])

    def get_context_length(self, request_id: str) -> int:
        return self.context_lengths.get(request_id, 0)

    def get_cache(self) -> Tuple[mx.array, mx.array]:
        """Returns the global cache tensors."""
        return self.key_cache, self.value_cache
