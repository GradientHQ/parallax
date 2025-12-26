"""
Block-based Prefix Cache implementation using Radix Tree.
"""

import heapq
import time
from typing import Callable, Dict, List, Optional, Tuple

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class BlockTreeNode:
    """Radix Tree node for managing block-level prefix cache."""

    counter = 0

    def __init__(self, block_id: Optional[int] = None, token_ids: Optional[List[int]] = None):
        self.block_id = block_id
        self.token_ids = token_ids or []
        self.children: Dict[int, "BlockTreeNode"] = {}
        self.parent: Optional["BlockTreeNode"] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()

        self.node_id = BlockTreeNode.counter
        BlockTreeNode.counter += 1

    def __lt__(self, other: "BlockTreeNode"):
        """For heap sorting based on last access time."""
        return self.last_access_time < other.last_access_time

    def is_full_block(self, block_size: int) -> bool:
        """Check if this is a full block."""
        return len(self.token_ids) == block_size


class BlockRadixCache:
    """Block-based Radix Cache for KV Cache reuse."""

    def __init__(
        self,
        block_size: int,
        max_cached_blocks: int = 1000,
        on_block_evict: Optional[Callable[[int], None]] = None,
    ):
        """
        Args:
            block_size: Number of tokens per block
            max_cached_blocks: Maximum number of blocks to cache
            on_block_evict: Callback function when a block is evicted, receives block_id
        """
        self.block_size = block_size
        self.max_cached_blocks = max_cached_blocks
        self.on_block_evict = on_block_evict

        self.root = BlockTreeNode(block_id=None, token_ids=[])
        self.root.lock_ref = 1

        self.num_cached_blocks = 0
        self.request_to_nodes: Dict[str, List[BlockTreeNode]] = {}

        logger.info(
            f"BlockRadixCache initialized: block_size={block_size}, max_cached_blocks={max_cached_blocks}"
        )

    def match_prefix(self, token_ids: List[int]) -> Tuple[List[int], int]:
        """
        Match prefix and find reusable blocks.

        Args:
            token_ids: Complete token sequence

        Returns:
            matched_blocks: List of reusable block IDs
            matched_tokens: Number of matched tokens
        """
        matched_blocks = []
        matched_tokens = 0

        current_node = self.root

        num_full_blocks = len(token_ids) // self.block_size

        for block_idx in range(num_full_blocks):
            block_start = block_idx * self.block_size
            block_end = block_start + self.block_size
            block_tokens = token_ids[block_start:block_end]

            first_token = block_tokens[0]
            if first_token not in current_node.children:
                logger.debug(
                    f"Prefix match stopped at block {block_idx}: first_token {first_token} not in children"
                )
                break

            child_node = current_node.children[first_token]

            if child_node.token_ids != block_tokens:
                logger.debug(
                    f"Prefix match stopped at block {block_idx}: token mismatch. "
                    f"Expected {block_tokens[:5]}..., got {child_node.token_ids[:5]}..."
                )
                break

            matched_blocks.append(child_node.block_id)
            matched_tokens += self.block_size
            current_node = child_node
            current_node.last_access_time = time.monotonic()

        logger.debug(
            f"Prefix match: {matched_tokens}/{len(token_ids)} tokens, "
            f"{len(matched_blocks)} blocks reused"
        )

        return matched_blocks, matched_tokens

    def insert_block(
        self,
        token_ids: List[int],
        block_id: int,
        parent_path: Optional[List[BlockTreeNode]] = None,
        lock: bool = False,
    ) -> BlockTreeNode:
        """
        Insert a full block into the radix tree.

        Args:
            token_ids: Token sequence for this block (must be block_size length)
            block_id: Physical block ID
            parent_path: Parent node path (optional, for faster lookup)
            lock: Whether to lock the node (increment ref count) immediately

        Returns:
            The inserted node
        """
        assert (
            len(token_ids) == self.block_size
        ), f"Token length {len(token_ids)} must equal block_size {self.block_size}"

        if parent_path:
            parent_node = parent_path[-1] if parent_path else self.root
        else:
            parent_node = self.root

        first_token = token_ids[0]

        if first_token in parent_node.children:
            existing_node = parent_node.children[first_token]
            if existing_node.token_ids == token_ids:
                logger.debug(f"Block already exists in cache: {token_ids[:5]}...")
                if lock:
                    existing_node.lock_ref += 1
                    existing_node.last_access_time = time.monotonic()
                return existing_node

        new_node = BlockTreeNode(block_id=block_id, token_ids=token_ids)
        new_node.parent = parent_node
        if lock:
            new_node.lock_ref += 1

        parent_node.children[first_token] = new_node

        self.num_cached_blocks += 1

        logger.debug(
            f"Inserted new block: block_id={block_id}, "
            f"tokens={token_ids[:5]}..., total_cached={self.num_cached_blocks}"
        )

        if self.num_cached_blocks > self.max_cached_blocks:
            self._evict_lru_blocks(self.num_cached_blocks - self.max_cached_blocks)

        return new_node

    def increase_lock_ref(self, nodes: List[BlockTreeNode]):
        """Increase reference count for node path."""
        for node in nodes:
            if node == self.root:
                continue
            node.lock_ref += 1
            node.last_access_time = time.monotonic()

    def decrease_lock_ref(self, nodes: List[BlockTreeNode]):
        """Decrease reference count for node path."""
        for node in nodes:
            if node == self.root:
                continue
            if node.lock_ref > 0:
                node.lock_ref -= 1

            if node.lock_ref == 0:
                logger.debug(
                    f"Node {node.node_id} (block_id={node.block_id}) ref count = 0, evictable"
                )

    def register_request(self, request_id: str, nodes: List[BlockTreeNode]):
        """Register nodes used by request."""
        self.request_to_nodes[request_id] = nodes
        self.increase_lock_ref(nodes)

    def release_request(self, request_id: str):
        """Release request and decrease reference count."""
        if request_id not in self.request_to_nodes:
            return

        nodes = self.request_to_nodes[request_id]
        self.decrease_lock_ref(nodes)
        del self.request_to_nodes[request_id]

        logger.debug(f"Released request {request_id}, decreased ref count for {len(nodes)} nodes")

    def _evict_lru_blocks(self, num_blocks: int):
        """Evict LRU blocks."""
        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_blocks and leaves:
            node = heapq.heappop(leaves)

            if node == self.root:
                break

            if node.lock_ref > 0:
                continue

            self._delete_leaf(node)
            num_evicted += 1

            if node.parent and len(node.parent.children) == 0:
                heapq.heappush(leaves, node.parent)

        logger.info(f"Evicted {num_evicted} blocks from cache")

    def _collect_leaves(self) -> List[BlockTreeNode]:
        """Collect all leaf nodes."""
        leaves = []
        stack = [self.root]

        while stack:
            node = stack.pop()
            if len(node.children) == 0 and node != self.root:
                leaves.append(node)
            else:
                stack.extend(node.children.values())

        return leaves

    def _delete_leaf(self, node: BlockTreeNode):
        """Delete a leaf node and free the physical block."""
        if node.parent:
            for key, child in list(node.parent.children.items()):
                if child == node:
                    del node.parent.children[key]
                    break

        # Free the physical block via callback
        if self.on_block_evict and node.block_id is not None:
            self.on_block_evict(node.block_id)

        self.num_cached_blocks -= 1
        logger.debug(f"Deleted node {node.node_id} (block_id={node.block_id})")

    def pretty_print(self):
        """Print the entire tree structure (for debugging)."""
        self._print_helper(self.root, 0)
        print(f"Total cached blocks: {self.num_cached_blocks}")

    def _print_helper(self, node: BlockTreeNode, indent: int):
        """Recursively print the tree."""
        tokens_preview = node.token_ids[:5] if len(node.token_ids) > 5 else node.token_ids
        print(
            " " * indent + f"Node {node.node_id}: block_id={node.block_id}, "
            f"tokens={tokens_preview}..., ref={node.lock_ref}, "
            f"children={len(node.children)}"
        )
        for child in node.children.values():
            self._print_helper(child, indent + 2)

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "num_cached_blocks": self.num_cached_blocks,
            "max_cached_blocks": self.max_cached_blocks,
            "num_requests": len(self.request_to_nodes),
        }
