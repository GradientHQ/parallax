from typing import List, Set

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class BlockAllocator:
    """Manages allocation of physical block indices."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Initialize free blocks stack
        self.free_blocks: List[int] = list(range(num_blocks))
        self.used_blocks: Set[int] = set()

    def allocate(self, num_blocks_needed: int) -> List[int]:
        """Allocates `num_blocks_needed` physical blocks."""
        if len(self.free_blocks) < num_blocks_needed:
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


class SlotAllocator:
    """Manages allocation of request slots (indices)."""

    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self.free_slots: List[int] = list(range(num_slots))
        self.used_slots: Set[int] = set()

    def allocate(self) -> int:
        """Allocates a single slot."""
        if not self.free_slots:
            return -1
        slot = self.free_slots.pop()
        self.used_slots.add(slot)
        return slot

    def free(self, slot: int):
        """Frees the given slot."""
        if slot in self.used_slots:
            self.used_slots.remove(slot)
            self.free_slots.append(slot)
        else:
            logger.warning(f"Double free detected for slot {slot}")

    def get_num_free_slots(self) -> int:
        return len(self.free_slots)
