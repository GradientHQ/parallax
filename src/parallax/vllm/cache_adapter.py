from typing import Optional


class VLLMCacheAdapter:

    def __init__(self, kv_cache_manager, block_size: int):
        self.kv_cache_manager = kv_cache_manager
        self.block_size = block_size

    def has_request(self, request_id: str) -> bool:
        return False

    def allocate_request(
        self, request_id: str, prompt_len: int, pre_alloc_size: Optional[int] = None
    ) -> bool:
        if self.kv_cache_manager is None:
            return True

        block_pool = self.kv_cache_manager.block_pool
        if block_pool is None:
            return True

        alloc_len = pre_alloc_size if pre_alloc_size is not None else prompt_len

        needed_blocks = (alloc_len + self.block_size - 1) // self.block_size

        num_free_blocks = block_pool.get_num_free_blocks()

        if num_free_blocks >= needed_blocks:
            return True

        return False

    def free_request(self, request_id: str):
        pass

    def release_request(self, request_id: str):
        pass
