from typing import Optional


class SGLangCacheAdapter:
    """
    Adapter for SGLang memory pool to be used with Parallax Scheduler.
    """

    def __init__(self, model_runner):
        self.model_runner = model_runner

    def has_request(self, request_id: str) -> bool:
        return False

    def allocate_request(
        self, request_id: str, prompt_len: int, pre_alloc_size: Optional[int] = None
    ) -> bool:

        allocator = self.model_runner.token_to_kv_pool_allocator
        if allocator is None:
            return True

        alloc_len = pre_alloc_size if pre_alloc_size is not None else prompt_len
        available_size = allocator.available_size()

        if available_size >= alloc_len:
            return True

        return False

    def free_request(self, request_id: str):
        pass

    def release_request(self, request_id: str):
        pass
