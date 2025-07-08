"""
Simplified KV Cache Manager for Parallax Server

This module implements a simplified key-value (KV) cache system to
avoid materializing the entier KV cache pool.
This is a dictionary-based approach where each request has its own growing KV cache.

Core Components:

KVCache:
    - MLX-LM style growing cache that dynamically allocates memory as needed
    - Supports efficient update and fetch operations
    - Automatically handles memory expansion in chunks

KVCacheManager:
    - Uses a dictionary mapping request_id to KVCache instances
    - Supports adding, updating, releasing requests' KV Cache
    - Performs necessary memory checks to avoid exceeding limits
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from parallax.server.request import Request, RequestStatus
from parallax.server.server_info import HardwareInfo
from parallax.utils.logging_config import get_logger

logger = get_logger(__name__)


class KVCache:
    """Per-Request KV cache for a single request.
    Dynamically grows the cache in chunks of block_size.
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        dtype: mx.Dtype,
        block_size: int = 64,
        num_initial_tokens: int = 128,
    ):
        """
        Args:
            num_kv_heads: The number of key-value heads.
            head_dim: The dimension of each head.
            num_layers: The number of layers.
            dtype: The data type of the cache.
            block_size: Source length dim growth step size.
            num_initial_tokens: The number of tokens to initialize the cache with.
        """
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.block_size = block_size

        num_initial_tokens = self.round_up_to_step(num_initial_tokens)
        # (num_layers, num_kv_heads, seq_len, head_dim)
        self.keys = mx.zeros((num_layers, num_kv_heads, num_initial_tokens, head_dim), dtype)
        self.values = mx.zeros((num_layers, num_kv_heads, num_initial_tokens, head_dim), dtype)
        self.num_tokens = num_initial_tokens
        self.offset = 0

    def round_up_to_step(self, seq_len: int) -> int:
        """
        Rounds up to the nearest multiple of the block_size.
        """
        return (seq_len + self.block_size - 1) // self.block_size * self.block_size

    def needs_grow(self, seq_len: int) -> bool:
        """Checks if the cache needs to grow."""
        return (self.offset + seq_len) > self.num_tokens

    def fetch(self) -> Tuple[mx.array, mx.array]:
        """Fetches the KV cache for the request."""
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def update(self, keys: mx.array, values: mx.array) -> None:
        """
        Updates the cache with new key-value pairs.

        Args:
            keys: New keys to add, shape (num_layers, num_kv_heads, target_len, head_dim)
            values: New values to add, shape (num_layers, num_kv_heads, target_len, head_dim)
        """
        prev = self.offset
        seq_len = keys.shape[2]
        # Grow the cache based on the block_size size
        if self.needs_grow(seq_len):
            num_layers, num_kv_heads, _, head_dim = keys.shape
            n_steps = (self.block_size + seq_len - 1) // self.block_size
            k_shape = (num_layers, num_kv_heads, n_steps * self.block_size, head_dim)
            v_shape = (num_layers, num_kv_heads, n_steps * self.block_size, head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)

            if prev % self.block_size != 0:
                self.keys = self.keys[..., :prev, :]
                self.values = self.values[..., :prev, :]
            self.keys = mx.concatenate([self.keys, new_k], axis=2)
            self.values = mx.concatenate([self.values, new_v], axis=2)
            self.num_tokens = self.keys.shape[2]

        # Update with new keys and values
        self.offset += seq_len
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values


class KVCacheManager:
    """Manager for KVCache instances."""

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        dtype: mx.Dtype,
        block_size: int = 64,
        max_num_tokens: Optional[int] = None,
        cache_memory_fraction: float = 0.5,
    ):
        """
        Args:
            num_kv_heads: The number of key-value heads.
            head_dim: The dimension of each head.
            num_layers: The number of layers.
            dtype: The data type of the cache.
            block_size: Source length dim growth step size.
            max_num_tokens: The maximum number of tokens in the cache.
            cache_memory_fraction: The fraction of memory to use for the cache.
        """
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self.block_size = block_size

        self.request_caches: Dict[str, KVCache] = {}
        self.tokens_in_cache = 0

        self.hw_info = HardwareInfo.detect()
        self.total_ram_gb = self.hw_info.total_ram_gb
        available_cache_size = round(
            (self.total_ram_gb * 1024**3 - mx.get_active_memory()) * cache_memory_fraction
        )
        per_token_cache_size = num_layers * num_kv_heads * head_dim * 2 * dtype.size
        self.max_num_tokens = available_cache_size // per_token_cache_size
        if max_num_tokens is not None:
            self.max_num_tokens = min(self.max_num_tokens, max_num_tokens)

    def round_up_to_step(self, seq_len: int) -> int:
        """
        Rounds up to the nearest multiple of the block_size.
        """
        return (seq_len + self.block_size - 1) // self.block_size * self.block_size

    def has_request(self, request_id: str) -> bool:
        """
        Checks if the request is in the cache.
        """
        return request_id in self.request_caches

    def request_length(self, request_id: str) -> int:
        """
        Returns the length of key/value in the request.
        """
        return self.request_caches[request_id].offset

    def request_num_tokens(self, request_id: str) -> int:
        """
        Returns the number of tokens (including slots not yet filled) in the request.
        """
        assert self.has_request(request_id), "request not in cache"
        return self.request_caches[request_id].num_tokens

    def gather_kv_cache(self, request_id: str) -> Tuple[mx.array, mx.array]:
        """
        Gathers the KV cache for the request.
        """
        assert self.has_request(request_id), "request not in cache"
        return self.request_caches[request_id].fetch()

    def add_request(self, request: Request, num_tokens: int = 128) -> bool:
        """Adds a request to the cache.

        Args:
            request: The request to add.
            num_tokens: The number of tokens in the request.

        Returns:
            True if the request is added.
        """
        assert (
            request.status == RequestStatus.PREFILLING
        ), "add_request can only be called for prefilling requests"

        if request.request_id in self.request_caches:
            logger.warning(f"Request {request.request_id} already in cache")
            return True

        num_tokens = self.round_up_to_step(num_tokens)
        if self.tokens_in_cache + num_tokens > self.max_num_tokens:
            logger.warning(
                f"can't add request {request.request_id} to cache: {self.tokens_in_cache} + "
                f"{num_tokens} > {self.max_num_tokens}"
            )
            return False

        self.request_caches[request.request_id] = KVCache(
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            num_layers=self.num_layers,
            dtype=self.dtype,
            block_size=self.block_size,
            num_initial_tokens=num_tokens,
        )
        self.tokens_in_cache += self.request_num_tokens(request.request_id)
        return True

    def release_request(self, request_id: str) -> bool:
        """
        Releases the request from the cache.
        """
        assert self.has_request(request_id), "request not in cache"
        self.tokens_in_cache -= self.request_num_tokens(request_id)
        del self.request_caches[request_id]
        return True

    def update_requests(
        self, requests: List[Request], keys: mx.array, values: mx.array, lengths: List[int]
    ) -> bool:
        """
        Updates the requests in the cache.

        Args:
            requests: The requests to update.
            keys: The keys to update.
            values: The values to update.
            lengths: The lengths of the requests.

        Returns:
            True if requests are updated.
        """
        batch_size, num_layers, n_kv_heads, _, head_dim = keys.shape
        # Validate
        assert keys.shape == values.shape, "key and value must have the same shape"
        assert num_layers == self.num_layers, "key and value must have the same number of layers"
        assert batch_size == len(requests), "key and value must have the same batch size"
        assert len(lengths) == batch_size, "lengths must have the same batch size as requests"
        assert (
            n_kv_heads == self.num_kv_heads
        ), "key and value must have the same number of key-value heads"
        assert head_dim == self.head_dim, "key and value must have the same head dimension"
        # TODO: Use vmap for better performance
        for request, key, value, length in zip(requests, keys, values, lengths):
            length = length.item()
            assert self.has_request(request.request_id), "request not in cache"
            if self.tokens_in_cache + self.round_up_to_step(length) > self.max_num_tokens:
                logger.warning(
                    f"can't add request {request.request_id} to cache: "
                    f"{self.tokens_in_cache} + {length} > {self.max_num_tokens}"
                )
                return False
            self.request_caches[request.request_id].update(
                key[..., :length, :], value[..., :length, :]
            )
            self.tokens_in_cache += self.round_up_to_step(length)
        return True
