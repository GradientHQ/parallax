from abc import ABC, abstractmethod
from typing import Any, Optional

import mlx.core as mx


class BaseCache(ABC):
    """Abstract base class for layer-level cache."""

    @abstractmethod
    def get_cache(self) -> Any:
        pass

    def get_indexer_cache(self) -> Optional[mx.array]:
        return None

    def get_indexer_cache(self) -> Optional[mx.array]:
        return None
