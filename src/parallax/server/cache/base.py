from abc import ABC, abstractmethod
from typing import Any


class BaseCache(ABC):
    """Abstract base class for layer-level cache."""

    @abstractmethod
    def get_cache(self) -> Any:
        pass
