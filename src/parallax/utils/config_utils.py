"""
Configuration utilities for handling model configs.

Provides VLM-aware config access for models with text_config/vision_config structure.
"""

from typing import Any, List, Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelConfigAccessor:
    """
    VLM-aware model configuration accessor.

    VLM (Vision-Language Models) typically have a nested config structure:
    - text_config: Contains text model parameters (num_hidden_layers, eos_token_id, etc.)
    - vision_config: Contains vision encoder parameters

    This class provides unified access to config values, automatically handling
    the nested structure for VLM models.
    """

    def __init__(self, config: Any):
        """
        Initialize the config accessor.

        Args:
            config: Model configuration (dict or object with attributes)
        """
        self._config = config
        self._is_vlm: Optional[bool] = None

    @property
    def is_vlm(self) -> bool:
        """Check if the model is a VLM (has both text_config and vision_config)."""
        if self._is_vlm is None:
            text_config = self._raw_get("text_config")
            vision_config = self._raw_get("vision_config")
            self._is_vlm = text_config is not None and vision_config is not None
        return self._is_vlm

    @property
    def text_config(self) -> Optional[Any]:
        """Get the text_config if available."""
        return self._raw_get("text_config")

    @property
    def vision_config(self) -> Optional[Any]:
        """Get the vision_config if available."""
        return self._raw_get("vision_config")

    def _raw_get(self, key: str, default: Any = None) -> Any:
        """
        Low-level config access without VLM-aware logic.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Config value or default
        """
        if isinstance(self._config, dict):
            return self._config.get(key, default)
        return getattr(self._config, key, default)

    def _get_from_subconfig(self, subconfig: Any, key: str) -> Optional[Any]:
        """Get a value from a subconfig (dict or object)."""
        if subconfig is None:
            return None
        if isinstance(subconfig, dict):
            return subconfig.get(key)
        return getattr(subconfig, key, None)

    def get(
        self,
        key: str,
        default: Any = None,
        fallback_keys: Optional[List[str]] = None,
        prefer_text_config: bool = False,
    ) -> Any:
        """
        Get a configuration value with VLM-aware logic.

        For VLM models, text-related parameters (num_hidden_layers, eos_token_id, etc.)
        are typically stored in 'text_config' rather than at the root level.

        Args:
            key: The primary key to look up.
            default: Default value if key is not found anywhere.
            fallback_keys: Alternative keys to try if the primary key is not found.
            prefer_text_config: If True and this is a VLM model, look in text_config first.

        Returns:
            The configuration value or default.
        """
        # For VLM models, prefer text_config for text-related parameters
        if prefer_text_config and self.is_vlm:
            value = self._get_from_subconfig(self.text_config, key)
            if value is not None:
                return value

        # Try primary key at root level
        value = self._raw_get(key)
        if value is not None:
            return value

        # Try fallback keys at root level
        if fallback_keys:
            for fallback_key in fallback_keys:
                value = self._raw_get(fallback_key)
                if value is not None:
                    return value

        # For non-VLM models, also check text_config as fallback
        # (some models might have this structure without being full VLMs)
        if not self.is_vlm and prefer_text_config:
            value = self._get_from_subconfig(self.text_config, key)
            if value is not None:
                return value

        return default

    def get_num_hidden_layers(self) -> Optional[int]:
        """Get the number of hidden layers (common parameter)."""
        return self.get(
            "num_hidden_layers",
            fallback_keys=["n_layer", "num_layers"],
            prefer_text_config=True,
        )

    def get_eos_token_id(self) -> Optional[int]:
        """Get the EOS token ID."""
        return self.get("eos_token_id", prefer_text_config=True)

    def build_mm_config(self) -> dict:
        """
        Build the multimodal configuration dictionary.

        Returns:
            Dictionary containing multimodal-related config values.
        """
        vision_config_raw = self._raw_get("vision_config", {})

        # Normalize vision_config to dict format
        if vision_config_raw is None:
            vision_config = {}
        elif isinstance(vision_config_raw, dict):
            vision_config = vision_config_raw
        else:
            # Convert object-style config to dict
            vision_config = {
                "spatial_merge_size": getattr(vision_config_raw, "spatial_merge_size", None),
                "tokens_per_second": getattr(vision_config_raw, "tokens_per_second", None),
            }

        # Get image_token_id with fallbacks for different models
        # Kimi K2.5 uses 'media_placeholder_token_id' instead of 'image_token_id'
        image_token_id = self._raw_get("image_token_id")
        if image_token_id is None:
            image_token_id = self._raw_get("media_placeholder_token_id")

        return {
            "model_type": self._raw_get("model_type"),
            "image_token_id": image_token_id,
            "vision_start_token_id": self._raw_get("vision_start_token_id"),
            "vision_end_token_id": self._raw_get("vision_end_token_id"),
            "video_token_id": self._raw_get("video_token_id"),
            "audio_token_id": self._raw_get("audio_token_id"),
            "vision_config": vision_config,
        }


# ============================================================================
# Convenience functions for simple use cases
# ============================================================================


def is_vlm_model(config: Any) -> bool:
    """
    Check if the config represents a VLM (Vision-Language Model).

    VLM models have both text_config and vision_config.

    Args:
        config: Model configuration (dict or object)

    Returns:
        True if the model is a VLM
    """
    if isinstance(config, dict):
        text_config = config.get("text_config")
        vision_config = config.get("vision_config")
    else:
        text_config = getattr(config, "text_config", None)
        vision_config = getattr(config, "vision_config", None)

    return text_config is not None and vision_config is not None


def get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """
    Get config value with text_config fallback for VLM models.

    This is a simple function interface for cases where you don't need
    the full ModelConfigAccessor functionality.

    Args:
        config: Model configuration (dict or object)
        key: Configuration key to look up
        default: Default value if not found

    Returns:
        Configuration value or default
    """
    # Try root level first
    if isinstance(config, dict):
        value = config.get(key)
    else:
        value = getattr(config, key, None)

    if value is not None:
        return value

    # Fallback to text_config (for VLM models)
    if isinstance(config, dict):
        text_config = config.get("text_config", {})
    else:
        text_config = getattr(config, "text_config", {})

    if isinstance(text_config, dict):
        return text_config.get(key, default)
    elif text_config is not None:
        return getattr(text_config, key, default)

    return default
