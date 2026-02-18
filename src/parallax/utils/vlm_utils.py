"""
VLM (Vision Language Model) utilities for MLX backend.

Provides image loading and preprocessing functionality for multimodal models.
"""

import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

# Lazy imports for optional dependencies
_PIL_AVAILABLE = None
_REQUESTS_AVAILABLE = None


def _check_pil():
    global _PIL_AVAILABLE
    if _PIL_AVAILABLE is None:
        try:
            _PIL_AVAILABLE = True
        except ImportError:
            _PIL_AVAILABLE = False
    return _PIL_AVAILABLE


def _check_requests():
    global _REQUESTS_AVAILABLE
    if _REQUESTS_AVAILABLE is None:
        try:
            _REQUESTS_AVAILABLE = True
        except ImportError:
            _REQUESTS_AVAILABLE = False
    return _REQUESTS_AVAILABLE


def load_image(source: Any) -> "Image.Image":
    """
    Load an image from various sources.

    Supports:
        - URL (http/https)
        - Local file path
        - Base64 encoded data URL
        - PIL Image object (pass through)
        - Dict with "url" key

    Args:
        source: Image source (URL, path, base64, or PIL Image)

    Returns:
        PIL Image in RGB format

    Raises:
        ImportError: If PIL is not installed
        ValueError: If source type is not supported
    """
    if not _check_pil():
        raise ImportError(
            "PIL (Pillow) is required for image loading. Install with: pip install Pillow"
        )

    from PIL import Image

    # Handle dict with url key
    if isinstance(source, dict):
        source = source.get("url")

    # Pass through PIL Image
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if not isinstance(source, str):
        raise ValueError(f"Unsupported image source type: {type(source)}")

    # Load from URL
    if source.startswith("http://") or source.startswith("https://"):
        if not _check_requests():
            raise ImportError(
                "requests is required for URL loading. Install with: pip install requests"
            )
        import requests

        response = requests.get(source, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    # Load from base64 data URL
    if source.startswith("data:image"):
        # Format: data:image/png;base64,<base64_data>
        header, encoded = source.split(",", 1)
        image_data = base64.b64decode(encoded)
        return Image.open(BytesIO(image_data)).convert("RGB")

    # Load from local file path
    return Image.open(source).convert("RGB")


def process_images_for_vlm(
    images: List[Any],
    processor: Any,
    text: str = "",
    model_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process images for VLM input using a HuggingFace processor.

    Args:
        images: List of image sources (URLs, paths, base64, or PIL Images)
        processor: HuggingFace processor (e.g., Qwen2VLProcessor, LlavaProcessor)
        text: Text prompt to process alongside images
        model_type: Optional model type for model-specific processing

    Returns:
        Dictionary containing:
            - pixel_values: numpy array of processed images
            - image_grid_thw: (optional) grid sizes for dynamic resolution models
            - image_sizes: (optional) original image sizes
            - input_ids: (optional) tokenized input with image tokens
    """
    if not images:
        return {}

    # Load all images
    pil_images = []
    image_sizes = []
    for img_src in images:
        try:
            img = load_image(img_src)
            pil_images.append(img)
            image_sizes.append((img.height, img.width))
        except Exception as e:
            logger.warning(f"Failed to load image: {e}")
            continue

    if not pil_images:
        logger.warning("No images were successfully loaded")
        return {}

    # Process with HuggingFace processor
    try:
        inputs = processor(
            text=text,
            images=pil_images,
            return_tensors="np",  # Use numpy for MLX compatibility
        )
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        return {}

    result = {"image_sizes": image_sizes}

    # Extract pixel_values
    if hasattr(inputs, "pixel_values"):
        pixel_values = inputs.pixel_values
        if hasattr(pixel_values, "numpy"):
            pixel_values = pixel_values.numpy()
        result["pixel_values"] = pixel_values
    elif "pixel_values" in inputs:
        pixel_values = inputs["pixel_values"]
        if hasattr(pixel_values, "numpy"):
            pixel_values = pixel_values.numpy()
        result["pixel_values"] = pixel_values

    # Extract image_grid_thw (for Qwen-VL style models)
    if hasattr(inputs, "image_grid_thw"):
        grid_thw = inputs.image_grid_thw
        if hasattr(grid_thw, "numpy"):
            grid_thw = grid_thw.numpy()
        result["image_grid_thw"] = grid_thw
    elif "image_grid_thw" in inputs:
        grid_thw = inputs["image_grid_thw"]
        if hasattr(grid_thw, "numpy"):
            grid_thw = grid_thw.numpy()
        result["image_grid_thw"] = grid_thw

    # Extract input_ids if available (some processors expand image tokens)
    if hasattr(inputs, "input_ids"):
        input_ids = inputs.input_ids
        if hasattr(input_ids, "numpy"):
            input_ids = input_ids.numpy()
        result["input_ids"] = input_ids.flatten().tolist()
    elif "input_ids" in inputs:
        input_ids = inputs["input_ids"]
        if hasattr(input_ids, "numpy"):
            input_ids = input_ids.numpy()
        result["input_ids"] = input_ids.flatten().tolist()

    logger.debug(
        f"Processed {len(pil_images)} images, "
        f"pixel_values shape: {result.get('pixel_values', np.array([])).shape}"
    )

    return result


def create_vlm_inputs_from_request(
    image_urls: List[str],
    processor: Any,
    text: str = "",
    model_type: Optional[str] = None,
) -> Optional["VLMInputs"]:
    """
    Create VLMInputs object from image URLs and processor.

    This is a convenience function that combines image processing
    and VLMInputs creation.

    Args:
        image_urls: List of image URLs or paths
        processor: HuggingFace processor
        text: Text prompt
        model_type: Optional model type

    Returns:
        VLMInputs object or None if processing fails
    """
    from parallax.server.request import VLMInputs

    if not image_urls:
        return None

    processed = process_images_for_vlm(
        images=image_urls,
        processor=processor,
        text=text,
        model_type=model_type,
    )

    if not processed or "pixel_values" not in processed:
        return None

    return VLMInputs(
        pixel_values=processed["pixel_values"],
        image_grid_thw=processed.get("image_grid_thw"),
        image_sizes=processed.get("image_sizes"),
        images_processed=False,
    )


def get_image_token_count(
    image_grid_thw: Optional[np.ndarray] = None,
    image_size: Optional[Tuple[int, int]] = None,
    patch_size: int = 14,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
) -> int:
    """
    Calculate the number of image tokens for a given image.

    Different VLM models have different token counting strategies:
    - LLaVA: Fixed number based on image size / patch_size
    - Qwen-VL: Dynamic based on image_grid_thw (temporal * height * width)

    Args:
        image_grid_thw: Grid sizes (temporal, height, width) for Qwen-VL style
        image_size: Image size (height, width) for LLaVA style
        patch_size: Size of each image patch
        merge_size: Merge factor for Qwen-VL (reduces tokens by merge_size^2)
        temporal_patch_size: Temporal patch size for video

    Returns:
        Number of image tokens
    """
    if image_grid_thw is not None:
        # Qwen-VL style: t * h * w tokens
        if isinstance(image_grid_thw, np.ndarray):
            t, h, w = image_grid_thw.flatten()[:3]
        else:
            t, h, w = image_grid_thw
        # After merge: (t * h * w) / (merge_size^2)
        return int((t * h * w) // (merge_size**2))

    if image_size is not None:
        # LLaVA style: (H / patch_size) * (W / patch_size)
        h, w = image_size
        return (h // patch_size) * (w // patch_size)

    # Default fallback
    return 576  # Common default for 336x336 image with 14x14 patches
