"""
Multimodal utilities for SGLang backend.
"""

import logging
from io import BytesIO
from typing import Any, List, Optional, Tuple

import requests
import torch
from PIL import Image
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)

logger = logging.getLogger(__name__)


def load_image(url: Any) -> Image.Image:
    import base64

    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str):
        raise ValueError(f"Unsupported image url type: {type(url)}")

    if url.startswith("http"):
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_data = BytesIO(response.content)
        return Image.open(image_data).convert("RGB")
    elif url.startswith("data:image"):
        header, encoded = url.split(",", 1)
        # Strip whitespace from base64 encoded data (some clients add spaces after comma)
        encoded = encoded.strip()
        image_data = BytesIO(base64.b64decode(encoded))
        return Image.open(image_data).convert("RGB")
    else:
        return Image.open(url).convert("RGB")


def process_images(
    image_urls: List[Any],
    processor: Any,
    input_text: str = "",
    mm_config: Optional[dict] = None,
) -> tuple[List[MultimodalDataItem], Optional[torch.Tensor], List[int]]:
    if not image_urls:
        return [], None, []

    mm_items = []
    images = []

    for url in image_urls:
        try:
            image = load_image(url)
            images.append(image)
            logger.debug(f"Loaded image: size={image.size}, mode={image.mode}")
        except Exception as e:
            logger.exception(f"Failed to load image {url}: {e}")
            continue

    if not images:
        return [], None, []

    try:
        # Check if this is a Kimi K2.5 processor (has different interface)
        processor_class_name = processor.__class__.__name__
        if processor_class_name == "KimiK25Processor":
            # Kimi K2.5 requires special handling:
            # 1. Expand image tokens based on actual image size
            # 2. Use 'medias' parameter instead of 'images'

            # The image token for Kimi K2.5 is <|media_pad|>
            image_token = "<|media_pad|>"

            # Expand single image tokens to the correct number of tokens
            if image_token in input_text and hasattr(processor, "media_processor"):
                parts = input_text.split(image_token)
                result = [parts[0]]
                for i, (image, part) in enumerate(zip(images, parts[1:])):
                    try:
                        # Calculate how many tokens this image needs
                        num_tokens = processor.media_processor.media_tokens_calculator(
                            {"type": "image", "image": image}
                        )
                        logger.debug(f"Kimi K2.5: Image {i} expanded to {num_tokens} tokens")
                    except Exception as e:
                        logger.warning(
                            f"Failed to calculate media tokens for image {i}: {e}, using 1"
                        )
                        num_tokens = 1
                    result.append(image_token * num_tokens + part)
                input_text = "".join(result)
                logger.debug(f"Kimi K2.5: Expanded input_text length: {len(input_text)}")

            # Kimi K2.5 requires 'medias' parameter with specific format
            medias = [{"type": "image", "image": img} for img in images]
            inputs = processor(medias=medias, text=input_text, return_tensors="pt")
        else:
            # Standard HuggingFace processor interface
            inputs = processor(text=input_text, images=images, return_tensors="pt")

        if inputs is None:
            logger.error("Processor returned None")
            return [], None, []

        # Debug: Log all keys returned by the processor
        logger.debug(f"Processor output keys: {list(inputs.keys())}")
        for key, value in inputs.items():
            if hasattr(value, "shape"):
                logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif hasattr(value, "__len__"):
                logger.debug(f"  {key}: len={len(value)}, type={type(value)}")
            else:
                logger.debug(f"  {key}: {type(value)}")

        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            logger.error("Processor output missing pixel_values")
            return [], None, []

        logger.debug(
            f"pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}, device: {pixel_values.device}"
        )
        logger.debug(
            f"pixel_values stats: min={pixel_values.min().item():.4f}, max={pixel_values.max().item():.4f}, mean={pixel_values.mean().item():.4f}"
        )

        expanded_input_ids = inputs.get("input_ids")
        if expanded_input_ids is not None:
            expanded_input_ids = expanded_input_ids.flatten().tolist()
        else:
            expanded_input_ids = []

        logger.debug(f"Processor expanded input_ids length: {len(expanded_input_ids)}")

        # Handle different field names: Kimi K2.5 uses 'grid_thws', others use 'image_grid_thw'
        is_kimi_k25 = processor_class_name == "KimiK25Processor"
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is None:
            image_grid_thw = inputs.get("grid_thws")
        image_sizes = inputs.get("image_sizes")

        logger.debug(f"image_grid_thw: {image_grid_thw}, is_kimi_k25: {is_kimi_k25}")

        # Determine the correct field name for grid data
        # Kimi K2.5 expects 'grid_thws', others expect 'image_grid_thw'
        grid_field_name = "grid_thws" if is_kimi_k25 else "image_grid_thw"

        model_specific_data = {}
        if image_grid_thw is not None:
            model_specific_data[grid_field_name] = image_grid_thw
        if image_sizes is not None:
            model_specific_data["image_sizes"] = image_sizes

        if image_grid_thw is not None and len(image_grid_thw) == len(images):
            num_images = len(images)

            patches_per_image = []
            for i in range(num_images):
                grid = image_grid_thw[i]
                if isinstance(grid, torch.Tensor):
                    num_patches = int(torch.prod(grid).item())
                else:
                    num_patches = int(torch.prod(torch.tensor(grid)).item())
                patches_per_image.append(num_patches)

            patch_start = 0
            for i in range(num_images):
                num_patches = patches_per_image[i]
                item_pixel_values = pixel_values[patch_start : patch_start + num_patches]
                item_grid_thw = image_grid_thw[i : i + 1]

                item = MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=item_pixel_values,
                    model_specific_data={
                        grid_field_name: item_grid_thw,
                    },
                )
                mm_items.append(item)
                patch_start += num_patches
        else:
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=pixel_values,
                model_specific_data=model_specific_data,
            )
            mm_items.append(item)

        return mm_items, image_grid_thw, expanded_input_ids

    except Exception as e:
        logger.exception(f"Failed to process images: {e}")
        return [], None, []


def get_image_token_offsets(
    input_ids: List[int],
    image_token_id: Optional[int],
    vision_start_id: Optional[int] = None,
    vision_end_id: Optional[int] = None,
) -> List[Tuple[int, int]]:
    offsets = []

    if vision_start_id is not None and vision_end_id is not None:
        start_indices = [i for i, tok in enumerate(input_ids) if tok == vision_start_id]
        end_indices = [i for i, tok in enumerate(input_ids) if tok == vision_end_id]

        for start, end in zip(start_indices, end_indices):
            if start < end:
                offsets.append((start + 1, end - 1))
    elif image_token_id is not None:
        start = None
        for i, tok in enumerate(input_ids):
            if tok == image_token_id:
                if start is None:
                    start = i
            elif start is not None:
                offsets.append((start, i - 1))
                start = None
        if start is not None:
            offsets.append((start, len(input_ids) - 1))

    return offsets


def compute_mrope_positions(
    input_ids: List[int],
    image_grid_thw: Optional[torch.Tensor],
    mm_config: dict,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

    import logging

    logger = logging.getLogger(__name__)

    seq_len = len(input_ids)

    def get_default_positions():
        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(3, 1)
        delta = torch.zeros(1, dtype=torch.long)
        return positions, delta

    if image_grid_thw is None:
        return get_default_positions()

    image_token_id = mm_config.get("image_token_id")
    vision_start_id = mm_config.get("vision_start_token_id")
    video_token_id = mm_config.get("video_token_id")
    model_type = mm_config.get("model_type")
    vision_config = mm_config.get("vision_config", {})

    spatial_merge_size = (
        vision_config.get("spatial_merge_size") if isinstance(vision_config, dict) else None
    )
    tokens_per_second = (
        vision_config.get("tokens_per_second") if isinstance(vision_config, dict) else None
    )

    if image_token_id is None or spatial_merge_size is None:
        logger.debug(
            f"Missing mrope config: image_token_id={image_token_id}, "
            f"spatial_merge_size={spatial_merge_size}. Using default positions."
        )
        return get_default_positions()

    try:
        from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=spatial_merge_size,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_id,
            model_type=model_type,
            input_ids=input_ids_tensor,
            image_grid_thw=image_grid_thw,
            tokens_per_second=tokens_per_second,
        )

        mrope_positions = mrope_positions.squeeze(1)
        return mrope_positions, mrope_position_delta

    except Exception as e:
        logger.warning(f"Failed to compute mrope_positions: {e}. Using default positions.")
        return get_default_positions()


def prepare_sglang_multimodal_inputs(
    mm_items: List[MultimodalDataItem],
    image_grid_thw: Optional[torch.Tensor] = None,
    mm_config: Optional[dict] = None,
    input_ids: Optional[List[int]] = None,
) -> MultimodalInputs:
    mm_config = mm_config or {}

    # Extract token IDs from config
    image_token_id = mm_config.get("image_token_id")
    vision_start_id = mm_config.get("vision_start_token_id")
    vision_end_id = mm_config.get("vision_end_token_id")
    video_token_id = mm_config.get("video_token_id")
    audio_token_id = mm_config.get("audio_token_id")

    for item in mm_items:
        if item.pad_value is None:
            item.set_pad_value()

    mrope_positions = None
    mrope_position_delta = None

    if input_ids is not None:
        mrope_positions, mrope_position_delta = compute_mrope_positions(
            input_ids, image_grid_thw, mm_config
        )

    return MultimodalInputs(
        mm_items=mm_items,
        im_token_id=image_token_id,
        im_start_id=vision_start_id,
        im_end_id=vision_end_id,
        video_token_id=video_token_id,
        audio_token_id=audio_token_id,
        mrope_positions=mrope_positions,
        mrope_position_delta=mrope_position_delta,
    )


def process_multimodal_request(
    image_urls: List[Any],
    input_ids: List[int],
    processor: Any,
    tokenizer: Any,
    mm_config: dict,
) -> Tuple[Optional[MultimodalInputs], List[int]]:
    input_text = ""
    if tokenizer is not None:
        try:
            input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
            logger.debug(f"Decoded input text (length={len(input_text)}): {input_text[:100]}...")
        except Exception as e:
            logger.warning(f"Failed to decode input_ids: {e}")

    mm_items, image_grid_thw, expanded_input_ids = process_images(
        image_urls,
        processor,
        input_text=input_text,
        mm_config=mm_config,
    )

    if not mm_items:
        return None, list(input_ids)

    if expanded_input_ids and len(expanded_input_ids) > len(input_ids):
        logger.debug(f"Using expanded input_ids: {len(input_ids)} -> {len(expanded_input_ids)}")
        input_ids_for_offsets = expanded_input_ids
    else:
        input_ids_for_offsets = list(input_ids)

    image_token_id = mm_config.get("image_token_id")
    vision_start_id = mm_config.get("vision_start_token_id")
    vision_end_id = mm_config.get("vision_end_token_id")

    offsets = get_image_token_offsets(
        input_ids_for_offsets,
        image_token_id,
        vision_start_id,
        vision_end_id,
    )

    if len(offsets) == len(mm_items):
        for item, offset in zip(mm_items, offsets):
            item.offsets = [offset]
    elif len(offsets) > 0:
        for item in mm_items:
            item.offsets = offsets

    for item in mm_items:
        item.set_pad_value()

    combined_grid_thw = None
    if image_grid_thw is not None:
        combined_grid_thw = image_grid_thw
    else:
        grids = []
        for item in mm_items:
            grid = item.model_specific_data.get("image_grid_thw")
            if grid is not None:
                grids.append(grid)
        if grids:
            combined_grid_thw = torch.cat(grids, dim=0)

    multimodal_inputs = prepare_sglang_multimodal_inputs(
        mm_items=mm_items,
        image_grid_thw=combined_grid_thw,
        mm_config=mm_config,
        input_ids=input_ids_for_offsets,
    )

    padding_pattern = MultiModalityDataPaddingPatternMultimodalTokens()
    padded_input_ids = padding_pattern.pad_input_tokens(input_ids_for_offsets, multimodal_inputs)

    logger.debug(
        f"Successfully processed {len(mm_items)} images, "
        f"offsets={offsets}, padded_input_ids_len={len(padded_input_ids)}"
    )

    return multimodal_inputs, padded_input_ids
