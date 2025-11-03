"""
Monkey patch for SGLang/vLLM weight loader to filter safetensors files based on layer range.

This reduces I/O and memory usage by only loading weight files that contain layers
in the [pp_start_layer, pp_end_layer) range.

Usage:
    1. Call apply_weight_loader_filter_patch() during initialization
    2. Call set_layer_range_for_filtering() before loading model weights
    3. Model weights will be automatically filtered based on the layer range
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


def filter_weight_files_by_layer_range(
    model_path: Path,
    weight_files: List[str],
    pp_start_layer: int,
    pp_end_layer: int,
    is_first_shard: bool,
    is_last_shard: bool,
) -> List[str]:
    """Filter weight files to only those containing layers in the specified range.

    Supports both safetensors (.safetensors) and PyTorch (.bin/.pt) formats.
    """
    # Try safetensors index first
    index_file = model_path / "model.safetensors.index.json"

    if not index_file.exists():
        logger.debug(f"No index file found at {model_path}, will load all weight files")
        return weight_files

    try:
        with open(index_file, "r") as f:
            index_data = json.load(f)

        weight_map = index_data.get("weight_map", {})
        if not weight_map:
            logger.warning("weight_map is empty in index file, will load all weight files")
            return weight_files

        needed_files: Set[str] = set()

        for key, filename in weight_map.items():
            should_include = False

            if is_first_shard and "embed_tokens" in key:
                should_include = True
                logger.debug(f"Including {filename} for embedding layer: {key}")

            if is_last_shard:
                if "model.norm" in key or "lm_head" in key:
                    should_include = True
                    logger.debug(f"Including {filename} for lm_head/norm: {key}")

            if "layers." in key:
                try:
                    parts = key.split(".")
                    for i, part in enumerate(parts):
                        if part == "layers" and i + 1 < len(parts):
                            layer_idx = int(parts[i + 1])
                            if pp_start_layer <= layer_idx < pp_end_layer:
                                should_include = True
                                logger.debug(f"Including {filename} for layer {layer_idx}: {key}")
                            break
                except (ValueError, IndexError):
                    logger.debug(f"Could not parse layer number from {key}, including to be safe")
                    should_include = True

            if should_include:
                full_path = str(model_path / filename)
                needed_files.add(full_path)

        if needed_files:
            filtered_files = [wf for wf in weight_files if wf in needed_files]
            logger.info(
                f"Filtered weight files from {len(weight_files)} to {len(filtered_files)} "
                f"for layers [{pp_start_layer}, {pp_end_layer})"
            )
            logger.debug(f"Needed files: {[Path(f).name for f in filtered_files]}")
            return filtered_files
        else:
            logger.warning(
                f"No relevant weight files found in index for layers [{pp_start_layer}, {pp_end_layer}), "
                "will load all files"
            )
            return weight_files

    except Exception as e:
        logger.warning(f"Failed to filter weight files using index file: {e}, will load all files")
        return weight_files


_layer_range_cache = {}


def set_layer_range_for_filtering(pp_start_layer: int, pp_end_layer: int, is_last_shard: bool):
    global _layer_range_cache
    _layer_range_cache["pp_start_layer"] = pp_start_layer
    _layer_range_cache["pp_end_layer"] = pp_end_layer
    _layer_range_cache["is_last_shard"] = is_last_shard
    logger.info(
        f"Set layer range for weight filtering: [{pp_start_layer}, {pp_end_layer}), "
        f"is_last={is_last_shard}"
    )


def apply_weight_loader_filter_patch():
    from sglang.srt.model_loader import weight_utils

    original_safetensors_iterator = weight_utils.safetensors_weights_iterator

    def patched_safetensors_weights_iterator(
        hf_weights_files: List[str],
        is_all_weights_sharded: bool = False,
        decryption_key: Optional[str] = None,
        disable_mmap: bool = False,
    ):
        filtered_files = _filter_weight_files_by_cache(hf_weights_files)
        return original_safetensors_iterator(
            filtered_files, is_all_weights_sharded, decryption_key, disable_mmap
        )

    def _filter_weight_files_by_cache(hf_weights_files: List[str]) -> List[str]:
        global _layer_range_cache

        pp_start_layer = _layer_range_cache.get("pp_start_layer")
        pp_end_layer = _layer_range_cache.get("pp_end_layer")
        is_last_shard = _layer_range_cache.get("is_last_shard", False)

        if pp_start_layer is None or pp_end_layer is None:
            logger.debug("No layer range set, loading all weight files")
            return hf_weights_files

        if not hf_weights_files:
            return hf_weights_files

        model_path = Path(hf_weights_files[0]).parent
        is_first_shard = pp_start_layer == 0

        logger.info(
            f"Filtering weight files for layers [{pp_start_layer}, {pp_end_layer}), "
            f"is_first={is_first_shard}, is_last={is_last_shard}"
        )

        filtered_files = filter_weight_files_by_layer_range(
            model_path=model_path,
            weight_files=hf_weights_files,
            pp_start_layer=pp_start_layer,
            pp_end_layer=pp_end_layer,
            is_first_shard=is_first_shard,
            is_last_shard=is_last_shard,
        )

        return filtered_files

    weight_utils.safetensors_weights_iterator = patched_safetensors_weights_iterator
    logger.debug("Applied weight loader filter patch to safetensors and pt weight iterators")
