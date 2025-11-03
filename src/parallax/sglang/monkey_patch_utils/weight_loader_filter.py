"""
Monkey patch for SGLang/vLLM weight loader to filter safetensors files based on layer range.

This reduces I/O and memory usage by only loading weight files that contain layers
in the [pp_start_layer, pp_end_layer) range.
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
    """
    Filter weight files based on layer range using model.safetensors.index.json.

    Args:
        model_path: Path to the model directory
        weight_files: List of all weight file paths
        pp_start_layer: Starting layer index (inclusive)
        pp_end_layer: Ending layer index (exclusive)
        is_first_shard: Whether this is the first shard (needs embedding)
        is_last_shard: Whether this is the last shard (needs lm_head and norm)

    Returns:
        Filtered list of weight files containing only relevant layers
    """
    index_file = model_path / "model.safetensors.index.json"

    if not index_file.exists():
        logger.debug(f"No index file found at {index_file}, will load all weight files")
        return weight_files

    try:
        with open(index_file, "r") as f:
            index_data = json.load(f)

        weight_map = index_data.get("weight_map", {})
        if not weight_map:
            logger.warning("weight_map is empty in index file, will load all weight files")
            return weight_files

        needed_files: Set[str] = set()

        # Check which files contain layers/weights we need
        for key, filename in weight_map.items():
            should_include = False

            # Check for embedding layer (first shard)
            if is_first_shard and "embed_tokens" in key:
                should_include = True
                logger.debug(f"Including {filename} for embedding layer: {key}")

            # Check for lm_head and norm (last shard)
            if is_last_shard:
                if "model.norm" in key or "lm_head" in key:
                    should_include = True
                    logger.debug(f"Including {filename} for lm_head/norm: {key}")

            # Check for decoder layers in range
            # Common patterns: "model.layers.0.", "layers.0.", "model.decoder.layers.0."
            if "layers." in key:
                try:
                    # Try to extract layer number from key
                    parts = key.split(".")

                    # Find the "layers" index and get the next part as layer number
                    for i, part in enumerate(parts):
                        if part == "layers" and i + 1 < len(parts):
                            layer_idx = int(parts[i + 1])
                            if pp_start_layer <= layer_idx < pp_end_layer:
                                should_include = True
                                logger.debug(f"Including {filename} for layer {layer_idx}: {key}")
                            break
                except (ValueError, IndexError):
                    # If we can't parse the layer number, include it to be safe
                    logger.debug(f"Could not parse layer number from {key}, including to be safe")
                    should_include = True

            if should_include:
                # Convert relative filename to full path
                full_path = str(model_path / filename)
                needed_files.add(full_path)

        # Filter weight_files to only include needed ones
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


def apply_weight_loader_filter_patch():
    """
    Apply monkey patch to filter weight files before loading.

    This patches the get_model_filenames function in vLLM/SGLang to filter
    out weight files that don't contain layers in the current shard's range.
    """
    try:
        from sglang.srt.model_loader import weight_utils
        from sglang.srt.distributed import get_pp_group

        original_get_model_filenames = weight_utils.get_model_filenames

        def patched_get_model_filenames(model_name_or_path: str, **kwargs):
            """Patched version that filters weight files by layer range."""
            # Get original file list
            weight_files = original_get_model_filenames(model_name_or_path, **kwargs)

            # Try to get PP group info
            try:
                pp_group = get_pp_group()
                if pp_group is None:
                    logger.debug("No PP group found, skipping weight file filtering")
                    return weight_files

                pp_start_layer = getattr(pp_group, "pp_start_layer", None)
                pp_end_layer = getattr(pp_group, "pp_end_layer", None)

                if pp_start_layer is None or pp_end_layer is None:
                    logger.debug(
                        f"PP layer range not set (start={pp_start_layer}, end={pp_end_layer}), "
                        "skipping weight file filtering"
                    )
                    return weight_files

                model_path = Path(model_name_or_path)
                is_first_shard = pp_start_layer == 0

                # We need to know the total number of layers to determine if this is the last shard
                # For now, we'll assume if end_layer is very large, it's the last shard
                # A more robust solution would read the config file
                is_last_shard = False
                try:
                    config_file = model_path / "config.json"
                    if config_file.exists():
                        with open(config_file, "r") as f:
                            config = json.load(f)
                            num_hidden_layers = config.get("num_hidden_layers", 0)
                            is_last_shard = pp_end_layer >= num_hidden_layers
                except Exception as e:
                    logger.debug(f"Could not determine if last shard: {e}")

                logger.info(
                    f"Filtering weight files for shard: layers [{pp_start_layer}, {pp_end_layer}), "
                    f"is_first={is_first_shard}, is_last={is_last_shard}"
                )

                filtered_files = filter_weight_files_by_layer_range(
                    model_path=model_path,
                    weight_files=weight_files,
                    pp_start_layer=pp_start_layer,
                    pp_end_layer=pp_end_layer,
                    is_first_shard=is_first_shard,
                    is_last_shard=is_last_shard,
                )

                return filtered_files

            except Exception as e:
                logger.warning(f"Error in weight file filtering: {e}, using all files")
                return weight_files

        # Apply the patch
        weight_utils.get_model_filenames = patched_get_model_filenames
        logger.info("Applied weight loader filter patch")

    except ImportError as e:
        logger.warning(f"Could not import SGLang weight_utils, skipping patch: {e}")
    except Exception as e:
        logger.error(f"Failed to apply weight loader filter patch: {e}")
