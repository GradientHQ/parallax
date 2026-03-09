import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def should_include_weight_key(
    key: str,
    start_layer: int,
    end_layer: int,
    is_first_shard: bool,
    is_last_shard: bool,
    tie_word_embeddings: bool = False,
) -> bool:
    if is_first_shard and "embed_tokens" in key and key.startswith("model."):
        return True

    if is_last_shard:
        if "model.norm" in key or "lm_head" in key:
            return True
        if tie_word_embeddings and "embed" in key and key.startswith("model.embed_tokens"):
            return True

    if "layers." in key:
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                layer_idx = int(parts[i + 1])
                return start_layer <= layer_idx < end_layer

    return False


def filter_weight_files_by_layer_range_for_load(
    model_path: Path,
    weight_files: List[str],
    start_layer: int,
    end_layer: int,
    is_first_shard: bool,
    is_last_shard: bool,
    config: Optional[Dict] = None,
) -> List[str]:
    index_file = model_path / "model.safetensors.index.json"

    if not index_file.exists():
        logger.debug(f"No index file found at {index_file}, cannot filter weight files")
        return weight_files

    with open(index_file, "r") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    if not weight_map:
        logger.debug("weight_map is empty in index file")
        return weight_files

    tie_word_embeddings = False
    if config:
        tie_word_embeddings = config.get("tie_word_embeddings", False)
    else:
        config_file = model_path / "config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                cfg = json.load(f)
                tie_word_embeddings = cfg.get("tie_word_embeddings", False)

    needed_files: Set[str] = set()

    for key, filename in weight_map.items():
        if filename in needed_files:
            continue
        if should_include_weight_key(
            key=key,
            start_layer=start_layer,
            end_layer=end_layer,
            is_first_shard=is_first_shard,
            is_last_shard=is_last_shard,
            tie_word_embeddings=tie_word_embeddings,
        ):
            needed_files.add(filename)

    if not needed_files:
        logger.debug(
            f"No relevant weight files found in index for layers [{start_layer}, {end_layer})"
        )
        return weight_files

    filtered_files = []
    for wf in weight_files:
        wf_name = Path(wf).name
        if wf_name in needed_files:
            filtered_files.append(wf)

    logger.debug(
        f"Filtered weight files from {len(weight_files)} to {len(filtered_files)} "
        f"for layers [{start_layer}, {end_layer})"
    )

    return filtered_files


def determine_needed_weight_files_for_download(
    model_path: Path,
    start_layer: int,
    end_layer: int,
    config: Optional[Dict] = None,
) -> List[str]:
    is_first_shard = start_layer == 0

    is_last_shard = False
    if config:
        num_hidden_layers = config.get("num_hidden_layers", 0)
        is_last_shard = end_layer >= num_hidden_layers
    else:
        config_file = model_path / "config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                cfg = json.load(f)
                num_hidden_layers = cfg.get("num_hidden_layers", 0)
                is_last_shard = end_layer >= num_hidden_layers

    index_file = model_path / "model.safetensors.index.json"

    if not index_file.exists():
        logger.debug(f"Index file not found at {index_file}, checking for single weight file")
        # For non-sharded models, look for single weight file
        single_weight_files = [
            "model.safetensors",
            "pytorch_model.bin",
            "model.bin",
        ]
        for weight_file in single_weight_files:
            if (model_path / weight_file).exists():
                logger.debug(f"Found single weight file: {weight_file}")
                return [weight_file]

        logger.debug("No weight files found (neither index nor single file)")
        return []

    with open(index_file, "r") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    if not weight_map:
        logger.debug("weight_map is empty in index file")
        return []

    tie_word_embeddings = False
    if config:
        tie_word_embeddings = config.get("tie_word_embeddings", False)

    needed_files: Set[str] = set()

    for key, filename in weight_map.items():
        if filename in needed_files:
            continue
        if should_include_weight_key(
            key=key,
            start_layer=start_layer,
            end_layer=end_layer,
            is_first_shard=is_first_shard,
            is_last_shard=is_last_shard,
            tie_word_embeddings=tie_word_embeddings,
        ):
            needed_files.add(filename)

    result = sorted(list(needed_files))
    logger.debug(
        f"Determined {len(result)} weight files needed for layers [{start_layer}, {end_layer})"
    )
    return result
