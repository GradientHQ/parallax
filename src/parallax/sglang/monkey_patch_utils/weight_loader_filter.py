import json
import logging
from pathlib import Path
from typing import List, Set

logger = logging.getLogger(__name__)


def filter_weight_files_by_layer_range(
    model_path: Path,
    weight_files: List[str],
    pp_start_layer: int,
    pp_end_layer: int,
    is_first_shard: bool,
    is_last_shard: bool,
) -> List[str]:
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
            if filename in needed_files:
                continue
            should_include = False

            if is_first_shard and "embed_tokens" in key:
                should_include = True

            if is_last_shard:
                if "model.norm" in key or "lm_head" in key:
                    should_include = True

            if "layers." in key:
                try:
                    parts = key.split(".")
                    for i, part in enumerate(parts):
                        if part == "layers" and i + 1 < len(parts):
                            layer_idx = int(parts[i + 1])
                            if pp_start_layer <= layer_idx < pp_end_layer:
                                should_include = True
                            break
                except (ValueError, IndexError):
                    # Could not parse layer number, include to be safe
                    should_include = True

            if should_include:
                full_path = str(model_path / filename)
                needed_files.add(full_path)

        if needed_files:
            filtered_files = [wf for wf in weight_files if wf in needed_files]
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

    filtered_files = filter_weight_files_by_layer_range(
        model_path=model_path,
        weight_files=hf_weights_files,
        pp_start_layer=pp_start_layer,
        pp_end_layer=pp_end_layer,
        is_first_shard=is_first_shard,
        is_last_shard=is_last_shard,
    )

    return filtered_files


def apply_weight_loader_filter_patch():
    import glob as glob_module

    original_glob = glob_module.glob

    def patched_glob(pathname, **kwargs):
        files = original_glob(pathname, **kwargs)
        if (
            isinstance(files, list)
            and files
            and any(f.endswith((".safetensors", ".bin", ".pt")) for f in files)
        ):

            # Filter if we have layer range set
            global _layer_range_cache
            if _layer_range_cache.get("pp_start_layer") is not None:
                filtered = _filter_weight_files_by_cache(files)
                return filtered

        return files

    glob_module.glob = patched_glob
