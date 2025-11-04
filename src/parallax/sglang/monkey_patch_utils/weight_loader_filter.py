import logging
from pathlib import Path
from typing import List

from parallax.utils.weight_filter_utils import (
    filter_weight_files_by_layer_range as shared_filter,
)

logger = logging.getLogger(__name__)


def filter_weight_files_by_layer_range(
    model_path: Path,
    weight_files: List[str],
    pp_start_layer: int,
    pp_end_layer: int,
    is_first_shard: bool,
    is_last_shard: bool,
) -> List[str]:
    return shared_filter(
        model_path=model_path,
        weight_files=weight_files,
        start_layer=pp_start_layer,
        end_layer=pp_end_layer,
        is_first_shard=is_first_shard,
        is_last_shard=is_last_shard,
        config={},
    )


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
