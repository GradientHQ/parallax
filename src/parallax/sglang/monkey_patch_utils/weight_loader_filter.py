import logging
import os
from pathlib import Path
from typing import List

from parallax.utils.weight_filter_utils import (
    filter_weight_files_by_layer_range_for_load,
)

logger = logging.getLogger(__name__)

_layer_range_cache = {}


def set_layer_range_for_filtering(pp_start_layer: int, pp_end_layer: int, num_hidden_layers: int):
    global _layer_range_cache
    _layer_range_cache["pp_start_layer"] = pp_start_layer
    _layer_range_cache["pp_end_layer"] = pp_end_layer
    _layer_range_cache["num_hidden_layers"] = num_hidden_layers


def _filter_weight_files_by_cache(hf_weights_files: List[str]) -> List[str]:
    global _layer_range_cache

    pp_start_layer = _layer_range_cache.get("pp_start_layer")
    pp_end_layer = _layer_range_cache.get("pp_end_layer")
    num_hidden_layers = _layer_range_cache.get("num_hidden_layers")

    if pp_start_layer is None or pp_end_layer is None:
        logger.debug("No layer range set, loading all weight files")
        return hf_weights_files

    if not hf_weights_files:
        return hf_weights_files

    model_path = Path(hf_weights_files[0]).parent
    is_first_shard = pp_start_layer == 0
    is_last_shard = pp_end_layer >= num_hidden_layers

    filtered_files = filter_weight_files_by_layer_range_for_load(
        model_path=model_path,
        weight_files=hf_weights_files,
        start_layer=pp_start_layer,
        end_layer=pp_end_layer,
        is_first_shard=is_first_shard,
        is_last_shard=is_last_shard,
    )

    if os.environ.get("PARALLAX_STEP3P5_DEBUG", "0") == "1":
        logger.warning(
            "[step3p5-debug][weight-filter] model_path=%s range=[%s,%s) hidden_layers=%s "
            "is_first=%s is_last=%s files_in=%d files_out=%d out_names=%s",
            model_path,
            pp_start_layer,
            pp_end_layer,
            num_hidden_layers,
            is_first_shard,
            is_last_shard,
            len(hf_weights_files),
            len(filtered_files),
            [Path(f).name for f in filtered_files],
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
