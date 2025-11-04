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
    is_last_shard = num_hidden_layers is not None and pp_end_layer >= num_hidden_layers

    logger.debug(
        f"Filtering weight files: start_layer={pp_start_layer}, end_layer={pp_end_layer}, "
        f"is_first_shard={is_first_shard}, is_last_shard={is_last_shard}, "
        f"input_files={len(hf_weights_files)}"
    )

    filtered_files = filter_weight_files_by_layer_range(
        model_path=model_path,
        weight_files=hf_weights_files,
        pp_start_layer=pp_start_layer,
        pp_end_layer=pp_end_layer,
        is_first_shard=is_first_shard,
        is_last_shard=is_last_shard,
    )

    logger.debug(
        f"Filtered to {len(filtered_files)} files: {[Path(f).name for f in filtered_files]}"
    )
    return filtered_files


def apply_weight_loader_filter_patch():
    import glob as glob_module

    original_glob = glob_module.glob

    def patched_glob(pathname, **kwargs):
        files = original_glob(pathname, **kwargs)
        logger.debug(
            f"patched_glob called: pathname={pathname}, num_files={len(files) if isinstance(files, list) else 'N/A'}"
        )

        if (
            isinstance(files, list)
            and files
            and any(f.endswith((".safetensors", ".bin", ".pt")) for f in files)
        ):
            logger.debug(f"Found weight files, checking layer range cache...")
            # Filter if we have layer range set
            global _layer_range_cache
            if _layer_range_cache.get("pp_start_layer") is not None:
                logger.debug(
                    f"Layer range set: start={_layer_range_cache.get('pp_start_layer')}, end={_layer_range_cache.get('pp_end_layer')}"
                )
                filtered = _filter_weight_files_by_cache(files)
                logger.debug(f"Filtered from {len(files)} to {len(filtered)} weight files")
                return filtered
            else:
                logger.debug("Layer range not set, loading all weight files")

        return files

    glob_module.glob = patched_glob

    # Patch os.listdir
    import os

    original_listdir = os.listdir

    def patched_listdir(path):
        files = original_listdir(path)
        logger.debug(f"patched_listdir called: path={path}, num_files={len(files)}")

        # Convert to full paths for filtering
        if any(f.endswith((".safetensors", ".bin", ".pt")) for f in files):
            logger.debug(f"Found weight files in listdir")
            global _layer_range_cache
            if _layer_range_cache.get("pp_start_layer") is not None:
                full_paths = [
                    os.path.join(path, f)
                    for f in files
                    if f.endswith((".safetensors", ".bin", ".pt"))
                ]
                if full_paths:
                    filtered_paths = _filter_weight_files_by_cache(full_paths)
                    filtered_names = [os.path.basename(f) for f in filtered_paths]
                    # Keep non-weight files
                    result = [
                        f for f in files if not f.endswith((".safetensors", ".bin", ".pt"))
                    ] + filtered_names
                    logger.debug(f"Filtered listdir from {len(files)} to {len(result)} files")
                    return result

        return files

    os.listdir = patched_listdir

    # Patch Path.glob
    from pathlib import Path as PathlibPath

    original_path_glob = PathlibPath.glob

    def patched_path_glob(self, pattern):
        files = list(original_path_glob(self, pattern))
        logger.debug(f"patched_path_glob called: pattern={pattern}, num_files={len(files)}")

        if files and any(str(f).endswith((".safetensors", ".bin", ".pt")) for f in files):
            logger.debug(f"Found weight files in Path.glob")
            global _layer_range_cache
            if _layer_range_cache.get("pp_start_layer") is not None:
                str_files = [
                    str(f) for f in files if str(f).endswith((".safetensors", ".bin", ".pt"))
                ]
                if str_files:
                    filtered_strs = _filter_weight_files_by_cache(str_files)
                    filtered_paths = [PathlibPath(f) for f in filtered_strs]
                    # Keep non-weight files
                    result = [
                        f for f in files if not str(f).endswith((".safetensors", ".bin", ".pt"))
                    ] + filtered_paths
                    logger.debug(f"Filtered Path.glob from {len(files)} to {len(result)} files")
                    return iter(result)

        return iter(files)

    PathlibPath.glob = patched_path_glob

    # Patch safetensors.torch.load_file to intercept actual file loading
    try:
        import safetensors.torch

        original_load_file = safetensors.torch.load_file

        def patched_load_file(filename, *args, **kwargs):
            logger.debug(f"patched_load_file called: filename={filename}")
            return original_load_file(filename, *args, **kwargs)

        safetensors.torch.load_file = patched_load_file
    except ImportError:
        logger.debug("safetensors module not available for patching")

    # Patch json.load to intercept index file reading
    import json
    import builtins

    original_open = builtins.open

    def patched_open(file, mode="r", *args, **kwargs):
        result = original_open(file, mode, *args, **kwargs)
        if isinstance(file, str) and file.endswith(".index.json"):
            logger.debug(f"patched_open called for index file: {file}")
        return result

    builtins.open = patched_open
