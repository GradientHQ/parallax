import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from huggingface_hub import hf_hub_download as _hf_hub_download
from huggingface_hub import snapshot_download as _snapshot_download
from modelscope import snapshot_download as _ms_snapshot_download
from modelscope.hub.file_download import model_file_download as _ms_model_file_download

from parallax.utils.weight_filter_utils import (
    normalize_language_model_weight_key,
    should_include_weight_key,
)

logger = logging.getLogger(__name__)
_USE_MODELSCOPE_ENV = "USE_MODELSCOPE"

__all__ = [
    "download_model_file",
    "download_model_snapshot",
    "selective_model_download",
]


def download_model_snapshot(
    repo_id: str,
    allow_patterns: Optional[list[str] | str] = None,
    ignore_patterns: Optional[list[str] | str] = None,
    local_dir: Optional[str | Path] = None,
    local_files_only: bool = False,
) -> Path:
    if _use_modelscope():
        return Path(
            _ms_snapshot_download(
                model_id=repo_id,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                local_dir=str(local_dir) if local_dir is not None else None,
                local_files_only=local_files_only,
            )
        )

    return Path(
        _snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            local_dir=local_dir,
            local_files_only=local_files_only,
        )
    )


def download_model_file(
    repo_id: str,
    filename: str,
    local_files_only: bool = False,
) -> Path:
    if _use_modelscope():
        return Path(
            _ms_model_file_download(
                model_id=repo_id,
                file_path=filename,
                local_files_only=local_files_only,
            )
        )

    return Path(
        _hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_files_only=local_files_only,
        )
    )


def selective_model_download(
    repo_id: str,
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
    local_files_only: bool = False,
) -> Path:
    local_path = Path(repo_id)
    if local_path.exists():
        logger.debug(f"Using local model path: {local_path}")
        return local_path

    logger.debug(f"Downloading model metadata for {repo_id}")
    model_path = download_model_snapshot(
        repo_id=repo_id,
        ignore_patterns=_EXCLUDE_WEIGHT_PATTERNS,
        local_files_only=local_files_only,
    )
    logger.debug(f"Downloaded model metadata to {model_path}")

    if start_layer is not None and end_layer is not None:
        logger.debug(f"Determining required weight files for layers [{start_layer}, {end_layer})")

        needed_weight_files = _determine_needed_weight_files_for_download(
            model_path=model_path,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        if not needed_weight_files:
            logger.debug("Could not determine specific weight files, downloading all")
            download_model_snapshot(repo_id=repo_id, local_files_only=local_files_only)
        else:
            # Step 3: Download only the needed weight files
            logger.info(f"Downloading {len(needed_weight_files)} weight files")

            for weight_file in needed_weight_files:
                # Check if file already exists in local cache before downloading
                weight_file_path = model_path / weight_file
                if weight_file_path.exists():
                    continue

                logger.debug(f"Downloading {weight_file}")
                try:
                    download_model_file(
                        repo_id=repo_id,
                        filename=weight_file,
                        local_files_only=local_files_only,
                    )
                except Exception as e:
                    logger.error(f"Failed to download {weight_file} for {repo_id}: {e}")
                    logger.error(
                        "This node cannot reach Hugging Face Hub to download weight files. "
                        "Please check network connectivity or pre-download the model."
                    )
                    raise

            logger.debug(f"Downloaded weight files for layers [{start_layer}, {end_layer})")
    else:
        logger.debug("No layer range specified, downloading all model files")
        download_model_snapshot(repo_id=repo_id, local_files_only=local_files_only)

    return model_path


_EXCLUDE_WEIGHT_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "pytorch_model*.bin",
    "model*.safetensors",
    "weight*.safetensors",
]


def _determine_needed_weight_files_for_download(
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
            from parallax.utils.utils import normalize_model_config

            with open(config_file, "r") as f:
                cfg = normalize_model_config(json.load(f))
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
        key = normalize_language_model_weight_key(key)
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


def _use_modelscope() -> bool:
    return _USE_MODELSCOPE_ENV in os.environ
