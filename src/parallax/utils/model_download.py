import logging
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download as _hf_hub_download
from huggingface_hub import snapshot_download as _snapshot_download
from modelscope import snapshot_download as _ms_snapshot_download
from modelscope.hub.file_download import model_file_download as _ms_model_file_download

from parallax.utils.weight_filter_utils import (
    determine_needed_weight_files_for_download,
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

        needed_weight_files = determine_needed_weight_files_for_download(
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


def _use_modelscope() -> bool:
    return _USE_MODELSCOPE_ENV in os.environ
