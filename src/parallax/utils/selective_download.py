import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)


def determine_needed_weight_files(model_path: Path, start_layer: int, end_layer: int):
    from parallax.utils.weight_filter_utils import (
        determine_needed_weight_files as determine_files,
    )

    return determine_files(model_path, start_layer, end_layer)


def selective_model_download(
    repo_id: str,
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    logger.debug(f"Downloading model metadata for {repo_id}")

    ignore_patterns = [
        "*.safetensors",
        "*.bin",
        "*.pt",
        "*.pth",
        "pytorch_model*.bin",
        "model*.safetensors",
    ]

    model_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        ignore_patterns=ignore_patterns,
        force_download=force_download,
    )
    model_path = Path(model_path)
    logger.debug(f"Downloaded model metadata to {model_path}")

    if start_layer is not None and end_layer is not None:
        logger.debug(f"Determining required weight files for layers [{start_layer}, {end_layer})")

        needed_weight_files = determine_needed_weight_files(
            model_path=model_path,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        if not needed_weight_files:
            logger.debug("Could not determine specific weight files, downloading all")
            snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                force_download=force_download,
            )
        else:
            # Step 3: Download only the needed weight files
            logger.info(f"Downloading {len(needed_weight_files)} weight files")

            for weight_file in needed_weight_files:
                logger.debug(f"Downloading {weight_file}")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=weight_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                )

            logger.debug(f"Downloaded weight files for layers [{start_layer}, {end_layer})")
    else:
        logger.debug("No layer range specified, downloading all model files")
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            force_download=force_download,
        )

    return model_path


def get_model_path_with_selective_download(
    model_path_or_repo: str,
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
) -> Path:
    return selective_model_download(
        repo_id=model_path_or_repo,
        start_layer=start_layer,
        end_layer=end_layer,
    )
