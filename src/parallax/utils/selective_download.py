import inspect
import logging
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)
from parallax.utils.weight_filter_utils import (
    determine_needed_weight_files_for_download,
)

# Monkey patch HfApi.repo_info to add short timeout for faster failure on network issues
# This prevents snapshot_download from hanging silently when Hugging Face Hub is unreachable
_original_repo_info = HfApi.repo_info
_REPO_INFO_TIMEOUT = float(os.environ.get("PARALLAX_HF_REPO_INFO_TIMEOUT", "5.0"))


def _repo_info_with_timeout(self, repo_id, repo_type=None, revision=None, **kwargs):
    """Wrapper for HfApi.repo_info that injects a short timeout if not provided."""
    if "timeout" not in kwargs:
        kwargs["timeout"] = _REPO_INFO_TIMEOUT
        logger.debug(f"Injecting timeout={_REPO_INFO_TIMEOUT}s for repo_info call to {repo_id}")
    return _original_repo_info(
        self, repo_id=repo_id, repo_type=repo_type, revision=revision, **kwargs
    )


# Only apply monkey patch if repo_info accepts timeout parameter
_repo_info_signature = inspect.signature(_original_repo_info)
if "timeout" in _repo_info_signature.parameters or "kwargs" in str(_repo_info_signature):
    HfApi.repo_info = _repo_info_with_timeout
    logger.debug(f"Applied monkey patch to HfApi.repo_info with timeout={_REPO_INFO_TIMEOUT}s")
else:
    logger.warning(
        "HfApi.repo_info does not accept 'timeout' parameter - monkey patch skipped. "
        "Network timeout issues may still occur."
    )

EXCLUDE_WEIGHT_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "pytorch_model*.bin",
    "model*.safetensors",
    "weight*.safetensors",
]


def download_metadata_only(
    repo_id: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    local_files_only: bool = False,
) -> Path:
    # If a local path is provided, return it directly without contacting HF Hub
    local_path = Path(repo_id)
    if local_path.exists():
        return local_path

    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        ignore_patterns=EXCLUDE_WEIGHT_PATTERNS,
        force_download=force_download,
        local_files_only=local_files_only,
    )
    return Path(path)


def selective_model_download(
    repo_id: str,
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    local_files_only: bool = False,
) -> Path:
    # Handle local model directory
    local_path = Path(repo_id)
    if local_path.exists():
        model_path = local_path
        logger.debug(f"Using local model path: {model_path}")
        is_remote = False
    else:
        logger.debug(f"Downloading model metadata for {repo_id}")
        model_path = download_metadata_only(
            repo_id=repo_id,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
        )
        logger.debug(f"Downloaded model metadata to {model_path}")
        is_remote = True

    if start_layer is not None and end_layer is not None:
        logger.debug(f"Determining required weight files for layers [{start_layer}, {end_layer})")

        needed_weight_files = determine_needed_weight_files_for_download(
            model_path=model_path,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        if is_remote:
            if not needed_weight_files:
                logger.debug("Could not determine specific weight files, downloading all")
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            else:
                # Step 3: Download only the needed weight files
                logger.info(f"Downloading {len(needed_weight_files)} weight files")

                for weight_file in needed_weight_files:
                    # Check if file already exists in local cache before downloading
                    weight_file_path = model_path / weight_file
                    if weight_file_path.exists():
                        logger.debug(
                            f"Weight file {weight_file} already exists locally, skipping download"
                        )
                        continue

                    logger.debug(f"Downloading {weight_file}")
                    try:
                        hf_hub_download(
                            repo_id=repo_id,
                            filename=weight_file,
                            cache_dir=cache_dir,
                            force_download=force_download,
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
            # Local path: skip any downloads
            logger.debug("Local model path detected; skipping remote weight downloads")
    else:
        # No layer range specified
        if is_remote:
            logger.debug("No layer range specified, downloading all model files")
            snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )
        else:
            logger.debug("No layer range specified and using local path; nothing to download")

    return model_path


def get_model_path_with_selective_download(
    model_path_or_repo: str,
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
    local_files_only: bool = False,
) -> Path:
    return selective_model_download(
        repo_id=model_path_or_repo,
        start_layer=start_layer,
        end_layer=end_layer,
        local_files_only=local_files_only,
    )
