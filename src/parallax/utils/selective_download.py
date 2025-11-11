import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)
from parallax.utils.weight_filter_utils import (
    determine_needed_weight_files_for_download,
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
) -> Path:
    # If a local path is provided, return it directly without contacting HF Hub
    local_path = Path(repo_id)
    if local_path.exists():
        return local_path

    # Try local cache first, then download if needed
    try:
        # First attempt: try to use local cache (offline-first)
        logger.debug(f"Attempting to load metadata from local cache for {repo_id}")
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            ignore_patterns=EXCLUDE_WEIGHT_PATTERNS,
            force_download=False,
            local_files_only=True,
        )
        logger.debug(f"Successfully loaded metadata from local cache for {repo_id}")
        return Path(path)
    except Exception as e:
        # If local cache fails, try to download from Hugging Face
        logger.info(
            f"Failed to load metadata from local cache for {repo_id}: {e}. "
            f"Attempting to download from Hugging Face..."
        )
        try:
            path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                ignore_patterns=EXCLUDE_WEIGHT_PATTERNS,
                force_download=force_download,
                local_files_only=False,
            )
            logger.info(f"Successfully downloaded metadata from Hugging Face for {repo_id}")
            return Path(path)
        except Exception as download_error:
            logger.error(
                f"Failed to download metadata from Hugging Face for {repo_id}: {download_error}. "
                f"Local cache error: {e}"
            )
            raise RuntimeError(
                f"Cannot load model metadata for {repo_id}. "
                f"Local cache failed: {e}. "
                f"Network download also failed: {download_error}. "
                f"Please ensure you have network access or the model is cached locally."
            ) from download_error


def selective_model_download(
    repo_id: str,
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
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
                # Try local cache first, then download if needed
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=cache_dir,
                        force_download=False,
                        local_files_only=True,
                    )
                    logger.debug(f"Successfully loaded all files from local cache for {repo_id}")
                except Exception as e:
                    logger.info(
                        f"Failed to load all files from local cache for {repo_id}: {e}. "
                        f"Attempting to download from Hugging Face..."
                    )
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=False,
                    )
            else:
                # Step 3: Download only the needed weight files
                logger.info(f"Downloading {len(needed_weight_files)} weight files")

                for weight_file in needed_weight_files:
                    logger.debug(f"Downloading {weight_file}")
                    # Try local cache first, then download if needed
                    try:
                        hf_hub_download(
                            repo_id=repo_id,
                            filename=weight_file,
                            cache_dir=cache_dir,
                            force_download=False,
                            local_files_only=True,
                        )
                        logger.debug(f"Successfully loaded {weight_file} from local cache")
                    except Exception as e:
                        logger.info(
                            f"Failed to load {weight_file} from local cache: {e}. "
                            f"Attempting to download from Hugging Face..."
                        )
                        hf_hub_download(
                            repo_id=repo_id,
                            filename=weight_file,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            local_files_only=False,
                        )

                logger.debug(f"Downloaded weight files for layers [{start_layer}, {end_layer})")
        else:
            # Local path: skip any downloads
            logger.debug("Local model path detected; skipping remote weight downloads")
    else:
        # No layer range specified
        if is_remote:
            logger.debug("No layer range specified, downloading all model files")
            # Try local cache first, then download if needed
            try:
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    force_download=False,
                    local_files_only=True,
                )
                logger.debug(f"Successfully loaded all files from local cache for {repo_id}")
            except Exception as e:
                logger.info(
                    f"Failed to load all files from local cache for {repo_id}: {e}. "
                    f"Attempting to download from Hugging Face..."
                )
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=False,
                )
        else:
            logger.debug("No layer range specified and using local path; nothing to download")

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
