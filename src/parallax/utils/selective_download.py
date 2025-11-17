import logging
import os
import socket
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
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


def _resolve_hf_endpoint() -> tuple[str, int, bool]:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    parsed = urlparse(endpoint)
    scheme = parsed.scheme or "https"
    host = parsed.netloc or parsed.path or "huggingface.co"
    if ":" in host:
        host_only, port_str = host.split(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            port = 443 if scheme == "https" else 80
        host = host_only
    else:
        port = 443 if scheme == "https" else 80
    use_https = scheme == "https"
    return host, port, use_https


def _quick_hf_reachability_check(repo_id: str, timeout_s: float = 3.0) -> bool:
    host, port, _ = _resolve_hf_endpoint()
    try:
        socket.create_connection((host, port), timeout=timeout_s).close()
        return True
    except OSError:
        # Try an HTTP-level check which may respect proxies
        try:
            endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
            base = endpoint.rstrip("/")
            # We don't care about response code, only that we can reach the server fast
            url = f"{base}/api/models/{repo_id}"
            requests.get(url, timeout=timeout_s, allow_redirects=True)
            return True
        except Exception:
            return False


def _is_network_error(e: Exception) -> bool:
    return (
        isinstance(e, requests.exceptions.RequestException)
        or isinstance(e, OSError)
        or isinstance(e, socket.error)
    )


def _handle_download_exception(
    e: Exception,
    base_fmt: str,
    base_args: tuple,
    network_hint: str | None = None,
) -> None:
    if _is_network_error(e):
        logger.error(base_fmt + " due to network error: %s", *base_args, e)
        if network_hint:
            logger.error(network_hint)
    else:
        logger.error(base_fmt + ": %s", *base_args, e)


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

    try:
        # Quick pre-check to avoid hanging on unreachable networks
        if not local_files_only:
            ok = _quick_hf_reachability_check(repo_id)
            if not ok:
                logger.error(
                    "Cannot reach Hugging Face endpoint before download (pre-check failed). "
                    "This node likely has no egress or DNS to the Hub."
                )
                logger.error(
                    "Please verify network connectivity, proxy settings, firewall rules, or set "
                    "`local_files_only=True` / provide a local model path."
                )
                raise RuntimeError("Hugging Face Hub not reachable from this node")

        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            ignore_patterns=EXCLUDE_WEIGHT_PATTERNS,
            force_download=force_download,
            local_files_only=local_files_only,
        )
        return Path(path)
    except Exception as e:  # broad catch so we can provide a clearer message
        _handle_download_exception(
            e,
            "Failed to download model metadata for %s",
            (repo_id,),
            (
                "This is likely a network/connectivity issue (cannot reach Hugging Face Hub). "
                "Please check your network, proxy settings or pre-download the model locally and "
                "provide a local path instead of a repo id."
            ),
        )
        raise


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
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                except Exception as e:
                    _handle_download_exception(
                        e,
                        "Failed to download all model files for %s",
                        (repo_id,),
                        (
                            "Cannot download model weights because of network issues. "
                            "Check connectivity to Hugging Face Hub or provide local files."
                        ),
                    )
                    raise
            else:
                # Step 3: Download only the needed weight files
                logger.info(f"Downloading {len(needed_weight_files)} weight files")

                for weight_file in needed_weight_files:
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
                        _handle_download_exception(
                            e,
                            "Failed to download weight file %s for %s",
                            (weight_file, repo_id),
                            (
                                "This usually means the node cannot reach Hugging Face Hub. "
                                "Verify network connectivity, firewall/egress rules, or proxy settings."
                            ),
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
            try:
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            except Exception as e:
                _handle_download_exception(
                    e,
                    "Failed to download model %s",
                    (repo_id,),
                    (
                        "Model download failed because the node appears to have no network access to "
                        "Hugging Face Hub. Please check network connectivity or pre-download the model."
                    ),
                )
                raise
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
