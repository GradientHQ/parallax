import argparse
import os
import sys
from pathlib import Path

# Add src to sys.path to allow importing parallax modules
# Assuming script is in scripts/ directory, so src is at ../src
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.append(str(src_dir))

try:
    from parallax.utils.selective_download import selective_model_download
    from parallax_utils.logging_config import get_logger, set_log_level
except ImportError:
    print(
        f"Error: Could not import parallax modules. Please ensure 'src' directory is in PYTHONPATH or script is located in 'scripts/'. Added path: {src_dir}"
    )
    sys.exit(1)

logger = get_logger("download_shard")


def main():
    parser = argparse.ArgumentParser(
        description="Download specific layers of a model from Hugging Face Hub."
    )
    parser.add_argument(
        "--model-repo", type=str, required=True, help="Hugging Face model repository ID"
    )
    parser.add_argument(
        "--start-layer", type=int, required=True, help="Start layer index (inclusive)"
    )
    parser.add_argument("--end-layer", type=int, required=True, help="End layer index (exclusive)")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default=None,
        help="Local directory to save the model. If not provided, uses default Hugging Face cache.",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()
    set_log_level(args.log_level)

    # Convert output_dir to absolute path if provided
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        logger.info(
            f"Downloading model {args.model_repo} layers [{args.start_layer}, {args.end_layer}) to {output_dir}"
        )
    else:
        output_dir = None
        logger.info(
            f"Downloading model {args.model_repo} layers [{args.start_layer}, {args.end_layer}) to default Hugging Face cache"
        )

    try:
        # Note: selective_model_download uses 'cache_dir' argument which is usually passed to hf_hub_download.
        # hf_hub_download uses cache_dir as the base for its cache structure (models--owner--repo/...).
        # If the user wants to download DIRECTLY to output_dir without the HF cache structure,
        # selective_download might need adjustment or we accept the HF cache structure.
        # Based on selective_download.py implementation:
        # It calls snapshot_download(..., cache_dir=cache_dir) or hf_hub_download(..., cache_dir=cache_dir).
        # So it will create the standard HF cache structure inside output_dir.

        model_path = selective_model_download(
            repo_id=args.model_repo,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            cache_dir=output_dir,
        )
        logger.info(f"Successfully downloaded/verified model shard. Cache location: {model_path}")
    except Exception as e:
        logger.error(f"Failed to download model shard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
