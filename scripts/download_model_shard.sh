#!/bin/bash

# Example usage of the download_shard.py script

# Default values
MODEL_REPO=${1:-"Qwen/Qwen2.5-7B-Instruct"}
START_LAYER=${2:-0}
END_LAYER=${3:-10}
OUTPUT_DIR=${4}  # Optional, defaults to empty/unset

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================================"
echo "Downloading shard for model: $MODEL_REPO"
echo "Layers: [$START_LAYER, $END_LAYER)"
if [ -z "$OUTPUT_DIR" ]; then
    echo "Output Directory: Default Hugging Face Cache"
    OUTPUT_ARG=""
else
    echo "Output Directory: $OUTPUT_DIR"
    OUTPUT_ARG="--output-dir $OUTPUT_DIR"
fi
echo "========================================================"

# Ensure PYTHONPATH includes src
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

python "${SCRIPT_DIR}/download_shard.py" \
    --model-repo "$MODEL_REPO" \
    --start-layer "$START_LAYER" \
    --end-layer "$END_LAYER" \
    $OUTPUT_ARG

if [ $? -eq 0 ]; then
    echo "========================================================"
    echo "Download completed successfully."
    echo "========================================================"
else
    echo "========================================================"
    echo "Download failed."
    echo "========================================================"
    exit 1
fi
