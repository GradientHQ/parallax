import glob
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx
import transformers
from huggingface_hub import hf_hub_download, snapshot_download


def process_adapter_config(model_id):
    """
    Process the adapter_config.json file associated with a Hugging Face model ID.
    """
    # Check if the model directory already exists locally.
    # Note: The local directory uses the raw `model_id` (slashes are not replaced).
    local_dir = model_id

    # Check if the local directory exists
    if not os.path.exists(local_dir):
        print(f"Model directory does not exist: {local_dir}")
        print(f"Downloading model from Hugging Face: {model_id} -> {local_dir}")
        try:
            snapshot_download(
                repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False, revision="main"
            )
            print(f"Model downloaded to: {local_dir}")
        except Exception as e:
            raise RuntimeError(f"Model download failed: {str(e)}")

    # Check if adapter_config.json exists
    adapter_config_path = os.path.join(local_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"adapter_config.json not found at: {adapter_config_path}")

    # Load adapter_config.json
    with open(adapter_config_path, "r") as f:
        config = json.load(f)

    # 1. Handle the 'fine_tune_type' field
    if "fine_tune_type" not in config:
        peft_type = config.get("peft_type", "lora").lower()
        config["fine_tune_type"] = peft_type
        print(f"Added fine_tune_type field: {peft_type}")

    # 2. Handle the 'num_layers' field
    if "num_layers" not in config:
        config_path = os.path.join(local_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found at: {config_path}")

        with open(config_path, "r") as f:
            model_config = json.load(f)

        if "num_hidden_layers" not in model_config:
            raise ValueError("config.json is missing the 'num_hidden_layers' field")

        config["num_layers"] = model_config["num_hidden_layers"]
        print(f"Added num_layers field: {model_config['num_hidden_layers']}")

    # 3. Handle the 'lora_parameters' field
    if "lora_parameters" not in config:
        # Extract LoRA parameters from config
        r = config.get("r", 8)
        lora_alpha = config.get("lora_alpha", 20.0)
        lora_dropout = config.get("lora_dropout", 0.0)

        config["lora_parameters"] = {
            "rank": int(r),
            "scale": float(lora_alpha),
            "dropout": float(lora_dropout),
        }
        print(f"Added lora_parameters field: {config['lora_parameters']}")

    # Save the updated config to the current working directory
    output_path = os.path.join(os.getcwd(), "adapter_config.json")
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Processing complete! Updated adapter_config.json saved to: {output_path}")
    return output_path


def trans_adapter_config(model_id):
    """
    Process and generate the adapter_config.json file (wrapper function).
    """
    process_adapter_config(model_id)


# Step 2: Convert safetensors files


def fetch_from_hub(path_or_hf_repo: str):
    """
    Load model weights, config, and tokenizer from a local path or Hugging Face Hub.
    If the path does not exist locally, download it from HF first.
    """
    model_path = Path(path_or_hf_repo)

    # Check if it's a local directory
    if not model_path.exists():
        print(f"[INFO] Downloading {path_or_hf_repo} from Hugging Face...")
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
            )
        )
    else:
        print(f"[INFO] Using local model from {model_path}")

    # Load weight files
    weight_files = glob.glob(f"{model_path}/*.safetensors")
    if len(weight_files) == 0:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())

    # Load config and tokenizer
    config = transformers.AutoConfig.from_pretrained(path_or_hf_repo)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path_or_hf_repo)

    return weights, config.to_dict(), tokenizer, model_path


def save_adapter(weights, tokenizer, config):
    """Save the adapter weights into a single adapters.safetensors file."""
    # Save all weights into one file
    mx.save_safetensors("adapters.safetensors", weights, metadata={"format": "mlx"})
    print("[INFO] Saved adapters to adapters.safetensors")


def trans_safetensors(model_path: str):
    print("[INFO] Loading model...")
    weights, config, tokenizer, model_path = fetch_from_hub(model_path)

    # Set data type
    dtype = mx.float16
    weights = {k: v.astype(dtype) for k, v in weights.items()}

    # Save adapter to a single file
    print("[INFO] Saving adapter...")
    save_adapter(weights, tokenizer, config)

    print("[INFO] Conversion complete!")


def download_adapter_config(repo_id):
    adapter_config_path = hf_hub_download(repo_id=repo_id, filename="adapter_config.json")
    output_path = os.path.join(os.getcwd(), "adapter_config.json")
    if os.path.isfile(adapter_config_path):
        shutil.copy(adapter_config_path, output_path)
        return True
    else:
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python process_adapter.py <Hugging Face Model ID> or model_path")
        print("Example: python process_adapter.py Qwen/Qwen3-0.6B or ./xxx")
        sys.exit(1)

    model_id = sys.argv[1]
    try:
        trans_adapter_config(model_id)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    trans_safetensors(model_id)
