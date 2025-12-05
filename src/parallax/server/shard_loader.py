"""
Loads sharded MLX models from Hugging Face Hub or local paths.
"""

import glob
import importlib
import json
import pathlib
import types
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import safetensors
from huggingface_hub import snapshot_download
from mlx import nn
from mlx.utils import tree_unflatten
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear
from mlx_lm.utils import _download, load_config

from parallax.server.model import ShardedModel
from parallax.utils.tokenizer_utils import load_tokenizer
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

MODEL_CLASS_MAP = {
    "kimi_k2": "mlx_lm.models.deepseek_v3",
}


class MLXModelLoader:
    """
    Handles downloading model assets from Hugging Face (if needed) and loading
    a specified shard of an MLX model.
    """

    def __init__(
        self,
        model_path_or_hf_repo: str,
        *,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_hfcache: bool = False,
    ):
        """
        Initializes the model loader.

        Args:
            model_path_or_hf_repo (str): The Hugging Face Hub model ID or a local path
                                         to the model directory.
            start_layer (Optional[int]): The starting layer index for the shard (inclusive).
                                         Defaults to the beginning of the model.
            end_layer (Optional[int]): The ending layer index for the shard (exclusive).
                                       Defaults to the end of the model.
            use_hfcache (bool): If True, use local Hugging Face cache only (no network download).
        """
        self.model_path_str = model_path_or_hf_repo
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.use_hfcache = use_hfcache
        self.register_block_class()

    def register_block_class(self):
        """Automatically read all EntryClass from models directory and generate block class map."""
        self.block_class_map = {}

        # Get models directory path
        models_dir = pathlib.Path(__file__).parent.parent / "models"

        # Find all .py files in models directory (excluding __init__.py)
        model_files = [f for f in models_dir.glob("*.py") if f.name != "__init__.py"]

        for model_file in model_files:
            try:
                # Import the module
                module_name = f"parallax.models.{model_file.stem}"
                module = importlib.import_module(module_name)

                # Get EntryClass from the module
                if hasattr(module, "EntryClass"):
                    entry_class = getattr(module, "EntryClass")

                    # Get architecture from class attribute
                    if hasattr(entry_class, "get_architecture"):
                        architecture = entry_class.get_architecture()
                        self.block_class_map[architecture] = entry_class
                        # logger.info(f"Registered {architecture} -> {entry_class.__name__}")
                    else:
                        logger.warning(f"No architecture attribute found in {entry_class.__name__}")

            except Exception as e:
                logger.warning(f"Failed to load model from {model_file}: {e}")

    def linear_to_lora_layers(
        self,
        model: nn.Module,
        num_layers: int,
        config: Dict,
        use_dora: bool = False,
    ):
        """
        Convert some of the models linear layers to lora layers.

        Args:
            model (nn.Module): The neural network model.
            num_layers (int): The number of blocks to convert to lora layers
            starting from the last layer.
            config (dict): More configuration parameters for LoRA, including the
            rank, scale, and optional layer keys.
            use_dora (bool): If True, uses DoRA instead of LoRA.
            Default: ``False``
        """

        def to_lora(layer):
            if not use_dora and hasattr(layer, "to_lora"):
                return layer.to_lora(
                    r=config["rank"],
                    scale=config["scale"],
                    dropout=config["dropout"],
                )

            if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
                LoRALayer = DoRALinear if use_dora else LoRALinear
            elif isinstance(layer, (SwitchLinear, QuantizedSwitchLinear)):
                if use_dora:
                    raise ValueError(f"{type(layer).__name__} doesn't support DoRA yet.")
                LoRALayer = LoRASwitchLinear
            elif isinstance(layer, (nn.Embedding, nn.QuantizedEmbedding)):
                LoRALayer = DoRAEmbedding if use_dora else LoRAEmbedding
            else:
                raise ValueError(f"Can't convert layer of type {type(layer).__name__} to LoRA")

            return LoRALayer.from_base(
                layer,
                r=config["rank"],
                scale=config["scale"],
                dropout=config["dropout"],
            )

        if (keys := config.get("keys", None)) is None:
            keys = set()

            def get_keys_for_lora(p, m):
                types = (
                    nn.Linear,
                    nn.QuantizedLinear,
                    SwitchLinear,
                    QuantizedSwitchLinear,
                    nn.Embedding,
                    nn.QuantizedEmbedding,
                )
                if hasattr(m, "to_lora") or isinstance(m, types):
                    keys.add(p)

            for l in model.layers:
                l.apply_to_modules(get_keys_for_lora)

        for l in model.layers[-max(num_layers, 0) :]:
            lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in keys]
            if lora_layers:
                l.update_modules(tree_unflatten(lora_layers))

        lora_modules = [(k, to_lora(m)) for k, m in model.named_modules() if k in keys]
        if lora_modules:
            model.update_modules(tree_unflatten(lora_modules))

    def load_lora(self, base_model: nn.Module, adapter_path: str) -> nn.Module:
        """
        Loads LoRA weights from the specified path and applies them to the base model.

        Args:
            adapter_path (str): Path to the LoRA weights file (safetensors format).
            base_model (nn.Module): The base model to which LoRA weights will be applied.

        Returns:
            nn.Module: The base model with LoRA weights applied.
        """

        adapter_path = pathlib.Path(adapter_path)
        if not adapter_path.exists():
            try:
                logger.info(
                    f"Adapter path {adapter_path} not found locally. Attempting to download from Hugging Face..."
                )
                downloaded_path = snapshot_download(
                    repo_id=str(adapter_path), local_dir=str(adapter_path)
                )
                adapter_path = pathlib.Path(downloaded_path)
                logger.info(f"Downloaded adapter to {adapter_path}")
            except Exception as e:
                logger.error(f"Failed to download adapter from Hugging Face: {e}")
                raise FileNotFoundError(
                    f"The adapter path does not exist: {adapter_path}. Download failed: {e}"
                )
        with open(adapter_path / "adapter_config.json", "r") as fid:
            config = types.SimpleNamespace(**json.load(fid))
        fine_tune_type = getattr(config, "fine_tune_type", "lora")
        if fine_tune_type != "full":
            self.linear_to_lora_layers(
                base_model,
                config.num_layers,
                config.lora_parameters,
                use_dora=(fine_tune_type == "dora"),
            )
        base_model.load_weights(str(adapter_path / "adapters.safetensors"), strict=False)
        return base_model

    def load(
        self, lazy: bool = False, strict: bool = True, use_selective_download: bool = True
    ) -> Tuple[nn.Module, Dict[str, Any], Any]:
        """
        Loads the specified model shard by loading only the necessary weights
        from the safetensor files, saving significant memory.

        Args:
            lazy (bool): If False, evaluates model parameters to ensure they are loaded
                         into memory. Defaults to False.
            strict (bool): If True, raises an exception if weights do not match.
                           Defaults to True.
            use_selective_download (bool): If True, only download necessary weight files
                                          from Hugging Face. Defaults to True.
        Returns:
            A tuple containing the loaded sharded MLX model and its configuration dictionary.
        """
        if use_selective_download and self.start_layer is not None and self.end_layer is not None:
            from parallax.utils.selective_download import (
                get_model_path_with_selective_download,
            )

            logger.info(
                f"Using selective download for layers [{self.start_layer}, {self.end_layer})"
            )
            model_path = get_model_path_with_selective_download(
                self.model_path_str,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                local_files_only=self.use_hfcache,
            )
        else:
            model_path = _download(self.model_path_str)

        config = load_config(model_path)
        tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id", None))

        architectures = config.get("architectures", None)
        if architectures is None:
            raise ValueError("architectures not found in config.json")
        if len(architectures) != 1:
            raise ValueError("only one architecture is supported")
        architecture = architectures[0]
        block_class = self.block_class_map.get(architecture, None)
        if block_class is None:
            raise ValueError(f"block_class not found for architecture: {architecture}")

        num_hidden_layers = config.get("num_hidden_layers", 0)
        current_start_layer = self.start_layer if self.start_layer is not None else 0
        current_end_layer = self.end_layer if self.end_layer is not None else num_hidden_layers

        # We need the model object to know its structure and which layers it owns.
        # This part mirrors the logic from the provided utils.py to get model_args.
        model_type = config.get("model_type")
        if not model_type:
            raise ValueError("model_type not found in config.json")

        if model_type in MODEL_CLASS_MAP:
            model_class = MODEL_CLASS_MAP[model_type]
        else:
            model_class = f"mlx_lm.models.{model_type}"

        try:
            arch_module = importlib.import_module(model_class)
            model_args_class = getattr(arch_module, "ModelArgs")
            model_args = model_args_class.from_dict(config)

        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to load architecture for model_type '{model_type}'.") from e

        dtype = getattr(mx, config.get("torch_dtype", "bfloat16"))

        # Extract the base model name from model_id_original if it's a repo ID
        model_id = self.model_path_str
        if "/" in model_id:
            model_id = model_id.split("/")[-1]
        else:  # If it's already a clean name or a local path (take basename)
            model_id = pathlib.Path(model_id).name
        model_shard = ShardedModel(
            config=model_args,
            model_id=model_id,
            start_layer=current_start_layer,
            end_layer=current_end_layer,
            block_class=block_class,
            dtype=dtype,
        )

        weight_files = glob.glob(str(model_path / "model*.safetensors"))
        if not weight_files:
            weight_files = glob.glob(str(model_path / "weight*.safetensors"))

        # Sort weight files by name for consistent loading order
        weight_files = sorted(weight_files)

        # Use shared utility to filter weight files
        from parallax.utils.weight_filter_utils import (
            filter_weight_files_by_layer_range_for_load,
        )

        weight_files = filter_weight_files_by_layer_range_for_load(
            model_path=model_path,
            weight_files=weight_files,
            start_layer=current_start_layer,
            end_layer=current_end_layer,
            is_first_shard=model_shard.is_first_shard,
            is_last_shard=model_shard.is_last_shard,
            config=config,
        )

        if not weight_files and strict:
            raise FileNotFoundError(f"No safetensors found in {model_path}")

        # Instead of loading all weights, we iterate through files and keys,
        # loading only what we need.
        shard_weights = {}
        layer_key_prefix = "model.layers"  # Common prefix

        for file_idx, wf in enumerate(weight_files):
            logger.debug(
                f"Scanning weight file {file_idx + 1}/{len(weight_files)}: {pathlib.Path(wf).name}"
            )

            with safetensors.safe_open(wf, framework="pt") as f:
                for key in f.keys():
                    is_needed = False
                    remapped_key = None

                    # Check if the key belongs to the shard and remap it
                    if (
                        model_shard.is_first_shard
                        and "embed_tokens" in key
                        and key.startswith("model.")
                    ):
                        is_needed = True
                        remapped_key = key.replace("model.", "", 1)
                        if model_shard.is_last_shard and config.get("tie_word_embeddings", False):
                            # Also add lm_head mapping for tied embeddings
                            lm_head_key = remapped_key.replace("embed_tokens", "lm_head")
                            shard_weights[lm_head_key] = mx.array(f.get_tensor(key))
                    elif model_shard.is_last_shard:
                        if "model.norm" in key:
                            is_needed = True
                            remapped_key = key.replace("model.", "", 1)
                        if "lm_head" in key:
                            is_needed = True
                            remapped_key = key
                        elif (
                            config.get("tie_word_embeddings", False)
                            and "embed_tokens" in key
                            and key.startswith("model.embed_tokens")
                        ):
                            is_needed = True
                            remapped_key = key.replace("model.", "", 1).replace(
                                "embed_tokens", "lm_head"
                            )
                    if layer_key_prefix in key:
                        try:
                            parts = key.split(".")
                            layer_idx = int(parts[2])
                            if current_start_layer <= layer_idx < current_end_layer:
                                is_needed = True
                                local_layer_idx = layer_idx - current_start_layer
                                remapped_key = f"layers.{local_layer_idx}.{'.'.join(parts[3:])}"
                        except (ValueError, IndexError):
                            continue

                    # If the key is needed, load only that tensor from the file
                    if is_needed:
                        # Load tensor
                        tensor = f.get_tensor(key)
                        weight_array = mx.array(tensor)

                        # Only convert dtype for non-quantized weights
                        # Quantized weights (uint32, int32) and their scales/biases should keep their original dtype
                        # Scales are typically float32 and should not be downcast to bfloat16
                        is_quantized_param = weight_array.dtype in (mx.uint32, mx.int32, mx.uint8)
                        if not is_quantized_param:
                            weight_array = weight_array.astype(dtype)

                        shard_weights[remapped_key] = weight_array

        if (quantization := config.get("quantization", None)) is not None:
            logger.debug("Model is quantized. Applying quantization parameters...")

            def class_predicate(p, m):
                # Handle custom per-layer quantizations from the config
                qcfg = config.get("quantization", {})
                # Direct key (Parallax remapped keys usually drop the 'model.' prefix)
                if p in qcfg:
                    override = qcfg[p]
                    if isinstance(override, dict):
                        logger.debug(
                            f"[quantize] Using override for '{p}': bits={override.get('bits')} group_size={override.get('group_size')}"
                        )
                    return override
                # Allow config keys that still include the original 'model.' prefix (as in mlx-lm)
                prefixed = f"model.{p}"
                if prefixed in qcfg:
                    override = qcfg[prefixed]
                    return override
                if not hasattr(m, "to_quantized"):
                    return False
                # Handle legacy models by checking if quantized weights exist
                return f"{p}.scales" in shard_weights

            nn.quantize(
                model_shard,
                group_size=quantization["group_size"],
                bits=quantization["bits"],
                mode=quantization.get("mode", "affine"),
                class_predicate=class_predicate,
            )

        model_shard.load_weights(list(shard_weights.items()), strict=strict)

        if not lazy:
            mx.eval(model_shard.parameters())
        model_shard.eval()
        logger.info(
            "Successfully loaded model shard (layers [%d-%d)), memory usage: %.3f GB",
            current_start_layer,
            current_end_layer,
            mx.get_active_memory() / 1024**3,
        )
        return model_shard, config, tokenizer
