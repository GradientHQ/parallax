"""
Loads sharded MLX models from Hugging Face Hub or local paths.
"""

import glob
import importlib
import json
import pathlib
import types
from typing import Any, Dict, Optional, Tuple, Type

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx import nn
from mlx.utils import tree_unflatten
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear
from mlx_lm.utils import _download, load_config

from parallax.server.model import ShardedModel
from parallax.utils.config_utils import get_config_value
from parallax.utils.tokenizer_utils import load_tokenizer
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

MODEL_CLASS_MAP = {
    "kimi_k2": "mlx_lm.models.deepseek_v3",
}


VLM_TEXT_CONFIG_MAP = {
    "qwen3_vl": "qwen3",
    "qwen2_vl": "qwen2",
    "qwen2_5_vl": "qwen2",
    "kimi_vl": "deepseek_v3",
}

VLM_SPECIAL_PROJECTOR_MAP = {
    "llava": ("mlx_vlm.models.llava.llava", "LlavaMultiModalProjector"),
    "llava_next": ("mlx_vlm.models.llava_next.llava_next", "LlavaMultiModalProjector"),
    "kimi_vl": ("mlx_vlm.models.kimi_vl.kimi_vl", "KimiVLMultiModalProjector"),
}


def _get_vlm_classes(
    model_type: str, config: Dict[str, Any]
) -> Tuple[Optional[Type[nn.Module]], Optional[Type[nn.Module]], Optional[Dict[str, Any]]]:
    """
    Get VLM-specific classes for a given model type.

    Args:
        model_type: The model type from config.json
        config: Full model configuration

    Returns:
        Tuple of (vision_tower_class, projector_class, vision_config)
        Returns (None, None, None) if not a VLM model
    """
    vision_config = config.get("vision_config")
    if vision_config is None:
        return None, None, None

    try:
        vision_module_path = f"mlx_vlm.models.{model_type}"
        vision_module = importlib.import_module(vision_module_path)
        vision_tower_class = getattr(vision_module, "VisionModel")

        projector_class = None
        if model_type in VLM_SPECIAL_PROJECTOR_MAP:
            proj_module_path, proj_class_name = VLM_SPECIAL_PROJECTOR_MAP[model_type]
            proj_module = importlib.import_module(proj_module_path)
            projector_class = getattr(proj_module, proj_class_name)
            logger.info(f"Loaded VLM classes for {model_type}: VisionModel + {proj_class_name}")
        else:
            logger.info(f"Loaded VLM classes for {model_type}: VisionModel (projector integrated)")

        return vision_tower_class, projector_class, vision_config

    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to load VLM classes for {model_type}: {e}")
        return None, None, None


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
                        logger.debug(f"Registered {architecture} -> {entry_class.__name__}")
                    else:
                        logger.warning(f"No architecture attribute found in {entry_class.__name__}")

            except ImportError as e:
                # Log more details for import errors (often missing dependencies)
                logger.warning(
                    f"Failed to import {model_file.name}: {e} (install required dependencies)"
                )
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
        self, lazy: bool = False, strict: bool = False, use_selective_download: bool = True
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

        # We need the model object to know its structure and which layers it owns.
        # This part mirrors the logic from the provided utils.py to get model_args.
        model_type = config.get("model_type")
        if not model_type:
            raise ValueError("model_type not found in config.json")

        config_for_args = config
        model_class_type = model_type

        if model_type in VLM_TEXT_CONFIG_MAP:
            text_config = config.get("text_config", {})
            if text_config:
                config_for_args = {**config, **text_config}
                if "num_hidden_layers" not in config and "num_hidden_layers" in text_config:
                    config["num_hidden_layers"] = text_config["num_hidden_layers"]
            model_class_type = VLM_TEXT_CONFIG_MAP[model_type]
            logger.info(
                f"VLM model {model_type} using {model_class_type} ModelArgs with text_config"
            )

        num_hidden_layers = config.get("num_hidden_layers", 0)
        current_start_layer = self.start_layer if self.start_layer is not None else 0
        current_end_layer = self.end_layer if self.end_layer is not None else num_hidden_layers

        if model_class_type in MODEL_CLASS_MAP:
            model_class = MODEL_CLASS_MAP[model_class_type]
        else:
            model_class = f"mlx_lm.models.{model_class_type}"

        try:
            arch_module = importlib.import_module(model_class)
            model_args_class = getattr(arch_module, "ModelArgs")
            model_args = model_args_class.from_dict(config_for_args)

        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Failed to load architecture for model_type '{model_type}' (using {model_class})."
            ) from e

        dtype = getattr(mx, config.get("torch_dtype", "bfloat16"))

        # Extract the base model name from model_id_original if it's a repo ID
        model_id = self.model_path_str
        if "/" in model_id:
            model_id = model_id.split("/")[-1]
        else:  # If it's already a clean name or a local path (take basename)
            model_id = pathlib.Path(model_id).name

        vision_tower_class, projector_class, vision_config = _get_vlm_classes(model_type, config)
        is_vlm = vision_config is not None and vision_tower_class is not None

        image_token_index = (
            config.get("image_token_index")
            or config.get("image_token_id")
            or config.get("media_placeholder_token_id")
        )
        vision_feature_layer = config.get("vision_feature_layer", -2)
        vision_feature_select_strategy = config.get("vision_feature_select_strategy", "default")

        if is_vlm:
            logger.info(
                f"Detected VLM model: {model_type}, "
                f"image_token_index={image_token_index}, "
                f"vision_feature_layer={vision_feature_layer}"
            )

        model_shard = ShardedModel(
            config=model_args,
            model_id=model_id,
            start_layer=current_start_layer,
            end_layer=current_end_layer,
            block_class=block_class,
            dtype=dtype,
            # VLM parameters
            vision_config=vision_config if is_vlm else None,
            vision_tower_class=vision_tower_class,
            multi_modal_projector_class=projector_class,
            image_token_index=image_token_index,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
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
            is_vlm=is_vlm,
        )

        if not weight_files and strict:
            raise FileNotFoundError(f"No safetensors found in {model_path}")

        # Instead of loading all weights, we iterate through files and keys,
        # loading only what we need.
        shard_weights = {}

        layer_key_prefixes = [
            ("language_model.model.layers.", 3),  # mlx-vlm style: parts[3] is layer index
            ("model.language_model.layers.", 3),  # HF VLM style: parts[3] is layer index
            ("model.layers.", 2),  # Standard style: parts[2] is layer index
        ]

        vlm_weight_prefixes = [
            "vision_tower.",
            "vision_model.",
            "visual.",  # Qwen-VL style
            "multi_modal_projector.",
            "mm_projector.",
        ]

        tie_word_embeddings = get_config_value(config, "tie_word_embeddings", False)

        for file_idx, wf in enumerate(weight_files):
            logger.debug(
                f"Scanning weight file {file_idx + 1}/{len(weight_files)}: {pathlib.Path(wf).name}"
            )

            f = mx.load(wf)
            for key in f.keys():
                is_needed = False
                remapped_key = None

                if model_shard.is_first_shard and "embed_tokens" in key:
                    is_needed = True
                    if "language_model.model.embed_tokens" in key:
                        remapped_key = key.replace("language_model.model.", "")
                    elif "language_model.embed_tokens" in key:
                        remapped_key = key.split("language_model.")[-1]
                    elif key.startswith("model."):
                        remapped_key = key.replace("model.", "", 1)
                    else:
                        remapped_key = key
                    if model_shard.is_last_shard and tie_word_embeddings:
                        lm_head_key = remapped_key.replace("embed_tokens", "lm_head")
                        shard_weights[lm_head_key] = f[key]

                elif model_shard.is_last_shard:
                    if ".norm." in key or key.endswith(".norm.weight"):
                        is_final_norm = (
                            "language_model.model.norm" in key
                            or "language_model.norm" in key
                            or (key.startswith("model.norm") and "layers" not in key)
                        )
                        if is_final_norm:
                            is_needed = True
                            if "language_model.model.norm" in key:
                                remapped_key = key.replace("language_model.model.", "")
                            elif "language_model.norm" in key:
                                remapped_key = key.split("language_model.")[-1]
                            else:
                                remapped_key = key.replace("model.", "", 1)
                    if "lm_head" in key:
                        is_needed = True
                        if key.startswith("language_model."):
                            remapped_key = key.replace("language_model.", "")
                        else:
                            remapped_key = key
                    elif tie_word_embeddings and "embed_tokens" in key:
                        is_needed = True
                        if "language_model.model.embed_tokens" in key:
                            remapped_key = key.replace("language_model.model.", "").replace(
                                "embed_tokens", "lm_head"
                            )
                        elif "language_model.embed_tokens" in key:
                            remapped_key = key.split("language_model.")[-1].replace(
                                "embed_tokens", "lm_head"
                            )
                        else:
                            remapped_key = key.replace("model.", "", 1).replace(
                                "embed_tokens", "lm_head"
                            )

                # VLM: Load vision tower and projector weights on first shard
                if model_shard.is_first_shard and is_vlm:
                    for prefix in vlm_weight_prefixes:
                        if key.startswith(prefix):
                            is_needed = True
                            remapped_key = key
                            break
                        if key.startswith(f"model.{prefix}"):
                            is_needed = True
                            remapped_key = key.replace("model.", "", 1)
                            break

                if not is_needed:
                    for layer_prefix, layer_idx_pos in layer_key_prefixes:
                        if layer_prefix in key:
                            try:
                                parts = key.split(".")
                                layer_idx = int(parts[layer_idx_pos])
                                if current_start_layer <= layer_idx < current_end_layer:
                                    is_needed = True
                                    local_layer_idx = layer_idx - current_start_layer
                                    # Remap to layers.{local_idx}.{rest}
                                    rest_parts = parts[layer_idx_pos + 1 :]
                                    remapped_key = (
                                        f"layers.{local_layer_idx}.{'.'.join(rest_parts)}"
                                    )
                                break
                            except (ValueError, IndexError):
                                continue

                # If the key is needed, load only that tensor from the file
                if is_needed:
                    # Load tensor (Lazy in MLX)
                    weight_array = f[key]

                    # Only convert dtype for non-quantized weights
                    # Quantized weights (uint32, int32) and their scales/biases should keep their original dtype
                    # Scales are typically float32 and should not be downcast to bfloat16
                    is_quantized_param = weight_array.dtype in (mx.uint32, mx.int32, mx.uint8)
                    if not is_quantized_param and weight_array.dtype != dtype:
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
                # Handle pipeline shards: map local layer index to global index for overrides.
                if p.startswith("layers."):
                    parts = p.split(".")
                    if len(parts) > 2 and parts[1].isdigit():
                        global_idx = int(parts[1]) + current_start_layer
                        global_key = "model.layers." + str(global_idx) + "." + ".".join(parts[2:])
                        if global_key in qcfg:
                            override = qcfg[global_key]
                            if isinstance(override, dict):
                                logger.debug(
                                    f"[quantize] Using override for '{global_key}' (local '{p}'): bits={override.get('bits')} group_size={override.get('group_size')}"
                                )
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

        # Log weight keys before loading
        logger.info(
            f"Loading {len(shard_weights)} weights. Sample keys: {list(shard_weights.keys())[:20]}"
        )

        # Try strict mode first to catch any mismatch, then fall back to non-strict
        try:
            model_shard.load_weights(list(shard_weights.items()), strict=True)
        except Exception as e:
            logger.warning(f"Strict weight loading failed: {e}. Retrying with strict=False.")
            model_shard.load_weights(list(shard_weights.items()), strict=False)
        model_shard.shard_layers()

        # Log VLM-specific weight loading info
        if is_vlm and model_shard.is_first_shard:
            vlm_weight_count = sum(
                1
                for k in shard_weights.keys()
                if any(
                    k.startswith(p)
                    for p in [
                        "vision_tower",
                        "vision_model",
                        "visual",
                        "multi_modal_projector",
                        "mm_projector",
                    ]
                )
            )
            logger.info(f"Loaded {vlm_weight_count} VLM weights (vision_tower + projector)")

        logger.info(f"Total weights loaded: {len(shard_weights)}")

        shard_weights.clear()

        mx.eval(model_shard.parameters())
        # Synchronize processes to avoid timeout
        mx.eval(mx.distributed.all_sum(mx.array(1.0)))
        model_shard.eval()

        vlm_info = f", VLM={is_vlm}" if is_vlm else ""
        logger.info(
            "Successfully loaded model shard (layers [%d-%d)%s), memory usage: %.3f GB",
            current_start_layer,
            current_end_layer,
            vlm_info,
            mx.get_active_memory() / 1024**3,
        )
        return model_shard, config, tokenizer

    def update_weight_from_disk(self, model_shard: nn.Module, refit_weight_path: str):
        """Runtime weight refit from disk"""
        weight_files = glob.glob(refit_weight_path + "/*.safetensors")
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {refit_weight_path}")

        logger.info(f"Begin refit weight from path: {refit_weight_path}")
        shard_weights = {}
        layer_key_prefix = "model.layers"  # Common prefix

        for wf in weight_files:
            # Use mx.load for lazy loading
            f = mx.load(wf)
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
                    if model_shard.is_last_shard and self.config.get("tie_word_embeddings", False):
                        shard_weights["lm_head.weight"] = f[key]
                elif model_shard.is_last_shard:
                    if "model.norm" in key:
                        is_needed = True
                        remapped_key = key.replace("model.", "", 1)
                    if "lm_head" in key:
                        is_needed = True
                        remapped_key = key
                    elif (
                        self.config.get("tie_word_embeddings", False)
                        and "embed" in key
                        and key.startswith("model.embed_tokens")
                    ):
                        # TODO: we don't need load lm_head in this case
                        # as we will pass hidden_states to FirstPeer
                        # see request.py for details
                        is_needed = True
                        remapped_key = "lm_head.weight"
                if layer_key_prefix in key:
                    try:
                        parts = key.split(".")
                        layer_idx = int(parts[2])
                        if self.start_layer <= layer_idx < self.end_layer:
                            is_needed = True
                            local_layer_idx = layer_idx - self.start_layer
                            remapped_key = f"layers.{local_layer_idx}.{'.'.join(parts[3:])}"
                    except (ValueError, IndexError):
                        continue

                # If the key is needed, load only that tensor from the file
                if is_needed:
                    shard_weights[remapped_key] = f[key]

        if (quantization := self.config.get("quantization", None)) is not None:
            logger.info("Model is quantized. Applying quantization parameters...")

            def class_predicate(p, m):
                # Handle custom per-layer quantizations from the config
                qcfg = self.config.get("quantization", {})
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
                    if isinstance(override, dict):
                        logger.debug(
                            f"[quantize] Using override for '{prefixed}' (mapped to '{p}'): bits={override.get('bits')} group_size={override.get('group_size')}"
                        )
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

        model_shard.load_weights(list(shard_weights.items()), strict=False)
        mx.eval(model_shard.parameters())
        model_shard.eval()
        logger.info(
            "Successfully updated model shard from %s, memory usage: %.3f GB",
            refit_weight_path,
            mx.get_active_memory() / 1024**3,
        )
