"""
Loads sharded MLX models from Hugging Face Hub or local paths.
"""

import glob
import importlib
import json
import pathlib
import types
from copy import copy
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_unflatten
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear
from mlx_lm.utils import _download, load_config

from parallax.server.model import ShardedModel
from parallax.utils.model_download import download_model_snapshot
from parallax.utils.tokenizer_utils import load_tokenizer
from parallax.utils.utils import normalize_model_config
from parallax.utils.weight_filter_utils import (
    normalize_language_model_weight_key,
    should_include_weight_key,
)
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

MODEL_CLASS_MAP = {
    "kimi_k2": "mlx_lm.models.deepseek_v3",
    "minimax_m2": "mlx_lm.models.minimax",
    "minimax_m3": "parallax.models.minimax_m3",
    "qwen3_5_moe": "mlx_lm.models.qwen3_5",
}

ARCHITECTURE_CLASS_ALIASES = {
    "GlmMoeDsaForCausalLM": "DeepseekV32ForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration": "MiniMaxM3SparseForCausalLM",
    "Qwen3_5MoeForConditionalGeneration": "Qwen3_5ForConditionalGeneration",
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

        for alias, target in ARCHITECTURE_CLASS_ALIASES.items():
            if target in self.block_class_map:
                self.block_class_map[alias] = self.block_class_map[target]

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

            for layer in model.layers:
                layer.apply_to_modules(get_keys_for_lora)

        for layer in model.layers[-max(num_layers, 0) :]:
            lora_layers = [(k, to_lora(m)) for k, m in layer.named_modules() if k in keys]
            if lora_layers:
                layer.update_modules(tree_unflatten(lora_layers))

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
                downloaded_path = download_model_snapshot(
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

    @staticmethod
    def _to_local_shard_model_key(key: str, start_layer: int) -> str:
        """Convert a global model weight key to the shard-local model key layout."""
        if key.startswith("model.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                parts[2] = str(int(parts[2]) - start_layer)
                return ".".join(parts)
        return key

    @staticmethod
    def _remap_sanitized_key_to_shard(
        key: str,
        *,
        end_layer: int,
        is_first_shard: bool,
        is_last_shard: bool,
        tie_word_embeddings: bool,
    ) -> list[str]:
        if key.startswith("model.layers."):
            parts = key.split(".")
            if len(parts) <= 3 or not parts[2].isdigit():
                return []
            local_layer_idx = int(parts[2])
            if 0 <= local_layer_idx < end_layer:
                return [f"layers.{local_layer_idx}.{'.'.join(parts[3:])}"]
            return []

        remapped_keys = []
        if key.startswith("model.embed_tokens"):
            if is_first_shard:
                remapped_keys.append(key.replace("model.", "", 1))
            if is_last_shard and tie_word_embeddings:
                remapped_keys.append(
                    key.replace("model.", "", 1).replace("embed_tokens", "lm_head", 1)
                )
            return remapped_keys

        if is_last_shard:
            if key.startswith("model.norm"):
                return [key.replace("model.", "", 1)]
            if key.startswith("lm_head"):
                return [key]
        return []

    @staticmethod
    def _make_mlx_lm_sanitizer(arch_module, model_args):
        sanitizer_cls = getattr(arch_module, "TextModel", None)
        if sanitizer_cls is None or not hasattr(sanitizer_cls, "sanitize"):
            sanitizer_cls = getattr(arch_module, "Model", None)
        if sanitizer_cls is None or not hasattr(sanitizer_cls, "sanitize"):
            return None

        sanitizer = sanitizer_cls.__new__(sanitizer_cls)
        sanitizer.args = model_args
        return sanitizer.sanitize

    def _apply_mlx_lm_sanitize(
        self,
        arch_module,
        model_args,
        local_weights: Dict[str, mx.array],
        *,
        num_layers: int,
    ) -> Dict[str, mx.array]:
        if not local_weights:
            return local_weights

        sanitizer_args = copy(model_args)
        if hasattr(sanitizer_args, "num_hidden_layers"):
            sanitizer_args.num_hidden_layers = num_layers

        sanitizer = self._make_mlx_lm_sanitizer(arch_module, sanitizer_args)
        if sanitizer is None:
            return local_weights

        try:
            return sanitizer(local_weights)
        except Exception as e:
            logger.warning("Failed to apply MLX-LM weight sanitize: %s", e)
            return local_weights

    @staticmethod
    def _cast_weight_array(weight_array: mx.array, dtype: mx.Dtype) -> mx.array:
        is_quantized_param = weight_array.dtype in (mx.uint32, mx.int32, mx.uint8)
        if not is_quantized_param and weight_array.dtype != dtype:
            return weight_array.astype(dtype)
        return weight_array

    @staticmethod
    def _load_mlx_lm_module_and_args(
        model_type: str,
        config: Dict[str, Any],
        block_class: Optional[type] = None,
    ):
        if block_class is not None and hasattr(block_class, "prepare_mlx_lm_config"):
            config = block_class.prepare_mlx_lm_config(config)

        if model_type in MODEL_CLASS_MAP:
            model_class = MODEL_CLASS_MAP[model_type]
        else:
            model_class = f"mlx_lm.models.{model_type}"

        arch_module = importlib.import_module(model_class)
        if hasattr(arch_module, "TextModelArgs"):
            model_args_class = getattr(arch_module, "TextModelArgs")
        else:
            model_args_class = getattr(arch_module, "ModelArgs")

        model_args = model_args_class.from_dict(config)
        if block_class is not None and hasattr(block_class, "attach_mlx_lm_model_args"):
            block_class.attach_mlx_lm_model_args(config, model_args)
        return arch_module, model_args

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
            from parallax.utils.model_download import selective_model_download

            logger.info(
                f"Using selective download for layers [{self.start_layer}, {self.end_layer})"
            )
            model_path = selective_model_download(
                self.model_path_str,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                local_files_only=self.use_hfcache,
            )
        else:
            model_path = _download(self.model_path_str)

        config = normalize_model_config(load_config(model_path))
        self.config = config
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
        if hasattr(block_class, "validate_shard_start"):
            block_class.validate_shard_start(config, current_start_layer)

        try:
            arch_module, model_args = self._load_mlx_lm_module_and_args(
                model_type,
                config,
                block_class,
            )
            self.arch_module = arch_module
            self.model_args = model_args

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

        for file_idx, wf in enumerate(weight_files):
            logger.debug(
                f"Scanning weight file {file_idx + 1}/{len(weight_files)}: {pathlib.Path(wf).name}"
            )

            f = mx.load(wf)
            for key in f.keys():
                model_key = normalize_language_model_weight_key(key)
                if should_include_weight_key(
                    model_key,
                    start_layer=current_start_layer,
                    end_layer=current_end_layer,
                    is_first_shard=model_shard.is_first_shard,
                    is_last_shard=model_shard.is_last_shard,
                    tie_word_embeddings=config.get("tie_word_embeddings", False),
                ):
                    local_key = self._to_local_shard_model_key(model_key, current_start_layer)
                    shard_weights[local_key] = f[key]

        sanitized_weights = self._apply_mlx_lm_sanitize(
            arch_module,
            model_args,
            shard_weights,
            num_layers=current_end_layer - current_start_layer,
        )
        if sanitized_weights is not shard_weights:
            shard_weights.clear()

        remapped_shard_weights = {}
        for key, weight_array in sanitized_weights.items():
            remapped_keys = self._remap_sanitized_key_to_shard(
                key,
                end_layer=current_end_layer - current_start_layer,
                is_first_shard=model_shard.is_first_shard,
                is_last_shard=model_shard.is_last_shard,
                tie_word_embeddings=config.get("tie_word_embeddings", False),
            )
            for remapped_key in remapped_keys:
                remapped_shard_weights[remapped_key] = self._cast_weight_array(weight_array, dtype)
        sanitized_weights.clear()

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
                return f"{p}.scales" in remapped_shard_weights

            nn.quantize(
                model_shard,
                group_size=quantization["group_size"],
                bits=quantization["bits"],
                mode=quantization.get("mode", "affine"),
                class_predicate=class_predicate,
            )

        model_shard.load_weights(list(remapped_shard_weights.items()), strict=strict)
        model_shard.shard_layers()

        remapped_shard_weights.clear()

        mx.eval(model_shard.parameters())
        # Synchronize processes to avoid timeout
        mx.eval(mx.distributed.all_sum(mx.array(1.0)))
        model_shard.eval()
        logger.info(
            "Successfully loaded model shard (layers [%d-%d)), memory usage: %.3f GB",
            current_start_layer,
            current_end_layer,
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
        start_layer = model_shard.start_layer
        end_layer = model_shard.end_layer

        for wf in weight_files:
            # Use mx.load for lazy loading
            f = mx.load(wf)
            for key in f.keys():
                model_key = normalize_language_model_weight_key(key)
                if should_include_weight_key(
                    model_key,
                    start_layer=start_layer,
                    end_layer=end_layer,
                    is_first_shard=model_shard.is_first_shard,
                    is_last_shard=model_shard.is_last_shard,
                    tie_word_embeddings=self.config.get("tie_word_embeddings", False),
                ):
                    local_key = self._to_local_shard_model_key(model_key, start_layer)
                    shard_weights[local_key] = f[key]

        arch_module = getattr(self, "arch_module", None)
        model_args = getattr(self, "model_args", getattr(model_shard, "config", None))
        if arch_module is None:
            model_type = self.config.get("model_type")
            model_class = MODEL_CLASS_MAP.get(model_type, f"mlx_lm.models.{model_type}")
            arch_module = importlib.import_module(model_class)

        sanitized_weights = self._apply_mlx_lm_sanitize(
            arch_module,
            model_args,
            shard_weights,
            num_layers=end_layer - start_layer,
        )
        dtype = getattr(model_shard, "dtype", mx.bfloat16)
        remapped_shard_weights = {}
        for key, weight_array in sanitized_weights.items():
            remapped_keys = self._remap_sanitized_key_to_shard(
                key,
                end_layer=end_layer - start_layer,
                is_first_shard=model_shard.is_first_shard,
                is_last_shard=model_shard.is_last_shard,
                tie_word_embeddings=self.config.get("tie_word_embeddings", False),
            )
            for remapped_key in remapped_keys:
                remapped_shard_weights[remapped_key] = self._cast_weight_array(weight_array, dtype)

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
                return f"{p}.scales" in remapped_shard_weights

            nn.quantize(
                model_shard,
                group_size=quantization["group_size"],
                bits=quantization["bits"],
                mode=quantization.get("mode", "affine"),
                class_predicate=class_predicate,
            )

        model_shard.load_weights(list(remapped_shard_weights.items()), strict=False)
        mx.eval(model_shard.parameters())
        model_shard.eval()
        logger.info(
            "Successfully updated model shard from %s, memory usage: %.3f GB",
            refit_weight_path,
            mx.get_active_memory() / 1024**3,
        )
