import concurrent.futures
import json
import math
from pathlib import Path

from parallax_utils.logging_config import get_logger
from scheduling.model_info import ModelInfo

logger = get_logger(__name__)

# Supported model list - key: model name, value: MLX model name (same as key if no MLX variant)
MODELS = {
    # =============================== for quickly test ===================================#
    "Qwen/Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    # ======================================= End ========================================#
    #
    #
    #
    # ===============================newly added models===================================#
    # Moonshot Kimi Models
    "moonshotai/Kimi-K2-Instruct": "mlx-community/Kimi-K2-Instruct-4bit",
    "moonshotai/Kimi-K2-Instruct-0905": "mlx-community/Kimi-K2-Instruct-0905-mlx-DQ3_K_M",
    "moonshotai/Kimi-K2-Thinking": "mlx-community/Kimi-K2-Thinking",
    # OpenAI GPT-OSS Models
    "openai/gpt-oss-20b": "mlx-community/gpt-oss-20b-MXFP4-Q8",
    "openai/gpt-oss-120b": "mlx-community/gpt-oss-120b-4bit",
    "openai/gpt-oss-safeguard-20b": "lmstudio-community/gpt-oss-safeguard-20b-MLX-MXFP4",
    "openai/gpt-oss-safeguard-120b": "lmstudio-community/gpt-oss-safeguard-120b-MLX-MXFP4",
    # zai-org GLM4 Models
    "zai-org/GLM-4.6": "mlx-community/GLM-4.6-4bit",
    "zai-org/GLM-4.6-FP8": "mlx-community/GLM-4.6-4bit",
    "zai-org/GLM-4.5-Air": "lmstudio-community/GLM-4.5-Air-MLX-8bit",
    # Other Models
    "MiniMaxAI/MiniMax-M2": "mlx-community/MiniMax-M2-4bit",
    # ======================================= End ========================================#
    #
    #
    #
    # =============================== Major Models =====================================#
    # DeepSeek Models
    "deepseek-ai/DeepSeek-V3.1": "mlx-community/DeepSeek-V3.1-4bit",
    "deepseek-ai/DeepSeek-V3": "mlx-community/DeepSeek-V3-4bit",
    "deepseek-ai/DeepSeek-V2.5-1210": "mlx-community/DeepSeek-V2.5-1210-4bit",
    "deepseek-ai/DeepSeek-R1": "mlx-community/DeepSeek-R1-4bit",
    # Qwen 2.5 Series
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
    # Qwen 3 Series (small models)
    "Qwen/Qwen3-0.6B-FP8": "Qwen/Qwen3-0.6B-MLX-8bit",
    "Qwen/Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-1.7B-FP8": "Qwen/Qwen3-1.7B-MLX-8bit",
    "Qwen/Qwen3-4B": "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-FP8": "Qwen/Qwen3-4B-MLX-8bit",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Instruct-2507-FP8": "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit",
    "Qwen/Qwen3-4B-Thinking-2507": "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-4B-Thinking-2507-FP8": "lmstudio-community/Qwen3-4B-Thinking-2507-MLX-8bit",
    "Qwen/Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-FP8": "Qwen/Qwen3-8B-MLX-8bit",
    "Qwen/Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen/Qwen3-14B-FP8": "Qwen/Qwen3-14B-MLX-8bit",
    "Qwen/Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen/Qwen3-32B-FP8": "Qwen/Qwen3-32B-MLX-8bit",
    # Qwen 3 MoE Models
    "Qwen/Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8": "lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-8bit",
    "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8": "lmstudio-community/Qwen3-30B-A3B-Thinking-2507-MLX-8bit",
    # Qwen 3 Next Series
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit",
    "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8": "mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit",
    "Qwen/Qwen3-Next-80B-A3B-Thinking": "mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit",
    "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8": "mlx-community/Qwen3-Next-80B-A3B-Thinking-8bit",
    # Qwen 3 Large MoE Models
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8": "mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit",
    "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8": "mlx-community/Qwen3-235B-A22B-Thinking-2507-8bit",
    "Qwen/Qwen3-235B-A22B-GPTQ-Int4": "mlx-community/Qwen3-235B-A22B-4bit",
    # Llama Models
    "nvidia/Llama-3.1-8B-Instruct-FP8": "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
    "nvidia/Llama-3.1-70B-Instruct-FP8": "mlx-community/Meta-Llama-3.1-70B-Instruct-8bit",
    "nvidia/Llama-3.3-70B-Instruct-FP8": "mlx-community/Llama-3.3-70B-Instruct-8bit",
    # ======================================= End ========================================#
}

NODE_JOIN_COMMAND_LOCAL_NETWORK = """parallax join"""

NODE_JOIN_COMMAND_PUBLIC_NETWORK = """parallax join -s {scheduler_addr} """


def get_model_info(model_name, use_hfcache: bool = False):
    def _load_config_only(name: str) -> dict:
        local_path = Path(name)
        if local_path.exists():
            config_path = local_path / "config.json"
            with open(config_path, "r") as f:
                return json.load(f)

        # Hugging Face only – download just config.json
        from huggingface_hub import hf_hub_download  # type: ignore

        config_file = hf_hub_download(
            repo_id=name, filename="config.json", local_files_only=use_hfcache
        )
        with open(config_file, "r") as f:
            return json.load(f)

    config = _load_config_only(model_name)

    quant_method = config.get("quant_method", None)
    quantization_config = config.get("quantization_config", None)
    if quant_method is None and quantization_config is not None:
        quant_method = quantization_config.get("quant_method", None)

    if quant_method is None:
        param_bytes_per_element = 2
    elif quant_method == "fp8":
        param_bytes_per_element = 1
    elif quant_method in ("mxfp4", "int4", "awq", "gptq", "compressed-tensors"):
        param_bytes_per_element = 0.5
    else:
        param_bytes_per_element = 1
        logger.warning(
            f"model_name:{model_name} quant_method {quant_method} not supported in get_model_info method"
        )

    mlx_param_bytes_per_element = param_bytes_per_element
    mlx_model_name = MODELS.get(model_name, model_name)

    if mlx_model_name != model_name:
        mlx_config = _load_config_only(mlx_model_name)
        mlx_quant_dict = mlx_config.get("quantization_config", None)
        if mlx_quant_dict and "bits" in mlx_quant_dict:
            mlx_param_bytes_per_element = mlx_quant_dict["bits"] / 8

    # get local experts
    num_local_experts = config.get("num_local_experts", None)
    if num_local_experts is None:
        num_local_experts = config.get("num_experts", None)
    if num_local_experts is None:
        num_local_experts = config.get("n_routed_experts", None)

    model_info = ModelInfo(
        model_name=model_name,
        mlx_model_name=mlx_model_name,
        head_size=config.get("head_dim", 128),
        qk_nope_head_dim=config.get("qk_nope_head_dim", None),
        qk_rope_head_dim=config.get("qk_rope_head_dim", None),
        hidden_dim=config.get("hidden_size", 0),
        intermediate_dim=config.get("intermediate_size", 0),
        num_attention_heads=config.get("num_attention_heads", 0),
        num_kv_heads=config.get("num_key_value_heads", 0),
        vocab_size=config.get("vocab_size", 0),
        num_layers=config.get("num_hidden_layers", 0),
        ffn_num_projections=3,
        param_bytes_per_element=param_bytes_per_element,
        mlx_param_bytes_per_element=mlx_param_bytes_per_element,
        cache_bytes_per_element=2,
        embedding_bytes_per_element=2,
        num_local_experts=num_local_experts,
        num_experts_per_tok=config.get("num_experts_per_tok", None),
        moe_intermediate_dim=config.get("moe_intermediate_size", None),
    )
    return model_info


def get_model_info_with_try_catch(model_name, use_hfcache: bool = False):
    try:
        return get_model_info(model_name, use_hfcache)
    except Exception as e:
        logger.debug(f"Error loading config.json for {model_name}: {e}")
        return None


def get_model_info_dict(use_hfcache: bool = False):
    model_name_list = list(MODELS.keys())
    with concurrent.futures.ThreadPoolExecutor() as executor:
        model_info_dict = dict(
            executor.map(
                lambda name: (name, get_model_info_with_try_catch(name, use_hfcache)),
                model_name_list,
            )
        )
    return model_info_dict


model_info_dict_cache = None


def init_model_info_dict_cache(use_hfcache: bool = False):
    global model_info_dict_cache
    if model_info_dict_cache is not None:
        return
    model_info_dict_cache = get_model_info_dict(use_hfcache)


def get_model_info_dict_cache():
    if model_info_dict_cache is None:
        return {}
    return model_info_dict_cache


def get_model_list():
    model_name_list = list(MODELS.keys())
    model_info_dict = get_model_info_dict_cache()

    def build_single_model(model_name, model_info):
        return {
            "name": model_name,
            "vram_gb": math.ceil(estimate_vram_gb_required(model_info)),
        }

    results = [
        build_single_model(model_name, model_info_dict.get(model_name, None))
        for model_name in model_name_list
    ]
    return results


def estimate_vram_gb_required(model_info):
    if model_info is None:
        return 0

    param_mem_ratio = 0.65
    return (
        (
            model_info.embedding_io_bytes
            + model_info.num_layers * model_info.decoder_layer_io_bytes(roofline=False)
        )
        * 1.0
        / 1024
        / 1024
        / 1024
        / param_mem_ratio
    )


def get_node_join_command(scheduler_addr, is_local_network):
    if scheduler_addr:
        if is_local_network:
            return {
                "command": NODE_JOIN_COMMAND_LOCAL_NETWORK.format(scheduler_addr=scheduler_addr),
            }
        else:
            return {
                "command": NODE_JOIN_COMMAND_PUBLIC_NETWORK.format(scheduler_addr=scheduler_addr),
            }
    else:
        return None
