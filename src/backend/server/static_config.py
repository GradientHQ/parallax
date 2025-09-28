import json

from huggingface_hub import hf_hub_download

from scheduling.model_info import ModelInfo

# Supported model list
MODEL_LIST = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-FP8",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-32B-FP8",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "openai/openai/gpt-oss-20b",
    "openai/openai/gpt-oss-120b",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

NODE_JOIN_COMMAND_LOCAL_NETWORK_LINUX_MAC = (
    """bash scripts/join_local.sh -m {model_name} -s {scheduler_addr}"""
)

NODE_JOIN_COMMAND_PUBLIC_NETWORK_LINUX_MAC = """bash scripts/join.sh -m {model_name} -i ${{ip-address-of-current-node}} -s {scheduler_addr}"""

NODE_JOIN_COMMAND_LOCAL_NETWORK_WINDOWS = (
    """parallax join bash scripts/join_local.sh -m {model_name} -s {scheduler_addr}"""
)

NODE_JOIN_COMMAND_PUBLIC_NETWORK_WINDOWS = """parallax join bash scripts/join.sh -m {model_name} -i ${{ip-address-of-current-node}} -s {scheduler_addr}"""


def get_model_info(model_name):
    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        f.close()

    # get quant method
    quant_method = config.get("quant_method", None)
    quantization_config = config.get("quantization_config", None)
    if quant_method is None and quantization_config is not None:
        quant_method = quantization_config.get("quant_method", None)

    if quant_method is None:
        param_bytes_per_element = 2
    elif quant_method == "fp8":
        param_bytes_per_element = 1
    elif quant_method in ("mxfp4", "int4", "awq", "gptq"):
        param_bytes_per_element = 0.5

    model_info = ModelInfo(
        model_name=model_name,
        head_size=config.get("head_dim", 128),
        hidden_dim=config.get("hidden_size", 0),
        intermediate_dim=config.get("intermediate_size", 0),
        num_attention_heads=config.get("num_attention_heads", 0),
        num_kv_heads=config.get("num_key_value_heads", 0),
        vocab_size=config.get("vocab_size", 0),
        num_layers=config.get("num_hidden_layers", 0),
        ffn_num_projections=3,
        param_bytes_per_element=param_bytes_per_element,
        cache_bytes_per_element=2,
        embedding_bytes_per_element=2,
        num_local_experts=config.get("num_experts", None),
        num_experts_per_tok=config.get("num_experts_per_tok", None),
    )
    return model_info


def get_model_list():
    return MODEL_LIST


def get_node_join_command(model_name, scheduler_addr, is_local_network):
    if model_name and scheduler_addr:
        if is_local_network:
            return {
                "linux/mac": NODE_JOIN_COMMAND_LOCAL_NETWORK_LINUX_MAC.format(
                    model_name=model_name, scheduler_addr=scheduler_addr
                ),
                "windows": NODE_JOIN_COMMAND_LOCAL_NETWORK_WINDOWS.format(
                    model_name=model_name, scheduler_addr=scheduler_addr
                ),
            }
        else:
            return {
                "linux/mac": NODE_JOIN_COMMAND_PUBLIC_NETWORK_LINUX_MAC.format(
                    model_name=model_name, scheduler_addr=scheduler_addr
                ),
                "windows": NODE_JOIN_COMMAND_PUBLIC_NETWORK_WINDOWS.format(
                    model_name=model_name, scheduler_addr=scheduler_addr
                ),
            }
    else:
        return None
