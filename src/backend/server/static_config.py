import json

from huggingface_hub import hf_hub_download

from scheduling.model_info import ModelInfo

# Supported model list
MODEL_LIST = [
    "Qwen/Qwen3-0.6B",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2-Instruct-0905",
    "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
    "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
    # "Qwen/Qwen3-8B",
    # "Qwen/Qwen3-8B-FP8",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-32B-FP8",
    # "Qwen/Qwen3-30B-A3B",
    # "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    # "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
    # "Qwen/Qwen2.5-3B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "nvidia/Llama-3.3-70B-Instruct-FP8",
    "nvidia/Llama-3.1-70B-Instruct-FP8",
    "nvidia/Llama-3.1-8B-Instruct-FP8",
]

NODE_JOIN_COMMAND_LOCAL_NETWORK = """parallax join"""

NODE_JOIN_COMMAND_PUBLIC_NETWORK = """parallax join -s {scheduler_addr} """

PUBLIC_RELAY_SERVERS = [
    "/dns4/relay-lattica.gradient.network/udp/18080/quic-v1/p2p/12D3KooWDaqDAsFupYvffBDxjHHuWmEAJE4sMDCXiuZiB8aG8rjf",
    "/dns4/relay-lattica.gradient.network/tcp/18080/p2p/12D3KooWDaqDAsFupYvffBDxjHHuWmEAJE4sMDCXiuZiB8aG8rjf",
]


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

    # get local experts
    num_local_experts = config.get("num_local_experts", None)
    if num_local_experts is None:
        num_local_experts = num_local_experts = config.get("num_experts", None)

    model_info = ModelInfo(
        model_name=model_name,
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
        cache_bytes_per_element=2,
        embedding_bytes_per_element=2,
        num_local_experts=num_local_experts,
        num_experts_per_tok=config.get("num_experts_per_tok", None),
        moe_intermediate_dim=config.get("moe_intermediate_size", None),
    )
    return model_info


def get_model_list():
    return MODEL_LIST


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
