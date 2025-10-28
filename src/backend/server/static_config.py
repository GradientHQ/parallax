import json
from pathlib import Path

from scheduling.model_info import ModelInfo

# Supported model list
MODEL_LIST = [
    "Qwen/Qwen3-0.6B",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2-Instruct-0905",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
    "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
    "Qwen/Qwen3-0.6B-FP8",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-1.7B-FP8",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-FP8",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Instruct-2507-FP8",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-4B-Thinking-2507-FP8",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-FP8",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-14B-FP8",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-32B-FP8",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
    "Qwen/Qwen3-235B-A22B-GPTQ-Int4",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "nvidia/Llama-3.3-70B-Instruct-FP8",
    "nvidia/Llama-3.1-70B-Instruct-FP8",
    "nvidia/Llama-3.1-8B-Instruct-FP8",
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V2",
    "MiniMaxAI/MiniMax-M2",
]

NODE_JOIN_COMMAND_LOCAL_NETWORK = """parallax join"""

NODE_JOIN_COMMAND_PUBLIC_NETWORK = """parallax join -s {scheduler_addr} """


def get_model_info(model_name):
    def _load_config_only(name: str) -> dict:
        local_path = Path(name)
        if local_path.exists():
            config_path = local_path / "config.json"
            with open(config_path, "r") as f:
                return json.load(f)

        # Hugging Face only – download just config.json
        from huggingface_hub import hf_hub_download  # type: ignore

        config_file = hf_hub_download(repo_id=name, filename="config.json")
        with open(config_file, "r") as f:
            return json.load(f)

    config = _load_config_only(model_name)

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

    # Only for hack, fix it when support different quantization bits
    # if "minimax-m2" in model_name.lower():
    #     param_bytes_per_element = 0.5

    # get local experts
    num_local_experts = config.get("num_local_experts", None)
    if num_local_experts is None:
        num_local_experts = config.get("num_experts", None)
    if num_local_experts is None:
        num_local_experts = config.get("n_routed_experts", None)

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
