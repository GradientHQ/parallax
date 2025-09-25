import json

from huggingface_hub import hf_hub_download

from scheduling.model_info import ModelInfo


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
