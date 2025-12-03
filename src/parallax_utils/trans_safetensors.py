#!/usr/bin/env python3  
 
import argparse  
import copy  
import glob  
import json  
from pathlib import Path  
  
import mlx.core as mx  
import mlx.nn as nn  
import numpy as np  
import transformers  
from huggingface_hub import snapshot_download  
from mlx.utils import tree_flatten, tree_unflatten  
  
  
def fetch_from_hub(path_or_hf_repo: str):  
    """  
    从本地路径或 Hugging Face 下载模型  
    首先检查是否为本地目录，如果不是则从 HF 下载  
    """  
    model_path = Path(path_or_hf_repo)  
      
    # 检查是否为本地目录  
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
      
    # 加载权重文件  
    weight_files = glob.glob(f"{model_path}/*.safetensors")  
    if len(weight_files) == 0:  
        raise FileNotFoundError(f"No safetensors found in {model_path}")  
  
    weights = {}  
    for wf in weight_files:  
        weights.update(mx.load(wf).items())  
  
    # 加载配置和分词器  
    config = transformers.AutoConfig.from_pretrained(path_or_hf_repo)  
    tokenizer = transformers.AutoTokenizer.from_pretrained(path_or_hf_repo)  
      
    return weights, config.to_dict(), tokenizer, model_path  
  
  
def quantize(weights, config, args):  
    """量化模型"""  
    quantized_config = copy.deepcopy(config)  
      
    # 这里需要根据具体模型架构加载模型  
    # 简化版本，实际使用时需要适配具体模型  
    print("[INFO] Quantizing model ( actual model loading would be here )")  
      
    # 更新配置  
    quantized_config["quantization"] = {  
        "group_size": args.q_group_size,  
        "bits": args.q_bits,  
    }  
      
    # 简化处理 - 实际需要模型量化逻辑  
    quantized_weights = {k: v.astype(mx.float16) for k, v in weights.items()}  
      
    return quantized_weights, quantized_config  
  
  
def save_adapter(weights, tokenizer, config):  
    """保存适配器为单个 adapters.safetensors 文件"""  
    # 直接保存所有权重到单个文件  
    mx.save_safetensors(  
        "adapters.safetensors",   
        weights,   
        metadata={"format": "mlx"}  
    )  
    print("[INFO] Saved adapters to adapters.safetensors")  
  
  
def main():  
    parser = argparse.ArgumentParser(  
        description="Convert model to single adapters.safetensors file"  
    )  
    parser.add_argument(  
        "--model-path",  
        type=str,  
        required=True,  
        help="Path to local model directory or Hugging Face repo ID",  
    )  
    parser.add_argument(  
        "-q",  
        "--quantize",  
        help="Generate a quantized adapter",  
        action="store_true",  
    )  
    parser.add_argument(  
        "--q-group-size",  
        help="Group size for quantization",  
        type=int,  
        default=64,  
    )  
    parser.add_argument(  
        "--q-bits",  
        help="Bits per weight for quantization",  
        type=int,  
        default=4,  
    )  
    parser.add_argument(  
        "--dtype",  
        help="Type to save the parameters, ignored if -q is given",  
        type=str,  
        choices=["float16", "bfloat16", "float32"],  
        default="float16",  
    )  
      
    args = parser.parse_args()  
      
    print("[INFO] Loading model...")  
    weights, config, tokenizer, model_path = fetch_from_hub(args.model_path)  
      
    # 设置数据类型  
    dtype = mx.float16 if args.quantize else getattr(mx, args.dtype)  
    weights = {k: v.astype(dtype) for k, v in weights.items()}  
      
    # 量化  
    if args.quantize:  
        print("[INFO] Quantizing...")  
        weights, config = quantize(weights, config, args)  
      
    # 保存适配器到单个文件  
    print("[INFO] Saving adapter...")  
    save_adapter(weights, tokenizer, config)  
      
    print("[INFO] Conversion complete!")  
  
  
if __name__ == "__main__":  
    main()
