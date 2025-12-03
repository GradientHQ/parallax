import os
import json
import shutil
from huggingface_hub import snapshot_download

def process_adapter_config(model_id):
    """
    处理 Hugging Face 模型 ID 对应的 adapter_config.json
    """
    # 检查模型 ID 是否在本地存在（通过检查模型目录是否存在）
    # 注意：model_id 作为 Hugging Face 模型 ID，本地目录将创建为 model_id 的简化版本（移除斜杠）
    # local_dir = model_id.replace('/', '_').replace('\\', '_')
    local_dir = model_id   
   
    # 检查本地目录是否存在
    if not os.path.exists(local_dir):
        print(f"模型目录不存在: {local_dir}")
        print(f"正在从 Hugging Face 下载模型: {model_id} -> {local_dir}")
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                revision="main"
            )
            print(f"模型已下载到: {local_dir}")
        except Exception as e:
            raise RuntimeError(f"模型下载失败: {str(e)}")
    
    # 检查 adapter_config.json 是否存在
    adapter_config_path = os.path.join(local_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"未找到 adapter_config.json 文件: {adapter_config_path}")
    
    # 读取 adapter_config.json
    with open(adapter_config_path, 'r') as f:
        config = json.load(f)
    
    # 1. 处理 fine_tune_type 字段
    if "fine_tune_type" not in config:
        peft_type = config.get("peft_type", "lora").lower()
        config["fine_tune_type"] = peft_type
        print(f"添加 fine_tune_type 字段: {peft_type}")
    
    # 2. 处理 num_layers 字段
    if "num_layers" not in config:
        config_path = os.path.join(local_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"未找到 config.json 文件: {config_path}")
        
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        if "num_hidden_layers" not in model_config:
            raise ValueError("config.json 中缺少 num_hidden_layers 字段")
        
        config["num_layers"] = model_config["num_hidden_layers"]
        print(f"添加 num_layers 字段: {model_config['num_hidden_layers']}")
    
    # 3. 处理 lora_parameters 字段
    if "lora_parameters" not in config:
        # 从 config 中提取参数
        r = config.get("r", 8)
        lora_alpha = config.get("lora_alpha", 20.0)
        lora_dropout = config.get("lora_dropout", 0.0)
        
        config["lora_parameters"] = {
            "rank": int(r),
            "scale": float(lora_alpha),
            "dropout": float(lora_dropout)
        }
        print(f"添加 lora_parameters 字段: {config['lora_parameters']}")
    
    # 保存到当前目录
    output_path = os.path.join(os.getcwd(), "adapter_config.json")
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"处理完成! 新的 adapter_config.json 已保存到: {output_path}")
    return output_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python process_adapter.py <Hugging Face 模型ID>")
        print("示例: python process_adapter.py Qwen/Qwen3-0.6B")
        sys.exit(1)
    
    model_id = sys.argv[1]
    try:
        process_adapter_config(model_id)
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)
