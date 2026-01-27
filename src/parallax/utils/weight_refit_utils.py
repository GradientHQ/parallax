import base64
import glob
import hashlib
import json
import os
import shutil
import struct
from typing import Dict

import torch
from safetensors.torch import save_file

# CID constants
CIDV1 = 0x01
RAW_CODEC = 0x55
SHA2_256_CODE = 0x12
SHA2_256_SIZE = 0x20  # 32 bytes


def calculate_cid_manual(data: bytes) -> str:
    sha256_digest = hashlib.sha256(data).digest()
    multihash = bytes([SHA2_256_CODE, SHA2_256_SIZE]) + sha256_digest
    cid_bytes = bytes([CIDV1, RAW_CODEC]) + multihash
    base32_str = base64.b32encode(cid_bytes).decode("ascii").lower().rstrip("=")
    cid_string = "b" + base32_str
    return cid_string


def inplace_insert_value_with_idx(tensor_list, value, idx):
    while len(tensor_list) < idx + 1:
        tensor_list.append(None)
    tensor_list[idx] = value


def concat_weight_partition(original_tensors, save_directory=None):
    """
    Concat partial weight into one safetensor.
    Partitioned weight should be named in the following format:
    {original_name}_part{i}
    e.g. model.embed_tokens.weight_part0

    If save_directory is None, use direct mode to update weight from tensor in host memory.
    Otherwise save tensors to disk and update weights from disk.
    """
    sorted_keys = sorted(original_tensors.keys())
    tensors = {}
    res_tensors = {}
    prev_key = None
    concate_list = []
    file_idx = 0
    max_size = 1024 * 1024 * 1024  # max size 1GB
    param_size = 0
    for key in sorted_keys:
        val = original_tensors[key]
        if "part" not in key:
            tensors[key] = val
            param_size += val.numel() * val.element_size()
            if param_size > max_size:
                if save_directory is None:
                    res_tensors.update(tensors)
                else:
                    save_file_name = save_directory + "/model_" + str(file_idx) + ".safetensors"
                    save_file(tensors, save_file_name)
                    file_idx += 1
                param_size = 0
                tensors = {}
            continue

        name_split = key.split(".")
        cur_name_list = name_split[:-1]
        weight_name = name_split[-1]
        cur_idx = int(weight_name.removeprefix("weight_part"))
        if prev_key is None:
            inplace_insert_value_with_idx(concate_list, val, cur_idx)
            prev_key = key
        else:
            prev_name_list = prev_key.split(".")[:-1]
            if prev_name_list == cur_name_list:
                inplace_insert_value_with_idx(concate_list, val, cur_idx)
            else:
                concate_result = torch.cat(concate_list, 0)
                cur_name_list.append("weight")
                final_key = ".".join(cur_name_list)
                tensors[final_key] = concate_result
                param_size += val.numel() * val.element_size()
                if param_size > max_size:
                    if save_directory is None:
                        res_tensors.update(tensors)
                    else:
                        save_file_name = save_directory + "/model_" + str(file_idx) + ".safetensors"
                        save_file(tensors, save_file_name)
                        file_idx += 1
                    param_size = 0
                    tensors = {}

                # for next tensor
                concate_list = []
                inplace_insert_value_with_idx(concate_list, val, cur_idx)
            prev_key = key

    if concate_list:
        concate_result = torch.cat(concate_list, 0)
        cur_name_list = prev_key.split(".")[:-1]
        cur_name_list.append("weight")
        final_key = ".".join(cur_name_list)
        tensors[final_key] = concate_result

    if save_directory is None:
        res_tensors.update(tensors)
        return res_tensors
    else:
        save_file_name = save_directory + "/model_" + str(file_idx) + ".safetensors"
        save_file(tensors, save_file_name)
        return {}


def is_block_needed(key, is_first_shard, is_last_shard, start_layer, end_layer) -> bool:
    if is_first_shard and "embed_tokens" in key and key.startswith("model."):
        return True

    if is_last_shard:
        if "model.norm" in key or "lm_head" in key:
            return True
        if "embed" in key and key.startswith("model.embed_tokens"):
            return True

    if "layers." in key:
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                layer_idx = int(parts[i + 1])
                return start_layer <= layer_idx < end_layer

    return False


def filer_weight_cid_list(start_layer, end_layer, hidden_layers, index_map):
    """
    Filters block cids that worker node needs to download.
    Arguments:
        start_layer: local start layer
        end_layer: local end layer
        index_map:  Dict[str],  key(weight_name): value(cid)
    Returns:
        cid: List[int].  cid list that current worker node holds.
    """
    is_first_shard = start_layer == 0
    is_last_shard = end_layer == hidden_layers

    res = set()
    for key in index_map.keys():
        if is_block_needed(key, is_first_shard, is_last_shard, start_layer, end_layer):
            value = index_map.get(key)
            res.add(value)

    return list(res)


def remove_list_dirs(dir_list):
    for dir_path in dir_list:
        if os.path.isdir(dir_path):
            try:
                shutil.rmtree(dir_path)
            except OSError:
                pass
        else:
            continue


def release_disk_storage():
    """Remove lattica storage files before get blocks"""
    storage_dir = "/tmp"
    storage_dirs = glob.glob(storage_dir + "/*.storage")
    key_dirs = glob.glob(storage_dir + "/*.key")
    dht_dirs = glob.glob(storage_dir + "/*.dht")
    remove_list_dirs(storage_dirs)
    remove_list_dirs(key_dirs)
    remove_list_dirs(dht_dirs)


def parse_safetensors_from_memory(raw_data: bytes) -> Dict[str, torch.Tensor]:
    """
    Convert binary in memory to safetensors
    """
    header_size = struct.unpack("<Q", raw_data[:8])[0]

    header_data = raw_data[8 : 8 + header_size]
    header = json.loads(header_data.decode("utf-8"))

    header.pop("__metadata__", None)

    tensors = {}
    buffer_start = 8 + header_size
    buffer = raw_data[buffer_start:]

    for name, info in header.items():
        if name == "__metadata__":
            continue

        dtype = info["dtype"]
        shape = info["shape"]
        begin, end = info["data_offsets"]

        dtype_map = {
            "F16": (torch.float16, 2),
            "BF16": (torch.bfloat16, 2),
            "F32": (torch.float32, 4),
            "F64": (torch.float64, 8),
            "I8": (torch.int8, 1),
            "I16": (torch.int16, 2),
            "I32": (torch.int32, 4),
            "I64": (torch.int64, 8),
            "U8": (torch.uint8, 1),
            "BOOL": (torch.bool, 1),
        }
        torch_dtype, item_size = dtype_map[dtype]

        num_elements = 1
        for dim in shape:
            num_elements *= dim
        total_bytes = num_elements * item_size

        tensor_data = buffer[begin : begin + total_bytes]
        tensor = torch.frombuffer(tensor_data, dtype=torch_dtype).clone()
        tensor = tensor.reshape(shape)

        tensors[name] = tensor

    return tensors
