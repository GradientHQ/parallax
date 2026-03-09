## This is a patch file for sglang GPT-OSS model to support loading mxFP4 MoE experts weights

import math

import torch
from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_rank,
    get_moe_tensor_parallel_world_size,
)
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gpt_oss import GptOssForCausalLM


def _parallax_load_mxfp4_experts_weights(self, weights):
    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    mxfp4_block = 32

    moe_tp_rank = get_moe_tensor_parallel_rank()
    moe_tp_size = get_moe_tensor_parallel_world_size()
    moe_ep_rank = get_moe_expert_parallel_rank()
    moe_ep_size = get_moe_expert_parallel_world_size()

    intermediate_size = self.config.intermediate_size
    assert (
        intermediate_size % mxfp4_block == 0
    ), f"{intermediate_size=} must be divisible by {mxfp4_block=}"
    intermediate_size_block = intermediate_size // mxfp4_block

    per_rank_intermediate_size_block = math.ceil(intermediate_size_block / moe_tp_size)

    per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block

    # Calculate common slicing bounds for current rank
    assert self.config.num_local_experts % moe_ep_size == 0
    moe_num_global_experts = self.config.num_local_experts
    moe_num_local_experts = self.config.num_local_experts // moe_ep_size

    moe_tp_rank_start = moe_tp_rank * per_rank_intermediate_size
    moe_tp_rank_end = min((moe_tp_rank + 1) * per_rank_intermediate_size, intermediate_size)

    moe_ep_rank_start = moe_ep_rank * moe_num_local_experts
    moe_ep_rank_end = (moe_ep_rank + 1) * moe_num_local_experts

    for name, weight in weights:
        ############################################################################
        ## TODO: remove when sglang code support pipeline parallelism
        ## This is a patch code for sgalng
        layer_id = get_layer_id(name)
        if (
            layer_id is not None
            and hasattr(self.model, "start_layer")
            and (layer_id < self.model.start_layer or layer_id >= self.model.end_layer)
        ):
            continue
        ## End of patch
        ############################################################################
        weight = weight.cuda()

        if "gate_up_proj_blocks" in name:
            # Handle MLP gate and up projection weights
            new_name = name.replace("gate_up_proj_blocks", "w13_weight")

            # flat weight from (E, 2 * N, block_size, entry_per_block)
            # to (E, 2 * N, -1), shouldn't trigger copy for contiguous
            weight = weight.view(moe_num_global_experts, 2 * intermediate_size, -1).contiguous()

            narrow_weight = weight[
                moe_ep_rank_start:moe_ep_rank_end,
                2 * moe_tp_rank_start : 2 * moe_tp_rank_end,
                ...,
            ]

            param = params_dict[new_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(
                param,
                narrow_weight,
                weight_name=new_name,
                shard_id=None,
                expert_id=None,
            )
            loaded_params.add(new_name)

        elif "down_proj_blocks" in name:
            # Handle MLP down projection weights
            new_name = name.replace("down_proj_blocks", "w2_weight")
            # same flatten here, but since 2 mx4 value are packed in 1
            # uint8, divide by 2
            weight = weight.view(moe_num_global_experts, -1, intermediate_size // 2).contiguous()
            narrow_weight = weight[
                moe_ep_rank_start:moe_ep_rank_end,
                ...,
                moe_tp_rank_start // 2 : moe_tp_rank_end // 2,
            ]

            param = params_dict[new_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(
                param,
                narrow_weight,
                weight_name=new_name,
                shard_id=None,
                expert_id=None,
            )
            loaded_params.add(new_name)

        elif "gate_up_proj_scales" in name:
            # Handle MLP gate and up projection weights scale
            new_name = name.replace("gate_up_proj_scales", "w13_weight_scale")
            narrow_weight = weight[
                moe_ep_rank_start:moe_ep_rank_end,
                2 * moe_tp_rank_start : 2 * moe_tp_rank_end,
                ...,
            ]

            param = params_dict[new_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(
                param,
                narrow_weight,
                weight_name=new_name,
                shard_id=None,
                expert_id=None,
            )
            loaded_params.add(new_name)

        elif "down_proj_scales" in name:
            # Handle MLP down projection weights
            new_name = name.replace("down_proj_scales", "w2_weight_scale")
            narrow_weight = weight[
                moe_ep_rank_start:moe_ep_rank_end,
                ...,
                moe_tp_rank_start // mxfp4_block : moe_tp_rank_end // mxfp4_block,
            ]

            param = params_dict[new_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(
                param,
                narrow_weight,
                weight_name=new_name,
                shard_id=None,
                expert_id=None,
            )
            loaded_params.add(new_name)
        elif "gate_up_proj_bias" in name:
            # Handle MLP gate and up projection biases
            new_name = name.replace("gate_up_proj_bias", "w13_weight_bias")

            narrow_weight = weight[
                moe_ep_rank_start:moe_ep_rank_end,
                2 * moe_tp_rank_start : 2 * moe_tp_rank_end,
            ]

            param = params_dict[new_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(
                param,
                narrow_weight,
                weight_name=new_name,
                shard_id=None,
                expert_id=None,
            )
            loaded_params.add(new_name)

        elif "down_proj_bias" in name:
            narrow_weight = weight[moe_ep_rank_start:moe_ep_rank_end, ...]
            if moe_tp_rank != 0:
                narrow_weight = torch.zeros_like(narrow_weight)

            # Handle MLP down projection bias
            new_name = name.replace("down_proj_bias", "w2_weight_bias")
            param = params_dict[new_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(
                param,
                narrow_weight,
                weight_name=new_name,
                shard_id=None,
                expert_id=None,
            )
            loaded_params.add(new_name)

    return loaded_params


def apply_gpt_oss_monkey_patch():
    GptOssForCausalLM._load_mxfp4_experts_weights = _parallax_load_mxfp4_experts_weights
