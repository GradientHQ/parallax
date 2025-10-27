from sglang.srt.layers.utils import get_layer_id
import logging
from typing import Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.utils import (
    BumpAllocator,
    add_prefix,
    get_compiler_backend,
    is_non_idle_and_non_empty,
    make_layers,
)
from sglang.srt.models.minimax_m2 import MiniMaxM2ForCausalLM, get_spec_layer_idx_from_weight_name


def monkey_patch_load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    """Load model weights with proper mapping for MiniMax architecture."""

    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=self.config.num_local_experts,
    )

    params_dict = dict(self.named_parameters())
    logger = logging.getLogger(__name__)

    weight_name_map = {
        "lm_head.weight": "model.embed_tokens.weight",
    }

    def resolve_param(name: str):
        """Resolve weight name to actual parameter, handling tied weights and PP filtering."""
        if name in weight_name_map:
            mapped_name = weight_name_map[name]
            if mapped_name in params_dict:
                logger.debug("Mapped '%s' -> '%s' (tied weight)", name, mapped_name)
                return mapped_name, params_dict[mapped_name]

        if name in params_dict:
            return name, params_dict[name]

        alt = f"model.{name}"
        if alt in params_dict:
            return alt, params_dict[alt]

        matches = [k for k in params_dict.keys() if k.endswith(name)]
        if len(matches) == 1:
            return matches[0], params_dict[matches[0]]

        if name in ("model.norm.weight", "model.embed_tokens.weight"):
            logger.debug("Weight '%s' not found (PP-sliced)", name)
            return None, None

        if ("lm_head" in name) or ("embed" in name):
            sample = [k for k in params_dict.keys() if ("lm_head" in k) or ("embed" in k)]
            if not sample:
                sample = list(params_dict.keys())[:50]
            logger.warning("Failed to resolve '%s'. Sample params: %s", name, sample)
        return None, None

    loaded_params: Set[str] = set()
    for name, loaded_weight in weights:
        layer_id = get_layer_id(name)
        if (
            layer_id is not None
            and hasattr(self.model, "start_layer")
            and (layer_id < self.model.start_layer or layer_id >= self.model.end_layer)
        ):
            continue

        if "rotary_emb.inv_freq" in name:
            continue

        spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
        if spec_layer is not None:
            continue

        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            name = name.replace(weight_name, param_name)
            if name.endswith(".bias") and name not in params_dict:
                continue

            resolved_name, param = resolve_param(name)
            if param is None:
                if name not in ("model.norm.weight", "model.embed_tokens.weight"):
                    logger.warning("Skipping weight '%s' (no matching parameter)", name)
                continue
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                resolved_name, param = resolve_param(name)
                if param is None:
                    if name not in ("model.norm.weight", "model.embed_tokens.weight"):
                        logger.warning("Skipping expert weight '%s' (no matching parameter)", name)
                    continue
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue

                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                resolved_name, param = resolve_param(name)
                if param is None:
                    if name not in ("model.norm.weight", "model.embed_tokens.weight"):
                        logger.warning("Skipping weight '%s' (no matching parameter)", name)
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params


def apply_minimax_m2_monkey_patch():
    """Apply monkey patches to MiniMax M2 for PP support and weight loading."""
    import sglang.srt.models.minimax_m2 as m2_module

    orig_init = m2_module.MiniMaxM2ForCausalLM.__init__

    def pp_init(self, config, quant_config=None, prefix=""):
        orig_init(self, config, quant_config, prefix)
        self.pp_group = get_pp_group()

    def pp_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids, positions, forward_batch, inputs_embeds, pp_proxy_tensors
        )

        if isinstance(hidden_states, PPProxyTensors):
            return hidden_states

        pp_group = getattr(self, "pp_group", None) or get_pp_group()
        if pp_group.is_last_rank:
            return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
        else:
            return hidden_states

    m2_module.MiniMaxM2ForCausalLM.__init__ = pp_init
    m2_module.MiniMaxM2ForCausalLM.forward = pp_forward
    m2_module.MiniMaxM2ForCausalLM.load_weights = monkey_patch_load_weights
