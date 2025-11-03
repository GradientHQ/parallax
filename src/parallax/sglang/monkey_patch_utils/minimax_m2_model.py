## This is a patch file for sglang MiniMax M2 model to support pipeline parallelism

import logging
from typing import Iterable, Optional, Set, Tuple

import torch
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.minimax_m2 import get_spec_layer_idx_from_weight_name

logger = logging.getLogger(__name__)


def monkey_patch_load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    """Load model weights with proper mapping for MiniMax architecture."""

    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    # Params for weights, fp8 weight scales, fp8 activation scales
    # (param_name, weight_name, expert_id, shard_id)
    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=self.config.num_local_experts,
    )

    params_dict = dict(self.named_parameters())
    loaded_params: Set[str] = set()
    for name, loaded_weight in weights:
        if "lm_head" in name:
            pp_group = getattr(self, "pp_group", None) or get_pp_group()
            if not pp_group.is_last_rank:
                logger.debug("Skipping lm_head weight '%s' on non-last PP rank", name)
                continue

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
            continue  # skip spec decode layers for main model

        for param_name, weight_name, shard_id in stacked_params_mapping:
            # Skip non-stacked layers and experts (experts handled below).
            if weight_name not in name:
                continue
            # We have mlp.experts[0].gate_proj in the checkpoint.
            # Since we handle the experts below in expert_params_mapping,
            # we need to skip here BEFORE we update the name, otherwise
            # name will be updated to mlp.experts[0].gate_up_proj, which
            # will then be updated below in expert_params_mapping
            # for mlp.experts[0].gate_gate_up_proj, which breaks load.
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(
                    param,
                    loaded_weight,
                    name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                param = params_dict[name]
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
        ##########################################################################
        ## TODO: remove when sglang code support pipeline parallelism
        ## This is a patch code for sgalng
        pp_group = getattr(self, "pp_group", None) or get_pp_group()
        if pp_group.is_last_rank:
            return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
        else:
            return hidden_states
        ## End of patch
        ##########################################################################

    m2_module.MiniMaxM2ForCausalLM.__init__ = pp_init
    m2_module.MiniMaxM2ForCausalLM.forward = pp_forward
    m2_module.MiniMaxM2ForCausalLM.load_weights = monkey_patch_load_weights
