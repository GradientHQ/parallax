## This is a patch file for sglang Step3.5 model to support PP-compatible forward return types

import logging
import os
from collections import Counter
from typing import Optional

import torch
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

logger = logging.getLogger(__name__)


def apply_step3p5_monkey_patch():
    """Patch Step3p5ForCausalLM.forward to handle PPProxyTensors/tuple/tensor outputs."""
    try:
        import sglang.srt.models.step3p5 as step3p5_module
    except Exception as e:
        logger.warning(f"Failed to import step3p5 for monkey patch: {e}")
        return

    debug_enabled = os.environ.get("PARALLAX_STEP3P5_DEBUG", "0") == "1"
    original_load_weights = step3p5_module.Step3p5ForCausalLM.load_weights

    def _iter_weights_with_stats(weights, stats, start_layer: int, end_layer: int):
        for name, loaded_weight in weights:
            stats["total"] += 1
            if name.startswith("model.layers."):
                stats["layer_total"] += 1
                layer_id = None
                parts = name.split(".")
                if len(parts) >= 3:
                    try:
                        layer_id = int(parts[2])
                    except ValueError:
                        layer_id = None
                if layer_id is not None:
                    if layer_id < start_layer:
                        stats["layer_below_start"] += 1
                    elif layer_id >= end_layer:
                        stats["layer_above_end"] += 1
                    else:
                        stats["layer_in_range"] += 1
            elif name.startswith("model.embed_tokens."):
                stats["embed"] += 1
            elif name.startswith("model.norm."):
                stats["norm"] += 1
            elif name.startswith("lm_head."):
                stats["lm_head"] += 1
            else:
                stats["other"] += 1
            yield name, loaded_weight

    def _debug_load_weights(self, weights, is_nextn=False):
        if not debug_enabled:
            return original_load_weights(self, weights, is_nextn=is_nextn)

        pp_group = getattr(self, "pp_group", None)
        start_layer = int(getattr(getattr(self, "model", None), "start_layer", -1))
        end_layer = int(getattr(getattr(self, "model", None), "end_layer", -1))
        stats = Counter()

        logger.warning(
            "[step3p5-debug][load_weights][start] pp_rank=%s world=%s is_first=%s is_last=%s "
            "start_layer=%s end_layer=%s is_nextn=%s",
            getattr(pp_group, "rank_in_group", None),
            getattr(pp_group, "world_size", None),
            getattr(pp_group, "is_first_rank", None),
            getattr(pp_group, "is_last_rank", None),
            start_layer,
            end_layer,
            is_nextn,
        )

        try:
            ret = original_load_weights(
                self,
                _iter_weights_with_stats(weights, stats, start_layer, end_layer),
                is_nextn=is_nextn,
            )
        except Exception:
            logger.exception(
                "[step3p5-debug][load_weights][error] stats=%s",
                dict(stats),
            )
            raise

        named_params = list(self.named_parameters())
        has_embed_param = any(n.startswith("model.embed_tokens.") for n, _ in named_params)
        has_norm_param = any(n.startswith("model.norm.") for n, _ in named_params)
        has_lm_head_param = any(n.startswith("lm_head.") for n, _ in named_params)

        logger.warning(
            "[step3p5-debug][load_weights][done] stats=%s named_params=%d has_embed=%s "
            "has_norm=%s has_lm_head=%s",
            dict(stats),
            len(named_params),
            has_embed_param,
            has_norm_param,
            has_lm_head_param,
        )
        return ret

    @torch.no_grad()
    def _pp_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **kwargs,
    ):
        model_output = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if isinstance(model_output, PPProxyTensors):
            return model_output

        if isinstance(model_output, tuple):
            hidden_states, hidden_states_before_norm = model_output
        else:
            hidden_states, hidden_states_before_norm = model_output, None

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                hidden_states_before_norm=hidden_states_before_norm,
            )
        return hidden_states

    step3p5_module.Step3p5ForCausalLM.forward = _pp_forward
    step3p5_module.Step3p5ForCausalLM.load_weights = _debug_load_weights
