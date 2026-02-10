## This is a patch file for sglang Step3.5 model to support PP-compatible forward return types

import logging
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

