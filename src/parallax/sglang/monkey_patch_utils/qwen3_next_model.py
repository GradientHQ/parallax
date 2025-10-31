import logging
from typing import Iterable, Optional, Set, Tuple

from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import is_cuda
from torch import nn

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()


# ---- Minimal method-level monkey patch to reuse sglang source ----
# Due to Qwen3NextModel not support pipeline parallelism (PP) natively
def apply_qwen3_next_monkey_patch():
    """Apply minimal monkey patches to sglang's qwen3_next to support PP without copying code.

    We override only a few methods:
    - Qwen3NextModel.__init__: build layers with PP slicing, gate embed/norm by first/last rank.
    - Qwen3NextModel.forward: accept/return PPProxyTensors between stages.
    - Qwen3NextForCausalLM.__init__: remove single-rank assertion, keep original wiring.
    - Qwen3NextForCausalLM.forward: only last rank computes logits; others pass proxies.
    - Qwen3NextForCausalLM.load_weights: pre-filter weights by layer_id to load only local slice.
    """
    import torch
    from sglang.srt.distributed import get_pp_group
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
    from sglang.srt.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    from sglang.srt.server_args import get_global_server_args
    from sglang.srt.utils import add_prefix, is_cuda, make_layers

    try:
        import sglang.srt.models.qwen3_next as m
    except Exception as e:  # Fallback: keep current module as-is
        logger.warning(
            f"Failed to import sglang.srt.models.qwen3_next for monkey patch: {e}. Using local copy."
        )
        return

    # --- Patch Qwen3NextModel.__init__ ---
    def _pp_model_init(
        self, config, quant_config: Optional[QuantizationConfig] = None, prefix: str = ""
    ):
        nn.Module.__init__(self)
        self.config = config
        self.pp_group = get_pp_group()
        alt_stream = torch.cuda.Stream() if is_cuda() else None

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(idx: int, prefix: str):
            layer_class = m.ALL_DECODER_LAYER_TYPES[config.layers_block_type[idx]]
            return layer_class(
                config,
                idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
        )
        if self.pp_group.is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)
        self.infer_count = 0

    # --- Patch Qwen3NextModel.forward ---
    def _pp_model_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **kwargs,
    ):
        if self.pp_group.is_first_rank:
            hidden_states = (
                inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
            )
            residual = None
        else:
            assert (
                pp_proxy_tensors is not None
            ), "pp_proxy_tensors must be provided on non-first PP ranks"
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                layer_id=i,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors({"hidden_states": hidden_states, "residual": residual})
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states

    # --- Patch Qwen3NextForCausalLM.__init__ (remove single-rank assert) ---
    def _pp_for_causal_init(
        self,
        config: Qwen3NextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.config = config
        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.model = m.Qwen3NextModel(config, quant_config, prefix=add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        ).float()
        self.logits_processor = m.LogitsProcessor(config)

    # --- Patch Qwen3NextForCausalLM.forward ---
    @torch.no_grad()
    def _pp_for_causal_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            inputs_embeds,
            pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
        else:
            return hidden_states

    # --- Patch Qwen3NextForCausalLM.load_weights (filter by PP slice) ---
    orig_load_weights = m.Qwen3NextForCausalLM.load_weights

    def _pp_load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> Set[str]:
        """Filter incoming weights to only those relevant for this PP slice.

        Rules:
        - Layer weights: keep only if layer_id in [start, end).
        - Non-layer weights (layer_id is None):
            * keep if they correspond to names present in current params_dict (e.g., model.norm on last rank,
              embed on first rank, lm_head on all ranks), or
            * keep if they match known mapping keywords (so original loader can rename and resolve), or
            * keep if they are explicitly skipped by original loader (e.g., rotary_emb.inv_freq), harmless to pass.
        This prevents KeyError like 'model.norm.weight' on non-last ranks where norm is a PPMissingLayer.
        """
        start = getattr(self.model, "start_layer", None)
        end = getattr(self.model, "end_layer", None)
        params_dict = dict(self.named_parameters())
        mapping_keywords = (
            "q_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "self_attn",
        )

        filtered: list[tuple[str, torch.Tensor]] = []
        for name, w in weights:
            layer_id = get_layer_id(name)
            if layer_id is not None:
                if (start is None) or (start <= layer_id < end):
                    filtered.append((name, w))
            else:
                if (
                    (name in params_dict)
                    or any(k in name for k in mapping_keywords)
                    or ("rotary_emb.inv_freq" in name)
                ):
                    filtered.append((name, w))

        return orig_load_weights(self, filtered, is_mtp=is_mtp)

    # Bind patches
    m.Qwen3NextModel.__init__ = _pp_model_init  # type: ignore
    m.Qwen3NextModel.forward = _pp_model_forward  # type: ignore
    m.Qwen3NextForCausalLM.__init__ = _pp_for_causal_init  # type: ignore
    m.Qwen3NextForCausalLM.forward = _pp_for_causal_forward  # type: ignore
    m.Qwen3NextForCausalLM.load_weights = _pp_load_weights  # type: ignore
