"""
Postprocesses logit_outputs to get tokens with different sampling methods
specified by requests.

Components:
    SamplingBatchInfo: Sampling info for a batch of requests
    Sampler: Module class for sampling.
TODO: Add penalizer support.
"""

import dataclasses
from functools import partial

import mlx.core as mx
from mlx import nn

from parallax.server.request import Request
from parallax.server.sampling.sampling_params import SamplingParams


@dataclasses.dataclass
class SamplingBatchInfo:
    """Maintains batched sampling information"""

    # Basic batched sampling params
    temperatures: mx.array
    top_ps: mx.array
    top_ks: mx.array
    min_ps: mx.array

    # Whether all requests use greedy sampling
    is_all_greedy: bool

    # Whether any request needs min_p sampling
    need_min_p_sampling: bool

    @classmethod
    def from_reqs(cls, reqs: list[Request]):
        """Retrieves sampling infos from a list of requests"""
        for r in reqs:
            if r.sampling_params is None:
                r.sampling_params = SamplingParams()

        is_all_greedy = all(r.sampling_params.top_k <= 1 for r in reqs)
        need_min_p_sampling = any(r.sampling_params.min_p > 0 for r in reqs)

        temperatures = mx.array(
            [r.sampling_params.temperature for r in reqs], dtype=mx.float32
        ).reshape(-1, 1)
        top_ps = mx.array([r.sampling_params.top_p for r in reqs], dtype=mx.float32)
        top_ks = mx.array([r.sampling_params.top_k for r in reqs], dtype=mx.int32)
        min_ps = mx.array([r.sampling_params.min_p for r in reqs], dtype=mx.float32)

        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            is_all_greedy=is_all_greedy,
            need_min_p_sampling=need_min_p_sampling,
        )
        return ret


class Sampler(nn.Module):
    """Sampler that completes Topk/Topp sampling for logits"""

    def __call__(self, logits: mx.array, sampling_info: SamplingBatchInfo):
        """Run a sampler & compute logprobs and update logits accordingly

        Args:
            logits: Logits from the model forward
            sampling_info: Metadata for sampling
        Returns:
            next_token_ids: next token IDs.
        """
        batch_next_token_ids = None
        if sampling_info.is_all_greedy:
            # Use argmax if all requests use greedy sampling
            batch_next_token_ids = mx.argmax(logits, axis=-1)
        else:
            logits = logits / sampling_info.temperatures.reshape(-1, 1)
            logits[:] = mx.softmax(logits, axis=-1)
            batch_next_token_ids = apply_top_k_top_p_min_p_sampling(
                logits,
                sampling_info.top_ks,
                sampling_info.top_ps,
                sampling_info.min_ps,
                sampling_info.need_min_p_sampling,
            )
        return batch_next_token_ids


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_top_k_top_p_min_p_sampling(
    logits: mx.array,
    top_ks: mx.array,
    top_ps: mx.array,
    min_ps: mx.array,
    need_min_p_sampling: bool,
):
    """Mlx compiled kernel for calculating topk/topp/minp sampling"""
    probs_idx = mx.argsort(-logits, axis=-1)
    probs_sort = mx.take_along_axis(logits, probs_idx, axis=-1)
    probs_sum = mx.cumsum(probs_sort, axis=-1)
    top_k_mask = mx.arange(0, logits.shape[-1]).reshape(1, -1) < top_ks.reshape(-1, 1)
    probs_sort = probs_sort * top_k_mask
    top_p_mask = (probs_sum - probs_sort) <= top_ps.reshape(-1, 1)
    probs_sort = probs_sort * top_p_mask
    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        min_p_mask = probs_sort >= min_p_thresholds.reshape(-1, 1)
        probs_sort = probs_sort * min_p_mask

    probs_sort = mx.log(probs_sort)
    sampled_index = mx.random.categorical(probs_sort, num_samples=1)
    batch_next_token_ids = mx.take_along_axis(probs_idx, indices=sampled_index, axis=1)

    return batch_next_token_ids.reshape(-1)
