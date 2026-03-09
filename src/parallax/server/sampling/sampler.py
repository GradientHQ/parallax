"""
Postprocesses logit_outputs to get tokens with different sampling methods
specified by requests.

Components:
    SamplingBatchInfo: Sampling info for a batch of requests
    Sampler: Module class for sampling.
    SamplerOutput: Return type carrying token_ids and optional logprobs per request.
"""

import dataclasses
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
from mlx import nn
import numpy as np

from parallax.server.request import Request
from parallax.server.sampling.sampling_params import SamplingParams
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

# Sentinel for "mask out this token" in logit_bias (OpenAI convention)
LOGIT_BIAS_MASK_OUT = -100.0


@dataclasses.dataclass
class SamplerLogprobsResult:
    """Per-request logprobs when sampling_params.logprobs is True."""

    chosen_logprob: float
    top_logprobs_dict: Dict[int, float]  # token_id -> log probability


def _log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    """Numerically stable log_softmax."""
    x_max = mx.max(x, axis=axis, keepdims=True)
    return x - x_max - mx.log(mx.sum(mx.exp(x - x_max), axis=axis, keepdims=True))


def _extract_logprobs_for_batch(
    log_probs: mx.array,
    batch_next_token_ids: mx.array,
    top_logprobs_per_req: List[int],
) -> List[Optional[SamplerLogprobsResult]]:
    """
    Extract chosen_logprob and top_logprobs_dict for each request.
    log_probs: (batch, vocab_size), log probabilities.
    batch_next_token_ids: (batch,) or (batch, 1), chosen token ids.
    """
    batch_size = log_probs.shape[0]
    if batch_next_token_ids.ndim > 1:
        batch_next_token_ids = batch_next_token_ids.squeeze(-1)
    results: List[Optional[SamplerLogprobsResult]] = []
    for i in range(batch_size):
        k = top_logprobs_per_req[i] if i < len(top_logprobs_per_req) else 0
        chosen_id = int(batch_next_token_ids[i])
        chosen_logprob = float(log_probs[i, chosen_id])
        top_logprobs_dict: Dict[int, float] = {}
        if k > 0:
            row = log_probs[i]
            top_k_indices = mx.argsort(-row)[:k]
            for pos in range(top_k_indices.size):
                tid = int(top_k_indices[pos])
                top_logprobs_dict[tid] = float(row[tid])
        results.append(SamplerLogprobsResult(chosen_logprob=chosen_logprob, top_logprobs_dict=top_logprobs_dict))
    return results


def _apply_logit_bias(logits: mx.array, logit_biases: List[Optional[Dict[int, float]]]) -> mx.array:
    """Add per-request logit bias; bias=-100 sets logit to -inf. logits shape: (batch, vocab_size)."""
    if not logit_biases or all(d is None or len(d) == 0 for d in logit_biases):
        return logits
    batch_size, vocab_size = logits.shape
    bias_add = np.zeros((batch_size, vocab_size), dtype=np.float32)
    mask_out = np.zeros((batch_size, vocab_size), dtype=bool)
    for i in range(min(batch_size, len(logit_biases))):
        d = logit_biases[i]
        if not d:
            continue
        for token_id, bias in d.items():
            if token_id < 0 or token_id >= vocab_size:
                logger.warning(
                    "logit_bias: ignoring token_id=%s (out of vocab_size [0, %s))",
                    token_id,
                    vocab_size,
                )
                continue
            if bias == LOGIT_BIAS_MASK_OUT:
                mask_out[i, token_id] = True
            else:
                bias_add[i, token_id] = bias
    logits = logits + mx.array(bias_add, dtype=logits.dtype)
    if np.any(mask_out):
        neg_inf = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        logits = mx.where(mx.array(mask_out), neg_inf, logits)
    return logits


def _apply_presence_frequency_penalty(
    logits: mx.array,
    output_ids_per_req: List[List[int]],
    presence_penalties: np.ndarray,
    frequency_penalties: np.ndarray,
    vocab_size: int,
) -> mx.array:
    """
    Apply presence_penalty and frequency_penalty from generation history.
    penalty[j] = presence_penalty * I(j in history) + frequency_penalty * count(j).
    Vectorized per row via np.bincount; only loops over batch dimension.
    """
    batch_size = logits.shape[0]
    if batch_size == 0:
        return logits
    presence_penalties = np.asarray(presence_penalties, dtype=np.float32).ravel()
    frequency_penalties = np.asarray(frequency_penalties, dtype=np.float32).ravel()
    penalty = np.zeros((batch_size, vocab_size), dtype=np.float32)
    for i in range(batch_size):
        tokens = output_ids_per_req[i] if i < len(output_ids_per_req) else []
        if not tokens:
            continue
        p_p, f_p = (
            presence_penalties[i] if i < len(presence_penalties) else 0.0,
            frequency_penalties[i] if i < len(frequency_penalties) else 0.0,
        )
        if p_p == 0.0 and f_p == 0.0:
            continue
        counts = np.bincount(tokens, minlength=vocab_size)
        penalty[i, :] = p_p * (counts > 0).astype(np.float32) + f_p * counts
    return logits - mx.array(penalty, dtype=logits.dtype)


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

    # Per-request logit bias (OpenAI logit_bias): token_id -> additive bias in [-100, 100]
    logit_biases: List[Optional[Dict[int, float]]]

    # When True, sampler returns logprobs (chosen_logprob + top_logprobs_dict) per request
    need_logprobs: bool
    # Per-request K for top logprobs (0 = do not return top list for that request)
    top_logprobs_per_req: List[int]

    # For presence_penalty / frequency_penalty: generated token ids so far per request
    output_ids_per_req: List[List[int]]
    # Per-request presence_penalty and frequency_penalty (length batch)
    presence_penalties: np.ndarray
    frequency_penalties: np.ndarray

    @classmethod
    def from_reqs(cls, reqs: list[Request]):
        """Retrieves sampling infos from a list of requests."""
        for r in reqs:
            if r.sampling_params is None:
                r.sampling_params = SamplingParams()

        is_all_greedy = all(r.sampling_params.top_k <= 1 for r in reqs)
        need_min_p_sampling = any(r.sampling_params.min_p > 0 for r in reqs)
        need_logprobs = any(
            getattr(r.sampling_params, "logprobs", False) for r in reqs
        )
        top_logprobs_per_req = [
            (r.sampling_params.top_logprobs or 0)
            if getattr(r.sampling_params, "logprobs", False)
            else 0
            for r in reqs
        ]
        output_ids_per_req = [
            list(getattr(r, "output_ids", []) or []) for r in reqs
        ]
        presence_penalties = np.array(
            [r.sampling_params.presence_penalty for r in reqs], dtype=np.float32
        )
        frequency_penalties = np.array(
            [r.sampling_params.frequency_penalty for r in reqs], dtype=np.float32
        )

        temperatures = mx.array(
            [r.sampling_params.temperature for r in reqs], dtype=mx.float32
        ).reshape(-1, 1)
        top_ps = mx.array([r.sampling_params.top_p for r in reqs], dtype=mx.float32)
        top_ks = mx.array([r.sampling_params.top_k for r in reqs], dtype=mx.int32)
        min_ps = mx.array([r.sampling_params.min_p for r in reqs], dtype=mx.float32)
        logit_biases = [
            r.sampling_params.logit_bias if r.sampling_params.logit_bias else None
            for r in reqs
        ]

        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            is_all_greedy=is_all_greedy,
            need_min_p_sampling=need_min_p_sampling,
            logit_biases=logit_biases,
            need_logprobs=need_logprobs,
            top_logprobs_per_req=top_logprobs_per_req,
            output_ids_per_req=output_ids_per_req,
            presence_penalties=presence_penalties,
            frequency_penalties=frequency_penalties,
        )
        return ret


class Sampler(nn.Module):
    """Sampler that completes Topk/Topp sampling for logits"""

    def __call__(
        self,
        logits: mx.array,
        sampling_info: SamplingBatchInfo,
    ) -> Union[
        mx.array,
        Tuple[mx.array, Optional[List[SamplerLogprobsResult]]],
    ]:
        """Run sampler; optionally compute and return logprobs.

        Args:
            logits: Logits from the model forward, shape (batch, vocab_size).
            sampling_info: Metadata for sampling.

        Returns:
            If sampling_info.need_logprobs is False: batch_next_token_ids only (mx.array).
            If need_logprobs is True: (batch_next_token_ids, logprobs_info) where
            logprobs_info is a list of SamplerLogprobsResult per request (chosen_logprob + top_logprobs_dict).
        """
        # Apply logit processors before softmax (order: logit_bias -> presence/frequency penalty)
        logits = _apply_logit_bias(logits, sampling_info.logit_biases)
        output_ids_per_req = getattr(sampling_info, "output_ids_per_req", [])
        presence_penalties = getattr(sampling_info, "presence_penalties", np.array([]))
        frequency_penalties = getattr(sampling_info, "frequency_penalties", np.array([]))
        if output_ids_per_req and (np.any(presence_penalties != 0) or np.any(frequency_penalties != 0)):
            logits = _apply_presence_frequency_penalty(
                logits,
                output_ids_per_req,
                presence_penalties,
                frequency_penalties,
                vocab_size=logits.shape[1],
            )
        need_logprobs = getattr(sampling_info, "need_logprobs", False)
        top_logprobs_per_req = getattr(sampling_info, "top_logprobs_per_req", [])

        if sampling_info.is_all_greedy:
            logits_scaled = logits / sampling_info.temperatures.reshape(-1, 1)
            batch_next_token_ids = mx.argmax(logits_scaled, axis=-1)
        else:
            logits_scaled = logits / sampling_info.temperatures.reshape(-1, 1)
            probs = mx.softmax(logits_scaled, axis=-1)
            batch_next_token_ids = apply_top_k_top_p_min_p_sampling(
                probs,
                sampling_info.top_ks,
                sampling_info.top_ps,
                sampling_info.min_ps,
                sampling_info.need_min_p_sampling,
            )

        if not need_logprobs:
            return batch_next_token_ids

        try:
            log_probs = _log_softmax(logits_scaled, axis=-1)
            logprobs_info = _extract_logprobs_for_batch(
                log_probs,
                batch_next_token_ids,
                top_logprobs_per_req,
            )
            return (batch_next_token_ids, logprobs_info)
        except Exception as e:
            logger.warning(
                "logprobs computation failed (e.g. OOM), returning tokens without logprobs: %s",
                e,
                exc_info=True,
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

    return batch_next_token_ids
