"""
Store information about a SGLang batch.
The following is the flow of data structures for a batch in SGLang:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch
"""

from types import SimpleNamespace
from typing import List

import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_batch_info import (
    SamplingBatchInfo as SGLSamplingBatchInfo,
)
from sglang.srt.sampling.sampling_params import SamplingParams as SGLSamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from parallax.server.request import Request
from parallax.server.sampling.sampling_params import (
    SamplingParams as ParallaxSamplingParams,
)
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


def transform_sampling_params_to_sglang(old_params: ParallaxSamplingParams) -> SGLSamplingParams:
    """Transforms Parallax SamplingParams to SGLang.SamplingParams format"""
    params = SGLSamplingParams(
        max_new_tokens=old_params.max_new_tokens,
        min_new_tokens=old_params.min_new_tokens,
        temperature=old_params.temperature,
        top_p=old_params.top_p,
        min_p=old_params.min_p,
        top_k=old_params.top_k,
        stop_token_ids=old_params.stop_token_ids,
        ignore_eos=old_params.ignore_eos,
        stop=old_params.stop_strs,
        repetition_penalty=old_params.repetition_penalty,
        presence_penalty=old_params.presence_penalty,
        json_schema=old_params.json_schema,
    )
    return params


def transform_requests_to_sglang(old_requests: List[Request]) -> List[Req]:
    """Transforms Parallax Request to SGLang.Req format"""
    reqs = []
    for old_req in old_requests:
        sampling_params = transform_sampling_params_to_sglang(old_req.sampling_params)
        req = Req(
            rid=old_req.request_id,
            origin_input_text="",
            origin_input_ids=old_req.input_ids,
            sampling_params=sampling_params,
        )
        req.init_next_round_input()
        reqs.append(req)
    return reqs


def form_sgl_batch_prefill(
    requests: List[Request],
    model_runner: ModelRunner,
) -> ForwardBatch:
    """Initialize a prefill ScheduleBatch -> ModelWorkerBatch -> ForwardBatch workflow"""
    sgl_reqs = transform_requests_to_sglang(requests)

    def dummy_evict(*args):
        pass

    dummy_tree_cache = SimpleNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        evictable_size=0,
    )
    dummy_tree_cache.evict = dummy_evict
    schedule_batch = ScheduleBatch.init_new(
        reqs=sgl_reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    schedule_batch.prepare_for_extend()
    model_worker_batch = schedule_batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    return schedule_batch, forward_batch


def select_batch(
    origin_batch: ScheduleBatch,
    keep_indices: List[int],
) -> ScheduleBatch:
    """
    Copy a subset of requests to form a new ScheduleBatch from the running ScheduleBatch.
    Since the requests are not necessary selected in the loop, we need to copy by indicies to select
    the real requests to run.
    """
    ret = origin_batch.copy()
    if keep_indices is None or len(keep_indices) == 0:
        return None

    keep_indices_device = torch.tensor(keep_indices, dtype=torch.int64).to(
        origin_batch.device, non_blocking=True
    )

    ret.token_to_kv_pool_allocator = origin_batch.token_to_kv_pool_allocator
    ret.req_to_token_pool = origin_batch.req_to_token_pool
    ret.tree_cache = origin_batch.tree_cache

    if origin_batch.model_config.is_encoder_decoder:
        ret.encoder_lens = origin_batch.encoder_lens[keep_indices_device]
        ret.encoder_lens_cpu = [origin_batch.encoder_lens_cpu[i] for i in keep_indices]

    ret.reqs = [origin_batch.reqs[i] for i in keep_indices]
    if origin_batch.multimodal_inputs is not None:
        ret.multimodal_inputs = [origin_batch.multimodal_inputs[i] for i in keep_indices]
    ret.seq_lens_cpu = origin_batch.seq_lens_cpu[keep_indices]
    ret.req_pool_indices = origin_batch.req_pool_indices[keep_indices_device]
    ret.seq_lens = origin_batch.seq_lens[keep_indices_device]
    ret.orig_seq_lens = origin_batch.orig_seq_lens[keep_indices_device]

    if origin_batch.out_cache_loc is not None:
        ret.out_cache_loc = origin_batch.out_cache_loc[keep_indices_device]
    ret.seq_lens_sum = ret.seq_lens.sum().item()

    if origin_batch.output_ids is not None:
        ret.output_ids = origin_batch.output_ids[keep_indices_device]

    ret.return_logprob = any(req.return_logprob for req in origin_batch.reqs)
    if ret.return_logprob:
        ret.top_logprobs_nums = [origin_batch.top_logprobs_nums[i] for i in keep_indices]
        ret.token_ids_logprobs = [origin_batch.token_ids_logprobs[i] for i in keep_indices]
    else:
        ret.top_logprobs_nums = None
        ret.token_ids_logprobs = None

    ret.has_stream = any(req.stream for req in origin_batch.reqs)
    ret.has_grammar = any(req.grammar for req in origin_batch.reqs)

    ret.sampling_info = SGLSamplingBatchInfo.from_schedule_batch(
        ret, origin_batch.model_config.vocab_size
    )

    return ret


def find_index(running_batch: ScheduleBatch, request_id: str):
    """Helper function for finding the requests in the running batch by request_id"""
    for index, req in enumerate(running_batch.reqs):
        if req.rid == request_id:
            return index
    logger.exception(
        f"Request {request_id} not found in running batch, size: {len(running_batch.reqs)}, \
        reqs: {[request.rid for request in running_batch.reqs]}"
    )
    return -1


def form_sgl_batch_decode(
    requests: List[Request],
    model_runner: ModelRunner,
    running_batch: ScheduleBatch,
    is_first_rank: bool,
) -> ForwardBatch:
    """
    Forms the decoding batch in this round.
    The returned ScheduleBatch is a copy of subset of the running batch.
    ModelWorkerBatch -> ForwardBatch are generated from the selected ScheduleBatch.
    """
    ready_indices = list(
        filter(lambda x: x != -1, [find_index(running_batch, req.request_id) for req in requests])
    )
    ret = select_batch(running_batch, ready_indices)
    if is_first_rank:
        output_ids = []
        for request in requests:
            output_ids.append(request.output_ids[-1])
        ret.output_ids = torch.tensor(output_ids, dtype=torch.int64).to(
            ret.device, non_blocking=True
        )
    else:
        # Set an empty output_ids tensor
        batch_size = len(ready_indices)
        ret.output_ids = torch.empty(batch_size, dtype=torch.int64).to(
            ret.device, non_blocking=True
        )
    ret.prepare_for_decode()
    # TODO: this is a hack to make the seq_lens correct due to select_batch is not refference running batch's seq_lens
    # need to fix this
    running_batch.seq_lens[ready_indices] += 1
    running_batch.seq_lens_cpu[ready_indices] += 1
    running_batch.orig_seq_lens[ready_indices] += 1

    model_worker_batch = ret.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)

    return forward_batch


def release_sglang_request(running_batch: ScheduleBatch, request_id: str):
    """Release KV Cache and other resources for finished/aborted requests."""
    if running_batch is None or running_batch.is_empty():
        return
    seq_lens_cpu = running_batch.seq_lens.cpu().numpy()
    idx = find_index(running_batch, request_id)
    req = running_batch.reqs.pop(idx)

    # Free kv cache
    page_size = running_batch.token_to_kv_pool_allocator.page_size
    last_uncached_pos = (len(req.prefix_indices) // page_size) * page_size
    token_indices = running_batch.req_to_token_pool.req_to_token[
        req.req_pool_idx, last_uncached_pos : seq_lens_cpu[idx]
    ]
    running_batch.token_to_kv_pool_allocator.free(token_indices)
    running_batch.req_to_token_pool.free(req.req_pool_idx)
