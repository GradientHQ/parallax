"""
Store information about a SGLang batch.
The following is the flow of data structures for a batch in SGLang:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.radix_cache import RadixCache as PageRadixCache
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
from parallax.utils.chunked_prefill import set_request_prefill_chunk
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SGLPrefillBatch:
    schedule_batch: ScheduleBatch
    forward_batch: ForwardBatch
    requests: List[Request]
    chunked_req: Optional[Req]


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


def transform_requests_to_sglang(
    old_requests: List[Request], page_tree_cache: Optional[PageRadixCache] = None
) -> List[Req]:
    """Transforms Parallax Request to SGLang.Req format"""
    reqs = []
    for old_req in old_requests:
        sampling_params = transform_sampling_params_to_sglang(old_req.sampling_params)
        req = Req(
            rid=old_req.request_id,
            origin_input_text="",
            origin_input_ids=old_req.origin_input_ids or old_req.input_ids,
            sampling_params=sampling_params,
            lora_id=old_req.lora_id,
        )

        # Debug: Log before cache lookup
        if page_tree_cache is not None:
            logger.debug(
                f"[PageRadixCache] Before init_next_round_input for request {old_req.request_id}: "
                f"input_ids length={len(old_req.input_ids)}, "
                f"page_tree_cache available"
            )

        req.init_next_round_input(page_tree_cache)

        # Debug: Log after cache lookup
        if page_tree_cache is not None:
            prefix_indices_len = len(req.prefix_indices) if hasattr(req, "prefix_indices") else 0
            input_len = len(req.origin_input_ids) if hasattr(req, "origin_input_ids") else 0
            logger.debug(
                f"[PageRadixCache] After init_next_round_input for request {old_req.request_id}: "
                f"prefix_indices length={prefix_indices_len}, "
                f"origin_input_ids length={input_len}, "
                f"matched_tokens={prefix_indices_len}, "
                f"cache_hit_ratio={prefix_indices_len/input_len if input_len > 0 else 0:.2%}"
            )

        reqs.append(req)
    return reqs


def _get_or_create_sgl_req(
    old_req: Request,
    sgl_req_by_rid: Dict[str, Req],
) -> Req:
    req = sgl_req_by_rid.get(old_req.request_id)
    if req is not None:
        req.lora_id = old_req.lora_id
        return req

    sampling_params = transform_sampling_params_to_sglang(old_req.sampling_params)
    req = Req(
        rid=old_req.request_id,
        origin_input_text="",
        origin_input_ids=old_req.origin_input_ids or old_req.input_ids,
        sampling_params=sampling_params,
        lora_id=old_req.lora_id,
    )
    sgl_req_by_rid[old_req.request_id] = req
    return req


def _set_dp_token_counts(schedule_batch: ScheduleBatch, model_runner: ModelRunner) -> None:
    num_tokens = schedule_batch.extend_num_tokens
    dp_size = model_runner.dp_size
    if dp_size > 1:
        schedule_batch.global_num_tokens = [num_tokens] * dp_size
        schedule_batch.global_num_tokens_for_logprob = [num_tokens] * dp_size


def _default_running_batch() -> ScheduleBatch:
    return ScheduleBatch(reqs=[], batch_is_full=False)


def form_sgl_chunked_batch_prefill(
    requests: List[Request],
    model_runner: ModelRunner,
    tree_cache,
    sgl_req_by_rid: Dict[str, Req],
    running_batch: Optional[ScheduleBatch],
    chunked_req: Optional[Req],
    chunked_prefill_size: Optional[int],
    max_num_tokens_per_batch: int,
) -> Optional[SGLPrefillBatch]:
    """Build a SGLang prefill batch using SGLang's native chunking policy."""

    if not requests:
        return None

    running_batch = running_batch or _default_running_batch()
    parallax_req_by_rid = {req.request_id: req for req in requests}

    adder = PrefillAdder(
        page_size=model_runner.server_args.page_size,
        tree_cache=tree_cache,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        running_batch=running_batch,
        new_token_ratio=1.0,
        rem_input_tokens=max_num_tokens_per_batch,
        rem_chunk_tokens=chunked_prefill_size,
        max_running_requests=getattr(model_runner, "max_running_requests", None),
    )

    active_chunked_rid = (
        chunked_req.rid
        if chunked_req is not None and chunked_req.rid in parallax_req_by_rid
        else None
    )
    if active_chunked_rid is not None:
        chunked_req.init_next_round_input()
        chunked_req = adder.add_chunked_req(chunked_req)

    for old_req in requests:
        if old_req.request_id == active_chunked_rid:
            continue
        if old_req.request_id not in parallax_req_by_rid:
            continue

        sgl_req = _get_or_create_sgl_req(old_req, sgl_req_by_rid)
        if sgl_req is chunked_req:
            continue

        sgl_req.init_next_round_input(tree_cache)
        res = adder.add_one_req(
            sgl_req,
            has_chunked_req=(chunked_req is not None),
            truncation_align_size=None,
        )
        if res != AddReqResult.CONTINUE:
            break

    can_run_list = adder.can_run_list
    if not can_run_list:
        return None

    if adder.new_chunked_req is not None:
        chunked_req = adder.new_chunked_req

    selected_requests: List[Request] = []
    for sgl_req in can_run_list:
        req = parallax_req_by_rid.get(sgl_req.rid)
        if req is None:
            continue
        is_middle_chunk = chunked_req is not None and sgl_req is chunked_req
        set_request_prefill_chunk(
            req,
            chunk_end=len(sgl_req.fill_ids),
            is_chunked=is_middle_chunk,
        )
        selected_requests.append(req)

    if not selected_requests:
        return None

    schedule_batch = ScheduleBatch.init_new(
        reqs=can_run_list,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        chunked_req=chunked_req,
    )
    schedule_batch.prepare_for_extend()
    _set_dp_token_counts(schedule_batch, model_runner)

    model_worker_batch = schedule_batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    return SGLPrefillBatch(
        schedule_batch=schedule_batch,
        forward_batch=forward_batch,
        requests=selected_requests,
        chunked_req=chunked_req,
    )


def form_sgl_batch_prefill(
    requests: List[Request],
    model_runner: ModelRunner,
    page_tree_cache: Optional[PageRadixCache] = None,
) -> ForwardBatch:
    """Initialize a prefill ScheduleBatch -> ModelWorkerBatch -> ForwardBatch workflow"""

    tree_cache = page_tree_cache
    if tree_cache is None:
        cache_params = CacheInitParams(
            disable=True,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            page_size=model_runner.server_args.page_size,
        )
        tree_cache = ChunkCache(cache_params)

    sgl_reqs = transform_requests_to_sglang(requests, tree_cache)

    schedule_batch = ScheduleBatch.init_new(
        reqs=sgl_reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    schedule_batch.prepare_for_extend()
    _set_dp_token_counts(schedule_batch, model_runner)

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

    num_tokens = len(ready_indices)
    dp_size = model_runner.dp_size
    if dp_size > 1:
        ret.global_num_tokens = [num_tokens] * dp_size
        ret.global_num_tokens_for_logprob = [num_tokens] * dp_size

    model_worker_batch = ret.get_model_worker_batch()
    if requests[0].lora_id is not None:
        model_worker_batch.lora_ids = [req.lora_id or "" for req in requests]
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)

    return forward_batch


def release_sglang_request(running_batch: ScheduleBatch, request_id: str):
    """Release KV Cache and other resources for finished/aborted requests."""
    if running_batch is None or running_batch.is_empty():
        return
    seq_lens_cpu = running_batch.seq_lens.cpu().numpy()
    idx = find_index(running_batch, request_id)
    req = running_batch.reqs.pop(idx)

    # use running batch's tree cache to release kv cache
    tree_cache = running_batch.tree_cache

    if isinstance(tree_cache, PageRadixCache):
        tree_cache.cache_finished_req(req)
    else:
        page_size = running_batch.token_to_kv_pool_allocator.page_size
        last_uncached_pos = (len(req.prefix_indices) // page_size) * page_size
        end_pos = last_uncached_pos + seq_lens_cpu[idx]
        running_batch.seq_lens = torch.cat(
            (running_batch.seq_lens[:idx], running_batch.seq_lens[idx + 1 :])
        )
        running_batch.seq_lens_cpu = torch.cat(
            (running_batch.seq_lens_cpu[:idx], running_batch.seq_lens_cpu[idx + 1 :])
        )
        running_batch.orig_seq_lens = torch.cat(
            (running_batch.orig_seq_lens[:idx], running_batch.orig_seq_lens[idx + 1 :])
        )

        # Free kv cache
        token_indices = running_batch.req_to_token_pool.req_to_token[req.req_pool_idx][
            last_uncached_pos:end_pos
        ]
        running_batch.token_to_kv_pool_allocator.free(token_indices)
        running_batch.req_to_token_pool.free(req.req_pool_idx)
        running_batch.req_pool_indices = torch.cat(
            (running_batch.req_pool_indices[:idx], running_batch.req_pool_indices[idx + 1 :])
        )
