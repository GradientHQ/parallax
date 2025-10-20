"""
为 vLLM 后端组装批次输入，接口尽量与 sglang 的 form_* 类似，便于在 executor 中切换。
注意：这里不做 KV 管理，由各 peer 内的模型自行处理（性能可能不及 vLLM 内部调度）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from parallax.server.request import Request


def _pad_2d(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """将一组 1D token 张量 padding 为 2D (B, L_max)，返回 padded 和 长度 tensor。"""
    if not seqs:
        return torch.empty(0, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    max_len = max(x.numel() for x in seqs)
    bsz = len(seqs)
    padded = torch.full((bsz, max_len), pad_id, dtype=seqs[0].dtype, device=seqs[0].device)
    lengths = torch.empty(bsz, dtype=torch.long, device=seqs[0].device)
    for i, x in enumerate(seqs):
        L = x.numel()
        padded[i, :L] = x
        lengths[i] = L
    return padded, lengths


def form_vllm_batch_prefill(
    batched_requests: List[Request], pad_token_id: int
) -> Dict[str, Any]:
    """首个 peer: 使用 input_ids；中间/最后 peer: 由 executor 传入 intermediate_tensors。
    这里仅组装 input_ids/lengths/requests（供首个 peer 使用）。
    """
    if len(batched_requests) == 0:
        return None
    # 收集 tokens（first peer 情况）
    token_lists: List[torch.Tensor] = []
    for req in batched_requests:
        assert hasattr(req, "input_ids") and req.input_ids is not None
        # 将 list[int] 转为 torch tensor
        token_lists.append(torch.tensor(req.input_ids, dtype=torch.long, device="cuda"))
    input_ids, lengths = _pad_2d(token_lists, pad_token_id)
    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "requests": batched_requests,
    }


def form_vllm_batch_decode(
    batched_requests: List[Request], is_first_peer: bool
) -> Dict[str, Any]:
    """解码批次：
    - 首个 peer: 仅传最后一个 token。
    - 中间/最后 peer: 由 executor 提供 intermediate_tensors。
    这里只组装首个 peer 所需的输入。
    """
    if len(batched_requests) == 0:
        return None
    if not is_first_peer:
        # 非首个 peer 不需要 tokens 输入
        return {
            "input_ids": None,
            "lengths": torch.tensor([1 for _ in batched_requests], device="cuda"),
            "requests": batched_requests,
        }
    last_tokens: List[torch.Tensor] = []
    for req in batched_requests:
        assert req.output_ids is not None and len(req.output_ids) > 0
        last_tokens.append(torch.tensor([req.output_ids[-1]], dtype=torch.long, device="cuda"))
    input_ids, lengths = _pad_2d(last_tokens, pad_id=0)
    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "requests": batched_requests,
    }


