"""
Tool call parser for gpt-oss Harmony-style output.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

tool_call_start = "<|start|>assistant<|channel|>commentary to=functions."
tool_call_end = "<|end|>"
_TOOL_CALL_BLOCK_RE = re.compile(
    re.escape(tool_call_start) + r".*?(?:" + re.escape(tool_call_end) + r"|$)",
    re.DOTALL,
)

_HEADER_RE = re.compile(
    r"^(?P<name>[^\s<]+)\s*(?:<\|constrain\|>(?P<constraint>[^<]*))?\s*<\|message\|>",
    re.DOTALL,
)


def _parse_args(payload: str) -> Any:
    payload = payload.strip()
    if not payload:
        return {}
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(payload)
    except json.JSONDecodeError:
        return payload
    return obj


def _parse_segment(segment: str) -> List[Dict[str, Any]]:
    segment = segment.lstrip()
    match = _HEADER_RE.match(segment)
    if not match:
        return []
    name = match.group("name")
    payload = segment[match.end() :]
    end_idx = payload.find(tool_call_end)
    if end_idx != -1:
        payload = payload[:end_idx]
    args = _parse_args(payload)
    return [{"name": name, "arguments": args}]


def parse_tool_call(text: str, tools: List[Any] | None = None):
    del tools
    tool_calls: List[Dict[str, Any]] = []
    if tool_call_start in text:
        idx = 0
        while True:
            start_idx = text.find(tool_call_start, idx)
            if start_idx == -1:
                break
            start_idx += len(tool_call_start)
            end_idx = text.find(tool_call_end, start_idx)
            segment = text[start_idx:end_idx] if end_idx != -1 else text[start_idx:]
            tool_calls.extend(_parse_segment(segment))
            idx = end_idx if end_idx != -1 else len(text)
    else:
        tool_calls.extend(_parse_segment(text))

    if not tool_calls:
        raise ValueError("No function provided.")
    return tool_calls if len(tool_calls) > 1 else tool_calls[0]


def strip_tool_calls(text: str) -> str:
    return _TOOL_CALL_BLOCK_RE.sub("", text)
