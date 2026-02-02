"""
Reasoning parser for gpt-oss Harmony-style output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import re

analysis_start = "<|channel|>analysis<|message|>"
analysis_end = "<|end|>"
final_start = "<|channel|>final<|message|>"
assistant_start = "<|start|>assistant"
_ANALYSIS_BLOCK_RE = re.compile(
    re.escape(analysis_start) + r"(.*?)" + re.escape(analysis_end),
    re.DOTALL,
)


def _strip_final_tag(text: str) -> str:
    if not text:
        return text
    return text.replace(final_start, "")


def _strip_assistant_start(text: str) -> str:
    if not text:
        return text
    if not text.startswith(assistant_start):
        return text
    stripped = text[len(assistant_start) :]
    if stripped.startswith(final_start):
        stripped = stripped[len(final_start) :]
    if stripped.startswith("\r\n") or stripped.startswith("\n"):
        stripped = stripped.lstrip("\r\n")
    return stripped


def extract_reasoning(text: str) -> Tuple[str, str]:
    if not text:
        return text, ""
    matches = _ANALYSIS_BLOCK_RE.findall(text)
    reasoning = "".join(matches)
    cleaned = _ANALYSIS_BLOCK_RE.sub("", text)
    cleaned = _strip_final_tag(cleaned)
    cleaned = _strip_assistant_start(cleaned)
    return cleaned, reasoning


@dataclass
class GptOssReasoningState:
    in_reasoning: bool = False
    buffer: str = ""
    current_channel: str | None = None
    in_message: bool = False
    awaiting_start_name: bool = False
    awaiting_channel_name: bool = False
    awaiting_constrain_value: bool = False
    suppress_until_end: bool = False

    def extract_from_segment(self, segment: str) -> Tuple[str, str]:
        if not segment:
            return segment, ""
        if segment.startswith("<|") and segment.endswith("|>"):
            if segment == "<|start|>":
                self.awaiting_start_name = True
                return "", ""
            if segment == "<|channel|>":
                self.awaiting_channel_name = True
                return "", ""
            if segment == "<|message|>":
                self.in_message = True
                return "", ""
            if segment == "<|end|>":
                self.in_message = False
                if self.suppress_until_end:
                    self.suppress_until_end = False
                return "", ""
            if segment == "<|constrain|>":
                self.awaiting_constrain_value = True
                return "", ""
        if self.awaiting_constrain_value:
            self.awaiting_constrain_value = False
            return "", ""
        if self.awaiting_start_name:
            self.awaiting_start_name = False
            if segment == "assistant":
                return "", ""
        if self.awaiting_channel_name:
            self.awaiting_channel_name = False
            self.current_channel = segment
            if self.current_channel.startswith("commentary to=functions."):
                self.suppress_until_end = True
            self.in_reasoning = self.current_channel == "analysis"
            return "", ""
        if not self.in_message:
            return "", ""
        if self.suppress_until_end:
            return "", ""
        if self.current_channel == "analysis":
            return "", segment
        if self.current_channel == "final":
            return segment, ""
        return "", ""

    def finalize(self) -> str:
        if not self.in_reasoning or not self.buffer:
            return ""
        reasoning = self.buffer
        self.buffer = ""
        self.in_reasoning = False
        return reasoning
