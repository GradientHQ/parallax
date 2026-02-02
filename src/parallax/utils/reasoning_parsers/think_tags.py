"""
Reasoning parser for <think> ... </think> tagged output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ThinkTagReasoningState:
    start_token: str = "<think>"
    end_token: str = "</think>"
    in_reasoning: bool = False
    buffer: str = ""
    strip_leading_newlines: bool = False
    strip_reasoning_newlines: bool = False

    def extract_from_segment(self, segment: str) -> Tuple[str, str]:
        if not segment:
            return segment, ""
        if self.strip_leading_newlines and not self.in_reasoning:
            idx = 0
            while idx < len(segment) and segment[idx] in "\r\n":
                idx += 1
            segment = segment[idx:]
            if idx < len(segment):
                self.strip_leading_newlines = False
            else:
                return "", ""
        if self.strip_reasoning_newlines and self.in_reasoning:
            idx = 0
            while idx < len(segment) and segment[idx] in "\r\n":
                idx += 1
            segment = segment[idx:]
            if idx < len(segment):
                self.strip_reasoning_newlines = False
            else:
                return "", ""
        output_chunks = []
        reasoning_chunks = []
        idx = 0
        while idx < len(segment):
            if not self.in_reasoning:
                start_idx = segment.find(self.start_token, idx)
                if start_idx == -1:
                    output_chunks.append(segment[idx:])
                    break
                if start_idx > idx:
                    output_chunks.append(segment[idx:start_idx])
                idx = start_idx + len(self.start_token)
                self.in_reasoning = True
                self.strip_reasoning_newlines = True
                while idx < len(segment) and segment[idx] in "\r\n":
                    idx += 1
                if idx < len(segment):
                    self.strip_reasoning_newlines = False
            else:
                end_idx = segment.find(self.end_token, idx)
                if end_idx == -1:
                    self.buffer += segment[idx:]
                    break
                self.buffer += segment[idx:end_idx]
                reasoning_chunks.append(self.buffer)
                self.buffer = ""
                self.in_reasoning = False
                idx = end_idx + len(self.end_token)
                self.strip_leading_newlines = True
                while idx < len(segment) and segment[idx] in "\r\n":
                    idx += 1
                if idx < len(segment):
                    self.strip_leading_newlines = False
        return "".join(output_chunks), "".join(reasoning_chunks)

    def finalize(self) -> str:
        if not self.in_reasoning or not self.buffer:
            return ""
        reasoning = self.buffer
        self.buffer = ""
        self.in_reasoning = False
        return reasoning
