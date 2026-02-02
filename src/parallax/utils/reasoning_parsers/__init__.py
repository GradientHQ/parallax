"""
Reasoning parsers for model-specific output formats.
"""

from __future__ import annotations

from typing import Optional

from parallax.utils.reasoning_parsers import gpt_oss, think_tags


def get_reasoning_state(model_name: Optional[str]):
    if model_name and "gpt-oss" in model_name.lower():
        return gpt_oss.GptOssReasoningState()
    return think_tags.ThinkTagReasoningState()
