"""
vLLM backend integration for Parallax distributed inference.

This module provides vLLM model runner with pipeline parallelism support.
"""

from parallax.vllm.model_runner import (
    ParallaxVLLMEngine,
    form_vllm_engine_args,
    initialize_vllm_model_runner,
)

__all__ = [
    "ParallaxVLLMEngine",
    "form_vllm_engine_args",
    "initialize_vllm_model_runner",
]
