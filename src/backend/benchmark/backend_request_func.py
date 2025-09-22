"""
Adapted from vLLM: https://github.com/vllm-project/vllm/blob/v0.7.2/benchmarks/backend_request_func.py
"""

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Union

import aiohttp
import huggingface_hub.constants
from modelscope import snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None
    best_of: int = 1
    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": (
                request_func_input.model_name
                if request_func_input.model_name
                else request_func_input.model
            ),
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("text")
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!"
                        )
                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """Send a streaming request to an OpenAI-compatible Chat Completions API.

    This implementation measures client-side latencies consistently:
    - TTFT (time-to-first-token) is recorded at the arrival time of the first
      non-empty content delta.
    - ITL (inter-token latencies) are measured between subsequent non-empty
      content deltas.
    - Overall latency is measured up to the last non-empty content delta, not
      the trailing usage summary event. This aligns TPOT and ITL.

    Args:
        request_func_input: Request parameters and payload settings.
        pbar: Optional progress bar to update upon completion.

    Returns:
        RequestFuncOutput populated with generated text, token counts, and
        timing metrics captured from the streaming response.
    """
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        content = [{"type": "text", "text": request_func_input.prompt}]
        if request_func_input.multi_modal_content:
            content.append(request_func_input.multi_modal_content)
        payload = {
            "model": (
                request_func_input.model_name
                if request_func_input.model_name
                else request_func_input.model
            ),
            "messages": [
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        # Timestamp of last non-empty content token we observed
        last_token_timestamp = st
        # Timestamp of last received event (used as a fallback if no tokens arrive)
        first_content_received = False
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                content_str = content.strip() if isinstance(content, str) else ""

                                # Only act on non-empty content tokens
                                if content_str:
                                    if not first_content_received:
                                        first_content_received = True
                                        output.ttft = timestamp - st
                                        last_token_timestamp = timestamp
                                    else:
                                        output.itl.append(timestamp - last_token_timestamp)
                                        last_token_timestamp = timestamp

                                    generated_text += content
                            elif usage := data.get("usage"):
                                # Capture token count from trailing usage event
                                output.output_tokens = usage.get("completion_tokens")

                            # Always record last event timestamp for fallback latency

                    output.generated_text = generated_text
                    if first_content_received:
                        output.success = True
                        # Latency is measured to the last non-empty content token
                        output.latency = last_token_timestamp - st
                    else:
                        output.success = False
                        output.error = (
                            "Never received a non-empty content delta to calculate TTFT. "
                            "This response will be marked as failed!"
                        )
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


def get_model(pretrained_model_name_or_path: str) -> str:

    model_path = snapshot_download(
        model_id=pretrained_model_name_or_path,
        local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
    )
    return model_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if pretrained_model_name_or_path is not None and not os.path.exists(
        pretrained_model_name_or_path
    ):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    else:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "sglang": async_request_openai_completions,
    "parallax": async_request_openai_chat_completions,
    "llama.cpp": async_request_openai_completions,
}
