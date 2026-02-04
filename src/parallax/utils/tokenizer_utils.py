"""
Implements parallax detokenizers for performance.
"""

import json
import uuid
from dataclasses import dataclass
from functools import partial
from json import JSONDecodeError
from typing import Any, Callable, Dict, List, Optional, Tuple

from mlx_lm.tokenizer_utils import (
    BPEStreamingDetokenizer,
    NaiveStreamingDetokenizer,
    SPMStreamingDetokenizer,
    _is_bpe_decoder,
    _is_spm_decoder,
    _is_spm_decoder_no_space,
)
from mlx_lm.tokenizer_utils import load as _mlx_load_tokenizer


class ParallaxNaiveStreamingDetokenizer(NaiveStreamingDetokenizer):
    """A custom BPE streaming detokenizer that add an argument 'tokenizer'"""

    def __init__(self, tokenizer, tokenmap):
        self._tokenizer = tokenizer
        self._tokenizer.decode([0])
        self.reset()


class ParallaxBPEStreamingDetokenizer(BPEStreamingDetokenizer):
    """A custom BPE streaming detokenizer that skips initializing tokenmap"""

    def __init__(self, tokenizer, tokenmap):
        self.clean_spaces = tokenizer.clean_up_tokenization_spaces
        self.tokenmap = tokenmap
        self.reset()
        self.make_byte_decoder()


class ParallaxSPMStreamingDetokenizer(SPMStreamingDetokenizer):
    """A custom SPM streaming detokenizer that skips initializing tokenmap"""

    def __init__(self, tokenizer, tokenmap, trim_space=True):
        self.trim_space = trim_space
        self._sep = "\u2581".encode()
        self.tokenmap = tokenmap
        self.reset()


def _get_spm_tokenmap(tokenizer):
    """Initialize spm tokenmap for reuse"""
    # Extract the tokens in a list from id to text
    tokenmap = [""] * (max(tokenizer.vocab.values()) + 1)
    for value, tokenid in tokenizer.vocab.items():
        if value.startswith("<0x"):
            # Replace bytes with their value
            tokenmap[tokenid] = bytes([int(value[3:5], 16)])
        else:
            tokenmap[tokenid] = value.encode()
    return tokenmap


def _get_bpe_tokenmap(tokenizer):
    """Initialize bpe tokenmap for reuse"""
    # Extract the tokens in a list from id to text
    tokenmap = [None] * len(tokenizer.vocab)
    for value, tokenid in tokenizer.vocab.items():
        tokenmap[tokenid] = value
    return tokenmap


def load_detokenizer(model_path, tokenizer):
    """Load a huggingface tokenizer and try to infer the type of streaming
    detokenizer to use.

    Note, to use a fast streaming tokenizer, pass a local file path rather than
    a Hugging Face repo ID.
    """
    detokenizer_class = ParallaxNaiveStreamingDetokenizer
    tokenmap = None

    tokenizer_file = model_path / "tokenizer.json"
    if tokenizer_file.exists():
        with open(tokenizer_file, "r", encoding="utf-8") as fid:
            try:
                tokenizer_content = json.load(fid)
            except JSONDecodeError as e:
                raise JSONDecodeError("Failed to parse tokenizer.json", e.doc, e.pos)

        if "decoder" in tokenizer_content:
            if _is_spm_decoder(tokenizer_content["decoder"]):
                detokenizer_class = ParallaxSPMStreamingDetokenizer
                tokenmap = _get_spm_tokenmap(tokenizer)
            elif _is_spm_decoder_no_space(tokenizer_content["decoder"]):
                detokenizer_class = partial(ParallaxSPMStreamingDetokenizer, trim_space=False)
                tokenmap = _get_spm_tokenmap(tokenizer)
            elif _is_bpe_decoder(tokenizer_content["decoder"]):
                detokenizer_class = ParallaxBPEStreamingDetokenizer
                tokenmap = _get_bpe_tokenmap(tokenizer)

    return detokenizer_class, tokenmap


def load_tokenizer(model_path, trust_remote_code=True, tokenizer_config_extra=None, **kwargs):
    """
    Wrapper function for MLX load_tokenizer that defaults trust_remote_code to True.
    This is needed for models like Kimi-K2 that contain custom code.

    Args:
        model_path: Path to the model
        trust_remote_code: Whether to trust remote code (defaults to True)
        tokenizer_config_extra: Extra config to pass to AutoTokenizer.from_pretrained
        **kwargs: Additional arguments to pass to the original load_tokenizer

    Returns:
        The loaded tokenizer
    """
    if tokenizer_config_extra is None:
        tokenizer_config_extra = {}

    # Add trust_remote_code to the tokenizer config
    if trust_remote_code:
        tokenizer_config_extra = tokenizer_config_extra.copy()
        tokenizer_config_extra["trust_remote_code"] = True

    return _mlx_load_tokenizer(model_path, tokenizer_config_extra=tokenizer_config_extra, **kwargs)


@dataclass
class ToolCallState:
    has_tool_calling: bool
    tool_call_start: Optional[str]
    tool_call_end: Optional[str]
    tool_parser: Optional[Callable[[str, Any], Dict[str, Any]]]
    tools: Optional[List[Any]]
    stream: bool
    in_tool_call: bool = False
    tool_text: str = ""
    tool_call_idx: int = 0
    made_tool_call: bool = False

    @classmethod
    def from_tokenizer(cls, tokenizer, tools: Optional[List[Any]], stream: bool):
        has_tool_calling = bool(getattr(tokenizer, "has_tool_calling", False))
        tool_parser = getattr(tokenizer, "tool_parser", None)
        tool_call_start = getattr(tokenizer, "tool_call_start", None)
        tool_call_end = getattr(tokenizer, "tool_call_end", None)
        if not (has_tool_calling and tool_parser and tool_call_start and tool_call_end):
            return cls(
                has_tool_calling=False,
                tool_call_start=None,
                tool_call_end=None,
                tool_parser=None,
                tools=tools,
                stream=stream,
            )
        return cls(
            has_tool_calling=True,
            tool_call_start=tool_call_start,
            tool_call_end=tool_call_end,
            tool_parser=tool_parser,
            tools=tools,
            stream=stream,
        )

    def _format_tool_call(self, tool_call: Dict[str, Any]):
        tool_call_id = tool_call.pop("id", None) or str(uuid.uuid4())
        tool_call["arguments"] = json.dumps(tool_call["arguments"], ensure_ascii=False)
        out = {
            "function": tool_call,
            "type": "function",
            "id": tool_call_id,
        }
        if self.stream:
            out["index"] = self.tool_call_idx
            self.tool_call_idx += 1
        return out

    def _parse_tool_text(self, tool_text: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        try:
            parsed = self.tool_parser(tool_text, self.tools)
        except Exception:
            fallback_text = f"{self.tool_call_start}{tool_text}{self.tool_call_end}"
            return [], fallback_text
        if isinstance(parsed, list):
            return [self._format_tool_call(tc) for tc in parsed], None
        return [self._format_tool_call(parsed)], None

    def extract_from_segment(self, segment: str) -> Tuple[str, List[Dict[str, Any]]]:
        if not self.has_tool_calling or not segment:
            return segment, []
        output_chunks: List[str] = []
        new_tool_calls: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(segment):
            if not self.in_tool_call:
                start_pos = segment.find(self.tool_call_start, idx)
                if start_pos == -1:
                    output_chunks.append(segment[idx:])
                    break
                if start_pos > idx:
                    output_chunks.append(segment[idx:start_pos])
                self.in_tool_call = True
                self.made_tool_call = True
                self.tool_text = ""
                idx = start_pos + len(self.tool_call_start)
            else:
                end_pos = segment.find(self.tool_call_end, idx)
                if end_pos == -1:
                    self.tool_text += segment[idx:]
                    break
                self.tool_text += segment[idx:end_pos]
                parsed_calls, fallback_text = self._parse_tool_text(self.tool_text)
                if parsed_calls:
                    new_tool_calls.extend(parsed_calls)
                if fallback_text:
                    output_chunks.append(fallback_text)
                self.tool_text = ""
                self.in_tool_call = False
                idx = end_pos + len(self.tool_call_end)
        return "".join(output_chunks), new_tool_calls
