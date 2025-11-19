"""
Implements parallax detokenizers for performance.
"""

import json
from functools import partial
from json import JSONDecodeError

from mlx_lm.tokenizer_utils import (
    BPEStreamingDetokenizer,
    NaiveStreamingDetokenizer,
    SPMStreamingDetokenizer,
    _is_bpe_decoder,
    _is_spm_decoder,
    _is_spm_decoder_no_space,
)
from mlx_lm.tokenizer_utils import load_tokenizer as _mlx_load_tokenizer


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
