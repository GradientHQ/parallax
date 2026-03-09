"""
Sampling parameters of each request
"""

from typing import Dict, List, Optional, Union


def _normalize_logit_bias(
    value: Optional[Union[Dict[int, float], Dict[str, float]]]
) -> Optional[Dict[int, float]]:
    """Normalize logit_bias to Dict[int, float], converting string keys to int."""
    if value is None or len(value) == 0:
        return None
    return {int(k): float(v) for k, v in value.items()}


class SamplingParams:
    """Sampling parameter class for a single request"""

    def __init__(
        self,
        max_new_tokens: int = 128,
        min_new_tokens: int = 0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = -1,
        stop_token_ids: Optional[List[int]] = None,
        ignore_eos: bool = False,
        stop_strs: Optional[Union[str, List[str]]] = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        json_schema: Optional[str] = None,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        logit_bias: Optional[Union[Dict[int, float], Dict[str, float]]] = None,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.top_k = top_k
        if stop_token_ids:
            self.stop_token_ids = set(stop_token_ids)
        else:
            self.stop_token_ids = None
        self.ignore_eos = ignore_eos
        self.stop_strs = stop_strs
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.json_schema = json_schema
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.logit_bias = _normalize_logit_bias(logit_bias)

        # Some special cases
        if self.temperature == 0.0:
            # greedy sampling
            self.temperature = 1.0
            self.top_k = 1

    def verify(self):
        """Basic verifications for the sampling parameters"""
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be non-negetive, got {self.temperature}.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if not 0.0 < self.min_p <= 1.0:
            raise ValueError(f"min_p must be in (0, 1], got {self.min_p}.")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(f"frequency_penalty must be in [-2, 2], got {self.frequency_penalty}.")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(f"presence_penalty must be in [-2, 2], got {self.presence_penalty}.")
        if not 0.0 <= self.repetition_penalty <= 2.0:
            raise ValueError(
                f"repetition_penalty must be in [0, 2], got {self.repetition_penalty}."
            )
        if self.logprobs and (self.top_logprobs is None or not 0 <= self.top_logprobs <= 20):
            raise ValueError(
                "when logprobs is True, top_logprobs must be in [0, 20], "
                f"got {self.top_logprobs}."
            )
        if self.logit_bias is not None:
            for token_id, bias in self.logit_bias.items():
                if not -100.0 <= bias <= 100.0:
                    raise ValueError(
                        f"logit_bias values must be in [-100, 100], got {bias} for token {token_id}."
                    )
