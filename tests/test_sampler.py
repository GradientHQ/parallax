"""
Test for the Sampler class
"""

import unittest

import mlx.core as mx
import pytest
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

from parallax.server.request import InitialRequest
from parallax.server.sampling.sampler import Sampler, SamplingBatchInfo
from parallax.server.sampling.sampling_params import SamplingParams

pytestmark = pytest.mark.mlx


class TestSampler(unittest.TestCase):
    """Tests the correctness of topk/topp/minp sampling"""

    def test_top_k_zero_and_negative_are_not_greedy(self):
        """top_k<=0 means unlimited top-k, not greedy sampling."""
        req_top_k_zero = InitialRequest.from_prompt_ids([1, 2], 1, 3)
        req_top_k_zero.sampling_params = SamplingParams(top_k=0)
        req_top_k_negative = InitialRequest.from_prompt_ids([1, 2], 1, 3)
        req_top_k_negative.sampling_params = SamplingParams(top_k=-1)

        self.assertFalse(SamplingBatchInfo.from_reqs([req_top_k_zero]).is_all_greedy)
        self.assertFalse(SamplingBatchInfo.from_reqs([req_top_k_negative]).is_all_greedy)

    def test_top_k_zero_keeps_full_distribution(self):
        """The custom sampling kernel treats top_k=0 as no top-k mask."""
        logits = mx.log(mx.array([[0.1, 0.2, 0.3, 0.4]]))
        sampling_info = SamplingBatchInfo(
            temperatures=mx.array([1.0], dtype=mx.float32).reshape(-1, 1),
            top_ps=mx.array([1.0], dtype=mx.float32),
            top_ks=mx.array([0], dtype=mx.int32),
            min_ps=mx.array([0.0], dtype=mx.float32),
            is_all_greedy=False,
            need_min_p_sampling=False,
        )

        sampled_token_ids = []
        for seed in range(10):
            mx.random.seed(seed)
            sampled_token_ids.append(int(Sampler()(logits, sampling_info)[0]))

        self.assertTrue(any(token_id != 3 for token_id in sampled_token_ids))

    def test_sampling(self):
        """Sampling test method"""
        temperatures = mx.array([0.5, 0.95, 1.0], dtype=mx.float32)
        top_ks = mx.array([3, 3, 1], dtype=mx.int32)
        top_ps = mx.array([0.8, 0.9, 1.0], dtype=mx.float32)
        min_ps = mx.array([0.2, 0.05, 0.05], dtype=mx.float32)

        probs = mx.array([[0.2, 0.0, 0.7, 0.1], [0.1, 0.0, 0.0, 0.9], [0.5, 0.3, 0.1, 0.1]])
        logits = mx.log(probs)

        # test sampling
        mx.random.seed(42)
        sampling_info = SamplingBatchInfo(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            is_all_greedy=False,
            need_min_p_sampling=True,
        )
        sampler = Sampler()
        batch_next_token_ids = sampler(logits, sampling_info)

        # calculate mx refs
        mx.random.seed(42)
        logits = mx.array([apply_top_k(logits[i], int(top_ks[i])) for i in range(3)])
        logits = mx.array([apply_top_p(logits[i], float(top_ps[i])) for i in range(3)])
        logits = mx.array(
            [apply_min_p(logits[i].reshape(1, -1), float(min_ps[i])).reshape(-1) for i in range(3)]
        )
        logits = logits / temperatures.reshape(-1, 1)
        next_token_ids_ref = mx.random.categorical(logits)

        self.assertEqual(batch_next_token_ids.shape, (3,))
        self.assertTrue(mx.allclose(batch_next_token_ids, next_token_ids_ref))


if __name__ == "__main__":
    unittest.main()
