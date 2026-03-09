"""
Test for the Sampler class
"""

import unittest

import numpy as np
import mlx.core as mx
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

from parallax.server.sampling.sampler import Sampler, SamplingBatchInfo


class TestSampler(unittest.TestCase):
    """Tests the correctness of topk/topp/minp sampling"""

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
            logit_biases=[None, None, None],
            need_logprobs=False,
            top_logprobs_per_req=[0, 0, 0],
            output_ids_per_req=[[], [], []],
            presence_penalties=np.zeros(3, dtype=np.float32),
            frequency_penalties=np.zeros(3, dtype=np.float32),
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

        mx.allclose(batch_next_token_ids, next_token_ids_ref)


if __name__ == "__main__":
    unittest.main()
