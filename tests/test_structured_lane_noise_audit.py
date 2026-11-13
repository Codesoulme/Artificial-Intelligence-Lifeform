import unittest

import torch
from torch.nn import functional as F

from scripts.deepseek_spora_structured_lane_noise_audit import projection_for, score
from spora.structured_text_memory import StructuredTextMemoryCodec


class StructuredLaneNoiseTests(unittest.TestCase):
    def test_projection_matches_actual_fact_lane(self):
        embedding = torch.randn(12, 20, generator=torch.Generator().manual_seed(19))
        codec = StructuredTextMemoryCodec(embedding, memory_width=90, identity_slots=3, fact_slots=5,
                                          identity_slot_width=15, fact_slot_width=9, seed=29071)
        self.assertTrue(torch.equal(codec.fact_projection, projection_for(20, 9, 29071)))

    def test_score_counts_ties_and_perturbation(self):
        codebook = F.normalize(torch.tensor([[1., 0.], [0., 1.], [1., 0.]]), dim=-1)
        ids = torch.tensor([0, 1])
        noise = torch.tensor([[0., 0.], [1., -2.]])
        clean = score(codebook, ids, noise, 0.0, batch_size=1)
        self.assertEqual(clean['correct_unique'], 1)
        self.assertEqual(clean['ties'], 1)
        disturbed = score(codebook, ids, noise, 1.0, batch_size=1)
        self.assertEqual(disturbed['correct_unique'], 0)
        self.assertLess(disturbed['margin_min'], 0)


if __name__ == '__main__':
    unittest.main()
