import unittest

import torch
from torch import nn

from scripts.deepseek_spora_lane_embedding_bridge_pilot import metrics


class IdentityBridge(nn.Module):
    def forward(self, values):
        return values


class LaneEmbeddingBridgeTests(unittest.TestCase):
    def test_rank_uses_same_score_matrix_and_rejects_missing_target(self):
        embeddings = torch.eye(3)
        codebook = embeddings.clone()
        result = metrics(IdentityBridge(), codebook, embeddings, torch.tensor([0, 2]), torch.tensor([0, 1, 2]))
        self.assertEqual(result['correct_unique'], 2)
        self.assertEqual(result['mean_rank'], 1)
        with self.assertRaises(ValueError):
            metrics(IdentityBridge(), codebook, embeddings, torch.tensor([2]), torch.tensor([0, 1]))

    def test_identical_embedding_is_not_counted_as_unique_match(self):
        embeddings = torch.tensor([[1., 0.], [1., 0.]])
        result = metrics(IdentityBridge(), embeddings, embeddings, torch.tensor([0]), torch.tensor([0, 1]))
        self.assertEqual(result['correct_unique'], 0)


if __name__ == '__main__':
    unittest.main()
