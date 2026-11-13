import unittest

import torch
from scripts.deepseek_spora_oracle_bridge_intervention import oracle_embeddings


class OracleBridgeTests(unittest.TestCase):
    def test_correct_slot_lookup_and_empty_mask(self):
        codebook = torch.eye(3)
        embeddings = torch.tensor([[3., 0.], [0., 4.], [-3., 4.]])
        raw = torch.tensor([[[0., 2., 0.], [0., 0., 0.], [0., 0., 5.]]])
        result = oracle_embeddings(raw, codebook, embeddings)
        torch.testing.assert_close(result, torch.tensor([[[0., 1.], [0., 0.], [-0.6, 0.8]]]))

    def test_rejects_mismatched_vocabulary(self):
        with self.assertRaises(ValueError):
            oracle_embeddings(torch.ones(1, 1, 3), torch.eye(3), torch.ones(2, 4))


if __name__ == '__main__':
    unittest.main()
