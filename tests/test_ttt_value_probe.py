import unittest

import torch

from scripts.deepseek_spora_ttt_value_probe import probe


class ValueProbeTests(unittest.TestCase):
    def test_centroids_use_training_labels_and_hold_out_identity(self):
        features = torch.tensor([[[1., 0.], [0., 1.]],
                                 [[2., 0.], [0., 2.]],
                                 [[3., 0.], [0., 3.]]])
        predictions, correct = probe(features, [0, 1], ['a', 'b'], ['held'])
        self.assertEqual(predictions, [0, 1])
        self.assertEqual(correct, 2)
        altered = features.clone()
        altered[2] = altered[2].flip(0)
        predictions, correct = probe(altered, [0, 1], ['a', 'b'], ['held'])
        self.assertEqual(predictions, [1, 0])
        self.assertEqual(correct, 0)

    def test_centering_removes_per_identity_offset(self):
        features = torch.tensor([[[10., 0.], [0., 10.]],
                                 [[110., 100.], [100., 110.]]])
        predictions, correct = probe(features, [0, 1], ['a'], ['held'], center=True)
        self.assertEqual(predictions, [0, 1])
        self.assertEqual(correct, 2)


if __name__ == '__main__':
    unittest.main()
