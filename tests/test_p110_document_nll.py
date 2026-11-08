import unittest

import numpy as np
import torch
from torch.nn import functional as F

from scripts.eval_p110_document_nll import evaluate, loss_sum, segments, selected_windows
from spora.document_stream import DocumentStream


class ContextModel(torch.nn.Module):
    def forward(self, tokens, state):
        # Make every prediction depend on all prior tokens, so leakage is observable.
        context = tokens.cumsum(dim=1).float()
        return torch.stack([context * .2, -context * .1, context * .05,
                            -context * .05, context * .1, -context * .2], dim=-1)


class DocumentNllTests(unittest.TestCase):
    def test_segments_and_boundary_validation(self):
        # BOS=1, EOS=2; also cover a partial first document and a trailing BOS.
        self.assertEqual(segments([4, 2, 1, 3, 2, 1], 1), [(0, 2), (2, 5)])
        valid = np.array([1, 4, 2, 1, 3, 2], dtype=np.uint16)
        DocumentStream(valid, 1, 2, 0)
        with self.assertRaises(ValueError):
            DocumentStream(np.array([1, 4, 1, 3, 2], dtype=np.uint16), 1, 2, 0)

    def test_mask_and_token_weighted_loss(self):
        logits = torch.tensor([[[2., 0.], [0., 2.], [1., 0.]]])
        labels = torch.tensor([[0, -100, 1]])
        total, count = loss_sum(logits, labels)
        expected = F.cross_entropy(logits[0, [0, 2]], torch.tensor([0, 1]), reduction='sum')
        self.assertEqual(count, 2)
        self.assertAlmostEqual(total, float(expected), places=6)
        self.assertEqual(loss_sum(logits, torch.full_like(labels, -100)), (0.0, 0))

    def test_paired_targets_context_reset_and_weighting(self):
        values = np.array([1, 4, 2, 1, 3, 5, 2, 1, 4, 2], dtype=np.uint16)
        model = ContextModel()
        result = evaluate(model, None, values, [0], 9, 1, 'cpu')
        self.assertEqual(result['cross_document_bos_targets'], 2)
        self.assertEqual(result['stream_all']['targets'], 9)
        self.assertEqual(result['stream_matched']['targets'], 7)
        self.assertEqual(result['document_within_window']['targets'], 7)
        pieces = [torch.tensor([[1, 4, 2]]), torch.tensor([[1, 3, 5, 2]]), torch.tensor([[1, 4, 2]])]
        expected = sum(float(F.cross_entropy(model(p[:, :-1], None).flatten(0, 1),
                                             p[:, 1:].flatten(), reduction='sum')) for p in pieces)
        self.assertAlmostEqual(result['document_within_window']['nll'], expected / 7, places=6)
        self.assertNotAlmostEqual(result['stream_matched']['nll'], expected / 7, places=5)

    def test_window_selection_reproducible_and_last_start_allowed(self):
        self.assertEqual(selected_windows(10, 4, 20, 77), selected_windows(10, 4, 20, 77))
        self.assertIn(5, selected_windows(10, 4, 100, 77))
        with self.assertRaises(ValueError):
            selected_windows(4, 4, 1, 0)


if __name__ == '__main__':
    unittest.main()
