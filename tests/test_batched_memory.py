import unittest
from types import SimpleNamespace

import torch

from spora.batched_memory import batch_states
from spora.sleep_memory import SealedPage


def state(current_scale, history_scale):
    return SimpleNamespace(
        current=[torch.eye(3).unsqueeze(0) * current_scale for _ in range(2)],
        pages=[[[torch.eye(3).unsqueeze(0) * history_scale, torch.eye(3).unsqueeze(0)]
                for _ in range(2)],
               [[torch.eye(3).unsqueeze(0) * (history_scale + 1), torch.eye(3).unsqueeze(0)]
                for _ in range(2)]])


class BatchedMemoryTests(unittest.TestCase):
    def test_batches_current_and_all_history_pages_in_order(self):
        a, b = state(1., 2.), state(3., 4.)
        batched = batch_states([a, b])
        self.assertEqual(batched.current[0].shape, (2, 3, 3))
        self.assertEqual(batched.pages[0][1][0].shape, (2, 3, 3))
        self.assertTrue(torch.equal(batched.current[0][0], a.current[0][0]))
        self.assertTrue(torch.equal(batched.current[0][1], b.current[0][0]))
        self.assertTrue(torch.equal(batched.pages[1][0][0][0], a.pages[1][0][0][0]))
        self.assertTrue(torch.equal(batched.pages[1][0][0][1], b.pages[1][0][0][0]))
        batched.current[0].zero_()
        self.assertTrue(torch.equal(a.current[0], torch.eye(3).unsqueeze(0)))

    def test_batches_sealed_history_pages(self):
        def sealed(scale):
            factors = [[torch.eye(3)[:, :1] * scale, torch.eye(3)[:1, :] * scale, torch.ones(6)]
                       for _ in range(2)]
            return SealedPage(factors, 1)
        a = SimpleNamespace(current=[torch.eye(3)[None] for _ in range(2)], pages=[sealed(1.)])
        b = SimpleNamespace(current=[torch.eye(3)[None] * 2 for _ in range(2)], pages=[sealed(3.)])
        batched = batch_states([a, b])
        self.assertEqual(batched.pages[0][0][0].shape, (2, 3, 3))
        self.assertTrue(torch.equal(batched.pages[0][0][0][0], a.pages[0].materialize(0)[0][0]))
        self.assertTrue(torch.equal(batched.pages[0][0][0][1], b.pages[0].materialize(0)[0][0]))

    def test_rejects_mismatched_or_nonfinite_states(self):
        with self.assertRaisesRegex(ValueError, '(?i)at least one'):
            batch_states([])
        with self.assertRaisesRegex(ValueError, 'page counts'):
            batch_states([state(1., 1.), SimpleNamespace(current=state(1., 1.).current, pages=[])])
        invalid = state(1., 1.)
        invalid.current[0][0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            batch_states([invalid])


if __name__ == '__main__':
    unittest.main()
