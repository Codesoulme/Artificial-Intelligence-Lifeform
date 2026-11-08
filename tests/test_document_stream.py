import random
import unittest

import numpy as np

import torch
from spora.document_stream import DocumentStream, trim_ignored_tail


class DocumentStreamTests(unittest.TestCase):
    def test_each_sample_stays_in_one_document_and_padding_is_ignored(self):
        source = np.array([1, 10, 11, 2, 1, 20, 21, 22, 2], dtype=np.uint16)
        sampler = DocumentStream(source, bos=1, eos=2, pad=0)
        x, y = sampler.batch(random.Random(12), 100, 6, 'cpu')
        for inputs, targets in zip(x.tolist(), y.tolist()):
            valid = [value for value in targets if value != -100]
            self.assertTrue(valid)
            self.assertFalse(10 in inputs and 20 in inputs)
            self.assertNotIn(1, valid)
            self.assertTrue(all(value == 0 for value in inputs[len(valid):]))
            self.assertTrue(all(value == -100 for value in targets[len(valid):]))

    def test_rejects_unpaired_document_boundaries(self):
        with self.assertRaises(ValueError):
            DocumentStream(np.array([1, 10, 1, 20, 2], dtype=np.uint16), 1, 2, 0)

    def test_trim_keeps_context_and_last_supervised_position_across_rows(self):
        x = torch.tensor([[1, 10, 11, 12, 0, 0], [1, 20, 21, 22, 0, 0]])
        y = torch.tensor([[-100, 11, -100, -100, -100, -100], [-100, -100, 22, -100, -100, -100]])
        actual_x, actual_y = trim_ignored_tail(x, y)
        torch.testing.assert_close(actual_x, x[:, :3])
        torch.testing.assert_close(actual_y, y[:, :3])
        with self.assertRaises(ValueError):
            trim_ignored_tail(x, torch.full_like(x, -100))


if __name__ == '__main__':
    unittest.main()
