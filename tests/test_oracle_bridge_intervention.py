import unittest
from contextvars import ContextVar
from types import SimpleNamespace

import torch
from spora.research_oracle_bridge import oracle_embeddings, context_oracle_embeddings


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

    def test_reads_actual_context_and_rejects_missing_context(self):
        context = ContextVar('test', default=None)
        decoder = SimpleNamespace(_memory_context=context)
        codec = SimpleNamespace(codebook=torch.eye(3), slot_count=1, slot_width=3)
        embedding = torch.eye(3)
        codes = torch.ones(1, 1, 3)
        with self.assertRaisesRegex(ValueError, 'explicit memory'):
            context_oracle_embeddings(decoder, codec, embedding, codes)
        address = torch.tensor([[1., 0., 0.]])
        matrix = torch.zeros(1, 3, 3)
        matrix[0, 1, 0] = 1.
        token = context.set((((matrix,), (matrix,)), (address, address), None, None))
        try:
            result = context_oracle_embeddings(decoder, codec, embedding, codes)
        finally:
            context.reset(token)
        torch.testing.assert_close(result, torch.tensor([[[0., 1., 0.]]]))


if __name__ == '__main__':
    unittest.main()
