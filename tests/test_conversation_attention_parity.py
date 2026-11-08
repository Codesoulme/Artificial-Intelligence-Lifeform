import unittest

import torch

from spora.conversation import ConversationBlock, ConversationConfig


class ConversationAttentionParityTests(unittest.TestCase):
    def test_incremental_chunks_match_full_causal_forward(self):
        torch.manual_seed(20261108)
        block = ConversationBlock(ConversationConfig(width=32, heads=4, hidden=64, depth=2,
                                                      context=64, memory_layers=(0, 1))).eval()
        inputs = torch.randn(2, 19, 32)
        with torch.no_grad():
            expected, _ = block(inputs)
            past = None
            pieces = []
            start = 0
            for size in (7, 1, 4, 1, 6):
                output, past = block(inputs[:, start:start + size], past=past, use_past=True)
                pieces.append(output)
                start += size
        torch.testing.assert_close(torch.cat(pieces, dim=1), expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
