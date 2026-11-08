import tempfile
import unittest
from pathlib import Path

import torch

from spora.codepoint import CodepointVocabulary, windows
from spora.language_bridge import BOS, EOS, PAD, causal_loss
from spora.language_training import interface_state, load_interface
from spora.temporal import TemporalConfig, TemporalSPORA


class CodepointTests(unittest.TestCase):
    def test_roundtrip_and_unknown_are_deterministic(self):
        vocab = CodepointVocabulary(['你', '好'])
        self.assertEqual(vocab.decode(vocab.encode('你好')), '你好')
        self.assertEqual(vocab.encode('未知')[1], 259)
        self.assertEqual(vocab.size, 262)

    def test_windows_keep_boundaries_and_padding(self):
        vocab = CodepointVocabulary(['甲', '乙', '丙'])
        docs = [{'split': 'train', 'text': '甲乙丙', 'path': 'x'},
                {'split': 'development', 'text': '甲', 'path': 'y'}]
        result = windows(docs, vocab, 'train', 6)
        self.assertEqual(result.shape, (1, 6))
        self.assertEqual(int(result[0, 0]), BOS)
        self.assertEqual(int(result[0, 5]), PAD)
        self.assertIn(EOS, result.tolist()[0])

    def test_interface_vocab_and_context_are_strictly_bound(self):
        torch.manual_seed(7)
        organism = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2,
            memory_width=16, expert_rank=4, memory_layers=(0, 1), history_writer='rls'))
        vocab = CodepointVocabulary(['你', '好'])
        from spora.codepoint import CodepointLanguageBridge
        bridge = CodepointLanguageBridge(organism, vocab, context_layers=2)
        tokens = torch.tensor([vocab.encode('你好')])
        loss = causal_loss(bridge(tokens, __import__('spora.growing', fromlist=['new_state']).new_state(organism)), tokens)
        self.assertTrue(torch.isfinite(loss))
        payload = {'format': 'spora-codepoint-interface-v1', 'context_layers': 2,
                   'vocab_size': vocab.size, 'organism_sha256': 'parent',
                   'state': interface_state(bridge)}
        load_interface(CodepointLanguageBridge(organism, vocab, context_layers=2), payload, 'parent')
        with self.assertRaisesRegex(ValueError, 'vocabulary size'):
            load_interface(CodepointLanguageBridge(organism, CodepointVocabulary(['你']), context_layers=2), payload, 'parent')


if __name__ == '__main__':
    unittest.main()
