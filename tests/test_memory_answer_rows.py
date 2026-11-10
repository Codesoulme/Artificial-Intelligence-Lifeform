import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from scripts import deepseek_spora_memory_answer_distill as experiment


class FakeIds:
    def to(self, device):
        return self


class MemoryAnswerRowsTests(unittest.TestCase):
    def test_candidate_loader_uses_validated_checkpoint_position_length(self):
        base = SimpleNamespace(config=SimpleNamespace(hidden_size=8, num_attention_heads=2))
        block = experiment.SPORALayerReplacement(8, 2, max_length=12, layer_idx=6)
        metadata = {'format': 'spora-qwen-layer-replacement-candidate-v1',
                    'base_weight_sha256': experiment.WEIGHT_SHA, 'layer': 6,
                    'width': 8, 'heads': 2, 'max_length': 12}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'candidate.pt'
            torch.save({'metadata': metadata, 'state': block.state_dict()}, path)
            with patch.object(experiment.SPORALayerReplacement, 'to', lambda self, device: self):
                restored = experiment._load_candidate(path, 6, base)
                self.assertEqual(restored.max_length, 12)
                torch.testing.assert_close(restored.position.weight, block.position.weight)
                metadata['max_length'] = 0
                torch.save({'metadata': metadata, 'state': block.state_dict()}, path)
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    experiment._load_candidate(path, 6, base)

    def test_every_template_contributes_a_distinct_row(self):
        tokenizer = type('Tokenizer', (), {
            'eos_token_id': 1,
            '__call__': lambda self, *args, **kwargs: SimpleNamespace(input_ids=FakeIds()),
        })()
        facts = [dict(name=f'name{i}', key=i, old_value=0, new_value=1,
                      old_time=2*i, new_time=2*i+1, train_split=i < 6)
                 for i in range(8)]
        with patch.object(experiment, 'closed_think_inputs', return_value={}), \
             patch.object(experiment.torch, 'tensor', return_value=FakeIds()), \
             patch.object(experiment.torch, 'cat', return_value=FakeIds()):
            train, held = experiment._rows(facts, tokenizer)
        self.assertEqual(len(train), 6 * 3 * 2)
        self.assertEqual(len(held), 2 * 3 * 2)
        self.assertEqual(len({row['question'] for row in train}), len(train))
        self.assertEqual(len({row['question'] for row in held}), len(held))
        self.assertEqual({row['variant'] for row in train}, {0, 1})
        self.assertEqual({row['variant'] for row in held}, {0, 1})


if __name__ == '__main__':
    unittest.main()
