import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts import deepseek_spora_memory_answer_distill as experiment


class FakeIds:
    def to(self, device):
        return self


class MemoryAnswerRowsTests(unittest.TestCase):
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
