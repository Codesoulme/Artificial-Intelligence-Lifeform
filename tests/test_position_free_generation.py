import unittest

import torch

from scripts import deepseek_spora_position_free_generation as audit


class PositionFreeGenerationTests(unittest.TestCase):
    def test_tail_validation_checks_all_layers_before_use(self):
        sources = {'6': 'a', '12': 'b'}
        payload = {'format': 'spora-position-tail-v1', 'base_weight_sha256': audit.base.WEIGHT_SHA,
                   'sources': sources, 'old_length': 128, 'new_length': audit.LIMIT,
                   'tails': {'6': torch.zeros(audit.LIMIT - 128, 3),
                             '12': torch.zeros(audit.LIMIT - 128, 4)}}
        audit.validate_tail(payload, sources, {'6': 3, '12': 4})
        payload['tails']['12'][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'tensor'):
            audit.validate_tail(payload, sources, {'6': 3, '12': 4})

    def test_final_answer_omits_unfinished_reasoning(self):
        class Tokenizer:
            def decode(self, ids, skip_special_tokens):
                return ids

        self.assertEqual(audit.summarize_tail(Tokenizer(), 'unfinished'),
                         {'reasoning_closed': False, 'final_answer': ''})
        self.assertEqual(audit.summarize_tail(Tokenizer(), 'private</think>\nanswer'),
                         {'reasoning_closed': True, 'final_answer': 'answer'})


if __name__ == '__main__':
    unittest.main()
