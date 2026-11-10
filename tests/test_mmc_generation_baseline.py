import unittest

from scripts.deepseek_mmc_generation_baseline import parse_answer, summarize


class GenerationBaselineTests(unittest.TestCase):
    def test_accepts_only_terminated_unambiguous_answers(self):
        self.assertEqual(parse_answer('是。', 'closed', True), (1, 'parsed'))
        self.assertEqual(parse_answer('推理</think>\n否。', 'normal', True), (0, 'parsed'))
        self.assertEqual(parse_answer('是。', 'closed', False), (None, 'not_terminated'))
        self.assertEqual(parse_answer('是。', 'normal', True), (None, 'reasoning_not_closed'))
        self.assertEqual(parse_answer('是，但也可能不是。', 'closed', True), (None, 'unrecognized_answer'))
        self.assertEqual(parse_answer('a</think>b</think>是', 'normal', True), (None, 'reasoning_not_closed'))

    def test_summary_preserves_unanswered(self):
        rows = [{'mode': 'normal', 'label': 0, 'prediction': 1, 'ended': True, 'reason': 'parsed'},
                {'mode': 'normal', 'label': 1, 'prediction': None, 'ended': False, 'reason': 'not_terminated'}]
        result = summarize(rows, 'normal')
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['parsed'], 1)
        self.assertEqual(result['wrong_same'], 1)
        self.assertEqual(result['reasons']['not_terminated'], 1)


if __name__ == '__main__':
    unittest.main()
