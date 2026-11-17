import unittest
from scripts.prepare_dulemon_evidence_pairs_20261117 import build_pairs


class EvidencePairTests(unittest.TestCase):
    def setUp(self):
        self.dialogues = [{'conversation': ['Usr:你好', 'Bot:我喜欢篮球。']}]
        self.item = {'session': 0, 'evidence_turn': 1, 'speaker': 'Bot', 'span': [3, 5],
                     'question': 'Bot喜欢什么运动？', 'split': 'development'}

    def test_pair_is_causal_and_excludes_answer_from_earlier_context(self):
        before, after = build_pairs(self.dialogues, [self.item])
        self.assertIsNone(before['answer'])
        self.assertEqual(after['answer'], '篮球')
        self.assertEqual(len(before['context']) + 1, len(after['context']))
        self.assertEqual(before['source_sha256'], after['source_sha256'])
        self.assertNotIn(after['answer'], before['question'])

    def test_rejects_early_leak_duplicate_session_wrong_speaker_and_bounds(self):
        cases = [([{'conversation': ['Usr:篮球好看', 'Bot:我喜欢篮球。']}], [self.item]),
                 (self.dialogues, [self.item, self.item]),
                 (self.dialogues, [{**self.item, 'speaker': 'Usr'}]),
                 (self.dialogues, [{**self.item, 'span': [3, 99]}]),
                 (self.dialogues, [{**self.item, 'question': 'Bot喜欢篮球吗？'}])]
        for dialogues, items in cases:
            with self.subTest(items=items), self.assertRaises(ValueError):
                build_pairs(dialogues, items)
