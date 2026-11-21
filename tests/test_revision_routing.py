import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from spora.revision_routing import parse_revision_reference, select_numbered_revision


class RevisionRoutingTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(config=SimpleNamespace(key_count=8))
        self.state = SimpleNamespace(cursor=10, internal=SimpleNamespace(
            start_cursor=0, event_keys=[torch.tensor([1, 2, 1, 1, 1]), torch.tensor([1, 1, 2, 0, 1])],
            validate=Mock(),
        ))

    def test_parse_multiple_ordinal_forms_and_abstentions(self):
        self.assertEqual(parse_revision_reference('第２次修订完成后给小林登记的是什么？'),
                         {'status': 'anchored', 'number': 2, 'side': 'after'})
        self.assertEqual(parse_revision_reference('小林第十二次更正以前是哪样？'),
                         {'status': 'anchored', 'number': 12, 'side': 'before'})
        self.assertIsNone(parse_revision_reference('现在是哪样？'))
        for question, reason in (
            ('第1次更正前和第2次更正后各是什么？', 'multiple_correction_anchors'),
            ('第2次更正前后分别是哪样？', 'ambiguous_correction_side'),
            ('第2次更正的饮品是哪样？', 'missing_correction_side'),
            ('第0次更新前是哪样？', 'invalid_correction_ordinal'),
            ('翻译「第一次更正前是什么」这句话。', 'quoted_correction_reference'),
        ):
            self.assertEqual(parse_revision_reference(question), {'status': 'abstain', 'reason': reason})

    def test_numbered_revision_counts_only_committed_corrections_for_requested_key(self):
        rows = [{'action': action, 'valid_at': 2} for action in
                ('new', 'repeat', 'correct', 'support', 'defer', 'correct', 'correct')]
        with patch('spora.revision_routing.read_provenance_batch', return_value=rows) as reader:
            before = select_numbered_revision(self.model, self.state, 1, 2, 'before')
            after = select_numbered_revision(self.model, self.state, 1, 2, 'after')
            missing = select_numbered_revision(self.model, self.state, 1, 4, 'after')
        reader.assert_called_with(self.model, self.state, [(1, t) for t in (0, 2, 3, 4, 5, 6, 9)])
        self.assertEqual(before['event_time'], 3)
        self.assertEqual(after['event_time'], 6)
        self.assertEqual(after['correction_event'], 6)
        self.assertEqual(missing['reason'], 'correction_ordinal_not_found')

    def test_uncertain_indexed_event_blocks_ordinal_counting(self):
        rows = [{'action': 'new'}, {'reason': 'unknown_event'}] + [{'action': 'correct'}] * 5
        with patch('spora.revision_routing.read_provenance_batch', return_value=rows):
            result = select_numbered_revision(self.model, self.state, 1, 1, 'after')
        self.assertEqual(result, {'status': 'abstain', 'reason': 'unreadable_indexed_provenance', 'event_time': 2})
        self.state.internal.event_keys = None
        self.assertEqual(select_numbered_revision(self.model, self.state, 1, 1, 'after')['reason'],
                         'complete_event_key_index_required')
