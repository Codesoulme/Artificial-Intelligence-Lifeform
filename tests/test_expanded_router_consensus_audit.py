import unittest

from scripts.deepseek_spora_expanded_router_consensus_audit import agree_with_guard


class ExpandedRouterConsensusTests(unittest.TestCase):
    def test_matching_single_span_after_guard(self):
        row = {'guarded_accepted': True, 'guard_span': {'selected_text': '小林'}}
        self.assertTrue(agree_with_guard(row, ['小林']))
        self.assertFalse(agree_with_guard(row, ['小林华']))
        self.assertFalse(agree_with_guard(row, ['小林', '阿宁']))

    def test_rejection_or_missing_span_fails_closed(self):
        self.assertFalse(agree_with_guard({'guarded_accepted': False}, ['小林']))
        self.assertFalse(agree_with_guard({'guarded_accepted': True}, ['小林']))
        self.assertFalse(agree_with_guard({'guarded_accepted': True, 'guard_span': {'selected_text': ''}}, ['']))
        self.assertFalse(agree_with_guard({'guarded_accepted': True, 'guard_span': {'selected_text': '小林'}}, None))


if __name__ == '__main__':
    unittest.main()
