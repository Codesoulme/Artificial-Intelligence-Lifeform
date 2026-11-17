import unittest
from scripts.audit_oasst_context_transfer_20261118 import build_rows, summarize
from scripts.prepare_oasst_evidence_review_20261118 import chains


class TransferTests(unittest.TestCase):
    def test_source_walk_excludes_future_answer_and_other_branches(self):
        def node(key, parent, role, text):
            return dict(message_id=key, parent_id=parent, role=role, text=text, lang='zh',
                        message_tree_id='tree', review_result=True, deleted=False,
                        labels={k: {'value': 0} for k in ('pii', 'spam', 'hate_speech', 'sexual_content')})
        raw = [node('a', None, 'prompter', '主题'), node('b', 'a', 'assistant', '历史回答'),
               node('c', 'b', 'prompter', '详细解释'), node('d', 'c', 'assistant', '不应泄漏的后续答案')]
        queue, count = chains(raw, set())
        self.assertEqual(count, 1)
        self.assertEqual(queue[0]['message_ids'], ['a', 'b', 'c'])
        self.assertNotIn('不应泄漏', str(queue))
        self.assertEqual(chains(raw, {'tree'})[0], [])

    def test_context_swap_changes_information_not_question(self):
        queue = [{'case': i, 'group': str(i), 'message_ids': [f'{i}a', f'{i}b', f'{i}c'],
                  'messages': [{'role': 'user', 'content': f'主题{i}'},
                               {'role': 'assistant', 'content': f'历史{i}'},
                               {'role': 'user', 'content': f'追问{i}'}]} for i in range(2)]
        reviews = [{'case': i, 'group': str(i), 'current_message_id': f'{i}c',
                    'needs_previous_context': 1 if i == 0 else None, 'wrong_history_case': 1-i} for i in range(2)]
        rows = build_rows(queue, reviews)
        self.assertEqual([r['need'] for r in rows], [1, 0, 1])
        self.assertTrue(all(r['text'].endswith('追问0') for r in rows))
        self.assertIn('历史1', rows[2]['text'])
        self.assertNotIn('历史0', rows[2]['text'])
        reviews[0]['wrong_history_case'] = 0
        with self.assertRaises(ValueError):
            build_rows(queue, reviews)

    def test_always_requests_history_is_not_success(self):
        rows = [{'history_dependent': 1, 'variant': v, 'need': n}
                for v, n in [('current_only', 1), ('original_history', 0), ('wrong_history', 1)]]
        result = summarize(rows, [1, 1, 1], 0)
        self.assertEqual(result['dependent_all_three_correct'], 0)
        self.assertEqual(result['original_history']['fp'], 1)
