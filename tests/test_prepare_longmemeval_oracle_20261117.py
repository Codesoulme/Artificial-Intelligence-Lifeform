import unittest

from scripts.prepare_longmemeval_oracle_20261117 import prepare_case


class LongMemEvalPreparationTests(unittest.TestCase):
    def fixture(self):
        return {'question_id': 'hidden_abs', 'question_type': 'temporal-reasoning',
                'question': 'Which occurred first?', 'answer': 'Not enough information.',
                'question_date': '2023/05/28 (Sun) 21:04',
                'haystack_session_ids': ['answer_late', 'answer_early'],
                'haystack_dates': ['2023/05/28 (Sun) 20:00', '2023/05/28 (Sun) 10:00'],
                'haystack_sessions': [[{'role': 'user', 'content': 'Later', 'has_answer': True}],
                                      [{'role': 'assistant', 'content': 'Earlier'}]],
                'answer_session_ids': ['answer_late']}

    def test_sorted_public_data_has_no_gold_metadata(self):
        public, gold = prepare_case(self.fixture(), 'case-0001')
        self.assertEqual([s['turns'][0]['content'] for s in public['sessions']], ['Earlier', 'Later'])
        self.assertEqual(public['sessions'][1]['turns'][0], {'role': 'user', 'content': 'Later'})
        self.assertEqual(set(public), {'case_id', 'question_date', 'question', 'sessions'})
        self.assertEqual(gold['evidence_session_indexes'], [1])
        self.assertEqual(gold['evidence_turn_indexes'], [[1, 0]])
        self.assertTrue(gold['abstention'])
        self.assertNotIn('hidden_abs', repr(public))
        self.assertNotIn('answer_late', repr(public))

    def test_rejects_missing_evidence_and_duplicate_ids(self):
        record = self.fixture()
        record['answer_session_ids'] = ['missing']
        with self.assertRaises(ValueError):
            prepare_case(record, 'x')
        record = self.fixture()
        record['haystack_session_ids'] = ['answer_late', 'answer_late']
        with self.assertRaises(ValueError):
            prepare_case(record, 'x')
