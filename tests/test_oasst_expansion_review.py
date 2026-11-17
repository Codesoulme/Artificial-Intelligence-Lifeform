import copy
import unittest

from scripts.reconcile_oasst_expansion_20261118 import reconcile


def fixture():
    queue = [dict(case=i, group=f'tree{i}', partition='train_review' if i < 2 else 'calibration_review',
                  messages=[{'role': 'user', 'content': f'subject{i}'},
                            {'role': 'assistant', 'content': f'answer{i}'},
                            {'role': 'user', 'content': f'question{i}'}],
                  message_ids=[f'{i}a', f'{i}b', f'{i}c']) for i in range(3)]
    a = {'reviewer': 'synthetic A', 'reviews': [dict(case=r['case'], partition=r['partition'],
         context_need=1 if r['case'] == 0 else 0, history_resolves=True if r['case'] == 0 else None,
         quality_flag='', rationale='synthetic reason') for r in queue]}
    b = copy.deepcopy(a)
    b['reviewer'] = 'synthetic B'
    return queue, a, b


class ReviewTests(unittest.TestCase):
    def test_labels_are_missing_context_and_not_wrong_history(self):
        result, rows = reconcile(*fixture())
        self.assertEqual([r['missing_context'] for r in rows], [1, 0, 0, 0, 0, 0])
        self.assertEqual({r['variant'] for r in rows}, {'current_only', 'original_history'})
        self.assertEqual(rows[0]['history_message_ids'], [])
        self.assertEqual(rows[1]['history_message_ids'], ['0a', '0b'])
        self.assertEqual(result['counts']['calibration_review']['accepted'], 1)

    def test_disagreement_and_unresolved_are_excluded_without_relabeling(self):
        queue, a, b = fixture()
        b['reviews'][0]['history_resolves'] = False
        a['reviews'][1].update(context_need=None, history_resolves=None)
        b['reviews'][1].update(context_need=None, history_resolves=None)
        old = copy.deepcopy((a, b))
        result, rows = reconcile(queue, a, b)
        self.assertEqual([r['case'] for r in result['excluded']], [0, 1])
        self.assertEqual({r['case'] for r in rows}, {2})
        self.assertEqual((a, b), old)

    def test_coverage_partition_types_and_rationale_fail_closed(self):
        for mutation in ('missing', 'duplicate', 'partition', 'boolean', 'resolution', 'rationale'):
            with self.subTest(mutation=mutation):
                queue, a, b = fixture()
                if mutation == 'missing': b['reviews'].pop()
                if mutation == 'duplicate': b['reviews'][1] = copy.deepcopy(b['reviews'][0])
                if mutation == 'partition': b['reviews'][0]['partition'] = 'calibration_review'
                if mutation == 'boolean': b['reviews'][0]['context_need'] = True
                if mutation == 'resolution': b['reviews'][1]['history_resolves'] = False
                if mutation == 'rationale': b['reviews'][0]['rationale'] = ''
                with self.assertRaises(ValueError): reconcile(queue, a, b)

    def test_reserved_partition_and_duplicate_source_are_rejected(self):
        for mutation in ('holdout', 'tree', 'text'):
            with self.subTest(mutation=mutation):
                queue, a, b = fixture()
                if mutation == 'holdout': queue[0]['partition'] = 'reserved_holdout'
                if mutation == 'tree': queue[1]['group'] = queue[0]['group']
                if mutation == 'text': queue[1]['messages'][-1] = queue[0]['messages'][-1]
                with self.assertRaises(ValueError): reconcile(queue, a, b)
