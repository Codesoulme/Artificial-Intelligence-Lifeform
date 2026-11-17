import copy
import unittest
from scripts.prepare_oasst_donor_review_20261118 import build_pairs


def fixture():
    queue = [dict(case=i, group=f'g{i}', partition='train_review' if i < 3 else 'calibration_review',
                  message_ids=[f'{i}-1', f'{i}-2', f'{i}-3'],
                  messages=[{'role':'user','content':f'subject{i}'},
                            {'role':'assistant','content':f'artifact{i}'},
                            {'role':'user','content':f'question{i}'}]) for i in range(6)]
    accepted = [dict(case=r['case'], partition=r['partition']) for r in queue if r['case'] != 1]
    return queue, accepted


class DonorTests(unittest.TestCase):
    def test_pairs_are_stable_local_and_use_accepted_sources(self):
        q,a = fixture()
        rows = build_pairs(q,a)
        self.assertEqual([(r['case'],r['donor_case']) for r in rows], [(0,2),(2,0),(3,4),(4,5),(5,3)])
        self.assertEqual(rows, build_pairs(list(reversed(q)),list(reversed(a))))
        for row in rows:
            self.assertNotEqual(row['group'], row['donor_group'])
            self.assertEqual(row['question'], q[row['case']]['messages'][-1])
            self.assertEqual(row['donor_history'], q[row['donor_case']]['messages'][:-1])

    def test_holdout_partition_spoof_and_missing_donor_fail(self):
        for case in ('reserved','spoof','missing'):
            q,a = copy.deepcopy(fixture())
            if case == 'reserved': q[0]['partition']='reserved_holdout'
            if case == 'spoof': a[0]['partition']='calibration_review'
            if case == 'missing': a=[r for r in a if r['case'] != 2]
            with self.assertRaises(ValueError): build_pairs(q,a)
