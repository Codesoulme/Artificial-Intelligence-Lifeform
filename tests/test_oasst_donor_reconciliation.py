import copy
import unittest
from scripts.prepare_oasst_donor_review_20261118 import build_pairs
from scripts.reconcile_oasst_donor_review_20261118 import reconcile
from tests.test_oasst_donor_review import fixture


def reviewed():
    queue, accepted = fixture()
    for r in accepted:
        r['context_need'] = int(r['case'] == 0)
    pairs = build_pairs(queue,accepted)
    reviews = [dict(case=p['case'], donor_case=p['donor_case'], partition=p['partition'],
                    presented_missing_context=0, original_intent_preserved=p['case']!=0,
                    plausible_rebinding=p['case']==0, rationale='synthetic review',quality_flag='') for p in pairs]
    return pairs, {'reviewer':'C','reviews':reviews}, {'reviewer':'D','reviews':copy.deepcopy(reviews)}, accepted


class ReconciliationTests(unittest.TestCase):
    def test_answerable_rebinding_is_not_forced_to_missing(self):
        result, rows = reconcile(*reviewed())
        first = rows[0]
        self.assertEqual(first['missing_context'], 0)
        self.assertFalse(first['original_intent_preserved'])
        self.assertTrue(first['plausible_rebinding'])
        self.assertEqual(result['counts']['train_review']['naive_target_disagrees'],1)
        self.assertEqual(result['counts']['train_review']['answerable_but_intent_lost'],1)

    def test_disagreement_is_not_overwritten_by_original_label(self):
        pairs,c,d,originals=reviewed()
        d['reviews'][0]['presented_missing_context']=1
        snapshot=copy.deepcopy((c,d))
        result,rows=reconcile(pairs,c,d,originals)
        self.assertEqual(result['excluded'][0]['case'],0)
        self.assertNotIn(0,[r['case'] for r in rows])
        self.assertEqual((c,d),snapshot)

    def test_bad_coverage_donor_or_type_is_rejected(self):
        for change in ('coverage','donor','partition','type','intent','boundary'):
            with self.subTest(change=change):
                pairs,c,d,originals=reviewed()
                if change=='coverage': d['reviews'].pop()
                if change=='donor': d['reviews'][0]['donor_case']=999
                if change=='partition': d['reviews'][0]['partition']='reserved_holdout'
                if change=='type': d['reviews'][0]['presented_missing_context']=False
                if change=='intent': d['reviews'][0]['original_intent_preserved']=1
                if change=='boundary': pairs[0]['donor_case']=3
                with self.assertRaises(ValueError): reconcile(pairs,c,d,originals)
