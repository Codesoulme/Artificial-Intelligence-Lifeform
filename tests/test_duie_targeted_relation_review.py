import unittest
from scripts.duie_targeted_relation_candidates_20261117 import candidates
from scripts.duie_targeted_relation_review_20261117 import compare


class TargetedRelationReviewTests(unittest.TestCase):
    def test_candidate_pair_and_review_manifest(self):
        rows = [{'text': '小林出生地是上海，小王出生地是北京。', 'spo_list': [
            {'subject': '小林', 'subject_type': '人物', 'predicate': '出生地', 'object': '上海'},
            {'subject': '小王', 'subject_type': '人物', 'predicate': '出生地', 'object': '北京'}]}]
        selected = candidates(rows)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(selected[0]['pairs']), 2)
        self.assertEqual(compare(selected, {selected[0]['digest']: 'accepted'}), {'accepted': 1})
        with self.assertRaises(ValueError):
            compare(selected, {})

    def test_same_subject_does_not_make_a_pair(self):
        row = {'text': '小林出生地是上海，也可能是北京。', 'spo_list': [
            {'subject': '小林', 'subject_type': '人物', 'predicate': '出生地', 'object': '上海'},
            {'subject': '小林', 'subject_type': '人物', 'predicate': '出生地', 'object': '北京'}]}
        self.assertEqual(candidates([row]), [])
