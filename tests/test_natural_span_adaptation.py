import json
import unittest
from unittest.mock import patch

from scripts import deepseek_spora_natural_span_adaptation as audit


class NaturalSpanAdaptationTests(unittest.TestCase):
    def test_more_multiple_changes_only_group_sampling_proportions(self):
        baseline = audit.training_schedule(False)
        weighted = audit.training_schedule(True)
        self.assertEqual(baseline, ('weak_single', 'weak_multiple', 'synthetic_none'))
        self.assertEqual(weighted, baseline + ('weak_multiple',))

    def test_training_uses_disjoint_conversations_and_names(self):
        reviewed = [{'doc': 'held', 'names': ['欧阳'], 'subject': '欧阳', 'text': '欧阳喜欢什么？'}]
        candidates = [{'doc': 'one', 'name': '李某', 'text': '李某喜欢什么？'},
                      {'doc': 'two', 'name': '张某', 'text': '张某喜欢什么？'}]
        multi = {'conversation': ['王明和李白共同拍了什么电影？'],
                 'knowledge': [['王明', '性别', '男'], ['李白', '性别', '男']]}
        lines = [json.dumps(multi, ensure_ascii=False).encode('utf-8')]
        with patch.object(audit, 'SINGLE_COUNT', 2), patch.object(audit, 'MULTI_COUNT', 1), patch.object(
                audit.natural, 'select_candidates', return_value=candidates):
            rows = audit.build_training(lines, reviewed)
        self.assertEqual(sum(row['group'] == 'weak_single' for row in rows), 2)
        self.assertEqual(sum(row['group'] == 'weak_multiple' for row in rows), 1)
        self.assertTrue(all(row['doc'] != 'held' for row in rows))
        self.assertTrue(all('欧阳' not in row['question'] for row in rows))


if __name__ == '__main__':
    unittest.main()
