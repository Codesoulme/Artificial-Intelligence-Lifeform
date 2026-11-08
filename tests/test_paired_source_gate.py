import json
import tempfile
import unittest
from pathlib import Path

from spora.paired_source_gate import DATA, verify_p91_inputs
from spora.source_review import file_sha256


class SourceGateTests(unittest.TestCase):
    def check_plan(self, plan):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/'plan.json'; approval = root/'approval.json'
            path.write_text(json.dumps(plan), encoding='utf8')
            approval.write_text(json.dumps({'approved_for_pilot': True, 'plan_sha256': file_sha256(path),
                'architectures':['memory_bridge','standard_causal'], 'initialization':'paired_random',
                'seeds':plan['seeds']}), encoding='utf8')
            with self.assertRaises(ValueError):
                verify_p91_inputs(root, path, approval)

    def test_never_extend_p54_with_new_review(self):
        self.check_plan({'seeds':[1,2], 'schedule_steps':101, 'data':DATA,
                         'target_shares':{'story':1.0}, 'focus_every':0})

    def test_focus_and_language_ancestor_rejected(self):
        for extra in ({'focus_every':1}, {'initial_language':'weights.pt'}):
            plan = {'seeds':[1,2], 'schedule_steps':100, 'data':DATA,
                    'target_shares':{'story':1.0}, 'focus_every':0}
            plan.update(extra); self.check_plan(plan)

    def test_review_must_bind_exact_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); p = root/'plan.json'; a = root/'approval.json'
            p.write_text('{}', encoding='utf8'); a.write_text('{"approved_for_pilot": true}', encoding='utf8')
            with self.assertRaises(ValueError):
                verify_p91_inputs(root, p, a)


if __name__ == '__main__':
    unittest.main()
