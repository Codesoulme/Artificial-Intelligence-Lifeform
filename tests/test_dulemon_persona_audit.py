import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.deepseek_dulemon_persona_audit import normalize_turn, sample_archive, summarize
from scripts.deepseek_relative_identity_probe import ADDRESSEE, SPEAKER, UNKNOWN


class DuLeMonPersonaAuditTests(unittest.TestCase):
    def test_turn_parser_keeps_roles_and_single_annotation(self):
        self.assertEqual(normalize_turn('Bot: 你 好\tU1'), ('Bot', '你好', 'U1'))
        self.assertEqual(normalize_turn('Usr: 你 好'), ('Usr', '你好', ''))
        self.assertIsNone(normalize_turn('Other: 你好'))

    def test_persona_owner_is_relative_to_current_role(self):
        sessions = [
            {'conversation': ['Bot: 你 好', 'Usr: 我 喜欢 跳舞\tU1']},
            {'conversation': ['Usr: 你 好', 'Bot: 我 喜欢 跳舞\tB1']},
            {'conversation': ['Usr: 你 好', 'Bot: 你 喜欢 跳舞\tU1']},
            {'conversation': ['Bot: 你 好', 'Usr: 你 喜欢 跳舞\tB1']},
            {'conversation': ['Bot: 你 好', 'Usr: 我们 喜欢 跳舞\tU1,B1']},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('DuLeMon/both/test.json', '\n'.join(json.dumps(x, ensure_ascii=False) for x in sessions))
            rows = sample_archive(path, per_class=2)
            self.assertEqual(len(rows), 4)
            self.assertEqual({r['session'] for r in rows}, {0, 1, 2, 3})
            self.assertEqual({r['session']: r['label'] for r in rows},
                             {0: SPEAKER, 1: SPEAKER, 2: ADDRESSEE, 3: ADDRESSEE})
            self.assertEqual(rows, sample_archive(path, per_class=2))

    def test_weak_metrics_do_not_count_abstention_as_cross_person(self):
        metrics = summarize([{'label': SPEAKER, 'predicted': UNKNOWN},
                             {'label': SPEAKER, 'predicted': ADDRESSEE},
                             {'label': ADDRESSEE, 'predicted': ADDRESSEE}])
        self.assertEqual(metrics, {'examples': 3, 'weak_label_match': 1, 'abstained': 1, 'cross_person': 1})


if __name__ == '__main__':
    unittest.main()
