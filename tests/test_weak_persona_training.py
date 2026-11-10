import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.deepseek_relative_identity_probe import ADDRESSEE, SPEAKER
from scripts.deepseek_weak_persona_training import sample_training


class WeakPersonaSamplingTests(unittest.TestCase):
    def make_archive(self, directory, dialogues):
        path = Path(directory) / 'fixture.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('DuLeMon/both/train.json', '\n'.join(
                json.dumps({'conversation': turns}, ensure_ascii=False) for turns in dialogues))
        return path

    def test_balances_weak_labels_and_excludes_audit_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_archive(directory, [
                ['Bot:你好呀', 'Usr:我喜欢读书\tU1'],
                ['Bot:你好呀', 'Usr:你喜欢读书\tB1'],
                ['Bot:排除前句', 'Usr:我喜欢游泳\tU2'],
            ])
            rows = sample_training(path, {'排除前句'}, per_class=1)
            self.assertEqual({row['label'] for row in rows}, {SPEAKER, ADDRESSEE})
            self.assertEqual(len({row['session'] for row in rows}), 2)
            self.assertNotIn('我喜欢游泳', {row['text'] for row in rows})

    def test_rejects_insufficient_distinct_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_archive(directory, [
                ['Bot:你好呀', 'Usr:我喜欢读书\tU1', 'Bot:你喜欢读书\tU1'],
            ])
            with self.assertRaisesRegex(ValueError, 'Insufficient'):
                sample_training(path, set(), per_class=1)

    def test_rejects_invalid_count(self):
        with self.assertRaises(ValueError):
            sample_training('unused', set(), per_class=True)


if __name__ == '__main__':
    unittest.main()
