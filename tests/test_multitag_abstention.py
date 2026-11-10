import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.deepseek_multitag_abstention import sample_multitag
from scripts.deepseek_relative_identity_probe import UNKNOWN


class MultiTagSamplingTests(unittest.TestCase):
    def make_archive(self, directory, dialogues):
        path = Path(directory) / 'fixture.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('DuLeMon/both/train.json', '\n'.join(
                json.dumps({'conversation': turns}, ensure_ascii=False) for turns in dialogues))
        return path

    def test_selects_only_both_owners_and_respects_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_archive(directory, [
                ['Bot:你好啊', 'Usr:我也喜欢读书\tU1,B1'],
                ['Bot:这是前句', 'Usr:我们喜欢打球\tU2,B2'],
                ['Bot:你喜欢什么', 'Usr:我喜欢旅游\tU1,U2'],
                ['Bot:另一个前句', 'Usr:我们都是学生\tU1,B1'],
            ])
            rows = sample_multitag(path, {'这是前句'}, {3}, count=1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['text'], '我也喜欢读书')
            self.assertEqual(rows[0]['label'], UNKNOWN)

    def test_rejects_malformed_tags_and_shortage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_archive(directory, [
                ['Bot:你好啊', 'Usr:我也喜欢读书\tU1,B1,bad'],
            ])
            with self.assertRaisesRegex(ValueError, 'Insufficient'):
                sample_multitag(path, set(), set(), count=1)

    def test_rejects_invalid_count(self):
        with self.assertRaises(ValueError):
            sample_multitag('unused', set(), set(), count=True)


if __name__ == '__main__':
    unittest.main()
