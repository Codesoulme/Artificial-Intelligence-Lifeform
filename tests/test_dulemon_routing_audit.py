import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.deepseek_dulemon_routing_audit import sample_archive


class DuLeMonAuditTests(unittest.TestCase):
    def test_selection_is_deterministic_and_excludes_drink_topics(self):
        records = [
            {'conversation': ['Usr: 今天 天气 真好', 'Usr: 我 喜欢 喝茶', 'Bot: 你好']},
            {'conversation': ['Usr: 最近 工作 很忙', 'Usr: 今天 天气 真好']},
            {'conversation': ['Usr: 我 今天 去 散步']},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('DuLeMon/both/test.json', '\n'.join(json.dumps(r, ensure_ascii=False) for r in records))
            with patch('scripts.deepseek_dulemon_routing_audit.COUNT', 2):
                first, second = sample_archive(path), sample_archive(path)
            self.assertEqual(first, second)
            self.assertEqual(len({row['session'] for row in first}), 2)
            self.assertEqual(len({row['text'] for row in first}), 2)
            self.assertTrue(all('喝' not in row['text'] for row in first))

    def test_topic_mode_retains_distinct_turns_from_same_session(self):
        record = {'conversation': ['Usr: 我 喜欢 喝茶', 'Usr: 你 喜欢 喝 什么 饮料']}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('DuLeMon/both/test.json', json.dumps(record, ensure_ascii=False))
            selected = sample_archive(path, topic=True)
            self.assertEqual(len(selected), 2)
            self.assertEqual(len({row['session'] for row in selected}), 1)


if __name__ == '__main__':
    unittest.main()
