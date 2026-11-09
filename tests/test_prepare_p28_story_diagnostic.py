import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_p28_story_diagnostic import prepare, sha


class StoryCandidateTests(unittest.TestCase):
    def test_candidate_is_reproducible_and_unapproved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ('train', 'development'):
                rows = []
                for index in range(4):
                    text = f'{split} story {index}'
                    rows.append({'split': split, 'text': text, 'source_row_index': index,
                                 'group': split + str(index),
                                 'source_original_sha256': hashlib.sha256(text.encode()).hexdigest()})
                (root / split).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
            with patch('scripts.prepare_p28_story_diagnostic.TRAIN_SHA', sha(root / 'train')), \
                 patch('scripts.prepare_p28_story_diagnostic.DEV_SHA', sha(root / 'development')):
                first = prepare(root / 'train', root / 'development', root / 'a', 2, 2)
                second = prepare(root / 'train', root / 'development', root / 'b', 2, 2)
            self.assertEqual(first['manifest_sha256'], second['manifest_sha256'])
            self.assertEqual(first['status'], 'candidate_not_approved_for_training')
            self.assertFalse(first['training_started'])


if __name__ == '__main__':
    unittest.main()
