import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import deepseek_spora_diverse_kl_correction_audit as audit


class DiverseKLPromptSplitTests(unittest.TestCase):
    def test_reviewed_groups_are_split_without_overlap(self):
        records, reviews = [], []
        for index in range(18):
            group = f'group-{index}'
            marker = f'path-{index}'
            records.append({'group': group, 'sha256': marker,
                            'messages': [{'role': 'user', 'content': f'Prompt {index}'}]})
            reviews.append({'line': index + 1, 'group': group,
                            'path_sha256': marker, 'decision': 'pass'})
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'dialogue.jsonl'
            review = Path(directory) / 'review.json'
            data.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
            review.write_text(json.dumps({'row_reviews': reviews}), encoding='utf-8')
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            with patch.object(audit, 'DATA', data), patch.object(audit, 'REVIEW', review), \
                 patch.object(audit, 'EXPECTED_DATA_SHA', digest):
                train, held = audit.load_reviewed_prompts()
                self.assertEqual((len(train), len(held)), (8, 8))
                self.assertFalse({item['group'] for item in train} & {item['group'] for item in held})
                self.assertEqual((train, held), audit.load_reviewed_prompts())
                reviews[0]['path_sha256'] = 'mismatched'
                review.write_text(json.dumps({'row_reviews': reviews}), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Review does not match'):
                    audit.load_reviewed_prompts()


if __name__ == '__main__':
    unittest.main()
