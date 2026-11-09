import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_storybooks_hybrid import reviewed_documents


class StorybooksEvaluationTests(unittest.TestCase):
    def test_reviewed_selection_checks_source_and_excludes_attribution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'raw-v1').mkdir()
            items, reviews = [], []
            for index in range(29):
                story = f'{index:04d}'
                raw = f'# 标题\n\n##\n正文{index}\n\n##\n* License: test\n'.encode('utf8')
                digest = hashlib.sha256(raw).hexdigest()
                (root / 'raw-v1' / (story + '.md')).write_bytes(raw)
                items.append({'story': story, 'status': 'matched_pending_quality_and_rights_review',
                              'source_sha256': digest, 'paragraphs': 1,
                              'text_matches_page': True, 'allowed_page_license': True})
                reviews.append({'story': story, 'source_sha256': digest,
                                'decision': '通过' if index < 20 else '存疑'})
            (root / 'page-text-verification-v2.json').write_text(json.dumps({'stories': items}), encoding='utf8')
            (root / 'quality-review-a.json').write_text(json.dumps({'stories': reviews[:15]}), encoding='utf8')
            (root / 'quality-review-b.json').write_text(json.dumps({'reviews': reviews[15:]}), encoding='utf8')
            docs, _ = reviewed_documents(root)
            self.assertEqual(len(docs), 20)
            self.assertEqual(docs[0]['text'], '正文0')
            (root / 'raw-v1' / '0000.md').write_text('changed', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'verification mismatch'):
                reviewed_documents(root)


if __name__ == '__main__':
    unittest.main()
