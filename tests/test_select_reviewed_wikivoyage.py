import json
import tempfile
import unittest
from pathlib import Path

from scripts import select_reviewed_wikivoyage as selection


class ReviewedWikivoyageTests(unittest.TestCase):
    def test_selection_binding_and_incomplete_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source'
            source.mkdir()
            rows = []
            for i in range(2):
                text = f'中文样本{i}。'
                rows.append({'page_id': str(i), 'text': text,
                             'text_sha256': selection.sha(text.encode('utf8')),
                             'characters': len(text), 'content_tokens': 4})
            candidate = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode('utf8')
            sample = json.dumps(rows, ensure_ascii=False).encode('utf8')
            (source/'candidate.jsonl').write_bytes(candidate)
            (source/'review-sample.json').write_bytes(sample)
            (source/'manifest.json').write_text(json.dumps({'candidate_sha256': selection.sha(candidate)}), encoding='utf8')
            review = {'status': 'model_review_not_training_approval', 'candidate_sha256': selection.sha(candidate),
                      'sample_sha256': selection.sha(sample), 'sample_size': 2,
                      'pass_indices': [0], 'uncertain_indices': [1], 'reject_indices': []}
            review_path = root/'review.json'
            review_path.write_text(json.dumps(review), encoding='utf8')
            result = selection.select(source, review_path, root/'out')
            self.assertEqual(result['paragraphs'], 1)
            self.assertEqual(json.loads((root/'out'/'candidate.jsonl').read_text(encoding='utf8'))['page_id'], '0')
            review['uncertain_indices'] = []
            review_path.write_text(json.dumps(review), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'cover'):
                selection.select(source, review_path, root/'other')
            with self.assertRaises(FileExistsError):
                selection.select(source, review_path, root/'out')


if __name__ == '__main__':
    unittest.main()
