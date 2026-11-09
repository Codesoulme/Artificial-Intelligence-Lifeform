import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_p227_story_candidate as prep


class StoryCandidateTests(unittest.TestCase):
    def test_identity_and_review_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'runs').mkdir()
            queue, source, review = root/'queue', root/'source', root/'review'
            items = [{'sample_id': str(i), 'text': '中文'+str(i)} for i in range(81)]
            queue.write_text(''.join(json.dumps(r)+'\n' for r in items), encoding='utf8')
            source.write_bytes(b'source')
            original = {r['sample_id']: {'text': r['text'], 'normalized_sha256': 'n'+r['sample_id'],
                         'group_sha256': 'g'+r['sample_id'], 'source_original_sha256': 'o'+r['sample_id'],
                         'source_row_1based': int(r['sample_id'])+1, 'content_tokens': 3} for r in items}
            reviews = [{'index': i, 'sample_id': r['sample_id'], 'label': 'clean' if i == 0 else 'minor',
                        'reason': 'reviewed'} for i, r in enumerate(items)]
            review.write_text(json.dumps(reviews), encoding='utf8')
            with patch.object(prep, 'ROOT', root), patch.object(prep, 'SAMPLE', source), \
                 patch.object(prep, 'SAMPLE_SHA', prep.sha(source.read_bytes())), \
                 patch.object(prep, 'QUEUE_SHA', prep.sha(queue.read_bytes())), \
                 patch.object(prep, 'PARTS', ((0, 80),)), \
                 patch.object(prep, 'validated_sample', return_value=original):
                result = prep.build(queue, [review], root/'runs'/'out')
                self.assertEqual(result['counts']['records'], 1)
                saved = json.loads((root/'runs'/'out'/'candidate.jsonl').read_text(encoding='utf8'))
                self.assertEqual(saved['sample_id'], '0')
                reviews[1]['sample_id'] = '0'
                review.write_text(json.dumps(reviews), encoding='utf8')
                with self.assertRaisesRegex(ValueError, 'identity'):
                    prep.build(queue, [review], root/'runs'/'bad')
                self.assertFalse((root/'runs'/'bad').exists())


if __name__ == '__main__':
    unittest.main()
