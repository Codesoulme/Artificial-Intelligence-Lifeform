import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_meow_training_overlap import audit


class ProseOverlapTests(unittest.TestCase):
    def test_training_comparison_and_digest_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text = '春天来了我们一起去公园散步看花然后高高兴兴回家'
            (root / 'prose.txt').write_text(text, encoding='utf8')
            digest = hashlib.sha256(text.encode('utf8')).hexdigest()
            (root / 'manifest.json').write_text(json.dumps({'candidate_sha256': digest}), encoding='utf8')
            source = root / 'train.jsonl'
            source.write_text(json.dumps({'text': text}, ensure_ascii=False) + '\n', encoding='utf8')
            result = audit(root, {'fixture': source})
            self.assertEqual(result['comparisons']['fixture']['records'], 1)
            self.assertEqual(result['comparisons']['fixture']['matches']['documents_equal_full'], 1)
            with self.assertRaisesRegex(ValueError, 'training sources'):
                audit(root, {'fixture': root / 'test.jsonl'})
            (root / 'prose.txt').write_text('changed', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                audit(root, {'fixture': source})


if __name__ == '__main__':
    unittest.main()
