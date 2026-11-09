import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import audit_authored_text_overlap as audit
from scripts import audit_reviewed_wikivoyage_overlap as wiki_audit


class AuthoredOverlapTests(unittest.TestCase):
    def test_reviewed_wikivoyage_binds_each_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root/'candidate'
            candidate.mkdir()
            text = '这是一段足够长的中文文本用于核对训练材料。'
            raw = (json.dumps({'text': text, 'text_sha256': hashlib.sha256(text.encode('utf8')).hexdigest()},
                              ensure_ascii=False)+'\n').encode('utf8')
            (candidate/'candidate.jsonl').write_bytes(raw)
            (candidate/'manifest.json').write_text(json.dumps({'output_sha256': hashlib.sha256(raw).hexdigest(),
                                                               'paragraphs': 1}), encoding='utf8')
            first, second = root/'p28', root/'v5'
            first.write_text(json.dumps({'text': text}, ensure_ascii=False)+'\n', encoding='utf8')
            second.write_text(json.dumps({'text': '另一段文字'}, ensure_ascii=False)+'\n', encoding='utf8')
            expected = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for name, path in (('p28', first), ('v5', second))}
            with patch.object(audit, 'EXPECTED', expected):
                result = wiki_audit.audit(candidate, {'p28': first, 'v5': second})
            match = next(iter(result['comparisons']['p28']['by_candidate'].values()))
            self.assertEqual(match['documents_equal_full'], 1)

    def test_exact_overlap_and_digest_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root/'candidate'
            candidate.mkdir()
            text = '这是需要认真检查的完整中文文本内容。'
            raw = (json.dumps({'id': 'one', 'work_group': 'meow-v2-one-work', 'text': text},
                              ensure_ascii=False)+'\n').encode('utf8')
            (candidate/'candidate.jsonl').write_bytes(raw)
            (candidate/'manifest.json').write_text(json.dumps({'output_sha256': hashlib.sha256(raw).hexdigest(),
                                                               'documents': 1}), encoding='utf8')
            first, second = root/'p28', root/'v5'
            first.write_text(json.dumps({'text': text}, ensure_ascii=False)+'\n', encoding='utf8')
            second.write_text(json.dumps({'text': '完全不相同'}, ensure_ascii=False)+'\n', encoding='utf8')
            expected = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for name, path in (('p28', first), ('v5', second))}
            with patch.object(audit, 'EXPECTED', expected):
                result = audit.audit(candidate, {'p28': first, 'v5': second})
                self.assertEqual(result['comparisons']['p28']['by_candidate']['one']['documents_equal_full'], 1)
                self.assertEqual(result['comparisons']['v5']['by_candidate']['one']['documents_equal_full'], 0)
                second.write_text('{}\n', encoding='utf8')
                with self.assertRaises(ValueError):
                    audit.audit(candidate, {'p28': first, 'v5': second})


if __name__ == '__main__':
    unittest.main()
