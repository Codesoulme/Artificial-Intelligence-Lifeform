import json
from pathlib import Path
import tempfile
import unittest

from scripts import refine_p230_wikipedia_body as refine


class RefineTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'source'
        source.mkdir()
        rows = []
        for n in range(45):
            text = '这是一段完整的内容。' if n != 40 else '霍夫蟹（；Kiwa tyleri）是一个物种。'
            rows.append({'status': refine.STATUS, 'page_id': str(n), 'text': text,
                         'text_sha256': refine.digest(text.encode()), 'char_start': 7,
                         'char_end': 7 + len(text), 'bpe_tokens': 10,
                         'title': 'title', 'url': 'https://example.org',
                         'paragraphs': [{'char_start': 7, 'char_end': 7 + len(text)}]})
        raw = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
        excluded = b''
        (source / 'candidates.jsonl').write_bytes(raw)
        (source / 'excluded.jsonl').write_bytes(excluded)
        manifest = {'status': refine.STATUS, 'files': {'candidates.jsonl': refine.digest(raw),
                    'excluded.jsonl': refine.digest(excluded)}, 'counts': {'passages': 45, 'bpe_tokens': 450}}
        manifest_raw = json.dumps(manifest).encode()
        (source / 'manifest.json').write_bytes(manifest_raw)
        review = root / 'review.json'
        review.write_text(json.dumps([{'index': n, 'page_id': str(n), 'text_sha256': rows[n]['text_sha256']}
                                     for n in range(40)]), encoding='utf8')
        return source, review, refine.digest(manifest_raw), rows

    def test_signals_and_narrow_negatives(self):
        cases = {'霍夫蟹（；Kiwa tyleri）': 'empty_or_damaged_parentheses',
                 '系統（，縮寫為 ext）': 'empty_or_damaged_parentheses',
                 '人物（某人，）': 'empty_or_damaged_parentheses',
                 '花園（或）': 'empty_or_damaged_parentheses',
                 '龜（英語：，學名：）': 'empty_name_label',
                 '是一所法学院。': 'missing_opening_subject',
                 '2008年，學者將此本與另一版本對照。': 'isolated_reference',
                 '大約位於。': 'missing_location', '由先生於1980年發展。': 'missing_named_person',
                 '某兩種行為之間有關連。': 'unspecified_behaviors', '句子。。': 'repeated_sentence_stop'}
        for text, rule in cases.items():
            with self.subTest(text=text):
                self.assertIn(rule, refine.reasons(text))
        for text in ('这是一个完整句子。', '本书（英文：Example）是一部作品。', '這所學校是一所大學。'):
            self.assertEqual(refine.reasons(text), [])

    def test_build_reproducible_and_preserves_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, review, expected, rows = self.fixture(root)
            first = refine.build(source, [review], root / 'out', expected, count=4, seed='seed')
            second = refine.build(source, [review], root / 'out2', expected, count=4, seed='seed')
            self.assertEqual(first, second)
            self.assertEqual(first['counts']['quarantined_passages'], 1)
            self.assertEqual(first['counts']['rule_matches']['empty_or_damaged_parentheses'], 1)
            quarantine = json.loads((root / 'out/quarantined.jsonl').read_text(encoding='utf8'))
            self.assertEqual(quarantine['candidate'], rows[40])
            sample = json.loads((root / 'out/sample.json').read_text(encoding='utf8'))
            self.assertEqual({r['page_id'] for r in sample}, {'41', '42', '43', '44'})
            with self.assertRaises(FileExistsError):
                refine.build(source, [review], root / 'out', expected, count=4)

    def test_integrity_and_review_linkage_fail_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, review, expected, _ = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'manifest digest'):
                refine.build(source, [review], root / 'bad1', '0' * 64, count=1)
            (source / 'excluded.jsonl').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'file digest'):
                refine.build(source, [review], root / 'bad2', expected, count=1)
            (source / 'excluded.jsonl').write_bytes(b'')
            records = json.loads(review.read_text(encoding='utf8'))
            records[0]['text_sha256'] = '0' * 64
            review.write_text(json.dumps(records), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'review linkage'):
                refine.build(source, [review], root / 'bad3', expected, count=1)
            self.assertFalse(any((root / n).exists() for n in ('bad1', 'bad2', 'bad3')))


if __name__ == '__main__':
    unittest.main()
