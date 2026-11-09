import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import filter_p223_overlap as filt


class FilterOverlapTests(unittest.TestCase):
    def test_excludes_matches_and_rejects_missing_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source'
            source.mkdir()
            rows = [{'source_original_sha256': str(i), 'source_row_1based': i+1,
                     'origin_source': 'wiki', 'content_tokens': 5} for i in range(2)]
            raw = ''.join(json.dumps(r)+'\n' for r in rows).encode('utf8')
            (source/'candidate.jsonl').write_bytes(raw)
            (source/'manifest.json').write_text(json.dumps({'candidate_sha256': filt.sha(raw),
                                                           'tokenizer_sha256': 'vocab'}), encoding='utf8')
            matches = lambda n: {'documents_with_full_containment': 0,
                                 'windows': {'20': {'candidate_window_positions': n}}}
            audit = {'status': 'training_side_overlap_diagnostic', 'candidate_sha256': filt.sha(raw),
                     'comparisons': {'p28': {'sha256': 'p28', 'by_candidate': {'0': matches(0), '1': matches(0)}},
                                     'v5': {'sha256': 'v5', 'by_candidate': {'0': matches(0), '1': matches(1)}}}}
            audit_path = root/'audit.json'
            audit_path.write_text(json.dumps(audit), encoding='utf8')
            with patch.object(filt, 'EXPECTED', {'p28': 'p28', 'v5': 'v5'}):
                result = filt.build(source, audit_path, root/'out')
                self.assertEqual(result['counts']['records'], 1)
                self.assertEqual(result['counts']['removed'], 1)
                self.assertEqual(json.loads((root/'out'/'candidate.jsonl').read_text())['source_row_1based'], 1)
                del audit['comparisons']['v5']['by_candidate']['0']
                audit_path.write_text(json.dumps(audit), encoding='utf8')
                with self.assertRaisesRegex(ValueError, 'cover'):
                    filt.build(source, audit_path, root/'other')


if __name__ == '__main__':
    unittest.main()
