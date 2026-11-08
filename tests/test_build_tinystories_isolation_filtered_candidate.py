import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_tinystories_isolation_filtered_candidate import build


class IsolationFilteredCandidateTests(unittest.TestCase):
    def test_filters_only_audited_overlap_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / 'candidate'
            candidate.mkdir()
            rows = [
                {'record_id': 'a', 'text': '甲'},
                {'record_id': 'b', 'text': '乙'},
            ]
            candidate_file = candidate / 'candidate.jsonl'
            candidate_file.write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in rows), encoding='utf8')
            import hashlib
            candidate_sha = hashlib.sha256(candidate_file.read_bytes()).hexdigest()
            manifest = {
                'status': 'candidate_not_approved_for_training', 'training_approval': False,
                'training_started': False, 'candidate_sha256': candidate_sha, 'records': 2,
                'repository': 'r', 'revision': 'v', 'config': 'c', 'split': 'train',
                'source_shard_sha256': 's', 'tokenizer_sha256': 't',
                'license_declared': 'l', 'attribution_required': [],
            }
            (candidate / 'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
            audit = root / 'audit.json'
            audit.write_text(json.dumps({
                'status': 'candidate_not_approved_for_training', 'training_approval': False,
                'candidate_sha256': candidate_sha,
                'prior_overlap': {'by_candidate': {'a': {'exact_full': [{'split': 'train', 'line': 1}], 'leading_96_contained': []}, 'b': {'exact_full': [], 'leading_96_contained': []}}},
            }), encoding='utf8')
            result = build(candidate, audit, root / 'out')
            self.assertEqual(result['records'], 1)
            self.assertEqual(result['excluded_prior_story_overlap_records'], 1)
            self.assertEqual(json.loads((root / 'out/candidate.jsonl').read_text(encoding='utf8'))['record_id'], 'b')
            self.assertFalse(result['training_approval'])


if __name__ == '__main__':
    unittest.main()
