import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from spora.dialogue_candidate import validate_candidate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tokenizer_sha = 'a' * 64
        self.rows = {
            'context.jsonl': [{'pair_id': 'p', 'group': 'g', 'tokens': [1, 3, 4, 2], 'mask': [False, False, True, True]}],
            'last-user.jsonl': [{'pair_id': 'p', 'group': 'g', 'tokens': [1, 4, 2], 'mask': [False, True, True]}],
            'lineage.jsonl': [{'pair_id': 'p', 'group': 'g', 'target_tokens': 2}],
        }

    def write_fixture(self):
        for name, rows in self.rows.items():
            (self.root / name).write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        manifest = {'status': 'reviewed_candidate_not_approved_for_training', 'format': 'matched-dialogue-v1',
                    'tokenizer_sha256': self.tokenizer_sha, 'counts': {'pairs': 1, 'groups': 1, 'target_tokens': 2},
                    'files': {name: digest(self.root / name) for name in self.rows}}
        (self.root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return digest(self.root / 'manifest.json')

    def test_valid_candidate_and_digest(self):
        sha = self.write_fixture()
        self.assertEqual(validate_candidate(self.root, sha, self.tokenizer_sha, 8, 8)['target_tokens'], 2)
        with self.assertRaises(ValueError):
            validate_candidate(self.root, '0' * 64, self.tokenizer_sha, 8, 8)

    def test_rejects_target_and_mask_tampering_even_when_rehashed(self):
        self.rows['last-user.jsonl'][0]['tokens'][1] = 5
        sha = self.write_fixture()
        with self.assertRaises(ValueError):
            validate_candidate(self.root, sha, self.tokenizer_sha, 8, 8)
        self.rows['last-user.jsonl'][0]['tokens'][1] = 4
        self.rows['context.jsonl'][0]['mask'] = [False, True, False, True]
        sha = self.write_fixture()
        with self.assertRaises(ValueError):
            validate_candidate(self.root, sha, self.tokenizer_sha, 8, 8)

    def test_rejects_oversized_and_changed_tokenizer(self):
        sha = self.write_fixture()
        with self.assertRaises(ValueError):
            validate_candidate(self.root, sha, self.tokenizer_sha, 2, 8)
        with self.assertRaises(ValueError):
            validate_candidate(self.root, sha, 'b' * 64, 8, 8)


if __name__ == '__main__':
    unittest.main()
