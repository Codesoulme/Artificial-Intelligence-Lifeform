import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import sample_p237_dialogues as p


def row(family, dialogue=None):
    return {
        'status': p.STATUS, 'family_sha256': f'{family:064x}',
        'dialogue_sha256': f'{(family if dialogue is None else dialogue):064x}',
        'messages': [{'role': role, 'content': str(family)}
                     for role in ('user', 'assistant', 'user', 'assistant')],
        'tokens': [1, 2, 3], 'mask': [0, 1, 1],
        'token_count': 3, 'supervised_tokens': 2,
    }


def jsonl(rows):
    return b''.join((json.dumps(r) + '\n').encode() for r in rows)


class SampleTests(unittest.TestCase):
    def test_selection_is_stable_excludes_and_preserves_dialogue(self):
        rows = [row(i) for i in range(50)] + [row(1, 100)]
        excluded = {row(2)['family_sha256']}
        selected, eligible = p.select(rows, excluded)
        self.assertEqual((selected, eligible), p.select(list(reversed(rows)), excluded))
        self.assertEqual(eligible, 49)
        self.assertEqual(len({r['family_sha256'] for r in selected}), 40)
        self.assertTrue(excluded.isdisjoint(r['family_sha256'] for r in selected))
        self.assertTrue(all(len(r['messages']) == 4 for r in selected))

    def test_invalid_candidates_fail_even_when_excluded(self):
        valid = row(0)
        valid['mask'] = [False, True, True]
        p.validate_row(valid)
        for change in ({'status': 'approved'}, {'mask': [1]}, {'token_count': 2},
                       {'supervised_tokens': 1}, {'mask': [0, 0, 0]},
                       {'mask': [0, 2, 0]}, {'messages': []}):
            bad = row(0)
            bad.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                p.select([bad], {bad['family_sha256']}, count=0)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            p.select([row(0), row(0)], set(), count=0)
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            p.select([row(0)], set(), count=2)

    def test_build_pins_inputs_and_creates_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = [row(i) for i in range(100)]
            data = {'candidate': jsonl(candidates), 'p231': jsonl(candidates[:40]),
                    'p235': jsonl(candidates[40:60]), 'p236': jsonl(candidates[60:80])}
            manifest = {'status': p.STATUS,
                        'output_sha256': {'candidates.jsonl': p.digest(data['candidate']),
                                          'review-40.jsonl': p.digest(data['p231'])},
                        'counts': {'retained_dialogues': 100, 'retained_families': 100}}
            data['manifest'] = json.dumps(manifest).encode()
            paths = {name: root / name for name in data}
            for name, path in paths.items():
                path.write_bytes(data[name])
            pins = {name: p.digest(value) for name, value in data.items()}
            output = root / 'output'
            with patch.object(p, 'PATHS', paths), patch.object(p, 'PINS', pins), patch.object(p, 'OUTPUT', output):
                # Leave enough eligible families for the full sample.
                prior = [row(200 + i) for i in range(20)]
                paths['p236'].write_bytes(jsonl(prior))
                with self.assertRaisesRegex(ValueError, 'p236 digest mismatch'):
                    p.build()
                self.assertFalse(output.exists())
                pins['p236'] = p.digest(paths['p236'].read_bytes())
                result = p.build()
                payload = (output / 'sample.jsonl').read_bytes()
                sample = [json.loads(line) for line in payload.splitlines()]
                self.assertEqual(len(sample), 40)
                self.assertEqual(result['sample_sha256'], p.digest(payload))
                self.assertEqual(result, json.loads((output / 'manifest.json').read_bytes()))
                self.assertTrue(all(r['messages'] == candidates[int(r['messages'][0]['content'])]['messages'] for r in sample))
                self.assertTrue({r['family_sha256'] for r in sample}.isdisjoint(
                    {r['family_sha256'] for r in candidates[:60]} | {r['family_sha256'] for r in prior}))
                with self.assertRaises(FileExistsError):
                    p.build()
                self.assertEqual(payload, (output / 'sample.jsonl').read_bytes())


if __name__ == '__main__':
    unittest.main()
