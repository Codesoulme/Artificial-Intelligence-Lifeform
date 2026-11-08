import copy
import hashlib
import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'scripts/build_p99_reviewed_intro_candidate.py'
spec = importlib.util.spec_from_file_location('p99', PATH)
p99 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p99)


def fixture():
    shards, first = [], []
    for i in range(4):
        text = f'文档{i}'
        row = dict(page_id=str(i), group=f'g{i}', text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   text=text, similarity_component='shared' if i < 2 else f'c{i}', source_url='https://example.org', title='title')
        shards.append({'samples': [row]})
        first.append({'samples': [{k: row[k] for k in ('page_id', 'group', 'text_sha256')} | {'decision': '通过'}]})
    cross = [{'samples': [copy.deepcopy(first[i]['samples'][0]) for i in (0, 1)]},
             {'samples': [copy.deepcopy(first[i]['samples'][0]) for i in (2, 3)]}]
    cross[1]['samples'][1]['decision'] = '存疑'
    return shards, first, cross


class Verification(unittest.TestCase):
    def test_both_pass_and_component_partition(self):
        s, f, c = fixture()
        rows = p99.verify(s, f, c, 1)
        self.assertEqual(len(rows), 3)
        self.assertEqual(p99.partition(rows[0][0]['similarity_component']), p99.partition(rows[1][0]['similarity_component']))
        self.assertEqual(p99.partition('same'), p99.partition('same'))

    def test_reject_coverage_and_identity_changes(self):
        for which in ('first', 'cross', 'duplicate', 'text', 'decision'):
            s, f, c = fixture()
            if which == 'first': f[0]['samples'][0]['group'] = 'wrong'
            if which == 'cross': c[0]['samples'].pop()
            if which == 'duplicate': c[1]['samples'].append(copy.deepcopy(c[0]['samples'][0]))
            if which == 'text': s[0]['samples'][0]['text'] += 'x'
            if which == 'decision': f[0]['samples'][0]['decision'] = 'unknown'
            with self.subTest(which=which), self.assertRaises(ValueError):
                p99.verify(s, f, c, 1)

    def test_sha_verification(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.json'
            p.write_bytes(b'{}')
            self.assertEqual(p99.read_verified(p, p99.digest(b'{}')), {})
            with self.assertRaises(ValueError): p99.read_verified(p, p99.digest(b'other'))


if __name__ == '__main__': unittest.main()
