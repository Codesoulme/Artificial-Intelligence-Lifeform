import copy
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p100', ROOT / 'scripts/build_p100_reviewed_intro_candidate.py')
p100 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p100)


def fixture():
    shards, first, cross = [], [], []
    for i in range(4):
        text = f'document {i}'
        row = dict(page_id=str(i), group=f'g{i}', text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   text=text, similarity_component=f'c{i}', source_url='https://example.org', title='title')
        grade = {k: row[k] for k in ('page_id', 'group', 'text_sha256')} | {'decision': '通过'}
        shards.append({'samples': [row]})
        first.append({('结果' if i == 2 else 'samples'): [copy.deepcopy(grade)]})
        cross.append({'samples': [copy.deepcopy(grade)]})
    cross[3]['samples'][0]['decision'] = '存疑'
    return shards, first, cross


class CandidateTests(unittest.TestCase):
    def test_selection_and_shared_partition_rule(self):
        s, f, c = fixture()
        self.assertEqual(len(p100.verify(s, f, c, 1)), 3)
        spec2 = importlib.util.spec_from_file_location('p99', ROOT / 'scripts/build_p99_reviewed_intro_candidate.py')
        p99 = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(p99)
        self.assertEqual(p100.SALT, p99.SALT)
        for component in ('same', 'another', 'c0', 'c1'):
            self.assertEqual(p100.partition(component), p99.partition(component))

    def test_corruption_and_coverage(self):
        for defect in ('first', 'cross', 'duplicate', 'text', 'decision', 'page', 'component'):
            s, f, c = fixture()
            if defect == 'first': f[0]['samples'][0]['group'] = 'changed'
            if defect == 'cross': c[0]['samples'].clear()
            if defect == 'duplicate': c[1]['samples'].append(copy.deepcopy(c[0]['samples'][0]))
            if defect == 'text': s[0]['samples'][0]['text'] += 'x'
            if defect == 'decision': f[0]['samples'][0]['decision'] = 'unknown'
            if defect == 'page':
                s[1]['samples'][0]['page_id'] = s[0]['samples'][0]['page_id']
                f[1]['samples'][0]['page_id'] = s[0]['samples'][0]['page_id']
                c[1]['samples'][0]['page_id'] = s[0]['samples'][0]['page_id']
            if defect == 'component': s[1]['samples'][0]['similarity_component'] = 'c0'
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                p100.verify(s, f, c, 1)

    def test_isolation(self):
        s, f, c = fixture()
        selected = p100.verify(s, f, c, 1)
        p100.verify_isolation(selected, [{'page_id': 'unrelated', 'similarity_component': 'other'}])
        for row in ({'page_id': '0', 'similarity_component': 'other'},
                    {'page_id': 'other', 'similarity_component': 'c0'}):
            with self.assertRaises(ValueError): p100.verify_isolation(selected, [row])

    def test_hash_and_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'
            path.write_bytes(b'{}')
            self.assertEqual(p100.read_verified(path, p100.digest(b'{}')), {})
            with self.assertRaises(ValueError): p100.read_verified(path, p100.digest(b'changed'))
        with self.assertRaises(ValueError): p100.build(ROOT / 'runs/wikimedia-reviewed-intro-p99-v1')
        with self.assertRaises(ValueError): p100.build(ROOT / 'reports/not-allowed')


if __name__ == '__main__':
    unittest.main()
