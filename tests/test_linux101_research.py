import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('linux101_research', Path(__file__).resolve().parents[1] / 'scripts' / 'linux101_research.py')
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


def tree_fixture():
    files = {'docs/a.md': '正文'.encode(), 'README.md': b'readme', 'LICENSE': b'license'}
    blob = {path: research.digest_blob(data) for path, data in files.items()}
    def tree(items):
        body = b''.join(mode.encode() + b' ' + name.encode() + b'\0' + bytes.fromhex(sha) for name, mode, sha in items)
        return hashlib.sha1(b'tree ' + str(len(body)).encode() + b'\0' + body).hexdigest()
    docs = tree([('a.md', '100644', blob['docs/a.md'])])
    root = tree([('LICENSE', '100644', blob['LICENSE']), ('README.md', '100644', blob['README.md']), ('docs', '40000', docs)])
    entries = [{'path': path, 'type': 'blob', 'mode': '100644', 'sha': sha, 'size': len(files[path])} for path, sha in blob.items()]
    entries.append({'path': 'docs', 'type': 'tree', 'mode': '040000', 'sha': docs})
    return {'sha': root, 'truncated': False, 'tree': entries}, root


class Linux101Tests(unittest.TestCase):
    def test_tree_detects_truncation_and_tree_tampering(self):
        document, root = tree_fixture()
        self.assertEqual(len(research.validate_tree(document, root)), 3)
        document['truncated'] = True
        with self.assertRaises(ValueError): research.validate_tree(document, root)
        document['truncated'] = False
        document['tree'][0]['sha'] = '0' * 40
        with self.assertRaises(ValueError): research.validate_tree(document, root)

    def test_unsafe_paths_and_existing_output_rejected(self):
        for path in ('../escape', 'docs/../x.md', '/abs', 'docs\\x.md', 'docs/CON.md', 'C:/x', 'docs/a./x'):
            with self.subTest(path=path), self.assertRaises(ValueError): research.safe_path(path)
        with TemporaryDirectory() as tmp, patch.object(research, 'ROOT', Path(tmp)):
            research.create_dir('one')
            with self.assertRaises(FileExistsError): research.create_dir('one')
            with self.assertRaises(ValueError): research.create_dir('../escape')

    def test_paragraphs_preserve_lines_and_omit_fenced_and_indented_code(self):
        source = '# 标题\n\n第一行\n第二行\n\n```python\n代码中文\n```\n\n    中文代码\n\n另一段。\n'
        found = list(research.paragraphs(source))
        self.assertEqual([(p['start_line'], p['end_line'], p['text']) for p in found], [(3, 4, '第一行\n第二行'), (12, 12, '另一段。')])

    def test_changed_source_is_rejected(self):
        with TemporaryDirectory() as tmp, patch.object(research, 'ROOT', Path(tmp)):
            base = Path(tmp) / 'input'
            (base / 'source' / 'docs').mkdir(parents=True)
            (base / 'source' / 'docs' / 'a.md').write_bytes(b'changed')
            manifest = {'repository': research.REPO, 'commit': research.COMMIT, 'files': [{'path': 'docs/a.md', 'size': 3, 'git_blob_sha1': research.digest_blob(b'old'), 'sha256': hashlib.sha256(b'old').hexdigest()}]}
            (base / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            args = type('Args', (), {'source': 'input', 'output': 'samples', 'seed': 87, 'count': 1})()
            with self.assertRaises(ValueError): research.sample(args)
            self.assertFalse((Path(tmp) / 'samples').exists())

    def test_sampling_is_reproducible_and_rejects_inline_code(self):
        with TemporaryDirectory() as tmp, patch.object(research, 'ROOT', Path(tmp)):
            base = Path(tmp) / 'input'
            (base / 'source' / 'docs').mkdir(parents=True)
            data = '第一段。\n\n代码 `print` 不应入选。\n\n第二段。\n'.encode()
            (base / 'source' / 'docs' / 'a.md').write_bytes(data)
            manifest = {'repository': research.REPO, 'commit': research.COMMIT, 'files': [{'path': 'docs/a.md', 'size': len(data), 'git_blob_sha1': research.digest_blob(data), 'sha256': hashlib.sha256(data).hexdigest()}]}
            (base / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            for output in ('one', 'two'):
                args = type('Args', (), {'source': 'input', 'output': output, 'seed': 87, 'count': 2})()
                research.sample(args)
            one = json.loads((Path(tmp) / 'one' / 'samples.json').read_text(encoding='utf-8'))
            two = json.loads((Path(tmp) / 'two' / 'samples.json').read_text(encoding='utf-8'))
            self.assertEqual(one, two)
            self.assertEqual(one['candidate_count'], 2)
            self.assertEqual({p['text'] for p in one['samples']}, {'第一段。', '第二段。'})


if __name__ == '__main__':
    unittest.main()
