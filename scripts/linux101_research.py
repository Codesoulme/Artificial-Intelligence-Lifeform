"""Fetch and sample the fixed Linux 101 source for local research only."""
import argparse
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import quote

COMMIT = '4099db4a1f90e8e89cac3eca4bf2e36b2932e3f1'
REPO = 'https://github.com/ustclug/Linux101-docs'
API = 'https://api.github.com/repos/ustclug/Linux101-docs'
RAW = f'https://raw.githubusercontent.com/ustclug/Linux101-docs/{COMMIT}/'
ROOT = Path(__file__).resolve().parents[1] / 'runs' / 'research'
LIMIT = 1_000_000


def request(url, no_revoke):
    cmd = ['curl', '--fail', '--silent', '--show-error', '--location', '--proto', '=https',
           '--proto-redir', '=https', '--max-time', '60', '-H', 'User-Agent: SPORA-research']
    if no_revoke:
        cmd.append('--ssl-no-revoke')
    return subprocess.run(cmd + [url], check=True, capture_output=True).stdout


def digest_blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def safe_path(path):
    p = PurePosixPath(path)
    if (not path or path.startswith('/') or '\\' in path or any(
        not s or s in ('.', '..') or ':' in s or s.endswith((' ', '.')) or
        re.fullmatch(r'(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?', s)
        for s in path.split('/')) or str(p) != path):
        raise ValueError(f'Unsafe path: {path!r}')
    return p


def validate_tree(document, root_sha):
    if document.get('truncated') is not False or document.get('sha') != root_sha:
        raise ValueError('Tree is truncated or does not match commit tree')
    entries = document['tree']
    dirs = {'': []}
    seen = set()
    for item in entries:
        path = item['path']
        safe_path(path)
        if path.casefold() in seen:
            raise ValueError('Duplicate or case-colliding path')
        seen.add(path.casefold())
        if not re.fullmatch(r'[0-9a-f]{40}', item['sha']):
            raise ValueError('Invalid object hash')
        parent, _, name = path.rpartition('/')
        dirs.setdefault(parent, []).append((name, item))
        if item['type'] == 'tree':
            dirs.setdefault(path, [])
        elif item['type'] == 'blob':
            if item['mode'] not in ('100644', '100755', '120000') or not isinstance(item.get('size'), int) or item['size'] < 0:
                raise ValueError('Invalid blob metadata')
        elif item['type'] != 'commit' or item['mode'] != '160000':
            raise ValueError('Invalid tree entry')
    for directory in sorted(dirs, key=lambda x: x.count('/') + bool(x), reverse=True):
        contents = bytearray()
        for name, item in sorted(dirs[directory], key=lambda pair: (pair[0] + ('/' if pair[1]['type'] == 'tree' else '')).encode()):
            mode = '40000' if item['type'] == 'tree' else item['mode']
            contents.extend(mode.encode() + b' ' + name.encode() + b'\0' + bytes.fromhex(item['sha']))
        actual = hashlib.sha1(b'tree ' + str(len(contents)).encode() + b'\0' + contents).hexdigest()
        expected = root_sha if not directory else next((x['sha'] for x in entries if x['path'] == directory and x['type'] == 'tree'), None)
        if actual != expected:
            raise ValueError(f'Tree hash mismatch: {directory}')
    chosen = [x for x in entries if (x['path'].startswith('docs/') and x['path'].lower().endswith('.md')) or x['path'] in ('LICENSE', 'README.md')]
    if any(x['type'] != 'blob' or x['mode'] not in ('100644', '100755') or x['size'] >= LIMIT for x in chosen):
        raise ValueError('Selected item is not a regular blob below the size limit')
    if not {'LICENSE', 'README.md'}.issubset({x['path'] for x in chosen}) or not any(x['path'].startswith('docs/') for x in chosen):
        raise ValueError('Required source files missing')
    return sorted(chosen, key=lambda x: x['path'])


def create_dir(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Output name must contain only letters, digits, underscore or hyphen')
    ROOT.mkdir(parents=True, exist_ok=True)
    dest = ROOT / name
    dest.mkdir(exist_ok=False)
    return dest


def fetch(args):
    commit = json.loads(request(f'{API}/git/commits/{COMMIT}', args.ssl_no_revoke))
    if commit.get('sha') != COMMIT:
        raise ValueError('Commit mismatch')
    tree_sha = commit['tree']['sha']
    tree_bytes = request(f'{API}/git/trees/{tree_sha}?recursive=1', args.ssl_no_revoke)
    tree = json.loads(tree_bytes)
    selected = validate_tree(tree, tree_sha)
    dest = create_dir(args.output)
    records = []
    for item in selected:
        path = item['path']
        data = request(RAW + quote(path, safe='/'), args.ssl_no_revoke)
        if len(data) != item['size'] or digest_blob(data) != item['sha']:
            raise ValueError(f'Blob size/hash mismatch: {path}')
        target = dest / 'source' / Path(*safe_path(path).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(data)
        records.append({'path': path, 'git_blob_sha1': item['sha'], 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    manifest = {'repository': REPO, 'commit': COMMIT, 'tree_sha1': tree_sha,
                'tree_truncated': False, 'tree_metadata_sha256': hashlib.sha256(tree_bytes).hexdigest(),
                'ssl_no_revoke': args.ssl_no_revoke, 'training_approved': False,
                'attribution': 'LUG@USTC and contributors; see README.md, LICENSE and source notices. Third-party material requires separate review.',
                'files': records}
    (dest / 'tree.json').write_bytes(tree_bytes)
    (dest / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'directory': str(dest), 'files': len(records), 'bytes': sum(r['size'] for r in records)}, ensure_ascii=False))


def paragraphs(text):
    """Conservative Markdown blocks, with 1-based source lines; no context guarantee."""
    lines = text.splitlines()
    block = []
    fence = None
    html = False
    def emit():
        if not block:
            return None
        start, end = block[0][0], block[-1][0]
        value = '\n'.join(line for _, line in block)
        block.clear()
        # Reject inline code and Markdown extension directives as whole blocks.
        if (re.search(r'[\u4e00-\u9fff]', value) and '`' not in value and
                not re.search(r'(?m)^\s*(?:!!!|\?\?\?|\{[:%]|: |\$\$)', value)):
            return {'start_line': start, 'end_line': end, 'text': value}
        return None
    for n, line in enumerate(lines, 1):
        match = re.match(r'^\s*(`{3,}|~{3,})', line)
        if fence:
            if re.match(r'^\s*' + re.escape(fence[0]) + '{' + str(len(fence)) + r',}\s*$', line):
                fence = None
            continue
        if match:
            result = emit()
            if result: yield result
            fence = match.group(1)
            continue
        if re.match(r'^\s*<!--', line): html = True
        skip = (html or not line.strip() or line.startswith(('    ', '\t')) or
                bool(re.match(r'^\s*(?:#{1,6}\s|>|[-*+]\s|\d+[.)]\s|\||!\[|\[\^|\[[^]]+\]:|<|[-=]{3,}\s*$)', line)))
        if skip:
            result = emit()
            if result: yield result
        else:
            block.append((n, line))
        if html and '-->' in line: html = False
    result = emit()
    if result: yield result


def sample(args):
    source = ROOT / args.source
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.source) or source.is_symlink():
        raise ValueError('Invalid source directory')
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['commit'] != COMMIT or manifest['repository'] != REPO:
        raise ValueError('Unexpected source')
    candidates = []
    for record in manifest['files']:
        path = record['path']
        if not path.startswith('docs/') or not path.lower().endswith('.md'): continue
        data = (source / 'source' / Path(*safe_path(path).parts)).read_bytes()
        if len(data) != record['size'] or digest_blob(data) != record['git_blob_sha1'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError(f'Source changed: {path}')
        for p in paragraphs(data.decode('utf-8-sig')):
            p.update({'source_file': path, 'sha256': hashlib.sha256(p['text'].encode('utf-8')).hexdigest(),
                      'warning': '可能依赖附近代码、图表或上下文；排除代码不保证语意完整，请人工核对。'})
            candidates.append(p)
    if args.count < 1 or args.count > len(candidates):
        raise ValueError('Sample count outside candidate range')
    chosen = random.Random(args.seed).sample(candidates, args.count)
    dest = create_dir(args.output)
    result = {'source_commit': COMMIT, 'source_manifest_sha256': hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest(),
              'seed': args.seed, 'candidate_count': len(candidates), 'sample_count': len(chosen),
              'method': 'Python random.Random(seed).sample over source path and line order; paragraph text preserves line breaks.',
              'limitations': 'Heuristic Markdown filtering excludes fenced, indented and backtick inline code; other markup or unmarked code may remain. Context and third-party rights not verified.',
              'training_approved': False, 'samples': chosen}
    (dest / 'samples.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'directory': str(dest), 'candidates': len(candidates), 'samples': len(chosen)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    get = sub.add_parser('fetch')
    get.add_argument('--output', required=True)
    get.add_argument('--ssl-no-revoke', action='store_true')
    get.set_defaults(func=fetch)
    draw = sub.add_parser('sample')
    draw.add_argument('--source', required=True)
    draw.add_argument('--output', required=True)
    draw.add_argument('--count', type=int, default=20)
    draw.add_argument('--seed', type=int, default=87)
    draw.set_defaults(func=sample)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
