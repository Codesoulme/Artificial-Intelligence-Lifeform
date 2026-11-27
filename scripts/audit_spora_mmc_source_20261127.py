"""Inspect official MMC projected Chinese train/dev sources without model training."""
import hashlib
import json
import sys
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/research/mmc-source-audit-20261127-v1'
COMMIT = 'a7007d1d4556a3f4347a3d7b686f71d66bd1e2d9'
PREFIX = 'data_creation/data/jsonline_format/mmc_zh_corrected/'
FILES = {'train': PREFIX + 'train.chinese.512.jsonlines', 'dev': PREFIX + 'dev.chinese.512.jsonlines'}
HEADERS = {'User-Agent': 'SPORA-research', 'Accept': 'application/vnd.github+json'}


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        return response.read()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def blob_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def inspect_file(path):
    docs, scenes, exact = set(), set(), set()
    counts = {'rows': 0, 'clusters': 0, 'mentions': 0, 'multi_mention_clusters': 0,
              'invalid_spans': 0, 'invalid_subtoken_map': 0, 'invalid_sentence_map': 0,
              'sentence_speaker_shape_mismatch': 0, 'multi_speaker_rows': 0,
              'missing_fields': 0, 'duplicate_doc_key': 0, 'tokens_not_equal_subtokens': 0}
    with path.open(encoding='utf8') as source:
        for line in source:
            row = json.loads(line)
            counts['rows'] += 1
            if not all(k in row for k in ('doc_key', 'tokens', 'sentences', 'speakers', 'clusters', 'sentence_map', 'subtoken_map')):
                counts['missing_fields'] += 1; continue
            key = row['doc_key']
            if key in docs: counts['duplicate_doc_key'] += 1
            docs.add(key); scenes.add(key.rsplit('_', 1)[0])
            sentences, speakers = row['sentences'], row['speakers']
            subtokens = [token for sentence in sentences for token in sentence]
            flat_speakers = [speaker for sent in speakers for speaker in sent]
            exact.add(sha256(json.dumps(subtokens, ensure_ascii=False, separators=(',', ':')).encode('utf8')))
            if len(flat_speakers) != len(subtokens) or len(speakers) != len(sentences) or any(len(a) != len(b) for a, b in zip(sentences, speakers)):
                counts['sentence_speaker_shape_mismatch'] += 1
            if len(set(flat_speakers)) > 1: counts['multi_speaker_rows'] += 1
            if len(row['tokens']) != len(subtokens): counts['tokens_not_equal_subtokens'] += 1
            if len(row['subtoken_map']) != len(subtokens) or any(type(x) is not int or x < 0 or x >= len(row['tokens']) for x in row['subtoken_map']):
                counts['invalid_subtoken_map'] += 1
            if len(row['sentence_map']) != len(subtokens) or any(type(x) is not int or x < 0 for x in row['sentence_map']):
                counts['invalid_sentence_map'] += 1
            for cluster in row['clusters']:
                counts['clusters'] += 1
                counts['multi_mention_clusters'] += len(cluster) > 1
                for span in cluster:
                    counts['mentions'] += 1
                    if not (isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span)
                            and 0 <= span[0] <= span[1] < len(subtokens)):
                        counts['invalid_spans'] += 1
    return counts, docs, scenes, exact


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    def stage(name):
        item = {'stage': name, 'seconds': time.perf_counter() - start}
        (OUT / 'stage.json').write_text(json.dumps(item), encoding='utf8')
        print(json.dumps(item), flush=True)
    try:
        stage('fetch_tree')
        tree = json.loads(fetch('https://api.github.com/repos/boyuanzheng010/mmc/git/trees/' + COMMIT + '?recursive=1'))
        if tree.get('truncated'): raise ValueError('Incomplete repository tree')
        entries = {v['path']: v for v in tree['tree'] if v['type'] == 'blob'}
        results = {}; groups = {}
        for label, relative in FILES.items():
            stage('fetch_' + label)
            content = fetch('https://raw.githubusercontent.com/boyuanzheng010/mmc/' + COMMIT + '/' + relative)
            expected = entries[relative]
            if len(content) != expected['size'] or blob_sha(content) != expected['sha']:
                raise ValueError('Remote content does not match commit tree: ' + label)
            dest = OUT / (label + '.jsonlines')
            with dest.open('xb') as stream: stream.write(content)
            counts, docs, scenes, texts = inspect_file(dest)
            results[label] = {'path': relative, 'bytes': len(content), 'git_blob_sha1': expected['sha'],
                              'sha256': sha256(content), 'counts': counts}
            groups[label] = (docs, scenes, texts)
        overlap = {'doc_key': len(groups['train'][0] & groups['dev'][0]),
                   'scene_prefix': len(groups['train'][1] & groups['dev'][1]),
                   'exact_subtoken_sequences': len(groups['train'][2] & groups['dev'][2])}
        result = {'commit': COMMIT, 'files': results, 'overlap': overlap,
                  'root_license_files': [p for p in entries if '/' not in p and p.lower().startswith(('license', 'copying'))],
                  'seconds': time.perf_counter() - start}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
        stage('complete')
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__': main()
