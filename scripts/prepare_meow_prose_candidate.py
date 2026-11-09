"""Fetch and verify one author-licensed Chinese prose work; never trains."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = 'f6f8fcbd9f9bc9d7686013705a2bc69c66e7bea5'
BASE = f'https://raw.githubusercontent.com/tingyulingfeng/MeowMeow-Kitchen/{REVISION}/'
FILES = {
    'README.md': 'e99c3330c19291651ec5730f6e5c25bc833d147de89c3604ca1da061cdf3f2b7',
    'DATASET_CARD.md': 'ee1e4c21666884cfa930852891ecb254f708a1a261b7d521c1a56c5a4b512b39',
    'data/paragraphs.jsonl': '48c2a72adf43ab5ea16ea869a5401378c9b19704ec6d910bc64eaa3c10d2d3e9',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_paragraphs(raw):
    rows = [json.loads(line) for line in raw.decode('utf8').splitlines() if line.strip()]
    if len(rows) != 33:
        raise ValueError('Expected exactly 33 paragraphs')
    for index, row in enumerate(rows, 1):
        if (row.get('para_id') != index or row.get('id') != f'meow-v2-p{index:02d}'
                or not isinstance(row.get('text'), str) or not row['text'].strip()):
            raise ValueError('Invalid paragraph order or content')
    return '\n'.join(row['text'] for row in rows)


def prepare(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    downloads = {}
    for name, expected in FILES.items():
        with urlopen(BASE + name, timeout=30) as response:
            raw = response.read(200_001)
        if len(raw) > 200_000 or digest(raw) != expected:
            raise ValueError('Source size or digest mismatch: ' + name)
        downloads[name] = raw
    text = parse_paragraphs(downloads['data/paragraphs.jsonl'])
    output.mkdir(parents=False)
    for name, raw in downloads.items():
        (output / Path(name).name).write_bytes(raw)
    text_bytes = text.encode('utf8')
    (output / 'prose.txt').write_bytes(text_bytes)
    result = {'status': 'candidate_not_approved_for_training', 'training_started': False,
              'repository': 'https://github.com/tingyulingfeng/MeowMeow-Kitchen', 'revision': REVISION,
              'source_files_sha256': FILES, 'candidate_sha256': digest(text_bytes),
              'paragraphs': 33, 'characters': len(text), 'utf8_bytes': len(text_bytes),
              'source_declared_author': '听雨聆风的小厨房', 'source_declared_license': 'CC BY 4.0',
              'license_url': 'https://creativecommons.org/licenses/by/4.0/',
              'transformation': 'Join the 33 ordered original paragraph text fields with newline; omit commentary.',
              'limitations': ['One work by one author cannot support general language pretraining.',
                              'Attribution and modification disclosures must accompany any reuse.',
                              'Candidate is not approved; history and evaluation overlap still require review.',
                              'Source claims about authorship and license are not independently verified.']}
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
