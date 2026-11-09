"""Prepare the Chinese side of a fixed bilingual literature parquet; never trains."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'runs/p192-chinese-vietnamese-literature-20261109/dataset_zh_vi.parquet'
EXPECTED_SOURCE_SHA = '2676db2aabdb3d6b5c2618b7cc8fa9ad283ae4e80bb0c8366065134295ce3b4e'
EXPECTED_ROWS = 396307


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cjk_ratio(text):
    return sum('\u3400' <= c <= '\u9fff' for c in text) / max(1, len(text))


def prepare(source, output):
    if output.exists():
        raise FileExistsError(output)
    if sha(source) != EXPECTED_SOURCE_SHA:
        raise ValueError('source hash mismatch')
    table = pq.read_table(source, columns=['zh', 'vi'])
    rows = table.to_pylist()
    if len(rows) != EXPECTED_ROWS:
        raise ValueError('row count mismatch')
    seen = set()
    kept = []
    rejected = {'empty': 0, 'low_cjk_ratio': 0, 'email_or_url': 0, 'exact_duplicate': 0}
    for index, row in enumerate(rows):
        text = (row.get('zh') or '').strip()
        if not text:
            rejected['empty'] += 1
            continue
        if cjk_ratio(text) < 0.5:
            rejected['low_cjk_ratio'] += 1
            continue
        if re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|https?://|www\.', text, re.I):
            rejected['email_or_url'] += 1
            continue
        digest = hashlib.sha256(text.encode('utf8')).hexdigest()
        if digest in seen:
            rejected['exact_duplicate'] += 1
            continue
        seen.add(digest)
        kept.append({'record_id': f'zh_row_{index}', 'source_row_index': index, 'text_sha256': digest, 'text': text, 'characters': len(text), 'cjk_ratio': cjk_ratio(text), 'review_status': 'candidate_not_approved_for_training'})
    output.mkdir(parents=False)
    candidate = output / 'candidate.jsonl'
    with candidate.open('x', encoding='utf8', newline='\n') as stream:
        for row in kept:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    manifest = {'status': 'candidate_not_approved_for_training', 'training_approval': False, 'training_started': False,
                'repository': 'moa/Chinese-Vietnamese-literature', 'revision': 'e24a1c862f03466ef35f345ee34661e1b4e4b6e6',
                'license': 'CC BY-SA 4.0', 'source_sha256': sha(source), 'source_rows': len(rows), 'candidate_rows': len(kept),
                'candidate_sha256': sha(candidate), 'rejected': rejected,
                'limitations': ['Source description says mainly public-domain works but does not provide a complete per-work rights manifest.', 'No story/work IDs are present in the two-column parquet schema, so sequential family boundaries remain unresolved.', 'Candidate is literary continuation material, not dialogue or general knowledge.', 'No training authorization.']}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
