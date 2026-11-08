"""Materialize only explicitly approved rows from the fixed dialogue review."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/teaching-curriculum-v4'
REVIEW = ROOT / 'reports/P30-clean-teacher-256-review.json'
OUTPUT = ROOT / 'runs/teaching-curriculum-v5'
REPORT = ROOT / 'reports/P30-reviewed-teacher-preparation.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not REVIEW.exists():
        raise FileNotFoundError(REVIEW)
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError('Preserve prior result; choose a new curriculum version')
    review = json.loads(REVIEW.read_text('utf-8'))
    if review.get('approved_for_pilot') is not True:
        raise ValueError('Independent review did not approve the pilot')
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text('utf-8'))
    if review.get('source_manifest_sha256') != sha(SOURCE / 'manifest.json'):
        raise ValueError('Review covers a different source manifest')
    rows = review.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Review rows required')
    approved = {}
    for item in rows:
        number = item.get('source_row_1based')
        if type(number) is not int or number in approved:
            raise ValueError('Invalid or duplicate reviewed row')
        if item.get('approved') is True:
            approved[number] = item.get('teacher_sha256')
    if not approved:
        raise ValueError('No approved rows')
    OUTPUT.mkdir(parents=True)
    for name in ('foundations.jsonl', 'foundations-tokens.jsonl'):
        shutil.copyfile(SOURCE / name, OUTPUT / name)
    kept = 0
    targets = 0
    seen = set()
    with (SOURCE / 'teacher.jsonl').open(encoding='utf-8') as raw, \
         (SOURCE / 'teacher-tokens.jsonl').open(encoding='utf-8') as encoded, \
         (OUTPUT / 'teacher.jsonl').open('x', encoding='utf-8') as out_raw, \
         (OUTPUT / 'teacher-tokens.jsonl').open('x', encoding='utf-8') as out_encoded:
        for number, (raw_line, encoded_line) in enumerate(zip(raw, encoded), 1):
            if number not in approved:
                continue
            raw_digest = hashlib.sha256(raw_line.encode('utf-8')).hexdigest()
            expected = approved[number]
            if expected and expected != raw_digest:
                raise ValueError('Reviewed row changed: '+str(number))
            if raw_digest in seen:
                raise ValueError('Duplicate approved source row')
            seen.add(raw_digest)
            out_raw.write(raw_line)
            out_encoded.write(encoded_line)
            targets += sum(json.loads(encoded_line)['mask'])
            kept += 1
    result = {
        'status': 'prepared_reviewed_dialogue_subset',
        'source': 'runs/teaching-curriculum-v4',
        'source_manifest_sha256': sha(SOURCE / 'manifest.json'),
        'review_sha256': sha(REVIEW),
        'tokenizer_sha256': source_manifest['tokenizer_sha256'],
        'teacher_rows_kept': kept,
        'teacher_targets_kept': targets,
        'reviewed_rows_requested': len(rows),
        'files': {p.name: sha(p) for p in OUTPUT.iterdir() if p.is_file()},
        'scope': 'Only rows explicitly approved by the fixed 256-row review. This is a bounded noisy dialogue pilot, not a factual corpus certification or chat acceptance.',
        'foundations_inherited': True,
    }
    (OUTPUT / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('teacher_rows_kept','teacher_targets_kept','reviewed_rows_requested')}))


if __name__ == '__main__':
    main()
