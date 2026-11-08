"""Measure exact normalized transcript containment in training prose only."""
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NONWORD = re.compile(r'[^\w]+', re.UNICODE)


def norm(text):
    return NONWORD.sub('', unicodedata.normalize('NFKC', text).casefold())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--transcript', required=True)
    p.add_argument('--transcript-sha256', required=True)
    p.add_argument('--prose-data', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    transcript, data, output = ROOT / args.transcript, ROOT / args.prose_data, ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    if digest(transcript) != args.transcript_sha256:
        raise ValueError('Transcript checksum mismatch')
    prose = data / 'pretrain-train.jsonl'
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf8'))
    prose_hash = digest(prose)
    if prose_hash != manifest['files'][prose.name]:
        raise ValueError('Prose checksum mismatch')
    sentences = set()
    with transcript.open(encoding='utf8', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            value = norm(row.get('sentence') or '')
            if value:
                sentences.add(value)
    minimum = 12
    eligible = {value for value in sentences if len(value) >= minimum}
    prefixes = {}
    for value in eligible:
        prefixes.setdefault(value[:minimum], []).append(value)
    matched, document_hits, document_count = set(), 0, 0
    with prose.open(encoding='utf8') as f:
        for line in f:
            document_count += 1
            text = norm(json.loads(line)['text'])
            candidates = set()
            for i in range(len(text) - minimum + 1):
                candidates.update(prefixes.get(text[i:i + minimum], ()))
            hits = {value for value in candidates if value in text}
            if hits:
                document_hits += 1
                matched.update(hits)
    result = dict(status='analysis_only_not_approved_for_training', transcript_sha256=digest(transcript),
                  prose_sha256=prose_hash, prose_manifest_sha256=digest(data / 'manifest.json'),
                  evaluator_sha256=digest(Path(__file__)), prose_documents=document_count,
                  transcript_unique_normalized=len(sentences), minimum_characters=minimum,
                  eligible_unique_sentences=len(eligible), matched_unique_sentences=len(matched),
                  matched_documents=document_hits,
                  scope='Exact full sentence containment after NFKC, casefold and non-word removal, sentences at least 12 characters; training prose only. Does not detect paraphrase or near duplicates, and does not establish source provenance.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
