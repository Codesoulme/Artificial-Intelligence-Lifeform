"""Audit P230 full-text containment in two fixed historical training files."""
import collections
import hashlib
import json
from pathlib import Path
import sys

import ahocorasick

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_authored_text_overlap import EXPECTED
from scripts.prepare_wikimedia_intro_candidate import norm

CANDIDATE = ROOT / 'runs/p230-wikipedia-body-refined-v1-20261109'
SOURCES = {
    'p28': ROOT / 'runs/p28-stories-data/candidate-v3/pretrain-train.jsonl',
    'v5': ROOT / 'runs/conversation-data-v5/pretrain-train.jsonl',
}
OUTPUT = ROOT / 'runs/p231-wikipedia-ancestor-overlap-v1-20261109.json'
MANIFEST_SHA256 = '22f9466980324f0dc22ee93e2a4b8b2428ec2bf65d08fcec9ceec61ab809fa02'
CANDIDATE_SHA256 = 'a81f0c484c189e7971a01c437312a17b1b2ef9f38cb6cedd66662e1a74092fc8'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_candidates(directory, manifest_sha=MANIFEST_SHA256,
                    candidate_sha=CANDIDATE_SHA256, expected_count=9092):
    manifest_raw = (directory / 'manifest.json').read_bytes()
    if digest(manifest_raw) != manifest_sha:
        raise ValueError('Candidate manifest digest mismatch')
    manifest = json.loads(manifest_raw)
    raw = (directory / 'candidates.jsonl').read_bytes()
    if digest(raw) != candidate_sha or manifest['files']['candidates.jsonl'] != candidate_sha:
        raise ValueError('Candidate digest mismatch')
    rows = []
    identities = set()
    for line in raw.splitlines():
        row = json.loads(line)
        page, start, end, text = (row['page_id'], row['char_start'],
                                  row['char_end'], row['text'])
        if (not isinstance(page, str) or not page or type(start) is not int or
                type(end) is not int or not isinstance(text, str) or start < 0 or
                end <= start or end - start != len(text) or
                digest(text.encode('utf8')) != row['text_sha256']):
            raise ValueError('Invalid candidate identity or text digest')
        identity = (page, start, end)
        if identity in identities:
            raise ValueError('Duplicate candidate identity')
        identities.add(identity)
        normalized = norm(text)
        if not normalized:
            raise ValueError('Empty normalized candidate')
        rows.append((f'{page}:{start}:{end}', normalized))
    if len(rows) != expected_count or manifest['counts']['retained_passages'] != expected_count:
        raise ValueError('Candidate count mismatch')
    return rows


def scan(rows, sources, expected):
    if set(sources) != {'p28', 'v5'} or set(expected) != set(sources):
        raise ValueError('Two fixed historical sources required')
    prefixes = collections.defaultdict(list)
    for index, (_, text) in enumerate(rows):
        prefixes[text[:24]].append(index)
    machine = ahocorasick.Automaton()
    for prefix, indices in prefixes.items():
        machine.add_word(prefix, indices)
    machine.make_automaton()
    counts = {name: [0] * len(rows) for name in sources}
    metadata = {}
    for name, path in sources.items():
        hasher = hashlib.sha256()
        records = 0
        with path.open('rb') as stream:
            for line in stream:
                hasher.update(line)
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get('text')
                if not isinstance(text, str):
                    raise ValueError('Invalid historical record')
                records += 1
                normalized = norm(text)
                possible = set()
                for _, indices in machine.iter(normalized):
                    possible.update(indices)
                for index in possible:
                    if rows[index][1] in normalized:
                        counts[name][index] += 1
        actual = hasher.hexdigest()
        if actual != expected[name]:
            raise ValueError('Historical source digest mismatch: ' + name)
        metadata[name] = {'sha256': actual, 'records': records,
                          'matched_candidates': sum(n > 0 for n in counts[name])}
    return counts, metadata


def audit(directory=CANDIDATE, sources=SOURCES):
    rows = load_candidates(directory)
    counts, metadata = scan(rows, sources, EXPECTED)
    return {
        'status': 'analysis_only_not_approved_for_training',
        'candidate_manifest_sha256': MANIFEST_SHA256,
        'candidate_sha256': CANDIDATE_SHA256,
        'script_sha256': digest(Path(__file__).read_bytes()),
        'candidate_count': len(rows),
        'sources': metadata,
        'matched_any_source': sum(any(counts[s][i] for s in counts) for i in range(len(rows))),
        'method': ('NFKC, casefold, remove nonword characters; Aho-Corasick indexes the first '
                   '24 normalized characters (or the entire shorter candidate), then checks full normalized candidate containment '
                   'in each normalized historical record. Counts are distinct source records, '
                   'not occurrences. Candidate IDs are page_id:char_start:char_end.'),
        'limitations': [
            'Exact normalized full containment only; partial matches, edits, paraphrases and semantic near-duplicates can remain.',
            'Normalization removes punctuation and whitespace and can produce matches that are not byte-identical.',
            'Only the two specified training files are checked; a match does not prove model exposure.',
            'No development or final-test data was read; this audit is not training approval.',
        ],
        'candidates': [{'candidate_id': identity,
                        'sources': {name: counts[name][i] for name in counts}}
                       for i, (identity, _) in enumerate(rows)],
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = audit()
    with OUTPUT.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('candidate_count', 'sources', 'matched_any_source')}), flush=True)


if __name__ == '__main__':
    main()
