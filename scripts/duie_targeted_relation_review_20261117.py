"""Summarize a single-reviewer development audit of six natural-text candidates."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import pyarrow.parquet as parquet
from scripts.audit_duie_subject_relations import SOURCE, SOURCE_SHA
from scripts.duie_targeted_relation_candidates_20261117 import candidates

OUT = ROOT / 'runs/duie-targeted-relation-review-20261117.json'
REVIEW = {
    '3b7fab3f6580c22b1cf2d764a15a2b15747edb42422888c053ee4d5ffe0c6a52': 'two_distinct_supported_pairs',
    '6230fd40a4990201e0cd8998a1f3ebcd103beae91769faa82f565d43c1d96a0d': 'one_couple_reciprocal_labels_not_two_pairs',
    'def10183ec182ad8e64870ce0525e685ff1370b1c08913e4ca574ff2e64857fe': 'same_person_alias_and_nested_location',
    'a3411ce3b1869e7c9292b3b178367e1e2868a426011bfc55eb82b45b8fd2a593': 'same_person_alias_and_nested_location',
    'fd7d440dd3b4e13283d93adef24a971fb5511e3319c2803b9cd7b70c24e1f937': 'alias_and_different_person_insufficient_relation',
    '917b650d1800dced453fe9fceec5b71086656b72112d7feb16a89b5d9ce16833': 'two_distinct_supported_pairs',
}


def compare(rows, review):
    ids = [row['digest'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(review):
        raise ValueError('Reviewed candidate manifest mismatch')
    return dict(Counter(review[digest] for digest in ids))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SOURCE_SHA:
        raise ValueError('Source mismatch')
    selected = candidates(parquet.read_table(SOURCE).to_pylist())
    result = {'summary': compare(selected, REVIEW), 'rows': len(selected), 'source_sha256': digest,
              'review_method': 'one model manually reviewed text and annotated relation candidates',
              'limitations': ['Selection and labels were reviewed after seeing data; development only.',
                  'No independent reviewer, model evaluation, truth verification, or write authorization.',
                  'Data rights remain unresolved; raw text, names and triples not exported.']}
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != digest:
        raise ValueError('Source changed')
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
