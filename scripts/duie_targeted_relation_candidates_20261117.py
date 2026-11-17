"""Select local weakly-labeled multi-person attribution candidates; no raw text export."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import pyarrow.parquet as parquet
from scripts.audit_duie_subject_relations import SOURCE, SOURCE_SHA, PERSON_TYPES

OUT = ROOT / 'runs/duie-targeted-relation-candidates-20261117.json'
RELATIONS = {'妻子', '丈夫', '出生地', '毕业院校', '国籍'}


def candidates(rows):
    output = []
    for row in rows:
        text, facts = row.get('text'), row.get('spo_list')
        if not isinstance(text, str) or not isinstance(facts, list) or not 10 <= len(text) <= 160:
            continue
        valid = []
        for f in facts:
            subject, predicate, obj = f.get('subject'), f.get('predicate'), f.get('object')
            if (f.get('subject_type') in PERSON_TYPES and predicate in RELATIONS
                    and isinstance(subject, str) and 2 <= len(subject) <= 8
                    and isinstance(obj, str) and 1 <= len(obj) <= 30
                    and text.count(subject) == 1 and text.count(obj) == 1 and subject not in obj and obj not in subject):
                valid.append((subject, predicate, obj))
        triples = []
        for a in valid:
            for b in valid:
                if a[1] == b[1] and a[0] != b[0] and a[2] != b[2] and a[0] not in b[0] and b[0] not in a[0]:
                    triples.append((a, b))
        if triples:
            output.append({'digest': hashlib.sha256(text.encode('utf8')).hexdigest(),
                           'pairs': sorted(set(triples)), 'text': text})
    return output


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SOURCE_SHA:
        raise ValueError('Source mismatch')
    rows = candidates(parquet.read_table(SOURCE).to_pylist())
    counts = Counter(pair[0][1] for row in rows for pair in row['pairs'])
    result = {'rows': len(rows), 'ordered_pairs_by_relation': dict(counts), 'source_sha256': digest,
              'limitations': ['Heuristic candidate counts only; annotations are not verified fact ownership.',
                  'Text and triple values stay in local source; no extraction, model inference, or training.',
                  'Data mirror license and original text rights remain unverified.']}
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != digest:
        raise ValueError('Source changed')
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
