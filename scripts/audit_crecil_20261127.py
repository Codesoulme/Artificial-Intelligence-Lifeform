"""Aggregate-only, read-only schema and overlap audit of CRECIL train/dev."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'runs/research/crecil-source-20261127-v1/Final_Data'
OUT = ROOT / 'runs/research/crecil-source-20261127-v1/audit.json'
EXPECTED = {'train': '8b197f6a857130577462813c28e2f8229f945c419ed18340a9a6270ed643a351',
            'dev': 'f3eef68ed89379e6d3acb4e685d557bf4bdaadad45dd15795141f9e86f8e764a'}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    splits, digests, stats = {}, {}, {}
    for name in ('train', 'dev'):
        raw = (DATA / (name + '.json')).read_bytes()
        digests[name] = hashlib.sha256(raw).hexdigest()
        if digests[name] != EXPECTED[name]:
            raise ValueError('Unexpected input digest')
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError('Unexpected root')
        splits[name] = data
        lengths, counts, labels = [], [], collections.Counter()
        missing_x = missing_y = missing_both = malformed = 0
        keys = set()
        for item in data:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError('Unexpected item structure')
            turns, relations = item
            if not isinstance(turns, list) or not all(isinstance(x, str) for x in turns):
                raise ValueError('Unexpected turns')
            lengths.append(len(turns));counts.append(len(relations))
            text = '\n'.join(turns)
            for relation in relations:
                keys.update(relation.keys())
                if not all(k in relation for k in ('x', 'y', 'r', 'rid')):
                    malformed += 1
                    continue
                labels.update(str(x) for x in relation['rid'])
                a, b = relation['x'] in text, relation['y'] in text
                missing_x += not a;missing_y += not b;missing_both += not a and not b
        stats[name] = {'documents': len(data), 'turns': sum(lengths), 'turns_min': min(lengths),
                       'turns_max': max(lengths), 'zero_relations': sum(n == 0 for n in counts),
                       'relation_records': sum(counts), 'relation_ids': dict(sorted(labels.items())),
                       'relation_keys': sorted(keys), 'malformed_relation_records': malformed,
                       'x_not_literal_in_dialogue': missing_x, 'y_not_literal_in_dialogue': missing_y,
                       'both_not_literal_in_dialogue': missing_both}
    def turn_hash(item):
        return hashlib.sha256(json.dumps(item[0], ensure_ascii=False, separators=(',', ':')).encode('utf8')).hexdigest()
    tr = collections.Counter(turn_hash(x) for x in splits['train'])
    de = collections.Counter(turn_hash(x) for x in splits['dev'])
    def triples(data):
        return {(r['x'], r['y'], str(rid)) for item in data for r in item[1] for rid in r['rid']}
    train_rel, dev_rel = triples(splits['train']), triples(splits['dev'])
    result = {'source_sha256': digests, 'stats': stats,
              'overlap': {'shared_exact_dialogues': len(tr.keys() & de.keys()),
                          'train_unique_dialogues': len(tr), 'dev_unique_dialogues': len(de),
                          'train_duplicate_dialogue_rows': sum(tr.values()) - len(tr),
                          'dev_duplicate_dialogue_rows': sum(de.values()) - len(de),
                          'train_unique_ordered_relation_triples': len(train_rel),
                          'dev_unique_ordered_relation_triples': len(dev_rel),
                          'shared_ordered_relation_triples': len(train_rel & dev_rel)},
              'limitations': ['Literal name occurrence does not establish identity, evidence or completeness.',
                              'No semantic annotation validation; official test split not accessed.',
                              'No relation text or source utterances are exported.']}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
