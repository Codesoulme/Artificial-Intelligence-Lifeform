"""Rescore immutable development outputs without touching model weights."""
import hashlib
import json
from pathlib import Path

from scripts.spora_memory_generation_metrics import asserted_fact, score_pair

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ('semantic-bridge-pilot', 16), ('semantic-mlp-bridge-pilot', 16),
    ('codebook-upper-bound-pilot', 16), ('full-fact-semantic-bridge-pilot', 16),
    ('full-fact-codebook-upper-bound-320', 16), ('learned-codec-bridge-pilot', 16),
    ('late-residual-pilot', 16), ('late-residual-strict-pilot', 16),
    ('fact-only-late-residual-strict-pilot', 16), ('all-facts-late-residual-strict-pilot', 18),
)
OUT = ROOT / 'reports/semantic-bridge-rescored-20261112.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    runs = []
    for name, train_count in RUNS:
        path = ROOT / f'runs/continuous-reader-{name}-20261112/result.json'
        raw = path.read_bytes()
        data = json.loads(raw)
        own = [r for r in data['after'] if r['group'] == 'held_name' and r['condition'] == 'memory']
        swapped = {r['expected']: r for r in data['after'] if r['condition'] == 'swapped'}
        if len(own) != 18 or len({r['expected'] for r in own}) != 18:
            raise ValueError('Unexpected original fruit schedule')
        scored = []
        for i, row in enumerate(own):
            item = dict(expected=row['expected'], answer=row['answer'], name=row['name'],
                        split='seen_value_new_name' if i < train_count else 'new_value_new_name',
                        correct=asserted_fact(row['answer'], row['name'], row['expected']))
            if row['expected'] in swapped:
                alternative = own[(i + 1) % len(own)]['expected']
                item['replacement'] = alternative
                item['swapped_answer'] = swapped[row['expected']]['answer']
                item.update(score_pair(row['answer'], item['swapped_answer'], name=row['name'],
                                       original=row['expected'], replacement=alternative))
            scored.append(item)
        counts = {}
        for split in ('seen_value_new_name', 'new_value_new_name'):
            subset = [r for r in scored if r['split'] == split]
            counts[split] = {'correct': sum(r['correct'] for r in subset), 'total': len(subset)}
        off_before = [r['token_ids'] for r in data['before'] if r['condition'] == 'off']
        off_after = [r['token_ids'] for r in data['after'] if r['condition'] == 'off']
        runs.append(dict(run=name, input_sha256=hashlib.sha256(raw).hexdigest(),
                         candidate_sha256=data['candidate_sha256'], summary=counts,
                         pairs_correct=sum(r.get('pair_correct', False) for r in scored),
                         pairs_total=len(swapped),
                         swapped_correct=sum(r.get('swapped_correct', False) for r in scored),
                         off_probe_count=len(off_before), off_tokens_unchanged=off_before == off_after,
                         seconds=data['seconds'], rows=scored))
    report = {'status': 'retrospective_development_rescoring', 'runs': runs,
              'limitations': ['No new generalization test; these outputs were already inspected.',
                              'Original training coverage is reconstructed from versioned run scripts.',
                              'Exact assertions avoid list/negation false positives but can reject paraphrases.',
                              'Off controls here are fruit questions, not general conversation quality.',
                              'Codebook variants are diagnostic retrieval paths, not mathematical performance bounds.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps([{k: r[k] for k in ('run', 'summary', 'pairs_correct', 'pairs_total', 'swapped_correct')}
                      for r in runs], ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
