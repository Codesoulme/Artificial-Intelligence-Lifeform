"""Reconcile independent reviews without opening the reserved holdout as text.

Outputs supervise missing conversational references, not factual answerability.
No wrong-history label is inferred from a donor's unrelated source tree.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/oasst-context-expansion-20261118'
PARTITIONS = ('train_review', 'calibration_review')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_review(queue, document):
    expected = {r['case']: r for r in queue}
    reviews = document['reviews']
    if not isinstance(document.get('reviewer'), str) or not document['reviewer'].strip():
        raise ValueError('Reviewer description required')
    found = {}
    for row in reviews:
        case = row['case']
        if type(case) is not int or case not in expected or case in found:
            raise ValueError('Duplicate, unknown, or non-integer review case')
        if row['partition'] != expected[case]['partition']:
            raise ValueError('Review partition mismatch')
        need, resolves = row['context_need'], row['history_resolves']
        if need is not None and (type(need) is not int or need not in (0, 1)):
            raise ValueError('Invalid context label')
        if resolves is not None and type(resolves) is not bool:
            raise ValueError('Invalid resolution label')
        if need == 0 and resolves is not None:
            raise ValueError('Standalone requests must have null resolution')
        if not isinstance(row.get('rationale'), str) or not row['rationale'].strip():
            raise ValueError('Rationale required')
        if not isinstance(row.get('quality_flag'), str):
            raise ValueError('Quality flag required')
        found[case] = row
    if set(found) != set(expected):
        raise ValueError('Incomplete review coverage')
    return found


def reconcile(queue, reviewer_a, reviewer_b):
    if any(r['partition'] not in PARTITIONS for r in queue):
        raise ValueError('Reserved or unknown partition prohibited')
    for key in ('case', 'group'):
        if len({r[key] for r in queue}) != len(queue):
            raise ValueError('Duplicate source ' + key)
    if any(type(r['case']) is not int for r in queue):
        raise ValueError('Invalid source case')
    current = [r['messages'][-1]['content'].strip() for r in queue]
    if len(set(current)) != len(current):
        raise ValueError('Duplicate current-turn text')
    a, b = validate_review(queue, reviewer_a), validate_review(queue, reviewer_b)
    accepted, excluded, inputs = [], [], []
    for row in sorted(queue, key=lambda r: r['case']):
        case = row['case']
        pair_a = (a[case]['context_need'], a[case]['history_resolves'])
        pair_b = (b[case]['context_need'], b[case]['history_resolves'])
        if pair_a != pair_b or pair_a[0] is None or (pair_a[0] == 1 and pair_a[1] is not True):
            excluded.append({'case': case, 'partition': row['partition'],
                             'reason': 'disagreement' if pair_a != pair_b else 'ambiguous_or_unresolved',
                             'reviewer_a': a[case], 'reviewer_b': b[case]})
            continue
        need = pair_a[0]
        accepted.append({'case': case, 'partition': row['partition'], 'context_need': need,
                         'history_resolves': pair_a[1],
                         'quality_flags': [a[case]['quality_flag'], b[case]['quality_flag']]})
        for variant, messages, label in (
            ('current_only', row['messages'][-1:], need),
            ('original_history', row['messages'], 0),
        ):
            inputs.append({'case': case, 'partition': row['partition'], 'group': row['group'],
                           'variant': variant, 'context_need': need, 'missing_context': label,
                           'messages': messages, 'current_message_id': row['message_ids'][-1],
                           'history_message_ids': row['message_ids'][:-1] if variant == 'original_history' else [],
                           'quality_flags': accepted[-1]['quality_flags']})
    counts = {}
    for part in PARTITIONS:
        chosen = [r for r in accepted if r['partition'] == part]
        counts[part] = {'accepted': len(chosen), 'dependent': sum(r['context_need'] for r in chosen),
                        'independent': sum(r['context_need'] == 0 for r in chosen),
                        'excluded': sum(r['partition'] == part for r in excluded), 'input_rows': 2*len(chosen)}
    return {'accepted': accepted, 'excluded': excluded, 'counts': counts}, inputs


def main():
    out = SOURCE / 'reconciled'
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf8'))
    expected_files = {n+'.json' for n in (*PARTITIONS, 'reserved_holdout')}
    if set(manifest['files']) != expected_files:
        raise ValueError('Unexpected source files')
    # Holdout bytes are hashed only; never decoded, parsed, displayed or scored.
    paths = {n: SOURCE/n for n in expected_files}
    paths.update({n: SOURCE/n for n in ('manifest.json', 'annotation-rules.txt', 'reviewer-a.json', 'reviewer-b.json')})
    paths['reconciliation_script'] = Path(__file__)
    hashes = {n: digest(p) for n, p in paths.items()}
    for name, expected in manifest['files'].items():
        if hashes[name] != expected:
            raise ValueError('Queue hash mismatch: ' + name)
    if hashes['annotation-rules.txt'] != manifest['rules_sha256']:
        raise ValueError('Rules changed')
    queue = []
    for part in PARTITIONS:
        rows = json.loads(paths[part+'.json'].read_text(encoding='utf8'))
        if len(rows) != manifest['counts'][part] or any(r['partition'] != part for r in rows):
            raise ValueError('Queue count or partition changed')
        queue.extend(rows)
    reviews = [json.loads(paths[n].read_text(encoding='utf8')) for n in ('reviewer-a.json', 'reviewer-b.json')]
    result, inputs = reconcile(queue, *reviews)
    result.update(source_sha256=hashes, status='reviewed_current_and_original_only',
                  reviewer_kind='two agents, not human ground truth',
                  wrong_history_status='pending_separate_pair_review',
                  holdout_status='bytes_hash_checked_only_not_parsed_or_scored',
                  target='missing conversational referent or constraint, not factual truth or complete answerability')
    if any(digest(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Sources changed during reconciliation')
    out.mkdir()
    for part in PARTITIONS:
        path = out/(part+'_inputs.json')
        path.write_text(json.dumps([r for r in inputs if r['partition'] == part], ensure_ascii=False, indent=2), encoding='utf8')
    result['output_sha256'] = {p.name: digest(p) for p in out.glob('*_inputs.json')}
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
