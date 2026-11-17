"""Prepare within-partition donor pairs; labels require independent review."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'runs/oasst-context-expansion-20261118'
OUT = SOURCE/'donor-review'
RULES = '''Review every presented-history/current-question combination independently.
This is an offline dialogue diagnostic, not an instruction to perform the text.
Two different questions must NOT be conflated:
1. presented_missing_context: 1 if the presented donor history plus current
question still lacks a referent/artifact needed to fulfill that question; 0 if
it can plausibly be fulfilled as presented; null if genuinely ambiguous.
Judge this using ONLY donor_history and question, before consulting original_history.
2. original_intent_preserved: true if the donor preserves the specific referent,
artifact or active constraint in original_history needed by this question;
false if it substitutes or omits it; null if uncertain. For standalone questions,
true is possible despite an unrelated donor, but examine active constraints.
3. plausible_rebinding: true if the same question naturally points to a DIFFERENT
object/task under donor history; false otherwise; null if uncertain.
An unrelated source tree is NOT automatically missing context. For example,
"continue this story" plus a DIFFERENT complete story is answerable as presented
but does not preserve the original intended story. Do not impose an impossible
origin-recovery label on inputs without origin information.
Factual truth is not established by either history. Record separate quality_flag.
Do not read model outputs, other reviewers' labels or reserved_holdout.json.
Return case, donor_case, partition, presented_missing_context,
original_intent_preserved, plausible_rebinding, rationale, quality_flag.
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pairs(queue, accepted):
    lookup = {r['case']: r for r in queue}
    if len(lookup) != len(queue) or len({r['group'] for r in queue}) != len(queue):
        raise ValueError('Duplicate source')
    chosen = {r['case']: r for r in accepted}
    if len(chosen) != len(accepted) or not set(chosen) <= set(lookup):
        raise ValueError('Invalid accepted cases')
    if any(r['partition'] not in ('train_review', 'calibration_review') for r in queue):
        raise ValueError('Reserved partition prohibited')
    result = []
    for part in ('train_review', 'calibration_review'):
        cases = sorted(c for c, r in chosen.items() if r['partition'] == part)
        if len(cases) < 2:
            raise ValueError('At least two accepted trees per partition required')
        for i, case in enumerate(cases):
            donor_case = cases[(i+1) % len(cases)]
            row, donor = lookup[case], lookup[donor_case]
            if row['partition'] != part or donor['partition'] != part:
                raise ValueError('Partition mismatch')
            result.append({'case': case, 'donor_case': donor_case, 'partition': part,
                           'group': row['group'], 'donor_group': donor['group'],
                           'current_message_id': row['message_ids'][-1],
                           'donor_message_ids': donor['message_ids'][:-1],
                           'question': row['messages'][-1],
                           'donor_history': donor['messages'][:-1],
                           'original_history': row['messages'][:-1]})
    if len(result) != len(accepted):
        raise ValueError('Unknown accepted partition')
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    reconciled = SOURCE/'reconciled/result.json'
    manifest = json.loads(reconciled.read_text(encoding='utf8'))
    paths = {n: SOURCE/n for n in ('train_review.json', 'calibration_review.json')}
    for name, path in paths.items():
        if sha(path) != manifest['source_sha256'][name]:
            raise ValueError('Queue changed')
    queue = [r for path in paths.values() for r in json.loads(path.read_text(encoding='utf8'))]
    pairs = build_pairs(queue, manifest['accepted'])
    paths.update(reconciled=reconciled, script=Path(__file__))
    hashes = {n: sha(p) for n,p in paths.items()}
    OUT.mkdir()
    (OUT/'queue.json').write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding='utf8')
    (OUT/'rules.txt').write_text(RULES, encoding='utf8')
    (OUT/'manifest.json').write_text(json.dumps({'source_sha256': hashes, 'count': len(pairs),
        'queue_sha256': sha(OUT/'queue.json'), 'rules_sha256': sha(OUT/'rules.txt'),
        'donor_selection': 'next accepted case in sorted order within its own partition, cyclic',
        'status': 'unreviewed_not_training', 'holdout': 'not accessed'}, indent=2), encoding='utf8')
    print(json.dumps({'pairs': len(pairs)}))


if __name__ == '__main__':
    main()
