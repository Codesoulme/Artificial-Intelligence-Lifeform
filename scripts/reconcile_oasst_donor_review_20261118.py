"""Keep observable sufficiency separate from preservation of hidden source intent."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_oasst_donor_review_20261118 import OUT, SOURCE, sha


def validate(pairs, review):
    lookup = {r['case']: r for r in pairs}
    found = {}
    if not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip():
        raise ValueError('Reviewer required')
    for item in review['reviews']:
        case = item['case']
        if type(case) is not int or case not in lookup or case in found:
            raise ValueError('Unknown or duplicate case')
        pair = lookup[case]
        if (type(item['donor_case']) is not int or item['donor_case'] != pair['donor_case']
                or item['partition'] != pair['partition']):
            raise ValueError('Wrong donor or partition')
        value = item['presented_missing_context']
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ValueError('Bad missing-context label')
        for field in ('original_intent_preserved', 'plausible_rebinding'):
            value = item[field]
            if value is not None and type(value) is not bool:
                raise ValueError('Bad boolean label')
        if not isinstance(item.get('rationale'), str) or not item['rationale'].strip():
            raise ValueError('Missing rationale')
        if not isinstance(item.get('quality_flag'), str):
            raise ValueError('Missing quality flag')
        found[case] = item
    if set(found) != set(lookup):
        raise ValueError('Incomplete coverage')
    return found


def reconcile(pairs, review_c, review_d, originals):
    if len({p['case'] for p in pairs}) != len(pairs):
        raise ValueError('Duplicate pairs')
    by_case = {r['case']: r for r in originals}
    if len(by_case) != len(originals) or set(by_case) != {r['case'] for r in pairs}:
        raise ValueError('Original label coverage mismatch')
    for p in pairs:
        if (p['partition'] not in ('train_review', 'calibration_review')
                or p['partition'] != by_case[p['case']]['partition']
                or p['donor_case'] not in by_case
                or by_case[p['donor_case']]['partition'] != p['partition']
                or p['group'] == p['donor_group'] or p['case'] == p['donor_case']):
            raise ValueError('Invalid donor boundary')
    c,d = validate(pairs, review_c), validate(pairs, review_d)
    fields = ('presented_missing_context', 'original_intent_preserved', 'plausible_rebinding')
    accepted, excluded, rows = [], [], []
    for p in sorted(pairs, key=lambda p:p['case']):
        case = p['case']
        labels = tuple(c[case][k] for k in fields)
        other = tuple(d[case][k] for k in fields)
        if labels != other or any(v is None for v in labels):
            excluded.append({'case': case, 'reason': 'disagreement' if labels != other else 'ambiguous',
                             'reviewer_c': c[case], 'reviewer_d': d[case]})
            continue
        row = {'case': case, 'donor_case': p['donor_case'], 'partition': p['partition'],
               **dict(zip(fields, labels)), 'original_context_need': by_case[case]['context_need'],
               'quality_flags': [c[case]['quality_flag'], d[case]['quality_flag']]}
        accepted.append(row)
        rows.append({**row, 'group': p['group'], 'donor_group': p['donor_group'],
                     'messages': p['donor_history']+[p['question']],
                     'current_message_id': p['current_message_id'],
                     'history_message_ids': p['donor_message_ids'],
                     'variant': 'donor_history', 'missing_context': labels[0]})
    counts = {}
    for part in ('train_review', 'calibration_review'):
        subset = [r for r in accepted if r['partition'] == part]
        counts[part] = {'reviewed': sum(p['partition']==part for p in pairs), 'accepted': len(subset),
                        'missing_context': sum(r['presented_missing_context'] for r in subset),
                        'source_intent_lost': sum(not r['original_intent_preserved'] for r in subset),
                        'plausible_rebinding': sum(r['plausible_rebinding'] for r in subset),
                        'naive_target_disagrees': sum(r['presented_missing_context'] != r['original_context_need'] for r in subset),
                        'answerable_but_intent_lost': sum(r['presented_missing_context']==0 and not r['original_intent_preserved'] for r in subset)}
    return {'accepted': accepted, 'excluded': excluded, 'counts': counts}, rows


def main():
    out = OUT/'reconciled'
    if out.exists():
        raise FileExistsError(out)
    paths = {n:OUT/n for n in ('manifest.json','queue.json','rules.txt','reviewer-c.json','reviewer-d.json')}
    paths.update(original_labels=SOURCE/'reconciled/result.json', script=Path(__file__))
    hashes = {n:sha(p) for n,p in paths.items()}
    manifest = json.loads(paths['manifest.json'].read_text(encoding='utf8'))
    for name,key in (('queue.json','queue_sha256'),('rules.txt','rules_sha256')):
        if hashes[name] != manifest[key]:
            raise ValueError('Source changed: '+name)
    if hashes['original_labels'] != manifest['source_sha256']['reconciled']:
        raise ValueError('Original reviews changed')
    pairs = json.loads(paths['queue.json'].read_text(encoding='utf8'))
    c,d = [json.loads(paths[n].read_text(encoding='utf8')) for n in ('reviewer-c.json','reviewer-d.json')]
    originals = json.loads(paths['original_labels'].read_text(encoding='utf8'))['accepted']
    result, rows = reconcile(pairs,c,d,originals)
    if any(sha(p)!=hashes[n] for n,p in paths.items()):
        raise ValueError('Sources changed during review')
    out.mkdir()
    path = out/'donor_inputs.json'
    path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    result.update(source_sha256=hashes, output_sha256={path.name:sha(path)},
                  status='reviewed_pairs_not_deployment_evidence',
                  limits=['Two agent reviews, not independent human truth.',
                          'Do not treat missing-context classification as source identity authentication.',
                          'No reserved holdout was accessed.'])
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
