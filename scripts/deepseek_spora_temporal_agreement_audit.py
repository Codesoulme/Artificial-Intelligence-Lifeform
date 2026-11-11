"""Offline retrospective temporal-head agreement audit; no runtime changes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/deepseek-spora-temporal-gate-train-20261111/result.json'
CANDIDATE = ROOT / 'runs/deepseek-spora-temporal-gate-v3-20261112/result.json'
FRESH = ROOT / 'runs/deepseek-spora-temporal-v3-holdout-20261112.json'
OUTPUT = ROOT / 'runs/deepseek-spora-temporal-agreement-audit-20261112.json'
ABSTAIN = 3


def summarize(rows, mode):
    out = []
    for row in rows:
        baseline, candidate = row['baseline'], row['candidate']
        prediction = (baseline if mode == 'baseline' else candidate if mode == 'candidate'
                      else baseline if baseline == candidate else ABSTAIN)
        out.append({**row, 'prediction': prediction})
    return {'total': len(out), 'correct': sum(r['prediction'] == r['label'] for r in out),
            'wrong_read': sum(r['prediction'] != ABSTAIN and r['prediction'] != r['label'] for r in out),
            'false_abstain': sum(r['prediction'] == ABSTAIN and r['label'] != ABSTAIN for r in out),
            'read': sum(r['prediction'] != ABSTAIN for r in out)}


def unique_by_question(rows):
    result = {}
    for row in rows:
        question = row['question']
        if question in result:
            raise ValueError('Duplicate question in source')
        result[question] = row
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base, candidate, fresh = [json.loads(path.read_text(encoding='utf-8')) for path in (BASE, CANDIDATE, FRESH)]
    if base['temporal_sha256'] == candidate['temporal_sha256']:
        raise ValueError('Identical candidate and baseline')
    if (base['feature_variant_sha256'] != candidate['feature_variant_sha256']
            or fresh['candidate_sha256'] != candidate['temporal_sha256']):
        raise ValueError('Source provenance mismatch')
    left, right = unique_by_question(base['rows']), unique_by_question(candidate['rows'])
    shared = sorted(left.keys() & right.keys())
    changed_labels = [q for q in shared if left[q]['label'] != right[q]['label']]
    old_rows = [{'question': q, 'label': left[q]['label'], 'baseline': left[q]['predicted'],
                 'candidate': right[q]['predicted']} for q in shared if q not in changed_labels]
    fresh_rows = [{'question': r['question'], 'label': r['label'], 'baseline': r['baseline'],
                   'candidate': r['candidate']} for r in fresh['rows']]
    if any(r[k] not in (0, 1, 2, 3) for r in old_rows + fresh_rows for k in ('label', 'baseline', 'candidate')):
        raise ValueError('Invalid time class')
    result = {'status': 'retrospective_temporal_agreement_only', 'excluded_changed_labels': changed_labels,
              'source_sha256': {'baseline': base['temporal_sha256'], 'candidate': candidate['temporal_sha256']},
              'results': {name: {mode: summarize(rows, mode) for mode in ('baseline', 'candidate', 'agreement')}
                          for name, rows in (('old_development', old_rows), ('new_development', fresh_rows))},
              'disagreements': {name: [r for r in rows if r['baseline'] != r['candidate']]
                                for name, rows in (('old_development', old_rows), ('new_development', fresh_rows))},
              'limitations': ['Post-hoc analysis on previously examined small authored development sets.',
                              'Agreement is not calibrated and models can make the same mistake.',
                              'No new extraction, integrated runtime, memory selection or answer generation.',
                              'Temporal ambiguity may require conversational context and different labels.',
                              'No runtime setting or real individual changed.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result['results'], ensure_ascii=False))


if __name__ == '__main__':
    main()
