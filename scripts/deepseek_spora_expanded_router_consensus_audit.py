"""Combine saved development outputs to expose correlated name truncation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_long_memory_generate_parity as p

OLD = ROOT / 'runs/deepseek-spora-original-layer-name-router-v3-20261114'
NEW = ROOT / 'runs/deepseek-spora-original-layer-name-router-expanded-20261114'
MULTI = ROOT / 'runs/deepseek-spora-original-multi-span-pilot-20261114'
GATES = ROOT / 'runs/deepseek-spora-original-layer-request-gates-v2-20261114'
CAL = ROOT / 'runs/deepseek-spora-original-layer-joint-calibration-20261114.json'
OUTPUT = ROOT / 'runs/deepseek-spora-expanded-router-consensus-audit-20261114.json'


def agree_with_guard(router_row, spans):
    if not router_row.get('guarded_accepted') or not isinstance(spans, list) or len(spans) != 1:
        return False
    guard = router_row.get('guard_span')
    return isinstance(guard, dict) and isinstance(spans[0], str) and bool(spans[0]) and spans[0] == guard.get('selected_text')


def load_result(folder, name, key):
    report_path, artifact_path = folder / 'result.json', folder / name
    report = json.loads(report_path.read_text(encoding='utf-8'))
    digest = p.sha(artifact_path)
    if report[key] != digest:
        raise ValueError('Artifact checksum mismatch')
    return report, [(report_path, p.sha(report_path)), (artifact_path, digest)]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    old, old_protected = load_result(OLD, 'router.pt', 'router_sha256')
    new, new_protected = load_result(NEW, 'router.pt', 'router_sha256')
    multi, multi_protected = load_result(MULTI, 'locator.pt', 'locator_sha256')
    gates, gates_protected = load_result(GATES, 'gates.pt', 'gates_sha256')
    cal_sha = p.sha(CAL)
    cal = json.loads(CAL.read_text(encoding='utf-8'))
    if (any(report['base_sha256'] != p.WEIGHT_SHA for report in (old, new, multi, gates))
            or old['checkpoint_sha256'] != new['checkpoint_sha256']
            or new['checkpoint_sha256'] != multi['checkpoint_sha256']
            or gates['checkpoint_sha256'] != new['checkpoint_sha256']
            or old['basis_sha256'] != new['basis_sha256']
            or cal['gates_sha256'] != gates['gates_sha256'] or cal['source_sha256'] != p.sha(GATES / 'result.json')):
        raise ValueError('Research source provenance mismatch')
    old_by_question = {row['question']: row for row in old['rows']}
    multi_by_question = {row['question']: row for row in multi['rows'] if row['split'] == 'prior_name_development'}
    gate_by_question = {row['question']: row for row in gates['rows'] if row['split'] == 'prior_name_development'}
    new_by_question = {row['question']: row for row in new['rows']}
    if (set(old_by_question) != set(multi_by_question) or set(old_by_question) != set(gate_by_question)
            or not set(old_by_question) <= set(new_by_question)):
        raise ValueError('Development question sets do not match')
    rows = []
    for question, previous in old_by_question.items():
        candidate, second, gate = new_by_question[question], multi_by_question[question], gate_by_question[question]
        if candidate['group'] != previous['group'] or candidate['expected_key'] != previous['expected_key']:
            raise ValueError('Development labels differ')
        gate_pass = (gate['single_score'] >= cal['thresholds']['single_joint']
                     and gate['read_score'] >= cal['thresholds']['read'])
        agreement = agree_with_guard(candidate, second['spans'])
        rows.append({'question': question, 'group': candidate['group'], 'expected_key': candidate['expected_key'],
                     'old_guarded_accepted': previous['guarded_accepted'],
                     'old_correct': previous['guarded_correct_key'],
                     'new_guarded_accepted': candidate['guarded_accepted'],
                     'new_correct': candidate['guarded_correct_key'],
                     'new_span': candidate.get('guard_span'), 'multi_spans': second['spans'],
                     'gates_accepted': gate_pass, 'agreement': agreement,
                     'combined_accepted': gate_pass and agreement,
                     'combined_correct': gate_pass and agreement and candidate['guarded_correct_key']})
    summary = {}
    for group in sorted({row['group'] for row in rows}):
        subset = [row for row in rows if row['group'] == group]
        summary[group] = {'cases': len(subset), **{field: sum(r[field] for r in subset) for field in
                          ('old_guarded_accepted', 'old_correct', 'new_guarded_accepted', 'new_correct',
                           'combined_accepted', 'combined_correct')}}
    for path, digest in old_protected + new_protected + multi_protected + gates_protected + [(CAL, cal_sha)]:
        if p.sha(path) != digest:
            raise ValueError('Source artifact changed')
    report = {'summary': summary, 'rows': rows, 'old_router_sha256': old['router_sha256'],
              'new_router_sha256': new['router_sha256'], 'multi_sha256': multi['locator_sha256'],
              'gates_sha256': gates['gates_sha256'], 'calibration_sha256': cal_sha,
              'old_threshold': old['threshold'], 'new_threshold': new['threshold'],
              'limitations': ['Offline combination of separately recorded outputs; no live memory was read.',
                              'Previously inspected artificial development cases and shared base features.',
                              'Same selected text can arise from common truncation and is not proof of identity.',
                              'Joint threshold is a previously analyzed development choice, not deployed.',
                              'Only shared earlier questions; added held templates not evaluated by all components.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
