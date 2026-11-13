"""Offline joint-threshold analysis for already trained original-layer gates."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_spora_long_memory_generate_parity as p

SOURCE = ROOT / 'runs/deepseek-spora-original-layer-request-gates-v2-20261114'
OUTPUT = ROOT / 'runs/deepseek-spora-original-layer-joint-calibration-20261114.json'


def joint_single_threshold(rows, read_threshold):
    """Reject calibration non-single rows which could pass the read gate."""
    if not math.isfinite(read_threshold):
        raise ValueError('Finite read threshold required')
    eligible = []
    for row in rows:
        if row.get('split') != 'calibration':
            continue
        if row.get('single') not in (0, 1):
            raise ValueError('Binary single label required')
        single_score, read_score = float(row['single_score']), float(row['read_score'])
        if not math.isfinite(single_score) or not math.isfinite(read_score):
            raise ValueError('Finite calibration scores required')
        if row['single'] == 0 and read_score >= read_threshold:
            eligible.append(row)
    if not eligible:
        raise ValueError('No calibration non-single cases pass read gate')
    maximum = max(row['single_score'] for row in eligible)
    threshold = float(torch.nextafter(torch.tensor(maximum, dtype=torch.float32),
                                      torch.tensor(float('inf'), dtype=torch.float32)))
    return threshold, eligible


def summarize(rows, single_threshold, read_threshold):
    groups = {}
    for split in ('calibration', 'prior_name_development', 'new_challenge'):
        groups[split] = {}
        for group in sorted({r['group'] for r in rows if r['split'] == split}):
            subset = [r for r in rows if r['split'] == split and r['group'] == group]
            gate_pass = lambda r: r['single_score'] >= single_threshold and r['read_score'] >= read_threshold
            item = {'cases': len(subset), 'gates_accepted': sum(gate_pass(r) for r in subset)}
            if split == 'prior_name_development':
                item['combined_accepted'] = sum(gate_pass(r) and r['prior']['guarded_accepted'] for r in subset)
                item['combined_correct_key'] = sum(gate_pass(r) and r['prior']['guarded_correct_key'] for r in subset)
            groups[split][group] = item
    return groups


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report_path, gates_path = SOURCE / 'result.json', SOURCE / 'gates.pt'
    source_sha, gates_sha = p.sha(report_path), p.sha(gates_path)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report['gates_sha256'] != gates_sha or report['base_sha256'] != p.WEIGHT_SHA:
        raise ValueError('Source provenance mismatch')
    rows = report['rows']
    read_threshold = report['thresholds']['read']
    current = report['thresholds']['single']
    candidate, eligible = joint_single_threshold(rows, read_threshold)
    summaries = {'current': summarize(rows, current, read_threshold),
                 'joint': summarize(rows, candidate, read_threshold)}
    if p.sha(report_path) != source_sha or p.sha(gates_path) != gates_sha:
        raise ValueError('Source files changed')
    result = {'source_sha256': source_sha, 'gates_sha256': gates_sha,
              'thresholds': {'read': read_threshold, 'single_current': current, 'single_joint': candidate},
              'eligible_calibration_negatives': [{'question': r['question'], 'group': r['group'],
                                                   'single_score': r['single_score'], 'read_score': r['read_score']}
                                                  for r in eligible],
              'summary': summaries,
              'limitations': ['Data and candidate selected after inspecting earlier development outcomes.',
                              'Small hand-authored calibration sample provides no false-acceptance guarantee.',
                              'Offline composition with previously recorded identity results; not a live route.',
                              'Single-person and read scores can correlate and fail together.',
                              'No new threshold is installed into the default or experimental runtime.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['thresholds']), flush=True)
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
