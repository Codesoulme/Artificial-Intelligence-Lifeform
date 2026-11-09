"""Offline audit of adding the finite P26 verifier to recorded composite routes."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch

from scripts.open_base_addressed_alignment import QUERY_PATH, QUERY_SHA
from scripts.open_base_baseline import sha
from spora.growing import load
from spora.language_training import frozen_digest


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    runs = (ROOT / 'runs').resolve()
    if not source.is_relative_to(runs) or not output.is_relative_to(runs) or output.exists():
        raise ValueError('Expected source and new output under runs')
    source_sha = sha(source)
    if sha(QUERY_PATH) != QUERY_SHA:
        raise ValueError('P26 checkpoint mismatch')
    with source.open(encoding='utf-8') as stream:
        report = json.load(stream)
    if report.get('status') != 'read_only_composite_development_only':
        raise ValueError('Unexpected composite report')
    torch.set_num_threads(2)
    reader, state, _ = load(QUERY_PATH, 'cpu')
    before = frozen_digest(reader, state)
    rows = []
    for row in report['rows']:
        try:
            parsed = reader.chinese_query.parse([row['question']])[0]
            supported = (parsed['accepted'] and parsed['kind'] == 'current'
                         and row['predicted_addresses'] == [parsed['entity'], parsed['entity']])
        except ValueError:
            parsed, supported = None, False
        route = row['route'] if supported else 'defer'
        correct = (route == row['expected_route'] and
                   (not row['label'] or row['predicted_addresses'] == [row['expected_key']] * 2))
        rows.append({'category': row['category'], 'question': row['question'], 'expected_route': row['expected_route'],
                     'prior_route': row['route'], 'audited_route': route, 'route_correct': correct, 'p26': parsed})
    if sha(QUERY_PATH) != QUERY_SHA or sha(source) != source_sha or frozen_digest(reader, state) != before:
        raise ValueError('Protected input changed')
    summary = {}
    for group in ('known', 'unknown', 'stress_current', 'stress_other'):
        selected = [r for r in rows if (r['category'] == group if group != 'stress_other' else
                                         r['category'].startswith('stress_') and r['category'] != 'stress_current')]
        summary[group] = {'examples': len(selected), 'route_correct': sum(r['route_correct'] for r in selected),
                          'deferred': sum(r['audited_route'] == 'defer' for r in selected)}
    result = {'status': 'offline_p26_composite_audit_only', 'source_sha256': source_sha, 'p26_sha256': QUERY_SHA,
              'summary': summary, 'rows': rows,
              'limitations': ['Offline audit, not an integrated runtime or independent test.',
                              'P26 has a restricted character domain and can reject valid requests.',
                              'Recorded composite data and stress questions were previously inspected.',
                              'No answer generation, live write, or broad safety claim.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
