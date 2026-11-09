"""Compare a learned gate with the legacy P26 current-query recognizer on recorded cases."""
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
from spora.finite_question_guard import valid_finite_question_form
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
        data = json.load(stream)
    if data.get('status') != 'handwritten_adversarial_development_only':
        raise ValueError('Unexpected source report')
    torch.set_num_threads(2)
    reader, state, _ = load(QUERY_PATH, 'cpu')
    before = frozen_digest(reader, state)
    rows = []
    for row in data['rows']:
        try:
            parsed = reader.chinese_query.parse([row['question']])[0]
            p26_accept = parsed['accepted'] and parsed['kind'] == 'current'
            error = None
        except ValueError as exc:
            parsed, p26_accept, error = None, False, str(exc)
        p26_accept = bool(p26_accept and valid_finite_question_form(row['question']))
        rows.append({'question': row['question'], 'label': row['label'], 'category': row['category'],
                     'gate_accept': row['accepted'], 'p26_accept': p26_accept,
                     'both_accept': bool(p26_accept and row['accepted']), 'p26_parsed': parsed, 'error': error})
    if sha(QUERY_PATH) != QUERY_SHA or sha(source) != source_sha or frozen_digest(reader, state) != before:
        raise ValueError('Protected inputs changed')
    summary = {}
    for field in ('gate_accept', 'p26_accept', 'both_accept'):
        summary[field] = {'correct': sum(row[field] == bool(row['label']) for row in rows),
                          'false_accept': sum(row[field] and not row['label'] for row in rows),
                          'false_reject': sum(not row[field] and bool(row['label']) for row in rows)}
    report = {'status': 'p26_router_crosscheck_development_only', 'source_sha256': source_sha,
              'p26_sha256': QUERY_SHA, 'summary': summary, 'rows': rows,
              'limitations': ['Handwritten development questions; P26 training exposure is not fully known.',
                              'Neither gate nor P26 checks whether a supported object has a memory.',
                              'Combining recognizers can reduce recall and does not prove safety.',
                              'No model answer or memory write is performed.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
