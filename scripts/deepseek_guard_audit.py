"""Audit a finite-format guard against a previously recorded intent-gate result."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.open_base_baseline import sha
from spora.finite_question_guard import valid_finite_question_form


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    runs = (ROOT / 'runs').resolve()
    if not source.is_relative_to(runs) or not output.is_relative_to(runs) or output.exists():
        raise ValueError('Expected source and new output under runs')
    digest = sha(source)
    with source.open(encoding='utf-8') as stream:
        recorded = json.load(stream)
    if recorded.get('status') != 'finite_intent_gate_development_only':
        raise ValueError('Unexpected source report')
    results = {}
    for split in ('held', 'boundary_development'):
        rows = []
        for row in recorded['results'][split]['rows']:
            form_ok = valid_finite_question_form(row['question'])
            accepted = row['accepted'] and form_ok
            rows.append({'question': row['question'], 'category': row['category'], 'label': row['label'],
                         'gate_accepted': row['accepted'], 'form_accepted': form_ok, 'accepted': accepted})
        categories = {}
        for category in sorted({row['category'] for row in rows}):
            group = [row for row in rows if row['category'] == category]
            categories[category] = {'examples': len(group), 'accepted': sum(row['accepted'] for row in group),
                                    'correct': sum(row['accepted'] == bool(row['label']) for row in group)}
        results[split] = {'examples': len(rows), 'correct': sum(row['accepted'] == bool(row['label']) for row in rows),
                          'categories': categories, 'rows': rows}
    if sha(source) != digest:
        raise ValueError('Recorded gate result changed')
    report = {'status': 'offline_finite_format_guard_audit', 'source_sha256': digest, 'results': results,
              'limitations': ['Audits recorded gate outputs; no integrated model generation is tested.',
                              'Supports exactly one canonical Arabic entity identifier in 0..63.',
                              'Format validation does not establish request intent or memory existence.',
                              'Synthetic templates and manually inspected boundary cases are not final acceptance.',
                              'A supported-looking but ambiguous or malicious question can still pass.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({name: {k: v for k, v in result.items() if k != 'rows'} for name, result in results.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
