"""Retrospective conjunction of two offline fingerprint-veto audits."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = [ROOT / 'runs/deepseek-spora-fingerprint-route-audit-v2-20261111.json',
          ROOT / 'runs/deepseek-spora-fingerprint-fresh-audit-20261111.json']
OUTPUT = ROOT / 'runs/deepseek-spora-fingerprint-conjunction-20261111.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = []
    for path in INPUTS:
        payload = path.read_bytes()
        source = json.loads(payload)
        rows = source['rows']
        known = [r for r in rows if r['group'] == 'known']
        unknown = [r for r in rows if r['group'] == 'unknown']
        summary = {'known_total': len(known),
                   'known_accepted': sum(r['gated_accepted'] and r['candidate_gated_accepted'] for r in known),
                   'unknown_total': len(unknown),
                   'unknown_accepted': sum(r['gated_accepted'] and r['candidate_gated_accepted'] for r in unknown)}
        result.append({'source': path.name, 'source_sha256': hashlib.sha256(payload).hexdigest(),
                       'summary': summary})
    report = {'status': 'retrospective_development_analysis', 'results': result,
              'limitations': ['Conjunction was chosen after seeing both component results.',
                              'Both locators share a model and can make correlated errors.',
                              'Tiny authored samples are not independent safety evaluations.',
                              'This analysis does not evaluate factual answer correctness or perform inference.',
                              'The proposed conjunction is not integrated into runtime.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
