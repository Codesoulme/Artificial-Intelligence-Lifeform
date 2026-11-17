"""Rebuild paired features and verify all saved heads in a separate process."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'runs/open-base-deps')]
import torch
from spora.research_question_context_pair import pair_features, question_rows
from spora.research_evidence_calibration import calibrate
from scripts.train_oasst_context_crossfit_20261118 import sha

OUT = ROOT/'runs/oasst-context-pair-ablation-20261118'
SOURCE = ROOT/'runs/oasst-context-transfer-20261118'


def main():
    target = OUT/'reload-audit.json'
    if target.exists():
        raise FileExistsError(target)
    result = json.loads((OUT/'result.json').read_text(encoding='utf8'))
    for filename, digest in result['artifacts'].items():
        if sha(OUT/filename) != digest:
            raise ValueError('Export hash mismatch')
    manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf8'))
    for key, path in [('inputs', SOURCE/'inputs.json'), ('features', SOURCE/'features.pt'),
                      ('pair_code', ROOT/'spora/research_question_context_pair.py')]:
        if sha(path) != manifest['source_sha256'][key]:
            raise ValueError('Feature source changed')
    rows = manifest['rows']
    original = json.loads((SOURCE/'inputs.json').read_text(encoding='utf8'))
    index = {(r['case'], r['variant']): i for i, r in enumerate(original)}
    indices = [index[r['case'], r['variant']] for r in rows]
    base = torch.load(SOURCE/'features.pt', map_location='cpu', weights_only=True)['last'][indices]
    q = base[question_rows(rows)]
    heads = torch.load(OUT/'heads.pt', map_location='cpu', weights_only=True)
    records = []
    for run in result['runs']:
        x = pair_features(base, q, run['pooling'])
        maximum = 0.
        decisions = [None]*len(rows)
        for fold, expected in zip(manifest['folds'], run['folds']):
            if fold['fold'] != expected['fold']:
                raise ValueError('Fold alignment mismatch')
            key = f'{run["pooling"]}-{run["seed"]}-{fold["fold"]}'
            head = torch.nn.Linear(x.shape[-1], 1)
            head.load_state_dict(heads[key]['state'], strict=True)
            with torch.no_grad():
                scores = head(x).squeeze(-1).tolist()
            for split in ('train', 'calibration', 'test'):
                actual = [scores[i] for i in fold[split]]
                if len(actual) != len(expected['scores'][split]):
                    raise ValueError('Score count mismatch')
                error = max(abs(a-b) for a, b in zip(actual, expected['scores'][split]))
                maximum = max(maximum, error)
                if not error < 1e-5:
                    raise ValueError('Scores changed')
            ci = fold['calibration']
            policies, _ = calibrate([scores[i] for i in ci], [rows[i]['need'] for i in ci])
            if abs(policies['balanced'] - heads[key]['threshold']) >= 1e-5:
                raise ValueError('Calibration changed')
            for i in fold['test']:
                decisions[i] = int(scores[i]>=heads[key]['threshold'])
        if decisions != run['predictions']:
            raise ValueError('Predictions changed')
        records.append({'mode': run['pooling'], 'seed': run['seed'], 'max_score_error': maximum,
                        'parameters_per_head': x.shape[-1]+1})
    target.write_text(json.dumps({'verified': records, 'scope': 'Saved-feature classifier reload only'}, indent=2), encoding='utf8')
    print(json.dumps(records))


if __name__ == '__main__':
    main()
