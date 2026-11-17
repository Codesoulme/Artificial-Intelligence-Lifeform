"""Reload audit plus calibration-only control on the identical purged folds."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'runs/open-base-deps')]
import torch
from spora.research_evidence_calibration import calibrate
from scripts.train_oasst_context_crossfit_20261118 import metrics

OUT = ROOT/'runs/oasst-context-crossfit-20261118'
SOURCE = ROOT/'runs/oasst-context-transfer-20261118'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = OUT/'reload-and-calibration-control.json'
    if target.exists():
        raise FileExistsError(target)
    result = json.loads((OUT/'result.json').read_text(encoding='utf8'))
    for filename, expected in result['artifacts'].items():
        if sha(OUT/filename) != expected:
            raise ValueError('Export mismatch')
    manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf8'))
    for key, filename in [('features', 'features.pt'), ('result', 'result.json'), ('inputs', 'inputs.json')]:
        if sha(SOURCE/filename) != manifest['source_sha256'][key]:
            raise ValueError('Protected source mismatch')
    rows = manifest['rows']
    original = json.loads((SOURCE/'inputs.json').read_text(encoding='utf8'))
    lookup = {(r['case'], r['variant']): i for i, r in enumerate(original)}
    indices = [lookup[r['case'], r['variant']] for r in rows]
    features = torch.load(SOURCE/'features.pt', map_location='cpu', weights_only=True)
    heads = torch.load(OUT/'heads.pt', map_location='cpu', weights_only=True)
    parent = json.loads((SOURCE/'result.json').read_text(encoding='utf8'))
    audits = []
    for run in result['runs']:
        x = features[run['pooling']][indices]
        fixed = next(r for r in parent['results'] if (r['pooling'], r['seed'])==(run['pooling'], run['seed']))
        fixed_scores = [fixed['scores'][i] for i in indices]
        control_predictions, reloaded = [None]*len(rows), [None]*len(rows)
        details, max_error = [], 0.
        for fold, record in zip(manifest['folds'], run['folds']):
            if fold['fold'] != record['fold']:
                raise ValueError('Fold order mismatch')
            saved = heads[f'{run["pooling"]}-{run["seed"]}-{fold["fold"]}']
            head = torch.nn.Linear(x.shape[-1], 1)
            head.load_state_dict(saved['state'], strict=True)
            with torch.no_grad():
                scores = head(x).squeeze(-1).tolist()
            for split in ('train', 'calibration', 'test'):
                expected = record['scores'][split]
                actual = [scores[i] for i in fold[split]]
                if len(expected) != len(actual):
                    raise ValueError('Score count mismatch')
                error = max(abs(a-b) for a, b in zip(actual, expected))
                max_error = max(max_error, error)
                if error > 1e-5:
                    raise ValueError('Reloaded scores differ')
            ci = fold['calibration']
            thresholds, _ = calibrate([scores[i] for i in ci], [rows[i]['need'] for i in ci])
            if abs(thresholds['balanced']-saved['threshold']) > 1e-5:
                raise ValueError('Reloaded calibration differs')
            control_thresholds, _ = calibrate([fixed_scores[i] for i in ci], [rows[i]['need'] for i in ci])
            for i in fold['test']:
                reloaded[i] = int(scores[i]>=saved['threshold'])
                control_predictions[i] = int(fixed_scores[i]>=control_thresholds['balanced'])
            details.append({'fold': fold['fold'], 'control_threshold': control_thresholds['balanced']})
        if reloaded != run['predictions']:
            raise ValueError('Reloaded decisions differ')
        audits.append({'pooling': run['pooling'], 'seed': run['seed'], 'max_abs_score_error': max_error,
                       'calibration_only_folds': details, 'calibration_only_predictions': control_predictions,
                       'calibration_only_metrics': metrics(rows, control_predictions)})
    data = {'audit': audits, 'scope': 'Head reload on saved features; no memory or GPU re-extraction test.',
            'control': 'Retrospective diagnostic, identical folds, no test labels used to fit thresholds.'}
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(audits))


if __name__ == '__main__':
    main()
