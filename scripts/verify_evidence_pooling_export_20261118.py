"""Verify exported probe heads against recorded frozen features in a new process."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from spora.research_evidence_calibration import calibrate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    p = args.directory
    result = json.loads((p / 'result.json').read_text(encoding='utf8'))
    manifest = json.loads((p / 'manifest.json').read_text(encoding='utf8'))
    for filename in ('manifest.json', 'features.pt', 'heads.pt'):
        if hashlib.sha256((p / filename).read_bytes()).hexdigest() != result['artifacts'][filename]:
            raise ValueError('Artifact mismatch: ' + filename)
    features = torch.load(p / 'features.pt', map_location='cpu', weights_only=True)
    heads = torch.load(p / 'heads.pt', map_location='cpu', weights_only=True)
    if features['manifest_sha256'] != result['artifacts']['manifest.json']:
        raise ValueError('Feature manifest mismatch')
    a = len(manifest['splits']['train'])
    b = a + len(manifest['splits']['calibration'])
    checked = []
    for run in result['results']:
        key = f"{run['pooling']}-{run['seed']}"
        x = features['features'][run['pooling']]
        head = torch.nn.Linear(x.shape[-1], 1)
        head.load_state_dict(heads[key]['state'], strict=True)
        with torch.no_grad():
            actual = head(x).squeeze(-1).tolist()
        expected = [v for split in manifest['splits'] for v in run['scores'][split]]
        if len(actual) != len(expected):
            raise ValueError('Wrong number of rows')
        error = max(abs(i - j) for i, j in zip(actual, expected))
        if not error < 1e-5:
            raise ValueError('Reloaded logits differ: ' + key)
        policies, _ = calibrate(actual[a:b], [r['need'] for r in manifest['splits']['calibration']])
        for name, value in policies.items():
            if abs(value - run['thresholds'][name]) >= 1e-5 or value != heads[key]['thresholds'][name]:
                raise ValueError('Calibration mismatch: ' + key)
        checked.append({'head': key, 'max_abs_logit_error': error})
    audit = {'artifacts_verified': result['artifacts'], 'checked': checked,
             'scope': 'Saved classifier/features only; not a persistent organism or GPU re-extraction audit.'}
    with (p / 'reload-verification.json').open('x', encoding='utf8') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(checked))


if __name__ == '__main__':
    main()
