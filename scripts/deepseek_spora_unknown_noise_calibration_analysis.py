"""Compare midpoint thresholds on clean and noise-augmented calibration rows."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/deepseek-spora-continuous-unknown-calibration-20261112.json'
OUTPUT = ROOT / 'runs/deepseek-spora-unknown-noise-calibration-analysis-20261112.json'


def select_threshold(rows):
    if not rows or not any(r['known'] for r in rows) or all(r['known'] for r in rows):
        raise ValueError('Calibration requires known and unwritten examples')
    values = sorted(set(r['norm'] for r in rows))
    thresholds = [values[0] - 1.0] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [values[-1] + 1.0]
    def score(t):
        errors = sum((r['norm'] >= t) != r['known'] for r in rows)
        margin = min(abs(r['norm'] - t) for r in rows)
        return errors, -margin, t
    return min(thresholds, key=score)


def evaluate(rows, threshold):
    return {'known': sum(r['known'] for r in rows), 'unwritten': sum(not r['known'] for r in rows),
            'false_known': sum(not r['known'] and r['norm'] >= threshold for r in rows),
            'false_unknown': sum(r['known'] and r['norm'] < threshold for r in rows)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    rows = source['rows']
    results = []
    for seed in sorted({r['seed'] for r in rows}):
        for site in sorted({r['site'] for r in rows}):
            group = [r for r in rows if r['seed'] == seed and r['site'] == site]
            for policy, levels in (('clean_midpoint', (0.0,)), ('noise_augmented', (0.0, 0.1, 0.3))):
                train = [r for r in group if r['key'] % 2 == 0 and r['noise_level'] in levels]
                threshold = select_threshold(train)
                for level in (0.0, 0.1, 0.3):
                    test = [r for r in group if r['key'] % 2 == 1 and r['noise_level'] == level]
                    if len(train) != 32 * len(levels) or len(test) != 32:
                        raise ValueError('Unexpected calibration split size')
                    results.append({'seed': seed, 'site': site, 'policy': policy, 'noise_level': level,
                                    'threshold': threshold, 'calibration': evaluate(train, threshold),
                                    'test': evaluate(test, threshold)})
    output = {'source': str(SOURCE), 'checkpoint_sha256': source['checkpoint_sha256'],
              'base_sha256': source['base_sha256'], 'rows': results,
              'limitations': ['Retrospective analysis of already inspected development data.',
                              'Same matrix, synthetic noise directions and text distribution across split.',
                              'Unwritten in this probe does not guarantee absent from the original individual.',
                              'Threshold classifies norm only, not factual correctness or decoding quality.',
                              'No runtime threshold or model changes.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
