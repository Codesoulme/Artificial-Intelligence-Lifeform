"""Compare saved research reader parameters without loading or updating a model."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_local_baseline as base

START = ROOT / 'runs/deepseek-spora-hidden-bridge-distill-20261115/candidate.pt'
SPAN = ROOT / 'runs/deepseek-spora-multitoken-span-bridge-64-20261117'
CONTROL = ROOT / 'runs/deepseek-spora-multitoken-span-control-64-20261117'
OUT = ROOT / 'runs/deepseek-spora-multitoken-reader-update-audit-20261117.json'
KEYS = tuple(f'native_sites.{site}.{name}' for site in (0, 1)
             for name in ('read_out.weight', 'read_gate.weight', 'read_gate.bias'))


def compare_tensors(reference, current):
    if (reference.shape != current.shape or not reference.is_floating_point()
            or not current.is_floating_point() or not bool(torch.isfinite(reference).all())
            or not bool(torch.isfinite(current).all())):
        raise ValueError('Incompatible or nonfinite tensors')
    difference = current.float() - reference.float()
    return {'reference_dtype': str(reference.dtype), 'current_dtype': str(current.dtype),
            'reference_norm': float(reference.float().norm()), 'difference_norm': float(difference.norm()),
            'difference_max_abs': float(difference.abs().max()),
            'changed': int((difference != 0).sum()), 'total': difference.numel()}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {'start': START, 'span': SPAN/'candidate.pt', 'control': CONTROL/'candidate.pt',
             'span_report': SPAN/'result.json', 'control_report': CONTROL/'result.json'}
    hashes = {name: base.sha(path) for name, path in paths.items()}
    span_report = json.loads(paths['span_report'].read_text(encoding='utf-8'))
    control_report = json.loads(paths['control_report'].read_text(encoding='utf-8'))
    if any(report['teacher_candidate_sha256'] != hashes['start'] or
           report['candidate_sha256'] != hashes[name]
           for name, report in (('span', span_report), ('control', control_report))):
        raise ValueError('Candidate lineage mismatch')
    payloads = {name: torch.load(paths[name], map_location='cpu', weights_only=True)
                for name in ('start', 'span', 'control')}
    if any(key not in p['reader'] for p in payloads.values() for key in KEYS):
        raise ValueError('Missing reader parameters')
    comparisons = {}
    for source, target in (('start', 'span'), ('start', 'control'), ('control', 'span')):
        comparisons[f'{source}_to_{target}'] = {
            key: compare_tensors(payloads[source]['reader'][key], payloads[target]['reader'][key]) for key in KEYS}
    if any(base.sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Input changed during comparison')
    result = {'status': 'offline_parameter_comparison_no_optimizer', 'source_sha256': hashes,
              'comparisons': comparisons,
              'limitations': ['Parameter difference does not measure contribution to generated answers.',
                              'Checkpoint changes combine all losses and the full 64-step optimizer trajectory.',
                              'Only selected native reader tensors were compared.',
                              'This is not an independent candidate reload or long-term retention evaluation.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({name: {key: value['difference_norm'] for key, value in rows.items()}
                      for name, rows in comparisons.items()}), flush=True)


if __name__ == '__main__':
    main()
