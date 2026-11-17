"""Compare post-training recovery and final research checkpoint tensors."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_local_baseline as base

RUN = ROOT / 'runs/deepseek-spora-multitoken-span-continuation-128-20261117'
RECOVERY = ROOT / 'runs/deepseek-spora-multitoken-span-continuation-128-recovery-20261117.pt'
OUT = ROOT / 'runs/deepseek-spora-multitoken-span-continuation-recovery-audit-20261117.json'
FIELDS = ('reader', 'bridge', 'codec_state', 'adapters', 'direct_adapters',
          'final_adapters', 'cross_adapters', 'logit_adapters')


def compare_tree(left, right, path='root'):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError(f'Tensor metadata mismatch at {path}')
        if not torch.equal(left, right):
            raise ValueError(f'Tensor mismatch at {path}')
        return 1
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError(f'Dictionary mismatch at {path}')
        return sum(compare_tree(left[k], right[k], f'{path}.{k}') for k in left)
    if isinstance(left, (tuple, list)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise ValueError(f'Sequence mismatch at {path}')
        return sum(compare_tree(a, b, f'{path}[{i}]') for i, (a, b) in enumerate(zip(left, right)))
    if left != right:
        raise ValueError(f'Value mismatch at {path}')
    return 0


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {'recovery': RECOVERY, 'candidate': RUN/'candidate.pt', 'result': RUN/'result.json'}
    hashes = {name: base.sha(path) for name, path in paths.items()}
    report = json.loads(paths['result'].read_text(encoding='utf-8'))
    if report['candidate_sha256'] != hashes['candidate']:
        raise ValueError('Candidate provenance mismatch')
    recovery, candidate = (torch.load(paths[k], map_location='cpu', weights_only=True)
                           for k in ('recovery', 'candidate'))
    if (recovery['start_candidate_sha256'] != report['teacher_candidate_sha256']
            or recovery['source_sha256'] != candidate['source_sha256']
            or recovery['base_sha256'] != candidate['base_sha256']):
        raise ValueError('Recovery provenance mismatch')
    counts = {field: compare_tree(recovery[field], candidate[field], field) for field in FIELDS}
    if any(base.sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected file changed')
    result = {'status': 'independent_process_checkpoint_tensor_comparison',
              'equal_tensors_by_field': counts, 'source_sha256': hashes,
              'limitations': ['No model inference, optimizer state, mid-training recovery or power-loss test.',
                              'Only the named model tensor fields and stated lineage were checked.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
