"""Measure numerical survival of the trained cross-attention residual.

Uses the existing independently reloaded candidate and generation/parity audit.
The forward hook is diagnostic only and never mutates a parameter or output.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_bridge_reload_audit as audit

torch = audit.torch
s = audit.s
audit.SOURCE = ROOT / 'runs/continuous-reader-cross-attention-pilot-20261113'
audit.OUT = ROOT / 'runs/deepseek-spora-cross-precision-audit-20261113'
audit.EXTRA_WORDS = ('桑葚', '枇杷')
audit.EXTRA_KEYS = (3,)
audit.EXTRA_GROUP_PREFIX = 'unseen'
audit.EXTRA_ONLY = True
audit.VERIFY_CODEC_READS = True
audit.PARITY_VALUE_INDICES = (0, 1, 3, 9, 16, 17)

original = s.MemorySlotCrossAttention.forward
measurements = []


def measured(module, x):
    captured = []
    hook = module.out.register_forward_hook(lambda _m, _i, output: captured.append(output.detach()))
    try:
        result = original(module, x)
    finally:
        hook.remove()
    if captured:
        with torch.no_grad():
            base = module.base(x).detach()
            intended = captured[0].float() * torch.sigmoid(module.gate.float())
            actual = result.float() - base.float()
            measurements.append({
                'tokens': int(x.shape[1]), 'dtype': str(base.dtype),
                'intended_l2': float(intended.norm()), 'surviving_l2': float(actual.norm()),
                'intended_max': float(intended.abs().max()), 'surviving_max': float(actual.abs().max()),
                'intended_nonzero': int(torch.count_nonzero(intended)),
                'surviving_nonzero': int(torch.count_nonzero(actual)),
                'elements': intended.numel(),
                'error_l2': float((actual - intended).norm()),
            })
    return result


if __name__ == '__main__':
    s.MemorySlotCrossAttention.forward = measured
    try:
        audit.main()
    finally:
        s.MemorySlotCrossAttention.forward = original
    if not measurements:
        raise ValueError('No active cross-attention calls were observed')
    total = sum(row['elements'] for row in measurements)
    result = {'calls': len(measurements), 'elements': total,
              'intended_nonzero': sum(row['intended_nonzero'] for row in measurements),
              'surviving_nonzero': sum(row['surviving_nonzero'] for row in measurements),
              'all_surviving_zero_calls': sum(row['surviving_nonzero'] == 0 for row in measurements),
              'rows': measurements,
              'interpretation': 'Measures residual survival at the o_proj addition only; it does not establish correct semantic use or logits effect.'}
    with (audit.OUT / 'precision.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)
