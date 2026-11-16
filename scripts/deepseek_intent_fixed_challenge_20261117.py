"""Read-only challenge of an existing finite-beverage direct-answer gate.

Negative labels prohibit the finite beverage-name answer path, not necessarily
all memory retrieval: answering an attribute may require retrieval and reasoning.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_local_baseline as base
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate

GATE = ROOT / 'runs/deepseek-spora-intent-attribute-probe-20261117'
CASES = ROOT / 'configs/intent-attribute-challenge-20261117.json'
OUT = ROOT / 'runs/deepseek-spora-intent-fixed-challenge-20261117.json'


def summarize(rows):
    result = {}
    for group in sorted({row['group'] for row in rows}):
        subset = [row for row in rows if row['group'] == group]
        result[group] = {'total': len(subset), 'accepted': sum(row['accepted'] for row in subset),
                         'expected_accept': sum(row['read'] for row in subset),
                         'correct': sum(row['accepted'] == row['read'] for row in subset)}
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {'base': base.MODEL / 'model.safetensors', 'layer6': PAIR / 'layer-6.pt',
             'layer12': PAIR / 'layer-12.pt', 'gate': GATE / 'gate.pt', 'cases': CASES}
    before = {key: base.sha(path) for key, path in paths.items()}
    cases = json.loads(CASES.read_text(encoding='utf-8'))
    if (len(cases) != 36 or len({r['question'] for r in cases}) != len(cases)
            or any(type(r.get('read')) is not bool for r in cases)):
        raise ValueError('Invalid fixed cases')
    payload = torch.load(paths['gate'], map_location='cpu', weights_only=True)
    report = json.loads((GATE / 'result.json').read_text(encoding='utf-8'))
    if (before['base'] != base.WEIGHT_SHA or before['gate'] != report['gate_sha256']
            or payload['source_sha256'] != {k: before[k] for k in ('base', 'layer6', 'layer12')}
            or payload['feature'] != 'early_mean'):
        raise ValueError('Intent probe provenance mismatch')
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed(skip_replaced=(6, 12)).eval()
    for index in (6, 12):
        model.model.layers[index] = _load_candidate(paths[f'layer{index}'], index, model).eval()
    model.requires_grad_(False)
    gate = nn.Linear(model.config.hidden_size, 1).eval()
    gate.load_state_dict(payload['state'], strict=True)
    rows = []
    with torch.inference_mode():
        for case in cases:
            inputs = tokenizer(case['question'], return_tensors='pt', add_special_tokens=False).to('cuda')
            output = model.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            features = F.normalize(output.hidden_states[1][0].float().mean(0).cpu(), dim=-1)
            score = float(gate(features).sigmoid().item())
            rows.append({**case, 'score': score, 'accepted': score >= payload['threshold']})
    if any(base.sha(path) != before[key] for key, path in paths.items()):
        raise ValueError('Protected input changed')
    result = {'groups': summarize(rows), 'threshold': payload['threshold'], 'rows': rows,
              'source_sha256': before, 'seconds': time.perf_counter()-start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Manually authored targeted development challenge, not population representative.',
                              'Cases authored after prior findings and templates were inspected.',
                              'Labels concern direct beverage-name answer, not general memory retrieval.',
                              'Offline classifier only, no identity, temporal mode, generation or real state.',
                              'No threshold adjustment or training on this challenge.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['groups'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
