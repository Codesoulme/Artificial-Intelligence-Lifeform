"""Audit the candidate intent probe on the earlier route challenge, read-only."""
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
CASES = ROOT / 'runs/deepseek-spora-route-generalization-probe-20261117.json'
OUT = ROOT / 'runs/deepseek-spora-intent-attribute-audit-20261117.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {'base': base.MODEL / 'model.safetensors', 'layer6': PAIR / 'layer-6.pt',
             'layer12': PAIR / 'layer-12.pt', 'gate': GATE / 'gate.pt', 'cases': CASES}
    before = {key: base.sha(path) for key, path in paths.items()}
    payload = torch.load(paths['gate'], map_location='cpu', weights_only=True)
    report = json.loads((GATE / 'result.json').read_text(encoding='utf-8'))
    if (before['base'] != base.WEIGHT_SHA or before['gate'] != report['gate_sha256']
            or payload['source_sha256'] != {k: before[k] for k in ('base', 'layer6', 'layer12')}
            or payload['feature'] != 'early_mean'):
        raise ValueError('Intent probe provenance mismatch')
    cases = json.loads(CASES.read_text(encoding='utf-8'))['rows']
    if len(cases) != 18:
        raise ValueError('Unexpected source challenge size')
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = load_full_streamed(skip_replaced=(6, 12)).eval()
    for index in (6, 12):
        model.model.layers[index] = _load_candidate(paths[f'layer{index}'], index, model).eval()
    model.requires_grad_(False)
    gate = nn.Linear(model.config.hidden_size, 1).eval()
    gate.load_state_dict(payload['state'], strict=True)
    rows = []
    with torch.inference_mode():
        for row in cases:
            inputs = tokenizer(row['question'], return_tensors='pt', add_special_tokens=False).to('cuda')
            output = model.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            features = F.normalize(output.hidden_states[1][0].float().mean(0).cpu(), dim=-1)
            score = float(gate(features).sigmoid().item())
            # An unknown entity can still be a genuine request to read a
            # beverage preference. Entity existence is a separate gate.
            label = row['group'] in ('current', 'past', 'unknown')
            rows.append({'question': row['question'], 'expected_read': label, 'group': row['group'],
                         'score': score, 'accepted': score >= payload['threshold'],
                         'prior_route_accepted': row['results']['v2']['result']['accepted']})
    if any(base.sha(path) != before[key] for key, path in paths.items()):
        raise ValueError('Protected input changed')
    summary = {'positives': sum(r['expected_read'] for r in rows),
               'negatives': sum(not r['expected_read'] for r in rows),
               'true_accepts': sum(r['expected_read'] and r['accepted'] for r in rows),
               'false_rejects': sum(r['expected_read'] and not r['accepted'] for r in rows),
               'false_accepts': sum(not r['expected_read'] and r['accepted'] for r in rows)}
    result = {'summary': summary, 'rows': rows, 'source_sha256': before,
              'seconds': time.perf_counter()-start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Questions were inspected before probe training and informed the design; not independent test.',
                              'Offline intent classification only; no temporal mode, identity, state read, or generation.',
                              'Current and past requests are both positive; accepted requests may still be unsafe or wrong.',
                              'Synthetic questions and manually assigned labels; no deployment or real user state.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
