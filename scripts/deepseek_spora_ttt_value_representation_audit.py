"""Inspect fast-weight and adapter representations under controlled fact swaps."""
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_crossed_values_pilot as crossed
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_ttt_ntp import attach_read_adapters, install, use_memory

SOURCE = crossed.OUT
OUT = ROOT / 'runs/deepseek-spora-ttt-value-representation-audit-20261114.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    meta = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    adapter_sha = p.sha(SOURCE / 'adapters.pt')
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    payload = torch.load(SOURCE / 'adapters.pt', map_location='cpu', weights_only=True)
    if adapter_sha != meta['adapter_sha256'] or base_sha != p.WEIGHT_SHA or payload['base_sha256'] != base_sha:
        raise ValueError('Candidate or base provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    wrappers = install(model, tuple(payload['layers']))
    adapters = attach_read_adapters(wrappers, rank=payload['rank'])
    for layer, adapter in adapters.items():
        adapter.load_state_dict(payload['adapters'][str(layer)], strict=True)
    model.requires_grad_(False)
    values = crossed.TRAIN_VALUES + crossed.HELD_VALUES
    rows = []
    records = {}
    for name in crossed.HELD_NAMES:
        question = crossed.TEMPLATES[0].format(name=name)
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        for value in values:
            case = pilot.build_case(model, tokenizer, wrappers, name, value, crossed.TEMPLATES[0])
            memory = pilot.device_memory(case['memory'])
            captured = {}
            handles = []
            for layer, adapter in adapters.items():
                def capture(module, args, output, layer=layer):
                    captured[layer] = (args[0][0, -1].detach().float().cpu(),
                                       output[0, -1].detach().float().cpu())
                handles.append(adapter.register_forward_hook(capture))
            try:
                with torch.inference_mode(), use_memory(memory):
                    logits = model(**prompt, use_cache=False, return_dict=True).logits[0, -1].float()
            finally:
                for handle in handles:
                    handle.remove()
            if set(captured) != set(adapters):
                raise ValueError('Missing per-layer adapter observations')
            token_ids = tokenizer.encode(value, add_special_tokens=False)
            target = token_ids[0]
            rank = int((logits > logits[target]).sum()) + 1
            winner = int(logits.argmax())
            records[(name, value)] = captured
            rows.append({'name': name, 'value': value, 'group': 'seen' if value in crossed.TRAIN_VALUES else 'unseen',
                         'target_token_ids': token_ids, 'target_first_token_rank': rank,
                         'target_first_token_logit': float(logits[target]),
                         'winner_token_id': winner,
                         'winner_token_text': tokenizer.decode([winner], skip_special_tokens=False),
                         'layers': {str(layer): {'raw_norm': float(pair[0].norm()),
                                                  'adapter_norm': float(pair[1].norm())}
                                    for layer, pair in captured.items()}})
            print(json.dumps({'name': name, 'value': value, 'target_rank': rank}, ensure_ascii=False), flush=True)
    comparisons = []
    for left, right in itertools.combinations(records, 2):
        a, b = records[left], records[right]
        kind = ('same_value_cross_name' if left[1] == right[1] and left[0] != right[0]
                else 'different_value_same_name' if left[0] == right[0] and left[1] != right[1]
                else None)
        if kind is None:
            continue
        layers = {}
        for layer in adapters:
            layers[str(layer)] = {}
            for index, label in enumerate(('raw', 'adapted')):
                x, y = a[layer][index], b[layer][index]
                layers[str(layer)][label] = {'cosine': float(F.cosine_similarity(x, y, dim=0)),
                                              'difference_norm': float((x-y).norm())}
        comparisons.append({'kind': kind, 'left': list(left), 'right': list(right), 'layers': layers})
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(SOURCE / 'adapters.pt') != adapter_sha:
        raise ValueError('Protected file changed')
    summary = {}
    for group in ('seen', 'unseen'):
        subset = [row for row in rows if row['group'] == group]
        summary[group] = {'cases': len(subset),
                          'first_token_top1': sum(row['target_first_token_rank'] == 1 for row in subset),
                          'first_token_top10': sum(row['target_first_token_rank'] <= 10 for row in subset)}
    result = {'status': 'ttt_value_representation_diagnostic', 'summary': summary, 'rows': rows,
              'comparisons': comparisons, 'base_sha256': base_sha, 'adapter_sha256': adapter_sha,
              'seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two names, six values, one fixed question per name and one fact per state.',
                              'First-token rank does not establish full answer correctness, especially for multi-token values.',
                              'Feature separability and cosine distance do not prove recoverable semantics or causality.',
                              'No automatic writing, persistence, or unknown-fact behavior evaluated.']}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
