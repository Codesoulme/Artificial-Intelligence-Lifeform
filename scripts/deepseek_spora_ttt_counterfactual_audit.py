"""Check whether a fixed question follows changes to its TTT memory state."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_ttt_ntp import attach_read_adapters, install

SOURCE = ROOT / 'runs/deepseek-spora-ttt-read-adapter-pilot-v4-20261114'
OUT = ROOT / 'runs/deepseek-spora-ttt-counterfactual-audit-20261114.json'
NAMES = ('小林', '阿宁', '小赵', '小郑')
VALUES = ('酸奶', '豆浆')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    adapter_path = SOURCE / 'adapters.pt'
    metadata = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    adapter_sha = p.sha(adapter_path)
    payload = torch.load(adapter_path, map_location='cpu', weights_only=True)
    if metadata['adapter_sha256'] != adapter_sha or payload['base_sha256'] != p.WEIGHT_SHA:
        raise ValueError('Candidate provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    wrappers = install(model, tuple(payload['layers']))
    adapters = attach_read_adapters(wrappers, rank=payload['rank'])
    for layer_id, adapter in adapters.items():
        adapter.load_state_dict(payload['adapters'][str(layer_id)], strict=True)
    model.requires_grad_(False)
    rows = []
    for name in NAMES:
        pair = []
        for value in VALUES:
            case = pilot.build_case(model, tokenizer, wrappers, name, value)
            answer = pilot.generate(model, tokenizer, case['question'], pilot.device_memory(case['memory']))
            row = {'name': name, 'value': value, 'question': case['question'], 'result': answer,
                   'exact': answer['answer'].strip().rstrip('。.!！ ') == value}
            rows.append(row)
            pair.append(row)
        print(json.dumps({'name': name, 'correct': sum(row['exact'] for row in pair),
                          'changed': pair[0]['result']['token_ids'] != pair[1]['result']['token_ids']},
                         ensure_ascii=False), flush=True)
    if p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Base model changed')
    pairs = [rows[i:i + 2] for i in range(0, len(rows), 2)]
    report = {'status': 'ttt_counterfactual_diagnostic', 'rows': rows,
              'summary': {'cases': len(rows), 'exact': sum(r['exact'] for r in rows),
                          'pairs': len(pairs), 'both_correct': sum(all(r['exact'] for r in pair) for pair in pairs),
                          'changed': sum(pair[0]['result']['token_ids'] != pair[1]['result']['token_ids']
                                         for pair in pairs)},
              'adapter_sha256': adapter_sha, 'base_sha256': p.WEIGHT_SHA,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Four names and two familiar values with explicit isolated fact writes.',
                              'A changed answer does not by itself establish correct memory use.',
                              'No persistent reload, natural routing, or open-domain evaluation.']}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
