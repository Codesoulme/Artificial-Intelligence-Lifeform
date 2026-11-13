"""Reload the TTT read adapter and test unseen question forms."""
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
OUT = ROOT / 'runs/deepseek-spora-ttt-read-adapter-reload-audit-v4-20261114'
ALT_TEMPLATES = (
    '{name}通常会选什么饮料？',
    '关于{name}，过去记录的是哪种饮品？',
    '请从记忆中找出{name}的饮料偏好，只写名称。',
)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    payload = torch.load(SOURCE / 'adapters.pt', map_location='cpu', weights_only=True)
    result = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    adapter_sha = p.sha(SOURCE / 'adapters.pt')
    if payload.get('base_sha256') != p.WEIGHT_SHA or result.get('adapter_sha256') != adapter_sha:
        raise ValueError('Adapter/base provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True,
                                                trust_remote_code=False)
    model = load_full_streamed().eval()
    wrappers = install(model, tuple(payload['layers']))
    adapters = attach_read_adapters(wrappers, rank=payload['rank'])
    for layer_id, adapter in adapters.items():
        adapter.load_state_dict(payload['adapters'][str(layer_id)], strict=True)
    model.requires_grad_(False)
    cases = [pilot.build_case(model, tokenizer, wrappers, name, value, pilot.TRAIN_TEMPLATES[0])
             for name, value in pilot.HELD]
    rows = []
    for item in cases:
        for template in ALT_TEMPLATES:
            question = template.format(name=item['name'])
            with_memory = pilot.generate(model, tokenizer, question,
                                         pilot.device_memory(item['memory']))
            no_memory = pilot.generate(model, tokenizer, question)
            rows.append({'name': item['name'], 'expected': item['value'],
                         'question': question, 'memory': with_memory,
                         'no_memory': no_memory,
                         'memory_contains': item['value'] in with_memory['answer'],
                         'no_memory_contains': item['value'] in no_memory['answer']})
            print(json.dumps({'completed': len(rows), 'expected': item['value'],
                              'memory_contains': rows[-1]['memory_contains']},
                             ensure_ascii=False), flush=True)
    if p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Base model changed')
    report = {
        'status': 'ttt_ntp_read_adapter_reload_audit',
        'source_adapter_sha256': adapter_sha,
        'base_sha256': p.WEIGHT_SHA,
        'summary': {'cases': len(rows),
                    'memory_contains': sum(row['memory_contains'] for row in rows),
                    'no_memory_contains': sum(row['no_memory_contains'] for row in rows)},
        'rows': rows,
        'limitations': [
            'Four held names and three new question templates; values remain in the finite pilot vocabulary.',
            'Fast memory is reconstructed from authored fact text during this audit.',
            'No natural-language routing, correction sequence or multi-relation state.',
            'The adapter was trained on one question family, so this is a reload and wording audit, not open chat.',
        ],
        'seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
    }
    OUT.mkdir()
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
