"""Compare whole-fact writes against oracle-selected value-boundary writes."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_crossed_values_pilot as crossed
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_ttt_ntp import attach_read_adapters, fresh_memory, install, sequential_ntp_write

OUT = ROOT / 'runs/deepseek-spora-ttt-selective-write-audit-20261114.json'
SOURCE = crossed.OUT


def selected_memory(model, tokenizer, wrappers, fact, value):
    encoded = tokenizer(fact, return_tensors='pt', add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.pop('offset_mapping')[0].tolist()
    start = fact.index(value)
    positions = [i - 1 for i, (left, right) in enumerate(offsets)
                 if i > 0 and left < start + len(value) and right > start]
    if not positions:
        raise ValueError('No aligned value token positions')
    inputs = {key: tensor.to('cuda') for key, tensor in encoded.items()}
    captured = {}
    handles = []
    for layer, wrapper in wrappers.items():
        def capture(module, args, output, layer=layer):
            captured[layer] = args[0][0].detach().float()
        handles.append(wrapper.base.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            result = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    indices = torch.tensor(positions, device='cuda', dtype=torch.long)
    keys = {layer: captured[layer].index_select(0, indices) for layer in wrappers}
    targets = {layer: result.hidden_states[layer + 1][0].detach().float().index_select(0, indices + 1)
               for layer in wrappers}
    memory = fresh_memory(wrappers)
    sequential_ntp_write(memory, keys, targets, eta=pilot.ETA, ridge=pilot.RIDGE)
    return memory, positions


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    meta = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    adapter_sha = p.sha(SOURCE / 'adapters.pt')
    if adapter_sha != meta['adapter_sha256'] or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Source digest mismatch')
    payload = torch.load(SOURCE / 'adapters.pt', map_location='cpu', weights_only=True)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    wrappers = install(model, tuple(payload['layers']))
    adapters = attach_read_adapters(wrappers, rank=payload['rank'])
    for layer, adapter in adapters.items():
        adapter.load_state_dict(payload['adapters'][str(layer)], strict=True)
    model.requires_grad_(False)
    rows = []
    for name in crossed.HELD_NAMES:
        for value in crossed.TRAIN_VALUES + crossed.HELD_VALUES:
            fact = f'{name}喜欢喝{value}。'
            question = crossed.TEMPLATES[0].format(name=name)
            memory, positions = selected_memory(model, tokenizer, wrappers, fact, value)
            answer = pilot.generate(model, tokenizer, question, memory)
            group = 'held_name' if value in crossed.TRAIN_VALUES else 'held_value'
            original = next(r for r in meta['rows'] if r['group'] == group and r['name'] == name and r['value'] == value)
            rows.append({'name': name, 'value': value, 'group': group, 'selected_positions': positions,
                         'whole_fact': original['answer'], 'whole_exact': original['exact'],
                         'selected': answer, 'selected_exact': crossed.exact(answer['answer'], value)})
            print(json.dumps({'name': name, 'value': value, 'selected_exact': rows[-1]['selected_exact']},
                             ensure_ascii=False), flush=True)
    if p.sha(SOURCE / 'adapters.pt') != adapter_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    summary = {}
    for group in ('held_name', 'held_value'):
        subset = [r for r in rows if r['group'] == group]
        summary[group] = {'cases': len(subset), 'whole_exact': sum(r['whole_exact'] for r in subset),
                          'selected_exact': sum(r['selected_exact'] for r in subset),
                          'tokens_changed': sum(r['whole_fact']['token_ids'] != r['selected']['token_ids'] for r in subset)}
    report = {'status': 'oracle_selected_write_diagnostic', 'summary': summary, 'rows': rows,
              'adapter_sha256': adapter_sha, 'base_sha256': p.WEIGHT_SHA,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Value boundaries are supplied by the experiment; this is not autonomous selection.',
                              'The adapter was trained on whole-fact states, not the selected-write distribution.',
                              'Two names and six values; outputs are development probes, not generalization estimates.',
                              'Whole-fact reference was generated in a different process.']}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
