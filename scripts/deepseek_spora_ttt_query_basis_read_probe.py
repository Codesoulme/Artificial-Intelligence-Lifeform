"""Evaluate compact query-basis bridges against fast-weight memory reads."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_joint_writer_reader_pilot import fact_cache
from scripts.deepseek_spora_ttt_persistence_interference import SOURCE
from scripts.deepseek_spora_ttt_query_basis_shared_probe import query_state
from spora.research_ttt_ntp import (attach_query_basis_bridges, attach_read_adapters, fresh_memory, install,
                                    sequential_ntp_write, use_query_bridges)

OUT = ROOT / 'runs/deepseek-spora-ttt-query-basis-read-v2-20261116'
TRAIN = (('小林', '酸奶'), ('阿宁', '豆浆'), ('小雨', '咖啡'), ('阿哲', '果汁'))
HELD = (('小赵', '豆浆'), ('小郑', '咖啡'), ('小王', '酸奶'), ('小孙', '果汁'))
UNKNOWN = ('小雪', '阿松')


def exact(answer, value):
    return value is not None and answer.strip().rstrip('。.!！ ').strip() == value


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {'base': base.MODEL / 'model.safetensors', 'adapter': SOURCE / 'adapters.pt',
             'source_report': SOURCE / 'result.json'}
    hashes = {key: base.sha(path) for key, path in paths.items()}
    source_report = json.loads(paths['source_report'].read_text(encoding='utf-8'))
    adapter = torch.load(paths['adapter'], map_location='cpu', weights_only=True)
    if (hashes['base'] != base.WEIGHT_SHA or hashes['adapter'] != source_report['adapter_sha256']
            or adapter['base_sha256'] != hashes['base']):
        raise ValueError('Source provenance mismatch')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    wrappers = install(model, tuple(adapter['layers']))
    readers = attach_read_adapters(wrappers, rank=adapter['rank'])
    for layer, reader in readers.items():
        reader.load_state_dict(adapter['adapters'][str(layer)], strict=True)
    model.requires_grad_(False)
    rows = []
    caches = {}
    for name, value in TRAIN + HELD:
        fact = f'{name}喜欢喝{value}。'
        cache = fact_cache(model, tokenizer, wrappers, fact, value)
        if any(cache['keys'][layer].shape[0] != 1 for layer in wrappers):
            raise ValueError('This probe requires one-token values')
        caches[(name, value)] = cache
    bases = {}
    queries = {name: query_state(model, tokenizer, wrappers,
                                 f'请只回答{name}喜欢喝什么饮料。') for name, _ in TRAIN}
    for layer in wrappers:
        query_rows = [queries[name][layer] for name, _ in TRAIN]
        key_rows = [caches[(name, value)]['keys'][layer][0].cpu() for name, value in TRAIN]
        bases[layer] = (torch.stack(query_rows), torch.stack(key_rows))
    bridges = attach_query_basis_bridges(wrappers, bases, ridge=10.)
    bridge_calls = {layer: 0 for layer in bridges}
    for layer, bridge in bridges.items():
        def count_call(module, args, output, layer=layer):
            bridge_calls[layer] += 1
        bridge.register_forward_hook(count_call)

    def evaluate(name, value, bridged):
        memory = fresh_memory(wrappers)
        sequential_ntp_write(memory, caches[(name, value)]['keys'], caches[(name, value)]['targets'],
                             eta=adapter['eta'], ridge=adapter['ridge'])
        before = dict(bridge_calls)
        with use_query_bridges(wrappers, bridges if bridged else None):
            result = pilot.generate(model, tokenizer, f'请只回答{name}喜欢喝什么饮料。', memory)
        calls = {layer: bridge_calls[layer] - before[layer] for layer in bridges}
        if any((count == 0) if bridged else (count != 0) for count in calls.values()):
            raise RuntimeError('Bridge execution did not match the selected condition')
        return {'answer': result['answer'], 'token_ids': result['token_ids'],
                'exact': exact(result['answer'], value), 'bridge_calls': calls}

    for split, cases in (('train', TRAIN), ('held', HELD)):
        for name, value in cases:
            rows.append({'split': split, 'name': name, 'value': value,
                         'baseline': evaluate(name, value, False), 'bridged': evaluate(name, value, True)})
    unknown = []
    for name in UNKNOWN:
        memory = fresh_memory(wrappers)
        result = pilot.generate(model, tokenizer, f'请只回答{name}喜欢喝什么饮料。', memory)
        unknown.append({'name': name, 'answer': result['answer'], 'token_ids': result['token_ids']})
    summary = {}
    for split in ('train', 'held'):
        subset = [row for row in rows if row['split'] == split]
        summary[split] = {'cases': len(subset), 'baseline_exact': sum(row['baseline']['exact'] for row in subset),
                          'bridged_exact': sum(row['bridged']['exact'] for row in subset),
                          'changed': sum(row['baseline']['token_ids'] != row['bridged']['token_ids'] for row in subset)}
    summary['unknown'] = {'cases': len(unknown), 'nonempty_answers': sum(bool(row['answer']) for row in unknown)}
    if any(base.sha(path) != hashes[key] for path, key in [(paths['base'], 'base'), (paths['adapter'], 'adapter'),
                                                            (paths['source_report'], 'source_report')]):
        raise ValueError('Protected input changed')
    result = {'summary': summary, 'rows': rows, 'unknown': unknown,
              'source_sha256': hashes, 'bridge_rows': len(TRAIN),
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Bridge is a compact four-person basis fit with oracle fact/value boundaries.',
                              'The bridge is applied only to fast-weight reads; no autonomous routing.',
                              'Held people and values are synthetic and limited to the development vocabulary.',
                              'Unknown questions use an empty fast state and have no learned rejection gate.',
                              'No online update, correction, persistence, crash recovery or open-domain chat.']}
    OUT.mkdir()
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
