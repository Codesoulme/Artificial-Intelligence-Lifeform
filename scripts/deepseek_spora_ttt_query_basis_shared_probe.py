"""Evaluate query-basis routing when several facts share one fast state."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_joint_writer_reader_pilot import fact_cache
from scripts.deepseek_spora_ttt_persistence_interference import SOURCE
from spora.research_ttt_ntp import (attach_query_basis_bridges, attach_read_adapters,
                                    fresh_memory, install, sequential_ntp_write, use_query_bridges)

OUT = ROOT / 'runs/deepseek-spora-ttt-query-basis-shared-v2-20261116'
TRAIN = (('小林', '酸奶'), ('阿宁', '豆浆'), ('小雨', '咖啡'), ('阿哲', '果汁'))
HELD = (('小赵', '豆浆'), ('小郑', '咖啡'), ('小王', '酸奶'), ('小孙', '果汁'))
UNKNOWN = ('小雪', '阿松')


def exact(answer, value):
    return value is not None and answer.strip().rstrip('。.!！ ').strip() == value


def query_state(model, tokenizer, wrappers, question):
    prompt = closed_think_inputs(tokenizer, question, 'cuda')
    captured = {}
    hooks = []
    for layer, wrapper in wrappers.items():
        def capture(module, args, output, layer=layer):
            # Hooks are observers; returning a tensor would replace down_proj output.
            captured[layer] = args[0][0, -1].detach().float().cpu()
        hooks.append(wrapper.base.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            model(**prompt, use_cache=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()
    return captured


def concatenate(caches, wrappers):
    return ({layer: torch.cat([item['keys'][layer] for item in caches], dim=0)
             for layer in wrappers},
            {layer: torch.cat([item['targets'][layer] for item in caches], dim=0)
             for layer in wrappers})


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
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
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True,
                                                   trust_remote_code=False)
    model = load_full_streamed().eval()
    wrappers = install(model, tuple(adapter['layers']))
    readers = attach_read_adapters(wrappers, rank=adapter['rank'])
    for layer, reader in readers.items():
        reader.load_state_dict(adapter['adapters'][str(layer)], strict=True)
    model.requires_grad_(False)

    all_cases = TRAIN + HELD
    caches = {(name, value): fact_cache(model, tokenizer, wrappers,
                                        f'{name}喜欢喝{value}。', value)
              for name, value in all_cases}
    train_bases = {}
    for layer in wrappers:
        queries = [query_state(model, tokenizer, wrappers,
                               f'请只回答{name}喜欢喝什么饮料。')[layer]
                   for name, _ in TRAIN]
        keys = [caches[(name, value)]['keys'][layer][0].cpu() for name, value in TRAIN]
        train_bases[layer] = (torch.stack(queries), torch.stack(keys))
    bridges = attach_query_basis_bridges(wrappers, train_bases, ridge=10.)
    bridge_calls = {layer: 0 for layer in bridges}
    for layer, bridge in bridges.items():
        def count_call(module, args, output, layer=layer):
            bridge_calls[layer] += 1
        bridge.register_forward_hook(count_call)

    def shared_memory(cases):
        selected = [caches[item] for item in cases]
        keys, targets = concatenate(selected, wrappers)
        return sequential_ntp_write(fresh_memory(wrappers), keys, targets,
                                    eta=adapter['eta'], ridge=adapter['ridge'])

    def evaluate(name, value, bridged):
        memory = shared_memory(all_cases)
        before = dict(bridge_calls)
        with use_query_bridges(wrappers, bridges if bridged else None):
            result = pilot.generate(model, tokenizer,
                                    f'请只回答{name}喜欢喝什么饮料。', memory)
        calls = {layer: bridge_calls[layer] - before[layer] for layer in bridges}
        if any((count == 0) if bridged else (count != 0) for count in calls.values()):
            raise RuntimeError('Bridge execution did not match the selected condition')
        return {'answer': result['answer'], 'token_ids': result['token_ids'],
                'exact': exact(result['answer'], value), 'bridge_calls': calls}

    rows = []
    for split, cases in (('train', TRAIN), ('held', HELD)):
        for name, value in cases:
            rows.append({'split': split, 'name': name, 'value': value,
                         'baseline': evaluate(name, value, False),
                         'bridged': evaluate(name, value, True)})

    correction = ('小林', '咖啡')
    correction_cases = all_cases + (correction,)
    correction_cache = fact_cache(model, tokenizer, wrappers,
                                  f'{correction[0]}喜欢喝{correction[1]}。', correction[1])
    caches[correction] = correction_cache
    baseline_memory = shared_memory(correction_cases)
    correction_rows = []
    for bridged in (False, True):
        for name, value in (('小林', '咖啡'), ('阿宁', '豆浆'), ('小孙', '果汁')):
            before = dict(bridge_calls)
            with use_query_bridges(wrappers, bridges if bridged else None):
                result = pilot.generate(model, tokenizer,
                                        f'请只回答{name}喜欢喝什么饮料。', baseline_memory)
            calls = {layer: bridge_calls[layer] - before[layer] for layer in bridges}
            if any((count == 0) if bridged else (count != 0) for count in calls.values()):
                raise RuntimeError('Correction bridge execution did not match the condition')
            correction_rows.append({'bridged': bridged, 'name': name, 'expected': value,
                                    'answer': result['answer'],
                                    'exact': exact(result['answer'], value), 'bridge_calls': calls})

    unknown_rows = []
    for name in UNKNOWN:
        with use_query_bridges(wrappers, bridges):
            result = pilot.generate(model, tokenizer,
                                    f'请只回答{name}喜欢喝什么饮料。', shared_memory(all_cases))
        unknown_rows.append({'name': name, 'answer': result['answer'],
                             'token_ids': result['token_ids']})
    summary = {}
    for split in ('train', 'held'):
        subset = [row for row in rows if row['split'] == split]
        summary[split] = {
            'cases': len(subset),
            'baseline_exact': sum(row['baseline']['exact'] for row in subset),
            'bridged_exact': sum(row['bridged']['exact'] for row in subset),
            'changed': sum(row['baseline']['token_ids'] != row['bridged']['token_ids']
                           for row in subset),
        }
    summary['correction'] = {
        'baseline_exact': sum(row['exact'] for row in correction_rows if not row['bridged']),
        'bridged_exact': sum(row['exact'] for row in correction_rows if row['bridged']),
    }
    summary['unknown'] = {'cases': len(unknown_rows),
                          'nonempty_answers': sum(bool(row['answer']) for row in unknown_rows)}
    if any(base.sha(path) != hashes[key] for path, key in [(paths['base'], 'base'),
                                                            (paths['adapter'], 'adapter'),
                                                            (paths['source_report'], 'source_report')]):
        raise ValueError('Protected input changed')
    result = {
        'summary': summary, 'rows': rows, 'correction_rows': correction_rows,
        'unknown': unknown_rows, 'source_sha256': hashes,
        'shared_cases': len(all_cases), 'bridge_rows': len(TRAIN),
        'seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'limitations': ['Eight synthetic facts share one temporary fast state.',
                        'Fact boundaries and identity/value associations are supplied.',
                        'Bridge is fitted from four people and has no learned rejection gate.',
                        'No persistence, crash recovery, autonomous observation or open-domain chat.'],
    }
    OUT.mkdir()
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                                     encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
