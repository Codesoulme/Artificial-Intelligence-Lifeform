"""Contrast shared and isolated factorized writes through the same language reader."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts import deepseek_local_baseline as base
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_joint_writer_reader_pilot import fact_cache
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_ttt_ntp import LowRankFastWeight, attach_read_adapters, install


NAMES = ('小赵', '小郑', '小王', '小孙', '小陈', '小程', '小余', '小俞',
         '周予', '周羽', '黄然', '黄冉', '叶楠', '叶南', '张琪', '张琦',
         '林晓', '林筱', '邓欣', '邓昕', '刘清', '刘晴', '任诗', '任思',
         '宋文', '宋雯', '侯晨', '侯辰', '顾安', '顾岸', '段元', '段源')
VALUES = ('酸奶', '咖啡', '豆浆', '果汁')
UNKNOWN = ('小雪', '周宇', '段园')
SOURCE = Path('runs/deepseek-spora-ttt-selective-crossed-values-20261116')


def event_batches():
    for start, end in ((0, 8), (8, 32)):
        new = [(NAMES[i], VALUES[(i + i // 4) % len(VALUES)]) for i in range(start, end)]
        fixes = [(name, VALUES[(VALUES.index(value) + 1) % len(VALUES)]) for name, value in new[-4:]]
        yield end, new + fixes


def empty(wrappers):
    result = {}
    for layer, wrapper in wrappers.items():
        out_width, in_width = wrapper.base.weight.shape
        device = wrapper.base.weight.device
        result[layer] = LowRankFastWeight(torch.empty(0, in_width, device=device),
                                          torch.empty(0, out_width, device=device))
    return result


def write(memory, cache, payload):
    return {layer: state.write(cache['keys'][layer], cache['targets'][layer], eta=payload['eta'], ridge=payload['ridge'])
            for layer, state in memory.items()}


def serialize(banks):
    return {name: {str(layer): (state.keys.cpu(), state.values.cpu()) for layer, state in memory.items()}
            for name, memory in banks.items()}


def restore(banks, wrappers):
    expected = {str(layer) for layer in wrappers}
    if any(set(bank) != expected for bank in banks.values()):
        raise ValueError('Saved fast-weight bank layers changed')
    out = {}
    for name, bank in banks.items():
        out[name] = {}
        for layer, wrapper in wrappers.items():
            device = wrapper.base.weight.device
            keys, values = bank[str(layer)]
            state = LowRankFastWeight(keys.to(device), values.to(device))
            if state.shape != tuple(wrapper.base.weight.shape):
                raise ValueError('Saved fast-weight bank shape changed')
            out[name][layer] = state
    return out


def read_all(model, tokenizer, shared, isolated, targets, previous):
    rows = []
    for index, (name, target) in enumerate(targets.items()):
        question = f'请只回答{name}喜欢喝什么饮料。'
        # Supply only the name and question. The benchmark supplies the bank address separately.
        states = {'shared': shared['all'], 'isolated': isolated[name],
                  'swapped': isolated[NAMES[index ^ 1]], 'off': None}
        rows.append({'name': name, 'expected': target, 'previous': previous.get(name),
                     'answers': {mode: pilot.generate(model, tokenizer, question, state)
                                 for mode, state in states.items()}})
    unknown = []
    for name in UNKNOWN:
        question = f'请只回答{name}喜欢喝什么饮料。'
        answers = {mode: pilot.generate(model, tokenizer, question, state) for mode, state in
                   (('shared_unrouted', shared['all']), ('isolated_missing', isolated.get(name)), ('off', None))}
        unknown.append({'name': name, 'answers': answers})
    return rows, unknown


def summarize(rows, unknown):
    result = {}
    expected = {row['name']: row['expected'] for row in rows}
    for mode in rows[0]['answers']:
        answers = [row['answers'][mode]['answer'].strip().rstrip('。.!！ ').strip() for row in rows]
        result[mode] = {'exact': sum(answer == row['expected'] for row, answer in zip(rows, answers, strict=True)),
                        'corrected_exact': sum(row['previous'] is not None and answer == row['expected']
                                            for row, answer in zip(rows, answers, strict=True)),
                        'stale': sum(answer == row['previous'] for row, answer in zip(rows, answers, strict=True)),
                        'swapped_distinct_exact': sum(answer == expected[NAMES[i ^ 1]] and answer != row['expected']
                                                      for i, (row, answer) in enumerate(zip(rows, answers, strict=True)))}
    result['unknown'] = {'without_bank_matches_off': sum(row['answers']['isolated_missing'] == row['answers']['off']
                                                         for row in unknown),
                         'shared_known_value': sum(row['answers']['shared_unrouted']['answer'].strip('。!！ ') in VALUES
                                                   for row in unknown)}
    return result


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    output = args.output
    report_path = output / 'result.json'
    checkpoint = base.MODEL / 'model.safetensors'
    adapter, adapter_report = SOURCE / 'adapters.pt', SOURCE / 'result.json'
    paths = (checkpoint, adapter, adapter_report)
    hashes = {str(path): base.sha(path) for path in paths}
    with adapter_report.open(encoding='utf-8') as stream:
        provenance = json.load(stream)
    payload = torch.load(adapter, weights_only=True, map_location='cpu')
    if (hashes[str(checkpoint)] != base.WEIGHT_SHA or hashes[str(adapter)] != provenance['adapter_sha256']
            or payload['base_sha256'] != base.WEIGHT_SHA or payload['format'] != 'spora-ttt-selective-crossed-values-v1'):
        raise ValueError('Frozen language reader or backbone changed')

    if args.verify:
        with report_path.open(encoding='utf-8') as stream:
            original = json.load(stream)
        if original['sources'] != hashes:
            raise ValueError('Saved experiment uses another base or reader')
    else:
        output.mkdir(parents=True, exist_ok=False)
    model = load_full_streamed().eval()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    wrappers = install(model, tuple(payload['layers']))
    adapters = attach_read_adapters(wrappers, rank=payload['rank'])
    for layer, reader in adapters.items():
        reader.load_state_dict(payload['adapters'][str(layer)], strict=True)
    model.requires_grad_(False)
    shared, isolated, targets, previous, stages = {'all': empty(wrappers)}, {}, {}, {}, []
    for count, events in event_batches():
        if args.verify and count != len(NAMES):
            continue
        saved_stage = original['stages'][-1] if args.verify else None
        timings = {'capture': 0., 'shared': 0., 'isolated': 0.}
        storage = {}
        if args.verify:
            banks = []
            for mode in ('shared', 'isolated'):
                path = output / f'{count}-{mode}.pt'
                if base.sha(path) != saved_stage['storage'][mode]['sha256']:
                    raise ValueError('Saved fast-weight state digest changed')
                banks.append(restore(torch.load(path, weights_only=True, map_location='cpu'), wrappers))
            shared, isolated = banks
            storage = saved_stage['storage']
            targets = {row['name']: row['expected'] for row in saved_stage['rows']}
            previous = {row['name']: row['previous'] for row in saved_stage['rows']}
        else:
            for name, value in events:
                if name in targets:
                    previous[name] = targets[name]
                targets[name] = value
                step = perf_counter()
                cache = fact_cache(model, tokenizer, wrappers, f'{name}喜欢喝{value}。', value)
                torch.cuda.synchronize()
                timings['capture'] += perf_counter() - step
                for mode in ('shared', 'isolated'):
                    step = perf_counter()
                    if mode == 'shared':
                        shared['all'] = write(shared['all'], cache, payload)
                    else:
                        existing = isolated[name] if name in isolated else empty(wrappers)
                        isolated[name] = write(existing, cache, payload)
                    torch.cuda.synchronize()
                    timings[mode] += perf_counter() - step
            for mode, bank in (('shared', shared), ('isolated', isolated)):
                path = output / f'{count}-{mode}.pt'
                atomic_save(serialize(bank), path)
                storage[mode] = {'bytes': path.stat().st_size, 'sha256': base.sha(path),
                                 'tensor_bytes': sum(t.numel() * t.element_size() for memory in bank.values()
                                                     for state in memory.values() for t in (state.keys, state.values))}
        rows, unknown = read_all(model, tokenizer, shared, isolated, targets, previous)
        summary = summarize(rows, unknown)
        print(count, summary, flush=True)
        if args.verify and (rows != saved_stage['rows'] or unknown != saved_stage['unknown']):
            raise ValueError('Reload changed a decoded token sequence')
        stages.append({'count': count, 'rows': rows, 'unknown': unknown, 'summary': summary, 'storage': storage,
                       'write_seconds': timings})
    if any(base.sha(path) != hashes[str(path)] for path in paths):
        raise ValueError('A frozen source changed during the run')
    result = {'sources': hashes, 'stages': stages, 'seconds': perf_counter() - started,
              'peak_cuda_bytes': torch.cuda.max_memory_allocated()}
    store_json(output / 'reload.json' if args.verify else report_path, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
