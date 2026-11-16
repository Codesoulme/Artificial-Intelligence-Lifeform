"""Compare normal and oracle write-key reads after sequential fast writes.

The oracle supplies identity and fact boundaries. This is a causal diagnostic,
not an autonomous routing or deployable memory interface.
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
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_joint_writer_reader_pilot import fact_cache
from scripts.deepseek_spora_ttt_protected_write_probe import protected_write
from spora.research_ttt_ntp import (attach_read_adapters, fresh_memory, install,
                                    sequential_ntp_write, use_query_bridges)

SOURCE = ROOT / 'runs/deepseek-spora-ttt-selective-crossed-values-20261116'
OUT = ROOT / 'runs/deepseek-spora-ttt-oracle-key-read-20261116'
EVENTS = (('小林', '酸奶'), ('阿宁', '豆浆'), ('小雨', '咖啡'), ('阿哲', '果汁'),
          ('小赵', '豆浆'), ('小郑', '咖啡'), ('小王', '酸奶'), ('小孙', '果汁'),
          ('小林', '豆浆'), ('小赵', '酸奶'))
CHECKPOINTS = (8, 10)
UNKNOWN = ('小雪', '阿松')


class FixedWriteKey(nn.Module):
    """Intervene only on fast-delta inputs with an oracle-selected key."""

    def __init__(self, key):
        super().__init__()
        if not isinstance(key, torch.Tensor) or key.ndim != 1 or not bool(torch.isfinite(key).all()):
            raise ValueError('A finite one-dimensional key is required')
        self.register_buffer('key', key.detach().float().clone())
        self.calls = 0

    def forward(self, hidden):
        if hidden.shape[-1] != self.key.shape[0]:
            raise ValueError('Oracle key has incompatible width')
        self.calls += 1
        return self.key.expand_as(hidden)


def exact(answer, value):
    return value is not None and answer.strip().rstrip('。.!！ ').strip() == value


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {'base': base.MODEL / 'model.safetensors', 'adapter': SOURCE / 'adapters.pt',
             'source_report': SOURCE / 'result.json'}
    hashes = {name: base.sha(path) for name, path in paths.items()}
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
    caches = [fact_cache(model, tokenizer, wrappers, f'{name}喜欢喝{value}。', value)
              for name, value in EVENTS]
    if any(cache['keys'][layer].shape[0] != 1 for cache in caches for layer in wrappers):
        raise ValueError('This diagnostic requires one-token values')
    states = {method: fresh_memory(wrappers) for method in ('ordinary', 'protected')}
    keys = {layer: {} for layer in wrappers}
    latest = {}
    stages = []
    for index, ((name, value), cache) in enumerate(zip(EVENTS, caches), start=1):
        sequential_ntp_write(states['ordinary'], cache['keys'], cache['targets'],
                             eta=adapter['eta'], ridge=adapter['ridge'])
        for layer, weight in states['protected'].items():
            key = cache['keys'][layer][0]
            other_keys = [old for person, old in keys[layer].items() if person != name]
            protected_write(weight, key, cache['targets'][layer][0], other_keys,
                            eta=adapter['eta'], ridge=adapter['ridge'])
            keys[layer][name] = key.detach().clone()
        latest[name] = (value, cache)
        if index not in CHECKPOINTS:
            continue
        rows = []
        for person, (expected, person_cache) in latest.items():
            row = {'name': person, 'expected': expected, 'conditions': {}, 'write_key_errors': {}}
            question = f'请只回答{person}喜欢喝什么饮料。'
            bridges = {layer: FixedWriteKey(person_cache['keys'][layer][0]) for layer in wrappers}
            for method, memory in states.items():
                row['write_key_errors'][method] = {}
                for layer, weight in memory.items():
                    key = person_cache['keys'][layer][0].float()
                    target = person_cache['targets'][layer][0].float()
                    read = F.linear(key, weight.float())
                    row['write_key_errors'][method][str(layer)] = float(
                        (read - target).norm() / target.norm().clamp_min(1e-12))
                for mode in ('natural', 'oracle_key'):
                    before = {layer: bridge.calls for layer, bridge in bridges.items()}
                    with use_query_bridges(wrappers, bridges if mode == 'oracle_key' else None):
                        answer = pilot.generate(model, tokenizer, question, memory)
                    calls = {layer: bridges[layer].calls - before[layer] for layer in bridges}
                    if any((count == 0) if mode == 'oracle_key' else (count != 0)
                           for count in calls.values()):
                        raise RuntimeError('Unexpected oracle intervention calls')
                    row['conditions'][f'{method}_{mode}'] = {
                        'answer': answer['answer'], 'token_ids': answer['token_ids'],
                        'exact': exact(answer['answer'], expected), 'bridge_calls': calls}
            rows.append(row)
        unknown = []
        for person in UNKNOWN:
            for method, memory in states.items():
                with use_query_bridges(wrappers):
                    answer = pilot.generate(model, tokenizer,
                                            f'请只回答{person}喜欢喝什么饮料。', memory)
                unknown.append({'name': person, 'method': method, 'answer': answer['answer']})
        summary = {condition: sum(row['conditions'][condition]['exact'] for row in rows)
                   for condition in rows[0]['conditions']}
        stages.append({'after_events': index, 'known': len(rows), 'summary': summary,
                       'rows': rows, 'unknown': unknown})
        print(json.dumps({'events': index, 'summary': summary}, ensure_ascii=False), flush=True)
    if any(base.sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected input changed')
    result = {'stages': stages, 'source_sha256': hashes,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Previously inspected synthetic people and values; no independent evaluation.',
                              'Identity, boundaries and the correct write key are supplied by the experiment.',
                              'Oracle key is fixed over generation; this is an intervention, not a learned router.',
                              'Adapter trained on isolated ordinary writes, not these shared states.',
                              'No persistence, crash recovery, unknown rejection or open-domain dialogue.']}
    OUT.mkdir()
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                                     encoding='utf-8')
    print(json.dumps({'stages': [stage['summary'] for stage in stages],
                      'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
