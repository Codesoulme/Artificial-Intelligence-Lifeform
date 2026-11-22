"""Compare fixed-prefix and distributed anchors at equal cache capacity."""

import argparse
import json
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from scripts.audit_spora_local_attention import answer, bounded_generation, cache_probe, memory_answers
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, prepare, records
from spora.integrated_checkpoint import _digest, load_integrated_runtime
from spora.language_training import frozen_digest
from spora.research_local_attention import anchor_attention, anchored_cache


def load_without_layout(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    meta, tensors = payload['metadata'], payload['tensors']
    if _digest(meta, tensors) != payload['integrity_sha256'] or meta['position_table_max_position'] != 1536:
        raise ValueError('Expected the verified 1536-position checkpoint')
    layout = meta.pop('anchored_attention')
    if layout['prefix'] != 64 or layout['window'] != 256:
        raise ValueError('Expected the verified 64/256 attention budget')
    payload['integrity_sha256'] = _digest(meta, tensors)
    with tempfile.TemporaryDirectory(prefix='spora-landmarks-') as folder:
        unmasked = Path(folder) / 'reference.pt'
        torch.save(payload, unmasked)
        runtime = load_integrated_runtime(unmasked, load_full_streamed(), MODEL)
    return runtime, tuple(layout['layers'])


def fact_cases(tokenizer, prose):
    specs = (
        (0, '排练通知写着集合点为松亭站。', '排练通知写的是哪个集合点？只答站名。', '松亭站'),
        (57, '缴费单注明应付款为七十九元。', '缴费单上的金额是多少？只答金额。', '七十九'),
        (117, '包裹登记栏记着收件人为许芸。', '包裹的收件人叫什么？只答姓名。', '许芸'),
    )
    for shift, fact, question, expected in specs:
        filler = prose[shift:shift + 1000]
        for offset in (0, 288, 544, 928):
            text = (f'{tokenizer.decode(filler[:offset])}\n\n{fact}\n\n'
                    f'{tokenizer.decode(filler[offset:])}\n\n{question}')
            yield text, offset, expected


def revision_cases(tokenizer, prose):
    records = (
        ('巡检', '集合站', '石桥站', '杏林站', '月台站'),
        ('展览', '布展组长', '顾晓岚', '沈秋禾', '陆明遥'),
        ('公演', '领票地点', '西塔大厅', '北桥大厅', '东岸大厅'),
        ('维修', '备用门牌', '甲七号', '辛六号', '乙二号'),
    )
    for index, (event, field, earlier, current, other) in enumerate(records):
        filler = prose[:1000]
        old = f'{event}一月的登记单：{field}填写的是{earlier}。'
        new = f'{event}二月更新后，{field}登记为{current}。'
        # Reverse chronological order for half the prompts so recency cannot solve the task.
        first, second = (old, new) if index % 2 == 0 else (new, old)
        distractor = f'另一场活动也填了{field}，内容是{other}。'
        sections = (tokenizer.decode(filler[:276]), first, tokenizer.decode(filler[276:560]), second,
                    tokenizer.decode(filler[560:892]), distractor, tokenizer.decode(filler[892:]))
        document = '\n\n'.join(sections)
        for period, expected, rejected in (('二月更新后', current, earlier), ('一月登记时', earlier, current)):
            question = f'请问{event}{period}的{field}是什么？只回答登记的内容。'
            yield f'{document}\n\n{question}', event, period, expected, (rejected, other)


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    source = ROOT / 'data/chinese-pilot/source.json'
    sources = {str(path): sha(path) for path in (args.checkpoint, source)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    data = json.loads(source.read_text(encoding='utf-8'))
    prose = tokenizer.encode('\n'.join(data['train'] + data['test'] + data['development']),
                             add_special_tokens=False)
    if len(prose) < 1117:
        raise ValueError('Expected enough prose for paired long prompts')
    runtime, layers = load_without_layout(args.checkpoint)
    decoder = runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    fixtures = records(runtime.organism, HELD)
    controls = [prepare(runtime.organism, fixtures[i]) for i in (0, 1, 6, 7, 16, 17, 22, 23)]
    fixed = tuple(position for start in range(80, 1536, 128) for position in range(start, start + 4))
    variants = {'full': (0, ()), 'fixed_prefix': (64, ()),
                'strided': (16, tuple(range(24, 24 + 48 * 31, 31))), 'blocks': (16, fixed)}
    if args.snapshot_only:
        variants = {'query_snapshot': (16, ())}
    elif args.full_prefill_control:
        variants = {'fixed_after_full_prefill': (64, ())}
    elif args.revisions:
        variants = {'full': (0, ()), 'fixed_after_full_prefill': (64, ()), 'query_snapshot': (16, ())}
        test_prose = tokenizer.encode('\n'.join(data['test'] + data['development'] + data['train']),
                                      add_special_tokens=False)
        dated_facts = tuple(revision_cases(tokenizer, test_prose))
    results = {}
    for name, (prefix, landmarks) in variants.items():
        snapshot = 48 if name == 'query_snapshot' else 0
        full_prefill = snapshot > 0 or name == 'fixed_after_full_prefill'
        scope = nullcontext()
        binder = nullcontext()

        def make_cache():
            if prefix:
                return anchored_cache(decoder.base.config, layers, 256, prefix, landmarks=landmarks)
            return DynamicCache(config=decoder.base.config)

        if prefix:
            scope = anchor_attention(decoder, layers, 256, prefix, landmarks=landmarks, snapshot=snapshot,
                                     full_prefill=full_prefill)
            binder = bounded_generation(decoder, make_cache)
        with scope, binder:
            cache, _ = cache_probe(decoder, torch.tensor([prose[:1024]], device='cuda'), make_cache)
            if full_prefill:
                cache['full_continuation_argmax_equal'] = cache.pop('continuation_argmax_equal')
                cache['full_max_logit_delta'] = cache.pop('max_logit_delta')
            facts = []
            if args.revisions:
                for text, event, period, expected, rejected in dated_facts:
                    row = answer(decoder, tokenizer, text, limit=24)
                    facts.append(dict(row, event=event, period=period, expected=expected, rejected=rejected,
                                      correct=expected in row['text'] and all(x not in row['text'] for x in rejected),
                                      exact=row['text'].strip('。！!，,；;？? \n') == expected))
            else:
                for text, offset, expected in fact_cases(tokenizer, prose):
                    row = answer(decoder, tokenizer, text, limit=24)
                    accepted = (expected, '79') if expected == '七十九' else (expected,)
                    facts.append(dict(row, offset=offset, expected=expected,
                                      correct=any(choice in row['text'] for choice in accepted)))
            memory = memory_answers(decoder, tokenizer, tokenizer.decode(prose[:1000]), controls)
        results[name] = {'prefix': prefix, 'landmarks': landmarks, 'snapshot': snapshot, 'cache': cache,
                         'full_prefill': full_prefill, 'facts': facts, 'memory': memory}
        print(name, 'facts', sum(row['correct'] for row in facts),
              'memory', sum(row['text'] == row['expected'] for row in memory),
              'KV bytes', cache['cache_bytes'], flush=True)
    if individual != frozen_digest(runtime.organism, runtime.state) or any(
        sha(Path(path)) != digest for path, digest in sources.items()
    ):
        raise ValueError('The audit changed a protected source')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, {'results': results, 'bounded_layers': layers, 'sources': sources,
                        'seconds': time.perf_counter() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--snapshot-only', action='store_true', help='Select old keys from the end of the prompt')
    modes.add_argument('--full-prefill-control', action='store_true', help='Trim the fixed prefix after full prefill')
    modes.add_argument('--revisions', action='store_true', help='Compare dated values with nearby distractors')
    main(parser.parse_args())
