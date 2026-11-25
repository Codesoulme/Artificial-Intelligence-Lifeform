"""Swap either of two isolated fast-weight records under an unchanged question."""

import argparse
from collections import Counter
from itertools import permutations
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_fast_weight_lexical_write import SOURCE, base, compress_dimension, tokens
from scripts.audit_spora_version_coverage import write_fresh
from scripts.audit_spora_version_reply_actions import FACTS, LIMIT, TRAINING
from scripts.audit_spora_versioned_memory_generation import MEMORY_FRACTION, NEUTRAL, TRAIN_NAMES, frozen_features, read_versions
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_read_adapter_pilot import generate
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_versioned_memory_pointer import VersionReplyActions


SEED = 32609
FIELDS = {'drinks': '饮品', 'places': '见面地点', 'items': '用品', 'books': '书目'}
EVENTS = {
    '闻韶宁': {'drinks': ('麦芽酸奶', '陈皮茶'), 'places': ('听涛长廊', '海棠茶室')},
    '方吟晚': {'places': ('旧邮局', '东巷展馆'), 'items': ('青瓷花瓶', '木柄剪刀')},
    '许照岚': {'items': ('蓝釉烛台', '软木杯垫'), 'drinks': ('冬瓜奶露', '姜枣豆乳')},
    '丁蔓庭': {'books': ('秋日来信', '白帆远行'), 'drinks': ('薄荷豆浆', '烘麦红茶')},
    '叶昕卓': {'places': ('南桥驿站', '芦苇栈道'), 'books': ('山间光影', '沿海旧照')},
    '贾宁杉': {'items': ('皮革尺套', '格纹手套'), 'places': ('云杉影院', '南坡书店')},
}
QUESTIONS = (
    '核对{name}留下的两栏，先报{first}，接着报{second}，都写全。',
    '跟我说说{name}登记过哪个{first}，以及哪项{second}，按这个顺序列。',
)


def training_rows(tokenizer):
    rows = []

    def add(question, parts):
        tail, actions, text = [], [], []
        for slot, piece in parts:
            ids = tokens(tokenizer, piece)
            tail.extend(ids)
            text.append(piece)
            if slot is None:
                actions.extend(('text', token) for token in ids)
            else:
                actions.append(('copy', slot))
                actions.extend(('skip', slot) for _ in ids[1:])
        if tokenizer.decode(tail) != ''.join(text):
            raise ValueError('Training pieces changed the decoded text')
        tail.append(tokenizer.eos_token_id)
        actions.append(('text', tokenizer.eos_token_id))
        rows.append((question, tail, actions, None))

    for index, name in enumerate(TRAIN_NAMES):
        for first, second in permutations(TRAINING, 2):
            a, b = FIELDS[first], FIELDS[second]
            values_a, values_b = TRAINING[first][2], TRAINING[second][2]
            value_a = values_a[index % len(values_a)]
            value_b = values_b[(index + 1) % len(values_b)]
            add(f'{name}登记的{a}和{b}是什么？只报名称。',
                ((0, value_a), (None, '，'), (1, value_b), (None, '。')))
            add(f'把{name}的{a}与{b}按顺序说完。',
                ((None, '记下的是'), (0, value_a), (None, '、'), (1, value_b), (None, '。')))
            add(f'请依次列出{name}的{a}和{b}，每项带上标签。',
                ((None, f'{a}：'), (0, value_a), (None, f'；{b}：'), (1, value_b), (None, '。')))
        for domain, label in FIELDS.items():
            if domain in TRAINING:
                values = TRAINING[domain][2]
                add(f'查一下{name}的{label}，给我名字就好。', ((0, values[index % len(values)]),))
    return rows


def fit(model, tokenizer, head, rows, steps):
    features, sizes = frozen_features(model, tokenizer, rows)
    features = features.clone()
    scaffold = {token: head.slots + index for index, token in enumerate(head.scaffold_tokens.tolist())}
    targets, start = [], 0
    for (_, _, actions, _), size in zip(rows, sizes, strict=True):
        copied = 0
        for offset, (kind, value) in enumerate(actions):
            if kind == 'text':
                targets.append((start + offset, copied, scaffold[value]))
            elif kind == 'copy':
                targets.append((start + offset, copied, value))
                copied |= 1 << value
        start += size
    targets = torch.tensor(targets)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    rng = torch.Generator().manual_seed(SEED)
    for step in range(steps):
        indices, progress, labels = targets[torch.randperm(len(targets), generator=rng)[:64]].T
        loss = F.cross_entropy(head(features[indices], progress), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, steps // 2, steps - 1):
            print('fit', step + 1, float(loss.detach()), flush=True)


@torch.inference_mode()
def reply(model, tokenizer, head, question, memories, *, require_coverage=False, history=(),
          max_new_tokens=LIMIT, memory_fraction=MEMORY_FRACTION):
    if not any(value is not None for value in memories):
        return {**generate(model, tokenizer, question, history=history, max_new_tokens=max_new_tokens), 'copied': []}
    require_coverage = require_coverage and all(value is not None for value in memories)
    inputs = closed_think_inputs(tokenizer, question, model.device, history=history)['input_ids']
    eos = tokenizer.eos_token_id
    tail, copied, past = [], [], None
    mask = 0
    while len(tail) < max_new_tokens:
        output = model.model(input_ids=inputs, past_key_values=past, use_cache=True)
        past = output.past_key_values
        hidden = output.last_hidden_state[0, -1]
        available = tuple(value is not None and len(value) <= max_new_tokens - len(tail) for value in memories)
        pending = any(fits and not (mask & (1 << slot)) for slot, fits in enumerate(available))
        slot, token = head.choose_slot(hidden, model.lm_head(hidden), copied=mask, available=available,
                                       memory_fraction=memory_fraction,
                                       blocked_token=eos if require_coverage and pending else None)
        if slot is not None:
            copied.append({'slot': slot, 'at': len(tail)})
            mask |= 1 << slot
            memory = memories[slot]
            tail.extend(memory.tolist())
            inputs = memory.unsqueeze(0)
        else:
            tail.append(token)
            if token == eos:
                break
            inputs = torch.tensor([[token]], device=inputs.device)
    return {'answer': tokenizer.decode(tail, skip_special_tokens=True).strip(), 'token_ids': tail, 'copied': copied}


def score(output, expected, foils, eos):
    text = output['answer']
    count = [text.count(value) for value in expected]
    ended = bool(output['token_ids'] and output['token_ids'][-1] == eos)
    stale = any(value in text for value in foils)
    order = all(text.find(a) < text.find(b) for a, b in zip(expected, expected[1:]))
    return {'slots': [n == 1 for n in count], 'foil': stale, 'ended': ended,
            'all': ended and all(n == 1 for n in count) and not stale and order}


def evaluate(model, tokenizer, head, memory, *, require_coverage=False):
    rows, counts = [], Counter()
    eos = tokenizer.eos_token_id
    for name, events in EVENTS.items():
        singles = {}
        for field, (old, current) in events.items():
            question = f'查一查{name}记过的{FIELDS[field]}，只给名字。'
            output = reply(model, tokenizer, head, question, (memory[field][name][1], None))
            single = score(output, [current], [old], eos)
            counts['single'] += single['all']
            singles[field] = {'question': question, 'expected': current, **output, 'score': single}
        counts['both_singles'] += all(item['score']['all'] for item in singles.values())
        for index, form in enumerate(QUESTIONS):
            fields = tuple(events) if index == 0 else tuple(reversed(events))
            question = form.format(name=name, first=FIELDS[fields[0]], second=FIELDS[fields[1]])
            if any(value in question for domain in fields for value in events[domain]):
                raise ValueError('Memory value leaked into the question')
            results = {}
            for mode, versions in (('current', (1, 1)), ('swap_first', (0, 1)),
                                   ('swap_second', (1, 0)), ('both_old', (0, 0))):
                expected = [events[field][version] for field, version in zip(fields, versions, strict=True)]
                foils = [events[field][1 - version] for field, version in zip(fields, versions, strict=True)]
                values = tuple(memory[field][name][version] for field, version in zip(fields, versions, strict=True))
                output = reply(model, tokenizer, head, question, values, require_coverage=require_coverage)
                result = score(output, expected, foils, eos)
                results[mode] = {'expected': expected, 'foils': foils, **output, 'score': result}
                counts[f'{mode}_both'] += result['all']
                counts[f'{mode}_foil'] += result['foil']
                counts[f'{mode}_first'] += result['slots'][0]
                counts[f'{mode}_second'] += result['slots'][1]
                counts[f'{mode}_ended'] += result['ended']
            counts['four_way'] += all(row['score']['all'] for row in results.values())

            empty = reply(model, tokenizer, head, question, (None, None), require_coverage=require_coverage)
            if {key: empty[key] for key in ('answer', 'token_ids')} != generate(model, tokenizer, question):
                raise ValueError('Empty memory changed frozen generation')
            counts['empty_with_value'] += any(value in empty['answer'] for domain in fields for value in events[domain])
            rows.append({'name': name, 'question': question, 'fields': fields, 'outputs': results,
                         'single': [singles[field] for field in fields], 'empty': empty})
        print('evaluated', len(rows), flush=True)
    neutral = {question: reply(model, tokenizer, head, question, (None, None), require_coverage=require_coverage)
               for question in NEUTRAL}
    return dict(counts), rows, neutral


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    model_path, adapter = base.MODEL / 'model.safetensors', SOURCE / 'adapters.pt'
    protected = {str(path): base.sha(path) for path in (model_path, adapter, base.MODEL / 'tokenizer.json',
                                                       base.MODEL / 'tokenizer_config.json')}
    if protected[str(model_path)] != base.WEIGHT_SHA:
        raise ValueError('Expected frozen DeepSeek checkpoint')
    result_path, version_path, head_path = (args.output / name for name in ('result.json', 'versions.pt', 'head.pt'))
    expected = json.loads(result_path.read_text(encoding='utf-8')) if args.verify else None
    if expected is not None and (protected != expected['sources'] or base.sha(version_path) != expected['versions_sha256']
                                 or base.sha(head_path) != expected['head_sha256']):
        raise ValueError('A replay artifact changed')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    rows = training_rows(tokenizer)
    train_values = {value for _, _, values in TRAINING.values() for value in values}
    held_values = [value for events in EVENTS.values() for values in events.values() for value in values]
    if (set(TRAIN_NAMES) & set(EVENTS) or train_values & set(held_values) or len(held_values) != len(set(held_values))
            or any(not 2 <= len(tokens(tokenizer, value)) <= 4 for value in held_values)):
        raise ValueError('Expected distinct held-out names and values of supported width')

    model = load_full_streamed()
    embeddings = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embeddings.shape[1]).to(embeddings.device)
    if expected is None:
        settings = torch.load(adapter, map_location='cpu', weights_only=True)
        banks = {}
        for field in FIELDS:
            records = [(name, *events[field]) for name, events in EVENTS.items() if field in events]
            banks[field] = write_fresh(model, tokenizer, columns, layer=settings['layers'][0], eta=settings['eta'],
                                       ridge=settings['ridge'], events={field: records}, facts=FACTS)
        atomic_save(banks, version_path)
    banks = torch.load(version_path, map_location='cpu', weights_only=True)
    codebook = F.normalize(embeddings.index_select(1, columns).float(), dim=-1)
    memory = {field: read_versions(bank, codebook) for field, bank in banks.items()}
    del codebook
    for name, events in EVENTS.items():
        for field, values in events.items():
            if [tokenizer.decode(ids.tolist()) for ids in memory[field][name]] != list(values):
                raise ValueError('The version tensors failed to recover an evaluation field')

    scaffold = sorted({value for _, _, actions, _ in rows for kind, value in actions if kind == 'text'})
    torch.manual_seed(SEED)
    head = VersionReplyActions(model.config.hidden_size, scaffold, slots=2)
    if expected is None:
        fit(model, tokenizer, head, rows, args.steps)
        atomic_save(head.state_dict(), head_path)
    else:
        head.load_state_dict(torch.load(head_path, map_location='cpu', weights_only=True))
    head.eval().to(model.device)
    summary, cases, neutral = evaluate(model, tokenizer, head, memory)
    if any(base.sha(Path(path)) != value for path, value in protected.items()):
        raise ValueError('A protected source changed')
    result = {'sources': protected, 'versions_sha256': base.sha(version_path), 'head_sha256': base.sha(head_path),
              'steps': args.steps if expected is None else expected['steps'], 'training_rows': len(rows),
              'summary': summary, 'cases': cases, 'neutral': neutral}
    if expected is not None and json.loads(json.dumps(result)) != {key: expected[key] for key in result}:
        raise ValueError('Independent reload changed a recorded output')
    store_json(args.output / ('reload.json' if args.verify else 'result.json'),
               {**result, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})
    print(summary, flush=True)
    if args.completion_control:
        control_start = perf_counter()
        control, rows, inactive = evaluate(model, tokenizer, head, memory, require_coverage=True)
        if inactive != neutral or any((after['single'], after['empty']) != (before['single'], before['empty'])
                                      for before, after in zip(cases, rows, strict=True)):
            raise ValueError('Completion control changed a single-field or empty answer')
        if any(base.sha(Path(path)) != value for path, value in protected.items()):
            raise ValueError('A protected source changed during completion control')
        store_json(args.output / 'completion-control.json', {
            **result, 'summary': control, 'cases': rows, 'neutral': inactive,
            'seconds': perf_counter() - control_start, 'peak_bytes': torch.cuda.max_memory_allocated(),
        })
        print('completion control', control, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=220)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--completion-control', action='store_true')
    options = parser.parse_args()
    if options.steps < 1:
        parser.error('At least one training step is required')
    if options.completion_control and not options.verify:
        parser.error('Completion control requires replaying the frozen baseline first')
    main(options)
