"""Reload recovered attention layouts and compare generation and persistent reads."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from time import perf_counter

import torch

from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, SOURCE, AutoTokenizer, encode, exchange, extend_positions, frozen_digest, load_full_streamed,
    load_integrated_runtime, recall_prompts, reply, run_dialogue, settings_for, sha, using,
)
from scripts.train_spora_recurrent_counterfactual import store_json
from scripts.train_spora_hybrid_language_recovery import DEFAULT_LAYOUTS, layout_name, parse_layout
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, evaluate, records
from spora.research_recurrent_decode import load_recurrent_replacements


SINGLE_QUESTIONS = {
    'market-sign': '只回答导览牌最后放在哪儿。',
    'radio-rehearsal': '只回答最新的广播录音时间。',
    'library-storage': '只回答新书标签最后用什么。',
    'printing-handoff': '只回答最后要交几份样张。',
    'seed-workshop': '只回答新牌子上最后写什么。',
}
FIELD_NAMES = {
    'market-sign': ('导览牌摆放点', '回收志愿者'),
    'radio-rehearsal': ('广播录音时间', '剪片头的人'),
    'library-storage': ('新书标签', '未登记书籍的存放处'),
    'printing-handoff': ('交付的样张数量', '领取样张的人'),
    'seed-workshop': ('新牌子上的桌名', '记录工具'),
}


def distance_prompts(tokenizer, source):
    rows = []
    for index, case in enumerate(source['recall']):
        fillers = []
        for offset in range(len(source['filler'])):
            prompt = source['filler'][(offset + index) % len(source['filler'])]
            exchange(fillers, prompt, '收到，接着说吧。')
        for variant, items in enumerate(case['values']):
            values = dict(zip(('old', 'new', 'keep'), items, strict=True))
            initial, update = [], []
            exchange(initial, case['first'].format_map(values), '我记下这项安排了。')
            exchange(update, case['update'].format_map(values), '好，我只修改这一项。')
            for question, required, kind in (
                    (SINGLE_QUESTIONS[case['name']], ('new',), 'single'),
                    (case['question'], ('new', 'keep'), 'dual')):
                query = {'role': 'user', 'content': question}
                near, far = initial + fillers + update, initial + update + fillers
                lengths = [len(encode(tokenizer, [*history, query])) for history in (near, far)]
                if lengths[0] != lengths[1] or not 600 <= lengths[0] <= 750:
                    raise ValueError(f'Unmatched prompt lengths for {case["name"]}: {lengths}')
                for distance, history in zip(('near', 'far'), (near, far), strict=True):
                    rows.append(dict(case=case['name'], variant=variant, length=f'{distance}_{kind}', values=values,
                                     required_keys=required, question=question, history=history, input_tokens=lengths[0]))
    return rows


def distance_summary(dialogue):
    counts = {}
    for row in dialogue['recall']:
        text = row['answer']
        values = row['values']
        counts.setdefault(row['length'], Counter()).update(
            total=1, passed=row['passed'], ended=row['eos'], new=values['new'] in text,
            old=values['old'] in text, retained=values['keep'] in text)
    return counts


def format_fields(fields, values):
    return '\n'.join(f'{field}：{value}' for field, value in zip(fields, values, strict=True))


def formatted_values(answer, fields):
    values = {}
    for line in answer.splitlines():
        match = re.fullmatch(r'\s*([^：:]+)[：:]\s*(.+?)\s*', line)
        if not match or match[1].strip() not in fields or match[1].strip() in values:
            return None
        values[match[1].strip()] = match[2].rstrip('。')
    return tuple(values[field] for field in fields) if values.keys() == set(fields) else None


@torch.inference_mode()
def revision_stages(decoder, tokenizer, prompts):
    settings = settings_for(decoder, tokenizer, 48)
    labels = [tokenizer.encode(letter, add_special_tokens=False) for letter in ('A', 'B')]
    if any(len(tokens) != 1 for tokens in labels) or labels[0] == labels[1]:
        raise ValueError('Choice labels must be distinct single tokens')
    label_ids = [tokens[0] for tokens in labels]
    rows, counts = [], {}
    for row in prompts:
        if row['required_keys'] != ('new', 'keep'):
            continue
        fields = FIELD_NAMES[row['case']]
        values = row['values']
        current = values['new'], values['keep']
        stale = values['old'], values['keep']
        question = f'按当前有效的安排填写以下两行，每行只写字段名和值。\n{format_fields(fields, ("…", "…"))}'
        supplied = f'已核对的当前值如下。\n{format_fields(fields, current)}\n\n{question}'
        history = row['history']
        distance = row['length'].removesuffix('_dual')
        result = dict(case=row['case'], variant=row['variant'], distance=distance, values=values)
        summary = counts.setdefault(distance, Counter())
        summary['total'] += 1
        for name, request in (('direct', question), ('supplied', supplied)):
            messages = [*history, {'role': 'user', 'content': request}]
            length = len(encode(tokenizer, messages))
            if length + settings['limit'] > decoder.position_table_max_position:
                raise ValueError('Revision response exceeds the position table')
            answer = reply(decoder, tokenizer, history, request, **settings)
            if answer['prompt_tokens'] != length:
                raise ValueError('Revision response used a different prompt length')
            text = answer['answer']
            answer.update(exact=bool(answer['eos'] and formatted_values(text, fields) == current),
                          has_new=values['new'] in text, has_old=values['old'] in text,
                          has_keep=values['keep'] in text)
            summary.update({f'{name}_{key}': answer[key] for key in ('exact', 'eos', 'has_new', 'has_old', 'has_keep')})
            result[name] = answer
        margins, lengths = [], []
        for options in ((current, stale), (stale, current)):
            request = ('以下两种填写只有更新字段不同。按当前有效的安排选出正确的一项，只回答 A 或 B。\n'
                       f'A\n{format_fields(fields, options[0])}\n\nB\n{format_fields(fields, options[1])}')
            ids = encode(tokenizer, [*history, {'role': 'user', 'content': request}])
            if len(ids) + 1 > decoder.position_table_max_position:
                raise ValueError('Revision choice exceeds the position table')
            tokens = torch.tensor([ids], device=decoder.base.device)
            logits = decoder(input_ids=tokens, use_cache=False, logits_to_keep=1).logits[0, -1, label_ids].float()
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite choice logits')
            margins.append((logits[0] - logits[1]).item())
            lengths.append(len(ids))
        choice = (margins[0] - margins[1]) / 2
        result['choice'] = dict(correct=choice > 0, order_agrees=margins[0] * margins[1] < 0,
                                margins=margins, average=choice, input_tokens=lengths)
        summary.update(choice_correct=choice > 0, choice_order_agrees=margins[0] * margins[1] < 0,
                       choice_correct_both=margins[0] > 0 and margins[1] < 0)
        rows.append(result)
    return dict(rows=rows, summary=counts)


def main(args):
    started = perf_counter()
    manifest = args.recovery / 'result.json'
    reference = json.loads(manifest.read_text(encoding='utf-8'))
    protected = {path: sha(path) for path in (args.checkpoint, SOURCE, manifest)}
    if reference['sources'][str(args.checkpoint)] != protected[args.checkpoint]:
        raise ValueError('Recovery used a different parent checkpoint')
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    transfer = []
    if args.transfer:
        protected[args.transfer] = sha(args.transfer)
        transfer = json.loads(args.transfer.read_text(encoding='utf-8'))
        questions = {turn[0] for case in source['open'] for turn in case['turns']}
        if not transfer or any(turn[0] in questions for case in transfer for turn in case['turns']):
            raise ValueError('Transfer dialogues must use distinct questions')
    layouts = tuple(tuple(layout) for layout in reference.get('layouts', DEFAULT_LAYOUTS))
    if (not layouts or any(parse_layout(','.join(map(str, layout))) != layout
                           or layout_name(layout) not in reference['results'] for layout in layouts)
            or len(set(layouts)) != len(layouts)):
        raise ValueError('Invalid layout manifest')
    if args.layout:
        if len(set(args.layout)) != len(args.layout) or not set(args.layout) <= set(layouts):
            raise ValueError('Requested layouts must be distinct and present in the recovery manifest')
        layouts = tuple(args.layout)
    if args.recall_transfer:
        protected[args.recall_transfer] = sha(args.recall_transfer)
        additional_recall = json.loads(args.recall_transfer.read_text(encoding='utf-8'))
        previous = {row['question'] for row in source['recall']}
        fresh = {row['question'] for row in additional_recall}
        if not fresh or len(fresh) != len(additional_recall) or fresh & previous:
            raise ValueError('Transfer recall must use distinct questions outside the existing suite')
        source['recall'] = additional_recall
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if args.revision_distance or args.revision_stages:
        prompts = distance_prompts(tokenizer, source)
        source['open'] = []
    else:
        prompts = recall_prompts(tokenizer, source, extended=True)
        if args.recall_transfer:
            prompts = [row for row in prompts if row['length'] != 'short']
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    base.lm_head.to(base.device)
    identity = frozen_digest(runtime.organism, runtime.state)
    if identity != reference['individual_sha256']:
        raise ValueError('Recovery used a different individual')
    memory = records(runtime.organism, HELD)
    parent_transfer = run_dialogue(decoder, tokenizer, [], transfer) if transfer else None
    results = {}
    if args.revision_stages:
        results['parent'] = revision_stages(decoder, tokenizer, prompts)
        print('parent', results['parent']['summary'], flush=True)
    elif args.revision_distance:
        dialogue = run_dialogue(decoder, tokenizer, prompts, [])
        persistent = evaluate(decoder, tokenizer, runtime.organism, memory)
        results['parent'] = dict(dialogue=dialogue, memory=persistent, distance=distance_summary(dialogue))
        print('parent', results['parent']['distance'], persistent['summary'], flush=True)
    for layout in layouts:
        name = layout_name(layout)
        directories = [args.recovery / name / f'layer-{index}' for index in layout]
        hashes = reference['results'][name].get('candidate_hashes')
        for directory in directories:
            for filename in ('attention.pt', 'comparison.json'):
                path = directory / filename
                protected[path] = sha(path)
        if hashes is not None and any(protected[directory / 'attention.pt'] != hashes.get(str(index))
                                      for index, directory in zip(layout, directories, strict=True)):
            raise ValueError('Recovered weights differ from the training result')
        replacements = load_recurrent_replacements(decoder, directories, protected[args.checkpoint])
        if set(replacements) != set(layout):
            raise ValueError('Loaded layout differs from the training result')
        if args.revision_stages:
            with using(replacements, 'recurrent'):
                results[name] = revision_stages(decoder, tokenizer, prompts)
            print(name, results[name]['summary'], flush=True)
            del replacements
            torch.cuda.empty_cache()
            continue
        with using(replacements, 'recurrent'):
            dialogue = run_dialogue(decoder, tokenizer, prompts, source['open'])
            additional = run_dialogue(decoder, tokenizer, [], transfer) if transfer else None
            persistent = evaluate(decoder, tokenizer, runtime.organism, memory)
        results[name] = dict(dialogue=dialogue, memory=persistent)
        if args.revision_distance:
            results[name]['distance'] = distance_summary(dialogue)
            print(name, results[name]['distance'], flush=True)
        if additional is not None:
            results[name]['transfer'] = additional
        print(name, dialogue['recall_passed'], dialogue['open_turns_passed'], persistent['summary'], flush=True)
        if additional is not None:
            print(name, 'transfer', additional['open_turns_passed'], additional['open_passed'], flush=True)
        del replacements
        torch.cuda.empty_cache()
    if identity != frozen_digest(runtime.organism, runtime.state) or any(
            sha(path) != digest for path, digest in protected.items()):
        raise RuntimeError('Source or individual changed in evaluation')
    result = dict(results=results, individual_sha256=identity,
                  sources={str(path): digest for path, digest in protected.items()}, seconds=perf_counter() - started)
    if parent_transfer is not None:
        result['parent_transfer'] = parent_transfer
    store_json(args.output, result)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('recovery', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--transfer', type=Path, help='Additional open multi-turn dialogue cases')
    cli.add_argument('--recall-transfer', type=Path, help='Replace recall cases with an independent long-dialogue set')
    cli.add_argument('--layout', type=parse_layout, action='append', help='Evaluate only the requested recovered layouts')
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--revision-distance', action='store_true', help='Compare edit distance and answer width on recall transfer')
    mode.add_argument('--revision-stages', action='store_true', help='Compare choice and formatted generation on recall transfer')
    args = cli.parse_args()
    if args.output.exists():
        cli.error('Choose a new output path')
    if (args.revision_distance or args.revision_stages) and not args.recall_transfer:
        cli.error('Revision comparisons require --recall-transfer')
    if args.revision_stages and args.transfer:
        cli.error('--revision-stages does not accept --transfer')
    main(args)
