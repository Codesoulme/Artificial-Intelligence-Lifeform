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


def distance_prompts(tokenizer, source, *, swap_values=False):
    rows = []
    for index, case in enumerate(source['recall']):
        fillers = []
        for offset in range(len(source['filler'])):
            prompt = source['filler'][(offset + index) % len(source['filler'])]
            exchange(fillers, prompt, '收到，接着说吧。')
        for variant, items in enumerate(case['values']):
            if swap_values:
                items = items[1], items[0], items[2]
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


@torch.inference_mode()
def score_value(decoder, tokenizer, prefix, prefix_ids, value):
    full = tokenizer.encode(prefix + value, add_special_tokens=False)
    target = full[len(prefix_ids):]
    if not target or full[:len(prefix_ids)] != prefix_ids:
        raise ValueError('Candidate value changes the tokenized prompt')
    if len(full) > decoder.position_table_max_position:
        raise ValueError('Candidate value exceeds the position table')
    inputs = torch.tensor([full[:-1]], device=decoder.base.device)
    labels = torch.tensor(target, device=inputs.device)
    logits = decoder(input_ids=inputs, use_cache=False, logits_to_keep=len(target)).logits[0].float()
    log_probs = logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(-1)
    if not torch.isfinite(log_probs).all():
        raise ValueError('Nonfinite candidate value likelihood')
    return dict(value=value, token_ids=target, log_probs=log_probs.tolist(),
                mean=log_probs.mean().item(), total=log_probs.sum().item())


def revision_likelihood(decoder, tokenizer, prompts, *, saved=None):
    rows, counts = [], {}
    cases = [(original, swapped) for original, swapped in zip(*prompts, strict=True)
             if original['length'] in ('near_single', 'far_single')]
    settings = settings_for(decoder, tokenizer, 48) if saved is not None else None
    reference_rows = saved['rows'] if saved is not None else [None] * len(cases)
    for (original, swapped), reference in zip(cases, reference_rows, strict=True):
        distance = original['length']
        field = FIELD_NAMES[original['case']][0]
        question = f'按当前有效的安排填写「{field}」，只写这个字段的值。'
        row = dict(case=original['case'], variant=original['variant'], distance=distance)
        if reference is not None and any(reference[key] != value for key, value in row.items()):
            raise ValueError('Saved score is for a different scenario')
        summary = counts.setdefault(distance, Counter())
        for name, history in (('original', original), ('swapped', swapped)):
            messages = [*history['history'], {'role': 'user', 'content': question}]
            prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + '</think>\n\n'
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            values = history['values']
            if reference is None:
                scores = {role: score_value(decoder, tokenizer, prefix, prefix_ids, values[role])
                          for role in ('new', 'old')}
            else:
                scores = reference[name]['scores']
                if reference[name]['input_tokens'] != len(prefix_ids) or any(
                        scores[role]['value'] != values[role] or
                        tokenizer.encode(prefix + values[role], add_special_tokens=False) !=
                        prefix_ids + scores[role]['token_ids'] for role in ('new', 'old')):
                    raise ValueError('Saved score does not match the prompt and candidates')
            row[name] = dict(scores=scores, input_tokens=len(prefix_ids),
                             mean_margin=scores['new']['mean'] - scores['old']['mean'],
                             total_margin=scores['new']['total'] - scores['old']['total'])
            if reference is not None:
                if len(prefix_ids) + settings['limit'] > decoder.position_table_max_position:
                    raise ValueError('Generation exceeds the position table')
                answer = reply(decoder, tokenizer, history['history'], question, **settings,
                               expected_prompt=prefix_ids)
                text = answer['answer']
                new_tokens = scores['new']['token_ids']
                answer.update(has_new=values['new'] in text, has_old=values['old'] in text,
                              starts_new=answer['token_ids'][:1] == new_tokens[:1],
                              exact=bool(answer['eos'] and text.rstrip('。') == values['new']))
                row[name]['generation'] = answer
                prefers_new = row[name]['mean_margin'] > 0
                summary.update(generation_exact=answer['exact'], generation_ended=answer['eos'],
                               generation_has_new=answer['has_new'], generation_has_old=answer['has_old'],
                               prefers_new_but_missing=prefers_new and not answer['has_new'],
                               prefers_new_but_old=prefers_new and answer['has_old'],
                               prefers_new_but_starts_elsewhere=prefers_new and not answer['starts_new'],
                               prefers_new_starts_new_but_missing=prefers_new and answer['starts_new']
                               and not answer['has_new'])
        first, second = row['original'], row['swapped']
        summary.update(pairs=1, mean_correct=sum(item['mean_margin'] > 0 for item in (first, second)),
                       total_correct=sum(item['total_margin'] > 0 for item in (first, second)),
                       both_mean=first['mean_margin'] > 0 and second['mean_margin'] > 0,
                       both_total=first['total_margin'] > 0 and second['total_margin'] > 0,
                       revision_gain=first['mean_margin'] + second['mean_margin'] > 0)
        rows.append(row)
    return dict(rows=rows, summary=counts)


def revision_field_switch(decoder, tokenizer, prompts):
    settings = settings_for(decoder, tokenizer, 48)
    rows, counts = [], {}
    for row in prompts:
        if row['required_keys'] != ('new',):
            continue
        distance = row['length'].removesuffix('_single')
        history, values = row['history'], row['values']
        responses = {}
        for role, field in zip(('new', 'keep'), FIELD_NAMES[row['case']], strict=True):
            question = f'按当前有效的安排填写「{field}」，只写这个字段的值。'
            prefix = encode(tokenizer, [*history, {'role': 'user', 'content': question}])
            if len(prefix) + settings['limit'] > decoder.position_table_max_position:
                raise ValueError('Field switch prompt exceeds the position table')
            answer = reply(decoder, tokenizer, history, question, **settings, expected_prompt=prefix)
            text = answer['answer']
            other = 'keep' if role == 'new' else 'new'
            answer.update(field=field, question=question, expected=values[role],
                          exact=answer['eos'] and text.rstrip('。') == values[role],
                          has_expected=values[role] in text, has_other=values[other] in text,
                          has_old=values['old'] in text)
            responses[role] = answer
            counts.setdefault(f'{distance}_{role}', Counter()).update(
                total=1, exact=answer['exact'], ended=answer['eos'], contains=answer['has_expected'],
                other_only=answer['has_other'] and not answer['has_expected'], old=answer['has_old'])
        counts.setdefault(distance, Counter()).update(
            pairs=1, both_exact=all(answer['exact'] for answer in responses.values()))
        rows.append(dict(case=row['case'], variant=row['variant'], distance=distance, values=values,
                         responses=responses))
    return dict(rows=rows, summary=counts)


def library_order_prompts(tokenizer, source):
    case = next(row for row in source['recall'] if row['name'] == 'library-storage')
    variants = {}
    for name, flipped in (('original', ()), ('initial_reversed', ('first',)),
                          ('revision_reversed', ('update',)), ('both_reversed', ('first', 'update'))):
        reordered = case.copy()
        for turn in flipped:
            left, right = case[turn].removesuffix('。').rsplit('，', 1)
            if ('{keep}' in right) != (turn == 'first'):
                raise ValueError('The library clauses do not match the expected field order')
            reordered[turn] = f'{right}，{left}。'
        recall = [reordered if entry is case else entry for entry in source['recall']]
        variants[name] = [row for row in distance_prompts(tokenizer, {**source, 'recall': recall})
                          if row['case'] == case['name'] and row['required_keys'] == ('new',)]
    return variants


def revision_field_order(decoder, tokenizer, prompts):
    results = {}
    for order, rows in prompts.items():
        results[order] = revision_field_switch(decoder, tokenizer, rows)
        if order == 'original':
            continue
        original = results['original']['rows']
        replay = results[order]['rows']
        for before, after in zip(original, replay, strict=True):
            for role in ('new', 'keep'):
                if before['responses'][role]['prompt_tokens'] != after['responses'][role]['prompt_tokens']:
                    raise ValueError('Field order changed the prompt token count')
    return dict(conditions=results, summary={order: result['summary'] for order, result in results.items()})


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
    revision_audit = {
        'revision_likelihood': revision_likelihood,
        'revision_stages': revision_stages,
        'revision_field_switch': revision_field_switch,
        'revision_field_order': revision_field_order,
    }.get(next((name for name in ('revision_likelihood', 'revision_stages', 'revision_field_switch', 'revision_field_order')
                if getattr(args, name)), None))
    if args.revision_field_order:
        prompts = library_order_prompts(tokenizer, source)
    elif args.revision_distance or revision_audit:
        prompts = distance_prompts(tokenizer, source)
        source['open'] = []
        if args.revision_likelihood:
            prompts = prompts, distance_prompts(tokenizer, source, swap_values=True)
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
    saved_likelihood = None
    if args.revision_likelihood_reference:
        path = args.revision_likelihood_reference
        protected[path] = sha(path)
        saved_likelihood = json.loads(path.read_text(encoding='utf-8'))
        if saved_likelihood['individual_sha256'] != identity:
            raise ValueError('Saved likelihood used a different individual')

    def run_revision(name):
        if saved_likelihood is None:
            return revision_audit(decoder, tokenizer, prompts)
        expected_sources = {Path(path).resolve(): digest for path, digest in saved_likelihood['sources'].items()}
        if any(expected_sources.get(path.resolve()) != digest for path, digest in protected.items()
               if path != args.revision_likelihood_reference):
            raise ValueError('Saved likelihood used different inputs')
        return revision_likelihood(decoder, tokenizer, prompts, saved=saved_likelihood['results'][name])

    results = {}
    if revision_audit:
        results['parent'] = run_revision('parent')
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
        if revision_audit:
            with using(replacements, 'recurrent'):
                results[name] = run_revision(name)
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
    mode.add_argument('--revision-likelihood', action='store_true', help='Score matched old and new values under reversed revisions')
    mode.add_argument('--revision-field-switch', action='store_true', help='Change only the requested field on each dialogue')
    mode.add_argument('--revision-field-order', action='store_true', help='Reorder the two library fields within each source turn')
    cli.add_argument('--revision-likelihood-reference', type=Path,
                     help='Reuse verified likelihoods and generate answers with the same prompts')
    args = cli.parse_args()
    if args.output.exists():
        cli.error('Choose a new output path')
    revision_probes = args.revision_stages or args.revision_likelihood or args.revision_field_switch or args.revision_field_order
    if (args.revision_distance or revision_probes) and not args.recall_transfer:
        cli.error('Revision comparisons require --recall-transfer')
    if revision_probes and args.transfer:
        cli.error('Revision probes do not accept --transfer')
    if args.revision_likelihood_reference and not args.revision_likelihood:
        cli.error('--revision-likelihood-reference requires --revision-likelihood')
    main(args)
