"""Test which pending revision a follow-up confirmation refers to."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_clause_intent import clause_features
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.prepare_spora_revision_samples import dialogue_samples
from scripts.train_spora_revision_intent import (
    DATA, DESIGN, EPISODES, EXTRA_NAMES, HELD, MODEL, OBSERVATIONS, ROLE_NAMES, AutoTokenizer, atomic_save, evaluate,
    extend_positions, frozen_digest, held_episodes, install, load_full_streamed, load_integrated_runtime,
    load_recurrent_replacements, packed, read, records, sha, store_json,
)
from scripts.train_spora_observed_episodes import encode_changes, state_for, state_values
from spora.research_clause_intent import ClauseIntent
from spora.research_revision_draft import RevisionDecision, RevisionDraft


SEED = 26212
MODES = ('last_turn', 'draft_and_history')
SPLITS = {
    'train': ('目前说的是{cue}这件事。', '刚才两项还混在一起。', '我同意，照刚才说的做。'),
    'calibration': ('刚讨论的是关于{cue}的调整。', '刚刚没说明在讨论哪项。', '这就定了，按说好的办。'),
    'held': ('我现在谈的是{cue}那一项。', '这两个待办还没分清。', '行，就照刚提到的安排吧。'),
}
TRAIN_CUES = {'time': '几点到', 'place': '具体位置', 'person': '负责的人', 'quantity': '件数', 'gear': '要用的东西'}
CAL_CUES = {'time': '什么时段', 'place': '地址', 'person': '经手者', 'quantity': '总量', 'gear': '备品'}
HELD_CUES = {'time': '哪天几时', 'place': '该往哪儿去', 'person': '由谁出面', 'quantity': '一共要留多少', 'gear': '报上的物件'}


def training_cases(specs, fixture, names, vocab, *, calibration=False):
    samples, _ = dialogue_samples(specs, fixture, names, vocab, calibration=calibration)
    per_topic = 6 if calibration else 24
    cases = []
    for i, row in enumerate(samples):
        topic = specs[i // per_topic]['case']
        field, value = next(iter(row['change'].items()))
        other = next(item for item in row['initial'] if item != field)
        choices = [item for item in vocab[other] if item != row['initial'][other]]
        cases.append({'key': f'{topic}-{i}', 'topic': topic, 'initial': row['initial'],
                      'drafts': ((field, value), (other, choices[i % len(choices)]))})
    return cases


def evaluation_cases(cases):
    held = []
    for i, case in enumerate(cases):
        field, new = next(iter(case['change'].items()))
        other = case['fields'][1]
        alternative = cases[i ^ 2]['initial'][other]
        if alternative == case['initial'][other] or cases[i ^ 2]['topic'] != case['topic']:
            raise ValueError('Held alternatives must be distinct values of the same field and topic')
        held.append({**case, 'drafts': ((field, new), (other, alternative))})
    return held


def episodes(cases, names, split, cues):
    focus, ambiguous, last = SPLITS[split]
    rows = []
    for case in cases:
        mapping = names[case['topic']]
        for order in (case['drafts'], case['drafts'][::-1]):
            pending = '\n'.join(f'{mapping[field]}拟改为{value}。' for field, value in order)
            for index, selected in enumerate((*order, None)):
                bridge = focus.format(cue=cues[selected[0]]) if selected is not None else ambiguous
                history = f'{pending}\n{bridge}\n'
                options = []
                for field, value in order:
                    request = f'核对草稿：{mapping[field]}改为{value}。\n{history}'
                    options.append({'field': field, 'value': value, 'text': request + last,
                                    'focus_start': len(request), 'matches': selected is not None and field == selected[0]})
                rows.append({'key': case['key'], 'order': [field for field, _ in order], 'bridge': bridge, 'last': last,
                             'target': index if selected is not None else None, 'options': options})
    return rows


def scoped_input(row, option, mode):
    if mode == 'last_turn':
        return row['last'], 0
    end = option['focus_start']
    if mode == 'scope_clause':
        return option['text'][:end - 1], end - len(row['bridge']) - 1
    return option['text'], end


def extract(featurizer, width, sets, mode):
    inputs = {}
    for rows in sets.values():
        for row in rows:
            for opt in row['options']:
                text, start = scoped_input(row, opt, mode)
                inputs[text] = {'text': text, 'focus_start': start}
    texts = sorted(inputs)
    hidden, masks = clause_features(featurizer, [inputs[text] for text in texts], width)
    return {text: (h, m) for text, h, m in zip(texts, hidden, masks, strict=True)}


def choices(rows, mode):
    return [scoped_input(row, opt, mode)[0] for row in rows for opt in row['options']]


def fit(model, features, rows, mode):
    rng = torch.Generator().manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=.002)
    for epoch in range(8):
        for ids in torch.randperm(len(rows), generator=rng).split(32):
            batch = [rows[i] for i in ids.tolist()]
            logits = model(*packed(features, choices(batch, mode)))
            actual = torch.tensor([int(item['matches']) for row in batch for item in row['options']])
            loss = F.cross_entropy(logits, actual)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print(mode, 'epoch', epoch + 1, round(float(loss.detach()), 5), flush=True)


@torch.inference_mode()
def score(model, features, rows, mode):
    margins = []
    for start in range(0, len(rows), 48):
        logits = model(*packed(features, choices(rows[start:start + 48], mode)))
        margins.extend((logits[:, 1] - logits[:, 0]).reshape(-1, 2).tolist())
    return margins


def resolve(margins, threshold, gap):
    a, b = margins
    if max(a, b) <= threshold or abs(a - b) <= gap:
        return None
    return int(b > a)


def calibrate(rows, scores):
    candidates = []
    for threshold in (-4., -2., 0., 2., 4., 6., 8.):
        for gap in (0., 1., 2., 4., 6., 8.):
            picks = [resolve(pair, threshold, gap) for pair in scores]
            wrong = sum(pick is not None and pick != row['target'] for row, pick in zip(rows, picks, strict=True))
            exact = sum(pick is not None and pick == row['target'] for row, pick in zip(rows, picks, strict=True))
            candidates.append((wrong, -exact, gap, threshold))
    _, _, gap, threshold = min(candidates)
    return threshold, gap


def audit(rows, scores, threshold, gap, cases, tokenizer):
    details = []
    by_key = {case['key']: case for case in cases}
    for row, pair in zip(rows, scores, strict=True):
        pick = resolve(pair, threshold, gap)
        state = by_key[row['key']]['initial']
        expected = dict(state)
        if row['target'] is not None:
            opt = row['options'][row['target']]
            expected[opt['field']] = opt['value']
        changes = {}
        if pick is not None:
            opt = row['options'][pick]
            draft = RevisionDraft(opt['field']).observe(RevisionDecision.PROPOSE, field=opt['field'], value=opt['value'])
            changes = draft.observe(RevisionDecision.CONFIRM, field=opt['field']).patch()
        after = state_values(tokenizer, state_for(tokenizer, state).update(encode_changes(tokenizer, changes)))
        details.append({**row, 'scores': pair, 'pick': pick, 'state': after, 'state_exact': after == expected})
    return {'summary': {'rows': len(rows), 'specific': sum(r['target'] is not None for r in rows),
                        'correct_commit': sum(r['pick'] is not None and r['pick'] == r['target'] for r in details),
                        'wrong_commit': sum(r['pick'] is not None and r['pick'] != r['target'] for r in details),
                        'ambiguous_commit': sum(r['pick'] is not None and r['target'] is None for r in details),
                        'state_exact': sum(r['state_exact'] for r in details),
                        'both_orders_all': sum(all(r['state_exact'] for r in details[i:i+6])
                                               for i in range(0, len(details), 6))}, 'rows': details}


def main(args):
    start = perf_counter()
    torch.set_num_threads(2)
    report_path, weight_path = args.output / 'result.json', args.output / 'reference.pt'
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    prior = read(args.events)
    protected = (args.checkpoint, DATA, DESIGN, EPISODES, OBSERVATIONS, args.events,
                 *(layer / name for layer in args.layers for name in ('attention.pt', 'comparison.json')))
    sources = {str(path): sha(path) for path in protected}
    frozen = {Path(path).resolve(): digest for path, digest in prior['sources'].items()}
    if any(sources[str(path)] != frozen.get(path.resolve()) for path in protected if path != args.events):
        raise ValueError('Frozen inputs differ from the revision event study')
    expected = read(report_path) if args.verify else None
    if expected is not None and (expected['sources'] != sources or expected['weights_sha256'] != sha(weight_path)):
        raise ValueError('Independent replay inputs changed')

    specs, source, fixture = read(EPISODES)['cases'], read(DATA), read(OBSERVATIONS)['topics']
    spoken = {topic: (*ROLE_NAMES[topic], EXTRA_NAMES[topic]) for topic in EXTRA_NAMES}
    names = {row['case']: dict(zip(row['fields'], spoken[row['case']], strict=True)) for row in specs}
    training = training_cases(specs, fixture, spoken, source['values']['train'])
    calibration = training_cases(specs, fixture, spoken, source['values']['held'], calibration=True)
    held = evaluation_cases(held_episodes({'cases': specs}))
    sets = {key: episodes(cases, names, key, cues) for key, cases, cues in
            (('train', training, TRAIN_CUES), ('calibration', calibration, CAL_CUES), ('held', held, HELD_CUES))}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != prior['individual_sha256']:
        raise ValueError('The individual changed')
    extend_positions(decoder, rotary, 2048)
    features = FrozenTurnFeatures(decoder, tokenizer, depth=20)
    saved = torch.load(weight_path, map_location='cpu', weights_only=True) if args.verify else {}
    results = {}
    for mode in ('scope_clause',) if args.scope_clause else MODES:
        encoded = extract(features, base.config.hidden_size, sets, mode)
        torch.manual_seed(SEED)
        model = ClauseIntent(base.config.hidden_size, recurrent=True)
        if args.verify:
            model.load_state_dict(saved[mode])
        else:
            fit(model, encoded, sets['train'], mode)
            saved[mode] = model.state_dict()
        model.eval()
        cut, gap = calibrate(sets['calibration'], score(model, encoded, sets['calibration'], mode))
        results[mode] = {'threshold': cut, 'gap': gap}
        for key, cases in (('calibration', calibration), ('held', held)):
            results[mode][key] = audit(sets[key], score(model, encoded, sets[key], mode), cut, gap, cases, tokenizer)
            print(mode, key, results[mode][key]['summary'], flush=True)
        del encoded
    if not args.verify:
        atomic_save(saved, weight_path)
    layers = load_recurrent_replacements(decoder, args.layers, sources[str(args.checkpoint)])
    with ExitStack() as stack:
        install(stack, layers)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    if (memory != prior['memory'] or frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != sources[str(path)] for path in protected)):
        raise ValueError('Frozen sources or durable memory changed')
    result = {'sources': sources, 'weights_sha256': sha(weight_path), 'individual_sha256': individual,
              'train_rows': len(sets['train']), 'arms': results, 'memory': memory}
    if args.verify:
        if any(expected[key] != value for key, value in result.items()):
            raise ValueError('Independent replay did not match')
        store_json(args.output / 'reload.json', {'weights_sha256': result['weights_sha256'],
                                                 'seconds': perf_counter() - start})
    else:
        store_json(report_path, {**result, 'seconds': perf_counter() - start})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'events', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('layers', nargs=2, type=Path)
    parser.add_argument('--scope-clause', action='store_true', help='Read the turn that establishes the referent')
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
