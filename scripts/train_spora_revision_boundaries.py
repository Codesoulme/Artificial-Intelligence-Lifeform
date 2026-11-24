"""Train full-span copying with explicit supervision for both endpoints."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.train_spora_described_revision_span import encode, make_rows, score
from scripts.train_spora_recorded_value_reranker import MENTIONED, candidates, choices, save_replay, span_report, transfer_cases
from scripts.train_spora_revision_intent import (
    DATA, DESIGN, EPISODES, EXTRA_NAMES, HELD, MODEL, OBSERVATIONS, ROLE_NAMES, TEST, AutoTokenizer,
    JointTurnValueWriter, atomic_save, evaluate, examples, extend_positions, frozen_digest, held_episodes, install,
    load_full_streamed, load_integrated_runtime, load_recurrent_replacements, read, records, sha, store_json, training_groups,
)
from scripts.train_spora_same_field_reference import audit, build, proposal_text, proposals, summary


SEED = 26253
MODES = ('joint', 'with_endpoints')
NESTED = {'place': ('{old}东侧柜', '新馆{old}'), 'time': ('{old}整', '下周的{old}'),
          'gear': ('{old}耐用版', '便携款{old}'), 'quantity': ('{old}半', '十{old}')}


def expanded_value(field, old, index):
    if field == 'quantity' and index % 2:
        if old.startswith('十'):
            return f'二{old}'
        if old.startswith('两'):
            return f'{old}半'
    return NESTED[field][index % 2].format(old=old)


def training_turns(groups, specs, schemas):
    turns, nested = [], 0
    for i, example in enumerate(groups):
        topic = specs[i // 24]['case']
        field, old, new = example['field'], example['old'], example['candidate']
        turns.extend((text, topic, field, new) for text in example['texts'][:2])
        if field not in NESTED:
            continue
        full = expanded_value(field, old, i)
        label = schemas[topic][field]
        text = (f'确认了，{label}的整项内容是{full}，按这个登记。' if i % 3
                else f'原先的{label}记成{old}，核对后的完整登记写{full}。')
        if text.count(full) != 1:
            raise ValueError('Nested example has more than one complete value')
        turns.append((text, topic, field, full))
        nested += 1
    return turns, nested


def endpoint_loss(logits, targets, width):
    selected = targets > 0
    spans = logits[selected, 1:].reshape(-1, width, width)
    truth = targets[selected] - 1
    row = torch.arange(len(spans), device=logits.device)
    start, end = truth // width, truth % width
    total = spans.flatten(1).logsumexp(dim=1)
    left = spans[row, start, :].logsumexp(dim=1)
    right = spans[row, :, end].logsumexp(dim=1)
    return (total - .5 * (left + right)).sum() / max(len(spans), 1)


def fit(model, data, *, endpoints):
    rng = torch.Generator().manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=.0005)
    turn_width = data[0].shape[1]
    for epoch in range(5):
        for batch in torch.randperm(len(data[0]), generator=rng).split(48):
            logits, targets = score(model, data, batch)
            loss = (F.cross_entropy(logits, targets, reduction='none') * torch.where(targets > 0, 2.5, 1.)).mean()
            if endpoints:
                loss = loss + .5 * endpoint_loss(logits, targets, turn_width)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        print('epoch', epoch + 1, round(float(loss.detach()), 4), flush=True)


def eval_cases(specs, held, schemas, names, fixtures, paths):
    same = build(proposals(held, schemas, fixtures), 'held', per_pair=4, key_kind='observation')
    first = sorted({(proposal_text(draft), draft['topic'], draft['field'], draft['value'], draft['old'])
                    for row in same for draft in row['options']})
    fields = {spec['case']: spec['fields'] for spec in specs}
    previous = examples(held, fields, names, TEST)
    splits = {'same_field': first,
              'proposals': [(group['texts'][1], row['topic'], group['field'], group['candidate'], group['old'])
                            for row, group in zip(held, previous, strict=True)],
              'confirmed': [(row['update'], row['topic'], *next(iter(row['change'].items())), row['old']) for row in held]}
    for name, path in paths.items():
        splits.update({f'{name}_{kind}': rows for kind, rows in transfer_cases(read(path)).items()})
    return same, splits


def report_choices(picked, same, scores, oracle, tokenizer, limits):
    actual = {(row['text'], row['topic']): {'automatic': row['choice']} for row in picked['same_field']}
    exact = summary(audit(same, scores, tokenizer, limits, actual))
    bounded = summary(audit(same, oracle, tokenizer, (0., limits[1]), actual))
    return {'spans': {key: span_report(rows) for key, rows in picked.items()}, 'rows': picked,
            'automatic': exact, 'given_topic': bounded}


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    report, weights = args.output / 'result.json', args.output / 'writers.pt'
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    parent_path = args.scope / 'result.json'
    writer_path, source_weights = args.writer / 'result.json', args.writer / 'writers.pt'
    parent, writer_parent = read(parent_path), read(writer_path)
    paths = {'transfer': args.transfer, 'fresh': args.fresh}
    frozen = (args.checkpoint, DATA, DESIGN, EPISODES, OBSERVATIONS, writer_path, source_weights,
              *(layer / name for layer in args.layers for name in ('attention.pt', 'comparison.json')))
    inputs = (*frozen, parent_path, *paths.values())
    sources = {str(path): sha(path) for path in inputs}
    inherited = {Path(path).resolve(): digest for path, digest in parent['sources'].items()}
    if any(sources[str(path)] != inherited.get(path.resolve()) for path in frozen):
        raise ValueError('Frozen sources differ from earlier scope experiments')
    if sources[str(source_weights)] != writer_parent['weights_sha256'] or parent['memory'] != writer_parent['memory']:
        raise ValueError('Frozen writer or replay changed')
    expected = read(report) if args.verify else None
    if expected is not None and (expected['sources'] != sources or expected['weights_sha256'] != sha(weights)):
        raise ValueError('Replay inputs changed')

    specs, source, fixtures = read(EPISODES)['cases'], read(DATA), read(OBSERVATIONS)['topics']
    names = {topic: (*ROLE_NAMES[topic], EXTRA_NAMES[topic]) for topic in EXTRA_NAMES}
    schemas = {row['case']: dict(zip(row['fields'], names[row['case']], strict=True)) for row in specs}
    same, tests = eval_cases(specs, held_episodes({'cases': specs}), schemas, names, fixtures, paths)
    groups = training_groups(source, specs, fixtures, names)
    train, nested = training_turns(groups, specs, schemas)
    if {row[0] for row in train} & {row[0] for rows in tests.values() for row in rows}:
        raise ValueError('Training and held texts overlap')
    if {row[3] for row in train} & {row[3] for name, rows in tests.items() if name.startswith('fresh_') for row in rows}:
        raise ValueError('Training and fresh target values overlap')
    prior_scope = parent['arms']['scoped']
    for row, check in zip(same, prior_scope['known_rows'], strict=True):
        target = row['target'] if row['confirm'] else None
        if (check['target'] != target or check['bridge'] != row['bridge'] or check['utterance'] != row['utterance']
                or check['topics'] != [draft['topic'] for draft in row['options']]):
            raise ValueError('Scope replay changed')
    frozen_scores = [(row['top'], row['gap'], row['action']) for row in prior_scope['known_rows']]
    oracle = [(row['target'] or 0, float(row['target'] is not None), action)
              for row, (_, _, action) in zip(same, frozen_scores, strict=True)]

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != parent['individual_sha256'] or individual != writer_parent['individual_sha256']:
        raise ValueError('The individual changed')
    extend_positions(decoder, rotary, 2048)
    feature = FrozenTurnFeatures(decoder, tokenizer, depth=12)
    original = torch.load(source_weights, map_location='cpu', weights_only=True)['identifier']
    saved = torch.load(weights, map_location='cpu', weights_only=True) if args.verify else {}
    prior_reranker, comparisons = None, {}
    if args.prior_reranker is not None:
        prior_weights = args.prior_reranker / 'reranker.pt'
        prior_report = read(args.prior_reranker / 'result.json')
        if (prior_report['weights_sha256'] != sha(prior_weights) or prior_report['individual_sha256'] != individual
                or prior_report['memory'] != parent['memory']):
            raise ValueError('Prior reranker or replay changed')
        prior_reranker = nn.Sequential(nn.Linear(12, 32), nn.GELU(), nn.Linear(32, 1)).eval()
        prior_reranker.load_state_dict(torch.load(prior_weights, map_location='cpu', weights_only=True))
    data = None if args.verify else encode(feature, make_rows(train, schemas), 'identifier')
    results = {}
    for mode in ('original', *MODES):
        print('writer', mode, flush=True)
        writer = JointTurnValueWriter(64, feature_width=base.config.hidden_size).to(base.device)
        writer.load_state_dict(saved[mode] if args.verify and mode != 'original' else original)
        if data is not None and mode != 'original':
            fit(writer, data, endpoints=mode == 'with_endpoints')
            saved[mode] = {name: value.detach().cpu() for name, value in writer.state_dict().items()}
        writer.eval()
        ranked = {name: candidates(writer, feature, rows, schemas) for name, rows in tests.items()}
        picked = {name: choices(None, rows, 'native') for name, rows in ranked.items()}
        results[mode] = report_choices(picked, same, frozen_scores, oracle, tokenizer, prior_scope['limits'])
        print(mode, results[mode]['spans'], 'state', results[mode]['automatic'], flush=True)
        if prior_reranker is not None:
            picked = {name: choices(prior_reranker, rows, MENTIONED) for name, rows in ranked.items()}
            comparison = report_choices(picked, same, frozen_scores, oracle, tokenizer, prior_scope['limits'])
            if mode == 'original' and comparison['automatic'] != prior_report['arms']['learned']['automatic']:
                raise ValueError('Prior reranker comparison did not reproduce the scope replay')
            comparisons[mode] = comparison
            print('prior reranker with', mode, comparison['spans'], 'state', comparison['automatic'], flush=True)
    del data
    if not args.verify:
        atomic_save(saved, weights)
    if results['original']['automatic'] != prior_scope['automatic']:
        raise ValueError('Frozen control does not reproduce the prior automatic replay')
    layers = load_recurrent_replacements(decoder, args.layers, sources[str(args.checkpoint)])
    with ExitStack() as stack:
        install(stack, layers)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    if (memory != parent['memory'] or frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != sources[str(path)] for path in inputs)):
        raise ValueError('Frozen sources or durable reads changed')
    result = {'sources': sources, 'weights_sha256': sha(weights), 'individual_sha256': individual,
              'train_turns': len(train), 'nested_turns': nested, 'arms': results, 'memory': memory}
    if args.verify:
        if any(expected[key] != value for key, value in result.items()):
            raise ValueError('Independent replay did not match')
        if not (args.output / 'reload.json').exists():
            store_json(args.output / 'reload.json', {'weights_sha256': result['weights_sha256'],
                                                     'seconds': perf_counter() - started})
    else:
        store_json(report, {**result, 'seconds': perf_counter() - started})
    if comparisons:
        save_replay(args.output / 'prior-reranker.json', {'weights_sha256': prior_report['weights_sha256'],
                                                         'arms': comparisons})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'scope', 'writer', 'transfer', 'fresh', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('layers', nargs=2, type=Path)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--prior-reranker', type=Path, help='Include an existing old-value reranker in the audit')
    main(parser.parse_args())
