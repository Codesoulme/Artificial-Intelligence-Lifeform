"""Learn to copy the proposed value when two values belong to the same field."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.train_spora_described_revision_span import assess, encode, make_rows, score, summarize
from scripts.train_spora_revision_events import compare, state_audit
from scripts.train_spora_revision_intent import (
    DATA, DESIGN, EPISODES, EXTRA_NAMES, HELD, MODEL, OBSERVATIONS, ROLE_NAMES, TEST, AutoTokenizer,
    JointTurnValueWriter, atomic_save, evaluate, examples, extend_positions, frozen_digest, held_episodes, install,
    load_full_streamed, load_integrated_runtime, load_recurrent_replacements, read, records, sha, store_json,
    training_groups,
)


SEED = 26205
ARMS = ('copy', 'copy_with_role_loss')
TRAIN_TURNS = (
    ('{topic}目前的{field}还是{old}。如果要动这栏，可以先考虑{new}。',
     '听说有人提过{new}，这可以作为{topic}的备选{field}。现在执行的仍是{old}。'),
    ('{topic}这次{field}按{old}准备。有人问过要不要换作{new}，还没同意。',
     '先把{new}列到讨论稿上。{topic}的{field}目前依旧用{old}，另等通知。'),
    ('给{topic}的{field}留个备查。现有那栏是{old}，如果再调整建议填{new}。',
     '{new}有人建议过，不过只是备查方案。{topic}的{field}目前登记在{old}。'),
    ('{topic}的{field}暂按{old}执行。我想过把它变成{new}，目前先别提交。',
     '是否改到{new}还要商量。至于{topic}的{field}，有效记录还是{old}。'),
)
HELD_TURNS = (
    ('手头那张{topic}单上，{field}写着{old}。有人把{new}拿来征求意见，还未落笔。',
     '刚发来的{new}还在征求意见，{topic}单里的{field}一栏仍然照{old}办。'),
    ('先把情况说清：{topic}已按{old}安排{field}，眼下讨论的是能否用{new}代替。',
     '眼下讨论的替代项是{new}。要是现在查看{topic}的{field}，上面写的还是{old}。'),
    ('{topic}现在执行的{field}是{old}，但这项另有一份未批的建议，上头填的是{new}。',
     '有一份尚未签字的提案写了{new}。照已生效的{topic}记录，{field}应当是{old}。'),
)


def paired_turns(specs, schemas, fixture, patterns):
    turns, previous = [], []
    for i, (topic, field, old, new) in enumerate(specs):
        for template in patterns[i % len(patterns)]:
            for current, candidate in ((old, new), (new, old)):
                text = template.format(topic=fixture[topic]['label'], field=schemas[topic][field],
                                       old=current, new=candidate)
                if text.count(current) != 1 or text.count(candidate) != 1:
                    raise ValueError(f'A contrast pair has ambiguous value boundaries: {text}')
                turns.append((text, topic, field, candidate))
                previous.append(current)
    return turns, previous


def rivals_for(rows, old_values, width):
    rivals = []
    for text, _, _, boundary in rows:
        if boundary is None:
            rivals.append((0, 0, 0))
            continue
        old = old_values[text]
        old_start = text.index(old)
        start, end = boundary
        rivals.append((1 + old_start * width + old_start + len(old) - 1,
                       1 + (start + 1) * width + end if start < end else 0,
                       1 + start * width + end - 1 if start < end else 0))
    return torch.tensor(rivals)


def fit(writer, data, rivals, *, role_loss):
    rng = torch.Generator().manual_seed(SEED)
    opt = torch.optim.AdamW(writer.parameters(), lr=.0005)
    for epoch in range(5):
        for indices in torch.randperm(len(data[0]), generator=rng).split(48):
            logits, targets = score(writer, data, indices)
            loss = (F.cross_entropy(logits, targets, reduction='none') * torch.where(targets > 0, 2.5, 1.)).mean()
            if role_loss:
                positive = targets > 0
                candidates = rivals[indices].to(targets.device)[positive]
                scores = logits[positive].gather(1, candidates)
                correct = logits[positive].gather(1, targets[positive, None])
                penalties = F.softplus(2. + scores - correct)[candidates > 0]
                loss = loss + .5 * penalties.sum() / max(penalties.numel(), 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(writer.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        print('epoch', epoch + 1, round(float(loss.detach()), 5), flush=True)


def contrast_summary(rows, old_values):
    exact = [row['given_field']['text'] == row['expected_value'] for row in rows]
    return {**summarize(rows), 'given_field_old': sum(row['given_field']['text'] == old for row, old in
                                                     zip(rows, old_values, strict=True)),
            'given_field_truncated': sum(row['given_field']['text'] in row['expected_value'] for row in rows) - sum(exact),
            'swap_pair_exact': sum(a and b for a, b in zip(exact[::2], exact[1::2], strict=True)),
            'all_four_exact': sum(all(exact[i:i + 4]) for i in range(0, len(exact), 4))}


def replay(rows, held, event_metrics, tokenizer):
    candidates = {(row['turn'], row['topic']): row['automatic'] for row in rows}
    revised = []
    for row in event_metrics:
        case = held[row['group']]
        candidate = candidates[row['first'], case['topic']]
        if case['change'] != {row['field']: row['candidate']}:
            raise ValueError('Frozen revision event labels differ from observed cases')
        revised.append({**row, 'predicted_field': candidate['field'], 'predicted_value': candidate['text']})
    states = state_audit(revised, held, tokenizer)
    scores = compare(revised)
    for mode in scores:
        scores[mode]['state_exact'] = sum(row['arms'][mode]['exact'] for row in states)
        scores[mode]['unwanted_mutation'] = sum(not row['expected_write'] and row['arms'][mode]['changed']
                                                 for row in states)
    return scores


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    result_path = args.output / 'result.json'
    weights_path = args.output / 'writers.pt'
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    prior = read(args.writer / 'result.json')
    event = read(args.events / 'result.json')
    expected = read(result_path) if args.verify else None
    paths = (args.checkpoint, DATA, DESIGN, EPISODES, OBSERVATIONS, args.writer / 'result.json', args.writer / 'writers.pt',
             args.events / 'result.json', *(layer / name for layer in args.layers for name in ('attention.pt', 'comparison.json')))
    sources = {str(path): sha(path) for path in paths}
    frozen = {Path(path).resolve(): digest for path, digest in prior['sources'].items()}
    if any(sources[str(path)] != frozen.get(path.resolve()) for path in paths
           if path.parent not in (args.writer, args.events)):
        raise ValueError('Frozen inputs differ from the prior span study')
    if sources[str(args.writer / 'writers.pt')] != prior['weights_sha256'] or event['memory'] != prior['memory']:
        raise ValueError('Frozen writer or event replay differs')
    if expected is not None and (expected['sources'] != sources or expected['weights_sha256'] != sha(weights_path)):
        raise ValueError('Replay sources changed')

    specs = read(EPISODES)['cases']
    names = {topic: (*ROLE_NAMES[topic], EXTRA_NAMES[topic]) for topic in EXTRA_NAMES}
    fields = {spec['case']: spec['fields'] for spec in specs}
    schemas = {topic: dict(zip(keys, names[topic], strict=True)) for topic, keys in fields.items()}
    fixture = read(OBSERVATIONS)['topics']
    groups = training_groups(read(DATA), specs, fixture, names)
    train_specs = [(spec['case'], group['field'], group['old'], group['candidate'])
                   for spec, start in zip(specs, range(0, len(groups), 24), strict=True)
                   for group in groups[start:start + 24]]
    train, old_train = paired_turns(train_specs, schemas, fixture, TRAIN_TURNS)
    episodes = held_episodes({'cases': specs})
    before = examples(episodes, fields, names, TEST)
    confirmed = [(row['update'], row['topic'], *next(iter(row['change'].items()))) for row in episodes]
    proposed = [(group['texts'][1], row['topic'], group['field'], group['candidate'])
                for row, group in zip(episodes, before, strict=True)]
    fresh, old_fresh = paired_turns([(topic, field, row['old'], value)
                                    for row, (_, topic, field, value) in zip(episodes, confirmed, strict=True)],
                                   schemas, fixture, HELD_TURNS)
    tests = {'confirmed': confirmed, 'proposed': proposed, 'swapped': fresh}
    if {row[0] for row in train} & {row[0] for cases in tests.values() for row in cases}:
        raise ValueError('Training and held turns overlap')

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != prior['individual_sha256'] or individual != event['individual_sha256']:
        raise ValueError('The individual changed')
    extend_positions(decoder, rotary, 2048)
    featurizer = FrozenTurnFeatures(decoder, tokenizer, depth=12)
    initial = torch.load(args.writer / 'writers.pt', map_location='cpu', weights_only=True)['identifier']
    saved = torch.load(weights_path, map_location='cpu', weights_only=True) if args.verify else {}
    groups = {name: make_rows(turns, schemas) for name, turns in tests.items()}
    tests_encoded = {name: encode(featurizer, rows, 'identifier') for name, rows in groups.items()}
    train_encoded = rivals = None
    if not args.verify:
        print('Encoding', len(train), 'contrast turns', flush=True)
        train_rows = make_rows(train, schemas)
        train_encoded = encode(featurizer, train_rows, 'identifier')
        rivals = rivals_for(train_rows, dict(zip((row[0] for row in train), old_train, strict=True)),
                            train_encoded[0].shape[1])
    arms = {}
    for mode in ('prior', *ARMS):
        print('writer', mode, flush=True)
        writer = JointTurnValueWriter(64, feature_width=base.config.hidden_size).to(base.device)
        writer.load_state_dict(saved[mode] if args.verify and mode != 'prior' else initial)
        if not args.verify and mode != 'prior':
            fit(writer, train_encoded, rivals, role_loss=mode == 'copy_with_role_loss')
            saved[mode] = {name: weight.detach().cpu() for name, weight in writer.state_dict().items()}
        writer.eval()
        outcomes = {}
        for name, turns in tests.items():
            rows = assess(writer, tests_encoded[name], turns, groups[name], schemas)
            summary = contrast_summary(rows, old_fresh) if name == 'swapped' else summarize(rows)
            outcomes[name] = {'summary': summary, 'rows': rows}
            print(name, summary, flush=True)
        outcomes['state_replay'] = replay(outcomes['confirmed']['rows'] + outcomes['proposed']['rows'], episodes,
                                          event['metrics']['held']['rows'], tokenizer)
        print('state replay', outcomes['state_replay'], flush=True)
        arms[mode] = outcomes
    del train_encoded, tests_encoded, rivals
    if not args.verify:
        atomic_save(saved, weights_path)

    layers = load_recurrent_replacements(decoder, args.layers, sources[str(args.checkpoint)])
    with ExitStack() as stack:
        install(stack, layers)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    if (memory != prior['memory'] or frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != sources[str(path)] for path in paths)):
        raise ValueError('Frozen sources or durable memory replay changed')
    result = {'sources': sources, 'weights_sha256': sha(weights_path), 'individual_sha256': individual,
              'train_turns': len(train), 'arms': arms, 'memory': memory}
    if args.verify:
        if any(expected[key] != value for key, value in result.items()):
            raise ValueError('Independent replay did not match')
        store_json(args.output / 'reload.json', {'weights_sha256': result['weights_sha256'], 'seconds': perf_counter() - started})
    else:
        store_json(result_path, {**result, 'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'writer', 'events', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('layers', nargs=2, type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
