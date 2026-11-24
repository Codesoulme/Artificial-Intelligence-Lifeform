"""Compare identifier and description conditioning for copying a revision candidate."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.prepare_spora_revision_samples import dialogue_samples
from scripts.train_spora_revision_intent import (
    DATA, DESIGN, EPISODES, EXTRA_NAMES, FIELDS, HELD, MODEL, OBSERVATIONS, ROLE_NAMES, TEST, AutoTokenizer,
    JointTurnValueWriter, atomic_save, evaluate, examples, extend_positions, frozen_digest, held_episodes, install,
    load_full_streamed, load_integrated_runtime, load_recurrent_replacements, read, records, sha, store_json,
    training_groups,
)
from spora.research_dialogue_working_state import SPAN_LIMIT, characters


SEED = 26196
MODES = ('identifier', 'description')


def make_rows(cases, schemas):
    rows = []
    for text, topic, field, expected in cases:
        if text.count(expected) != 1 or len(expected) > SPAN_LIMIT:
            raise ValueError('Expected one legal candidate span in each turn')
        start = text.index(expected)
        for candidate, description in schemas[topic].items():
            boundary = (start, start + len(expected) - 1) if candidate == field else None
            rows.append((text, FIELDS.index(candidate), description, boundary))
    return rows


def encode(featurizer, rows, mode):
    codes = characters([row[0] for row in rows])
    positions = torch.tensor([row[1] for row in rows])
    length = codes.shape[1]
    targets = torch.tensor([0 if span is None else 1 + span[0] * length + span[1] for *_, span in rows])
    aligned = None
    for first in range(0, len(rows), 48):
        batch = rows[first:first + 48]
        labels = [str(index + 1) if mode == 'identifier' else desc for _, index, desc, _ in batch]
        prefixes = [f'待定位字段：{label}。\n' for label in labels]
        features = featurizer([prefix + row[0] for prefix, row in zip(prefixes, batch, strict=True)])
        if aligned is None:
            aligned = features.new_zeros((len(rows), length, features.shape[-1]))
        for i, (prefix, (text, *_)) in enumerate(zip(prefixes, batch, strict=True)):
            aligned[first + i, :len(text)].copy_(features[i, len(prefix):len(prefix) + len(text)])
    return codes, positions, targets, aligned


def score(model, data, indices):
    codes, fields, targets, features = data
    device = model.embedding.weight.device
    fields = fields[indices].to(device)
    logits = model(codes[indices].to(device), features[indices].to(device))
    return logits[torch.arange(len(indices), device=device), fields], targets[indices].to(device)


def fit(model, data):
    rng = torch.Generator().manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    for epoch in range(6):
        for indices in torch.randperm(len(data[0]), generator=rng).split(48):
            logits, targets = score(model, data, indices)
            batch_loss = F.cross_entropy(logits, targets, reduction='none')
            loss = (batch_loss * torch.where(targets > 0, 2.5, 1.)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        print('epoch', epoch + 1, round(float(loss.detach()), 4), flush=True)


@torch.inference_mode()
def assess(model, data, cases, rows, schemas):
    length = data[0].shape[1]
    logits = torch.cat([score(model, data, ids)[0].cpu() for ids in torch.arange(len(rows)).split(24)])
    best, span = logits[:, 1:].max(dim=1)
    margins = (best - logits[:, 0]).tolist()
    candidates = []
    for (text, field, _, _), index, margin in zip(rows, span.tolist(), margins, strict=True):
        start, end = divmod(index, length)
        candidates.append({'field': FIELDS[field], 'text': text[start:end + 1], 'margin': round(margin, 5)})
    results = []
    offset = 0
    for text, topic, field, expected in cases:
        count = len(schemas[topic])
        choices = candidates[offset:offset + count]
        offset += count
        given = next(row for row in choices if row['field'] == field)
        automatic = max(choices, key=lambda row: row['margin'])
        results.append({'turn': text, 'topic': topic, 'expected_field': field, 'expected_value': expected,
                        'given_field': given, 'automatic': automatic, 'candidates': choices})
    return results


def summarize(rows):
    return {'count': len(rows),
            'given_field_exact': sum(row['given_field']['text'] == row['expected_value'] for row in rows),
            'automatic_field_exact': sum(row['automatic']['field'] == row['expected_field'] for row in rows),
            'automatic_exact': sum(row['automatic']['field'] == row['expected_field']
                                   and row['automatic']['text'] == row['expected_value'] for row in rows)}


def main(args):
    start = perf_counter()
    torch.set_num_threads(2)
    output = args.output
    report_path = output / 'result.json'
    weights_path = output / 'writers.pt'
    if not args.verify:
        output.mkdir(parents=True, exist_ok=False)
    parent = read(args.study / 'result.json')
    weights = args.study / 'revision-writers.pt'
    paths = (args.checkpoint, DATA, DESIGN, EPISODES, OBSERVATIONS, args.study / 'result.json', weights,
             *(layer / name for layer in args.layers for name in ('attention.pt', 'comparison.json')))
    sources = {str(path): sha(path) for path in paths}
    original = {Path(path).resolve(): digest for path, digest in parent['sources'].items()}
    if any(sources[str(path)] != original.get(path.resolve()) for path in paths if path.parent != args.study):
        raise ValueError('A frozen source differs from the prior writer study')
    if sources[str(weights)] != parent['weights_sha256']:
        raise ValueError('The initialization weights changed')
    expected = read(report_path) if args.verify else None
    if expected is not None and (expected['sources'] != sources or expected['weights_sha256'] != sha(weights_path)):
        raise ValueError('Replay sources changed')

    specs = read(EPISODES)['cases']
    names = {name: (*ROLE_NAMES[name], EXTRA_NAMES[name]) for name in EXTRA_NAMES}
    fields = {spec['case']: spec['fields'] for spec in specs}
    schemas = {topic: dict(zip(items, names[topic], strict=True)) for topic, items in fields.items()}
    source, fixture = read(DATA), read(OBSERVATIONS)['topics']
    plain, _ = dialogue_samples(specs, fixture, names, source['values']['train'])
    matched = training_groups(source, specs, fixture, names)
    train = []
    for i, (row, group) in enumerate(zip(plain, matched, strict=True)):
        topic = specs[i // 24]['case']
        field, value = next(iter(row['change'].items()))
        if (field, value) != (group['field'], group['candidate']):
            raise ValueError('Matched training turns have different annotations')
        train.extend((text, topic, field, value) for text in (row['update'][0], *group['texts'][:2]))
    held = held_episodes({'cases': specs})
    matched_held = examples(held, fields, names, TEST)
    confirmed = [(row['update'], row['topic'], *next(iter(row['change'].items()))) for row in held]
    proposed = [(group['texts'][1], row['topic'], group['field'], group['candidate'])
                for row, group in zip(held, matched_held, strict=True)]
    tests = {'confirmed': confirmed, 'proposed': proposed}
    if {row[0] for row in train} & {row[0] for cases in tests.values() for row in cases}:
        raise ValueError('Training and evaluation turns overlap')
    train_rows = make_rows(train, schemas)
    test_rows = {name: make_rows(cases, schemas) for name, cases in tests.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != parent['individual_sha256']:
        raise ValueError('The individual changed')
    extend_positions(decoder, rotary, 2048)
    featurizer = FrozenTurnFeatures(decoder, tokenizer, depth=12)
    initial = torch.load(weights, map_location='cpu', weights_only=True)['joint']['weights']
    saved = torch.load(weights_path, map_location='cpu', weights_only=True) if args.verify else {}
    results = {}
    for mode in MODES:
        print('writer', mode, flush=True)
        writer = JointTurnValueWriter(64, feature_width=base.config.hidden_size).to(base.device)
        writer.load_state_dict(saved[mode] if args.verify else initial)
        if not args.verify:
            data = encode(featurizer, train_rows, mode)
            fit(writer, data)
            del data
            saved[mode] = {name: weight.detach().cpu() for name, weight in writer.state_dict().items()}
        writer.eval()
        results[mode] = {}
        for name, cases in tests.items():
            encoded = encode(featurizer, test_rows[name], mode)
            rows = assess(writer, encoded, cases, test_rows[name], schemas)
            del encoded
            results[mode][name] = {'summary': summarize(rows), 'rows': rows}
            print(mode, name, results[mode][name]['summary'], flush=True)
    if not args.verify:
        atomic_save(saved, weights_path)

    layers = load_recurrent_replacements(decoder, args.layers, sources[str(args.checkpoint)])
    with ExitStack() as stack:
        install(stack, layers)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    if (memory != parent['memory'] or frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != sources[str(path)] for path in paths)):
        raise ValueError('Frozen sources or durable memory replay changed')
    result = {'sources': sources, 'weights_sha256': sha(weights_path), 'individual_sha256': individual,
              'train_turns': len(train), 'arms': results, 'memory': memory}
    if args.verify:
        if any(expected[key] != val for key, val in result.items()):
            raise ValueError('Independent replay did not match')
        store_json(output / 'reload.json', {'weights_sha256': result['weights_sha256'], 'seconds': perf_counter() - start})
    else:
        store_json(report_path, {**result, 'seconds': perf_counter() - start})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'study', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('layers', nargs=2, type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
