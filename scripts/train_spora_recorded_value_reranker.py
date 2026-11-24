"""Rerank revision spans using the value already stored in the selected record."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.train_spora_described_revision_span import encode, make_rows, score
from scripts.train_spora_revision_intent import (
    DATA, DESIGN, EPISODES, EXTRA_NAMES, FIELDS, HELD, MODEL, OBSERVATIONS, ROLE_NAMES, TEST, AutoTokenizer,
    JointTurnValueWriter, atomic_save, evaluate, examples, extend_positions, frozen_digest, held_episodes, install,
    load_full_streamed, load_integrated_runtime, load_recurrent_replacements, read, records, sha, store_json, training_groups,
)
from scripts.train_spora_revision_value_roles import TRAIN_TURNS, paired_turns
from scripts.train_spora_same_field_reference import audit, build, proposal_text, proposals, summary


SEED = 26244
TOP = 48
MODES = ('native', 'exclude_unchanged', 'learned')
MENTIONED = 'learned_when_old_mentioned'
BREAKS = frozenset('。！？；，、：,.!?;:\n《》「」')


def occurrences(text, value):
    if not value:
        return []
    return [(index, index + len(value)) for index in range(len(text)) if text.startswith(value, index)]


def span_features(text, start, stop, old, mentions, relative):
    span = text[start:stop]
    overlap = max((max(0, min(stop, end) - max(start, begin)) for begin, end in mentions), default=0)
    left = text[start - 1] if start else ''
    right = text[stop] if stop < len(text) else ''
    return [max(relative, -60.) / 12., len(span) / 14., start / len(text),
            float(span == old), overlap / len(span), float(any(end <= start for _, end in mentions)),
            float(any(begin >= stop for begin, _ in mentions)), float(not left or left in BREAKS),
            float(not right or right in BREAKS), float(any(char in BREAKS for char in span)),
            float(bool(old) and span in old), float(bool(old) and old in span)]


@torch.inference_mode()
def candidates(writer, featurizer, cases, schemas, *, training=False):
    turns = [case[:4] for case in cases]
    rows = make_rows(turns, schemas)
    if training:
        rows = [row for row in rows if row[-1] is not None]
    data = encode(featurizer, rows, 'identifier')
    width = data[0].shape[1]
    batches = [score(writer, data, batch)[0].cpu() for batch in torch.arange(len(rows)).split(24)]
    all_logits = torch.cat(batches)
    targets = data[2]
    del batches, data
    output = []
    offset = 0
    for text, topic, field, expected, old in cases:
        count = 1 if training else len(schemas[topic])
        logits = all_logits[offset:offset + count]
        ranked_fields = (logits[:, 1:].amax(dim=1) - logits[:, 0]).argmax().item()
        selected_field = FIELDS[rows[offset + ranked_fields][1]]
        raw = logits[ranked_fields, 1:]
        native = raw.argmax().item()
        top = raw.topk(min(TOP, raw.numel())).indices.tolist()
        ranked = [native, *(i for i in top if i != native and torch.isfinite(raw[i]).item())][:TOP]
        truth = int(targets[offset + ranked_fields]) - 1
        offset += count
        if training and truth not in ranked:
            ranked[-1] = truth
        highest = float(raw.max())
        mentions = occurrences(text, old)
        spans, feats = [], []
        for index in ranked:
            left, right = divmod(index, width)
            stop = right + 1
            spans.append(text[left:stop])
            feats.append(span_features(text, left, stop, old, mentions, float(raw[index]) - highest))
        output.append({'text': text, 'topic': topic, 'field': field, 'value': expected, 'old': old,
                       'predicted_field': selected_field, 'spans': spans, 'features': torch.tensor(feats),
                       'target': ranked.index(truth) if truth in ranked else -1})
    return output


def fit(model, proposals_to_rank):
    length = max(row['features'].shape[0] for row in proposals_to_rank)
    features = torch.stack([F.pad(row['features'], (0, 0, 0, length - len(row['spans']))) for row in proposals_to_rank])
    valid = torch.arange(length)[None, :] < torch.tensor([len(row['spans']) for row in proposals_to_rank])[:, None]
    targets = torch.tensor([row['target'] for row in proposals_to_rank])
    rng = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    for epoch in range(9):
        for batch in torch.randperm(len(targets), generator=rng).split(48):
            local = features[batch]
            logits = (local[:, :, 0] + model(local).squeeze(-1)).masked_fill(~valid[batch], -torch.inf)
            loss = F.cross_entropy(logits, targets[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print('reranker', epoch + 1, round(float(loss.detach()), 4), flush=True)


@torch.inference_mode()
def choices(model, rows, mode):
    results = []
    for row in rows:
        spans, features = row['spans'], row['features']
        if mode == 'native' or (mode == MENTIONED and not occurrences(row['text'], row['old'])):
            pick = 0
        elif mode == 'exclude_unchanged':
            pick = next((i for i, value in enumerate(spans) if value != row['old']), 0)
        else:
            pick = int((features[:, 0] + model(features).squeeze(-1)).argmax())
        proposal = {'field': row['predicted_field'], 'text': spans[pick]}
        results.append({key: row[key] for key in ('text', 'topic', 'field', 'value', 'old')}
                       | {'choice': proposal, 'candidate_reaches_target': row['target'] >= 0})
    return results


def span_report(rows):
    return {'turns': len(rows),
            'exact': sum(row['choice'] == {'field': row['field'], 'text': row['value']} for row in rows),
            'old': sum(row['choice']['text'] == row['old'] for row in rows),
            'truncated': sum(row['choice']['text'] != row['value'] and row['choice']['text'] in row['value'] for row in rows),
            'candidate_reaches_target': sum(row['candidate_reaches_target'] for row in rows)}


def main(args):
    start = perf_counter()
    torch.set_num_threads(2)
    report, weights = args.output / 'result.json', args.output / 'reranker.pt'
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    scope_path = args.scope / 'result.json'
    writer_path, writer_weights = args.writer / 'result.json', args.writer / 'writers.pt'
    scope, writer_parent = read(scope_path), read(writer_path)
    inputs = (args.checkpoint, DATA, DESIGN, EPISODES, OBSERVATIONS, scope_path, writer_path, writer_weights,
              *(layer / name for layer in args.layers for name in ('attention.pt', 'comparison.json')))
    sources = {str(path): sha(path) for path in inputs}
    inherited = {Path(path).resolve(): digest for path, digest in scope['sources'].items()}
    if any(sources[str(path)] != inherited.get(path.resolve()) for path in inputs if path != scope_path):
        raise ValueError('Frozen inputs differ from the observation scope experiment')
    if scope['memory'] != writer_parent['memory'] or sources[str(writer_weights)] != writer_parent['weights_sha256']:
        raise ValueError('Frozen span writer changed')
    expected = read(report) if args.verify else None
    if expected is not None and (expected['sources'] != sources or expected['weights_sha256'] != sha(weights)):
        raise ValueError('Replay sources changed')

    specs, source, fixtures = read(EPISODES)['cases'], read(DATA), read(OBSERVATIONS)['topics']
    names = {topic: (*ROLE_NAMES[topic], EXTRA_NAMES[topic]) for topic in EXTRA_NAMES}
    fields = {spec['case']: spec['fields'] for spec in specs}
    schemas = {topic: dict(zip(items, names[topic], strict=True)) for topic, items in fields.items()}
    held = held_episodes({'cases': specs})
    same_field = build(proposals(held, schemas, fixtures), 'held', per_pair=4, key_kind='observation')
    unique = sorted({(proposal_text(draft), draft['topic'], draft['field'], draft['value'], draft['old'])
                     for row in same_field for draft in row['options']})
    matched_held = examples(held, fields, names, TEST)
    tests = {'same_field': unique,
             'proposals': [(group['texts'][1], row['topic'], group['field'], group['candidate'], group['old'])
                           for row, group in zip(held, matched_held, strict=True)],
             'confirmed': [(row['update'], row['topic'], *next(iter(row['change'].items())), row['old']) for row in held]}
    previous = scope['arms']['scoped']
    for row, check in zip(same_field, previous['known_rows'], strict=True):
        target = row['target'] if row['confirm'] else None
        if (check['target'] != target or check['bridge'] != row['bridge'] or check['utterance'] != row['utterance']
                or check['topics'] != [draft['topic'] for draft in row['options']]):
            raise ValueError('Scope replay no longer matches the held cases')
    frozen_scores = [(row['top'], row['gap'], row['action']) for row in previous['known_rows']]
    oracle = [(row['target'] or 0, float(row['target'] is not None), action)
              for row, (_, _, action) in zip(same_field, frozen_scores, strict=True)]

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != scope['individual_sha256'] or individual != writer_parent['individual_sha256']:
        raise ValueError('The individual changed')
    extend_positions(decoder, rotary, 2048)
    feat = FrozenTurnFeatures(decoder, tokenizer, depth=12)
    writer = JointTurnValueWriter(64, feature_width=base.config.hidden_size).to(base.device).eval()
    writer.load_state_dict(torch.load(writer_weights, map_location='cpu', weights_only=True)['identifier'])
    torch.manual_seed(SEED)
    reranker = nn.Sequential(nn.Linear(12, 32), nn.GELU(), nn.Linear(32, 1))
    if args.verify:
        reranker.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    else:
        groups = training_groups(source, specs, fixtures, names)
        training = []
        for i, group in enumerate(groups):
            topic = specs[i // 24]['case']
            training.append((group['texts'][1], topic, group['field'], group['candidate'], group['old']))
        train_specs = [(topic, field, old, value) for _, topic, field, value, old in training]
        turns, stored = paired_turns(train_specs, schemas, fixtures, TRAIN_TURNS)
        training += [(*turn, old) for i, (turn, old) in enumerate(zip(turns, stored, strict=True)) if i % 4 in (0, 3)]
        if {row[0] for row in training} & {row[0] for cases in tests.values() for row in cases}:
            raise ValueError('Training and held turns overlap')
        fit(reranker, candidates(writer, feat, training, schemas, training=True))
        atomic_save(reranker.state_dict(), weights)
    reranker.eval()
    ranked = {name: candidates(writer, feat, rows, schemas) for name, rows in tests.items()}
    arms = {}
    for mode in (*MODES, MENTIONED):
        picked = {name: choices(reranker, rows, mode) for name, rows in ranked.items()}
        copied = {(row['text'], row['topic']): {'automatic': row['choice']} for row in picked['same_field']}
        outcome = summary(audit(same_field, frozen_scores, tokenizer, previous['limits'], copied))
        bounded = summary(audit(same_field, oracle, tokenizer, (0., previous['limits'][1]), copied))
        arms[mode] = {'spans': {name: span_report(rows) for name, rows in picked.items()}, 'choices': picked,
                      'automatic': outcome, 'given_topic': bounded}
        print(mode, arms[mode]['spans'], 'automatic', outcome, 'given topic', bounded, flush=True)
    evidence_gate = arms.pop(MENTIONED)
    if arms['native']['automatic'] != previous['automatic']:
        raise ValueError('Native writer control did not match the prior experiment')
    layers = load_recurrent_replacements(decoder, args.layers, sources[str(args.checkpoint)])
    with ExitStack() as stack:
        install(stack, layers)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    if (memory != scope['memory'] or frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != sources[str(path)] for path in inputs)):
        raise ValueError('Frozen sources or durable memory changed')
    result = {'sources': sources, 'weights_sha256': sha(weights), 'individual_sha256': individual,
              'arms': arms, 'memory': memory}
    if args.verify:
        if any(expected[key] != value for key, value in result.items()):
            raise ValueError('Independent replay did not match')
        store_json(args.output / 'old-evidence-gate.json', evidence_gate)
        store_json(args.output / 'reload.json', {'weights_sha256': result['weights_sha256'], 'seconds': perf_counter() - start})
    else:
        store_json(report, {**result, 'seconds': perf_counter() - start})
        store_json(args.output / 'old-evidence-gate.json', evidence_gate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'scope', 'writer', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('layers', nargs=2, type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
