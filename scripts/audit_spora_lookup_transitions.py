"""Audit explicit learned transitions of the active memory address set."""

import argparse
from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_recurrent_lookup_scope import (
    ASK, CHAT, READ, SEED, SCOPE_MEMBERS, PEOPLE, VALUES, FrozenTurnFeatures,
    describe, episodes, evaluate, features,
)
from scripts.audit_spora_paired_lookup_clarification import RECORDS
from scripts.spora_version_chat import VersionChat, VersionReplyActions, VersionStore, artifacts, base, compress_dimension, local_model
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save


KEEP, ADD, REMOVE = range(3)

PAIR_FORMS = (
    ('请核对{target}，原记录继续查。', '请核对{target}，原记录不再查。'),
    ('把{target}放进来，保留手头那项。', '把{target}放进来，撤掉手头那项。'),
    ('也看一下{target}，先前选的还有效。', '也看一下{target}，先前选的已作废。'),
)
HELD_PAIR = ('接下来查看{target}，最先那条也算数。', '接下来查看{target}，最先那条不算数。')
MIDDLE_VALUES = ('秋灯', '桃源路', '银铃', '麦茶', '潮汐')


def paired_rows(rows, forms):
    pairs = []
    # These views retain the opening and final question across both alternatives.
    for row in rows:
        if row.scope != 'replace':
            continue
        for append, switch in forms:
            target = describe(row.addresses[1])
            pairs.append((replace(row, scope='both', events=(row.events[0], append.format(target=target)),
                                  action=CHAT if row.action == CHAT else ASK, version=-1),
                          replace(row, events=(row.events[0], switch.format(target=target)))))
    return list(dict.fromkeys(pairs))


def apply_operations(active, operations):
    return (active & operations.eq(KEEP)) | operations.eq(ADD)


def operation_targets(rows):
    actions = {
        'retain': (KEEP, KEEP), 'replace': (REMOVE, ADD), 'both': (KEEP, ADD),
        'cancel': (REMOVE, REMOVE), 'mention': (KEEP, KEEP), 'absent': (KEEP, KEEP),
    }
    return torch.tensor([((ADD, KEEP), actions[row.scope]) for row in rows])


def normalize_row(row):
    """Each frozen representation sees references to its own address and the alternative."""
    def text_for(text, index):
        for position, address in enumerate(row.addresses):
            text = text.replace(describe(address), '本项记录' if position == index else '另一项记录')
        return text

    return [SimpleNamespace(events=tuple(text_for(text, index) for text in row.events),
                            question=row.question, addresses=(address, address))
            for index, address in enumerate(row.addresses)]


def transition_features(extract, groups, *, utterance_features=False):
    marked = {name: [view for row in rows for view in normalize_row(row)] for name, rows in groups.items()}
    if utterance_features:
        prompts = list(dict.fromkeys(text for views in marked.values() for row in views
                                     for text in (*row.events, row.question)))
        vectors = {}
        for offset in range(0, len(prompts), 48):
            batch = prompts[offset:offset + 48]
            vectors.update((text, vector[max(0, len(text) - 6):len(text)].float().mean(0))
                           for text, vector in zip(batch, extract(batch), strict=True))
        result = {}
        for split, views in marked.items():
            current = torch.stack([vectors[row.question] for row in views[::2]])
            history = current.new_zeros(len(current), 2, 2, current.shape[-1])
            valid = torch.zeros(len(current), 2, dtype=torch.bool)
            for index, view in enumerate(views):
                for turn, text in enumerate(view.events):
                    history[index // 2, turn, index % 2] = vectors[text]
                    valid[index // 2, turn] = True
            result[split] = current, history, valid
        return result
    result = {}
    for name, (current, history, valid) in features(extract, marked).items():
        result[name] = (current[:, 0].unflatten(0, (-1, 2)).mean(1),
                        history[:, :, 0].unflatten(0, (-1, 2)).transpose(1, 2), valid[::2])
    return result


class LookupTransitions(nn.Module):
    def __init__(self, width, size=64):
        super().__init__()
        self.operation = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, size), nn.GELU(), nn.Linear(size, 3))
        self.version = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, size), nn.GELU(), nn.Linear(size, 4))

    def advance(self, history, valid, *, opening=None):
        active = torch.zeros_like(history[:, 0, :, 0], dtype=torch.bool)
        snapshots = []
        for index, (turn, present) in enumerate(zip(history.unbind(1), valid.unbind(1), strict=True)):
            change = opening if index == 0 and opening is not None else apply_operations(active, self.operation(turn).argmax(-1))
            active = torch.where(present[:, None], change, active)
            snapshots.append(active)
        return torch.stack(snapshots, dim=1)


def fit(data, rows, steps, *, pairs=(), contrast=False):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    head = LookupTransitions(data[0].shape[-1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    operations = operation_targets(rows)
    # A bare version retains its meaning even when the address set is ambiguous.
    version_for = {row.question: row.version for row in rows if row.version >= 0}
    versions = torch.tensor([version_for.get(row.question, 3) for row in rows])
    pair_indices = torch.tensor(pairs, dtype=torch.long)
    pair_rng = torch.Generator().manual_seed(SEED + 1)
    for _ in range(steps):
        batch = torch.randperm(len(rows), generator=rng)[:32]
        current, history, valid = (value[batch] for value in data)
        predicted = head.operation(history)[valid].flatten(0, 1)
        targets = operations[batch][valid].flatten()
        operation_loss = F.cross_entropy(predicted, targets, reduction='sum') / max(1, targets.numel())
        loss = operation_loss + F.cross_entropy(head.version(current), versions[batch])
        if contrast:
            sampled = pair_indices[torch.randperm(len(pairs), generator=pair_rng)[:16]]
            old_address = data[1][sampled, 1, 0]
            scores = head.operation(old_address)
            keep_margin = scores[..., KEEP] - scores[..., REMOVE]
            loss += .25 * F.softplus(1. - keep_margin[:, 0] + keep_margin[:, 1]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


def predict(head, data, rows, *, thresholds=(0., 0.), oracle_scope=False, confirmed_opening=False):
    current, history, valid = data
    version_confidence, version = head.version(current).softmax(-1).max(-1)
    opening = torch.tensor([True, False]).expand(len(rows), -1) if confirmed_opening else None
    snapshots = head.advance(history, valid, opening=opening)
    members = snapshots[:, -1]
    scope_confidence = torch.where(valid[:, :, None], head.operation(history).softmax(-1).amax(-1), 1.).amin((1, 2))
    expected = torch.tensor([SCOPE_MEMBERS[row.scope] for row in rows], dtype=torch.bool)
    active = expected if oracle_scope else members
    candidate = active.long().argmax(1) + READ
    eligible = active.sum(1).eq(1) & version_confidence.ge(thresholds[1]) & scope_confidence.ge(thresholds[0])
    action = torch.where(version.eq(3), CHAT, torch.where(eligible, candidate, ASK))
    return list(zip(action.tolist(), version.tolist(), strict=True)), snapshots, expected


def scope_metrics(snapshots, expected, rows):
    counts = {}
    for row, first, final, target in zip(rows, snapshots[:, 0], snapshots[:, -1], expected, strict=True):
        group = counts.setdefault(row.scope, Counter())
        group.update(total=1, complete=torch.equal(final, target), opening=torch.equal(first, torch.tensor([True, False]))
                     if row.events else not first.any().item())
    return counts


def calibrated_thresholds(head, data, rows):
    def utility(cuts):
        plan, _, _ = predict(head, data, rows, thresholds=cuts)
        right = sum(action == row.action and (action < READ or version == row.version)
                    for row, (action, version) in zip(rows, plan, strict=True))
        bad = sum(action >= READ and (action, version) != (row.action, row.version)
                  for row, (action, version) in zip(rows, plan, strict=True))
        return right - 2 * bad, -bad, -sum(cuts)

    cutoffs = (0., .5, .65, .8, .9, .99, 1.01)
    return max(((scope, version) for scope in cutoffs for version in cutoffs), key=utility)


def plan_metrics(rows, predictions):
    grouped = {}
    for row, (action, version) in zip(rows, predictions, strict=True):
        group = grouped.setdefault(row.scope, Counter())
        group.update(total=1, correct=action == row.action and (action < READ or version == row.version),
                     correct_read=action == row.action and action >= READ and version == row.version,
                     unwanted_read=action >= READ and row.action < READ,
                     wrong_address=action >= READ and row.action >= READ and action != row.action,
                     wrong_version=action == row.action and action >= READ and version != row.version)
    return grouped


@torch.inference_mode()
def three_version_archive(path, binding, model, tokenizer, writer_file):
    today = date.today()
    values = {}
    with VersionStore(path, binding, create=True) as store:
        settings = torch.load(writer_file, map_location='cpu', weights_only=True)
        writer = VersionChat(store, model, tokenizer, None, None, settings, ordinary_tokens=12)
        for (name, field, first, current), middle in zip(RECORDS, MIDDLE_VALUES, strict=True):
            for offset, value in enumerate((first, middle, current)):
                effective = today - timedelta(days=2 - offset)
                result = writer.save(f'{name} | {field} | {value} | {effective}', today)
                values[result['record']] = value
    return values


def paired_scope_metrics(snapshots, pairs):
    states = snapshots[:, -1].reshape(len(pairs), 2, 2)
    target = torch.tensor([[True, True], [False, True]])
    return int(states.eq(target).flatten(1).all(1).sum())


def compare_paired(args, model, tokenizer, reader_file, writer_file, binding, groups):
    held_pairs = paired_rows(groups['held'], (HELD_PAIR,))
    training_pairs = paired_rows(groups['train'], PAIR_FORMS)
    training = list(dict.fromkeys([*groups['train'], *(row for pair in training_pairs for row in pair)]))
    index = {row: position for position, row in enumerate(training)}
    pair_indices = [(index[first], index[second]) for first, second in training_pairs]
    held = [*groups['held'], *(row for pair in held_pairs for row in pair)]
    groups = {**groups, 'train': training, 'held': held}
    extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
    data = transition_features(extract, groups, utterance_features=True)
    path = args.output / 'three-versions.sqlite3'
    values = three_version_archive(path, binding, model, tokenizer, writer_file)
    if len(values) != 3 * len(RECORDS) or any(value in text for row in held for text in (*row.events, row.question)
                                             for value in values.values()):
        raise ValueError('The replay archive must contain three versions per address with no visible answers')
    archive_digest = base.sha(path)
    embedding = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    weights = torch.load(reader_file, map_location='cpu', weights_only=True)
    reader = VersionReplyActions(model.config.hidden_size, weights['scaffold_tokens'], slots=2)
    reader.load_state_dict(weights)
    reader.eval().to(model.device)

    results = {}
    with VersionStore(path, binding) as store:
        catalog = store.catalog()
        if len(catalog) != len(RECORDS) or any(len(versions) != 3 for versions in catalog.values()):
            raise ValueError('Every address must have three distinct records after reopening')
        for name, contrast in (('classification', False), ('paired', True)):
            head = fit(data['train'], training, args.steps, pairs=pair_indices, contrast=contrast)
            checkpoint = args.output / f'{name}.pt'
            atomic_save(head.state_dict(), checkpoint)
            head.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
            with torch.inference_mode():
                thresholds = calibrated_thresholds(head, data['calibration'], groups['calibration'])
                plan, snapshots, expected = predict(head, data['held'], held, thresholds=thresholds)
                _, confirmed, _ = predict(head, data['held'], held, confirmed_opening=True)
                oracle, _, _ = predict(head, data['held'], held, oracle_scope=True)
                result = evaluate(held, plan, catalog, store, model, tokenizer, codebook, reader, values=values)
                paired_start = len(held) - 2 * len(held_pairs)
                result.update(thresholds=thresholds, scope=scope_metrics(snapshots, expected, held),
                              confirmed_scope=scope_metrics(confirmed, expected, held),
                              pairs_correct=paired_scope_metrics(snapshots[paired_start:], held_pairs),
                              confirmed_pairs_correct=paired_scope_metrics(confirmed[paired_start:], held_pairs),
                              pairs_total=len(held_pairs), oracle_scope_metrics=plan_metrics(held, oracle),
                              checkpoint_sha256=base.sha(checkpoint))
            results[name] = result
            print(name, thresholds, 'paired', result['pairs_correct'], result['confirmed_pairs_correct'],
                  result['summary'], flush=True)
    if base.sha(path) != archive_digest:
        raise RuntimeError('The replay altered the written archive')
    return dict(results=results, values=values, archive_sha256=archive_digest, training_rows=len(training),
                training_pairs=len(training_pairs), held_rows=len(held), original_held_rows=len(groups['held']) - 2 * len(held_pairs))


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    archive = args.prior / 'profile' / 'versions.sqlite3'
    manifest = json.loads((args.prior / 'result.json').read_text(encoding='utf-8'))
    reader_file, normalizer_file, writer_file, binding = artifacts(args.reader, args.normalizer)
    protected = (archive, reader_file, normalizer_file, writer_file, args.recurrent[0],
                 *(folder / 'attention.pt' for folder in args.recurrent[1:]))
    sources = {path: base.sha(path) for path in protected}
    if sources[archive] != manifest['archive_sha256']:
        raise ValueError('Archive does not match the completed replay')
    groups = {split: episodes(split) for split in PEOPLE}
    held = groups['held']
    if any(value in text for row in held for text in (*row.events, row.question) for value in VALUES.values()):
        raise ValueError('An evaluation prompt contains a stored value')
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    if args.paired_transitions:
        with local_model(args.recurrent) as model:
            result = compare_paired(args, model, tokenizer, reader_file, writer_file, binding, groups)
        if any(base.sha(path) != digest for path, digest in sources.items()):
            raise RuntimeError('A protected input changed')
        store_json(args.output / 'result.json', dict(**result, steps=args.steps,
                   sources={str(path): digest for path, digest in sources.items()}, seconds=perf_counter() - started))
        return
    with local_model(args.recurrent) as model, VersionStore(archive, binding) as store:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
        data = transition_features(extract, groups, utterance_features=args.utterance_features)
        head = fit(data['train'], groups['train'], args.steps)
        checkpoint = args.output / 'transitions.pt'
        atomic_save(head.state_dict(), checkpoint)
        head.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        with torch.inference_mode():
            thresholds = calibrated_thresholds(head, data['calibration'], groups['calibration'])
            plan, snapshots, expected = predict(head, data['held'], held, thresholds=thresholds)
            raw, _, _ = predict(head, data['held'], held)
            oracle, _, _ = predict(head, data['held'], held, oracle_scope=True)
            _, confirmed, _ = predict(head, data['held'], held, confirmed_opening=True)
            current, history, valid = data['held']
            swapped = head.advance(history.flip(2), valid)
            if not torch.equal(swapped, snapshots.flip(2)):
                raise RuntimeError('Candidate order changed the transition')
            metrics = scope_metrics(snapshots, expected, held)
            print('scope', metrics, flush=True)
            weights = torch.load(reader_file, map_location='cpu', weights_only=True)
            reader = VersionReplyActions(model.config.hidden_size, weights['scaffold_tokens'], slots=2)
            reader.load_state_dict(weights)
            reader.eval().to(model.device)
            embedding = model.get_input_embeddings().weight.detach()
            columns = compress_dimension(embedding.shape[1]).to(embedding.device)
            codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
            result = evaluate(held, plan, store.catalog(), store, model, tokenizer, codebook, reader)
    if any(base.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A protected input changed')
    print('calibrated', thresholds, result['summary'], flush=True)
    store_json(args.output / 'result.json', dict(**result, thresholds=thresholds, scope=metrics,
               snapshots=snapshots.tolist(), raw_plan=raw, oracle_scope_plan=oracle,
               confirmed_opening_scope=scope_metrics(confirmed, expected, held), utterance_features=args.utterance_features,
               raw_metrics=plan_metrics(held, raw), oracle_scope_metrics=plan_metrics(held, oracle),
               checkpoint_sha256=base.sha(checkpoint), steps=args.steps,
               sources={str(path): digest for path, digest in sources.items()}, seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--recurrent', nargs=3, type=Path, required=True)
    cli.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    cli.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    cli.add_argument('--steps', type=int, default=240)
    cli.add_argument('--utterance-features', action='store_true')
    cli.add_argument('--paired-transitions', action='store_true')
    args = cli.parse_args()
    if args.steps < 1:
        cli.error('--steps must be positive')
    main(args)
