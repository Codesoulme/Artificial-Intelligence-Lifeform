"""Audit explicit learned transitions of the active memory address set."""

import argparse
from collections import Counter
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
from scripts.spora_version_chat import VersionReplyActions, VersionStore, artifacts, base, compress_dimension, local_model
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save


KEEP, ADD, REMOVE = range(3)


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


def fit(data, rows, steps):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    head = LookupTransitions(data[0].shape[-1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    operations = operation_targets(rows)
    # A bare version retains its meaning even when the address set is ambiguous.
    version_for = {row.question: row.version for row in rows if row.version >= 0}
    versions = torch.tensor([version_for.get(row.question, 3) for row in rows])
    for _ in range(steps):
        batch = torch.randperm(len(rows), generator=rng)[:32]
        current, history, valid = (value[batch] for value in data)
        predicted = head.operation(history)[valid].flatten(0, 1)
        targets = operations[batch][valid].flatten()
        operation_loss = F.cross_entropy(predicted, targets, reduction='sum') / max(1, targets.numel())
        loss = operation_loss + F.cross_entropy(head.version(current), versions[batch])
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
    args = cli.parse_args()
    if args.steps < 1:
        cli.error('--steps must be positive')
    main(args)
