"""Choose a semantic memory target before validating the target's enrollment."""

import argparse
import json
from pathlib import Path

from scripts.train_spora_joint_name_role import (
    CALIBRATION, CALIBRATION_NAMES, F, FRESH, FRESH_NAMES, HELD_NAMES, HELD_TEXT, HELD_VALUES, MODEL, SEED,
    TRAIN, TRAIN_NAMES, AutoTokenizer, FrozenTurnFeatures, JointAddress, atomic_save, decisions, examples, fit,
    frozen_digest, load_full_streamed, load_integrated_runtime, pad_candidates, role_examples, sha, store_json,
    summary, torch,
)
from scripts.train_spora_name_role import FIELDS, TokenValue, identity_group, measure, observations, representations, summarize
from spora.research_name_role import (
    JointNameRoleHead, NameSpanHead, mention_features, select_nonoverlapping_spans, validate_enrollment,
)


FORMAT = 'spora-open-name-roles-v1'


def hidden_batch(featurize, rows):
    batch = featurize([row['text'] for row in rows])
    lengths = torch.tensor([len(row['text']) for row in rows])
    inside = torch.arange(batch.shape[1])[None, :] < lengths[:, None]
    return batch.masked_fill(~inside[..., None], 0), lengths


def fit_proposer(proposer, hidden, lengths, rows, *, epochs=12, lr=.003):
    valid = torch.arange(hidden.shape[1])[None, :, None] + torch.tensor(proposer.lengths) <= lengths[:, None, None]
    targets = torch.zeros_like(valid, dtype=torch.float32)
    for i, row in enumerate(rows):
        for start, stop in row['mentions']:
            targets[i, start, proposer.lengths.index(stop - start)] = 1.
    rng = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.AdamW(proposer.parameters(), lr=lr, weight_decay=.03)
    positive_weight = (valid.sum() - targets.sum()) / targets.sum() / 3
    for _ in range(epochs):
        for ids in torch.randperm(len(rows), generator=rng).split(64):
            optimizer.zero_grad(set_to_none=True)
            active = valid[ids]
            loss = F.binary_cross_entropy_with_logits(proposer(hidden[ids])[active], targets[ids][active],
                                                     pos_weight=positive_weight)
            loss.backward()
            optimizer.step()
    return proposer.eval().requires_grad_(False)


def proposals(proposer, hidden, lengths, floor):
    return [select_nonoverlapping_spans(scores, length, threshold=floor)
            for batch, sizes in zip(hidden.split(64), lengths.split(64), strict=True)
            for scores, length in zip(proposer(batch).tolist(), sizes.tolist(), strict=True)]


def span_metrics(rows, found):
    correct = sum(len(set(row['mentions']) & set(spans)) for row, spans in zip(rows, found, strict=True))
    return {'correct': correct, 'expected': sum(len(row['mentions']) for row in rows),
            'proposed': sum(map(len, found))}


def proposal_floor(proposer, hidden, lengths, rows):
    def f1(floor):
        stats = span_metrics(rows, proposals(proposer, hidden, lengths, floor))
        return 2 * stats['correct'] / (stats['expected'] + stats['proposed'])

    return max((0., 1., 2., 3.), key=f1)


def role_batch(rows, hidden, proposals_by_row):
    labelled, feats, counts = [], [], []
    for row, state, spans in zip(rows, hidden, proposals_by_row, strict=True):
        candidates = [(start, stop, i) for i, (start, stop) in enumerate(spans)]
        target = row.get('semantic_target')
        labelled.append({**row, 'candidates': candidates, 'complete': [span in row['mentions'] for span in spans],
                         'target': spans.index(target) if target in spans else None})
        counts.append(len(candidates))
        feats.extend(mention_features(state[:len(row['text'])], start, stop) for start, stop, _ in candidates)
    inputs = torch.stack(feats) if feats else torch.empty(0, 6, hidden.shape[-1])
    return labelled, pad_candidates(inputs, counts)


def resolve_rows(role, rows, hidden, spans_by_row, *, margin_floor=0.):
    labelled, features = role_batch(rows, hidden, spans_by_row)
    picked = decisions(role, labelled, *features)
    output = []
    for row, (candidate, margin) in zip(rows, picked, strict=True):
        span = candidate[:2] if candidate is not None and margin > margin_floor else None
        output.append((span, validate_enrollment(span, row['candidates'])))
    return output


def annotate_group(group, foreign, events, names):
    rows = []
    registry = identity_group(names)
    for kind, i, text in events:
        row = group[i]
        owner = i // len(FIELDS)
        actor = row['name']
        if kind == 'unknown':
            actor = foreign[owner]
        elif kind == 'suffix':
            actor += '越'
        people = [actor]
        if kind == 'two_names':
            people.append(names[(owner + 1) % len(names)])
        if kind != 'query' and row['field'] == 'contact':
            people.append(row['values'][1])
        mentions = sorted({(text.index(person), text.index(person) + len(person)) for person in people})
        rows.append({'text': text, 'mentions': mentions,
                     'candidates': registry.matching_spans(text, max_name_chars=8)})
    return rows


def replay_roles(held, featurize, proposer, floor, role, joint, tokenizer):
    configs = {'proposed': (role, floor, 0.), 'annotated': (role, None, 0.)}
    return replay_profiles(held, featurize, proposer, configs, joint, tokenizer)


def replay_profiles(held, featurize, proposer, configs, joint, tokenizer):
    return replay_proposer_profiles(held, featurize,
                                    {name: (proposer, *profile) for name, profile in configs.items()}, joint, tokenizer)


def replay_proposer_profiles(held, featurize, configs, joint, tokenizer):
    size = len(FIELDS) * 4
    groups = [held[start:start + size] for start in range(0, len(held), size)]
    result = {name: [] for name in configs}
    codec = TokenValue(len(tokenizer))
    for i, group in enumerate(groups):
        names = [row['name'] for row in group[::len(FIELDS)]]
        foreign = [row['name'] for row in groups[(i + 1) % len(groups)][::len(FIELDS)]]
        events = list(observations(group, foreign, i))
        rows = annotate_group(group, foreign, events, names)
        hidden, lengths = hidden_batch(featurize, rows)
        initial = joint(*representations(featurize, group))[:, 0]
        bounds = dict.fromkeys((proposer, floor) for proposer, _, floor, _ in configs.values())
        spans = {(proposer, floor): [row['mentions'] for row in rows] if floor is None else
                 proposals(proposer, hidden, lengths, floor) for proposer, floor in bounds}
        for name, (proposer, role, floor, margin) in configs.items():
            selected = [registered for _, registered in resolve_rows(
                role, rows, hidden, spans[proposer, floor], margin_floor=margin)]
            report = measure(group, events, selected, hidden.float(), initial, codec, tokenizer, joint, name,
                             selector=lambda _mode, _registry, _text, chosen, _gold: chosen)
            result[name].extend(report)
    return {name: {'summary': summarize(rows), 'rows': rows} for name, rows in result.items()}


def main(checkpoint, addresses, output, verify):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    hashes = {'checkpoint': sha(checkpoint), 'addresses': sha(addresses)}
    base = torch.load(addresses, map_location='cpu', weights_only=True)
    if base['format'] != 'spora-entity-field-address-probe-v1' or base['checkpoint_sha256'] != hashes['checkpoint']:
        raise ValueError('Field address weights do not match this model')
    hashes['individual'] = base['individual_sha256']
    saved = torch.load(verify / 'roles.pt', map_location='cpu', weights_only=True) if verify else None
    if saved is not None and (saved['format'] != FORMAT or saved['sources'] != hashes):
        raise ValueError('Name proposal weights do not match these sources')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    if frozen_digest(runtime.organism, runtime.state) != hashes['individual']:
        raise ValueError('Field address weights belong to a different individual')
    featurize = FrozenTurnFeatures(runtime.decoder, tokenizer)
    proposer, role = NameSpanHead(base['hidden']), JointNameRoleHead(base['hidden'])
    if saved is None:
        train = role_examples(TRAIN_NAMES, TRAIN, '荣安云庆岚薇青岳东梦', separate_boundary=True, annotate_roles=True)
        hidden, lengths = hidden_batch(featurize, train)
        print('Captured name training hidden states', len(train), flush=True)
        fit_proposer(proposer, hidden, lengths, train)
        training_spans = [sorted(set(row['mentions']) | {(a, b) for a, b, _ in row['candidates']}) for row in train]
        labels, (inputs, present) = role_batch(train, hidden, training_spans)
        fit(role, inputs, present, labels, boundary_weight=1.)
        del hidden, lengths, inputs, present
    else:
        proposer.load_state_dict(saved['proposer'], strict=True)
        role.load_state_dict(saved['role'], strict=True)
        for head in (proposer, role):
            head.eval().requires_grad_(False)
    cal = role_examples(CALIBRATION_NAMES, CALIBRATION, '嘉河亭琛柠添', separate_boundary=True, annotate_roles=True)
    fresh = role_examples(FRESH_NAMES, FRESH, '飞川宁华星明', separate_boundary=True, annotate_roles=True)
    floor = proposal_floor(proposer, *hidden_batch(featurize, cal), cal)
    hidden, lengths = hidden_batch(featurize, fresh)
    proposed = proposals(proposer, hidden, lengths, floor)
    print('Fresh span metrics', floor, span_metrics(fresh, proposed), flush=True)
    arms = {}
    for name, spans in (('proposed', proposed), ('annotated', [row['mentions'] for row in fresh])):
        selected = resolve_rows(role, fresh, hidden, spans)
        rows = []
        for row, candidates, (actor, eligible) in zip(fresh, spans, selected, strict=True):
            owner = eligible[2] if eligible is not None else None
            rows.append({**row, 'proposed': candidates, 'actor': actor, 'owner': owner,
                         'actor_correct': actor == row['semantic_target'], 'correct': owner == row['target']})
        arms[name] = {'summary': summary(rows), 'rows': rows}
        print(name, arms[name]['summary'], flush=True)
    joint = JointAddress(base['hidden']).eval().requires_grad_(False)
    joint.load_state_dict(base['heads']['JointAddress'], strict=True)
    held = examples(HELD_NAMES, HELD_TEXT, HELD_VALUES, SEED + 1)
    repairs = replay_roles(held, featurize, proposer, floor, role, joint, tokenizer)
    for name, record in repairs.items():
        print(name, record['summary'], flush=True)
    if (sha(checkpoint) != hashes['checkpoint'] or sha(addresses) != hashes['addresses']
            or frozen_digest(runtime.organism, runtime.state) != hashes['individual']):
        raise ValueError('Source weights or individual state changed during this run')
    output.mkdir(parents=True, exist_ok=False)
    report = {'format': FORMAT, 'sources': hashes, 'proposal_floor': floor, 'spans': span_metrics(fresh, proposed),
              'arms': arms, 'slot_replay': repairs}
    if saved is None:
        atomic_save({'format': FORMAT, 'sources': hashes, 'proposer': proposer.state_dict(),
                     'role': role.state_dict()}, output / 'roles.pt')
        store_json(output / 'result.json', report)
    else:
        previous = json.loads((verify / 'result.json').read_text(encoding='utf-8'))
        if json.loads(json.dumps(report)) != previous:
            raise ValueError('Reloaded role model did not reproduce the audit')
        store_json(output / 'result.json', {'identical': True, 'sources': hashes})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'addresses', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', type=Path)
    main(**vars(parser.parse_args()))
