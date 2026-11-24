"""Trace where verified name spans stop before a memory write."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.audit_spora_correction_roles import FORMAT as ROLE_FORMAT
from scripts.audit_spora_missing_name_roles import (
    HELD_NAMES, HELD_TEXT, HELD_VALUES, MODEL, SEED, AutoTokenizer, FrozenTurnFeatures, JointAddress, JointNameRoleHead,
    examples, frozen_digest, hidden_batch, load_full_streamed, load_integrated_runtime, role_batch, sha, store_json,
    torch,
)
from scripts.train_spora_name_role import FIELDS, TokenValue, measure, observations, representations, summarize
from scripts.train_spora_open_name_roles import annotate_group
from spora.research_name_role import compare_name_scores, validate_enrollment
from torch.nn import functional as F


def traces(head, rows, hidden):
    spans = [row['mentions'] for row in rows]
    _, (inputs, present) = role_batch(rows, hidden, spans)
    decision, boundary = head(inputs, present)
    selected, margins = compare_name_scores(decision, boundary, present)
    output = []
    for row, names, scores, edges, winner, margin in zip(
            rows, spans, decision, boundary, selected.tolist(), margins.tolist(), strict=True):
        candidates = []
        for i, span in enumerate(names):
            left, right = span
            enrolled = validate_enrollment(span, row['candidates'])
            candidates.append({'text': row['text'][left:right], 'span': span,
                               'role': float(scores[i] - F.logsigmoid(edges[i])),
                               'joint': float(scores[i]), 'boundary': float(edges[i]),
                               'owner': None if enrolled is None else enrolled[2]})
        picked = candidates[winner]
        abstain = float(scores[-1])
        accepted = picked['owner'] if margin > 0. else None
        output.append({'text': row['text'], 'field': row['field'], 'kind': row['kind'], 'target': row['target'],
                       'variant': row['variant'], 'names': candidates, 'abstain': abstain, 'winner': winner,
                       'margin': margin, 'owner': accepted})
    return output


def audit(rows):
    actual = Counter()
    for row in rows:
        selected = row['names'][row['winner']]
        choices = [candidate for candidate in row['names'] if candidate['owner'] == row['target']]
        actual['count'] += 1
        if row['target'] is None:
            actual['rejected'] += row['owner'] is None
            actual['without_boundary_wrong'] += selected['owner'] is not None and selected['joint'] > row['abstain']
            actual['without_abstention_wrong'] += selected['owner'] is not None and selected['boundary'] > 0.
            actual['rank_only_wrong'] += selected['owner'] is not None
            continue
        if len(choices) != 1:
            raise ValueError('An annotated update must have exactly one registered target')
        target = choices[0]
        actual['accepted'] += row['owner'] == row['target']
        actual['rank_correct'] += selected is target
        actual['raw_role_rank_correct'] += max(row['names'], key=lambda candidate: candidate['role']) is target
        actual['boundary_positive'] += target['boundary'] > 0.
        actual['target_beats_abstention'] += target['joint'] > row['abstain']
        actual['without_boundary_correct'] += selected is target and target['joint'] > row['abstain']
        actual['without_abstention_correct'] += selected is target and target['boundary'] > 0.
    return dict(actual)


def explicit_events(group, foreign, names, events):
    rewritten = []
    for kind, index, text in events:
        row = group[index]
        field = row['field']
        if kind != 'query' and field in ('place', 'contact'):
            owner = index // len(FIELDS)
            name = names[owner]
            if kind == 'unknown':
                name = foreign[owner]
            elif kind == 'suffix':
                name += '越'
            elif kind == 'two_names':
                name += '和' + names[(owner + 1) % len(names)]
            value = row['values'][1]
            field_name = '会合地点' if field == 'place' else '对接联系人'
            text = f'请修改{name}那条记录，新的{field_name}是{value}。'
        rewritten.append((kind, index, text))
    return rewritten


def diagnostic_rows(group, foreign, names, events, variant):
    result = annotate_group(group, foreign, events, names)
    for row, (kind, i, _) in zip(result, events, strict=True):
        row.update({'kind': kind, 'field': group[i]['field'], 'variant': variant,
                    'target': i // len(FIELDS) if kind in ('update', 'query') else None})
    return result


def main(checkpoint, addresses, trained, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    paths = {'checkpoint': checkpoint, 'addresses': addresses, 'roles': trained / 'roles.pt', 'report': trained / 'result.json'}
    sources = {name: sha(path) for name, path in paths.items()}
    roles = torch.load(trained / 'roles.pt', map_location='cpu', weights_only=True)
    record = json.loads((trained / 'result.json').read_text(encoding='utf-8'))
    if (roles['format'] != ROLE_FORMAT or record['sources'] != roles['sources']
            or any(roles['sources'][key] != sources[key] for key in ('checkpoint', 'addresses'))):
        raise ValueError('Role data does not match the requested checkpoint')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    before = frozen_digest(runtime.organism, runtime.state)
    if before != roles['sources']['individual']:
        raise ValueError('Role weights do not match the individual')
    featurize = FrozenTurnFeatures(runtime.decoder, tokenizer)
    source = torch.load(addresses, map_location='cpu', weights_only=True)
    if (source['format'] != 'spora-entity-field-address-probe-v1' or source['individual_sha256'] != before
            or source['checkpoint_sha256'] != sources['checkpoint']):
        raise ValueError('Field address data does not match the requested checkpoint')
    joint = JointAddress(source['hidden']).eval().requires_grad_(False)
    joint.load_state_dict(source['heads']['JointAddress'], strict=True)
    codec = TokenValue(len(tokenizer))
    heads = {}
    for name, state in roles['heads'].items():
        width = state['project.weight'].shape[-1] // 6
        head = JointNameRoleHead(width)
        head.load_state_dict(state, strict=True)
        heads[name] = head.eval().requires_grad_(False)
    held = examples(HELD_NAMES, HELD_TEXT, HELD_VALUES, SEED + 1)
    size = len(FIELDS) * 4
    groups = [held[start:start + size] for start in range(0, len(held), size)]
    observations_by_head = {name: [] for name in heads}
    replay = {name: {'original': [], 'explicit_role': []} for name in heads}
    for i, group in enumerate(groups):
        names = [row['name'] for row in group[::len(FIELDS)]]
        foreign = [row['name'] for row in groups[(i + 1) % len(groups)][::len(FIELDS)]]
        events = list(observations(group, foreign, i))
        initial = joint(*representations(featurize, group))[:, 0]
        for variant, batch in (('original', events), ('explicit_role', explicit_events(group, foreign, names, events))):
            rows = diagnostic_rows(group, foreign, names, batch, variant)
            hidden, _ = hidden_batch(featurize, rows)
            for name, head in heads.items():
                details = traces(head, rows, hidden)
                observations_by_head[name].extend(row for row in details if row['kind'] != 'query')
                selected = [validate_enrollment(row['names'][row['winner']]['span'], observation['candidates'])
                            if row['margin'] > 0. else None
                            for row, observation in zip(details, rows, strict=True)]
                replay[name][variant].extend(measure(
                    group, batch, selected, hidden.float(), initial, codec, tokenizer, joint, variant,
                    selector=lambda _mode, _registry, _text, chosen, _gold: chosen))
    metrics = {}
    for name, rows in observations_by_head.items():
        conditions = {(row['variant'], row['kind'], row['field'] if row['kind'] == 'update' else 'all')
                      for row in rows}
        metrics[name] = {}
        for variant, kind, field in sorted(conditions):
            group = [row for row in rows if row['variant'] == variant and row['kind'] == kind
                     and (field == 'all' or row['field'] == field)]
            key = '_'.join((variant, kind, field))
            metrics[name][key] = audit(group)
            print(name, key, metrics[name][key], flush=True)
        expected = record['slot_replay'][name + '_annotated']
        if json.loads(json.dumps(replay[name]['original'])) != expected['rows']:
            raise ValueError('Diagnostic decisions differ from the recorded fixed-field replay')
    slot_results = {name: {variant: {'summary': summarize(rows), 'rows': rows} for variant, rows in variants.items()}
                    for name, variants in replay.items()}
    for name, profiles in slot_results.items():
        for mode, item in profiles.items():
            print('Slots', name, mode, item['summary'], flush=True)
    after = {name: sha(path) for name, path in paths.items()}
    if sources != after or before != frozen_digest(runtime.organism, runtime.state):
        raise ValueError('Input weights or individual state changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    store_json(output, {'sources': sources | {'individual': before}, 'summary': metrics, 'rows': observations_by_head,
                        'slot_replay': slot_results})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'addresses', 'trained', 'output'):
        parser.add_argument(name, type=Path)
    main(**vars(parser.parse_args()))
