"""Measure the effect of automatic name boundaries on slot correction and recall."""

import argparse
from pathlib import Path

from scripts.audit_spora_entity_field_address import (
    F, FIELDS, HELD_NAMES, HELD_TEXT, HELD_VALUES, MODEL, SEED, AutoTokenizer, FrozenTurnFeatures,
    IdentityFingerprints, JointAddress, RecurrentFieldSlots, TokenValue, examples, frozen_digest,
    load_full_streamed, load_integrated_runtime, representations, sha, store_json, torch,
)
from scripts.audit_registered_name_teacher import annotate
from transformers import AutoModelForTokenClassification


def observations(group, foreign, index):
    for i, row in enumerate(group):
        name = row['name']
        if i % len(FIELDS) == (i // len(FIELDS) + index) % len(FIELDS):
            yield ('update', i, row['texts'][1])
        yield ('query', i, row['texts'][2])
        # Counterfactuals retain the request and value while changing only the subject.
        replacement = foreign[i // len(FIELDS)]
        other = group[((i // len(FIELDS) + 1) % 4) * len(FIELDS)]['name']
        for kind, value in (('unknown', replacement), ('suffix', name + '越'), ('two_names', name + '和' + other)):
            yield (kind, i, row['texts'][1].replace(name, value, 1))


def locate(mode, identities, text, spans, gold_name):
    if mode == 'confirmed':
        start = text.index(gold_name)
        return start, start + len(gold_name), identities.lookup(gold_name)
    candidates = identities.matching_spans(text, max_name_chars=8)
    if len(candidates) != 1:
        return None
    start, end, _ = candidates[0]
    if mode == 'fingerprint':
        return candidates[0]
    if [start, end] not in spans or (mode == 'neural_single' and len(spans) != 1):
        return None
    return candidates[0]


def measure(group, events, spans, hidden, initial, codec, tokenizer, head, mode, *, selector=locate, field_keys=None):
    slots = RecurrentFieldSlots(len(group), initial.shape[-1], codec.width)
    identities = IdentityFingerprints(4)
    for owner, row in enumerate(group[::len(FIELDS)]):
        identities.bind(owner, row['name'])
    values = [[tokenizer.encode(value, add_special_tokens=False) for value in row['values']] for row in group]
    for key, (first, _) in zip(initial, values, strict=True):
        slots.allocate(key, codec.encode(first))

    changed = {i for kind, i, _ in events if kind == 'update'}
    details = []
    for phase in ('update', 'query', 'negative'):
        for request, ((kind, i, text), predicted, state) in enumerate(zip(events, spans, hidden, strict=True)):
            if kind != phase and (phase != 'negative' or kind in ('update', 'query')):
                continue
            if mode == 'confirmed' and phase == 'negative':
                continue
            mention = selector(mode, identities, text, predicted, group[i]['name'])
            owner = selected = None
            if mention is not None:
                start, end, owner = mention
                if field_keys is None:
                    entity = F.normalize(state[start:end].mean(dim=0), dim=0)
                    context = F.normalize(state[:len(text)].mean(dim=0), dim=0)
                    key = head(entity, context)
                else:
                    key = field_keys[request]
                if key is not None:
                    scope = slice(owner * len(FIELDS), (owner + 1) * len(FIELDS))
                    if kind == 'update':
                        selected = slots.revise(key, codec.encode(values[i][1]), eligible=scope)
                    else:
                        selected = slots.select(key, eligible=scope)
            actual = codec.decode(slots.values[selected]) if selected is not None and kind == 'query' else None
            details.append({'kind': kind, 'text': text, 'slot': selected, 'target': i,
                            'owner': owner, 'predicted_spans': predicted, 'changed': i in changed,
                            'value_correct': kind == 'query' and selected == i and actual == values[i][int(i in changed)]})
    return details


def summarize(rows):
    updates = [row for row in rows if row['kind'] == 'update']
    queries = [row for row in rows if row['kind'] == 'query']
    return {'update_correct': sum(row['slot'] == row['target'] for row in updates),
            'update_wrong': sum(row['slot'] is not None and row['slot'] != row['target'] for row in updates),
            'update_skipped': sum(row['slot'] is None for row in updates),
            'revised_read_correct': sum(row['value_correct'] and row['changed'] for row in queries),
            'retained_read_correct': sum(row['value_correct'] and not row['changed'] for row in queries),
            'unsafe_negative_routes': {kind: sum(row['slot'] is not None for row in rows if row['kind'] == kind)
                                       for kind in ('unknown', 'suffix', 'two_names')},
            'negative_counts': {kind: sum(row['kind'] == kind for row in rows)
                                for kind in ('unknown', 'suffix', 'two_names')}}


def main(checkpoint, addresses, encoder, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    snapshots = {path: sha(path) for path in (checkpoint, addresses, encoder / 'model.safetensors')}
    saved = torch.load(addresses, map_location='cpu', weights_only=True)
    if saved['format'] != 'spora-entity-field-address-probe-v1' or saved['checkpoint_sha256'] != snapshots[checkpoint]:
        raise ValueError('Incompatible language and address weights')
    source = examples(HELD_NAMES, HELD_TEXT, HELD_VALUES, SEED + 1)
    size = 4 * len(FIELDS)
    groups = [source[start:start + size] for start in range(0, len(source), size)]
    batches = [list(observations(group, [row['name'] for row in groups[(i + 1) % len(groups)][::len(FIELDS)]], i))
               for i, group in enumerate(groups)]
    name_tokens = AutoTokenizer.from_pretrained(encoder, local_files_only=True, trust_remote_code=False)
    name_model = AutoModelForTokenClassification.from_pretrained(
        encoder, local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()
    predictions = [[item['bio'] for item in annotate(name_model, name_tokens, [text for _, _, text in batch])]
                   for batch in batches]
    del name_model
    print('Automatic name spans captured', sum(map(len, predictions)), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    before = frozen_digest(runtime.organism, runtime.state)
    if before != saved['individual_sha256']:
        raise ValueError('Address weights belong to a different individual state')
    head = JointAddress(saved['hidden'])
    head.load_state_dict(saved['heads']['JointAddress'], strict=True)
    head.eval().requires_grad_(False)
    features = FrozenTurnFeatures(runtime.decoder, tokenizer)
    codec = TokenValue(len(tokenizer))
    results = {mode: [] for mode in ('confirmed', 'fingerprint', 'neural_single', 'neural_registered')}
    for group, events, spans in zip(groups, batches, predictions, strict=True):
        initial = head(*representations(features, group))[:, 0]
        hidden = features([text for _, _, text in events]).float()
        for mode in results:
            results[mode].extend(measure(group, events, spans, hidden, initial, codec, tokenizer, head, mode))
    if before != frozen_digest(runtime.organism, runtime.state) or any(sha(p) != s for p, s in snapshots.items()):
        raise ValueError('Input weights or individual state changed during the audit')
    summary = {mode: summarize(rows) for mode, rows in results.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    store_json(output, {'sources': {path.name: value for path, value in snapshots.items()},
                        'individual_sha256': before, 'summary': summary, 'rows': results})
    for mode, record in summary.items():
        print(mode, record, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'addresses', 'encoder', 'output'):
        parser.add_argument(name, type=Path)
    main(**vars(parser.parse_args()))
