"""Test whether a read decision depends on the active slots as well as the question."""

import argparse
import json
from pathlib import Path

from scripts.audit_spora_working_state_read_gate import (
    DATA, MODEL, SEED, SOURCE, AutoTokenizer, FrozenTurnFeatures, dialogue_data, encode_questions, frozen_digest,
    load_full_streamed, load_integrated_runtime, measure, settings_for, sha, state_from, torch,
)
from spora.language_training import atomic_save
from spora.research_dialogue_working_state import FIELDS, WorkingTurnReader, WorkingTurnState


def present_fields(state):
    return [bool(value) for value in state.slots]


def fit(features, training, masks):
    questions = list(training)
    vectors = encode_questions(features, questions)
    pairs = [(index, mask, label == kind) for index, label in enumerate(training.values())
             for kind, mask in masks.items()]
    indices, slots, labels = zip(*pairs, strict=True)
    inputs = vectors[list(indices)]
    slots = torch.tensor(slots, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    gate = torch.nn.Linear(inputs.shape[-1], len(FIELDS))
    optimizer = torch.optim.AdamW(gate.parameters(), lr=.018, weight_decay=.05)
    for _ in range(160):
        optimizer.zero_grad(set_to_none=True)
        logits = read_scores(gate, inputs, slots)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
    return gate.eval().requires_grad_(False)


def read_scores(gate, questions, slots):
    return (gate(questions) * slots).sum(dim=-1) / slots.sum(dim=-1).clamp_min(1)


def fit_field_match(features, training, masks):
    names = tuple(masks)
    positive = {question: names.index(kind) for question, kind in training.items() if kind is not None}
    queries = encode_questions(features, list(positive))
    labels = torch.tensor(list(positive.values()))
    classifier = torch.nn.Linear(queries.shape[-1], len(names))
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=.015, weight_decay=.05)
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(classifier(queries), labels)
        loss.backward()
        optimizer.step()
    return classifier.eval().requires_grad_(False)


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    compare_path = args.copy / 'comparison.json'
    reader_path = args.copy / 'standard.pt'
    old_gate_path = args.gate / 'gate.pt'
    old_result_path = args.gate / 'comparison.json'
    protected = {path: sha(path) for path in (args.checkpoint, compare_path, reader_path, old_gate_path,
                                             old_result_path, DATA, SOURCE)}
    comparison = json.loads(compare_path.read_text(encoding='utf-8'))
    old_result = json.loads(old_result_path.read_text(encoding='utf-8'))
    if (comparison['arms']['standard']['sha256'] != protected[reader_path]
            or old_result['gate_sha256'] != protected[old_gate_path]
            or old_result['reader_sha256'] != protected[reader_path]):
        raise ValueError('Read experiment source differs from its records')
    gate_data = json.loads(SOURCE.read_text(encoding='utf-8'))
    train_data = json.loads(DATA.read_text(encoding='utf-8'))
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    digest = frozen_digest(runtime.organism, runtime.state)
    if digest != comparison['individual_sha256'] or digest != old_result['individual_sha256']:
        raise ValueError('Read experiment individual differs from its records')
    features = FrozenTurnFeatures(decoder, tokenizer)
    source_rows = dialogue_data(train_data, held=False, count=128)
    training = {row['question']: row['kind'] for row in source_rows}
    for kind, questions in gate_data['train'].items():
        training.update({question: kind for question in questions['read']})
        training.update(dict.fromkeys(questions['skip']))
    masks = {row['kind']: present_fields(state_from(row['initial'], tokenizer)) for row in source_rows}
    gate = fit(features, training, masks)
    field_match = fit_field_match(features, training, masks)
    expected_masks = torch.tensor(list(masks.values()), dtype=torch.bool)
    saved = torch.load(old_gate_path, map_location='cpu', weights_only=True)
    prior = torch.nn.Linear(gate.in_features, 1)
    prior.load_state_dict(saved['gate'], strict=True)
    prior.eval().requires_grad_(False)
    cases = comparison['arms']['standard']['splits']['new']['rows']
    representative = {row['case']['kind']: row for row in cases}
    held = [row for row in old_result['rows'] if row['split'] in ('new', 'hard', 'change_topic')]
    by_question = {row['case']['question']: row for row in cases}
    for (index, _, question) in gate_data['held']:
        by_question[question] = cases[index]
    topics = [row for row in held if row['split'] == 'change_topic']
    for row, reference in zip(topics, cases):
        by_question[row['question']] = reference
    questions = [row['question'] for row in held]
    if set(questions) & training.keys():
        raise ValueError('Evaluation questions overlap training')
    vectors = encode_questions(features, questions)
    rows = []
    for index, row in enumerate(held):
        question = row['question']
        reference = by_question[question]
        live = reference['case']['kind']
        for kind, alternate in representative.items():
            values = (reference if kind == live else alternate)['case']['current']
            state = state_from(values, tokenizer)
            mask = torch.tensor([present_fields(state)], dtype=torch.float32)
            vector = vectors[index:index + 1]
            score = read_scores(gate, vector, mask).sigmoid().item()
            expected = bool(row['expected'] and kind == live)
            intent = prior(vector).sigmoid().item() >= saved['threshold']
            inferred = int(field_match(vector).argmax())
            matches = bool((mask.bool() & expected_masks[inferred]).sum() >= 2)
            rows.append({'question': question, 'split': row['split'], 'question_kind': live, 'state_kind': kind,
                         'expected': expected, 'read': score >= .5, 'probability': score,
                         'question_only': intent, 'intention_and_fields': intent and matches,
                         'predicted_schema': tuple(masks)[inferred]})
    reader_saved = torch.load(reader_path, map_location='cpu', weights_only=True)
    if reader_saved['checkpoint_sha256'] != protected[args.checkpoint]:
        raise ValueError('Reader belongs to a different model')
    reader = WorkingTurnReader(decoder.base.config.hidden_size, layer_index=reader_saved.get('layer', 19))
    reader.load_state_dict(reader_saved['reader'], strict=True)
    reader.to(decoder.base.device).eval().requires_grad_(False)
    settings = settings_for(decoder, tokenizer, 48)
    samples = []
    for question in questions:
        related = [row for row in rows if row['question'] == question]
        if not any(row['expected'] for row in related):
            continue
        reference = by_question[question]
        for row in related:
            if row['state_kind'] == row['question_kind']:
                continue
            values = representative[row['state_kind']]['case']['current']
            state = state_from(values, tokenizer)
            empty = WorkingTurnState()
            replies = {}
            for arm, snapshot in (('always', state), ('off', empty),
                                  ('gated', state if row['intention_and_fields'] else empty)):
                generated = measure(decoder, reader, tokenizer, settings, [], question, snapshot)
                generated.pop('seconds')
                replies[arm] = generated
            if replies['gated'] != replies['always' if row['intention_and_fields'] else 'off']:
                raise ValueError('Gated reply did not reproduce its selected route')
            samples.append({'question': question, 'state_kind': row['state_kind'],
                            'read': row['intention_and_fields'], **replies})
        if len(samples) >= 6:
            break
    if (frozen_digest(runtime.organism, runtime.state) != digest
            or any(sha(path) != value for path, value in protected.items())):
        raise ValueError('State-conditioned read experiment changed its sources')
    path = args.output / 'gate.pt'
    atomic_save({'format': 'spora-state-conditioned-read-gate-v1', 'gate': gate.state_dict(),
                 'field_match': field_match.state_dict(), 'fields': FIELDS, 'schemas': masks,
                 'reader_sha256': protected[reader_path], 'intent_sha256': protected[old_gate_path], 'threshold': .5}, path)
    with (args.output / 'comparison.json').open('x', encoding='utf-8') as stream:
        json.dump({'sources': {str(path): value for path, value in protected.items()}, 'gate_sha256': sha(path),
                   'individual_sha256': digest, 'training_questions': len(training), 'rows': rows, 'samples': samples},
                  stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for name in ('new', 'hard', 'change_topic'):
        group = [row for row in rows if row['split'] == name]
        print(name, len(group), 'state-conditioned', sum(row['read'] == row['expected'] for row in group),
              'question-only', sum(row['question_only'] == row['expected'] for row in group),
              'intention-and-fields', sum(row['intention_and_fields'] == row['expected'] for row in group), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('copy', type=Path)
    parser.add_argument('gate', type=Path)
    parser.add_argument('output', type=Path)
    main(parser.parse_args())
