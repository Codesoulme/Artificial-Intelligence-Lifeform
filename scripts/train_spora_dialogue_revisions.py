"""Train field revision operations on frozen DeepSeek and evaluate new dialogues."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import torch

from scripts.audit_spora_dialogue_evidence import generate, load_spora_for_eval, scores
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.deepseek_local_baseline import sha
from scripts.spora_dialogue_revision_data import DEVELOPMENT, HELD, LABELS, ORACLE_PROMPTS, VALUES
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_dialogue_revisions import DialogueRevisionWriter, Revision, revise
from spora.research_dialogue_working_state import SPAN_LIMIT, characters


FIELDS = tuple(LABELS)
SEED = 31707
EPOCHS = 32


def scene_rows(turns, updates):
    state = {}
    rows = []
    for stage, change in zip(turns, updates, strict=True):
        rows.append({'text': stage['text'], 'before': state, 'change': change, 'after': revise(state, change)})
        state = rows[-1]['after']
    return rows


def training_rows(source, review):
    rows = []
    exported = {row['case']: row for row in review['candidates']['spora']}
    for case in source['rows']:
        reference = exported[case['case']]
        if reference['request'] != case['stages'][-1]['text']:
            raise ValueError('The reviewed trajectory no longer matches its annotation')
        rows.extend(scene_rows(case['stages'], DEVELOPMENT[case['case']]))
    for field, (old, new) in VALUES.items():
        label = LABELS[field]
        for value, alternative in ((old, new), (new, old)):
            before = {field: value}
            wishes = (
                (f'给这次任务的{label}设为{value}。', {}, {field: value}),
                (f'原{label}是{value}，改按{alternative}处理。', before, {field: alternative}),
                (f'{label}不再要求{value}，这一项撤掉。', before, {field: None}),
                (f'我提到{label}里的“{value}”只是举例，要求先不变。', before, {}),
                (f'维持原{label}，当前仍用{value}。', before, {}),
            )
            rows.extend({'text': text, 'before': earlier, 'change': change}
                        for text, earlier, change in wishes)
    for index, field in enumerate(FIELDS):
        other = FIELDS[(index + 1) % len(FIELDS)]
        first, _ = VALUES[field]
        old, new = VALUES[other]
        before = {field: first, other: old}
        rows.append({'text': f'{LABELS[field]}仍用{first}，{LABELS[other]}从{old}换到{new}。', 'before': before,
                     'change': {other: new}})
        rows.append({'text': f'{LABELS[other]}保留{old}，{LABELS[field]}的{first}不列入。', 'before': before,
                     'change': {field: None}})
    return rows


def labels(rows):
    operations, boundaries = [], []
    for row in rows:
        text, changes = row['text'], row['change']
        if changes.keys() - LABELS.keys():
            raise ValueError('Unsupported training field')
        kinds, spans = [], []
        for field in FIELDS:
            if field not in changes or changes[field] is None:
                kinds.append(Revision.REMOVE if field in changes else Revision.KEEP)
                spans.append((0, 0))
                continue
            value = changes[field]
            if not 0 < len(value) <= SPAN_LIMIT or value not in text:
                raise ValueError(f'The target must be a bounded span of the user turn: {field}, {value}, {text}')
            start = text.index(value) + 1
            kinds.append(Revision.SET)
            spans.append((start, start + len(value) - 1))
        operations.append(kinds)
        boundaries.append(spans)
    return torch.tensor(operations), torch.tensor(boundaries)


def state_text(state):
    lines = [f'{LABELS[field]}：{state[field]}' for field in FIELDS if field in state]
    return '\n'.join(lines)


def features(extractor, texts, states):
    codes = characters(texts)
    padded = None
    for start in range(0, len(texts), 24):
        part = texts[start:start + 24]
        before = states[start:start + 24]
        prefixes = ['现行要求\n' + state_text(state) + '\n用户发言\n' for state in before]
        raw = extractor([prefix + text for prefix, text in zip(prefixes, part, strict=True)])
        if padded is None:
            padded = raw.new_zeros((len(texts), codes.shape[1], raw.shape[-1]))
        for index, (prefix, text) in enumerate(zip(prefixes, part, strict=True)):
            padded[start + index, :len(text)] = raw[index, len(prefix):len(prefix) + len(text)]
    return codes, padded


def train(model, extractor, rows):
    texts = [row['text'] for row in rows]
    codes, hidden = features(extractor, texts, [row['before'] for row in rows])
    ops, spans = labels(rows)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    random = torch.Generator().manual_seed(SEED)
    trace = []
    for epoch in range(EPOCHS):
        total = 0.
        batches = torch.randperm(len(rows), generator=random).split(24)
        for indices in batches:
            loss = model.loss(codes[indices].to('cuda'), ops[indices].to('cuda'), spans[indices].to('cuda'), hidden[indices].to('cuda'))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
        trace.append(total / len(batches))
        if epoch in (0, EPOCHS // 2, EPOCHS - 1):
            print('revision loss', epoch + 1, trace[-1], flush=True)
    return trace


def rollout(model, extractor, examples, *, use_gold_history=False):
    before = [{} for _ in examples]
    results = [[] for _ in examples]
    for turn in range(len(examples[0])):
        expected = [scene[turn] for scene in examples]
        texts = [row['text'] for row in expected]
        context = [row['before'] for row in expected] if use_gold_history else before
        codes, hidden = features(extractor, texts, context)
        updates = model.updates(texts, codes.to('cuda'), hidden.to('cuda'))
        for index, (row, predicted) in enumerate(zip(expected, updates, strict=True)):
            state = revise(context[index], predicted)
            results[index].append({**row, 'used': context[index], 'predicted': predicted, 'computed': state})
            before[index] = state
    return results


def summary(rollouts):
    count = Counter(scenes=len(rollouts))
    for rows in rollouts:
        count['exact_final_state'] += rows[-1]['computed'] == rows[-1]['after']
        for row in rows:
            expected, predicted, computed = row['change'], row['predicted'], row['computed']
            count['turns'] += 1
            count['exact_updates'] += predicted == expected
            count['exact_state'] += computed == row['after']
            for field, value in expected.items():
                label = 'remove' if value is None else 'set'
                count[label] += 1
                count[label + '_ok'] += field in predicted and predicted[field] == value
            for field in predicted.keys() - expected.keys():
                count['unwanted_updates'] += 1
            for field in row['before'].keys() - expected.keys():
                count['retain'] += 1
                count['retain_ok'] += computed.get(field) == row['before'][field]
            for field, value in row['before'].items():
                count['carried_obsolete'] += field in expected and value != expected[field] and computed.get(field) == value
    return count


def answer(chat, question, history, spec):
    result = generate(chat.decoder, chat.tokenizer, history, question, min(256, chat.position_limit))
    del result['seconds']
    return {**result, 'score': scores(result['answer'], spec)}


def evaluate_output(chat, predicted, given):
    counts = Counter()
    rows = []
    for spec, rollout, oracle in zip(HELD, predicted, given, strict=True):
        name, task, stages, _ = spec
        history = []
        for stage in stages[:-1]:
            raw = answer(chat, stage['text'], history, stage)
            history.append((stage['text'], raw['answer']))
        modes = {'raw': answer(chat, stages[-1]['text'], history, stages[-1])}
        states = {'learned': rollout[-1]['computed'], 'true_state': rollout[-1]['after']}
        for mode, state in states.items():
            prompt = f'{task}\n按以下现行条件作答。\n{state_text(state)}'
            modes[mode] = {'prompt': prompt, **answer(chat, prompt, [], stages[-1])}
        counts.update({mode: out['score']['pass'] for mode, out in modes.items()})
        print(name, {mode: out['score']['pass'] for mode, out in modes.items()}, flush=True)
        rows.append({'name': name, 'predicted': rollout, 'given_prior': oracle, 'outputs': modes})
    return counts, rows


def natural_oracle(chat, primary):
    paired = []
    for name, _, stages, _ in HELD:
        original, = (row for row in primary['held'] if row['name'] == name)
        question = ORACLE_PROMPTS[name]
        output = answer(chat, question, [], stages[-1])
        print('oracle wording', name, output['score']['pass'], flush=True)
        paired.append({'name': name, 'natural_prompt': question, 'natural': output,
                       'structured': original['outputs']['true_state']})
    return {'natural_pass': sum(row['natural']['score']['pass'] for row in paired),
            'structured_pass': sum(row['structured']['score']['pass'] for row in paired), 'rows': paired}


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    path = args.output.resolve()
    if not args.verify:
        path.mkdir(parents=True, exist_ok=False)
    files = (args.evidence / 'spora.json', args.evidence / 'recovery-review.json')
    sources = {str(file): sha(file) for file in files}
    source, review = (json.loads(file.read_text(encoding='utf-8')) for file in files)
    if review['sources']['spora'] != sources[str(files[0])]:
        raise ValueError('Review artifacts do not belong to the paired comparison')
    saved, weights = path / 'result.json', path / 'head.pt'
    old = json.loads(saved.read_text(encoding='utf-8')) if args.verify else None
    if old is not None and (old['sources'] != sources or old['weights_sha'] != sha(weights)):
        raise ValueError('The replay inputs changed')
    training = training_rows(source, review)
    held = [scene_rows(stages, updates) for _, _, stages, updates in HELD]
    if {row['text'] for row in training}.intersection(row['text'] for scene in held for row in scene):
        raise ValueError('Held-out turns overlap training')
    labels(training)
    labels([row for scene in held for row in scene])
    chat = load_spora_for_eval(path)
    if old is not None and old['runtime'] != chat.awake_individual:
        raise ValueError('The original decoder configuration changed')
    capture = FrozenTurnFeatures(chat.decoder, chat.tokenizer)
    head = DialogueRevisionWriter(chat.decoder.base.config.hidden_size, FIELDS).to('cuda')
    if old is None:
        trace = train(head, capture, training)
        atomic_save({'state': head.cpu().state_dict(), 'fields': FIELDS}, weights)
        head.to('cuda')
    else:
        state = torch.load(weights, map_location='cpu', weights_only=True)
        if state['fields'] != FIELDS:
            raise ValueError('Working-state fields changed')
        head.load_state_dict(state['state'], strict=True)
        trace = old['training_loss']
    head.eval()
    predicted = rollout(head, capture, held)
    given = rollout(head, capture, held, use_gold_history=True)
    answer_counts, rows = evaluate_output(chat, predicted, given)
    report = {'sources': sources, 'runtime': chat.awake_individual, 'weights_sha': sha(weights),
              'training_examples': len(training), 'training_loss': trace,
              'predicted_prior': summary(predicted), 'correct_prior': summary(given), 'answer_counts': answer_counts,
              'held': rows}
    if old is not None and json.loads(json.dumps(report)) != {key: old[key] for key in report}:
        raise ValueError('The saved head did not reproduce held-out states or generated tokens')
    if any(sha(Path(name)) != digest for name, digest in sources.items()):
        raise ValueError('Protected supervision changed')
    store_json(path / ('reload.json' if args.verify else 'result.json'),
               {**report, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})
    print(json.dumps({key: report[key] for key in ('predicted_prior', 'correct_prior', 'answer_counts')}), flush=True)
    if args.verify:
        store_json(path / 'oracle-format.json', natural_oracle(chat, report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
