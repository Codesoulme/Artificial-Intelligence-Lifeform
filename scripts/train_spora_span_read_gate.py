"""Train a state-conditioned route for the frozen whole-field decoder."""

import argparse
from contextlib import ExitStack
from itertools import product
from pathlib import Path
import re
from time import perf_counter

from scripts.audit_spora_reply_coverage import measures
from scripts.train_spora_span_actions import (
    DATA, FRESH, HELD, MODEL, ORDINARY_TRAIN, SCENARIOS, SOURCE, AutoTokenizer, DynamicCache, SpanActionHead,
    TRANSFER_FIELDS, backbone_step, evaluate, extend_positions, fresh_cases, frozen_digest, generate, grade, install,
    load_full_streamed, load_integrated_runtime, load_recurrent_replacements, prompt_ids, read, recall_prompts,
    records, sha, state_for, store_json, torch,
)
from spora.language_training import atomic_save
from spora.research_dialogue_working_state import FIELDS, WorkingTurnState
from spora.research_span_read_gate import StateReadGate
from torch.nn import functional as F


DESIGN = DATA.with_name('working-state-relevance-v1.json')
SEED = 26124


def vector(fields):
    return torch.tensor([field in fields for field in FIELDS], dtype=torch.bool)


def topic_prompt(description):
    return f'这条记事的主题是{description}。'


def training_data(fixture):
    topics = {row['name']: row for row in fixture['topics']}
    if len(topics) != len(fixture['topics']) or any(
        tuple(topics[name]['fields']) != fields for name, fields in TRANSFER_FIELDS.items()
    ):
        raise ValueError('Topic field schema differs from the prior read tests')
    for row in topics.values():
        if (row['name'] == row['other'] or tuple(row['fields']) != tuple(topics[row['other']]['fields'])
                or set(row['fields']) - set(FIELDS) or len(row['fields']) != len(row['example'])):
            raise ValueError('Every distractor must have the same fields and a different topic')
    training = {q: row for row in topics.values() for q in row['train']}
    ordinary = (*ORDINARY_TRAIN, *fixture['cal_ordinary'], *fixture['held_ordinary'])
    splits = [set(training) | set(ORDINARY_TRAIN), {row['cal'] for row in topics.values()} | set(fixture['cal_ordinary']),
              {row['question'] for row in fresh_cases(FRESH)} | set(fixture['held_ordinary'])
              | {case['turns'][0][0] for case in read(SOURCE)['open']}]
    if any(a & b for i, a in enumerate(splits) for b in splits[i+1:]) or len(set(ordinary)) != len(ordinary):
        raise ValueError('Read gate question splits overlap')
    return topics, training


@torch.inference_mode()
def encode_prompts(base, prompts):
    vectors = {}
    for ids in dict.fromkeys(tuple(prompt) for prompt in prompts):
        cache = DynamicCache(config=base.config)
        vectors[ids] = backbone_step(base, ids, cache).detach().float().cpu()
    return vectors


def fit(gate, features, tokenizer, training, topics):
    def get(text):
        return features[tuple(prompt_ids(tokenizer, [], text))]

    questions = torch.stack([get(q) for q in (*training, *ORDINARY_TRAIN)])
    wants = torch.stack([vector(row['fields']) for row in training.values()] + [vector(())] * len(ORDINARY_TRAIN))
    needs = wants.any(dim=-1).float()
    topic_keys = [(row['name'], get(topic_prompt(key))) for row in topics.values() for key in row['train_keys']]
    pairs = [
        (questions[i], key, train['name'] == name, train['fields'] == topics[name]['fields'])
        for i, train in enumerate(training.values()) for name, key in topic_keys
    ]
    q, key, y, same_schema = zip(*pairs, strict=True)
    question_pairs, keys = torch.stack(q), torch.stack(key)
    labels, schema = torch.tensor(y), torch.tensor(same_schema)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=.003, weight_decay=.02)
    for step in range(160):
        need, required = gate.requirements(questions)
        match = gate.relevance(question_pairs, keys)
        plain = F.binary_cross_entropy_with_logits(need, needs)
        fields = F.binary_cross_entropy_with_logits(required, wants.float())
        topic = (F.softplus(-match[labels]).mean() + F.softplus(match[~labels]).mean()
                 + F.softplus(match[~labels & schema]).mean()) / 3
        loss = plain + fields + topic
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 40 == 0:
            print('gate step', step + 1, round(float(loss.detach()), 5), flush=True)


def calibration_specs(topics, ordinary):
    items = []
    for row in topics.values():
        fields = row['fields']
        other = topics[row['other']]
        items.extend(((row['cal'], row['cal_key'], fields, 'relevant'),
                      (row['cal'], other['cal_key'], fields, 'wrong_topic'),
                      (row['cal'], row['cal_key'], fields[1:], 'missing_field'),
                      (row['cal'], row['cal_key'], (), 'empty')))
    rows = list(topics.values())
    items.extend((question, rows[i]['cal_key'], rows[i]['fields'], 'ordinary') for i, question in enumerate(ordinary))
    return items


@torch.inference_mode()
def calibrate(gate, features, tokenizer, specs):
    def get(text):
        return features[tuple(prompt_ids(tokenizer, [], text))]
    q = torch.stack([get(question) for question, _, _, _ in specs])
    keys = torch.stack([get(topic_prompt(key)) for _, key, _, _ in specs])
    present = torch.stack([vector(fields) for _, _, fields, _ in specs])
    need, required = gate.requirements(q)
    match = gate.relevance(q, keys)
    groups = tuple(dict.fromkeys(row[3] for row in specs))
    masks = [torch.tensor([row[3] == name for row in specs]) for name in groups]
    expected = masks[0]
    grid = (-3., -1.5, 0., 1.5, 3.)
    best, best_thresholds, score = None, None, None
    for parameters in product(grid, repeat=3):
        chosen = gate.permits(need, required, match, present, parameters)
        correct = chosen == expected
        metrics = [float(correct[mask].float().mean()) for mask in masks]
        criteria = (min(metrics), sum(metrics), sum(metrics[1:]), -sum(abs(x) for x in parameters))
        if best is None or criteria > best:
            best, best_thresholds, score = criteria, parameters, metrics
    return best_thresholds, dict(zip(groups, score, strict=True))


class BoundReadGate:
    """Bind a precomputed memory key to the current action decoder."""

    def __init__(self, gate, key, thresholds):
        self.gate, self.key, self.thresholds = gate, key, thresholds

    def __call__(self, hidden, state):
        need, required = self.gate.requirements(hidden)
        match = self.gate.relevance(hidden, self.key)
        present = torch.tensor([bool(tokens) for tokens in state.slots], device=hidden.device)
        return self.gate.permits(need, required, match, present, self.thresholds)


def stored_state(tokenizer, topic):
    return state_for(tokenizer, dict(zip(topic['fields'], topic['example'], strict=True)))


@torch.inference_mode()
def audit_recall(decoder, tokenizer, head, gate, keys, thresholds, topics, cases, before, *, history):
    result, trace = [], []
    scope = 'long' if history else 'state_only'
    for index, case in enumerate(cases):
        data = case['values']
        other = cases[index ^ 1]
        replacement = other['values']['new']
        if other['case'] != case['case'] or replacement in data.values():
            raise ValueError('Recall pair must supply one compatible replacement')
        meta = topics[case['case']]
        updated, keep = meta['fields']
        prompt = prompt_ids(tokenizer, case['history'] if history else [], case['question'])
        bind = BoundReadGate(gate, keys[meta['name']], thresholds)
        distractor = topics[meta['other']]
        wrong = stored_state(tokenizer, distractor)
        partial = state_for(tokenizer, {keep: data['keep']})
        off = generate(decoder, tokenizer, head, prompt, WorkingTurnState())
        wrong_answer = generate(decoder, tokenizer, head, prompt, wrong,
                                should_read=BoundReadGate(gate, keys[distractor['name']], thresholds))
        partial_answer = generate(decoder, tokenizer, head, prompt, partial, should_read=bind)
        outputs = {}
        for mode, current, foil in (('matched', data['new'], replacement), ('swapped', replacement, data['new'])):
            state = state_for(tokenizer, {updated: current, keep: data['keep']})
            row = generate(decoder, tokenizer, head, prompt, state, should_read=bind)
            text = row['answer']
            forbid = (data['old'], foil)
            outputs[mode] = {**row, 'new': current in text, 'keep': data['keep'] in text,
                             'foil': any(value in text for value in forbid),
                             'passed': grade(row, [re.escape(current), re.escape(data['keep'])],
                                             [re.escape(value) for value in forbid])}
        previous = before[index]
        if (case['case'], case['variant'], data) != (previous['case'], previous['variant'], previous['values']):
            raise ValueError('Prior recall test differs from this case')
        result.append({'case': case['case'], 'variant': case['variant'], 'values': data,
                       'replacement': replacement, 'outputs': {scope: outputs}})
        trace.append({'case': case['case'], 'variant': case['variant'], 'wrong_topic': wrong_answer,
                      'missing_field': partial_answer, 'empty': off,
                      'unchanged_wrong': wrong_answer == off, 'unchanged_missing': partial_answer == off,
                      'previous_routes': {name: row['route'] for name, row in previous['outputs'][scope].items()}})
    return result, trace


@torch.inference_mode()
def audit_ordinary(decoder, tokenizer, head, gate, keys, thresholds, topics, fixture, baseline):
    rows = []
    questions = [row['question'] for row in baseline] + fixture['held_ordinary']
    context = [topics['clinic_booking']] * len(baseline) + [
        topics[name] for name in ('film_notice', 'choir_program', 'clinic_booking',
                                  'repair_pickup', 'pottery_inventory', 'garden_supply')
    ]
    for index, (question, topic) in enumerate(zip(questions, context, strict=True)):
        state = (state_for(tokenizer, {'place': fresh_cases(FRESH)[0]['values']['keep']})
                 if index < len(baseline) else stored_state(tokenizer, topic))
        prompt = prompt_ids(tokenizer, [], question)
        plain = generate(decoder, tokenizer, head, prompt, WorkingTurnState())
        original = generate(decoder, tokenizer, head, prompt, state)
        new = generate(decoder, tokenizer, head, prompt, state,
                       should_read=BoundReadGate(gate, keys[topic['name']], thresholds))
        if index < len(baseline):
            row = baseline[index]
            if original != row['with_state'] or any(plain[k] != v for k, v in row['native'].items()):
                raise ValueError('Original ordinary route no longer reproduces the protected baseline')
        rows.append({'question': question, 'topic': topic['name'], 'original': original, 'gated': new,
                     'plain': plain, 'unchanged_original': original == plain, 'unchanged_gated': new == plain})
    return rows


@torch.inference_mode()
def audit_observation_keys(base, tokenizer, gate, topics, cases, fresh, thresholds):
    """Test keys encoded from observed text in place of authored topic summaries."""
    def prompt(text):
        return tuple(prompt_ids(tokenizer, [], text))

    foreign = {
        name: prompt(f'我先记下{row["transfer_key"]}，具体是{"、".join(row["example"])}。')
        for name, row in topics.items()
    }
    observations = {(row['case'], row['variant']): row['history'][0] for row in cases}
    checks = []
    for split, examples in (('older', cases), ('fresh', fresh)):
        for row in examples:
            topic = topics[row['case']]
            first = observations[row['case'], row['variant']]
            if first['role'] != 'user':
                raise ValueError('The first observation must be user text')
            checks.append((split, row, prompt(row['question']), prompt(first['content']),
                           foreign[topic['other']], prompt(topic_prompt(topic['transfer_key']))))
    features = encode_prompts(base, (tokens for _, _, *inputs in checks for tokens in inputs))
    rows = []
    gate = gate.cpu().eval().requires_grad_(False)
    for split, row, question, observed, foreign_key, authored in checks:
        topic = topics[row['case']]
        q = features[question]
        need, required = gate.requirements(q)
        present = vector(topic['fields'])
        records = {}
        for label, key in (('authored', authored), ('observed', observed), ('foreign_observed', foreign_key)):
            topic_score = gate.relevance(q, features[key])
            records[label] = {'read': bool(gate.permits(need, required, topic_score, present, thresholds)),
                              'topic_logit': float(topic_score)}
        wanted = [field for field, score in zip(FIELDS, required, strict=True) if score >= thresholds[1]]
        rows.append({'split': split, 'case': row['case'], 'variant': row['variant'], 'decisions': records,
                     'intent_logit': float(need), 'required': wanted, 'available': topic['fields']})
    totals = {split: {name: sum(row['decisions'][name]['read'] for row in rows if row['split'] == split)
                      for name in rows[0]['decisions']} for split in ('older', 'fresh')}
    return {'rows': rows, 'permitted': totals}


def main(args):
    start = perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    action_path = args.actions / 'actions.pt'
    reference_path = args.actions / 'result.json'
    model_paths = [args.checkpoint, DATA, SOURCE, FRESH, SCENARIOS, *(
        folder / file for folder in args.candidates for file in ('attention.pt', 'comparison.json'))]
    paths = [*model_paths, action_path, reference_path, DESIGN]
    protected = {path: sha(path) for path in paths}
    digests = {str(path): digest for path, digest in protected.items()}
    reference = read(reference_path)
    prior_sources = {Path(name).resolve(): digest for name, digest in reference['sources'].items()}
    if (any(prior_sources.get(path.resolve()) != protected[path] for path in model_paths)
            or reference['sha256'] != protected[action_path]):
        raise ValueError('Prior action weights or sources have changed')
    fixture = read(DESIGN)
    topics, training = training_data(fixture)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    fresh = fresh_cases(FRESH)
    source = {**read(SOURCE), 'recall': read(SCENARIOS)['recall']}
    long = [row for row in recall_prompts(tokenizer, source, extended=True) if row['length'] == 'extended']
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != reference['individual_sha256']:
        raise ValueError('The individual changed')
    replacements = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    head = SpanActionHead(base.config.hidden_size).to(base.device)
    head.load_state_dict(torch.load(action_path, map_location='cpu', weights_only=True))
    head.eval().requires_grad_(False)
    gate = StateReadGate(base.config.hidden_size)
    gate_path = args.output / 'gate.pt'
    if args.verify or args.check_origins:
        expected = read(args.output / 'result.json')
        if expected['sources'] != digests or expected['gate_sha256'] != sha(gate_path):
            raise ValueError('Reload source changed')
        saved = torch.load(gate_path, map_location='cpu', weights_only=True)
        gate.load_state_dict(saved['weights'])
        thresholds = saved['thresholds']
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        install(stack, replacements)
        if args.check_origins:
            origin = audit_observation_keys(base, tokenizer, gate, topics, long, fresh, thresholds)
            if (frozen_digest(runtime.organism, runtime.state) != individual
                    or any(sha(path) != digest for path, digest in protected.items())):
                raise ValueError('Origin check changed protected inputs')
            store_json(args.output / 'observation-keys.json', {**origin, 'gate_sha256': sha(gate_path),
                                                               'seconds': perf_counter() - start})
            print('Observation key routes:', origin['permitted'], flush=True)
            return
        keys = {name: prompt_ids(tokenizer, [], topic_prompt(row['transfer_key'])) for name, row in topics.items()}
        if args.verify:
            features = encode_prompts(base, keys.values())
        else:
            specs = calibration_specs(topics, fixture['cal_ordinary'])
            texts = [*training, *ORDINARY_TRAIN, *(topic_prompt(key) for row in topics.values() for key in row['train_keys']),
                     *(text for question, key, _, _ in specs for text in (question, topic_prompt(key)))]
            features = encode_prompts(base, (*keys.values(), *(prompt_ids(tokenizer, [], text) for text in texts)))
            fit(gate, features, tokenizer, training, topics)
            thresholds, calibration = calibrate(gate, features, tokenizer, specs)
            print('calibration', thresholds, calibration, flush=True)
            atomic_save({'weights': gate.state_dict(), 'thresholds': thresholds}, gate_path)
        transfer = {name: features[tuple(ids)].to(base.device) for name, ids in keys.items()}
        gate.to(base.device).eval().requires_grad_(False)
        results = {}
        for name, cases, is_long, baseline in (
            ('fresh', fresh, False, reference['conditions']['fresh']),
            ('long', long, True, reference['conditions']['reused']),
        ):
            rows, controls = audit_recall(decoder, tokenizer, head, gate, transfer, thresholds, topics, cases,
                                          baseline['rows'], history=is_long)
            metrics = measures(rows)
            results[name] = {'summary': metrics, 'rows': rows, 'controls': controls}
            print(name, metrics, 'blocked wrong', sum(row['unchanged_wrong'] for row in controls),
                  'blocked partial', sum(row['unchanged_missing'] for row in controls), flush=True)
        ordinary = audit_ordinary(decoder, tokenizer, head, gate, transfer, thresholds, topics, fixture,
                                  reference['ordinary'])
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    if (memory != reference['memory'] or frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('Protected input or durable-memory response changed')
    core = {'recall': results, 'ordinary': ordinary, 'memory': memory}
    if args.verify:
        if any(expected[key] != value for key, value in core.items()):
            raise ValueError('Independent evaluation changed a decision or reply')
        store_json(args.output / 'reload.json', {'gate_sha256': sha(gate_path), 'seconds': perf_counter() - start})
    else:
        store_json(args.output / 'result.json', {**core, 'sources': digests, 'gate_sha256': sha(gate_path),
                                                 'individual_sha256': individual, 'calibration': calibration,
                                                 'thresholds': thresholds, 'seconds': perf_counter() - start})
        print('ordinary unchanged', sum(row['unchanged_original'] for row in ordinary),
              sum(row['unchanged_gated'] for row in ordinary), 'of', len(ordinary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('actions', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify', action='store_true')
    mode.add_argument('--check-origins', action='store_true')
    args = parser.parse_args()
    if not (args.verify or args.check_origins) and args.output.exists():
        parser.error('Choose a new output directory')
    main(args)
