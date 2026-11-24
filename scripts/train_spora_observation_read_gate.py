"""Learn a working-state read gate from raw observations and paired availability."""

import argparse
from contextlib import ExitStack
from itertools import product
import math
from pathlib import Path
import random
import re
from time import perf_counter

from torch.nn import functional as F

from scripts.train_spora_span_actions import scored_cases
from scripts.train_spora_span_read_gate import (
    DATA, DESIGN, FRESH, HELD, MODEL, ORDINARY_TRAIN, SCENARIOS, SOURCE, AutoTokenizer, DynamicCache, FIELDS,
    SpanActionHead, WorkingTurnState, evaluate, extend_positions, fresh_cases, frozen_digest, generate, grade, install,
    load_full_streamed, load_integrated_runtime, load_recurrent_replacements, measures, prompt_ids, read, recall_prompts,
    records, sha, state_for, store_json, torch, training_data, vector,
)
from spora.language_training import atomic_save
from spora.research_span_read_gate import ObservationReadGate


OBSERVATIONS = DATA.with_name('working-state-observation-training-v1.json')
SEED = 26132
SEALED_VALUES = {
    'clinic_booking': [('周五傍晚六点', '周一清晨六点半', '雁回卫生站'),
                       ('周日上午十一点', '周二午后一点十五', '远棠医务室')],
    'repair_pickup': [('蓝角配件仓', '旧松修理部', '邵迦南'), ('灰窗服务台', '露石代领柜', '唐栈青')],
    'pottery_inventory': [('十五袋', '十九串', '卜时宁'), ('九匣', '三十二束', '程遇弦')],
    'choir_program': [('河坡的信', '雾起茶窗边', '陶江禾'), ('去看朝霞', '春泥记忆', '沈吟西')],
    'garden_supply': [('砾石袋', '护腕剪', '榆梧台阶旁'), ('种土勺', '弯头掘根叉', '黄杨篱拐弯处')],
    'film_notice': [('黑檀前厅', '灰鹭影厅内门', '蒋沐江'), ('银荻后廊', '沙屿电影厅二号口', '饶望溪')],
}


def sealed_cases(fixture):
    return [{'case': name, 'variant': variant, 'question': fixture[name]['sealed'], 'history': [],
             'values': dict(zip(('old', 'new', 'keep'), values, strict=True))}
            for name, pairs in SEALED_VALUES.items() for variant, values in enumerate(pairs)]


def training_samples(topics, fixture, vocabulary, rng):
    questions, observations = [], {}
    for name, base in topics.items():
        row = fixture[name]
        fields, terms = tuple(base['fields']), row['terms']
        for text in base['train']:
            questions.append((name, text, fields))
        for text in row['generic']:
            questions.append((name, text, ()))
        for field, term in zip(fields, terms, strict=True):
            for text in (f'这一项{row["label"]}只要确认{term}，查一下之前登记的内容。',
                         f'按这份{row["label"]}记录，只告诉我{term}。'):
                questions.append((name, text, (field,)))
        for a, b in (terms, terms[::-1]):
            for text in (f'给我{row["label"]}当时写下的两项，{a}和{b}都要说。',
                         f'我要转述我们目前的{row["label"]}安排：{a}，另外{b}？'):
                questions.append((name, text, fields))
        observations[name] = [
            row['obs'].format_map({field: rng.choice(vocabulary[field]) for field in fields}) for _ in range(3)
        ]
    questions.extend(('', text, ()) for text in ORDINARY_TRAIN)
    return questions, observations


@torch.inference_mode()
def encode(base, tokenizer, texts):
    """Pool the contextual and input embeddings of the raw words."""
    features = {}
    for text in dict.fromkeys(texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError('An observation or question cannot be empty')
        input_ids = torch.tensor([ids], device=base.device)
        cache = DynamicCache(config=base.config)
        hidden = base.model(input_ids=input_ids, past_key_values=cache, use_cache=True).last_hidden_state[0].float()
        words = base.get_input_embeddings()(input_ids[0]).float()
        views = (hidden.mean(dim=0), hidden[-1], words.mean(dim=0))
        features[text] = torch.cat([F.normalize(view, dim=0) for view in views]).cpu() * math.sqrt(hidden.shape[-1])
    return features


def observation(topics, fixture, name):
    row = topics[name]
    return fixture[name]['obs'].format_map(dict(zip(row['fields'], row['example'], strict=True)))


def calibration_rows(topics, fixture, ordinary):
    rows = []
    for name, row in topics.items():
        fact = observation(topics, fixture, name)
        fields = tuple(row['fields'])
        rows.extend([(row['cal'], fact, fields, 'relevant'),
                     (row['cal'], observation(topics, fixture, row['other']), fields, 'wrong_topic'),
                     *[(row['cal'], fact, (field,), 'partial') for field in fields],
                     (row['cal'], fact, (), 'empty')])
    rows.extend((question, observation(topics, fixture, name), tuple(topics[name]['fields']), 'ordinary')
                for name, question in zip(topics, ordinary))
    return rows


def fit(gate, features, samples, observations, topics):
    questions = torch.stack([features[q] for _, q, _ in samples])
    requested = torch.stack([vector(fields) for _, _, fields in samples])
    intent = requested.any(dim=-1).float()
    all_topics = list(topics)
    keys = torch.stack([features[raw] for name in all_topics for raw in observations[name]])
    positive = [i for i, row in enumerate(samples) if row[2]]
    rng = random.Random(SEED)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=.0025, weight_decay=.02)
    cover_rows = [(index, vector(fields), 1.) for index in positive for _, _, fields in [samples[index]]]
    for index in positive:
        names = samples[index][2]
        cover_rows.extend((index, vector(set(names) - {field}), 0.) for field in names)
        cover_rows.append((index, vector({*names, rng.choice(FIELDS)}), 1.))
    cid = torch.tensor([row[0] for row in cover_rows])
    masks = torch.stack([row[1] for row in cover_rows])
    coverage_labels = torch.tensor([row[2] for row in cover_rows])
    for step in range(220):
        need, wanted = gate.requirements(questions)
        subset = rng.sample(positive, 32)
        pos, wrong, hard = [], [], []
        for i in subset:
            name = samples[i][0]
            pos.append(3 * all_topics.index(name) + rng.randrange(3))
            negative = rng.choice([other for other in all_topics if other != name])
            wrong.append(3 * all_topics.index(negative) + rng.randrange(3))
            hard.append(3 * all_topics.index(topics[name]['other']) + rng.randrange(3))
        topic = gate.relevance(questions[subset * 3], keys[pos + wrong + hard])
        a, b, c = topic.chunk(3)
        loss = (F.binary_cross_entropy_with_logits(need, intent)
                + F.binary_cross_entropy_with_logits(wanted, requested.float())
                + (F.softplus(-a).mean() + F.softplus(b).mean() + F.softplus(c).mean()) / 3
                + F.binary_cross_entropy_with_logits(gate.coverage(wanted[cid], masks), coverage_labels))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 55 == 0:
            print('observation route step', step + 1, float(loss.detach()), flush=True)


@torch.inference_mode()
def calibrate(gate, features, rows):
    questions = torch.stack([features[row[0]] for row in rows])
    keys = torch.stack([features[row[1]] for row in rows])
    masks = torch.stack([vector(row[2]) for row in rows])
    need, required = gate.requirements(questions)
    relevant = gate.relevance(questions, keys)
    groups = dict.fromkeys(row[3] for row in rows)
    masks_by_kind = {group: torch.tensor([row[3] == group for row in rows]) for group in groups}
    best = None
    for thresholds in product((-3., -1.5, 0., 1.5, 3.), repeat=3):
        read_decision = gate.allow(need, required, relevant, masks, thresholds)
        correct = read_decision == masks_by_kind['relevant']
        rates = {name: float(correct[chosen].float().mean()) for name, chosen in masks_by_kind.items()}
        rank = (min(rates.values()), sum(rates.values()), -sum(abs(v) for v in thresholds))
        if best is None or rank > best[0]:
            best = rank, thresholds, rates
    return best[1], best[2]


@torch.inference_mode()
def decide(gate, features, question, raw, fields, thresholds):
    q = features[question]
    need, required = gate.requirements(q)
    match = gate.relevance(q, features[raw])
    present = vector(fields)
    return {'read': bool(gate.allow(need, required, match, present, thresholds)),
            'scores': [float(need), float(match), float(gate.coverage(required, present))]}


def case_observation(row, fixture):
    if row['history']:
        first = row['history'][0]
        if first['role'] != 'user':
            raise ValueError('Observed history must begin with the user')
        return first['content']
    return fixture[row['case']]['first'].format_map(row['values'])


def decode_case(decoder, tokenizer, head, prompt, state, decision):
    return generate(decoder, tokenizer, head, prompt, state, should_read=lambda _hidden, _state: decision)


def audit_recall(decoder, tokenizer, head, gate, features, cases, topics, fixture, thresholds, *, long=False):
    rows, controls = [], []
    scope = 'long' if long else 'state_only'
    for i, case in enumerate(cases):
        name, question, values = case['case'], case['question'], case['values']
        updated, retained = fields = topics[name]['fields']
        prompt = prompt_ids(tokenizer, case['history'] if long else [], question)
        raw = case_observation(case, fixture)
        other = topics[name]['other']
        foreign = observation(topics, fixture, other)
        related = decide(gate, features, question, raw, fields, thresholds)
        negatives = {}
        for label, key, available in (('wrong_topic', foreign, fields), ('missing_update', raw, (retained,)),
                                      ('missing_retained', raw, (updated,)), ('empty', raw, ())):
            negatives[label] = decide(gate, features, question, key, available, thresholds)
        if long:
            foreign_fields = dict(zip(fields, topics[other]['example'], strict=True))
            missing = {retained: values['keep']}
            normal = generate(decoder, tokenizer, head, prompt, WorkingTurnState())
            for label, data in (('wrong_topic', foreign_fields), ('missing_update', missing)):
                rival = decode_case(decoder, tokenizer, head, prompt, state_for(tokenizer, data), negatives[label]['read'])
                if rival['route'] != negatives[label]['read'] or (not rival['route'] and rival != normal):
                    raise ValueError('A declined state did not reproduce the plain decoder')
        replacement = cases[i ^ 1]['values']['new']
        if cases[i ^ 1]['case'] != name or replacement in values.values():
            raise ValueError('Expected one changed slot for each paired question')
        outputs = {}
        for label, target, conflict in (('matched', values['new'], replacement),
                                        ('swapped', replacement, values['new'])):
            state = state_for(tokenizer, {updated: target, retained: values['keep']})
            result = decode_case(decoder, tokenizer, head, prompt, state, related['read'])
            raw_answer = result['answer']
            forbidden = (values['old'], conflict)
            outputs[label] = {**result, 'new': target in raw_answer, 'keep': values['keep'] in raw_answer,
                              'foil': any(x in raw_answer for x in forbidden),
                              'passed': grade(result, [re.escape(target), re.escape(values['keep'])],
                                              [re.escape(x) for x in forbidden])}
        rows.append({'case': name, 'variant': case['variant'], 'values': values, 'replacement': replacement,
                     'outputs': {scope: outputs}})
        controls.append({'case': name, 'variant': case['variant'], 'related': related, **negatives})
    metrics = measures(rows)
    rates = {name: sum(row[name]['read'] for row in controls)
             for name in ('related', 'wrong_topic', 'missing_update', 'missing_retained', 'empty')}
    return {'rows': rows, 'controls': controls, 'summary': metrics, 'read_counts': rates}


def audit_ordinary(decoder, tokenizer, head, gate, features, questions, topics, fixture, thresholds, baseline):
    rows = []
    for i, (q, name) in enumerate(questions):
        topic = topics[name]
        raw = observation(topics, fixture, name)
        values = dict(zip(topic['fields'], topic['example'], strict=True))
        if i < len(baseline):
            values = {'place': baseline[i]['place']}
        prompt = prompt_ids(tokenizer, [], q)
        route = decide(gate, features, q, raw, tuple(values), thresholds)
        answer = decode_case(decoder, tokenizer, head, prompt, state_for(tokenizer, values), route['read'])
        plain = generate(decoder, tokenizer, head, prompt, WorkingTurnState())
        rows.append({'question': q, 'topic': name, 'decision': route, 'answer': answer, 'plain': plain,
                     'unchanged': answer == plain})
    return rows


def replay_comparisons(decoder, tokenizer, head, sealed, controls, topics):
    """Expose effects of missed partial-state rejections and the unguarded action head."""
    original = scored_cases(decoder, tokenizer, head, sealed, ('state_only',))
    rows = []
    for case, control in zip(sealed, controls, strict=True):
        updated, retained = topics[case['case']]['fields']
        prompt = prompt_ids(tokenizer, [], case['question'])
        for label, field, value in (('missing_update', retained, case['values']['keep']),
                                    ('missing_retained', updated, case['values']['new'])):
            if control[label]['read']:
                result = decode_case(decoder, tokenizer, head, prompt, state_for(tokenizer, {field: value}), True)
                rows.append({'case': case['case'], 'variant': case['variant'], 'excluded': label, 'answer': result})
    return {'original_sealed': {'summary': measures(original), 'rows': original}, 'partial_replies': rows}


def main(args):
    started = perf_counter()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    action_path = args.actions / 'actions.pt'
    reference_path = args.actions / 'result.json'
    model_sources = (args.checkpoint, DATA, SOURCE, SCENARIOS, FRESH, *(
        folder / file for folder in args.candidates for file in ('attention.pt', 'comparison.json')))
    paths = (*model_sources, action_path, reference_path, DESIGN, OBSERVATIONS)
    protected = {path: sha(path) for path in paths}
    baseline = read(reference_path)
    sources = {Path(path).resolve(): digest for path, digest in baseline['sources'].items()}
    if (any(sources.get(path.resolve()) != protected[path] for path in model_sources)
            or baseline['sha256'] != protected[action_path]):
        raise ValueError('The action baseline uses another checkpoint or sources')
    topic_source = read(DESIGN)
    topics, _ = training_data(topic_source)
    fixture = read(OBSERVATIONS)['topics']
    if fixture.keys() != topics.keys():
        raise ValueError('Observation fixture and slot schemas differ')
    samples, observations = training_samples(topics, fixture, read(DATA)['values']['train'], random.Random(SEED))
    cal = calibration_rows(topics, fixture, topic_source['cal_ordinary'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    fresh = fresh_cases(FRESH)
    sealed = sealed_cases(fixture)
    source = {**read(SOURCE), 'recall': read(SCENARIOS)['recall']}
    long = [row for row in recall_prompts(tokenizer, source, extended=True) if row['length'] == 'extended']
    old_ordinary = [{'question': row['question'], 'place': fresh[0]['values']['keep']} for row in baseline['ordinary']]
    ordinary = [(row['question'], 'clinic_booking') for row in old_ordinary] + list(zip(
        topic_source['held_ordinary'], ('film_notice', 'choir_program', 'clinic_booking', 'repair_pickup',
                                         'pottery_inventory', 'garden_supply'), strict=True))
    train_questions = {row[1] for row in samples}
    all_test = {row['question'] for row in (*fresh, *sealed, *long)} | {q for q, _ in ordinary}
    calibration_questions = {row[0] for row in cal}
    if train_questions & (calibration_questions | all_test) or calibration_questions & all_test:
        raise ValueError('Training, calibration, and evaluation questions overlap')
    train_text = train_questions | {raw for key in observations.values() for raw in key}
    cal_text = {text for row in cal for text in row[:2]}
    all_text = all_test | {case_observation(row, fixture) for row in (*fresh, *sealed, *long)} | {
        observation(topics, fixture, name) for name in topics}
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != baseline['individual_sha256']:
        raise ValueError('The individual changed')
    candidates = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    head = SpanActionHead(base.config.hidden_size).to(base.device)
    head.load_state_dict(torch.load(action_path, map_location='cpu', weights_only=True))
    head.eval().requires_grad_(False)
    gate = ObservationReadGate(3 * base.config.hidden_size)
    weights = args.output / 'gate.pt'
    source_digests = {str(path): digest for path, digest in protected.items()}
    if args.verify:
        archived = read(args.output / 'result.json')
        if (archived['sources'] != source_digests or archived['gate_sha256'] != sha(weights)
                or archived['individual_sha256'] != individual):
            raise ValueError('The observation gate or evaluation sources changed')
        saved = torch.load(weights, map_location='cpu', weights_only=True)
        gate.load_state_dict(saved['weights'])
        thresholds = saved['thresholds']
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        install(stack, candidates)
        texts = sorted(all_text if args.verify else train_text | cal_text | all_text)
        print('Encoding distinct text inputs:', len(texts), flush=True)
        features = encode(base, tokenizer, texts)
        if not args.verify:
            fit(gate, features, samples, observations, topics)
            thresholds, calibration = calibrate(gate, features, cal)
            print('Calibration:', thresholds, calibration, flush=True)
            atomic_save({'weights': gate.state_dict(), 'thresholds': thresholds}, weights)
        gate.eval().requires_grad_(False)
        results = {}
        for name, cases in (('fresh', fresh), ('sealed', sealed), ('long', long)):
            result = audit_recall(decoder, tokenizer, head, gate, features, cases, topics, fixture, thresholds,
                                  long=name == 'long')
            results[name] = result
            print(name, result['read_counts'], result['summary'], flush=True)
        ordinary_results = audit_ordinary(decoder, tokenizer, head, gate, features, ordinary, topics, fixture,
                                          thresholds, old_ordinary)
        print('Ordinary unchanged:', sum(row['unchanged'] for row in ordinary_results), 'of', len(ordinary_results),
              flush=True)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
        if args.verify:
            comparison = replay_comparisons(decoder, tokenizer, head, sealed, results['sealed']['controls'], topics)
    if (memory != baseline['memory'] or frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('A protected input or persistent memory response changed')
    core = {'recall': results, 'ordinary': ordinary_results, 'memory': memory}
    elapsed = perf_counter() - started
    if args.verify:
        if any(archived[name] != value for name, value in core.items()):
            raise ValueError('Reload changed a gate score or generated answer')
        store_json(args.output / 'reload.json', {'gate_sha256': sha(weights), 'seconds': elapsed,
                                                 'sources': source_digests, **comparison})
        print('Original sealed:', comparison['original_sealed']['summary'], flush=True)
    else:
        store_json(args.output / 'result.json', {**core, 'sources': source_digests, 'gate_sha256': sha(weights),
                                                 'individual_sha256': individual, 'seconds': elapsed,
                                                 'thresholds': thresholds, 'calibration': calibration,
                                                 'training_questions': len(samples),
                                                 'observations': sum(map(len, observations.values()))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('actions', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', type=Path, nargs=2)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not args.verify and args.output.exists():
        parser.error('Choose a new output directory')
    main(args)
