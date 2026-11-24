"""Train field permissions and missing-value replies on the frozen SPORA backbone."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import random
import re
from time import perf_counter

from torch.nn import functional as F

from scripts.train_spora_span_actions import backbone_step, capture, tokens
from scripts.train_spora_observation_read_gate import (
    DATA, DESIGN, FRESH, HELD, MODEL, OBSERVATIONS, SCENARIOS, SOURCE, AutoTokenizer, DynamicCache, FIELDS,
    SpanActionHead, WorkingTurnState, case_observation, encode, evaluate, extend_positions, fresh_cases, frozen_digest,
    generate, grade, install, load_full_streamed, load_integrated_runtime, load_recurrent_replacements, measures,
    observation, prompt_ids, read, records, sealed_cases, sha, state_for, store_json, torch, training_data, training_samples,
)
from spora.language_training import atomic_save
from spora.research_field_access import (
    EXCLUDE, IGNORE, OPTIONAL, REQUIRE, FieldAccessHead, FieldAccessPlan, MissingFieldActions,
    partial_masks, partial_targets,
)
from spora.research_span_action import emit_action
from spora.research_span_read_gate import ObservationReadGate


SEED = 26142
NAMES = ('日期', '时间', '地点', '所需物品', '人员姓名', '数量', '方向')
CONTROL = (
    ('clinic_booking', '给家里发消息就只写我什么时候去复查，门诊是哪家先别说。', 'time', None, 'place'),
    ('repair_pickup', '给代领群只报领回维修件的柜台，领取者是谁请隐去。', 'place', None, 'person'),
    ('pottery_inventory', '只确认这批货实际留多少。收货者名字不要出现在回复里。', 'quantity', None, 'person'),
    ('choir_program', '节目单先核对歌曲标题，领唱姓名请别放进这条回复。', 'gear', None, 'person'),
    ('garden_supply', '我手里的短讯只缺花圃会合位置，随身工具那栏暂时不要写。', 'place', None, 'gear'),
    ('film_notice', '请只补入场走哪个门，主持是谁要留空。', 'place', None, 'person'),
    ('clinic_booking', '先告诉我这次复查去的门诊。如果已经定了钟点，也可以一并写。', 'place', 'time', None),
    ('repair_pickup', '先确认取修理件的人。柜台这项有登记的话也补在后面。', 'person', 'place', None),
    ('pottery_inventory', '报一下签收者名字，若库房已经点清总量，也可附上。', 'person', 'quantity', None),
    ('choir_program', '合唱单的歌名一定要给我，领唱那栏如果有资料也可以带上。', 'gear', 'person', None),
    ('garden_supply', '告诉晚到的人花圃在哪里碰面。随身物件清单有记录就一起写。', 'place', 'gear', None),
    ('film_notice', '主持人写上去。入口已经定下才需要加进这条消息。', 'person', 'place', None),
)
ORDINARY = (
    ('choir_program', '设计一份不提具体姓名的乐团招募启事。'),
    ('clinic_booking', '用通用话术写条就医提醒，别说去哪家门诊或几点。'),
    ('repair_pickup', '写一句售后服务评价，请避免提及具体取件者。'),
    ('pottery_inventory', '如何用简洁句子提醒仓管核货？不要代填单上的数量。'),
    ('garden_supply', '写一条园艺活动欢迎语，别代入之前留下的集合地点。'),
    ('film_notice', '起草一条影院秩序提示，不写本场放映的主持人姓名。'),
)


def labels(required=(), optional=(), excluded=()):
    roles = tuple(REQUIRE if field in required else OPTIONAL if field in optional else
                  EXCLUDE if field in excluded else IGNORE for field in FIELDS)
    return FieldAccessPlan(roles)


def training_roles(topics, fixture, samples):
    rows = {text: labels(required=fields) for _, text, fields in samples}
    for name, meta in topics.items():
        a, b = meta['fields']
        spec = fixture[name]
        topic = spec['label']
        for first, second, requested, extra in ((a, b, *spec['terms']), (b, a, *spec['terms'][::-1])):
            label = NAMES[FIELDS.index(second)]
            for form in (
                    f'按{topic}记事，给我{requested}即可，别透露{label}。',
                    f'说{requested}。这条{topic}回复里不要带{extra}。'):
                rows[form] = labels((first,), excluded=(second,))
            for form in (
                    f'我只要求明确{requested}。{label}有记录才附在{topic}通知上。',
                    f'{topic}这条记录先报{requested}，能找到{extra}也可以补充。'):
                rows[form] = labels((first,), (second,))
            rows[f'给{topic}写一段通用文案，不提具体{label}，其他内容取一般建议。'] = labels(excluded=(second,))
    return rows


def fit_roles(head, features, references, external):
    texts = list(references)
    data = torch.stack([features[text] for text in texts])
    target = torch.tensor([references[text].roles for text in texts])
    weights = torch.tensor([.40, 1., 1.25, 1.5])
    aux = torch.stack([features[row['text']] for row in external])
    aux_targets = torch.tensor([row['required'] for row in external], dtype=torch.long)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002, weight_decay=.025)
    rng = random.Random(SEED)
    place = FIELDS.index('place')
    for step in range(200):
        batch = rng.sample(range(len(texts)), 56)
        extra = rng.sample(range(len(external)), 24)
        logits = head(data[batch])
        aux_logits = head(aux[extra])[:, place]
        # CrossWOZ supervises address requests only; other role labels are left unspecified.
        address = aux_logits[:, REQUIRE] - torch.logsumexp(aux_logits[:, [IGNORE, OPTIONAL, EXCLUDE]], dim=-1)
        loss = (F.cross_entropy(logits.flatten(0, 1), target[batch].flatten(), weight=weights)
                + .18 * F.binary_cross_entropy_with_logits(address, aux_targets[extra].float()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            print('Access role step', step + 1, round(float(loss.detach()), 5), flush=True)


def markers_for(tokenizer):
    return [tuple(tokens(tokenizer, f'{name}尚未记录')) for name in NAMES]


def partial_plans(tokenizer, topics, vocabulary, markers, mode_vocab):
    rng = random.Random(SEED)
    out = []
    for meta in topics.values():
        fields = meta['fields']
        for present in ((fields[0],), (fields[1],), ()):
            for index in range(3):
                state = state_for(tokenizer, {field: rng.choice(vocabulary[field]) for field in present})
                missing = tuple(FIELDS.index(field) for field in fields if field not in present)
                q = meta['train'][index % len(meta['train'])]
                segments = []
                for field in present:
                    address = FIELDS.index(field)
                    segments.append((None, tokens(tokenizer, NAMES[address] + '为')))
                    segments.append((address, state.slots[address]))
                    segments.append((None, tokens(tokenizer, '。')))
                for address in missing:
                    segments.append((len(FIELDS) + address, markers[address]))
                    segments.append((None, tokens(tokenizer, '。')))
                token_ids, targets = partial_targets(segments, mode_vocab, tokenizer.eos_token_id)
                out.append((prompt_ids(tokenizer, [], q), token_ids, targets, state, missing))
    return out


def fit_partial(decoder, adapter, head, examples):
    base = decoder.base
    vocab = base.config.vocab_size
    with torch.no_grad():
        requests = [(prompt + answer, len(answer) + 1) for prompt, answer, _, _, _ in examples]
        raw = capture(decoder, base.model.layers[-1], requests)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=8e-4)
    rng = random.Random(SEED)
    for step in range(384):
        i = rng.randrange(len(examples))
        _, _, targets, state, missing = examples[i]
        positions = torch.tensor([target.offset for target in targets])
        hidden = base.model.norm(raw[i][positions].to(base.device))
        keys = head.field_keys(state, base.get_input_embeddings())
        present, absent = partial_masks(state, missing, targets, vocab, base.device)
        target = torch.tensor([action.action for action in targets], device=base.device)
        distribution = adapter(hidden, keys, present, absent, head, base.lm_head)
        chosen = distribution.gather(-1, target[:, None]).squeeze(-1)
        weights = torch.where(target >= vocab, 2., 1.)
        weights[-1] = 2.
        loss = -(chosen * weights).sum() / weights.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 96 == 0:
            print('Missing-action step', step + 1, round(float(loss.detach()), 5), flush=True)


@torch.inference_mode()
def partial_generate(decoder, tokenizer, head, adapter, prompt, state, missing, markers, limit=48, *,
                     required=(), coverage=None):
    base = decoder.base
    vocab = base.config.vocab_size
    cache = DynamicCache(config=base.config)
    hidden = backbone_step(base, prompt, cache)
    keys = head.field_keys(state, base.get_input_embeddings())
    remaining = {i for i, value in enumerate(state.slots) if value}
    unseen = set(missing)
    surface, reads, notices, ended = [], [], [], False
    while len(surface) < limit:
        budget = limit - len(surface)
        present = torch.tensor([i in remaining and len(value) <= budget for i, value in enumerate(state.slots)],
                               device=base.device)
        absent = torch.tensor([i in unseen and len(marker) <= budget for i, marker in enumerate(markers)],
                              device=base.device)
        scores = adapter(hidden, keys, present, absent, head, base.lm_head)
        if coverage is not None:
            owed = bool(remaining.intersection(required))
            if owed or unseen:
                scores[tokenizer.eos_token_id] = -torch.inf
            if owed and coverage == 'known_first':
                scores[vocab + len(FIELDS):] = -torch.inf
        action = int(scores.argmax())
        if action == tokenizer.eos_token_id:
            ended = True
            break
        if action >= vocab + len(FIELDS):
            address = action - vocab - len(FIELDS)
            unseen.remove(address)
            chunk = markers[address]
            notices.append(FIELDS[address])
        else:
            chunk = emit_action(action, state, remaining, budget, vocab)
            if action >= vocab:
                reads.append(FIELDS[action - vocab])
        surface.extend(chunk)
        if len(surface) < limit:
            hidden = backbone_step(base, chunk, cache)
    return {'answer': tokenizer.decode(surface, skip_special_tokens=True).strip(), 'eos': ended,
            'reads': reads, 'reported_missing': notices, 'route': True}


def decision(role_head, previous, features, question, raw, thresholds):
    hidden = features[question]
    plan = FieldAccessPlan(tuple(role_head(hidden).argmax(dim=-1).tolist()))
    intent, _ = previous.requirements(hidden)
    match = previous.relevance(hidden, features[raw])
    enable = bool(plan.requested and intent >= thresholds[0] and match >= thresholds[1])
    return plan, enable


def answer(decoder, tokenizer, head, adapter, markers, prompt, state, plan, enabled, *, coverage=None):
    if not enabled:
        return generate(decoder, tokenizer, head, prompt, WorkingTurnState())
    picked = plan.project(state)
    missing = plan.missing(picked)
    if missing:
        required = tuple(i for i, role in enumerate(plan.roles) if role == REQUIRE)
        return partial_generate(decoder, tokenizer, head, adapter, prompt, picked, missing, markers,
                                required=required, coverage=coverage)
    return generate(decoder, tokenizer, head, prompt, picked, should_read=lambda _hidden, _state: True)


def partial_cases(decoder, tokenizer, head, adapter, markers, prompt, case, fields, plan, enabled, *, coverage=None):
    a, b = fields
    values = case['values']
    results = []
    for missing, present, value, absent_value in ((a, b, values['keep'], values['new']),
                                                  (b, a, values['new'], values['keep'])):
        state = state_for(tokenizer, {present: value})
        reply = answer(decoder, tokenizer, head, adapter, markers, prompt, state, plan, enabled, coverage=coverage)
        correct = (reply.get('reported_missing', []) == [missing] and value in reply['answer']
                   and absent_value not in reply['answer'] and values['old'] not in reply['answer'] and reply['eos'])
        results.append({'case': case['case'], 'variant': case['variant'], 'expected_missing': missing,
                        'answer': reply, 'passed': correct})
    return results


@torch.inference_mode()
def audit_recall(decoder, tokenizer, head, adapter, roles, gate, features, markers, cases, topics, fixture, thresholds):
    rows, partial = [], []
    for i, case in enumerate(cases):
        name, q, values = case['case'], case['question'], case['values']
        a, b = topics[name]['fields']
        raw = case_observation(case, fixture)
        plan, enabled = decision(roles, gate, features, q, raw, thresholds)
        prompt = prompt_ids(tokenizer, [], q)
        replacement = cases[i ^ 1]['values']['new']
        if cases[i ^ 1]['case'] != name or replacement in values.values():
            raise ValueError('Expected two distinct observed-state arms')
        outputs = {}
        for mode, new, foil in (('matched', values['new'], replacement), ('swapped', replacement, values['new'])):
            state = state_for(tokenizer, {a: new, b: values['keep']})
            reply = answer(decoder, tokenizer, head, adapter, markers, prompt, state, plan, enabled)
            old = (values['old'], foil)
            text = reply['answer']
            outputs[mode] = {**reply, 'new': new in text, 'keep': values['keep'] in text,
                             'foil': any(x in text for x in old),
                             'passed': grade(reply, [re.escape(new), re.escape(values['keep'])],
                                             [re.escape(x) for x in old])}
        rows.append({'case': name, 'variant': case['variant'], 'values': values, 'replacement': replacement,
                     'outputs': {'state_only': outputs}, 'roles': list(plan.roles)})
        partial.extend(partial_cases(decoder, tokenizer, head, adapter, markers, prompt, case, (a, b), plan, enabled))
    return {'summary': measures(rows), 'rows': rows, 'partial': partial}


@torch.inference_mode()
def audit_controls(decoder, tokenizer, head, adapter, roles, gate, features, markers, topics, fixture, thresholds):
    results = []
    for name, q, required, optional, exclude in CONTROL:
        raw = observation(topics, fixture, name)
        prompt = prompt_ids(tokenizer, [], q)
        plan, enabled = decision(roles, gate, features, q, raw, thresholds)
        expected = labels((required,), (optional,), (exclude,))
        values = dict(zip(topics[name]['fields'], topics[name]['example'], strict=True))
        variants = [('full', values)]
        if optional:
            variants.append(('optional_absent', {required: values[required]}))
        outputs = {}
        for label, available in variants:
            reply = answer(decoder, tokenizer, head, adapter, markers, prompt, state_for(tokenizer, available),
                           plan, enabled)
            outputs[label] = {**reply, 'required_present': values[required] in reply['answer'],
                              'excluded_leaked': bool(exclude and values[exclude] in reply['answer']),
                              'optionally_reported': bool(optional and values[optional] in reply['answer'])}
        results.append({'case': name, 'question': q, 'expected': list(expected.roles), 'actual': list(plan.roles),
                        'outputs': outputs})
    ordinary = []
    for name, q in ORDINARY:
        raw = observation(topics, fixture, name)
        values = dict(zip(topics[name]['fields'], topics[name]['example'], strict=True))
        prompt = prompt_ids(tokenizer, [], q)
        plan, enabled = decision(roles, gate, features, q, raw, thresholds)
        reply = answer(decoder, tokenizer, head, adapter, markers, prompt, state_for(tokenizer, values), plan, enabled)
        baseline = generate(decoder, tokenizer, head, prompt, WorkingTurnState())
        ordinary.append({'question': q, 'same_answer': reply == baseline, 'roles': list(plan.roles),
                         'reply': reply, 'baseline': baseline})
    return {'field_controls': results, 'ordinary': ordinary}


@torch.inference_mode()
def oracle_partial(decoder, tokenizer, head, adapter, markers, cases, topics, *, coverage=None):
    """Test missing-value generation with the annotated fields fixed."""
    proof, empty = [], []
    for case in cases:
        topic = topics[case['case']]
        a, b = topic['fields']
        plan = labels((a, b))
        values = case['values']
        prompt = prompt_ids(tokenizer, [], case['question'])
        proof.extend(partial_cases(decoder, tokenizer, head, adapter, markers, prompt, case, (a, b), plan, True,
                                   coverage=coverage))
        if coverage is None and not case['variant']:
            response = answer(decoder, tokenizer, head, adapter, markers, prompt, WorkingTurnState(), plan, True)
            passed = (set(response.get('reported_missing', [])) == {a, b} and response['eos']
                      and not any(value in response['answer'] for value in values.values()) and not response['reads'])
            empty.append({'case': case['case'], 'passed': passed, 'answer': response})
    return proof, empty


@torch.inference_mode()
def learned_coverage(decoder, tokenizer, head, adapter, roles, gate, features, markers, cases, topics, fixture,
                     thresholds):
    rows = []
    for case in cases:
        name, question = case['case'], case['question']
        plan, enabled = decision(roles, gate, features, question, case_observation(case, fixture), thresholds)
        prompt = prompt_ids(tokenizer, [], question)
        rows.extend(partial_cases(decoder, tokenizer, head, adapter, markers, prompt, case, topics[name]['fields'],
                                  plan, enabled, coverage='eos_only'))
    return rows


def main(args):
    started = perf_counter()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    action_weights = args.actions / 'actions.pt'
    route_weights = args.observations / 'gate.pt'
    action_archive = args.actions / 'result.json'
    route_archive = args.observations / 'result.json'
    models = (args.checkpoint, DATA, SOURCE, SCENARIOS, FRESH, *(
        folder / file for folder in args.layers for file in ('attention.pt', 'comparison.json')))
    protected = {path: sha(path) for path in (*models, action_weights, route_weights, action_archive,
                                             route_archive, DESIGN, OBSERVATIONS, args.crosswoz)}
    reference, prior = read(action_archive), read(route_archive)
    action_sources = {Path(path).resolve(): value for path, value in reference['sources'].items()}
    if (any(action_sources.get(path.resolve()) != protected[path] for path in models)
            or reference['sha256'] != protected[action_weights] or prior['gate_sha256'] != protected[route_weights]):
        raise ValueError('The frozen candidate differs from the verified sources')
    rng = random.Random(SEED)
    topic_source = read(DESIGN)
    topics, _ = training_data(topic_source)
    fixture = read(OBSERVATIONS)['topics']
    values = read(DATA)['values']['train']
    sample, _ = training_samples(topics, fixture, values, rng)
    access = training_roles(topics, fixture, sample)
    external = read(args.crosswoz)['rows']
    cases = fresh_cases(FRESH) + sealed_cases(fixture)
    held_text = {row['question'] for row in cases} | {row[1] for row in CONTROL + ORDINARY}
    training = set(access) | {row['text'] for row in external}
    if training & held_text:
        raise ValueError('Field role evaluation overlaps training')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != prior['individual_sha256'] or individual != reference['individual_sha256']:
        raise ValueError('The individual changed')
    layers = load_recurrent_replacements(decoder, args.layers, protected[args.checkpoint])
    head = SpanActionHead(base.config.hidden_size).to(base.device)
    head.load_state_dict(torch.load(action_weights, map_location='cpu', weights_only=True))
    head.eval().requires_grad_(False)
    saved = torch.load(route_weights, map_location='cpu', weights_only=True)
    gate = ObservationReadGate(3 * base.config.hidden_size)
    gate.load_state_dict(saved['weights'])
    gate.eval().requires_grad_(False)
    thresholds = saved['thresholds']
    roles = FieldAccessHead(3 * base.config.hidden_size)
    partial = MissingFieldActions(base.config.hidden_size).to(base.device)
    markers = markers_for(tokenizer)
    prompts = held_text | {observation(topics, fixture, name) for name in topics} | {
        case_observation(row, fixture) for row in cases}
    sources = {str(path): digest for path, digest in protected.items()}
    weights = args.output / 'field-access.pt'
    if args.verify or args.audit_coverage or args.learned_coverage:
        expected = read(args.output / 'result.json')
        if (expected['sources'] != sources or expected['individual_sha256'] != individual
                or expected['sha256'] != sha(weights)):
            raise ValueError('An independent replay input changed')
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        roles.load_state_dict(checkpoint['roles'])
        partial.load_state_dict(checkpoint['missing'])
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        install(stack, layers)
        if args.learned_coverage:
            roles.eval().requires_grad_(False)
            partial.eval().requires_grad_(False)
            features = encode(base, tokenizer, sorted(prompts))
            readout = {}
            for name, held in (('fresh', fresh_cases(FRESH)), ('sealed', sealed_cases(fixture))):
                rows = learned_coverage(decoder, tokenizer, head, partial, roles, gate, features, markers, held, topics,
                                        fixture, thresholds)
                readout[name] = rows
                print(name, 'learned coverage', sum(row['passed'] for row in rows), '/', len(rows), flush=True)
            if (frozen_digest(runtime.organism, runtime.state) != individual
                    or any(sha(path) != digest for path, digest in protected.items())):
                raise ValueError('The learned coverage diagnostic changed a frozen source')
            store_json(args.output / 'learned-coverage.json', {'sha256': sha(weights), 'results': readout,
                                                               'seconds': perf_counter() - started})
            return
        if args.audit_coverage:
            roles.eval().requires_grad_(False)
            partial.eval().requires_grad_(False)
            audited = {}
            for mode in ('eos_only', 'known_first'):
                records_, _ = oracle_partial(decoder, tokenizer, head, partial, markers, sealed_cases(fixture),
                                             topics, coverage=mode)
                audited[mode] = records_
                print(mode, sum(row['passed'] for row in records_), '/', len(records_), flush=True)
            if (frozen_digest(runtime.organism, runtime.state) != individual
                    or any(sha(path) != digest for path, digest in protected.items())):
                raise ValueError('The coverage diagnostic changed a frozen source')
            store_json(args.output / 'coverage-diagnostic.json', {'sha256': sha(weights), 'arms': audited,
                                                                  'seconds': perf_counter() - started})
            return
        inputs = prompts if args.verify else training | prompts
        print('Encoding access training and held-out inputs', len(inputs), flush=True)
        features = encode(base, tokenizer, sorted(inputs))
        if not args.verify:
            fit_roles(roles, features, access, external)
            examples = partial_plans(tokenizer, topics, values, markers, base.config.vocab_size)
            fit_partial(decoder, partial, head, examples)
            atomic_save({'roles': roles.state_dict(),
                         'missing': {key: val.cpu() for key, val in partial.state_dict().items()}}, weights)
        roles.eval().requires_grad_(False)
        partial.eval().requires_grad_(False)
        result = {}
        for name, rows in (('fresh', cases[:len(cases) // 2]), ('sealed', cases[len(cases) // 2:])):
            recall = audit_recall(decoder, tokenizer, head, partial, roles, gate, features, markers, rows,
                                  topics, fixture, thresholds)
            result[name] = recall
            print(name, recall['summary'], 'partial', sum(row['passed'] for row in recall['partial']), flush=True)
        controls = audit_controls(decoder, tokenizer, head, partial, roles, gate, features, markers, topics,
                                  fixture, thresholds)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
        if args.verify:
            oracle, oracle_empty = oracle_partial(decoder, tokenizer, head, partial, markers, sealed_cases(fixture), topics)
    if (memory != reference['memory'] or frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != value for path, value in protected.items())):
        raise ValueError('A protected input or durable memory response changed')
    elapsed = perf_counter() - started
    core = {'recall': result, **controls, 'memory': memory}
    if args.verify:
        if any(expected[name] != value for name, value in core.items()):
            raise ValueError('The replay changed a field decision or response')
        store_json(args.output / 'reload.json', {'sha256': sha(weights), 'seconds': elapsed, 'oracle_partial': oracle, 'oracle_empty': oracle_empty})
        print('Fixed-plan partial:', sum(row['passed'] for row in oracle), '/', len(oracle), flush=True)
    else:
        store_json(args.output / 'result.json', {**core, 'sources': sources, 'individual_sha256': individual,
                                                 'sha256': sha(weights), 'access_questions': len(access),
                                                 'external_questions': len(external),
                                                 'partial_training_examples': len(examples), 'seconds': elapsed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('actions', type=Path)
    parser.add_argument('observations', type=Path)
    parser.add_argument('crosswoz', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('layers', nargs=2, type=Path)
    use = parser.add_mutually_exclusive_group()
    use.add_argument('--verify', action='store_true')
    use.add_argument('--audit-coverage', action='store_true')
    use.add_argument('--learned-coverage', action='store_true')
    args = parser.parse_args()
    if not (args.verify or args.audit_coverage or args.learned_coverage) and args.output.exists():
        parser.error('Choose a new output directory')
    main(args)
