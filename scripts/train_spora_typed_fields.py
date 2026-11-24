"""Train token-level reading plans and bind field names to whole-field actions."""

import argparse
from contextlib import ExitStack
import math
from pathlib import Path
import random
import re
from time import perf_counter

from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from scripts.train_spora_field_access import (
    CONTROL, ORDINARY, DATA, DESIGN, FRESH, HELD, MODEL, OBSERVATIONS, SCENARIOS, SOURCE, AutoTokenizer,
    DynamicCache, EXCLUDE, FIELDS, IGNORE, OPTIONAL, REQUIRE, FieldAccessPlan, MissingFieldActions, ObservationReadGate,
    SpanActionHead, WorkingTurnState, backbone_step, capture, case_observation, evaluate, extend_positions, fresh_cases,
    frozen_digest, generate, grade, install, labels, load_full_streamed, load_integrated_runtime,
    load_recurrent_replacements, measures, observation, partial_masks, partial_targets, prompt_ids, read, records,
    sealed_cases, sha, state_for, store_json, tokens, torch, training_data, training_roles, training_samples,
)
from spora.language_training import atomic_save
from spora.research_field_access import TokenFieldAccessHead, typed_field_surface


SEED = 26152
# Spoken labels follow episode roles. They are used for rendering, while the planner
# receives only the question tokens. The experiment provides the episode schema.
ROLE_NAMES = {
    'clinic_booking': ('复诊时间', '门诊地点'),
    'museum_visit': ('导览时间', '参观展馆'),
    'repair_pickup': ('领取柜台', '取件人'),
    'hotel_checkin': ('住宿地点', '预订人'),
    'pottery_inventory': ('出货数量', '收货人'),
    'ballot_count': ('得票数', '候选人'),
    'choir_program': ('曲名', '领唱人'),
    'tool_checkout': ('领用设备', '领用人'),
    'garden_supply': ('随身工具', '碰头地点'),
    'hiking_setup': ('登山装备', '集合地点'),
    'film_notice': ('观众入口', '主持人'),
    'team_meeting': ('会议时间', '会议室'),
}
FALLBACK_NAMES = dict(zip(FIELDS, ('日期', '时间', '地点', '物品', '姓名', '数量', '方向'), strict=True))
NEW_CONTROL = (
    ('clinic_booking', '提醒上只留几点到诊，那家门诊留给我自己补。', 'time', None, 'place'),
    ('repair_pickup', '签收回执写维修品的领取地址，姓名一格先空着。', 'place', None, 'person'),
    ('pottery_inventory', '这张标签只填要留几件，接货人不用公开。', 'quantity', None, 'person'),
    ('choir_program', '发给排版的内容只要歌曲名字，负责领唱的人由我另行通知。', 'gear', None, 'person'),
    ('garden_supply', '给晚到的人发个碰头位置，工具那项省略。', 'place', None, 'gear'),
    ('film_notice', '贴纸上只写从哪个门进，主持名字先留白。', 'place', None, 'person'),
    ('clinic_booking', '先给出此次就诊的门诊名称。到诊时间落实了才附上。', 'place', 'time', None),
    ('repair_pickup', '哪位代领必须说清，领取柜台查得到的话再写一行。', 'person', 'place', None),
    ('pottery_inventory', '收货者的姓名要写全；若已有确数，附带实际件数即可。', 'person', 'quantity', None),
    ('choir_program', '节目单曲名请确认，领唱一栏有登记才填。', 'gear', 'person', None),
    ('garden_supply', '会合地点先发一下，工具如果已经列了清单，顺便写上。', 'place', 'gear', None),
    ('film_notice', '必须确定主持人，取到观众入口资料时可一块发。', 'person', 'place', None),
)


def expand_training(topics, fixture, sample):
    rows = training_roles(topics, fixture, sample)
    for name, meta in topics.items():
        a, b = meta['fields']
        a_name, b_name = ROLE_NAMES[name]
        theme = fixture[name]['label']
        for use, other, requested, remaining in ((a, b, a_name, b_name), (b, a, b_name, a_name)):
            explicit = (
                f'{theme}的{requested}需要给我确认一下，{remaining}先留白。',
                f'只传达{requested}即可，我会另外告知{remaining}。',
                f'我来填{remaining}。你把已记下的{requested}发来。',
                f'请给出{theme}记事的{requested}，省略其中的{remaining}。',
            )
            extra = (
                f'先写{theme}的{requested}，查到{remaining}的话顺带提一下。',
                f'{requested}得有，{remaining}若确定了可作为补充。',
                f'这次要知道{requested}。还记有{remaining}的话就一并发过来。',
                f'{theme}里的{requested}是必须项，已有的{remaining}也能写进去。',
            )
            rows.update((q, labels((use,), excluded=(other,))) for q in explicit)
            rows.update((q, labels((use,), (other,))) for q in extra)
            rows[f'起草通用{theme}的说明，{remaining}这一栏不要代入现有记录。'] = labels(excluded=(other,))
    return rows


@torch.inference_mode()
def text_features(base, tokenizer, texts):
    pooled, sequence = {}, {}
    for text in sorted(texts):
        ids = torch.tensor([tokens(tokenizer, text)], device=base.device)
        cache = DynamicCache(config=base.config)
        hidden = base.model(input_ids=ids, past_key_values=cache, use_cache=True).last_hidden_state[0].float()
        words = base.get_input_embeddings()(ids[0]).float()
        scale = math.sqrt(hidden.shape[-1])
        pooled[text] = torch.cat([F.normalize(view, dim=0) for view in
                                  (hidden.mean(dim=0), hidden[-1], words.mean(dim=0))]).cpu() * scale
        sequence[text] = torch.cat((F.normalize(hidden, dim=-1), F.normalize(words, dim=-1)), dim=-1).cpu() * scale
    return pooled, sequence


def role_logits(model, features, texts):
    items = [features[q] for q in texts]
    device = next(model.parameters()).device
    values = pad_sequence(items, batch_first=True).to(device)
    lengths = torch.tensor([row.shape[0] for row in items], device=device)
    mask = torch.arange(values.shape[1], device=device)[None, :] < lengths[:, None]
    return model(values, mask)


def fit_roles(model, features, train, external):
    device = next(model.parameters()).device
    corpus = list(train.items())
    weights = torch.tensor([.40, 1., 1.25, 1.5], device=device)
    rng = random.Random(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.02)
    for step in range(240):
        batch = rng.sample(corpus, 40)
        aux = rng.sample(external, 16)
        logits = role_logits(model, features, [q for q, _ in batch] + [item['text'] for item in aux])
        target = torch.tensor([plan.roles for _, plan in batch], device=device)
        plac = logits[len(batch):, FIELDS.index('place')]
        aux_score = plac[:, REQUIRE] - torch.logsumexp(plac[:, [IGNORE, OPTIONAL, EXCLUDE]], dim=-1)
        aux_target = torch.tensor([item['required'] for item in aux], dtype=torch.float32, device=device)
        loss = (F.cross_entropy(logits[:len(batch)].flatten(0, 1), target.flatten(), weight=weights)
                + .18 * F.binary_cross_entropy_with_logits(aux_score, aux_target))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 60 == 0:
            print('Token field roles', step + 1, round(float(loss.detach()), 5), flush=True)


def domain_labels(topics, domain):
    joined = FALLBACK_NAMES | dict(zip(topics[domain]['fields'], ROLE_NAMES[domain], strict=True))
    return tuple(joined[field] for field in FIELDS)


def bound_surfaces(tokenizer, domain, topics, state):
    encode = lambda text: tokens(tokenizer, text)
    names = domain_labels(topics, domain)
    values = [typed_field_surface(encode, name, value) if value else ()
              for name, value in zip(names, state.slots, strict=True)]
    missing = [typed_field_surface(encode, name) for name in names]
    return values, missing


def make_training(tokenizer, topics, vocabulary, vocab):
    rng = random.Random(SEED)
    examples = []
    for domain, meta in topics.items():
        a, b = meta['fields']
        for populated in ((a, b), (a,), (b,), ()):
            for i in range(3):
                state = state_for(tokenizer, {field: rng.choice(vocabulary[field]) for field in populated})
                known, missing = bound_surfaces(tokenizer, domain, topics, state)
                needs = tuple(FIELDS.index(field) for field in (a, b) if field not in populated)
                action = []
                for field in ((a, b) if i % 2 else (b, a)):
                    j = FIELDS.index(field)
                    action.append((j, known[j]) if known[j] else (len(FIELDS) + j, missing[j]))
                if i == 1:
                    action.insert(0, (None, tokens(tokenizer, '目前的记录如下。')))
                request = meta['train'][i % len(meta['train'])]
                surface, targets = partial_targets(action, vocab, tokenizer.eos_token_id)
                examples.append((prompt_ids(tokenizer, [], request), surface, targets, state, needs))
    return examples


def fit_realizer(decoder, head, bound, samples):
    base = decoder.base
    vocab = base.config.vocab_size
    with torch.no_grad():
        caches = capture(decoder, base.model.layers[-1], [(p + s, len(s) + 1) for p, s, _, _, _ in samples])
    opt = torch.optim.AdamW(bound.parameters(), lr=8e-4)
    order = list(range(len(samples)))
    rng = random.Random(SEED)
    for step in range(480):
        if step % len(order) == 0:
            rng.shuffle(order)
        index = order[step % len(order)]
        _, _, targets, state, needs = samples[index]
        positions = torch.tensor([target.offset for target in targets])
        hidden = base.model.norm(caches[index][positions].to(base.device))
        keys = head.field_keys(state, base.get_input_embeddings())
        available, absent = partial_masks(state, needs, targets, vocab, base.device)
        target = torch.tensor([action.action for action in targets], device=base.device)
        scores = bound(hidden, keys, available, absent, head, base.lm_head)
        weight = torch.where(target >= vocab, 2., 1.)
        weight[-1] = 2.
        loss = -(scores.gather(-1, target[:, None]).squeeze(-1) * weight).sum() / weight.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bound.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        if (step + 1) % 120 == 0:
            print('Typed field output', step + 1, round(float(loss.detach()), 5), flush=True)


@torch.inference_mode()
def realize(decoder, tokenizer, head, adapter, prompt, state, plan, domain, topics, limit=48):
    base = decoder.base
    vocab = base.config.vocab_size
    projected = plan.project(state)
    copied, markers = bound_surfaces(tokenizer, domain, topics, projected)
    owed = {i for i, role in enumerate(plan.roles) if role == REQUIRE}
    reads_pending = {i for i, value in enumerate(copied) if value}
    unavailable = set(plan.missing(projected))
    cache = DynamicCache(config=base.config)
    hidden = backbone_step(base, prompt, cache)
    keys = head.field_keys(projected, base.get_input_embeddings())
    surface, reads, declared, ended = [], [], [], False
    while len(surface) < limit:
        budget = limit - len(surface)
        present = torch.tensor([i in reads_pending and len(value) <= budget for i, value in enumerate(copied)],
                               device=base.device)
        absent = torch.tensor([i in unavailable and len(marker) <= budget for i, marker in enumerate(markers)],
                              device=base.device)
        scores = adapter(hidden, keys, present, absent, head, base.lm_head)
        if owed:
            scores[tokenizer.eos_token_id] = -torch.inf
        action = int(scores.argmax())
        if action == tokenizer.eos_token_id:
            ended = True
            break
        if action < vocab:
            chunk = (action,)
        elif action < vocab + len(FIELDS):
            i = action - vocab
            chunk = copied[i]
            reads_pending.remove(i)
            owed.discard(i)
            reads.append(FIELDS[i])
        else:
            i = action - vocab - len(FIELDS)
            chunk = markers[i]
            unavailable.remove(i)
            owed.discard(i)
            declared.append(FIELDS[i])
        surface.extend(chunk)
        if len(surface) < limit:
            hidden = backbone_step(base, chunk, cache)
    return {'answer': tokenizer.decode(surface, skip_special_tokens=True).strip(), 'eos': ended, 'reads': reads,
            'reported_missing': declared, 'route': True, 'pending_required': [FIELDS[i] for i in sorted(owed)]}


def policy(model, gate, pooled, sequence, q, raw, thresholds):
    logits = role_logits(model, sequence, [q])[0]
    choice = FieldAccessPlan(tuple(logits.argmax(dim=-1).tolist()))
    need, _ = gate.requirements(pooled[q])
    relevance = gate.relevance(pooled[q], pooled[raw])
    return choice, bool(choice.requested and need >= thresholds[0] and relevance >= thresholds[1])


def run_answer(decoder, tokenizer, head, adapter, prompt, state, plan, read, domain, topics):
    if read:
        return realize(decoder, tokenizer, head, adapter, prompt, state, plan, domain, topics)
    return generate(decoder, tokenizer, head, prompt, WorkingTurnState())


@torch.inference_mode()
def audits(decoder, tokenizer, head, adapter, model, gate, features, cases, topics, fixture, thresholds, *, oracle=False):
    pooled, sequence = features
    whole, partial, empty = [], [], []
    for i, row in enumerate(cases):
        key, q, values = row['case'], row['question'], row['values']
        a, b = fields = topics[key]['fields']
        label_a, label_b = ROLE_NAMES[key]
        plan, enabled = ((labels(fields), True) if oracle else policy(
            model, gate, pooled, sequence, q, case_observation(row, fixture), thresholds))
        prompt = prompt_ids(tokenizer, [], q)
        def decode(values_to_use):
            return run_answer(decoder, tokenizer, head, adapter, prompt, state_for(tokenizer, values_to_use),
                              plan, enabled, key, topics)

        if not oracle:
            replacement = cases[i ^ 1]['values']['new']
            if cases[i ^ 1]['case'] != key or replacement in values.values():
                raise ValueError('Expected one counterfactual field value')
            outputs = {}
            for mode, value, foil in (('matched', values['new'], replacement), ('swapped', replacement, values['new'])):
                reply = decode({a: value, b: values['keep']})
                text = reply['answer']
                forbidden = (values['old'], foil)
                outputs[mode] = {**reply, 'new': value in text, 'keep': values['keep'] in text,
                                 'foil': any(old in text for old in forbidden),
                                 'passed': grade(reply, [re.escape(value), re.escape(values['keep'])],
                                                 [re.escape(old) for old in forbidden])}
            whole.append({'case': key, 'variant': row['variant'], 'values': values, 'replacement': replacement,
                          'outputs': {'state_only': outputs}, 'roles': list(plan.roles)})
        for available, unknown, value, known_name, absent_name in (
            (b, a, values['keep'], label_b, label_a), (a, b, values['new'], label_a, label_b),
        ):
            reply = decode({available: value})
            text = reply['answer']
            literal = (value in text and reply.get('reported_missing') == [unknown] and reply['eos']
                       and values['old'] not in text)
            bound = (literal and f'{known_name}为{value}。' in text and f'{absent_name}尚未记录。' in text
                     and f'{absent_name}为{value}' not in text)
            partial.append({'case': key, 'variant': row['variant'], 'field': unknown,
                            'bound_pass': bound, 'literal_pass': literal, 'answer': reply})
        if oracle and not row['variant']:
            reply = decode({})
            text = reply['answer']
            correct = (reply['eos'] and set(reply.get('reported_missing', [])) == set(fields)
                       and all(f'{label}尚未记录。' in text for label in (label_a, label_b))
                       and not any(value in text for value in values.values()))
            empty.append({'case': key, 'passed': correct, 'answer': reply})
    return {'rows': whole, 'summary': measures(whole) if whole else {}, 'partial': partial, 'empty': empty}


@torch.inference_mode()
def scoped(decoder, tokenizer, head, adapter, model, gate, features, topics, fixture, thresholds):
    pooled, sequence = features
    tests = []
    for domain, q, needed, optional, exclude in CONTROL + NEW_CONTROL:
        plan, enabled = policy(model, gate, pooled, sequence, q, observation(topics, fixture, domain), thresholds)
        prompt = prompt_ids(tokenizer, [], q)
        example = dict(zip(topics[domain]['fields'], topics[domain]['example'], strict=True))
        complete = run_answer(decoder, tokenizer, head, adapter, prompt, state_for(tokenizer, example),
                              plan, enabled, domain, topics)
        output = {'complete': complete}
        if optional:
            output['optional_absent'] = run_answer(decoder, tokenizer, head, adapter, prompt,
                                                   state_for(tokenizer, {needed: example[needed]}),
                                                   plan, enabled, domain, topics)
        tests.append({'case': domain, 'question': q, 'plan': list(plan.roles),
                      'expected': list(labels((needed,), (optional,), (exclude,)).roles),
                      'excluded_leak': bool(exclude and example[exclude] in complete['answer']),
                      'required_present': example[needed] in complete['answer'], 'outputs': output})
    regular = []
    for domain, q in ORDINARY:
        plan, enabled = policy(model, gate, pooled, sequence, q, observation(topics, fixture, domain), thresholds)
        prompt = prompt_ids(tokenizer, [], q)
        example = dict(zip(topics[domain]['fields'], topics[domain]['example'], strict=True))
        proposed = run_answer(decoder, tokenizer, head, adapter, prompt, state_for(tokenizer, example),
                              plan, enabled, domain, topics)
        plain = generate(decoder, tokenizer, head, prompt, WorkingTurnState())
        regular.append({'question': q, 'same_answer': proposed == plain, 'answer': proposed})
    return {'scope': tests, 'ordinary': regular}


@torch.inference_mode()
def infer_label_schema(decoder, tokenizer, head, adapter, model, gate, features, prototypes, held, topics, fixture,
                       thresholds):
    """Infer the role lexicon from the stored observation and test its real output."""
    pooled, sequence = features
    domains = tuple(topics)
    centroids = F.normalize(torch.stack([torch.stack([pooled[x] for x in prototypes[name]]).mean(dim=0)
                                         for name in domains]), dim=-1)
    checks = []
    for row in held:
        domain, q, values = row['case'], row['question'], row['values']
        raw = case_observation(row, fixture)
        picked = domains[int((centroids @ F.normalize(pooled[raw], dim=0)).argmax())]
        plan, enabled = policy(model, gate, pooled, sequence, q, raw, thresholds)
        prompt = prompt_ids(tokenizer, [], q)
        a, b = topics[domain]['fields']
        name_a, name_b = ROLE_NAMES[domain]
        partial = []
        for known, value, known_name, absent_field, absent_name in (
            (a, values['new'], name_a, b, name_b), (b, values['keep'], name_b, a, name_a),
        ):
            state = state_for(tokenizer, {known: value})
            reply = run_answer(decoder, tokenizer, head, adapter, prompt, state, plan, enabled, picked, topics)
            text = reply['answer']
            passed = (reply['eos'] and reply.get('reported_missing') == [absent_field]
                      and f'{known_name}为{value}。' in text and f'{absent_name}尚未记录。' in text
                      and values['old'] not in text)
            partial.append({'field': absent_field, 'passed': passed, 'answer': reply})
        checks.append({'expected_schema': domain, 'inferred_schema': picked, 'variant': row['variant'],
                       'partial': partial})
    return checks


def main(args):
    started = perf_counter()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    action_source = args.actions / 'actions.pt'
    action_report = args.actions / 'result.json'
    gate_source = args.gate / 'gate.pt'
    gate_report = args.gate / 'result.json'
    source = [args.checkpoint, DATA, DESIGN, OBSERVATIONS, SOURCE, SCENARIOS, FRESH, action_source, action_report,
              gate_source, gate_report, args.crosswoz,
              *(folder / file for folder in args.layers for file in ('attention.pt', 'comparison.json'))]
    protected = {path: sha(path) for path in source}
    parent = read(action_report)
    previous = read(gate_report)
    prior_sources = {Path(path).resolve(): digest for path, digest in parent['sources'].items()}
    bound_inputs = [args.checkpoint, DATA, SOURCE, SCENARIOS, FRESH, *(
        folder / file for folder in args.layers for file in ('attention.pt', 'comparison.json'))]
    if (any(prior_sources.get(path.resolve()) != protected[path] for path in bound_inputs)
            or parent['sha256'] != protected[action_source] or previous['gate_sha256'] != protected[gate_source]):
        raise ValueError('One of the frozen references changed')
    topics, _ = training_data(read(DESIGN))
    fixture = read(OBSERVATIONS)['topics']
    words = read(DATA)['values']['train']
    examples, prototypes = training_samples(topics, fixture, words, random.Random(SEED))
    train = expand_training(topics, fixture, examples)
    aux = read(args.crosswoz)['rows']
    fresh, sealed = fresh_cases(FRESH), sealed_cases(fixture)
    held = {row['question'] for row in (*fresh, *sealed)} | {row[1] for row in CONTROL + NEW_CONTROL + ORDINARY}
    train_text = set(train) | {row['text'] for row in aux}
    if train_text & held:
        raise ValueError('Held-out access questions overlap training')
    prompts = held | {observation(topics, fixture, key) for key in topics} | {
        case_observation(row, fixture) for row in (*fresh, *sealed)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != parent['individual_sha256'] or individual != previous['individual_sha256']:
        raise ValueError('The individual changed')
    candidates = load_recurrent_replacements(decoder, args.layers, protected[args.checkpoint])
    head = SpanActionHead(base.config.hidden_size).to(base.device)
    head.load_state_dict(torch.load(action_source, map_location='cpu', weights_only=True))
    head.eval().requires_grad_(False)
    saved = torch.load(gate_source, map_location='cpu', weights_only=True)
    gate = ObservationReadGate(3 * base.config.hidden_size)
    gate.load_state_dict(saved['weights'])
    gate.eval().requires_grad_(False)
    thresholds = saved['thresholds']
    model = TokenFieldAccessHead(2 * base.config.hidden_size).to(base.device)
    adapter = MissingFieldActions(base.config.hidden_size).to(base.device)
    weights = args.output / 'typed-fields.pt'
    sources = {str(path): value for path, value in protected.items()}
    if args.verify:
        expected = read(args.output / 'result.json')
        if (expected['sources'] != sources or expected['individual_sha256'] != individual
                or expected['sha256'] != sha(weights)):
            raise ValueError('The typed field checkpoint or test sources changed')
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['scope'])
        adapter.load_state_dict(checkpoint['realizer'])
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        install(stack, candidates)
        inputs = prompts if args.verify else train_text | prompts
        if args.verify:
            inputs |= {text for items in prototypes.values() for text in items}
        print('Encoding typed-access inputs:', len(inputs), flush=True)
        features = text_features(base, tokenizer, inputs)
        if not args.verify:
            fit_roles(model, features[1], train, aux)
            samples = make_training(tokenizer, topics, words, base.config.vocab_size)
            fit_realizer(decoder, head, adapter, samples)
            atomic_save({'scope': {key: value.cpu() for key, value in model.state_dict().items()},
                         'realizer': {key: value.cpu() for key, value in adapter.state_dict().items()}}, weights)
        model.eval().requires_grad_(False)
        adapter.eval().requires_grad_(False)
        results = {}
        for name, cases, oracle in (('fresh', fresh, False), ('sealed', sealed, False), ('fixed', sealed, True)):
            comparison = audits(decoder, tokenizer, head, adapter, model, gate, features, cases, topics, fixture,
                                thresholds, oracle=oracle)
            results[name] = comparison
            print(name, comparison['summary'], 'typed partial',
                  sum(row['bound_pass'] for row in comparison['partial']), flush=True)
        controls = scoped(decoder, tokenizer, head, adapter, model, gate, features, topics, fixture, thresholds)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
        if args.verify:
            inferred = infer_label_schema(decoder, tokenizer, head, adapter, model, gate, features, prototypes, sealed,
                                           topics, fixture, thresholds)
    if (memory != parent['memory'] or frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('The individual, frozen source, or memory replay changed')
    core = {'recall': results, **controls, 'memory': memory}
    elapsed = perf_counter() - started
    if args.verify:
        if any(expected[name] != value for name, value in core.items()):
            raise ValueError('Reload changed a typed field result')
        store_json(args.output / 'reload.json', {'sha256': sha(weights), 'seconds': elapsed, 'inferred_schema': inferred})
        print('Inferred schema', sum(row['expected_schema'] == row['inferred_schema'] for row in inferred),
              '/', len(inferred), 'typed partial', sum(x['passed'] for row in inferred for x in row['partial']),
              flush=True)
    else:
        store_json(args.output / 'result.json', {**core, 'individual_sha256': individual, 'sha256': sha(weights),
                                                 'sources': sources, 'role_questions': len(train),
                                                 'external_questions': len(aux), 'realizer_examples': len(samples),
                                                 'seconds': elapsed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('actions', type=Path)
    parser.add_argument('gate', type=Path)
    parser.add_argument('crosswoz', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('layers', nargs=2, type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not args.verify and args.output.exists():
        parser.error('Choose a fresh run folder')
    main(args)
