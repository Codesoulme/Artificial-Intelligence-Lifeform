"""Learn paired Chinese field permissions while preserving unrestricted memory reads."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import random
from time import perf_counter

from torch.nn import functional as F

from scripts.train_spora_typed_fields import (
    CONTROL, NEW_CONTROL, ORDINARY, ROLE_NAMES, DATA, DESIGN, FRESH, HELD, MODEL, OBSERVATIONS, SCENARIOS, SOURCE,
    AutoTokenizer, MissingFieldActions, ObservationReadGate, SpanActionHead, TokenFieldAccessHead, WorkingTurnState,
    audits, case_observation, evaluate, extend_positions, fresh_cases, frozen_digest, generate, install, labels,
    load_full_streamed, load_integrated_runtime, load_recurrent_replacements, observation, policy, prompt_ids, read,
    records, run_answer, sealed_cases, sha, state_for, store_json, text_features, torch, training_data, training_samples,
)
from spora.language_training import atomic_save
from spora.research_field_access import (
    FIRST_ONLY, FIRST_REQUIRED, KEEP_SCOPE, SECOND_ONLY, SECOND_REQUIRED, SKIP_MEMORY, PairedFieldScopeHead,
)


SEED = 26160
EXCLUDE_TRAIN = (
    '请只提供{a}，不要写{b}。',
    '先给我{a}，{b}请省略。',
    '{b}这次不对外说，回答里要有{a}。',
    '回复里请带{a}，{b}空着。',
    '把{a}留下，删去里面的{b}。',
    '这张消息填{a}就够了。{b}由我之后决定。',
    '我自己再填{b}，你把原来定的{a}先发过来。',
    '只报{a}，别把{b}一块报给别人。',
)
OPTION_TRAIN = (
    '{a}一定要告诉我，有{b}也可以补一句。',
    '已知的{b}可以附上，但必须报{a}。',
    '先给我{a}。{b}查到了才写。',
    '正文要完整提供{a}，{b}有记录的话放在后面。',
    '把{a}写清楚；如果已经定了{b}，也带上。',
    '{b}有就加，没有就留空，{a}请直接发来。',
)
FULL_TRAIN = (
    '把登记的{a}和{b}一起告诉我。',
    '这两栏都需要确认：{b}和{a}。',
    '帮我把{a}、{b}都写到更新后的通知里。',
)
SCOPE_HOLDOUT = (
    ('clinic_booking', '家里要发复查便条，{a}请替我填好，{b}这一栏暂不外发。'),
    ('repair_pickup', '代领群看得到的那一行写{a}。{b}请从这条消息里去掉。'),
    ('pottery_inventory', '单据只印一栏，就用{a}；另一栏{b}暂不入单。'),
    ('choir_program', '给排节目单的同事发{a}，不要在发出的这条文字里透露{b}。'),
    ('garden_supply', '群里现在只需要这次花圃活动的{a}；至于{b}，请先别传到群里。'),
    ('film_notice', '请把{a}写在现场通知上，{b}这项另外交代，不放到这句话中。'),
)


def field_prompt(q, first, second):
    return f'状态包含两项：甲是{first}，乙是{second}。\n用户要求：{q}'


def feature_pair(features, rows):
    ordered, swapped = [], []
    for domain, q in rows:
        a, b = ROLE_NAMES[domain]
        ordered.append(features[field_prompt(q, a, b)])
        swapped.append(features[field_prompt(q, b, a)])
    return torch.stack(ordered), torch.stack(swapped)


def scope_texts(rows):
    return {field_prompt(q, a, b) for domain, q in rows
            for a, b in (ROLE_NAMES[domain], ROLE_NAMES[domain][::-1])}


def training_rows(topics, fixture, examples):
    rows, pairs = {}, []
    for domain, q, _ in examples:
        if domain:
            rows[domain, q] = KEEP_SCOPE
    for domain in topics:
        role_a, role_b = ROLE_NAMES[domain]
        for patterns, options in ((EXCLUDE_TRAIN, (FIRST_ONLY, SECOND_ONLY)),
                                  (OPTION_TRAIN, (FIRST_REQUIRED, SECOND_REQUIRED))):
            for pattern in patterns:
                q_a = pattern.format(a=role_a, b=role_b)
                q_b = pattern.format(a=role_b, b=role_a)
                rows[domain, q_a], rows[domain, q_b] = options
                pairs.append(((domain, q_a), (domain, q_b)))
        for phrase in FULL_TRAIN:
            for a, b in ((role_a, role_b), (role_b, role_a)):
                rows[domain, phrase.format(a=a, b=b)] = KEEP_SCOPE
        topic = fixture[domain]['label']
        for role in ROLE_NAMES[domain]:
            for q in (
                f'给{topic}写一条通用说明，用常见例子，避免提到已记下的{role}。',
                f'起草一段{topic}介绍，别代填具体{role}。',
                f'{topic}的宣传语怎么写？先不要使用记录里的{role}。',
            ):
                rows[domain, q] = SKIP_MEMORY
    return rows, pairs


def calibration_rows(topics, fixture):
    rows = {}
    for key, topic in topics.items():
        a, b = ROLE_NAMES[key]
        for first, other, exclusive, optional in ((a, b, FIRST_ONLY, FIRST_REQUIRED),
                                                   (b, a, SECOND_ONLY, SECOND_REQUIRED)):
            rows[key, f'可直接转发的是{first}，{other}留在草稿里。'] = exclusive
            rows[key, f'请汇报{first}。若{other}已有正式信息，顺便列下。'] = optional
        rows[key, topic['cal']] = KEEP_SCOPE
        rows[key, f'写一句介绍{fixture[key]["label"]}的通用话，只写常规事项，别套用记下的{b}。'] = SKIP_MEMORY
    return rows


def fit(model, features, train, pairs):
    keys = list(train)
    first, second = feature_pair(features, keys)
    classes = torch.tensor([train[key] for key in keys])
    index = {key: i for i, key in enumerate(keys)}
    pairs = [(index[a], index[b]) for a, b in pairs]
    rng = random.Random(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.025)
    for step in range(220):
        matched = rng.sample(pairs, 16)
        selected = [i for pair in matched for i in pair] + rng.sample(range(len(keys)), 36)
        logits = model(first[selected], second[selected])
        truth = classes[selected]
        pair_logits = logits[:32].reshape(16, 2, -1)
        pair_true = truth[:32].reshape(16, 2, 1)
        partner = pair_true.flip(dims=(1,))
        advantage = (pair_logits.gather(-1, pair_true) - pair_logits.gather(-1, partner)).squeeze(-1)
        loss = F.cross_entropy(logits, truth) + .3 * F.softplus(1. - advantage).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 55 == 0:
            print('Joint field permission', step + 1, round(float(loss.detach()), 5), flush=True)


def decisions(model, features, rows, threshold):
    first, second = feature_pair(features, rows)
    conf, proposed = model(first, second).softmax(dim=-1).max(dim=-1)
    return torch.where(conf >= threshold, proposed, KEEP_SCOPE), conf


@torch.inference_mode()
def calibrate(model, features, cal):
    rows = list(cal)
    expected = torch.tensor([cal[row] for row in rows])
    best = None
    for threshold in (.4, .5, .6, .7, .8, .9):
        predicted, _ = decisions(model, features, rows, threshold)
        correct = predicted.eq(expected)
        rates = [float(correct[expected == action].float().mean()) for action in range(6)]
        rank = min(rates), sum(rates), rates[KEEP_SCOPE], threshold
        if best is None or rank > best[0]:
            best = rank, threshold, rates
    return best[1], best[2]


def scoped_plan(fields, mode):
    if mode == SKIP_MEMORY:
        return labels()
    if mode not in (FIRST_ONLY, SECOND_ONLY, FIRST_REQUIRED, SECOND_REQUIRED):
        raise ValueError('Expected a learned override mode')
    a, b = fields if mode in (FIRST_ONLY, FIRST_REQUIRED) else fields[::-1]
    return labels((a,), excluded=(b,)) if mode in (FIRST_ONLY, SECOND_ONLY) else labels((a,), (b,))


def guarded_resolver(model, original, gate, pooled, seq, topics, base_thresholds, confidence):
    @torch.inference_mode()
    def choose(domain, q, raw):
        picked, _ = decisions(model, pooled, [(domain, q)], confidence)
        mode = int(picked[0])
        if mode == KEEP_SCOPE:
            return policy(original, gate, pooled, seq, q, raw, base_thresholds)
        chosen = scoped_plan(topics[domain]['fields'], mode)
        topic_is_related = gate.relevance(pooled[q], pooled[raw]) >= base_thresholds[1]
        return chosen, bool(chosen.requested and topic_is_related)
    return choose


def paired_holdout(topics):
    cases = []
    for domain, template in SCOPE_HOLDOUT:
        fields = topics[domain]['fields']
        for swapped, names in enumerate((ROLE_NAMES[domain], ROLE_NAMES[domain][::-1])):
            required, excluded = fields if not swapped else fields[::-1]
            cases.append((domain, template.format(a=names[0], b=names[1]), required, None, excluded))
    return tuple(cases)


@torch.inference_mode()
def paired_test(decoder, tokenizer, head, adapter, resolver, topics, fixture, cases, *, both=False):
    captured = []
    for domain, q, required, optional, excluded in cases:
        raw = observation(topics, fixture, domain)
        plan, enabled = resolver(domain, q, raw)
        expected = labels((required,), (optional,), (excluded,))
        values = dict(zip(topics[domain]['fields'], topics[domain]['example'], strict=True))
        prompt = prompt_ids(tokenizer, [], q)
        reply = run_answer(decoder, tokenizer, head, adapter, prompt, state_for(tokenizer, values), plan, enabled,
                           domain, topics)
        required_present = values[required] in reply['answer']
        leaked = bool(excluded and values[excluded] in reply['answer'])
        arms = {}
        if optional:
            short = run_answer(decoder, tokenizer, head, adapter, prompt, state_for(tokenizer, {required: values[required]}),
                               plan, enabled, domain, topics)
            arms['optional_absent'] = short
            arms['optional_pass'] = (values[required] in short['answer'] and short['eos']
                                     and not short.get('reported_missing') and values[optional] not in short['answer'])
        captured.append({'case': domain, 'question': q, 'required': required, 'excluded': excluded,
                         'plan': list(plan.roles), 'expected': list(expected.roles), 'required_present': required_present,
                         'leaked': leaked, 'passed': required_present and not leaked and reply['eos'], 'answer': reply,
                         **arms})
    if both:
        if len(captured) % 2 or any(one['case'] != two['case'] or one['required'] == two['required']
                                    for one, two in zip(captured[::2], captured[1::2], strict=True)):
            raise ValueError('The paired holdout must alternate roles within each fixed state')
    return captured


@torch.inference_mode()
def ordinary_test(decoder, tokenizer, head, adapter, resolver, topics, fixture):
    rows = []
    for domain, q in ORDINARY:
        plan, enabled = resolver(domain, q, observation(topics, fixture, domain))
        values = dict(zip(topics[domain]['fields'], topics[domain]['example'], strict=True))
        prompt = prompt_ids(tokenizer, [], q)
        with_state = run_answer(decoder, tokenizer, head, adapter, prompt, state_for(tokenizer, values), plan, enabled,
                                domain, topics)
        without = generate(decoder, tokenizer, head, prompt, WorkingTurnState())
        rows.append({'question': q, 'answer': with_state, 'unchanged': with_state == without})
    return rows


def summary(rows, *, pairs=False):
    result = {'count': len(rows), 'exact_plan': sum(row['plan'] == row['expected'] for row in rows),
              'leaks': sum(row['leaked'] for row in rows), 'passed': sum(row['passed'] for row in rows),
              'optional_pass': sum(row.get('optional_pass', False) for row in rows)}
    if pairs:
        result['both'] = sum(a['passed'] and b['passed'] for a, b in zip(rows[::2], rows[1::2], strict=True))
    return result


def main(args):
    started = perf_counter()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    action_path, action_results = args.actions / 'actions.pt', args.actions / 'result.json'
    gate_path, gate_results = args.gate / 'gate.pt', args.gate / 'result.json'
    typed_path, typed_results = args.typed / 'typed-fields.pt', args.typed / 'result.json'
    sources = (args.checkpoint, DATA, DESIGN, OBSERVATIONS, SCENARIOS, SOURCE, FRESH, action_path, action_results,
               gate_path, gate_results, typed_path, typed_results,
               *(folder / part for folder in args.layers for part in ('attention.pt', 'comparison.json')))
    protected = {path: sha(path) for path in sources}
    parent, old_gate, old_typed = map(read, (action_results, gate_results, typed_results))
    if (parent['sha256'] != protected[action_path] or old_gate['gate_sha256'] != protected[gate_path]
            or old_typed['sha256'] != protected[typed_path]):
        raise ValueError('A frozen component differs from its test report')
    prior = {Path(path).resolve(): digest for path, digest in old_typed['sources'].items()}
    if any(prior.get(path.resolve()) != digest for path, digest in protected.items() if path not in (typed_path, typed_results)):
        raise ValueError('A protected source differs from the typed field checkpoint')
    topics, _ = training_data(read(DESIGN))
    fixture = read(OBSERVATIONS)['topics']
    train_samples, _ = training_samples(topics, fixture, read(DATA)['values']['train'], random.Random(SEED))
    train, contrast = training_rows(topics, fixture, train_samples)
    cal = calibration_rows(topics, fixture)
    fresh, sealed = fresh_cases(FRESH), sealed_cases(fixture)
    same_state = paired_holdout(topics)
    scope = CONTROL + NEW_CONTROL
    probes = {(row[0], row[1]) for row in (*scope, *same_state, *ORDINARY)} | {
        (row['case'], row['question']) for row in (*fresh, *sealed)
    }
    if (set(train) & probes or set(cal) & (probes | train.keys())):
        raise ValueError('Constraint training, calibration and evaluation overlap')
    plain_text = {q for _, q in probes} | {observation(topics, fixture, domain) for domain in topics} | {
        case_observation(row, fixture) for row in (*fresh, *sealed)
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    if any(run['individual_sha256'] != individual for run in (parent, old_gate, old_typed)):
        raise ValueError('Individual differs from the previous runs')
    layers = load_recurrent_replacements(decoder, args.layers, protected[args.checkpoint])
    head = SpanActionHead(base.config.hidden_size).to(base.device)
    head.load_state_dict(torch.load(action_path, map_location='cpu', weights_only=True))
    head.eval().requires_grad_(False)
    typed = torch.load(typed_path, map_location='cpu', weights_only=True)
    old_plan = TokenFieldAccessHead(2 * base.config.hidden_size).to(base.device)
    old_plan.load_state_dict(typed['scope'])
    old_plan.eval().requires_grad_(False)
    adapter = MissingFieldActions(base.config.hidden_size).to(base.device)
    adapter.load_state_dict(typed['realizer'])
    adapter.eval().requires_grad_(False)
    gate_weights = torch.load(gate_path, map_location='cpu', weights_only=True)
    gate = ObservationReadGate(3 * base.config.hidden_size)
    gate.load_state_dict(gate_weights['weights'])
    gate.eval().requires_grad_(False)
    scope_head = PairedFieldScopeHead(3 * base.config.hidden_size)
    thresholds = gate_weights['thresholds']
    saved = args.output / 'scope.pt'
    digests = {str(path): digest for path, digest in protected.items()}
    if args.verify:
        expected = read(args.output / 'result.json')
        if (expected['sources'] != digests or expected['individual_sha256'] != individual
                or expected['sha256'] != sha(saved)):
            raise ValueError('A scope checkpoint or frozen source changed')
        checkpoint = torch.load(saved, map_location='cpu', weights_only=True)
        scope_head.load_state_dict(checkpoint['state'])
        cutoff = checkpoint['cutoff']
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        install(stack, layers)
        needed = probes | cal.keys()
        if not args.verify:
            needed |= train.keys()
        prompts = plain_text | scope_texts(needed)
        print('Encoding paired field questions:', len(prompts), flush=True)
        features, sequence = text_features(base, tokenizer, prompts, keep_tokens=plain_text)
        if not args.verify:
            fit(scope_head, features, train, contrast)
        scope_head.eval().requires_grad_(False)
        current_cutoff, cal_metrics = calibrate(scope_head, features, cal)
        print('Scope calibration', current_cutoff, cal_metrics, flush=True)
        if args.verify:
            if cutoff != current_cutoff or expected['calibration'] != cal_metrics:
                raise ValueError('Independent scope calibration did not reproduce')
        else:
            cutoff = current_cutoff
            atomic_save({'state': scope_head.state_dict(), 'cutoff': cutoff}, saved)
        new = guarded_resolver(scope_head, old_plan, gate, features, sequence, topics, thresholds, cutoff)
        def original(domain, q, raw):
            return policy(old_plan, gate, features, sequence, q, raw, thresholds)
        typed = {}
        for name, held in (('fresh', fresh), ('sealed', sealed)):
            result = audits(decoder, tokenizer, head, adapter, old_plan, gate, (features, sequence), held, topics,
                            fixture, thresholds, scope_plan=new)
            typed[name] = result
            print(name, 'complete paired', result['summary']['state_only']['both_without_detected_repeats'],
                  'partial', sum(row['bound_pass'] for row in result['partial']), flush=True)
        pair_result = {}
        for name, resolver in (('original', original), ('paired', new)):
            rows = paired_test(decoder, tokenizer, head, adapter, resolver, topics, fixture, same_state, both=True)
            pair_result[name] = {'rows': rows, 'summary': summary(rows, pairs=True)}
            print(name, pair_result[name]['summary'], flush=True)
        known = paired_test(decoder, tokenizer, head, adapter, new, topics, fixture, scope)
        ordinary = ordinary_test(decoder, tokenizer, head, adapter, new, topics, fixture)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    if (memory != parent['memory'] or frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('The individual or a verified source changed')
    core = {'paired': pair_result, 'recall': typed, 'scope': known, 'ordinary': ordinary, 'memory': memory}
    seconds = perf_counter() - started
    if args.verify:
        if any(expected[name] != actual for name, actual in core.items()):
            raise ValueError('Independent scope replay changed a plan or answer')
        store_json(args.output / 'reload.json', {'sha256': sha(saved), 'seconds': seconds})
    else:
        store_json(args.output / 'result.json', {**core, 'sha256': sha(saved), 'individual_sha256': individual,
                                                 'sources': digests, 'seconds': seconds, 'training': len(train),
                                                 'contrast_pairs': len(contrast), 'cutoff': cutoff,
                                                 'calibration': cal_metrics})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('actions', type=Path)
    parser.add_argument('gate', type=Path)
    parser.add_argument('typed', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('layers', nargs=2, type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not args.verify and args.output.exists():
        parser.error('Choose a fresh run folder')
    main(args)
