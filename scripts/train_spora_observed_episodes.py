"""Learn observation writes and read back three-field working-state snapshots."""

import argparse
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
import random
from time import perf_counter

from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures, SemanticLocator
from scripts.train_spora_dialogue_working_state import annotated, dialogue_data, fit_writer
from scripts.train_spora_paired_field_scope import (
    PairedFieldScopeHead, guarded_resolver, scope_texts, training_samples,
)
from scripts.train_spora_typed_fields import (
    DATA, DESIGN, FRESH, HELD, MODEL, OBSERVATIONS, ROLE_NAMES, SCENARIOS, SOURCE, AutoTokenizer,
    MissingFieldActions, ObservationReadGate, SpanActionHead, TokenFieldAccessHead, WorkingTurnState, evaluate,
    expand_training, extend_positions, fit_roles, frozen_digest, install, labels, load_full_streamed,
    load_integrated_runtime, load_recurrent_replacements, policy, prompt_ids, read, records, run_answer,
    sha, state_for, store_json, text_features, torch, training_data,
)
from spora.language_training import atomic_save
from spora.research_dialogue_working_state import FIELDS, SPAN_LIMIT, JointTurnValueWriter, characters
from spora.research_observation_episode import ObservationEpisode


SEED = 26166
EPISODES = DATA.with_name('working-state-observed-episodes-v1.json')
EXTRA_NAMES = {'clinic_booking':'联络人', 'repair_pickup':'领取钟点', 'pottery_inventory':'交货处',
               'choir_program':'上台时刻', 'garden_supply':'带队人', 'film_notice':'开场时刻'}
RAW_KEY = '<episode-observation-vector>'


def render(template, words):
    return template.format_map(words)


def held_episodes(source):
    cases = []
    for spec in source['cases']:
        a, b, extra = spec['fields']
        for i, (old, new, alternate, keep, added) in enumerate(spec['values']):
            for arm, update in enumerate((new, alternate)):
                values = {'old': old, 'new': update, 'keep': keep, 'extra': added}
                cases.append({'key': f'{spec["case"]}-{i}-{arm}', 'topic': spec['case'], 'fields': spec['fields'],
                              'arm': arm, 'index': i, 'first': render(spec['first'], values),
                              'update': render(spec['update'], values), 'neutral': spec['neutral'],
                              'all': spec['all'], 'limited': spec['limited'], 'old': old,
                              'initial': {a: old, b: keep, extra: added},
                              'change': {a: update}, 'expected': {a: update, b: keep, extra: added}})
    return cases


def training_examples(source, specs, fixture, topics):
    rng = random.Random(SEED)
    pool = source['values']['train']
    rows = dialogue_data(source, held=False, count=40)
    base, prototypes = training_samples(topics, fixture, pool, random.Random(SEED))
    roles = expand_training(topics, fixture, base)
    for spec in specs:
        domain, fields = spec['case'], spec['fields']
        a, b, c = fields
        names = dict(zip(fields, (*ROLE_NAMES[domain], EXTRA_NAMES[domain]), strict=True))
        theme = fixture[domain]['label']
        for iteration in range(30):
            sequence = fields[iteration % 3:] + fields[:iteration % 3]
            parts = [f'{names[field]}为' + '{' + field + '}' for field in sequence]
            intro = (f'登记一下{theme}：', f'{theme}这次的记录是：', f'关于{theme}的三项情况，')[iteration % 3]
            first = intro + '，'.join(parts) + '。'
            choices = {field: rng.choice(pool[field]) for field in fields}
            changed = sequence[iteration % 2]
            replacement = rng.choice([value for value in pool[changed] if value != choices[changed]])
            others = [field for field in fields if field != changed]
            changes = (
                f'{theme}的{names[changed]}换为' + '{' + changed + '}，另外两项照常。',
                f'{names[others[0]]}及{names[others[1]]}沿用。现在{names[changed]}更新到' + '{' + changed + '}。',
                f'刚确认了{theme}，别用原{names[changed]}，改成' + '{' + changed + '}，其他也一样。',
            )[iteration % 3]
            query = f'把{theme}相关的{names[a]}、{names[b]}和{names[c]}都告诉我。'
            rows.append({'first': annotated(first, choices), 'update': annotated(changes, {changed: replacement}),
                         'question': query})
        for seq in (fields, fields[1:] + fields[:1], fields[::-1]):
            words = [names[field] for field in seq]
            for q in (
                f'{theme}的{words[0]}、{words[1]}、{words[2]}三项都要列上。',
                f'核对一下{theme}已经登记的全部信息：{words[0]}，以及{words[1]}、{words[2]}。',
                f'一起发给我{words[0]}和{words[1]}，也告诉我{words[2]}。',
                f'{theme}里{words[0]}、{words[1]}和{words[2]}各是什么？',
            ):
                roles[q] = labels(seq)
            for q in (
                f'{theme}通知只写{words[0]}与{words[1]}，{words[2]}先不公开。',
                f'{words[2]}留给我，给我的答复里需要有{words[0]}和{words[1]}。',
                f'请只填{words[0]}、{words[1]}两个格子，{words[2]}留白。',
                f'{theme}先说明{words[0]}和{words[1]}，内容不包含{words[2]}。',
            ):
                roles[q] = labels(seq[:2], excluded=(seq[2],))
    return rows, roles, prototypes


@torch.inference_mode()
def locate_writes(locator, cases):
    output = {}
    for case in cases:
        output[case['key']] = [locator(case[label]) for label in ('first', 'update', 'neutral', 'all', 'limited')]
    return output


def metrics(cases, votes):
    return {'initial_exact': sum(votes[row['key']][0] == row['initial'] for row in cases),
            'update_exact': sum(votes[row['key']][1] == row['change'] for row in cases),
            'after_update_exact': sum((votes[row['key']][0] | votes[row['key']][1]) == row['expected'] for row in cases),
            'after_neutral_exact': sum((votes[row['key']][0] | votes[row['key']][1] | votes[row['key']][2])
                                       == row['expected'] for row in cases),
            'neutral_write': sum(bool(votes[row['key']][2]) for row in cases),
            'query_attempts': sum(bool(vote) for row in cases for vote in votes[row['key']][3:])}


def encode_changes(tokenizer, values):
    encoded = {name: tuple(tokenizer.encode(value, add_special_tokens=False)) for name, value in values.items()}
    return {name: span for name, span in encoded.items() if 0 < len(span) <= SPAN_LIMIT}


def episodes(cases, votes, tokenizer, pooled, prototypes, topics):
    names = tuple(topics)
    origin = F.normalize(torch.stack([torch.stack([pooled[x] for x in prototypes[key]]).mean(dim=0)
                                      for key in names]), dim=-1)
    snapshots = {}
    for case in cases:
        raw = pooled[case['first']]
        picked = names[int((origin @ F.normalize(raw, dim=0)).argmax())]
        empty = ObservationEpisode(WorkingTurnState(), picked, raw)
        events = votes[case['key']]
        initial = empty.apply(encode_changes(tokenizer, events[0]))
        current = initial.apply(encode_changes(tokenizer, events[1])).apply(encode_changes(tokenizer, events[2]))
        expected = ObservationEpisode(state_for(tokenizer, case['expected']), case['topic'], raw)
        snapshots[case['key']] = {'initial': initial.payload(), 'current': current.payload(),
                                   'gold': expected.payload()}
    return snapshots


def state_values(tokenizer, state):
    return {field: tokenizer.decode(value, skip_special_tokens=True)
            for field, value in zip(FIELDS, state.slots, strict=True) if value}


def read_episode(decoder, tokenizer, head, adapter, route, planner, gate, features, episode, prompt, text,
                 topics, thresholds):
    pooled, seq = features
    inputs = {**pooled, RAW_KEY: episode.key}
    plan, read_now = policy(planner, gate, inputs, seq, text, RAW_KEY, thresholds)
    planned = {FIELDS[index] for index, role in enumerate(plan.roles) if role}
    if not (planned - set(topics[episode.topic]['fields'])):
        choose = guarded_resolver(route['model'], planner, gate, inputs, seq, topics, thresholds, route['cutoff'])
        plan, read_now = choose(episode.topic, text, RAW_KEY)
    reply = run_answer(decoder, tokenizer, head, adapter, prompt, episode.state, plan, read_now, episode.topic, topics)
    return reply, list(plan.roles)


def audit(decoder, tokenizer, head, adapter, route, planner, gate, features, snapshots, cases, topics, thresholds):
    output = []
    width = 3 * decoder.base.config.hidden_size
    vocab = decoder.base.config.vocab_size
    for case in cases:
        results = {}
        for arm in ('current', 'gold'):
            restored = ObservationEpisode.load(snapshots[case['key']][arm], topics=topics, width=width, vocab=vocab)
            predicted, answers = state_values(tokenizer, restored.state), {}
            for read_type in ('all', 'limited'):
                text = case[read_type]
                prompt = prompt_ids(tokenizer, [], text)
                reply, roles = read_episode(decoder, tokenizer, head, adapter, route, planner, gate, features,
                                            restored, prompt, text, topics, thresholds)
                expected = case['fields'] if read_type == 'all' else case['fields'][:2]
                required = set(expected)
                reported = set(reply['reads'])
                leak = bool(read_type == 'limited' and case['expected'][case['fields'][2]] in reply['answer'])
                correct = (reply['eos'] and reported == required and all(case['expected'][f] in reply['answer']
                            for f in required) and case['old'] not in reply['answer'] and not leak
                           and not reply.get('reported_missing'))
                answers[read_type] = {'reply': reply, 'roles': roles, 'passed': correct, 'excluded': leak}
            results[arm] = {'topic': restored.topic, 'state': predicted, 'revision': restored.revision,
                            'answers': answers}
        output.append({'key': case['key'], 'topic': case['topic'], 'expected': case['expected'], **results})
    return output


def summarize(rows):
    result = {}
    for mode in ('current', 'gold'):
        matching = [row[mode] for row in rows]
        result[mode] = {
            'topic': sum(row['topic'] == row[mode]['topic'] for row in rows),
            'state': sum(row['expected'] == row[mode]['state'] for row in rows),
            'all': sum(row['answers']['all']['passed'] for row in matching),
            'limited': sum(row['answers']['limited']['passed'] for row in matching),
            'leaked': sum(row['answers']['limited']['excluded'] for row in matching),
            'all_pairs': sum(a['answers']['all']['passed'] and b['answers']['all']['passed']
                             for a, b in zip(matching[::2], matching[1::2], strict=True)),
            'limited_pairs': sum(a['answers']['limited']['passed'] and b['answers']['limited']['passed']
                                 for a, b in zip(matching[::2], matching[1::2], strict=True)),
        }
    return result


@torch.inference_mode()
def audit_update_spans(module, features, cases):
    """Separate absent-score rejection from failure to rank the right interval."""
    device = module.embedding.weight.device
    rows = []
    for case in cases:
        text = case['update']
        (field, value), = case['change'].items()
        if text.count(value) != 1 or len(value) > SPAN_LIMIT:
            raise ValueError('The held update must identify one eligible correct span')
        begin = text.index(value)
        finish = begin + len(value) - 1
        scores = module(characters([text]).to(device), features(text).to(device))[0, FIELDS.index(field)].float()
        expected = scores[1 + begin * len(text) + finish]
        if not torch.isfinite(expected):
            raise ValueError('The annotated correction span was masked out')
        best, highest = scores[1:].max(dim=0)
        first, last = divmod(int(highest), len(text))
        rows.append({'key': case['key'], 'field': field, 'expected': value,
                     'best_span': text[first:last + 1], 'rank': 1 + int((scores[1:] > expected).sum()),
                     'best_minus_absent': float(best - scores[0]),
                     'expected_minus_absent': float(expected - scores[0])})
    return rows


def main(args):
    started = perf_counter()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    action_path, action_report = args.actions / 'actions.pt', args.actions / 'result.json'
    gate_path, gate_report = args.gate / 'gate.pt', args.gate / 'result.json'
    typed_path, typed_report = args.typed / 'typed-fields.pt', args.typed / 'result.json'
    scope_path, scope_report = args.scope / 'scope.pt', args.scope / 'result.json'
    writer_path, writer_report = args.writer / 'joint.pt', args.writer / 'comparison.json'
    paths = [args.checkpoint, DATA, SOURCE, SCENARIOS, DESIGN, OBSERVATIONS, FRESH, EPISODES, args.crosswoz,
             action_path, action_report, gate_path, gate_report, typed_path, typed_report, scope_path, scope_report,
             writer_path, writer_report, *(folder / file for folder in args.layers for file in ('attention.pt', 'comparison.json'))]
    protected = {path: sha(path) for path in paths}
    references = [read(path) for path in (action_report, gate_report, typed_report, scope_report, writer_report)]
    action, gate_audit, typed, scope, writer_audit = references
    checksums = ((action_path, action['sha256']), (gate_path, gate_audit['gate_sha256']),
                 (typed_path, typed['sha256']), (scope_path, scope['sha256']),
                 (writer_path, writer_audit['arms']['joint']['writer_sha256']))
    if any(protected[path] != sha_ for path, sha_ in checksums):
        raise ValueError('A source model changed after its independent audit')
    topics, _ = training_data(read(DESIGN))
    fixture = read(OBSERVATIONS)['topics']
    source = read(DATA)
    records_ = read(EPISODES)
    cases = held_episodes(records_)
    training, role_training, prototypes = training_examples(source, records_['cases'], fixture, topics)
    external = read(args.crosswoz)['rows']
    held_questions = {row[arm] for row in cases for arm in ('all', 'limited', 'neutral')}
    held_observation = {row[arm] for row in cases for arm in ('first', 'update')}
    if ((set(role_training) | {row['text'] for row in external}) & held_questions
            or {row[arm][0] for row in training for arm in ('first', 'update')} & held_observation):
        raise ValueError('Episode evaluation overlaps model training')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    if any(ref['individual_sha256'] != individual for ref in references):
        raise ValueError('The frozen individual changed')
    layers = load_recurrent_replacements(decoder, args.layers, protected[args.checkpoint])
    head = SpanActionHead(base.config.hidden_size).to(base.device)
    head.load_state_dict(torch.load(action_path, map_location='cpu', weights_only=True))
    head.eval().requires_grad_(False)
    typed_weights = torch.load(typed_path, map_location='cpu', weights_only=True)
    planner = TokenFieldAccessHead(2 * base.config.hidden_size).to(base.device)
    planner.load_state_dict(typed_weights['scope'])
    adapter = MissingFieldActions(base.config.hidden_size).to(base.device)
    adapter.load_state_dict(typed_weights['realizer'])
    adapter.eval().requires_grad_(False)
    gate_saved = torch.load(gate_path, map_location='cpu', weights_only=True)
    gate = ObservationReadGate(3 * base.config.hidden_size)
    gate.load_state_dict(gate_saved['weights'])
    gate.eval().requires_grad_(False)
    thresholds = gate_saved['thresholds']
    route = {'model': PairedFieldScopeHead(3 * base.config.hidden_size)}
    saved = torch.load(scope_path, map_location='cpu', weights_only=True)
    route['model'].load_state_dict(saved['state'])
    route['model'].eval().requires_grad_(False)
    route['cutoff'] = saved['cutoff']
    prior = torch.load(writer_path, map_location='cpu', weights_only=True)
    if prior['checkpoint_sha256'] != protected[args.checkpoint] or prior['feature_depth'] != 12:
        raise ValueError('Writer anchor differs from the active checkpoint')
    initial = JointTurnValueWriter(64, feature_width=base.config.hidden_size).to(base.device)
    initial.load_state_dict(prior['writer'])
    initial.eval().requires_grad_(False)
    writer = JointTurnValueWriter(64, feature_width=base.config.hidden_size).to(base.device)
    writer.load_state_dict(prior['writer'])
    features_at_turn = FrozenTurnFeatures(decoder, tokenizer, depth=12)

    @lru_cache(maxsize=256)
    def char_features(text):
        return features_at_turn([text])

    def locate(module):
        return lambda text: module.write(text, 0., features=char_features(text).to(base.device))

    path = args.output / 'working-state.pt'
    weights = args.output / 'observation-weights.pt'
    digests = {str(p): h for p, h in protected.items()}
    if args.verify or args.audit_spans:
        reference = read(args.output / 'result.json')
        if (reference['sources'] != digests or reference['individual_sha256'] != individual
                or reference['checkpoint_sha256'] != sha(weights) or reference['working_state_sha256'] != sha(path)):
            raise ValueError('A verified write checkpoint or snapshot changed')
        saved = torch.load(weights, map_location='cpu', weights_only=True)
        writer.load_state_dict(saved['writer'])
        planner.load_state_dict(saved['planner'])
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        install(stack, layers)
        if args.audit_spans:
            writer.eval().requires_grad_(False)
            scores = {name: audit_update_spans(module, char_features, cases)
                      for name, module in (('previous', initial), ('trained', writer))}
            if (frozen_digest(runtime.organism, runtime.state) != individual
                    or any(sha(path) != digest for path, digest in protected.items())):
                raise ValueError('Span audit changed an input or the individual')
            store_json(args.output / 'span-audit.json', {'checkpoint_sha256': sha(weights), 'update_scores': scores,
                                                         'seconds': perf_counter() - started})
            for name, rows in scores.items():
                print(name, 'top exact spans', sum(row['rank'] == 1 for row in rows),
                      'top five', sum(row['rank'] <= 5 for row in rows),
                      'correct over absent', sum(row['expected_minus_absent'] >= 0 for row in rows), flush=True)
            return
        raw_questions = {(domain, row[arm]) for domain in topics for row in cases for arm in ('all', 'limited')}
        plain = held_questions | {row['first'] for row in cases} | {x for items in prototypes.values() for x in items}
        train_text = set() if args.verify else set(role_training) | {row['text'] for row in external}
        inputs = plain | train_text | scope_texts(raw_questions)
        print('Encoding observations and plans:', len(inputs), flush=True)
        features = text_features(base, tokenizer, inputs, keep_tokens=held_questions | train_text)
        if not args.verify:
            fit_roles(planner, features[1], role_training, external)
            print('Training observation writes:', len(training), flush=True)
            loss = fit_writer(writer, training, source['no_write'], 7, base.device, features=features_at_turn)
        planner.eval().requires_grad_(False)
        writer.eval().requires_grad_(False)
        before = locate_writes(locate(initial), cases)
        after = locate_writes(locate(writer), cases)
        write_summary = {'previous': metrics(cases, before), 'trained': metrics(cases, after)}
        print('Write summary:', write_summary, flush=True)
        if args.verify:
            if after != reference['writes'] or before != reference['previous_writes']:
                raise ValueError('Independent replay produced different write operations')
        else:
            stored = episodes(cases, after, tokenizer, features[0], prototypes, topics)
            atomic_save(stored, path)
            atomic_save({'writer': {key: value.cpu() for key, value in writer.state_dict().items()},
                         'planner': {key: value.cpu() for key, value in planner.state_dict().items()}}, weights)
        answers = audit(decoder, tokenizer, head, adapter, route, planner, gate, features,
                        torch.load(path, map_location='cpu', weights_only=True), cases, topics, thresholds)
        read_summary = summarize(answers)
        print('Read summary:', read_summary, flush=True)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    if (memory != action['memory'] or frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('Episode testing changed a durable memory or its inputs')
    core = {'writing': write_summary, 'read': read_summary, 'rows': answers, 'writes': after,
            'previous_writes': before, 'memory': memory}
    seconds = perf_counter() - started
    if args.verify:
        if any(reference[name] != value for name, value in core.items()):
            raise ValueError('Reload produced a different reply or durable result')
        store_json(args.output / 'reload.json', {'checkpoint_sha256': sha(weights), 'working_state_sha256': sha(path),
                                                 'seconds': seconds})
    else:
        store_json(args.output / 'result.json', {**core, 'training_examples': len(training),
                                                 'role_questions': len(role_training), 'trace': loss,
                                                 'checkpoint_sha256': sha(weights), 'working_state_sha256': sha(path),
                                                 'individual_sha256': individual, 'sources': digests, 'seconds': seconds})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('actions', type=Path)
    parser.add_argument('gate', type=Path)
    parser.add_argument('typed', type=Path)
    parser.add_argument('scope', type=Path)
    parser.add_argument('writer', type=Path)
    parser.add_argument('crosswoz', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('layers', nargs=2, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify', action='store_true')
    mode.add_argument('--audit-spans', action='store_true')
    args = parser.parse_args()
    if not (args.verify or args.audit_spans) and args.output.exists():
        parser.error('Choose a fresh episode experiment directory')
    main(args)
